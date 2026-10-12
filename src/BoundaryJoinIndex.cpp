#include "whatif/BoundaryJoinIndex.h"

#include "whatif/CausalAdjustment.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace whatif {
namespace {

void appendSized(std::string &out, const std::string &value) {
  const std::uint64_t size = value.size();
  out.append(reinterpret_cast<const char *>(&size), sizeof(size));
  out.append(value);
}

void appendInstance(std::string &out, InstanceId value) {
  out.append(reinterpret_cast<const char *>(&value), sizeof(value));
}

Count checkedAdd(Count left, Count right) {
  if (right > std::numeric_limits<Count>::max() - left) {
    throw std::overflow_error("Boundary count addition overflow");
  }
  return left + right;
}

Count checkedMultiply(Count left, Count right) {
  if (left != 0 && right > std::numeric_limits<Count>::max() / left) {
    throw std::overflow_error("Boundary count multiplication overflow");
  }
  return left * right;
}

bool sameScope(const std::vector<VariableId> &left,
               const std::vector<VariableId> &right) {
  return std::set<VariableId>(left.begin(), left.end()) ==
         std::set<VariableId>(right.begin(), right.end());
}

std::vector<std::vector<VariableId>>
allNonEmptyScopes(const ParentTreeSpec &tree, std::size_t max_width) {
  if (tree.variables.size() > max_width) {
    throw std::runtime_error(
        "Parent Tree " + tree.id + " width " +
        std::to_string(tree.variables.size()) +
        " exceeds max_all_subset_width=" + std::to_string(max_width));
  }
  if (tree.variables.size() >= 63) {
    throw std::runtime_error("local cube masks support at most 62 variables");
  }
  const std::uint64_t all = (std::uint64_t{1} << tree.variables.size()) - 1;
  std::vector<std::vector<VariableId>> scopes;
  scopes.reserve(static_cast<std::size_t>(all));
  for (std::uint64_t mask = 1; mask <= all; ++mask) {
    std::vector<VariableId> scope;
    for (std::size_t variable = 0; variable < tree.variables.size();
         ++variable) {
      if ((mask & (std::uint64_t{1} << variable)) != 0) {
        scope.push_back(tree.variables[variable]);
      }
    }
    scopes.push_back(std::move(scope));
  }
  return scopes;
}

std::vector<VariableId>
canonicalTreeScope(const ParentTreeSpec &tree,
                   const std::vector<VariableId> &requested) {
  const std::unordered_set<VariableId> wanted(requested.begin(),
                                              requested.end());
  if (wanted.size() != requested.size()) {
    throw std::runtime_error("scope repeats a causal variable");
  }
  std::vector<VariableId> result;
  for (const auto &variable : tree.variables) {
    if (wanted.count(variable) != 0) {
      result.push_back(variable);
    }
  }
  if (result.size() != requested.size()) {
    throw std::runtime_error("scope is not contained in Parent Tree " +
                             tree.id);
  }
  return result;
}

std::string boundaryRowKey(const BoundaryRow &row) {
  auto bindings = row.bindings;
  auto roles = row.role_nodes;
  std::sort(bindings.begin(), bindings.end(),
            [](const auto &left, const auto &right) {
              return left.variable < right.variable;
            });
  std::sort(roles.begin(), roles.end(),
            [](const auto &left, const auto &right) {
              return left.role < right.role;
            });
  std::string key;
  for (const auto &binding : bindings) {
    appendSized(key, binding.variable);
    appendInstance(key, binding.instance_id);
    appendSized(key, binding.raw_value);
  }
  key.push_back('\x1e');
  for (const auto &role : roles) {
    appendSized(key, role.role);
    appendSized(key, role.node);
  }
  return key;
}

const VariableBindingCell *findBinding(const BoundaryRow &row,
                                       const VariableId &variable) {
  const auto found =
      std::find_if(row.bindings.begin(), row.bindings.end(),
                   [&](const auto &item) { return item.variable == variable; });
  return found == row.bindings.end() ? nullptr : &*found;
}

const RoleBinding *findRole(const BoundaryRow &row, const RoleId &role) {
  const auto found =
      std::find_if(row.role_nodes.begin(), row.role_nodes.end(),
                   [&](const auto &item) { return item.role == role; });
  return found == row.role_nodes.end() ? nullptr : &*found;
}

std::set<VariableId> variableSchema(const std::vector<BoundaryRow> &rows,
                                    const BoundaryView *view = nullptr) {
  std::set<VariableId> result;
  if (view != nullptr) {
    result.insert(view->scope.begin(), view->scope.end());
  }
  for (const auto &row : rows) {
    for (const auto &binding : row.bindings) {
      result.insert(binding.variable);
    }
  }
  return result;
}

std::set<RoleId> roleSchema(const std::vector<BoundaryRow> &rows) {
  std::set<RoleId> result;
  for (const auto &row : rows) {
    for (const auto &role : row.role_nodes) {
      result.insert(role.role);
    }
  }
  return result;
}

std::string joinKey(const BoundaryRow &row,
                    const std::vector<VariableId> &variables,
                    const std::vector<RoleId> &roles) {
  std::string key;
  for (const auto &variable : variables) {
    const auto *binding = findBinding(row, variable);
    if (binding == nullptr) {
      return {};
    }
    appendSized(key, variable);
    appendInstance(key, binding->instance_id);
  }
  key.push_back('\x1f');
  for (const auto &role : roles) {
    const auto *binding = findRole(row, role);
    if (binding == nullptr) {
      return {};
    }
    appendSized(key, role);
    appendSized(key, binding->node);
  }
  return key;
}

bool compatible(const BoundaryRow &left, const BoundaryRow &right) {
  for (const auto &binding : right.bindings) {
    const auto *found = findBinding(left, binding.variable);
    if (found != nullptr &&
        (found->instance_id != binding.instance_id ||
         found->raw_value != binding.raw_value)) {
      return false;
    }
  }

  const auto compatible_roles = [](const RoleBinding &a, const RoleBinding &b) {
    return (a.role == b.role) == (a.node == b.node);
  };
  for (std::size_t i = 0; i < left.role_nodes.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j)
      if (!compatible_roles(left.role_nodes[i], left.role_nodes[j])) return false;
    for (const auto &b : right.role_nodes)
      if (!compatible_roles(left.role_nodes[i], b)) return false;
  }
  for (std::size_t i = 0; i < right.role_nodes.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (!compatible_roles(right.role_nodes[i], right.role_nodes[j])) return false;
  return true;
}

BoundaryRow mergeRows(const BoundaryRow &left, const BoundaryRow &right) {
  BoundaryRow result = left;
  for (const auto &binding : right.bindings) {
    if (findBinding(result, binding.variable) == nullptr) {
      result.bindings.push_back(binding);
    }
  }
  for (const auto &role : right.role_nodes) {
    if (findRole(result, role.role) == nullptr) {
      result.role_nodes.push_back(role);
    }
  }
  result.multiplicity = checkedMultiply(left.multiplicity, right.multiplicity);
  return result;
}

std::vector<BoundaryRow> compactRows(std::vector<BoundaryRow> rows) {
  std::unordered_map<std::string, std::size_t> position;
  std::vector<BoundaryRow> result;
  result.reserve(rows.size());
  for (auto &row : rows) {
    auto key = boundaryRowKey(row);
    const auto found = position.find(key);
    if (found == position.end()) {
      position.emplace(std::move(key), result.size());
      result.push_back(std::move(row));
    } else {
      auto &count = result[found->second].multiplicity;
      count = checkedAdd(count, row.multiplicity);
    }
  }
  return result;
}

std::vector<BoundaryRow> joinRows(const std::vector<BoundaryRow> &left,
                                  const BoundaryView &right_view,
                                  OnlineScopeStats *stats = nullptr) {
  if (stats != nullptr) {
    ++stats->join_calls;
    stats->join_probe_rows += left.size();
  }
  if (left.empty() || right_view.rows.empty()) {
    return {};
  }
  const auto left_variables = variableSchema(left);
  const auto right_variables = variableSchema(right_view.rows, &right_view);
  const auto left_roles = roleSchema(left);
  const auto right_roles = roleSchema(right_view.rows);
  std::vector<VariableId> shared_variables;
  std::vector<RoleId> shared_roles;
  std::set_intersection(left_variables.begin(), left_variables.end(),
                        right_variables.begin(), right_variables.end(),
                        std::back_inserter(shared_variables));
  std::set_intersection(left_roles.begin(), left_roles.end(),
                        right_roles.begin(), right_roles.end(),
                        std::back_inserter(shared_roles));

  std::unordered_map<std::string, std::vector<const BoundaryRow *>> right_index;
  right_index.reserve(right_view.rows.size() * 2 + 1);
  for (const auto &row : right_view.rows) {
    right_index[joinKey(row, shared_variables, shared_roles)].push_back(&row);
  }

  std::vector<BoundaryRow> output;
  for (const auto &left_row : left) {
    const auto found =
        right_index.find(joinKey(left_row, shared_variables, shared_roles));
    if (found == right_index.end()) {
      continue;
    }
    for (const auto *right_row : found->second) {
      if (compatible(left_row, *right_row)) {
        if (stats != nullptr) ++stats->join_work_pairs;
        output.push_back(mergeRows(left_row, *right_row));
      }
    }
  }
  return compactRows(std::move(output));
}

} // namespace

std::string
MaterializedBoundaryIndex::attachmentViewId(const VariableId &variable) {
  return "attachment::" + std::to_string(variable.size()) + ":" + variable;
}

MaterializedBoundaryIndex MaterializedBoundaryIndex::build(
    std::vector<VariableInput> variables, std::vector<CausalEdge> causal_edges,
    std::vector<ParentTreeSpec> parent_trees, BoundaryBuildConfig config) {
  MaterializedBoundaryIndex index;
  index.causal_edges_ = causal_edges;
  index.parent_trees_ = parent_trees;
  if (config.precompiled_boundary_plan.has_value()) {
    index.plan_ = std::move(*config.precompiled_boundary_plan);
  } else {
    BoundaryRequirementPlanner planner(variables, causal_edges, parent_trees);
    index.plan_ = planner.plan(
        config.query_workload, config.include_all_reachable_pairs,
        config.prune_redundant_boundary_views, config.exact_scope_workload);
  }

  if (config.materialize_structural_boundaries) {
    std::unordered_map<VariableId, std::size_t> memberships;
    for (const auto &tree : parent_trees)
      for (const auto &v : tree.variables) ++memberships[v];
    for (const auto &tree : parent_trees) {
      const auto n = tree.variables.size();
      std::unordered_map<VariableId, std::size_t> positions;
      std::vector<bool> boundary(n, false);
      std::vector<std::vector<bool>> reach(n, std::vector<bool>(n, false));
      for (std::size_t i = 0; i < n; ++i) {
        positions.emplace(tree.variables[i], i);
        boundary[i] = memberships.at(tree.variables[i]) > 1;
        reach[i][i] = true;
      }
      for (const auto &edge : causal_edges) {
        const auto a = positions.find(edge.parent), b = positions.find(edge.child);
        if (a != positions.end() && b != positions.end()) reach[a->second][b->second] = true;
        else if (a != positions.end()) boundary[a->second] = true;
        else if (b != positions.end()) boundary[b->second] = true;
      }
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j)
            reach[i][j] = reach[i][j] || (reach[i][k] && reach[k][j]);
      for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) {
        if (a == b || !reach[a][b] || (!boundary[a] && !boundary[b])) continue;
        std::vector<VariableId> scope;
        for (std::size_t k = 0; k < n; ++k)
          if (reach[a][k] && reach[k][b]) scope.push_back(tree.variables[k]);
        if (std::any_of(index.plan_.requirements.begin(), index.plan_.requirements.end(),
            [&](const auto &r) { return r.tree_id == tree.id && sameScope(r.scope, scope); })) continue;
        BoundaryRequirement requirement;
        requirement.tree_id = tree.id;
        requirement.scope = scope;
        requirement.view_id = "structural::";
        appendSized(requirement.view_id, tree.id);
        for (const auto &v : scope) appendSized(requirement.view_id, v);
        requirement.direction_flags = (boundary[a] ? kBoundaryTargetSide : 0) |
                                      (boundary[b] ? kBoundarySourceSide : 0);
        index.plan_.requirements.push_back(std::move(requirement));
      }
    }
  }

  if (config.materialize_variable_attachments) {
    std::unordered_map<std::string, std::string> singleton_aliases;
    auto &requirements = index.plan_.requirements;
    requirements.erase(
        std::remove_if(requirements.begin(), requirements.end(),
                       [&](const auto &requirement) {
                         if (requirement.scope.size() != 1) {
                           return false;
                         }
                         singleton_aliases.emplace(
                             requirement.view_id,
                             attachmentViewId(requirement.scope.front()));
                         return true;
                       }),
        requirements.end());
    for (auto &query : index.plan_.queries) {
      std::vector<std::string> rewritten;
      std::unordered_set<std::string> seen;
      for (auto view_id : query.view_ids) {
        const auto alias = singleton_aliases.find(view_id);
        if (alias != singleton_aliases.end()) {
          view_id = alias->second;
        }
        if (seen.insert(view_id).second) {
          rewritten.push_back(std::move(view_id));
        }
      }
      query.view_ids = std::move(rewritten);
    }
  }

  if (config.materialize_variable_attachments) {
    for (const auto &variable : variables) {
      BoundaryView view;
      view.id = attachmentViewId(variable.id);
      view.tree_id = "@attachment";
      view.scope = {variable.id};
      view.rows.reserve(variable.instances.size());
      for (const auto &instance : variable.instances) {
        BoundaryRow row;
        row.bindings.push_back({variable.id, instance.id, instance.value});
        row.role_nodes = instance.roles;
        view.rows.push_back(std::move(row));
      }
      index.boundary_views_.push_back(std::move(view));
    }
  }

  std::unordered_map<VariableId, VariableInput> variable_inputs;
  for (auto &variable : variables) {
    if (!variable_inputs.emplace(variable.id, std::move(variable)).second) {
      throw std::runtime_error("duplicate variable input " + variable.id);
    }
  }

  std::unordered_map<std::string, std::vector<const BoundaryRequirement *>>
      requirements_by_tree;
  for (const auto &requirement : index.plan_.requirements) {
    requirements_by_tree[requirement.tree_id].push_back(&requirement);
  }

  for (const auto &tree : parent_trees) {
    std::vector<VariableInput> local_inputs;
    local_inputs.reserve(tree.variables.size());
    for (const auto &variable : tree.variables) {
      const auto found = variable_inputs.find(variable);
      if (found == variable_inputs.end()) {
        throw std::runtime_error("Parent Tree " + tree.id +
                                 " references missing input " + variable);
      }
      local_inputs.push_back(found->second);
    }

    if (config.stream_all_subset_cubes) {
      if (config.cube_policy != LocalCubePolicy::AllSubsets ||
          tree.variables.size() > config.max_all_subset_width ||
          !requirements_by_tree[tree.id].empty()) {
        throw std::runtime_error("streaming full cubes require AllSubsets within width limit and no boundary outputs");
      }
      auto tree_config = config.tree_executor;
      tree_config.store_all_supports = false;
      tree_config.store_value_counts = true;
      tree_config.enable_derive = false;
      tree_config.store_participation = false;
      tree_config.store_derive_results = false;
      whatif::core::FullLatticeIndex local_index(std::move(local_inputs),
                                                   std::move(tree_config));
      auto emit_cube = [&](auto mask, auto&& counts) {
        CubeTable cube;
        cube.tree_id = tree.id;
        cube.scope = local_index.variablesOf(mask);
        cube.counts = std::move(counts);
        if (config.streamed_cube_sink) {
          config.streamed_cube_sink(std::move(cube));
        } else {
          index.cubes_.push_back(std::move(cube));
        }
      };
      if (config.tree_executor.full_join_rollup)
        local_index.streamRollupCubes(emit_cube);
      else
        local_index.streamIndependentCubes(emit_cube);
      continue;
    }

    std::vector<std::vector<VariableId>> cube_scopes;
    if (config.cube_policy == LocalCubePolicy::AllSubsets) {
      cube_scopes = allNonEmptyScopes(tree, config.max_all_subset_width);
    } else {
      const auto found = config.explicit_cube_scopes.find(tree.id);
      if (found != config.explicit_cube_scopes.end()) {
        for (const auto &scope : found->second) {
          cube_scopes.push_back(canonicalTreeScope(tree, scope));
        }
      }
    }

    // An exact scope normally spans several split Trees. Keep the API robust
    // when a supplied exact scope becomes local after a later merge.
    for (const auto &query : index.plan_.queries) {
      if (!query.view_ids.empty() || !query.treatments.empty() ||
          !query.outcomes.empty()) {
        continue;
      }
      const std::unordered_set<VariableId> members(tree.variables.begin(),
                                                   tree.variables.end());
      if (std::all_of(query.global_scope.begin(), query.global_scope.end(),
                      [&](const auto &v) { return members.count(v) != 0; })) {
        auto scope = canonicalTreeScope(tree, query.global_scope);
        if (std::none_of(
                cube_scopes.begin(), cube_scopes.end(),
                [&](const auto &old) { return sameScope(old, scope); })) {
          cube_scopes.push_back(std::move(scope));
        }
      }
    }

    std::vector<std::vector<VariableId>> boundary_scopes;
    for (const auto *requirement : requirements_by_tree[tree.id]) {
      boundary_scopes.push_back(canonicalTreeScope(tree, requirement->scope));
    }

    std::vector<std::vector<VariableId>> target_scopes = cube_scopes;
    for (const auto &scope : boundary_scopes) {
      if (std::none_of(target_scopes.begin(), target_scopes.end(),
                       [&](const auto &existing) {
                         return sameScope(existing, scope);
                       })) {
        target_scopes.push_back(scope);
      }
    }
    if (target_scopes.empty()) {
      continue;
    }

    auto tree_config = config.tree_executor;
    tree_config.target_scopes = target_scopes;
    tree_config.source_output_scopes.clear();
    tree_config.binding_output_scopes = boundary_scopes;
    tree_config.store_all_supports = false;
    tree_config.store_value_counts = true;
    tree_config.enable_derive = false;
    tree_config.derive_materialization_budget_tuples = 0;
    tree_config.store_participation = false;
    tree_config.store_derive_results = false;
    if (config.streamed_cube_sink) {
      std::unordered_map<whatif::core::Mask, std::vector<VariableId>>
          cube_scope_by_mask;
      std::unordered_map<VariableId, std::size_t> local_position;
      for (std::size_t i = 0; i < local_inputs.size(); ++i) {
        local_position.emplace(local_inputs[i].id, i);
      }
      for (const auto &scope : cube_scopes) {
        whatif::core::Mask mask = 0;
        for (const auto &variable : scope) {
          const auto found = local_position.find(variable);
          if (found == local_position.end()) {
            throw std::runtime_error("Cube scope references missing local variable " +
                                     variable);
          }
          mask |= whatif::core::Mask{1} << found->second;
        }
        cube_scope_by_mask.emplace(mask, scope);
      }
      tree_config.streamed_value_count_sink =
          [&, tree_id = tree.id,
           cube_scope_by_mask = std::move(cube_scope_by_mask)](
              whatif::core::Mask mask,
              std::map<std::vector<std::string>, whatif::core::Count>
                  &&counts) {
            const auto found = cube_scope_by_mask.find(mask);
            if (found == cube_scope_by_mask.end()) {
              return;
            }
            CubeTable cube;
            cube.tree_id = tree_id;
            cube.scope = found->second;
            cube.counts = std::move(counts);
            config.streamed_cube_sink(std::move(cube));
          };
    }
    whatif::core::FullLatticeIndex local_index(std::move(local_inputs),
                                                  std::move(tree_config));
    local_index.build();

    if (!config.streamed_cube_sink) {
      for (const auto &scope : cube_scopes) {
        const auto mask = local_index.maskOf(scope);
        CubeTable cube;
        cube.tree_id = tree.id;
        cube.scope = scope;
        cube.counts = local_index.result().value_counts[mask];
        index.cubes_.push_back(std::move(cube));
      }
    }

    for (const auto *requirement : requirements_by_tree[tree.id]) {
      const auto mask = local_index.maskOf(requirement->scope);
      BoundaryView view;
      view.id = requirement->view_id;
      view.tree_id = tree.id;
      view.scope = requirement->scope;
      const std::unordered_set<VariableId> in_scope(requirement->scope.begin(),
                                                    requirement->scope.end());
      for (const auto &source_row :
           local_index.result().binding_outputs[mask]) {
        BoundaryRow row;
        for (std::size_t variable = 0;
             variable < local_index.variables().size(); ++variable) {
          const auto &variable_id = local_index.variables()[variable].id;
          if (in_scope.count(variable_id) == 0) {
            continue;
          }
          if (source_row.instance_ids[variable] ==
              whatif::core::kNoInstance) {
            throw std::runtime_error("materialized Boundary row misses " +
                                     variable_id);
          }
          row.bindings.push_back({variable_id,
                                  source_row.instance_ids[variable],
                                  source_row.values[variable]});
        }
        row.role_nodes = source_row.role_nodes;
        row.multiplicity = source_row.multiplicity;
        view.rows.push_back(std::move(row));
      }
      std::sort(view.rows.begin(), view.rows.end(),
                [](const auto &left, const auto &right) {
                  return boundaryRowKey(left) < boundaryRowKey(right);
                });
      index.boundary_views_.push_back(std::move(view));
    }
  }

  std::sort(index.cubes_.begin(), index.cubes_.end(),
            [](const auto &left, const auto &right) {
              if (left.tree_id != right.tree_id) {
                return left.tree_id < right.tree_id;
              }
              if (left.scope.size() != right.scope.size()) {
                return left.scope.size() < right.scope.size();
              }
              return left.scope < right.scope;
            });
  std::sort(
      index.boundary_views_.begin(), index.boundary_views_.end(),
      [](const auto &left, const auto &right) { return left.id < right.id; });
  index.rebuildCubeLookup();
  return index;
}

void MaterializedBoundaryIndex::rebuildCubeLookup() {
  cube_lookup_.clear();
  cube_lookup_.reserve(cubes_.size());
  for (std::size_t i = 0; i < cubes_.size(); ++i) {
    auto scope = cubes_[i].scope;
    std::sort(scope.begin(), scope.end());
    scope.erase(std::unique(scope.begin(), scope.end()), scope.end());
    // Preserve the first matching cube, just as the previous linear search did.
    cube_lookup_.emplace(std::move(scope), i);
  }
}

const CubeTable &
MaterializedBoundaryIndex::cube(const std::string &tree_id,
                                const std::vector<VariableId> &scope) const {
  const auto found =
      std::find_if(cubes_.begin(), cubes_.end(), [&](const auto &item) {
        return item.tree_id == tree_id && sameScope(item.scope, scope);
      });
  if (found == cubes_.end()) {
    throw std::runtime_error("unknown Tree-local cube");
  }
  return *found;
}

void MaterializedBoundaryIndex::materializeExactBoundaryViews(
    const std::vector<ExactScopeSpec> &queries) {
  std::set<std::vector<VariableId>> needed;
  std::map<std::vector<VariableId>, std::string> owner;
  for (const auto &query : queries) {
    for (const auto &tree : parent_trees_) {
      std::vector<VariableId> scope;
      for (const auto &variable : query.scope)
        if (std::find(tree.variables.begin(), tree.variables.end(), variable) != tree.variables.end())
          scope.push_back(variable);
      std::sort(scope.begin(), scope.end());
      scope.erase(std::unique(scope.begin(), scope.end()), scope.end());
      if (scope.size() < 2) continue;
      needed.insert(scope);
      owner.emplace(scope, tree.id);
    }
  }
  for (const auto &scope : needed) {
    if (std::any_of(boundary_views_.begin(), boundary_views_.end(),
                    [&](const auto &view) { return sameScope(view.scope, scope); })) continue;
    std::string id = "workload-boundary::";
    for (const auto &variable : scope) appendSized(id, variable);
    // Retain mappings, never derive these projections from aggregated counts.
    auto result = executeExactScope({id, scope}, {}, true);
    BoundaryView view;
    view.id = id;
    view.tree_id = owner.at(scope);
    view.scope = scope;
    view.rows = std::move(result.rows);
    boundary_views_.push_back(std::move(view));
  }
}

const BoundaryView &
MaterializedBoundaryIndex::boundaryView(const std::string &view_id) const {
  const auto found =
      std::find_if(boundary_views_.begin(), boundary_views_.end(),
                   [&](const auto &item) { return item.id == view_id; });
  if (found == boundary_views_.end()) {
    throw std::runtime_error("unknown Boundary view " + view_id);
  }
  return *found;
}

const BoundaryQueryPlan &
MaterializedBoundaryIndex::queryPlan(const std::string &query_id) const {
  const auto found =
      std::find_if(plan_.queries.begin(), plan_.queries.end(),
                   [&](const auto &item) { return item.query_id == query_id; });
  if (found == plan_.queries.end()) {
    throw std::runtime_error("unknown cross-tree query " + query_id);
  }
  return *found;
}

BoundaryJoinResult
MaterializedBoundaryIndex::execute(const std::string &query_id,
                                   std::vector<VariableId> output_scope,
                                   bool retain_rows) const {
  const auto &query = queryPlan(query_id);
  if (output_scope.empty()) {
    output_scope = query.global_scope;
  }
  const std::unordered_set<VariableId> global(query.global_scope.begin(),
                                              query.global_scope.end());
  std::unordered_set<VariableId> output_seen;
  for (const auto &variable : output_scope) {
    if (global.count(variable) == 0) {
      throw std::runtime_error("output scope is not contained in query scope");
    }
    if (!output_seen.insert(variable).second) {
      throw std::runtime_error("output scope repeats a variable");
    }
  }

  if (query.view_ids.empty()) {
    const auto local_cube =
        std::find_if(cubes_.begin(), cubes_.end(), [&](const auto &item) {
          return sameScope(item.scope, query.global_scope);
        });
    if (local_cube == cubes_.end()) {
      throw std::runtime_error(
          "Tree-local query scope was not materialized as a Cube");
    }
    std::unordered_map<VariableId, std::size_t> cube_position;
    for (std::size_t i = 0; i < local_cube->scope.size(); ++i) {
      cube_position.emplace(local_cube->scope[i], i);
    }
    BoundaryJoinResult result;
    result.query_id = query_id;
    result.global_scope = query.global_scope;
    result.output_scope = output_scope;
    for (const auto &entry : local_cube->counts) {
      ValueTuple values;
      values.reserve(output_scope.size());
      for (const auto &variable : output_scope) {
        const auto found = cube_position.find(variable);
        if (found == cube_position.end() ||
            found->second >= entry.first.size()) {
          throw std::runtime_error("Tree-local Cube misses output variable " +
                                   variable);
        }
        values.push_back(entry.first[found->second]);
      }
      result.value_counts[values] =
          checkedAdd(result.value_counts[values], entry.second);
      result.exact_binding_count =
          checkedAdd(result.exact_binding_count, entry.second);
    }
    return result;
  }

  std::vector<const BoundaryView *> remaining;
  remaining.reserve(query.view_ids.size());
  for (const auto &view_id : query.view_ids) {
    remaining.push_back(&boundaryView(view_id));
  }
  auto first = std::min_element(remaining.begin(), remaining.end(),
                                [](const auto *left, const auto *right) {
                                  return left->rows.size() < right->rows.size();
                                });
  std::vector<BoundaryRow> rows = (*first)->rows;
  remaining.erase(first);

  while (!remaining.empty() && !rows.empty()) {
    const auto current_variables = variableSchema(rows);
    const auto current_roles = roleSchema(rows);
    auto best = remaining.begin();
    std::size_t best_shared = 0;
    for (auto candidate = remaining.begin(); candidate != remaining.end();
         ++candidate) {
      std::size_t shared = 0;
      for (const auto &variable : (*candidate)->scope) {
        shared += current_variables.count(variable);
      }
      const auto candidate_roles = roleSchema((*candidate)->rows);
      for (const auto &role : candidate_roles) {
        shared += current_roles.count(role);
      }
      if (shared > best_shared ||
          (shared == best_shared &&
           (*candidate)->rows.size() < (*best)->rows.size())) {
        best = candidate;
        best_shared = shared;
      }
    }
    rows = joinRows(rows, **best);
    remaining.erase(best);
  }

  BoundaryJoinResult result;
  result.query_id = query_id;
  result.global_scope = query.global_scope;
  result.output_scope = output_scope;
  for (const auto &row : rows) {
    ValueTuple values;
    values.reserve(output_scope.size());
    for (const auto &variable : output_scope) {
      const auto *binding = findBinding(row, variable);
      if (binding == nullptr) {
        throw std::runtime_error("joined Boundary row misses output variable " +
                                 variable);
      }
      values.push_back(binding->raw_value);
    }
    result.value_counts[values] =
        checkedAdd(result.value_counts[values], row.multiplicity);
    result.exact_binding_count =
        checkedAdd(result.exact_binding_count, row.multiplicity);
  }
  if (retain_rows) {
    result.rows = std::move(rows);
  }
  return result;
}

BoundaryJoinResult
MaterializedBoundaryIndex::execute(const DirectedQuerySpec &input_query,
                                   std::vector<VariableId> output_scope,
                                   bool retain_rows) const {
  DirectedQuerySpec query = input_query;

  std::vector<VariableId> variables;
  std::unordered_set<VariableId> known;
  for (const auto &tree : parent_trees_) {
    for (const auto &variable : tree.variables) {
      if (known.insert(variable).second) {
        variables.push_back(variable);
      }
    }
  }
  for (const auto &view : boundary_views_) {
    for (const auto &variable : view.scope) {
      if (known.insert(variable).second) {
        variables.push_back(variable);
      }
    }
  }
  if (variables.empty()) {
    throw std::runtime_error(
        "dynamic adjustment execution needs indexed variables");
  }

  auto canonicalize = [&](const std::vector<VariableId> &input,
                          const char *description) {
    const std::unordered_set<VariableId> wanted(input.begin(), input.end());
    if (wanted.size() != input.size()) {
      throw std::runtime_error(std::string(description) +
                               " repeats a causal variable");
    }
    for (const auto &variable : input) {
      if (known.count(variable) == 0) {
        throw std::runtime_error(std::string(description) +
                                 " references unknown variable " + variable);
      }
    }
    std::vector<VariableId> result;
    for (const auto &variable : variables) {
      if (wanted.count(variable) != 0) {
        result.push_back(variable);
      }
    }
    return result;
  };

  query.treatments = canonicalize(query.treatments, "treatment set");
  query.outcomes = canonicalize(query.outcomes, "outcome set");
  query.conditioning_variables =
      canonicalize(query.conditioning_variables, "conditioning set");
  query.required_variables =
      canonicalize(query.required_variables, "required adjustment set");
  if (query.treatments.size() != 1 || query.outcomes.size() != 1) {
    throw std::runtime_error(
        "dynamic adjustment query requires one treatment and one outcome");
  }

  std::vector<std::pair<std::string, std::string>> dag_edges;
  dag_edges.reserve(causal_edges_.size());
  for (const auto &edge : causal_edges_) {
    dag_edges.emplace_back(edge.parent, edge.child);
  }
  whatif::CausalDag dag(variables, dag_edges);
  const auto treatment_mask = dag.maskOf(query.treatments);
  const auto outcome_mask = dag.maskOf(query.outcomes);
  if ((treatment_mask & outcome_mask) != 0) {
    throw std::runtime_error("treatment and outcome must be disjoint");
  }
  if (!dag.reaches(treatment_mask, outcome_mask)) {
    throw std::runtime_error(
        "treatment has no directed causal path to outcome");
  }

  auto condition_mask = dag.maskOf(query.conditioning_variables);
  const auto required_mask = dag.maskOf(query.required_variables);
  if (((condition_mask | required_mask) & (treatment_mask | outcome_mask)) !=
      0) {
    throw std::runtime_error(
        "adjustment variables must be disjoint from treatment and outcome");
  }
  condition_mask = dag.reduceConditioningUnderIntervention(
      treatment_mask, outcome_mask, condition_mask);
  query.conditioning_variables = dag.variablesOf(condition_mask);

  const auto all = (whatif::CausalMask{1} << variables.size()) - 1;
  auto adjustment_mask = condition_mask | required_mask;
  if (adjustment_mask == 0) {
    const auto analysis = dag.analyzeAdjustmentWithin(
        treatment_mask, outcome_mask, all & ~(treatment_mask | outcome_mask));
    if (!analysis.adjustment_exists) {
      throw std::runtime_error(
          "no observed adjustment set identifies the dynamic query");
    }
    adjustment_mask = analysis.adjustment;
  } else if (!dag.isValidAdjustment(treatment_mask, outcome_mask,
                                    adjustment_mask)) {
    throw std::runtime_error(
        "the supplied multi-variable adjustment set is invalid");
  }

  query.required_variables.clear();
  for (const auto &variable :
       dag.variablesOf(adjustment_mask & ~condition_mask)) {
    query.required_variables.push_back(variable);
  }
  const auto global_mask =
      dag.causalPathClosure(treatment_mask, outcome_mask) | adjustment_mask;
  std::vector<VariableId> global_scope;
  for (std::size_t variable = 0; variable < variables.size(); ++variable) {
    if ((global_mask & (whatif::CausalMask{1} << variable)) != 0) {
      global_scope.push_back(variables[variable]);
    }
  }
  if (query.id.empty()) {
    query.id = "dynamic:" +
               BoundaryRequirementPlanner::exactScopeQueryId(global_scope);
  }

  return executeExactScope({query.id, global_scope}, std::move(output_scope), retain_rows);
}

BoundaryJoinResult MaterializedBoundaryIndex::executeExactScope(
    const ExactScopeSpec &query, std::vector<VariableId> output_scope,
    bool retain_rows, bool cube_only, OnlineScopeStats *stats, bool plan_only) const {
  if (stats != nullptr) *stats = OnlineScopeStats{};
  if (retain_rows && cube_only) throw std::runtime_error("cube-only query cannot return bindings");
  const auto &global_scope = query.scope;
  std::unordered_set<VariableId> known;
  for (const auto &tree : parent_trees_)
    known.insert(tree.variables.begin(), tree.variables.end());
  std::unordered_set<VariableId> seen;
  if (global_scope.empty()) throw std::runtime_error("empty exact scope");
  for (const auto &variable : global_scope) {
    if (known.count(variable) == 0 || !seen.insert(variable).second)
      throw std::runtime_error("unknown or repeated exact scope variable");
  }

  if (output_scope.empty()) {
    output_scope = global_scope;
  } else {
    std::unordered_set<VariableId> output_seen;
    for (const auto &variable : output_scope)
      if (!output_seen.insert(variable).second)
        throw std::runtime_error("output scope repeats a variable");
  }
  const std::unordered_set<VariableId> global(global_scope.begin(),
                                              global_scope.end());
  for (const auto &variable : output_scope) {
    if (global.count(variable) == 0) {
      throw std::runtime_error("output scope is not contained in query scope");
    }
  }

  if (!retain_rows) {
    auto lookup_scope = global_scope;
    std::sort(lookup_scope.begin(), lookup_scope.end());
    const auto found_cube = cube_lookup_.find(lookup_scope);
    if (found_cube != cube_lookup_.end()) {
      const auto local_cube = cubes_.begin() + found_cube->second;
      if (stats != nullptr) {
        stats->cube_hit = true;
        stats->component_ids = {local_cube->tree_id};
      }
      if (plan_only) {
        BoundaryJoinResult plan;
        plan.query_id = query.id; plan.global_scope = global_scope; plan.output_scope = output_scope;
        return plan;
      }
      std::unordered_map<VariableId, std::size_t> position;
      for (std::size_t i = 0; i < local_cube->scope.size(); ++i) {
        position.emplace(local_cube->scope[i], i);
      }
      BoundaryJoinResult result;
      result.query_id = query.id;
      result.global_scope = global_scope;
      result.output_scope = output_scope;
      for (const auto &entry : local_cube->counts) {
        ValueTuple values;
        for (const auto &variable : output_scope) {
          values.push_back(entry.first.at(position.at(variable)));
        }
        result.value_counts[values] =
            checkedAdd(result.value_counts[values], entry.second);
        result.exact_binding_count =
            checkedAdd(result.exact_binding_count, entry.second);
      }
      return result;
    }
  }

  if (cube_only) throw std::runtime_error("index_unavailable: exact-scope cube not materialized");
  std::vector<const BoundaryView *> candidates;
  for (const auto &view : boundary_views_) {
    if (view.scope.empty() || !std::all_of(view.scope.begin(), view.scope.end(),
                                           [&](const auto &variable) {
                                             return global.count(variable) != 0;
                                           })) {
      continue;
    }
    candidates.push_back(&view);
    if (stats != nullptr && view.scope.size() > 1)
      ++stats->eligible_multivariable_views;
  }

  std::unordered_set<VariableId> uncovered(global.begin(), global.end());
  std::vector<const BoundaryView *> selected;
  std::unordered_set<std::string> selected_ids;
  while (!uncovered.empty()) {
    const BoundaryView *best = nullptr;
    std::size_t best_gain = 0;
    long double best_score = -1.0L;
    for (const auto *candidate : candidates) {
      if (selected_ids.count(candidate->id) != 0) {
        continue;
      }
      std::size_t gain = 0;
      for (const auto &variable : candidate->scope) {
        gain += uncovered.count(variable);
      }
      if (gain == 0) {
        continue;
      }
      const auto storage = std::max<std::size_t>(candidate->rows.size(), 1);
      const long double score =
          static_cast<long double>(gain) / static_cast<long double>(storage);
      if (best == nullptr || gain > best_gain ||
          (gain == best_gain && score > best_score) ||
          (score == best_score && gain == best_gain &&
           candidate->rows.size() < best->rows.size())) {
        best = candidate;
        best_gain = gain;
        best_score = score;
      }
    }
    if (best == nullptr) {
      throw std::runtime_error(
          "dynamic query is not covered by saved views; rebuild with "
          "materialize_variable_attachments=true");
    }
    selected.push_back(best);
    selected_ids.insert(best->id);
    for (const auto &variable : best->scope) {
      uncovered.erase(variable);
    }
  }

  if (stats != nullptr) {
    std::set<std::string> components;
    for (const auto *view : selected) {
      stats->view_ids.push_back(view->id);
      if (view->scope.size() > 1) ++stats->selected_multivariable_views;
      if (view->tree_id == "@attachment") ++stats->selected_attachment_views;
      components.insert(view->tree_id.empty() || view->tree_id == "@attachment" ? view->id : view->tree_id);
    }
    stats->component_ids.assign(components.begin(), components.end());
  }
  if (plan_only) {
    BoundaryJoinResult plan;
    plan.query_id = query.id; plan.global_scope = global_scope; plan.output_scope = output_scope;
    return plan;
  }
  auto first = std::min_element(selected.begin(), selected.end(),
                                [](const auto *left, const auto *right) {
                                  return left->rows.size() < right->rows.size();
                                });
  std::vector<BoundaryRow> rows = (*first)->rows;
  selected.erase(first);
  while (!selected.empty() && !rows.empty()) {
    const auto current_variables = variableSchema(rows);
    const auto current_roles = roleSchema(rows);
    auto best = selected.begin();
    std::size_t best_shared = 0;
    for (auto candidate = selected.begin(); candidate != selected.end();
         ++candidate) {
      std::size_t shared = 0;
      for (const auto &variable : (*candidate)->scope) {
        shared += current_variables.count(variable);
      }
      for (const auto &role : roleSchema((*candidate)->rows)) {
        shared += current_roles.count(role);
      }
      if (shared > best_shared ||
          (shared == best_shared &&
           (*candidate)->rows.size() < (*best)->rows.size())) {
        best = candidate;
        best_shared = shared;
      }
    }
    rows = joinRows(rows, **best, stats);
    selected.erase(best);
  }

  BoundaryJoinResult result;
  result.query_id = query.id;
  result.global_scope = global_scope;
  result.output_scope = output_scope;
  for (const auto &row : rows) {
    ValueTuple values;
    for (const auto &variable : output_scope) {
      const auto *binding = findBinding(row, variable);
      if (binding == nullptr) {
        throw std::runtime_error(
            "dynamic Boundary join misses output variable " + variable);
      }
      values.push_back(binding->raw_value);
    }
    result.value_counts[values] =
        checkedAdd(result.value_counts[values], row.multiplicity);
    result.exact_binding_count =
        checkedAdd(result.exact_binding_count, row.multiplicity);
  }
  if (retain_rows) {
    result.rows = std::move(rows);
  }
  return result;
}

} // namespace whatif
