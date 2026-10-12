#include "whatif/BoundaryJoinIndex.h"

#include <algorithm>
#include <deque>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace whatif {
namespace {

template <typename T>
void appendUnique(std::vector<T> &values, const T &value) {
  if (std::find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
}

std::string scopeKey(const std::string &tree_id,
                     const std::vector<VariableId> &scope) {
  std::string key = tree_id;
  key += "::";
  for (std::size_t i = 0; i < scope.size(); ++i) {
    if (i != 0) {
      key += ',';
    }
    key += scope[i];
  }
  return key;
}

bool intersects(const std::vector<VariableId> &left,
                const std::vector<VariableId> &right) {
  for (const auto &value : left) {
    if (std::find(right.begin(), right.end(), value) != right.end()) {
      return true;
    }
  }
  return false;
}

} // namespace

BoundaryRequirementPlanner::BoundaryRequirementPlanner(
    std::vector<VariableInput> variables, std::vector<CausalEdge> causal_edges,
    std::vector<ParentTreeSpec> parent_trees)
    : variables_(std::move(variables)), causal_edges_(std::move(causal_edges)),
      parent_trees_(std::move(parent_trees)) {}

std::string
BoundaryRequirementPlanner::reachablePairQueryId(const VariableId &treatment,
                                                 const VariableId &outcome) {
  return "do:" + treatment + "->" + outcome;
}

std::string BoundaryRequirementPlanner::exactScopeQueryId(
    const std::vector<VariableId> &scope) {
  auto canonical = scope;
  std::sort(canonical.begin(), canonical.end());
  canonical.erase(std::unique(canonical.begin(), canonical.end()),
                  canonical.end());
  std::string id = "exact-scope:";
  for (const auto &variable : canonical) {
    id += std::to_string(variable.size());
    id.push_back(':');
    id += variable;
    id.push_back('|');
  }
  return id;
}

BoundaryPlan BoundaryRequirementPlanner::plan(
    const std::vector<DirectedQuerySpec> &explicit_queries,
    bool include_all_reachable_pairs, bool prune_redundant_views,
    const std::vector<ExactScopeSpec> &exact_scopes) const {
  if (variables_.empty()) {
    throw std::runtime_error("Boundary planner needs causal variables");
  }
  if (parent_trees_.empty()) {
    throw std::runtime_error("Boundary planner needs Parent Trees");
  }

  const std::size_t n = variables_.size();
  std::unordered_map<VariableId, std::size_t> variable_index;
  std::vector<std::unordered_set<RoleId>> variable_roles(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (variables_[i].id.empty() ||
        !variable_index.emplace(variables_[i].id, i).second) {
      throw std::runtime_error("duplicate or empty causal variable ID");
    }
    variable_roles[i].insert(variables_[i].pattern_roles.begin(),
                             variables_[i].pattern_roles.end());
  }

  std::unordered_set<std::string> tree_ids;
  for (const auto &tree : parent_trees_) {
    if (tree.id.empty() || !tree_ids.insert(tree.id).second) {
      throw std::runtime_error("duplicate or empty Parent Tree ID");
    }
    if (tree.variables.empty()) {
      throw std::runtime_error("Parent Tree " + tree.id + " has no variables");
    }
    std::unordered_set<VariableId> local;
    for (const auto &variable : tree.variables) {
      if (variable_index.count(variable) == 0) {
        throw std::runtime_error("Parent Tree " + tree.id +
                                 " references unknown variable " + variable);
      }
      if (!local.insert(variable).second) {
        throw std::runtime_error("Parent Tree " + tree.id +
                                 " repeats variable " + variable);
      }
    }
  }

  std::vector<std::vector<std::size_t>> children(n);
  std::vector<std::size_t> indegree(n, 0);
  std::set<std::pair<std::size_t, std::size_t>> unique_edges;
  for (const auto &edge : causal_edges_) {
    const auto parent = variable_index.find(edge.parent);
    const auto child = variable_index.find(edge.child);
    if (parent == variable_index.end() || child == variable_index.end()) {
      throw std::runtime_error("causal edge references an unknown variable");
    }
    if (parent->second == child->second) {
      throw std::runtime_error("causal DAG cannot contain a self edge");
    }
    if (unique_edges.emplace(parent->second, child->second).second) {
      children[parent->second].push_back(child->second);
      indegree[child->second] += 1;
    }
  }

  std::deque<std::size_t> ready;
  for (std::size_t i = 0; i < n; ++i) {
    if (indegree[i] == 0) {
      ready.push_back(i);
    }
  }
  std::vector<std::size_t> topo;
  while (!ready.empty()) {
    const auto current = ready.front();
    ready.pop_front();
    topo.push_back(current);
    auto next = children[current];
    std::sort(next.begin(), next.end());
    for (const auto child : next) {
      if (--indegree[child] == 0) {
        ready.push_back(child);
      }
    }
  }
  if (topo.size() != n) {
    throw std::runtime_error("causal graph must be acyclic");
  }
  std::vector<std::size_t> topo_rank(n, 0);
  for (std::size_t rank = 0; rank < topo.size(); ++rank) {
    topo_rank[topo[rank]] = rank;
  }

  std::vector<std::vector<std::uint8_t>> reachable(
      n, std::vector<std::uint8_t>(n, 0));
  for (std::size_t i = 0; i < n; ++i) {
    reachable[i][i] = 1;
  }
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    const auto source = *it;
    for (const auto child : children[source]) {
      reachable[source][child] = 1;
      for (std::size_t target = 0; target < n; ++target) {
        if (reachable[child][target]) {
          reachable[source][target] = 1;
        }
      }
    }
  }

  std::vector<DirectedQuerySpec> workload;
  std::unordered_set<std::string> query_ids;
  if (include_all_reachable_pairs) {
    for (const auto source : topo) {
      for (const auto target : topo) {
        if (source == target || !reachable[source][target]) {
          continue;
        }
        DirectedQuerySpec query;
        query.id =
            reachablePairQueryId(variables_[source].id, variables_[target].id);
        query.treatments = {variables_[source].id};
        query.outcomes = {variables_[target].id};
        query_ids.insert(query.id);
        workload.push_back(std::move(query));
      }
    }
  }
  for (const auto &query : explicit_queries) {
    if (query.id.empty()) {
      throw std::runtime_error("explicit Boundary query needs an ID");
    }
    if (!query_ids.insert(query.id).second) {
      throw std::runtime_error("duplicate Boundary query ID: " + query.id);
    }
    workload.push_back(query);
  }
  std::unordered_map<std::string, std::vector<VariableId>> exact_by_id;
  for (const auto &exact : exact_scopes) {
    const std::string id =
        exact.id.empty() ? exactScopeQueryId(exact.scope) : exact.id;
    if (exact.scope.empty()) {
      throw std::runtime_error("exact Boundary scope cannot be empty");
    }
    if (!query_ids.insert(id).second) {
      throw std::runtime_error("duplicate Boundary query ID: " + id);
    }
    exact_by_id.emplace(id, exact.scope);
    DirectedQuerySpec placeholder;
    placeholder.id = id;
    workload.push_back(std::move(placeholder));
  }

  struct LocalView {
    std::size_t tree = 0;
    std::vector<VariableId> scope;
    std::unordered_set<RoleId> roles;
  };

  auto canonicalize = [&](const std::vector<VariableId> &values) {
    std::vector<VariableId> out;
    std::unordered_set<VariableId> seen;
    for (const auto &value : values) {
      if (variable_index.count(value) == 0) {
        throw std::runtime_error("Boundary query references unknown variable " +
                                 value);
      }
      if (seen.insert(value).second) {
        out.push_back(value);
      }
    }
    std::sort(out.begin(), out.end(), [&](const auto &left, const auto &right) {
      return topo_rank[variable_index.at(left)] <
             topo_rank[variable_index.at(right)];
    });
    return out;
  };

  auto local_views_connected = [&](const std::vector<LocalView> &views,
                                   const std::vector<std::uint8_t> &keep) {
    std::size_t first = views.size();
    std::size_t kept = 0;
    for (std::size_t i = 0; i < views.size(); ++i) {
      if (keep[i]) {
        first = std::min(first, i);
        kept += 1;
      }
    }
    if (kept <= 1) {
      return kept == 1;
    }
    std::vector<std::uint8_t> visited(views.size(), 0);
    std::deque<std::size_t> queue{first};
    visited[first] = 1;
    std::size_t reached = 0;
    while (!queue.empty()) {
      const auto left = queue.front();
      queue.pop_front();
      reached += 1;
      for (std::size_t right = 0; right < views.size(); ++right) {
        if (!keep[right] || visited[right] || left == right) {
          continue;
        }
        bool shared = intersects(views[left].scope, views[right].scope);
        if (!shared) {
          for (const auto &role : views[left].roles) {
            if (views[right].roles.count(role) != 0) {
              shared = true;
              break;
            }
          }
        }
        if (shared) {
          visited[right] = 1;
          queue.push_back(right);
        }
      }
    }
    return reached == kept;
  };

  BoundaryPlan result;
  std::unordered_map<std::string, std::size_t> requirement_index;
  for (const auto &raw_query : workload) {
    DirectedQuerySpec query = raw_query;
    const auto exact = exact_by_id.find(query.id);
    const bool is_exact_scope = exact != exact_by_id.end();
    std::vector<VariableId> requested_exact_scope;
    if (is_exact_scope) {
      requested_exact_scope = canonicalize(exact->second);
    } else {
      query.treatments = canonicalize(query.treatments);
      query.outcomes = canonicalize(query.outcomes);
      query.conditioning_variables = canonicalize(query.conditioning_variables);
      query.required_variables = canonicalize(query.required_variables);
      if (query.treatments.empty() || query.outcomes.empty()) {
        throw std::runtime_error(
            "Boundary query needs treatment and outcome variables");
      }
    }

    std::unordered_set<std::size_t> global;
    for (const auto &variable : requested_exact_scope) {
      global.insert(variable_index.at(variable));
    }
    for (const auto &treatment : query.treatments) {
      global.insert(variable_index.at(treatment));
    }
    for (const auto &outcome : query.outcomes) {
      global.insert(variable_index.at(outcome));
    }
    for (const auto &condition : query.conditioning_variables) {
      global.insert(variable_index.at(condition));
    }
    for (const auto &required : query.required_variables) {
      global.insert(variable_index.at(required));
    }
    for (const auto &treatment : query.treatments) {
      const auto source = variable_index.at(treatment);
      for (const auto &outcome : query.outcomes) {
        const auto target = variable_index.at(outcome);
        if (!reachable[source][target]) {
          continue;
        }
        for (std::size_t middle = 0; middle < n; ++middle) {
          if (reachable[source][middle] && reachable[middle][target]) {
            global.insert(middle);
          }
        }
      }
    }

    std::vector<VariableId> global_scope;
    for (const auto variable : topo) {
      if (global.count(variable) != 0) {
        global_scope.push_back(variables_[variable].id);
      }
    }

    bool tree_local = false;
    for (const auto &tree : parent_trees_) {
      const std::unordered_set<VariableId> members(tree.variables.begin(),
                                                   tree.variables.end());
      if (std::all_of(
              global_scope.begin(), global_scope.end(),
              [&](const auto &value) { return members.count(value) != 0; })) {
        tree_local = true;
        break;
      }
    }
    if (tree_local) {
      BoundaryQueryPlan query_plan;
      query_plan.query_id = query.id;
      query_plan.treatments = query.treatments;
      query_plan.outcomes = query.outcomes;
      query_plan.conditioning_variables = query.conditioning_variables;
      query_plan.required_variables = query.required_variables;
      query_plan.global_scope = global_scope;
      result.queries.push_back(std::move(query_plan));
      continue;
    }

    std::vector<LocalView> local_views;
    for (std::size_t tree_index = 0; tree_index < parent_trees_.size();
         ++tree_index) {
      LocalView local;
      local.tree = tree_index;
      for (const auto &variable : parent_trees_[tree_index].variables) {
        if (global.count(variable_index.at(variable)) == 0) {
          continue;
        }
        local.scope.push_back(variable);
        const auto index = variable_index.at(variable);
        local.roles.insert(variable_roles[index].begin(),
                           variable_roles[index].end());
      }
      if (!local.scope.empty()) {
        local_views.push_back(std::move(local));
      }
    }
    if (local_views.size() < 2) {
      throw std::runtime_error(
          "cross-tree query " + query.id +
          " is not covered by connected Parent Tree views");
    }

    std::vector<std::uint8_t> keep(local_views.size(), 1);
    auto valid_cover = [&](const std::vector<std::uint8_t> &candidate) {
      std::unordered_set<VariableId> covered;
      for (std::size_t i = 0; i < local_views.size(); ++i) {
        if (candidate[i]) {
          covered.insert(local_views[i].scope.begin(),
                         local_views[i].scope.end());
        }
      }
      if (!std::all_of(
              global_scope.begin(), global_scope.end(),
              [&](const auto &value) { return covered.count(value) != 0; })) {
        return false;
      }
      return local_views_connected(local_views, candidate);
    };
    if (prune_redundant_views) {
      for (std::size_t i = 0; i < local_views.size(); ++i) {
        keep[i] = 0;
        if (!valid_cover(keep)) {
          keep[i] = 1;
        }
      }
    }
    if (!valid_cover(keep)) {
      throw std::runtime_error("cross-tree query " + query.id +
                               " has disconnected materialized boundaries");
    }

    BoundaryQueryPlan query_plan;
    query_plan.query_id = query.id;
    query_plan.treatments = query.treatments;
    query_plan.outcomes = query.outcomes;
    query_plan.conditioning_variables = query.conditioning_variables;
    query_plan.required_variables = query.required_variables;
    query_plan.global_scope = global_scope;
    for (std::size_t i = 0; i < local_views.size(); ++i) {
      if (!keep[i]) {
        continue;
      }
      const auto &local = local_views[i];
      const auto &tree = parent_trees_[local.tree];
      const std::string view_id = scopeKey(tree.id, local.scope);
      query_plan.view_ids.push_back(view_id);

      std::uint8_t flags = kBoundaryNone;
      const bool has_treatment = intersects(local.scope, query.treatments);
      const bool has_outcome = intersects(local.scope, query.outcomes);
      if (has_treatment) {
        flags |= kBoundarySourceSide;
      }
      if (has_outcome) {
        flags |= kBoundaryTargetSide;
      }
      if (!has_treatment && !has_outcome) {
        flags |= kBoundaryTransit;
      }

      auto found = requirement_index.find(view_id);
      if (found == requirement_index.end()) {
        BoundaryRequirement requirement;
        requirement.view_id = view_id;
        requirement.tree_id = tree.id;
        requirement.scope = local.scope;
        requirement.direction_flags = flags;
        requirement.query_ids.push_back(query.id);
        requirement_index.emplace(view_id, result.requirements.size());
        result.requirements.push_back(std::move(requirement));
      } else {
        auto &requirement = result.requirements[found->second];
        requirement.direction_flags |= flags;
        appendUnique(requirement.query_ids, query.id);
      }
    }
    result.queries.push_back(std::move(query_plan));
  }

  std::sort(result.requirements.begin(), result.requirements.end(),
            [](const auto &left, const auto &right) {
              if (left.tree_id != right.tree_id) {
                return left.tree_id < right.tree_id;
              }
              if (left.scope.size() != right.scope.size()) {
                return left.scope.size() < right.scope.size();
              }
              return left.scope < right.scope;
            });
  std::sort(result.queries.begin(), result.queries.end(),
            [](const auto &left, const auto &right) {
              return left.query_id < right.query_id;
            });
  return result;
}

} // namespace whatif
