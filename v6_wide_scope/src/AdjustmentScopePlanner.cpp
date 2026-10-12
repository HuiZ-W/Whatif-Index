#include "causal_index_v5/AdjustmentScopePlanner.h"

#include "causal_index_v5/CausalAdjustment.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace causal_index_v5 {
namespace {

std::string scopeKey(const std::vector<VariableId> &scope) {
  std::string key;
  for (const auto &variable : scope) {
    key += std::to_string(variable.size());
    key.push_back(':');
    key += variable;
    key.push_back('|');
  }
  return key;
}

std::vector<VariableId>
canonicalTreeScope(const ParentTreeSpec &tree,
                   const std::vector<VariableId> &scope) {
  const std::unordered_set<VariableId> wanted(scope.begin(), scope.end());
  if (wanted.size() != scope.size()) {
    throw std::runtime_error("adjustment scope repeats a causal variable");
  }
  std::vector<VariableId> canonical;
  for (const auto &variable : tree.variables) {
    if (wanted.count(variable) != 0) {
      canonical.push_back(variable);
    }
  }
  return canonical;
}

bool containsScope(const ParentTreeSpec &tree,
                   const std::vector<VariableId> &scope) {
  const std::unordered_set<VariableId> members(tree.variables.begin(),
                                               tree.variables.end());
  return std::all_of(scope.begin(), scope.end(), [&](const auto &variable) {
    return members.count(variable) != 0;
  });
}

std::vector<VariableInput>
plannerMetadata(const std::vector<VariableInput> &variables) {
  std::vector<VariableInput> result;
  result.reserve(variables.size());
  for (const auto &variable : variables) {
    result.push_back({variable.id, variable.pattern_roles, {}});
  }
  return result;
}

std::vector<ExactScopeSpec> exactScopesOutsideTrees(
    const std::vector<ParentTreeSpec> &trees,
    const std::vector<std::vector<VariableId>> &atomic_scopes) {
  std::map<std::string, ExactScopeSpec> unique;
  for (auto scope : atomic_scopes) {
    if (scope.empty() ||
        std::any_of(trees.begin(), trees.end(), [&](const auto &tree) {
          return containsScope(tree, scope);
        })) {
      continue;
    }
    std::sort(scope.begin(), scope.end());
    scope.erase(std::unique(scope.begin(), scope.end()), scope.end());
    ExactScopeSpec exact;
    exact.id = BoundaryRequirementPlanner::exactScopeQueryId(scope);
    exact.scope = std::move(scope);
    unique.emplace(exact.id, std::move(exact));
  }
  std::vector<ExactScopeSpec> result;
  result.reserve(unique.size());
  for (auto &entry : unique) {
    result.push_back(std::move(entry.second));
  }
  return result;
}

} // namespace

AdjustmentScopePlan AdjustmentScopePlanner::compilePairwise(
    const std::vector<VariableInput> &variables,
    const std::vector<CausalEdge> &causal_edges,
    const std::vector<ParentTreeSpec> &parent_trees,
    const std::vector<std::vector<VariableId>> &atomic_parent_scopes,
    bool prune_redundant_boundary_views) {
  if (variables.empty()) {
    throw std::runtime_error("adjustment planner needs causal variables");
  }
  if (parent_trees.empty()) {
    throw std::runtime_error("adjustment planner needs Parent Trees");
  }

  std::vector<VariableId> variable_ids;
  variable_ids.reserve(variables.size());
  std::unordered_set<VariableId> unique_variables;
  for (const auto &variable : variables) {
    if (variable.id.empty() || !unique_variables.insert(variable.id).second) {
      throw std::runtime_error("duplicate or empty causal variable ID");
    }
    variable_ids.push_back(variable.id);
  }

  std::vector<std::pair<std::string, std::string>> dag_edges;
  dag_edges.reserve(causal_edges.size());
  for (const auto &edge : causal_edges) {
    dag_edges.emplace_back(edge.parent, edge.child);
  }
  causal_index_v5::CausalDag dag(variable_ids, dag_edges);
  const causal_index_v5::CausalMask all =
      (causal_index_v5::CausalMask{1} << variables.size()) - 1;

  AdjustmentScopePlan result;
  result.exact_scope_workload =
      exactScopesOutsideTrees(parent_trees, atomic_parent_scopes);
  for (std::size_t treatment = 0; treatment < variables.size(); ++treatment) {
    const auto treatment_mask = causal_index_v5::CausalMask{1} << treatment;
    for (std::size_t outcome = 0; outcome < variables.size(); ++outcome) {
      if (treatment == outcome) {
        continue;
      }
      result.candidate_queries += 1;
      const auto outcome_mask = causal_index_v5::CausalMask{1} << outcome;
      const auto analysis = dag.analyzeAdjustmentWithin(
          treatment_mask, outcome_mask, all & ~(treatment_mask | outcome_mask));
      if (!analysis.has_proper_causal_path) {
        result.trivial_queries_pruned += 1;
        continue;
      }
      if (!analysis.adjustment_exists) {
        result.unidentifiable_queries += 1;
        continue;
      }
      DirectedQuerySpec query;
      query.id = BoundaryRequirementPlanner::reachablePairQueryId(
          variables[treatment].id, variables[outcome].id);
      query.treatments = {variables[treatment].id};
      query.outcomes = {variables[outcome].id};
      query.required_variables = dag.variablesOf(analysis.adjustment);
      if (analysis.adjustment == 0) {
        result.unadjusted_queries += 1;
      } else {
        result.adjusted_queries += 1;
      }
      result.query_workload.push_back(std::move(query));
      result.retained_queries += 1;
    }
  }

  BoundaryRequirementPlanner boundary_planner(plannerMetadata(variables),
                                              causal_edges, parent_trees);
  result.boundary_plan = boundary_planner.plan(result.query_workload, false,
                                               prune_redundant_boundary_views,
                                               result.exact_scope_workload);

  std::unordered_map<std::string,
                     std::map<std::string, std::vector<VariableId>>>
      retained_by_tree;
  auto retain_scope = [&](const std::vector<VariableId> &scope,
                          const std::string &description) {
    if (scope.empty()) {
      return;
    }
    for (const auto &tree : parent_trees) {
      if (!containsScope(tree, scope)) {
        continue;
      }
      auto canonical = canonicalTreeScope(tree, scope);
      if (canonical.size() != scope.size()) {
        throw std::logic_error("failed to canonicalize " + description);
      }
      retained_by_tree[tree.id].emplace(scopeKey(canonical),
                                        std::move(canonical));
      return;
    }
    // A split atomic factor is retained by its exact Boundary query.
  };

  // These are the non-prunable base of the causal factor index.
  for (const auto &scope : atomic_parent_scopes) {
    retain_scope(scope, "atomic parent factor");
  }
  for (const auto &variable : variable_ids) {
    retain_scope({variable}, "singleton marginal");
  }

  // Tree-local queries use an exact Cube over their complete binding source.
  // Cross-tree queries are represented by the Boundary views in boundary_plan.
  for (const auto &query : result.boundary_plan.queries) {
    if (query.view_ids.empty()) {
      retain_scope(query.global_scope, "Tree-local adjusted query scope");
    }
  }

  for (const auto &tree : parent_trees) {
    if (tree.variables.size() >= 63) {
      throw std::runtime_error(
          "Parent Tree is too wide for local lattice accounting");
    }
    result.full_local_cube_scopes +=
        (std::size_t{1} << tree.variables.size()) - 1;
    auto found = retained_by_tree.find(tree.id);
    if (found == retained_by_tree.end()) {
      continue;
    }
    auto &scopes = result.explicit_cube_scopes[tree.id];
    for (auto &item : found->second) {
      scopes.push_back(std::move(item.second));
    }
    std::sort(scopes.begin(), scopes.end(),
              [](const auto &left, const auto &right) {
                if (left.size() != right.size()) {
                  return left.size() < right.size();
                }
                return left < right;
              });
    result.retained_local_cube_scopes += scopes.size();
  }
  if (result.retained_local_cube_scopes > result.full_local_cube_scopes) {
    throw std::logic_error("retained adjustment scopes exceed full lattice");
  }
  result.pruned_local_cube_scopes =
      result.full_local_cube_scopes - result.retained_local_cube_scopes;
  return result;
}

AdjustmentScopePlan AdjustmentScopePlanner::compileSingleDo(
    const std::vector<VariableInput> &variables,
    const std::vector<CausalEdge> &causal_edges,
    const std::vector<ParentTreeSpec> &parent_trees,
    const std::vector<std::vector<VariableId>> &atomic_parent_scopes,
    const std::vector<DirectedQuerySpec> &explicit_queries,
    bool prune_redundant_boundary_views) {
  if (variables.empty()) {
    throw std::runtime_error("single-do planner needs causal variables");
  }
  if (parent_trees.empty()) {
    throw std::runtime_error("single-do planner needs Parent Trees");
  }

  std::vector<VariableId> variable_ids;
  variable_ids.reserve(variables.size());
  std::unordered_set<VariableId> unique_variables;
  for (const auto &variable : variables) {
    if (variable.id.empty() || !unique_variables.insert(variable.id).second) {
      throw std::runtime_error("duplicate or empty causal variable ID");
    }
    variable_ids.push_back(variable.id);
  }

  std::vector<std::pair<std::string, std::string>> dag_edges;
  dag_edges.reserve(causal_edges.size());
  for (const auto &edge : causal_edges) {
    dag_edges.emplace_back(edge.parent, edge.child);
  }
  causal_index_v5::CausalDag dag(variable_ids, dag_edges);
  const causal_index_v5::CausalMask all =
      (causal_index_v5::CausalMask{1} << variables.size()) - 1;

  auto canonicalize = [&](const std::vector<VariableId> &input) {
    const std::unordered_set<VariableId> wanted(input.begin(), input.end());
    if (wanted.size() != input.size()) {
      throw std::runtime_error("single-do query repeats a causal variable");
    }
    for (const auto &variable : input) {
      if (unique_variables.count(variable) == 0) {
        throw std::runtime_error(
            "single-do query references unknown variable " + variable);
      }
    }
    std::vector<VariableId> output;
    for (const auto &variable : variable_ids) {
      if (wanted.count(variable) != 0) {
        output.push_back(variable);
      }
    }
    return output;
  };

  AdjustmentScopePlan result;
  result.exact_scope_workload =
      exactScopesOutsideTrees(parent_trees, atomic_parent_scopes);
  std::unordered_set<std::string> query_ids;
  auto compile_query = [&](DirectedQuerySpec query) {
    result.candidate_queries += 1;
    if (query.id.empty() || !query_ids.insert(query.id).second) {
      throw std::runtime_error("duplicate or empty single-do query ID: " +
                               query.id);
    }
    query.treatments = canonicalize(query.treatments);
    query.outcomes = canonicalize(query.outcomes);
    query.conditioning_variables = canonicalize(query.conditioning_variables);
    query.required_variables = canonicalize(query.required_variables);
    if (query.treatments.size() != 1 || query.outcomes.size() != 1) {
      result.unsupported_query_shapes_pruned += 1;
      return;
    }

    const auto treatment_mask = dag.maskOf(query.treatments);
    const auto outcome_mask = dag.maskOf(query.outcomes);
    if ((treatment_mask & outcome_mask) != 0) {
      throw std::runtime_error(
          "single-do treatment and outcome must be disjoint");
    }
    auto condition_mask = dag.maskOf(query.conditioning_variables);
    if ((condition_mask & (treatment_mask | outcome_mask)) != 0) {
      throw std::runtime_error("conditioning variables must be disjoint from "
                               "treatments and outcomes");
    }

    if (!dag.reaches(treatment_mask, outcome_mask)) {
      // The intervention cannot causally reach the requested outcome, so this
      // is not a non-trivial causal-effect query in the supported family.
      result.trivial_queries_pruned += 1;
      return;
    }

    const auto reduced_condition = dag.reduceConditioningUnderIntervention(
        treatment_mask, outcome_mask, condition_mask);
    result.redundant_condition_variables_removed +=
        causalMaskCount(condition_mask) - causalMaskCount(reduced_condition);
    condition_mask = reduced_condition;
    query.conditioning_variables = dag.variablesOf(condition_mask);

    std::unordered_set<VariableId> specified_adjustment(
        query.required_variables.begin(), query.required_variables.end());
    specified_adjustment.insert(query.conditioning_variables.begin(),
                                query.conditioning_variables.end());

    causal_index_v5::CausalMask chosen_adjustment = 0;
    if (!specified_adjustment.empty()) {
      const std::vector<VariableId> specified(specified_adjustment.begin(),
                                              specified_adjustment.end());
      chosen_adjustment = dag.maskOf(specified);
      if (!dag.isValidAdjustment(treatment_mask, outcome_mask,
                                 chosen_adjustment)) {
        result.unidentifiable_queries += 1;
        return;
      }
    } else {
      const auto analysis = dag.analyzeAdjustmentWithin(
          treatment_mask, outcome_mask, all & ~(treatment_mask | outcome_mask));
      if (!analysis.adjustment_exists) {
        result.unidentifiable_queries += 1;
        return;
      }
      chosen_adjustment = analysis.adjustment;
    }

    query.required_variables.clear();
    const std::unordered_set<VariableId> conditioned(
        query.conditioning_variables.begin(),
        query.conditioning_variables.end());
    for (const auto &variable : dag.variablesOf(chosen_adjustment)) {
      if (conditioned.count(variable) == 0) {
        query.required_variables.push_back(variable);
      }
    }

    if (chosen_adjustment == 0) {
      result.unadjusted_queries += 1;
    } else {
      result.adjusted_queries += 1;
    }
    result.query_workload.push_back(std::move(query));
    result.retained_queries += 1;
  };

  // Cover every ordered singleton treatment/outcome pair. The deterministic
  // adjustment selected for a pair may contain any number of variables.
  // Additional conditional workloads are accepted through explicit_queries
  // and normalized before their materialized scopes are planned.
  for (std::size_t treatment = 0; treatment < variables.size(); ++treatment) {
    for (std::size_t outcome = 0; outcome < variables.size(); ++outcome) {
      if (treatment == outcome) {
        continue;
      }
      DirectedQuerySpec query;
      query.id = BoundaryRequirementPlanner::reachablePairQueryId(
          variables[treatment].id, variables[outcome].id);
      query.treatments = {variables[treatment].id};
      query.outcomes = {variables[outcome].id};
      compile_query(std::move(query));
    }
  }
  for (const auto &query : explicit_queries) {
    compile_query(query);
  }

  BoundaryRequirementPlanner boundary_planner(plannerMetadata(variables),
                                              causal_edges, parent_trees);
  result.boundary_plan = boundary_planner.plan(result.query_workload, false,
                                               prune_redundant_boundary_views,
                                               result.exact_scope_workload);

  std::unordered_map<std::string,
                     std::map<std::string, std::vector<VariableId>>>
      retained_by_tree;
  auto retain_exact_scope = [&](const std::vector<VariableId> &scope) {
    if (scope.empty()) {
      return;
    }
    for (const auto &tree : parent_trees) {
      if (!containsScope(tree, scope)) {
        continue;
      }
      auto canonical = canonicalTreeScope(tree, scope);
      retained_by_tree[tree.id].emplace(scopeKey(canonical),
                                        std::move(canonical));
      return;
    }
    // A split atomic factor is retained by its exact Boundary query.
  };

  for (const auto &scope : atomic_parent_scopes) {
    retain_exact_scope(scope);
  }
  for (const auto &variable : variable_ids) {
    retain_exact_scope({variable});
  }

  // Every compiled tree-local query keeps its exact path/adjustment scope.
  // Explicit conditions may add a disconnected pre-treatment variable.
  for (const auto &query : result.boundary_plan.queries) {
    if (query.view_ids.empty()) {
      retain_exact_scope(query.global_scope);
    }
  }

  for (const auto &tree : parent_trees) {
    if (tree.variables.size() >= 63) {
      throw std::runtime_error(
          "Parent Tree is too wide for single-do scope accounting");
    }
    const std::uint64_t limit = std::uint64_t{1} << tree.variables.size();
    result.full_local_cube_scopes += static_cast<std::size_t>(limit - 1);

    auto &retained = retained_by_tree[tree.id];

    // Enumerate the bounded local lattice once at build time. A scope is a
    // reusable adjustment Cube when it is exactly a directed X-to-Y path
    // closure plus a valid, non-redundant adjustment set of any width.
    for (std::uint64_t local_mask = 1; local_mask < limit; ++local_mask) {
      std::vector<VariableId> scope;
      for (std::size_t bit = 0; bit < tree.variables.size(); ++bit) {
        if ((local_mask & (std::uint64_t{1} << bit)) != 0) {
          scope.push_back(tree.variables[bit]);
        }
      }
      if (scope.size() < 2) {
        continue;
      }
      const auto scope_mask = dag.maskOf(scope);
      bool valid_adjustment_scope = false;
      for (std::size_t treatment = 0;
           treatment < variables.size() && !valid_adjustment_scope;
           ++treatment) {
        const auto treatment_mask = causal_index_v5::CausalMask{1} << treatment;
        if ((scope_mask & treatment_mask) == 0) {
          continue;
        }
        for (std::size_t outcome = 0; outcome < variables.size(); ++outcome) {
          const auto outcome_mask = causal_index_v5::CausalMask{1} << outcome;
          if (treatment == outcome || (scope_mask & outcome_mask) == 0 ||
              !dag.reaches(treatment_mask, outcome_mask)) {
            continue;
          }
          const auto path = dag.causalPathClosure(treatment_mask, outcome_mask);
          if ((path & ~scope_mask) != 0) {
            continue;
          }
          const auto adjustment = scope_mask & ~path;
          if (dag.reduceConditioningUnderIntervention(
                  treatment_mask, outcome_mask, adjustment) != adjustment ||
              !dag.isValidAdjustment(treatment_mask, outcome_mask,
                                     adjustment)) {
            continue;
          }
          valid_adjustment_scope = true;
          break;
        }
      }
      if (valid_adjustment_scope) {
        result.valid_adjustment_local_cube_scopes += 1;
        retained.emplace(scopeKey(scope), std::move(scope));
      }
    }

    if (retained.empty()) {
      continue;
    }
    auto &scopes = result.explicit_cube_scopes[tree.id];
    for (auto &item : retained) {
      scopes.push_back(std::move(item.second));
    }
    std::sort(scopes.begin(), scopes.end(),
              [](const auto &left, const auto &right) {
                if (left.size() != right.size()) {
                  return left.size() < right.size();
                }
                return left < right;
              });
    result.retained_local_cube_scopes += scopes.size();
  }

  if (result.retained_local_cube_scopes > result.full_local_cube_scopes) {
    throw std::logic_error("retained single-do scopes exceed full lattice");
  }
  result.pruned_local_cube_scopes =
      result.full_local_cube_scopes - result.retained_local_cube_scopes;
  return result;
}

} // namespace causal_index_v5
