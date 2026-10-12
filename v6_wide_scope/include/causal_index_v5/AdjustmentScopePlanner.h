#pragma once

#include "causal_index_v5/BoundaryJoinIndex.h"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace causal_index_v5 {

// Compiled materialization workload for interventional queries. The plan keeps
// exact full-binding source scopes, not independent projections of a larger
// match universe.
struct AdjustmentScopePlan {
  std::vector<DirectedQuerySpec> query_workload;
  std::vector<ExactScopeSpec> exact_scope_workload;
  BoundaryPlan boundary_plan;
  std::unordered_map<std::string, std::vector<std::vector<VariableId>>>
      explicit_cube_scopes;

  std::size_t candidate_queries = 0;
  std::size_t retained_queries = 0;
  std::size_t adjusted_queries = 0;
  std::size_t unadjusted_queries = 0;
  std::size_t trivial_queries_pruned = 0;
  std::size_t unsupported_query_shapes_pruned = 0;
  std::size_t redundant_condition_variables_removed = 0;
  std::size_t unidentifiable_queries = 0;
  std::size_t full_local_cube_scopes = 0;
  std::size_t valid_adjustment_local_cube_scopes = 0;
  std::size_t retained_local_cube_scopes = 0;
  std::size_t pruned_local_cube_scopes = 0;
};

class AdjustmentScopePlanner {
public:
  // Compiles every ordered singleton-treatment/outcome pair having a proper
  // directed causal path. Its deterministic adjustment set may contain any
  // number of variables.
  static AdjustmentScopePlan compilePairwise(
      const std::vector<VariableInput> &variables,
      const std::vector<CausalEdge> &causal_edges,
      const std::vector<ParentTreeSpec> &parent_trees,
      const std::vector<std::vector<VariableId>> &atomic_parent_scopes,
      bool prune_redundant_boundary_views = true);

  // Compiles P(Y | do(X)) and P(Y | do(X), Z=z): treatment and outcome
  // are singletons, while Z may contain any number of adjustment variables.
  // Independently of the explicit query catalog, every bounded Parent Tree is
  // searched for all local path-closure plus valid-adjustment source scopes so
  // future multi-variable adjustment queries can hit prebuilt Cubes.
  // Redundant observations are removed in the intervention graph before
  // materialization. Atomic factors and singleton marginals are always kept.
  static AdjustmentScopePlan compileSingleDo(
      const std::vector<VariableInput> &variables,
      const std::vector<CausalEdge> &causal_edges,
      const std::vector<ParentTreeSpec> &parent_trees,
      const std::vector<std::vector<VariableId>> &atomic_parent_scopes,
      const std::vector<DirectedQuerySpec> &explicit_queries = {},
      bool prune_redundant_boundary_views = true);
};

} // namespace causal_index_v5
