#pragma once

#include "causal_index_v5/BoundaryJoinIndex.h"

#include <cstddef>
#include <vector>

namespace causal_index_v5 {

struct ParentTreePlanningStats {
  std::size_t atomic_factor_scopes = 0;
  std::size_t unique_atomic_factor_scopes = 0;
  std::size_t oversized_atomic_factors = 0;
  std::size_t reachable_causal_pairs = 0;
  std::size_t identifiable_query_scopes = 0;
  std::size_t localizable_query_scopes = 0;
  std::size_t oversized_query_scopes = 0;
  std::size_t maximal_query_seeds = 0;
  std::size_t covered_localizable_query_scopes = 0;
  std::size_t preserved_oversized_trees = 0;
  std::size_t subset_absorptions = 0;
  std::size_t bounded_merges = 0;
  std::size_t cost_merge_candidates = 0;
  std::size_t cost_rejected_merges = 0;
  double estimated_initial_tree_cost = 0.0;
  double estimated_final_tree_cost = 0.0;
  double estimated_boundary_cost = 0.0;
  double estimated_total_cost = 0.0;
};

struct ParentTreePlan {
  std::vector<ParentTreeSpec> trees;
  ParentTreePlanningStats stats;
};

class ParentTreePlanner {
public:
  // Produces query-aware physical Parent Trees. In addition to atomic factors,
  // every identifiable
  // PathClosure(X,Y) plus its deterministic adjustment is used as a required
  // hyperedge when it fits within the width bound. An atomic parent factor
  // wider than maximum_width remains one indivisible Parent Tree; K limits
  // only unions of smaller factors. Subsets are absorbed before incomparable
  // scopes are merged with a cardinality-aware estimate of prefix execution,
  // materialization, and repeated per-Tree setup. A merge is accepted only
  // when it lowers estimated work. Oversized query scopes spanning several
  // Trees are served by exact Boundary joins.
  static ParentTreePlan
  plan(const std::vector<VariableInput> &variables,
       const std::vector<CausalEdge> &causal_edges,
       const std::vector<std::vector<VariableId>> &atomic_parent_scopes,
       std::size_t maximum_width);
};

} // namespace causal_index_v5
