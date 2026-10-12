#include "causal_index_v5/AdjustmentScopePlanner.h"
#include "causal_index_v5/BoundaryJoinIndex.h"
#include "causal_index_v5/ParentTreePlanner.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace causal_index_v5;

namespace {

VariableInput
variable(const std::string &id, std::vector<RoleId> roles,
         std::vector<causal_index_v4::MatchInstance> instances = {}) {
  return {id, std::move(roles), std::move(instances)};
}

causal_index_v4::MatchInstance instance(InstanceId id,
                                        std::vector<RoleBinding> roles,
                                        const std::string &value) {
  return {id, "all", std::move(roles), value};
}

std::set<std::string> requirementScopes(const BoundaryPlan &plan,
                                        const std::string &tree_id) {
  std::set<std::string> result;
  for (const auto &requirement : plan.requirements) {
    if (requirement.tree_id != tree_id) {
      continue;
    }
    std::string scope;
    for (const auto &variable_id : requirement.scope) {
      scope += variable_id;
    }
    result.insert(scope);
  }
  return result;
}

void testDirectedBoundaryMarking() {
  const std::vector<std::string> ids{"X", "Y", "Z", "W", "A",
                                     "B", "C", "D", "E", "F"};
  std::vector<VariableInput> variables;
  std::vector<CausalEdge> edges;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    variables.push_back(variable(ids[i], {"P", "R" + ids[i]}));
    if (i != 0) {
      edges.push_back({ids[i - 1], ids[i]});
    }
  }
  const std::vector<ParentTreeSpec> trees{
      {"T1", {"X", "Y", "Z", "W"}},
      {"T2", {"W", "A", "B", "C"}},
      {"T3", {"C", "D", "E", "F"}},
  };

  BoundaryRequirementPlanner planner(variables, edges, trees);
  const auto plan = planner.plan({}, true);
  assert(requirementScopes(plan, "T1") ==
         std::set<std::string>({"XYZW", "YZW", "ZW"}));
  assert(requirementScopes(plan, "T2") ==
         std::set<std::string>({"WA", "WAB", "WABC", "ABC", "BC"}));
  // CD is the exact incoming Boundary. DE is Tree-local and belongs in its
  // Cube; it cannot connect an upstream Tree because it omits separator C.
  assert(requirementScopes(plan, "T3") ==
         std::set<std::string>({"CD", "CDE", "CDEF"}));

  const auto query_id =
      BoundaryRequirementPlanner::reachablePairQueryId("X", "F");
  const auto query =
      std::find_if(plan.queries.begin(), plan.queries.end(),
                   [&](const auto &item) { return item.query_id == query_id; });
  assert(query != plan.queries.end());
  assert(query->global_scope == ids);
  assert(query->view_ids == std::vector<std::string>(
                                {"T1::X,Y,Z,W", "T2::W,A,B,C", "T3::C,D,E,F"}));
}

std::vector<VariableInput> detailedVariables() {
  return {
      variable("X", {"P", "Post"},
               {
                   instance(1, {{"P", "p1"}, {"Post", "post10"}}, "en"),
                   instance(2, {{"P", "p1"}, {"Post", "post20"}}, "zh"),
               }),
      variable(
          "Y", {"P", "Post", "Tag"},
          {
              instance(11, {{"P", "p1"}, {"Post", "post10"}, {"Tag", "tagA"}},
                       "sports"),
              instance(12, {{"P", "p1"}, {"Post", "post20"}, {"Tag", "tagB"}},
                       "music"),
              instance(13, {{"P", "p1"}, {"Post", "post10"}, {"Tag", "tagC"}},
                       "tech"),
              // Same nodes/value as y1, but a distinct full motif binding.
              instance(14, {{"P", "p1"}, {"Post", "post10"}, {"Tag", "tagA"}},
                       "sports"),
          }),
      variable(
          "Z", {"P", "Tag", "Forum"},
          {
              instance(21, {{"P", "p1"}, {"Tag", "tagA"}, {"Forum", "forum7"}},
                       "public"),
              instance(22, {{"P", "p1"}, {"Tag", "tagB"}, {"Forum", "forum8"}},
                       "private"),
              instance(23, {{"P", "p1"}, {"Tag", "tagC"}, {"Forum", "forum9"}},
                       "public"),
          }),
      variable(
          "W", {"P", "Forum", "Comment"},
          {
              instance(31,
                       {{"P", "p1"}, {"Forum", "forum7"}, {"Comment", "c100"}},
                       "positive"),
              instance(32,
                       {{"P", "p1"}, {"Forum", "forum7"}, {"Comment", "c101"}},
                       "negative"),
              instance(33,
                       {{"P", "p1"}, {"Forum", "forum8"}, {"Comment", "c200"}},
                       "neutral"),
              instance(34,
                       {{"P", "p1"}, {"Forum", "forum9"}, {"Comment", "c300"}},
                       "positive"),
          }),
      variable("U", {"P", "Comment"},
               {
                   instance(41, {{"P", "p1"}, {"Comment", "c100"}}, "low"),
                   instance(42, {{"P", "p1"}, {"Comment", "c300"}}, "high"),
               }),
  };
}

MaterializedBoundaryIndex
buildDetailedIndex(causal_index_v4::ExecutionSchedule schedule =
                       causal_index_v4::ExecutionSchedule::DfsPrefixTree) {
  const std::vector<CausalEdge> edges{
      {"X", "Y"}, {"Y", "Z"}, {"Z", "W"}, {"W", "U"}};
  const std::vector<ParentTreeSpec> trees{
      {"T1", {"X", "Y"}},
      {"T2", {"Y", "Z", "W"}},
      {"T3", {"W", "U"}},
  };
  BoundaryBuildConfig config;
  config.max_all_subset_width = 4;
  config.tree_executor.enable_anchor_partition = true;
  config.tree_executor.anchor_role = "P";
  config.tree_executor.enable_value_partition_join = false;
  config.tree_executor.variable_order_policy =
      causal_index_v4::VariableOrderPolicy::PartialWorkloadGreedy;
  config.tree_executor.execution_schedule = schedule;
  return MaterializedBoundaryIndex::build(detailedVariables(), edges, trees,
                                          config);
}

void assertDetailedResult(const MaterializedBoundaryIndex &index) {
  assert(index.cubes().size() == 13); // 3 + 7 + 3 local subsets.
  const auto &xy_cube = index.cube("T1", {"X", "Y"});
  assert(xy_cube.counts.at({"en", "sports"}) == 2);
  assert(xy_cube.counts.at({"zh", "music"}) == 1);

  const auto &bridge = index.boundaryView("T2::Y,Z,W");
  assert(bridge.rows.size() == 6);

  const auto query_id =
      BoundaryRequirementPlanner::reachablePairQueryId("X", "U");
  const auto result = index.execute(query_id, {}, true);
  assert(result.exact_binding_count == 3);
  assert(result.rows.size() == 3);
  assert(result.value_counts.at(
             {"en", "sports", "public", "positive", "low"}) == 2);
  assert(result.value_counts.at({"en", "tech", "public", "positive", "high"}) ==
         1);

  const auto local_query =
      index.execute(BoundaryRequirementPlanner::reachablePairQueryId("X", "Y"));
  assert(local_query.exact_binding_count == 4);
  assert(local_query.value_counts.at({"en", "sports"}) == 2);
}

void testTreeCubeAndExactBoundaryJoin() {
  const auto index = buildDetailedIndex();
  assertDetailedResult(index);

  const std::string snapshot = "/tmp/causal-index-v5-boundary-test.bin";
  const auto storage = index.saveWithStats(snapshot);
  assert(storage.total_bytes == std::filesystem::file_size(snapshot));
  assert(storage.total_bytes ==
         storage.metadata_bytes + storage.cube_bytes + storage.boundary_bytes);
  assert(storage.cube_bytes > 0);
  assert(storage.boundary_bytes > 0);
  const auto loaded = MaterializedBoundaryIndex::load(snapshot);
  assert(loaded.plan().requirements.size() == index.plan().requirements.size());
  assert(loaded.cubes().size() == index.cubes().size());
  assert(loaded.boundaryViews().size() == index.boundaryViews().size());
  assertDetailedResult(loaded);

  const auto independent = buildDetailedIndex(
      causal_index_v4::ExecutionSchedule::IndependentSources);
  assertDetailedResult(independent);
  assert(independent.cubes().size() == index.cubes().size());
  assert(independent.boundaryViews().size() == index.boundaryViews().size());
  std::remove(snapshot.c_str());
}

void testExplicitAdjustmentExpandedQuery() {
  std::vector<VariableInput> variables{
      variable("Z", {"P", "Shared"}),
      variable("X", {"P", "Shared", "RX"}),
      variable("Y", {"P", "Shared", "RY"}),
  };
  const std::vector<CausalEdge> edges{{"Z", "X"}, {"Z", "Y"}};
  const std::vector<ParentTreeSpec> trees{{"ZX", {"Z", "X"}},
                                          {"ZY", {"Z", "Y"}}};
  BoundaryRequirementPlanner planner(variables, edges, trees);
  DirectedQuerySpec query;
  query.id = "adjusted:X->Y";
  query.treatments = {"X"};
  query.outcomes = {"Y"};
  query.conditioning_variables = {"Z"};
  query.required_variables = {"Z"};
  const auto plan = planner.plan({query}, false);
  assert(plan.queries.size() == 1);
  assert(plan.queries.front().global_scope ==
         std::vector<VariableId>({"Z", "X", "Y"}));
  assert(plan.queries.front().conditioning_variables ==
         std::vector<VariableId>({"Z"}));
  assert(plan.queries.front().required_variables ==
         std::vector<VariableId>({"Z"}));
  assert(plan.queries.front().view_ids ==
         std::vector<std::string>({"ZX::Z,X", "ZY::Z,Y"}));
}

void testBoundaryPruningAblation() {
  std::vector<VariableInput> variables{variable("X", {"P", "RX"}),
                                       variable("Y", {"P", "RY"}),
                                       variable("Z", {"P", "RZ"})};
  const std::vector<CausalEdge> edges{{"X", "Y"}, {"Y", "Z"}};
  const std::vector<ParentTreeSpec> trees{
      {"T1", {"X", "Y"}}, {"T2", {"Y", "Z"}}, {"T3", {"Y"}}};
  BoundaryRequirementPlanner planner(variables, edges, trees);
  const auto pruned = planner.plan({}, true, true);
  const auto unpruned = planner.plan({}, true, false);
  const auto query_id =
      BoundaryRequirementPlanner::reachablePairQueryId("X", "Z");
  const auto find_query = [&](const BoundaryPlan &plan) -> const auto & {
    const auto found = std::find_if(
        plan.queries.begin(), plan.queries.end(),
        [&](const auto &query) { return query.query_id == query_id; });
    assert(found != plan.queries.end());
    return *found;
  };
  assert(find_query(pruned).view_ids ==
         std::vector<std::string>({"T1::X,Y", "T2::Y,Z"}));
  assert(find_query(unpruned).view_ids ==
         std::vector<std::string>({"T1::X,Y", "T2::Y,Z", "T3::Y"}));
  assert(unpruned.requirements.size() > pruned.requirements.size());
}

void testCrossTreeJoinUsesCanonicalRoleOverlap() {
  std::vector<VariableInput> variables{
      variable("X", {"P", "Shared"},
               {
                   instance(1, {{"P", "p1"}, {"Shared", "s1"}}, "x1"),
                   instance(2, {{"P", "p1"}, {"Shared", "s2"}}, "x2"),
               }),
      variable("Y", {"P", "Shared", "Tail"},
               {
                   instance(3, {{"P", "p1"}, {"Shared", "s1"}, {"Tail", "t1"}},
                            "y1"),
                   instance(4, {{"P", "p1"}, {"Shared", "s3"}, {"Tail", "t2"}},
                            "y2"),
               }),
  };
  BoundaryBuildConfig config;
  config.max_all_subset_width = 2;
  config.tree_executor.enable_anchor_partition = false;
  config.tree_executor.enable_value_partition_join = false;
  const auto index =
      MaterializedBoundaryIndex::build(std::move(variables), {{"X", "Y"}},
                                       {{"TX", {"X"}}, {"TY", {"Y"}}}, config);
  const auto result =
      index.execute(BoundaryRequirementPlanner::reachablePairQueryId("X", "Y"));
  assert(result.exact_binding_count == 1);
  assert(result.value_counts.at({"x1", "y1"}) == 1);
}

void testAdjustmentScopePruning() {
  const std::vector<VariableInput> variables{
      variable("Z", {"P", "RZ"},
               {instance(1, {{"P", "p1"}, {"RZ", "z1"}}, "z")}),
      variable("X", {"P", "RX"},
               {instance(2, {{"P", "p1"}, {"RX", "x1"}}, "x")}),
      variable("Y", {"P", "RY"},
               {instance(3, {{"P", "p1"}, {"RY", "y1"}}, "y")}),
  };
  const std::vector<CausalEdge> edges{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}};
  const std::vector<ParentTreeSpec> trees{{"T", {"Z", "X", "Y"}}};
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"Z"}, {"Z", "X"}, {"Z", "X", "Y"}};

  const auto scope_plan = AdjustmentScopePlanner::compilePairwise(
      variables, edges, trees, atomic_scopes);
  assert(scope_plan.full_local_cube_scopes == 7);
  assert(scope_plan.retained_local_cube_scopes == 5);
  assert(scope_plan.pruned_local_cube_scopes == 2);
  const auto &retained = scope_plan.explicit_cube_scopes.at("T");
  const auto has_scope = [&](std::vector<VariableId> expected) {
    const std::set<VariableId> wanted(expected.begin(), expected.end());
    return std::any_of(
        retained.begin(), retained.end(), [&](const auto &scope) {
          return std::set<VariableId>(scope.begin(), scope.end()) == wanted;
        });
  };
  assert(has_scope({"Z", "X", "Y"}));
  assert(!has_scope({"X", "Y"}));

  const auto query_id =
      BoundaryRequirementPlanner::reachablePairQueryId("X", "Y");
  const auto query = std::find_if(
      scope_plan.query_workload.begin(), scope_plan.query_workload.end(),
      [&](const auto &item) { return item.id == query_id; });
  assert(query != scope_plan.query_workload.end());
  assert(query->required_variables == std::vector<VariableId>({"Z"}));

  DirectedQuerySpec conditional_query;
  conditional_query.id = "conditional:X->Y|Z";
  conditional_query.treatments = {"X"};
  conditional_query.outcomes = {"Y"};
  conditional_query.conditioning_variables = {"Z"};
  const auto multi_scope_plan = AdjustmentScopePlanner::compileSingleDo(
      variables, edges, trees, atomic_scopes, {conditional_query});
  const auto compiled_conditional = std::find_if(
      multi_scope_plan.query_workload.begin(),
      multi_scope_plan.query_workload.end(),
      [&](const auto &item) { return item.id == conditional_query.id; });
  assert(compiled_conditional != multi_scope_plan.query_workload.end());
  assert(compiled_conditional->treatments == std::vector<VariableId>({"X"}));
  assert(compiled_conditional->conditioning_variables ==
         std::vector<VariableId>({"Z"}));
  assert(compiled_conditional->required_variables.empty());

  BoundaryBuildConfig pruned_config;
  pruned_config.cube_policy = LocalCubePolicy::ExplicitScopes;
  pruned_config.explicit_cube_scopes = scope_plan.explicit_cube_scopes;
  pruned_config.include_all_reachable_pairs = false;
  pruned_config.query_workload = scope_plan.query_workload;
  pruned_config.tree_executor.enable_anchor_partition = false;
  pruned_config.tree_executor.enable_value_partition_join = false;
  pruned_config.tree_executor.plan_order_from_unpruned_lattice = true;
  const auto pruned =
      MaterializedBoundaryIndex::build(variables, edges, trees, pruned_config);

  auto full_config = pruned_config;
  full_config.cube_policy = LocalCubePolicy::AllSubsets;
  full_config.max_all_subset_width = 3;
  full_config.explicit_cube_scopes.clear();
  const auto full =
      MaterializedBoundaryIndex::build(variables, edges, trees, full_config);

  assert(pruned.cubes().size() == 5);
  assert(full.cubes().size() == 7);
  const auto pruned_result = pruned.execute(query_id);
  const auto full_result = full.execute(query_id);
  assert(pruned_result.exact_binding_count == full_result.exact_binding_count);
  assert(pruned_result.value_counts == full_result.value_counts);
}

void testAdjustmentPruningPreservesCrossTreeQueries() {
  const std::vector<CausalEdge> edges{
      {"X", "Y"}, {"Y", "Z"}, {"Z", "W"}, {"W", "U"}};
  const std::vector<ParentTreeSpec> trees{
      {"T1", {"X", "Y"}},
      {"T2", {"Y", "Z", "W"}},
      {"T3", {"W", "U"}},
  };
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"X"}, {"X", "Y"}, {"Y", "Z"}, {"Z", "W"}, {"W", "U"}};
  const auto variables = detailedVariables();
  const auto scope_plan = AdjustmentScopePlanner::compilePairwise(
      variables, edges, trees, atomic_scopes);

  BoundaryBuildConfig pruned_config;
  pruned_config.cube_policy = LocalCubePolicy::ExplicitScopes;
  pruned_config.explicit_cube_scopes = scope_plan.explicit_cube_scopes;
  pruned_config.include_all_reachable_pairs = false;
  pruned_config.query_workload = scope_plan.query_workload;
  pruned_config.tree_executor.enable_anchor_partition = true;
  pruned_config.tree_executor.anchor_role = "P";
  pruned_config.tree_executor.enable_value_partition_join = false;
  pruned_config.tree_executor.plan_order_from_unpruned_lattice = true;
  const auto pruned =
      MaterializedBoundaryIndex::build(variables, edges, trees, pruned_config);

  auto full_config = pruned_config;
  full_config.cube_policy = LocalCubePolicy::AllSubsets;
  full_config.max_all_subset_width = 4;
  full_config.explicit_cube_scopes.clear();
  const auto full =
      MaterializedBoundaryIndex::build(variables, edges, trees, full_config);

  assert(pruned.cubes().size() < full.cubes().size());
  for (const auto &query : scope_plan.boundary_plan.queries) {
    const auto pruned_result = pruned.execute(query.query_id);
    const auto full_result = full.execute(query.query_id);
    assert(pruned_result.exact_binding_count ==
           full_result.exact_binding_count);
    assert(pruned_result.value_counts == full_result.value_counts);
  }
}

void testSingleDoCausalConnectedScopePruning() {
  const std::vector<VariableInput> variables{
      variable("X", {"P", "RX"},
               {instance(1, {{"P", "p1"}, {"RX", "x1"}}, "x")}),
      variable("Y", {"P", "RY"},
               {instance(2, {{"P", "p1"}, {"RY", "y1"}}, "y")}),
      variable("Z", {"P", "RZ"},
               {instance(3, {{"P", "p1"}, {"RZ", "z1"}}, "z")}),
      variable("W", {"P", "RW"},
               {instance(4, {{"P", "p1"}, {"RW", "w1"}}, "w")}),
  };
  const std::vector<CausalEdge> edges{{"X", "Y"}, {"Y", "Z"}, {"Z", "W"}};
  const std::vector<ParentTreeSpec> trees{{"T", {"X", "Y", "Z", "W"}}};
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"X"}, {"X", "Y"}, {"Y", "Z"}, {"Z", "W"}};

  DirectedQuerySpec conditional;
  conditional.id = "explicit:X->W";
  conditional.treatments = {"X"};
  conditional.outcomes = {"W"};

  DirectedQuerySpec multi;
  multi.id = "multi:{X,Y}->W";
  multi.treatments = {"X", "Y"};
  multi.outcomes = {"W"};

  const auto scope_plan = AdjustmentScopePlanner::compileSingleDo(
      variables, edges, trees, atomic_scopes, {conditional, multi});
  assert(scope_plan.full_local_cube_scopes == 15);
  assert(scope_plan.retained_local_cube_scopes == 10);
  assert(scope_plan.pruned_local_cube_scopes == 5);

  const auto &retained = scope_plan.explicit_cube_scopes.at("T");
  const auto has_scope = [&](std::vector<VariableId> expected) {
    const std::set<VariableId> wanted(expected.begin(), expected.end());
    return std::any_of(
        retained.begin(), retained.end(), [&](const auto &scope) {
          return std::set<VariableId>(scope.begin(), scope.end()) == wanted;
        });
  };
  // A chain retains exactly its contiguous causal segments. Disconnected
  // supersets are no longer materialized merely because they contain XY.
  assert(!has_scope({"X", "Z"}));
  assert(!has_scope({"X", "W"}));
  assert(!has_scope({"Y", "W"}));
  assert(!has_scope({"X", "Y", "W"}));
  assert(!has_scope({"X", "Z", "W"}));
  assert(has_scope({"X", "Y", "Z"}));
  assert(has_scope({"Y", "Z", "W"}));
  assert(has_scope({"X", "Y", "Z", "W"}));

  const auto conditional_plan = std::find_if(
      scope_plan.boundary_plan.queries.begin(),
      scope_plan.boundary_plan.queries.end(),
      [&](const auto &query) { return query.query_id == conditional.id; });
  assert(conditional_plan != scope_plan.boundary_plan.queries.end());
  assert(conditional_plan->treatments == std::vector<VariableId>({"X"}));
  assert(conditional_plan->conditioning_variables.empty());
  assert(std::set<VariableId>(conditional_plan->global_scope.begin(),
                              conditional_plan->global_scope.end()) ==
         std::set<VariableId>({"X", "Y", "Z", "W"}));

  const auto multi_plan = std::find_if(
      scope_plan.boundary_plan.queries.begin(),
      scope_plan.boundary_plan.queries.end(),
      [&](const auto &query) { return query.query_id == multi.id; });
  assert(multi_plan == scope_plan.boundary_plan.queries.end());
  assert(scope_plan.unsupported_query_shapes_pruned == 1);

  BoundaryBuildConfig pruned_config;
  pruned_config.cube_policy = LocalCubePolicy::ExplicitScopes;
  pruned_config.explicit_cube_scopes = scope_plan.explicit_cube_scopes;
  pruned_config.include_all_reachable_pairs = false;
  pruned_config.precompiled_boundary_plan = scope_plan.boundary_plan;
  pruned_config.tree_executor.enable_anchor_partition = false;
  pruned_config.tree_executor.enable_value_partition_join = false;
  const auto pruned =
      MaterializedBoundaryIndex::build(variables, edges, trees, pruned_config);

  auto full_config = pruned_config;
  full_config.cube_policy = LocalCubePolicy::AllSubsets;
  full_config.max_all_subset_width = 4;
  full_config.explicit_cube_scopes.clear();
  const auto full =
      MaterializedBoundaryIndex::build(variables, edges, trees, full_config);
  assert(pruned.execute(conditional.id).value_counts ==
         full.execute(conditional.id).value_counts);
}

void testSingleDoPrunesOnlyNoEffectForkScope() {
  const std::vector<VariableInput> variables{
      variable("Z", {"P", "RZ"},
               {instance(1, {{"P", "p1"}, {"RZ", "z1"}}, "z")}),
      variable("X", {"P", "RX"},
               {instance(2, {{"P", "p1"}, {"RX", "x1"}}, "x")}),
      variable("Y", {"P", "RY"},
               {instance(3, {{"P", "p1"}, {"RY", "y1"}}, "y")}),
  };
  const std::vector<CausalEdge> edges{{"Z", "X"}, {"Z", "Y"}};
  const std::vector<ParentTreeSpec> trees{{"T", {"Z", "X", "Y"}}};
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"Z"}, {"Z", "X"}, {"Z", "Y"}};
  const auto plan = AdjustmentScopePlanner::compileSingleDo(
      variables, edges, trees, atomic_scopes);
  const auto &retained = plan.explicit_cube_scopes.at("T");
  const auto has_scope = [&](const std::set<VariableId> &wanted) {
    return std::any_of(
        retained.begin(), retained.end(), [&](const auto &scope) {
          return std::set<VariableId>(scope.begin(), scope.end()) == wanted;
        });
  };
  assert(!has_scope({"X", "Y"}));
  assert(!has_scope({"Z", "X", "Y"}));
  assert(plan.retained_local_cube_scopes == 5);
  assert(plan.pruned_local_cube_scopes == 2);
}

void testMultiVariableAdjustmentIsRetainedFromAlternativeSeparator() {
  const std::vector<VariableInput> variables{
      variable("Z", {"P", "RZ"}), variable("A", {"P", "RA"}),
      variable("B", {"P", "RB"}), variable("X", {"P", "RX"}),
      variable("Y", {"P", "RY"})};
  const std::vector<CausalEdge> edges{{"Z", "A"}, {"A", "X"}, {"Z", "B"},
                                      {"B", "X"}, {"Z", "Y"}, {"X", "Y"}};
  const std::vector<ParentTreeSpec> trees{{"T", {"Z", "A", "B", "X", "Y"}}};
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"Z"}, {"Z", "A"}, {"Z", "B"}, {"A", "B", "X"}, {"Z", "X", "Y"}};

  const auto plan = AdjustmentScopePlanner::compileSingleDo(
      variables, edges, trees, atomic_scopes);
  const auto query_id =
      BoundaryRequirementPlanner::reachablePairQueryId("X", "Y");
  const auto query =
      std::find_if(plan.query_workload.begin(), plan.query_workload.end(),
                   [&](const auto &item) { return item.id == query_id; });
  assert(query != plan.query_workload.end());
  assert(query->required_variables == std::vector<VariableId>({"A", "B"}));
}

void testAllSingletonAdjustmentAlternativesAreRetained() {
  const std::vector<VariableInput> variables{
      variable("B", {"P", "RB"}), variable("A", {"P", "RA"}),
      variable("X", {"P", "RX"}), variable("Y", {"P", "RY"})};
  const std::vector<CausalEdge> edges{
      {"B", "A"}, {"A", "X"}, {"B", "Y"}, {"X", "Y"}};
  const std::vector<ParentTreeSpec> trees{{"T", {"B", "A", "X", "Y"}}};
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"B"}, {"A"}, {"X"}, {"X", "Y"}};

  const auto plan = AdjustmentScopePlanner::compileSingleDo(
      variables, edges, trees, atomic_scopes);
  const auto &retained = plan.explicit_cube_scopes.at("T");
  const auto has_scope = [&](const std::set<VariableId> &wanted) {
    return std::any_of(
        retained.begin(), retained.end(), [&](const auto &scope) {
          return std::set<VariableId>(scope.begin(), scope.end()) == wanted;
        });
  };
  assert(has_scope({"A", "X", "Y"}));
  assert(has_scope({"B", "X", "Y"}));
}

void testRequiredMultiVariableAdjustmentIsRetained() {
  const std::vector<VariableInput> variables{
      variable("C1", {"P", "RC1"}), variable("C2", {"P", "RC2"}),
      variable("X", {"P", "RX"}), variable("Y", {"P", "RY"})};
  const std::vector<CausalEdge> edges{
      {"C1", "X"}, {"C1", "Y"}, {"C2", "X"}, {"C2", "Y"}, {"X", "Y"}};
  const std::vector<ParentTreeSpec> trees{{"T", {"C1", "C2", "X", "Y"}}};
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"C1"}, {"C2"}, {"C1", "C2", "X"}, {"C1", "C2", "X", "Y"}};

  const auto plan = AdjustmentScopePlanner::compileSingleDo(
      variables, edges, trees, atomic_scopes);
  const auto query_id =
      BoundaryRequirementPlanner::reachablePairQueryId("X", "Y");
  const auto query =
      std::find_if(plan.query_workload.begin(), plan.query_workload.end(),
                   [&](const auto &item) { return item.id == query_id; });
  assert(query != plan.query_workload.end());
  assert(query->required_variables == std::vector<VariableId>({"C1", "C2"}));
}

void testRedundantConditionIsCanonicalized() {
  const std::vector<VariableInput> variables{
      variable("X", {"P", "RX"}), variable("Y", {"P", "RY"}),
      variable("Z", {"P", "RZ"}), variable("W", {"P", "RW"})};
  const std::vector<CausalEdge> edges{{"X", "Y"}, {"Y", "Z"}, {"Z", "W"}};
  const std::vector<ParentTreeSpec> trees{{"T", {"X", "Y", "Z", "W"}}};
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"X"}, {"X", "Y"}, {"Y", "Z"}, {"Z", "W"}};

  DirectedQuerySpec query;
  query.id = "redundant:X|do(Z)->W";
  query.treatments = {"Z"};
  query.outcomes = {"W"};
  query.conditioning_variables = {"X"};
  const auto plan = AdjustmentScopePlanner::compileSingleDo(
      variables, edges, trees, atomic_scopes, {query});

  const auto compiled = std::find_if(
      plan.boundary_plan.queries.begin(), plan.boundary_plan.queries.end(),
      [&](const auto &item) { return item.query_id == query.id; });
  assert(compiled != plan.boundary_plan.queries.end());
  assert(compiled->conditioning_variables.empty());
  assert(std::set<VariableId>(compiled->global_scope.begin(),
                              compiled->global_scope.end()) ==
         std::set<VariableId>({"Z", "W"}));
  assert(plan.redundant_condition_variables_removed == 1);
}

void testDynamicMultiAdjustmentUsesSavedAttachments() {
  const std::vector<VariableInput> variables{
      variable("A", {"P", "RA"},
               {instance(1, {{"P", "p1"}, {"RA", "a1"}}, "a-low"),
                instance(2, {{"P", "p2"}, {"RA", "a2"}}, "a-high")}),
      variable("B", {"P", "RB"},
               {instance(11, {{"P", "p1"}, {"RB", "b1"}}, "b-low"),
                instance(12, {{"P", "p2"}, {"RB", "b2"}}, "b-high")}),
      variable("X", {"P", "RX"},
               {instance(21, {{"P", "p1"}, {"RX", "x1"}}, "x-off"),
                instance(22, {{"P", "p2"}, {"RX", "x2"}}, "x-on")}),
      variable("Y", {"P", "RY"},
               {instance(31, {{"P", "p1"}, {"RY", "y1"}}, "y-no"),
                instance(32, {{"P", "p2"}, {"RY", "y2"}}, "y-yes")}),
  };
  const std::vector<CausalEdge> edges{
      {"A", "X"}, {"A", "Y"}, {"B", "X"}, {"B", "Y"}, {"X", "Y"}};
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"A"}, {"B"}, {"X"}, {"X", "Y"}};
  const std::vector<ParentTreeSpec> global_tree{
      {"global", {"A", "B", "X", "Y"}}};
  const auto scope_plan = AdjustmentScopePlanner::compileSingleDo(
      variables, edges, global_tree, atomic_scopes);
  const auto &retained = scope_plan.explicit_cube_scopes.at("global");
  assert(std::any_of(retained.begin(), retained.end(), [](const auto &scope) {
    return std::set<VariableId>(scope.begin(), scope.end()) ==
           std::set<VariableId>({"A", "B", "X", "Y"});
  }));
  assert(scope_plan.valid_adjustment_local_cube_scopes > 0);

  BoundaryBuildConfig split_config;
  split_config.include_all_reachable_pairs = false;
  split_config.max_all_subset_width = 2;
  split_config.materialize_variable_attachments = true;
  split_config.tree_executor.enable_anchor_partition = false;
  split_config.tree_executor.enable_value_partition_join = false;
  const std::vector<ParentTreeSpec> split_trees{{"TA", {"A", "X"}},
                                                {"TB", {"B", "Y"}}};
  const auto split = MaterializedBoundaryIndex::build(
      variables, edges, split_trees, split_config);
  assert(split.boundaryViews().size() == variables.size());
  assert(split.boundaryView(MaterializedBoundaryIndex::attachmentViewId("A"))
             .rows.size() == 2);

  BoundaryBuildConfig direct_config;
  direct_config.cube_policy = LocalCubePolicy::ExplicitScopes;
  direct_config.explicit_cube_scopes = scope_plan.explicit_cube_scopes;
  direct_config.include_all_reachable_pairs = false;
  direct_config.max_all_subset_width = 4;
  direct_config.tree_executor.enable_anchor_partition = false;
  direct_config.tree_executor.enable_value_partition_join = false;
  const auto direct = MaterializedBoundaryIndex::build(
      variables, edges, global_tree, direct_config);
  assert(!direct.cube("global", {"A", "B", "X", "Y"}).counts.empty());

  DirectedQuerySpec query;
  query.id = "dynamic:X->Y|A,B";
  query.treatments = {"X"};
  query.outcomes = {"Y"};
  query.conditioning_variables = {"A", "B"};
  const auto split_result = split.execute(query, {"Y"}, true);
  const auto direct_result = direct.execute(query, {"Y"});
  assert(split_result.exact_binding_count == 2);
  assert(split_result.rows.size() == 2);
  assert(split_result.value_counts == direct_result.value_counts);

  const std::string snapshot = "/tmp/causal-index-v5-multiz-test.bin";
  split.save(snapshot);
  const auto loaded = MaterializedBoundaryIndex::load(snapshot);
  const auto loaded_result = loaded.execute(query, {"Y"});
  assert(loaded_result.exact_binding_count == split_result.exact_binding_count);
  assert(loaded_result.value_counts == split_result.value_counts);
  std::remove(snapshot.c_str());
}

void testOversizedAtomicFactorIsPreserved() {
  const std::vector<VariableInput> variables{
      variable("Y", {"P", "RY"},
               {instance(1, {{"P", "p1"}, {"RY", "y1"}}, "y")}),
      variable("X1", {"P", "R1"},
               {instance(11, {{"P", "p1"}, {"R1", "a1"}}, "x1")}),
      variable("X2", {"P", "R2"},
               {instance(21, {{"P", "p1"}, {"R2", "b1"}}, "x2a"),
                instance(22, {{"P", "p1"}, {"R2", "b2"}}, "x2b")}),
      variable("X3", {"P", "R3"},
               {instance(31, {{"P", "p1"}, {"R3", "c1"}}, "x3")}),
  };
  const std::vector<CausalEdge> edges{{"X1", "Y"}, {"X2", "Y"}, {"X3", "Y"}};
  const std::vector<std::vector<VariableId>> atomic_scopes{
      {"Y", "X1", "X2", "X3"}, {"X1", "X3"}};

  const auto physical =
      ParentTreePlanner::plan(variables, edges, atomic_scopes, 2);
  assert(physical.stats.oversized_atomic_factors == 1);
  assert(physical.stats.preserved_oversized_trees == 1);
  assert(physical.stats.subset_absorptions >= 1);
  assert(physical.trees.size() == 1);
  assert(physical.trees.front().variables.size() == 4);

  BoundaryBuildConfig config;
  config.max_all_subset_width = 4;
  config.include_all_reachable_pairs = false;
  config.tree_executor.enable_anchor_partition = false;
  config.tree_executor.enable_value_partition_join = false;
  const auto index = MaterializedBoundaryIndex::build(
      variables, edges, physical.trees, config);
  const auto &factor = index.cube(physical.trees.front().id,
                                  {"Y", "X1", "X2", "X3"});
  std::uint64_t total = 0;
  for (const auto &[values, count] : factor.counts) {
    (void)values;
    total += count;
  }
  assert(total == 2);
  const auto &subset =
      index.cube(physical.trees.front().id, {"X1", "X3"});
  assert(subset.counts.size() == 1);
}

void testQueryAwareParentTreeCoversChainClosures() {
  const std::vector<VariableId> ids{"X", "Y", "Z", "W", "A", "B"};
  std::vector<VariableInput> variables;
  std::vector<CausalEdge> edges;
  std::vector<std::vector<VariableId>> atomic_scopes{{"X"}};
  for (std::size_t i = 0; i < ids.size(); ++i) {
    variables.push_back(variable(ids[i], {"P", "R" + ids[i]}));
    if (i != 0) {
      edges.push_back({ids[i - 1], ids[i]});
      atomic_scopes.push_back({ids[i - 1], ids[i]});
    }
  }

  const auto plan =
      ParentTreePlanner::plan(variables, edges, atomic_scopes, ids.size());
  assert(plan.stats.reachable_causal_pairs == 15);
  assert(plan.stats.identifiable_query_scopes == 15);
  assert(plan.stats.localizable_query_scopes == 15);
  assert(plan.stats.oversized_query_scopes == 0);
  assert(plan.stats.covered_localizable_query_scopes == 15);
  assert(plan.trees.size() == 1);
  assert(plan.trees.front().variables == ids);
}

void testCostAwareParentTreeMergeAcceptsOnlyUsefulUnion() {
  auto cost_variable = [](const std::string &id, const std::string &role,
                          InstanceId base) {
    return variable(
        id, {"P", role},
        {instance(base, {{"P", "p1"}, {role, id + "1"}}, "v1"),
         instance(base + 1, {{"P", "p2"}, {role, id + "2"}}, "v2")});
  };
  const std::vector<VariableInput> variables{
      cost_variable("A", "RA", 1), cost_variable("B", "RB", 11),
      cost_variable("C", "RC", 21), cost_variable("D", "RD", 31)};

  const auto overlapping = ParentTreePlanner::plan(
      variables, {}, {{"A", "B"}, {"A", "C"}}, 3);
  assert(overlapping.trees.size() == 1);
  assert(overlapping.stats.bounded_merges == 1);
  assert(overlapping.stats.estimated_final_tree_cost <
         overlapping.stats.estimated_initial_tree_cost);

  const auto disjoint = ParentTreePlanner::plan(
      variables, {}, {{"A", "B"}, {"C", "D"}}, 4);
  assert(disjoint.trees.size() == 2);
  assert(disjoint.stats.bounded_merges == 0);
  assert(disjoint.stats.cost_rejected_merges >= 1);
  assert(disjoint.stats.estimated_final_tree_cost ==
         disjoint.stats.estimated_initial_tree_cost);
}

void testExactScopePreservesPopulationAndCubeOnly() {
  const std::vector<VariableInput> variables{
      variable("X", {"P"}, {instance(1, {{"P", "p1"}}, "a"), instance(2, {{"P", "p2"}}, "b")}),
      variable("M", {"P"}, {instance(3, {{"P", "p1"}}, "m")}),
      variable("Y", {"P"}, {instance(4, {{"P", "p1"}}, "yes"), instance(5, {{"P", "p2"}}, "no")})};
  BoundaryBuildConfig config;
  config.materialize_variable_attachments = true;
  auto index = MaterializedBoundaryIndex::build(variables, {{"X", "M"}, {"M", "Y"}}, {{"T1", {"X", "M"}}, {"T2", {"M", "Y"}}}, config);
  OnlineScopeStats stats;
  const auto exact = index.executeExactScope({"fixed", {"X", "Y"}}, {}, false, false, &stats);
  assert(exact.global_scope == std::vector<VariableId>({"X", "Y"}));
  assert(exact.exact_binding_count == 2);
  assert(exact.value_counts.at({"a", "yes"}) == 1);
  assert(exact.value_counts.at({"b", "no"}) == 1);
  assert(!stats.cube_hit);
  assert(stats.component_ids.size() == 2);
  assert(stats.join_calls == 1);
  assert(stats.join_work_pairs == 2);
  const auto planned = index.executeExactScope({"plan", {"X", "Y"}}, {}, false, false, &stats, true);
  assert(stats.component_ids.size() == 2 && stats.join_calls == 0);
  assert(planned.value_counts.empty());
  const auto local = index.executeExactScope({"local", {"X"}}, {}, false, true, &stats);
  assert(local.exact_binding_count == 2);
  assert(stats.cube_hit && stats.join_calls == 0 && stats.component_ids.size() == 1);
  const auto check_lookup = [](const MaterializedBoundaryIndex &candidate) {
    OnlineScopeStats hit;
    const auto reversed = candidate.executeExactScope(
        {"reversed", {"M", "X"}}, {}, false, true, &hit);
    assert(hit.cube_hit && hit.join_calls == 0);
    assert(reversed.value_counts.at({"m", "a"}) == 1);
    const auto projected = candidate.executeExactScope(
        {"projected", {"X", "M"}}, {"X"}, false, true);
    assert(projected.value_counts.at({"a"}) == 1);
    candidate.executeExactScope({"duplicate-tree-scope", {"M"}}, {},
                                false, true, &hit, true);
    assert(hit.component_ids == std::vector<std::string>({"T1"}));
  };
  check_lookup(index);
  auto copied = index;
  check_lookup(copied);
  const std::string lookup_path = "/tmp/causal-v6-hash-lookup-test.bin";
  index.save(lookup_path);
  check_lookup(MaterializedBoundaryIndex::load(lookup_path));
  std::remove(lookup_path.c_str());
  bool unavailable = false;
  try { (void)index.executeExactScope({"full", {"X", "Y"}}, {}, false, true); }
  catch (const std::runtime_error &) { unavailable = true; }
  assert(unavailable);
  bool rejected = false;
  try { (void)index.executeExactScope({"bad", {"X", "X"}}); }
  catch (const std::runtime_error &) { rejected = true; }
  assert(rejected);
  const auto dynamic = index.execute(DirectedQuerySpec{"dynamic", {"X"}, {"Y"}, {}, {}});
  assert(dynamic.exact_binding_count == 1);
}

} // namespace

void testExactScopePrefersMaterializedBoundaryViews() {
  BoundaryBuildConfig config;
  config.include_all_reachable_pairs = false;
  config.materialize_variable_attachments = true;
  config.cube_policy = LocalCubePolicy::ExplicitScopes;
  const std::vector<ParentTreeSpec> trees{
      {"T1", {"X", "Y"}}, {"T2", {"Y", "Z", "W"}}, {"T3", {"W", "U"}}};
  const ExactScopeSpec query{"cross-tree", {"X", "Y", "Z", "W", "U"}};
  auto attachments = MaterializedBoundaryIndex::build(detailedVariables(), {}, trees, config);
  const auto expected = attachments.executeExactScope(query);
  auto augmented = attachments;
  augmented.materializeExactBoundaryViews({query});
  const auto augmented_result = augmented.executeExactScope(query);
  assert(augmented_result.value_counts == expected.value_counts);
  const auto view_count = augmented.boundaryViews().size();
  augmented.materializeExactBoundaryViews({query});
  assert(augmented.boundaryViews().size() == view_count);
  config.exact_scope_workload = {query};
  auto boundary = MaterializedBoundaryIndex::build(detailedVariables(), {}, trees, config);
  OnlineScopeStats stats;
  const auto actual = boundary.executeExactScope(query, {}, false, false, &stats);
  assert(actual.exact_binding_count == expected.exact_binding_count);
  assert(actual.value_counts == expected.value_counts);
  assert(stats.selected_multivariable_views > 0);
  assert(stats.selected_attachment_views < 5);
  const auto missing = boundary.executeExactScope({"subset", {"X", "U"}});
  const auto expected_missing = attachments.executeExactScope({"subset", {"X", "U"}});
  assert(missing.value_counts == expected_missing.value_counts);
}

int main() {
  {
    const std::vector<std::string> ids{"X", "Y", "Z", "W", "A", "B", "C", "D", "E", "F"};
    std::vector<VariableInput> vars;
    std::vector<CausalEdge> edges;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      vars.push_back(variable(ids[i], {"P"}, {instance(i + 1, {{"P", "p"}}, "v")}));
      if (i) edges.push_back({ids[i - 1], ids[i]});
    }
    BoundaryBuildConfig config;
    config.include_all_reachable_pairs = false;
    config.materialize_structural_boundaries = true;
    config.materialize_variable_attachments = true;
    const auto index = MaterializedBoundaryIndex::build(vars, edges,
        {{"T1", {"X", "Y", "Z", "W"}}, {"T2", {"W", "A", "B", "C"}},
         {"T3", {"C", "D", "E", "F"}}}, config);
    assert(index.plan().queries.empty());
    assert(requirementScopes(index.plan(), "T1") == std::set<std::string>({"XYZW", "YZW", "ZW"}));
    assert(requirementScopes(index.plan(), "T2") == std::set<std::string>({"WABC", "WAB", "WA", "ABC", "BC"}));
    assert(requirementScopes(index.plan(), "T3") == std::set<std::string>({"CDEF", "CDE", "CD"}));
    OnlineScopeStats stats;
    const auto answer = index.executeExactScope({"later-query", {"Y", "Z", "W", "A", "B"}}, {"Y", "B"}, false, false, &stats);
    assert(answer.exact_binding_count == 1 && stats.selected_multivariable_views == 2);
    assert(stats.selected_attachment_views == 0);
  }
  testExactScopePrefersMaterializedBoundaryViews();
  testExactScopePreservesPopulationAndCubeOnly();
  testDirectedBoundaryMarking();
  testTreeCubeAndExactBoundaryJoin();
  testExplicitAdjustmentExpandedQuery();
  testBoundaryPruningAblation();
  testCrossTreeJoinUsesCanonicalRoleOverlap();
  testAdjustmentScopePruning();
  testAdjustmentPruningPreservesCrossTreeQueries();
  testSingleDoCausalConnectedScopePruning();
  testSingleDoPrunesOnlyNoEffectForkScope();
  testMultiVariableAdjustmentIsRetainedFromAlternativeSeparator();
  testAllSingletonAdjustmentAlternativesAreRetained();
  testRequiredMultiVariableAdjustmentIsRetained();
  testRedundantConditionIsCanonicalized();
  testDynamicMultiAdjustmentUsesSavedAttachments();
  testOversizedAtomicFactorIsPreserved();
  testQueryAwareParentTreeCoversChainClosures();
  testCostAwareParentTreeMergeAcceptsOnlyUsefulUnion();
  return 0;
}
