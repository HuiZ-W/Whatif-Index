#include "causal_index_v4/CausalAdjustment.h"
#include "causal_index_v4/FullLatticeIndex.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace causal_index_v4;

namespace {

void testGeneralizedAdjustmentSets() {
  {
    CausalDag chain({"X", "Y", "Z"}, {{"X", "Y"}, {"Y", "Z"}});
    const auto result = chain.analyzeAdjustment(chain.maskOf({"X"}), chain.maskOf({"Z"}));
    assert(result.has_proper_causal_path);
    assert(result.adjustment_exists);
    assert(result.adjustment == 0);
  }
  {
    CausalDag fork({"W", "X", "Y", "Z"},
                   {{"Z", "W"}, {"Z", "X"}, {"Z", "Y"}});
    const auto pair = fork.analyzeAdjustment(fork.maskOf({"X"}), fork.maskOf({"Y"}));
    assert(!pair.has_proper_causal_path);
    assert(pair.observationally_connected);
    assert(pair.adjustment_exists);
    assert(pair.adjustment == fork.maskOf({"Z"}));

    const auto multi = fork.analyzeAdjustment(fork.maskOf({"X", "Y"}),
                                              fork.maskOf({"W"}));
    assert(!multi.has_proper_causal_path);
    assert(multi.adjustment_exists);
    assert(multi.adjustment == fork.maskOf({"Z"}));
  }
  {
    CausalDag confounded({"X", "Y", "Z"},
                         {{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}});
    const auto result = confounded.analyzeAdjustment(confounded.maskOf({"X"}),
                                                     confounded.maskOf({"Y"}));
    assert(result.has_proper_causal_path);
    assert(result.adjustment_exists);
    assert(result.adjustment == confounded.maskOf({"Z"}));
  }
  {
    CausalDag collider({"M", "X", "Z"}, {{"X", "M"}, {"Z", "M"}});
    const auto result = collider.analyzeAdjustment(collider.maskOf({"X"}),
                                                  collider.maskOf({"Z"}));
    assert(!result.has_proper_causal_path);
    assert(!result.observationally_connected);
    assert(result.adjustment_exists);
    assert(result.adjustment == 0);
  }
  {
    CausalDag confounded({"X", "Y", "Z"},
                         {{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}});
    const auto unavailable = confounded.analyzeAdjustmentWithin(
        confounded.maskOf({"X"}), confounded.maskOf({"Y"}),
        confounded.maskOf({"X", "Y"}));
    assert(!unavailable.adjustment_exists);
    const auto available = confounded.analyzeAdjustmentWithin(
        confounded.maskOf({"X"}), confounded.maskOf({"Y"}),
        confounded.maskOf({"X", "Y", "Z"}));
    assert(available.adjustment_exists);
    assert(available.adjustment == confounded.maskOf({"Z"}));
  }

}

VariableInput makeVariable(const std::string& id,
                           std::vector<RoleId> roles,
                           std::vector<MatchInstance> instances) {
  return {id, std::move(roles), std::move(instances)};
}

std::vector<VariableInput> makeTriangleWorkload() {
  return {
      makeVariable("X", {"A", "C"}, {
          {1, "b1", {{"A", "a1"}, {"C", "c1"}}},
          {2, "b1", {{"A", "a2"}, {"C", "c2"}}},
      }),
      makeVariable("Y", {"A", "B"}, {
          {11, "b1", {{"A", "a1"}, {"B", "b1"}}},
          {12, "b1", {{"A", "a1"}, {"B", "b2"}}},
          {13, "b1", {{"A", "a2"}, {"B", "b2"}}},
      }),
      makeVariable("Z", {"B", "C"}, {
          {21, "b1", {{"B", "b1"}, {"C", "c1"}}},
          {22, "b1", {{"B", "b2"}, {"C", "c1"}}},
          {23, "b1", {{"B", "b2"}, {"C", "c2"}}},
      }),
  };
}

void testPartialWorkloadOrdersPreferSharedCore() {
  const std::vector<VariableInput> variables{
      makeVariable("A", {"Root#0"}, {{1, "b", {{"Root#0", "r1"}}}}),
      makeVariable("B", {"Root#0"}, {{2, "b", {{"Root#0", "r1"}}}}),
      makeVariable("X", {"Root#0"}, {{3, "b", {{"Root#0", "r1"}}}}),
      makeVariable("Y", {"Root#0"}, {{4, "b", {{"Root#0", "r1"}}}}),
  };
  auto run = [&](VariableOrderPolicy policy) {
    LatticeBuildConfig config;
    config.enable_derive = false;
    config.variable_order_policy = policy;
    config.source_output_scopes = {
        {{"X", "Y", "A"}, {{"X", "Y", "A"}}},
        {{"X", "Y", "B"}, {{"X", "Y", "B"}}},
    };
    FullLatticeIndex index(variables, config);
    index.build();
    assert(index.plan().task_order.size() == 4);
    const std::set<VariableId> first_two{
        index.plan().variable_order[0], index.plan().variable_order[1]};
    assert(first_two == std::set<VariableId>({"X", "Y"}));
    assert(index.exactCount(index.maskOf({"X", "Y", "A"})) == 1);
    assert(index.exactCount(index.maskOf({"X", "Y", "B"})) == 1);
  };
  run(VariableOrderPolicy::PartialWorkloadDp);
  run(VariableOrderPolicy::PartialWorkloadGreedy);
}

void testPruningKeepsTheUnprunedVariableOrder() {
  const std::vector<VariableInput> variables{
      makeVariable("A", {"Root#0"}, {{1, "b", {{"Root#0", "r1"}}}}),
      makeVariable("B", {"Root#0"}, {{2, "b", {{"Root#0", "r1"}}}}),
      makeVariable("C", {"Root#0"}, {{3, "b", {{"Root#0", "r1"}}}}),
      makeVariable("D", {"Root#0"}, {{4, "b", {{"Root#0", "r1"}}}}),
  };

  auto make_plan = [&](VariableOrderPolicy policy, bool pruned) {
    LatticeBuildConfig config;
    config.enable_derive = false;
    config.variable_order_policy = policy;
    config.variable_order_seed = 73;
    config.plan_order_from_unpruned_lattice = true;
    if (pruned) {
      config.source_output_scopes = {
          {{"A", "B", "C"}, {{"A", "B", "C"}}},
          {{"A", "B", "D"}, {{"A", "B", "D"}}},
      };
    }
    FullLatticeIndex index(variables, config);
    index.planOnly();
    return index.plan();
  };

  for (const auto policy : {VariableOrderPolicy::PartialWorkloadGreedy,
                            VariableOrderPolicy::Random}) {
    const auto full = make_plan(policy, false);
    const auto pruned = make_plan(policy, true);
    assert(full.variable_order == pruned.variable_order);
    assert(pruned.task_order.size() < full.task_order.size());

    auto cursor = full.task_order.begin();
    for (const auto task : pruned.task_order) {
      cursor = std::find(cursor, full.task_order.end(), task);
      assert(cursor != full.task_order.end());
      ++cursor;
    }
  }
}

void testFullLatticeCountsAndPlan() {
  LatticeBuildConfig config;
  config.cache_budget_tuples = 1;
  config.enable_derive = true;
  config.store_all_supports = true;
  config.store_participation = true;
  config.store_derive_results = true;
  FullLatticeIndex index(makeTriangleWorkload(), config);
  index.build();

  assert(index.variableCount() == 3);
  assert(index.plan().task_order.size() == 7);
  assert(index.plan().variable_order.size() == 3);
  assert(std::set<VariableId>(index.plan().variable_order.begin(), index.plan().variable_order.end()) ==
         std::set<VariableId>({"X", "Y", "Z"}));
  assert(!index.plan().materialization.empty());
  assert(index.plan().estimated_cost > 0.0);

  const Mask x = index.maskOf({"X"});
  const Mask y = index.maskOf({"Y"});
  const Mask z = index.maskOf({"Z"});
  const Mask xy = index.maskOf({"X", "Y"});
  const Mask xz = index.maskOf({"X", "Z"});
  const Mask yz = index.maskOf({"Y", "Z"});
  const Mask xyz = index.maskOf({"X", "Y", "Z"});

  assert(index.exactCount(x) == 2);
  assert(index.exactCount(y) == 3);
  assert(index.exactCount(z) == 3);
  assert(std::abs(index.estimatedCount(x) - 2.0) < 1e-9);
  assert(std::abs(index.estimatedCount(y) - 3.0) < 1e-9);
  assert(std::abs(index.estimatedCount(z) - 3.0) < 1e-9);
  assert(index.exactCount(xy) == 3);
  assert(index.exactCount(xz) == 3);
  assert(index.exactCount(yz) == 5);
  assert(index.exactCount(xyz) == 3);

  assert(index.result().emitted_rows == 22);
  assert(index.result().cache_stats.trie_nodes_visited > 0);
  assert(index.result().cache_stats.evictions > 0);
  assert(index.result().cache_stats.recomputations > 0);
  assert(!index.result().participation.empty());
  bool saw_nontrivial_bitmap = false;
  for (const auto& bitmap : index.result().participation) {
    if (!bitmap.has_partner.empty()) {
      saw_nontrivial_bitmap = true;
    }
  }
  assert(saw_nontrivial_bitmap);

  const auto derived = index.deriveFromSuperset(xz, 1 /* Y */);
  assert(derived.target == xz);
  assert(derived.superset == xyz);
  assert(derived.rows.size() == index.exactCount(xz));

  bool result_has_same_derive = false;
  for (const auto& item : index.result().derive_results) {
    if (item.target == xz && item.donor_variable == 1) {
      result_has_same_derive = item.rows.size() == index.exactCount(xz);
    }
  }
  assert(result_has_same_derive);
}

void testInjectivityPrunesConflictingInstances() {
  std::vector<VariableInput> variables{
      makeVariable("A", {"P", "Q"}, {
          {1, "b", {{"P", "p1"}, {"Q", "q1"}}},
      }),
      makeVariable("B", {"Q", "R"}, {
          {2, "b", {{"Q", "q1"}, {"R", "p1"}}}, // conflicts with A.P
          {3, "b", {{"Q", "q1"}, {"R", "r1"}}},
      }),
  };
  LatticeBuildConfig config;
  config.enable_anchor_partition = false;
  config.collect_partition_traces = true;
  FullLatticeIndex index(std::move(variables), config);
  index.build();
  const auto ab = index.maskOf({"A", "B"});
  assert(index.exactCount(ab) == 1);
  assert(index.result().supports[ab].empty());
  assert(index.result().cache_stats.trie_subtrees_skipped > 0);
  assert(!index.result().partition_traces.empty());
  const auto& trace = index.result().partition_traces.back();
  assert(trace.mode == "exact-trie");
  assert(trace.parent_partition_buckets == 1);
  assert(trace.candidate_partition_buckets == 1);
  assert(trace.bucket_count == 1);
  assert(trace.parent_bucket_sum == 1.0L);
  assert(trace.candidate_bucket_sum == 2.0L);
  assert(trace.partition_pairs == 2.0L);
  assert(trace.compatible_pairs == 1);
}

void testOnlineDeriveCanReplaceExtend() {
  std::vector<MatchInstance> x_instances;
  std::vector<MatchInstance> z_instances;
  for (int i = 0; i < 10; ++i) {
    x_instances.push_back({static_cast<InstanceId>(100 + i), "b", {{"A", "a" + std::to_string(i)}}});
    z_instances.push_back({static_cast<InstanceId>(200 + i), "b", {{"A", "a" + std::to_string(i)}}});
  }
  std::vector<VariableInput> variables{
      makeVariable("X", {"A"}, x_instances),
      makeVariable("Y", {"A", "B"}, {
          {1, "b", {{"A", "a0"}, {"B", "b0"}}},
      }),
      makeVariable("Z", {"A"}, z_instances),
  };
  LatticeBuildConfig config;
  config.enable_derive = true;
  config.store_all_supports = true;
  config.store_derive_results = true;
  FullLatticeIndex index(std::move(variables), config);
  index.build();

  const Mask xz = index.maskOf({"X", "Z"});
  assert(index.exactCount(xz) == 10);
  assert(index.result().cache_stats.derive_candidates_considered > 0);
  assert(index.result().cache_stats.derive_used > 0);
  bool saw_used_xz = false;
  for (const auto& derive : index.result().derive_results) {
    if (derive.target == xz && derive.used) {
      saw_used_xz = true;
      assert(derive.rows.size() == 10);
      assert(derive.projected_rows == 1);
      assert(derive.complement_rows == 9);
      assert(derive.derive_cost <= derive.extend_cost);
    }
  }
  assert(saw_used_xz);
}


void testStreamingSelectiveDeriveMaterialization() {
  std::vector<MatchInstance> x_instances;
  std::vector<MatchInstance> z_instances;
  for (int i = 0; i < 10; ++i) {
    x_instances.push_back({static_cast<InstanceId>(100 + i), "b", {{"A", "a" + std::to_string(i)}}});
    z_instances.push_back({static_cast<InstanceId>(200 + i), "b", {{"A", "a" + std::to_string(i)}}});
  }
  std::vector<VariableInput> variables{
      makeVariable("X", {"A"}, x_instances),
      makeVariable("Y", {"A", "B"}, {
          {1, "b", {{"A", "a0"}, {"B", "b0"}}},
      }),
      makeVariable("Z", {"A"}, z_instances),
  };
  LatticeBuildConfig config;
  config.enable_derive = true;
  config.store_all_supports = false;
  config.store_derive_results = true;
  config.derive_materialization_budget_tuples = 1000;
  FullLatticeIndex index(std::move(variables), config);
  index.build();

  const Mask xz = index.maskOf({"X", "Z"});
  assert(index.exactCount(xz) == 10);
  assert(!index.plan().derive_pinned_masks.empty());
  assert(index.result().cache_stats.derive_materialized_masks > 0);
  assert(index.result().cache_stats.derive_materialized_masks < index.result().supports.size() - 1);
  assert(index.result().cache_stats.derive_used > 0);
  std::size_t retained_supports = 0;
  for (const auto& rows : index.result().supports) {
    if (!rows.empty()) {
      retained_supports += 1;
    }
  }
  assert(retained_supports == index.result().cache_stats.derive_materialized_masks);
  bool saw_used_xz = false;
  for (const auto& derive : index.result().derive_results) {
    if (derive.target == xz && derive.used) {
      saw_used_xz = true;
      assert(derive.rows.size() == 10);
      assert(derive.projected_rows == 1);
      assert(derive.complement_rows == 9);
    }
  }
  assert(saw_used_xz);
}

void testValuePartitionStillUsesExactNodeEquality() {
  std::vector<VariableInput> variables{
      makeVariable("X", {"Person#0"}, {
          {1, "x1", {{"Person#0", "p1"}}, "x1"},
          {2, "x2", {{"Person#0", "p2"}}, "x2"},
      }),
      makeVariable("Y", {"Person#0"}, {
          {11, "y1", {{"Person#0", "p1"}}, "y1"},
          {12, "y2", {{"Person#0", "p2"}}, "y2"},
          {13, "y3", {{"Person#0", "p3"}}, "y3"},
      }),
  };
  auto attributes = std::make_shared<NodeAttributeStore>();
  (*attributes)["p1"]["gender"] = "female";
  (*attributes)["p2"]["gender"] = "male";
  // p1 and p3 deliberately collide in the value partition. The direct exact
  // comparison inside that bucket must still reject p1-p3.
  (*attributes)["p3"]["gender"] = "female";

  LatticeBuildConfig config;
  config.enable_derive = false;
  config.enable_anchor_partition = false;
  config.store_all_supports = true;
  config.node_attributes = attributes;
  config.value_partition_attributes_by_label["Person"] = {"gender"};
  config.value_partition_min_parent_rows = 1;
  config.value_partition_min_pair_reduction = 0.0;
  config.value_partition_bucket_overhead = 0.0;
  config.collect_partition_traces = true;
  config.worker_threads = 1;
  config.target_scopes = {{"X", "Y"}};
  FullLatticeIndex index(std::move(variables), config);
  index.build();

  const auto xy = index.maskOf({"X", "Y"});
  assert(index.exactCount(xy) == 2);
  assert(index.result().cache_stats.value_partition_joins == 1);
  assert(index.result().cache_stats.value_partition_features_selected == 1);
  assert(index.result().cache_stats.value_partition_buckets_built == 2);
  assert(index.result().cache_stats.value_partition_pair_checks == 3);
  assert(index.result().cache_stats.value_partition_compatible_pairs == 2);
  assert(index.result().cache_stats.value_partition_feature_uses.at("Person#0.gender") == 1);
  assert(index.result().partition_traces.size() == 1);
  const auto& trace = index.result().partition_traces.front();
  assert(trace.mode == "value");
  assert(trace.parent_partition_buckets == 2);
  assert(trace.candidate_partition_buckets == 2);
  assert(trace.bucket_count == 2);
  assert(trace.cartesian_pairs == 6.0L);
  assert(trace.partition_pairs == 3.0L);
  assert(trace.compatible_pairs == 2);
}

void testExecutionSchedulesProduceIdenticalCubes() {
  const std::vector<SourceOutputScope> workload{
      {{"X", "Y"}, {{"X", "Y"}}},
      {{"X", "Y", "Z"}, {{"X", "Z"}, {"X", "Y", "Z"}}},
  };
  std::vector<Count> expected_counts;
  std::vector<std::map<std::vector<std::string>, Count>> expected_cubes;

  for (const auto schedule : {
           ExecutionSchedule::DfsPrefixTree,
           ExecutionSchedule::ScopeOrderedWorkingPath,
           ExecutionSchedule::BoundedScopeCache,
           ExecutionSchedule::JoinSizeGreedyWorkingPath,
           ExecutionSchedule::CacheAwareGreedy,
           ExecutionSchedule::IndependentSources}) {
    LatticeBuildConfig config;
    config.enable_derive = false;
    config.enable_value_partition_join = false;
    config.variable_order_policy = VariableOrderPolicy::InstanceCountAscending;
    config.execution_schedule = schedule;
    if (schedule == ExecutionSchedule::BoundedScopeCache) {
      config.result_cache_capacity = 2;
    }
    config.source_output_scopes = workload;
    FullLatticeIndex index(makeTriangleWorkload(), config);
    index.build();

    if (expected_counts.empty()) {
      expected_counts = index.result().exact_counts;
      expected_cubes = index.result().value_counts;
    } else {
      assert(index.result().exact_counts == expected_counts);
      assert(index.result().value_counts == expected_cubes);
    }
    if (schedule == ExecutionSchedule::JoinSizeGreedyWorkingPath) {
      assert(index.result().cache_stats.evictions == 0);
      assert(index.result().cache_stats.recomputations == 0);
      assert(index.result().cache_stats.resident_tuples == 1);
    }
    if (schedule == ExecutionSchedule::CacheAwareGreedy) {
      assert(index.result().cache_stats.non_root_hits > 0);
    }
  }
}

void testScopeOrderPoliciesOnlyReorderCompleteScopes() {
  const std::vector<SourceOutputScope> workload{
      {{"Y", "Z"}, {{"Y", "Z"}}},
      {{"X", "Z"}, {{"X", "Z"}}},
      {{"X", "Y"}, {{"X", "Y"}}},
      {{"X", "Y", "Z"}, {{"X", "Y", "Z"}}},
  };

  auto build = [&](ScopeOrderPolicy scope_policy, std::uint64_t seed) {
    LatticeBuildConfig config;
    config.enable_derive = false;
    config.enable_value_partition_join = false;
    config.variable_order_policy = VariableOrderPolicy::PartialWorkloadGreedy;
    config.execution_schedule = ExecutionSchedule::ScopeOrderedWorkingPath;
    config.scope_order_policy = scope_policy;
    config.scope_order_seed = seed;
    config.source_output_scopes = workload;
    FullLatticeIndex index(makeTriangleWorkload(), config);
    index.build();
    return index;
  };

  auto alphabetical = build(ScopeOrderPolicy::Alphabetical, 73);
  const std::vector<Mask> expected_alphabetical{
      alphabetical.maskOf({"X", "Y"}),
      alphabetical.maskOf({"X", "Y", "Z"}),
      alphabetical.maskOf({"X", "Z"}),
      alphabetical.maskOf({"Y", "Z"}),
  };
  assert(alphabetical.plan().scope_order == expected_alphabetical);
  assert(alphabetical.result().cache_stats.working_path_tasks ==
         expected_alphabetical.size());
  assert(alphabetical.result().cache_stats.resident_tuples == 1);

  auto random_left = build(ScopeOrderPolicy::Random, 73);
  auto random_right = build(ScopeOrderPolicy::Random, 73);
  assert(random_left.plan().scope_order == random_right.plan().scope_order);
  assert(random_left.plan().scope_order != alphabetical.plan().scope_order);

  // Scope scheduling must not alter rho or any output semantics.
  assert(random_left.plan().variable_order ==
         alphabetical.plan().variable_order);
  assert(random_left.result().exact_counts ==
         alphabetical.result().exact_counts);
  assert(random_left.result().value_counts ==
         alphabetical.result().value_counts);

  auto expected_join_steps = [&](const FullLatticeIndex& index) {
    std::vector<std::vector<Mask>> paths;
    for (const Mask source : index.plan().scope_order) {
      std::vector<Mask> path;
      Mask prefix = 0;
      for (const auto variable : index.plan().variable_order_indices) {
        const Mask bit = Mask{1} << variable;
        if ((source & bit) != 0) {
          prefix |= bit;
          path.push_back(prefix);
        }
      }
      paths.push_back(std::move(path));
    }
    std::size_t joins = 0;
    for (std::size_t i = 0; i < paths.size(); ++i) {
      std::size_t reused = 0;
      if (i != 0) {
        while (reused < paths[i - 1].size() &&
               reused < paths[i].size() &&
               paths[i - 1][reused] == paths[i][reused]) {
          ++reused;
        }
      }
      joins += paths[i].size() - reused;
    }
    return joins;
  };
  assert(random_left.result().cache_stats.working_path_join_steps ==
         expected_join_steps(random_left));
  assert(random_left.result().task_order.size() ==
         expected_join_steps(random_left));
}

void testPrunedRandomScopeOrderFiltersUnprunedSchedule() {
  auto build = [](const std::vector<SourceOutputScope>& workload) {
    LatticeBuildConfig config;
    config.enable_derive = false;
    config.enable_value_partition_join = false;
    config.variable_order_policy = VariableOrderPolicy::PartialWorkloadGreedy;
    config.plan_order_from_unpruned_lattice = true;
    config.execution_schedule = ExecutionSchedule::ScopeOrderedWorkingPath;
    config.scope_order_policy = ScopeOrderPolicy::Random;
    config.scope_order_seed = 91;
    config.scope_order_from_unpruned_lattice = true;
    config.source_output_scopes = workload;
    FullLatticeIndex index(makeTriangleWorkload(), config);
    index.build();
    return index;
  };

  std::vector<SourceOutputScope> full_workload;
  const std::vector<std::vector<VariableId>> all_scopes{
      {"X"}, {"Y"}, {"Z"}, {"X", "Y"}, {"X", "Z"},
      {"Y", "Z"}, {"X", "Y", "Z"}};
  for (const auto& scope : all_scopes) {
    full_workload.push_back({scope, {scope}});
  }
  const std::vector<SourceOutputScope> pruned_workload{
      {{"X"}, {{"X"}}},
      {{"X", "Y"}, {{"X", "Y"}}},
      {{"Y", "Z"}, {{"Y", "Z"}}},
      {{"X", "Y", "Z"}, {{"X", "Y", "Z"}}},
  };

  const auto full = build(full_workload);
  const auto pruned = build(pruned_workload);
  const std::set<Mask> retained{
      pruned.maskOf({"X"}), pruned.maskOf({"X", "Y"}),
      pruned.maskOf({"Y", "Z"}), pruned.maskOf({"X", "Y", "Z"})};
  std::vector<Mask> filtered;
  for (const Mask scope : full.plan().scope_order) {
    if (retained.count(scope) != 0) {
      filtered.push_back(scope);
    }
  }
  assert(pruned.plan().variable_order == full.plan().variable_order);
  assert(pruned.plan().scope_order == filtered);
}

void testBoundedResultCacheOrdersAndEviction() {
  LatticeBuildConfig reference_config;
  reference_config.enable_derive = false;
  reference_config.enable_value_partition_join = false;
  reference_config.variable_order_policy =
      VariableOrderPolicy::PartialWorkloadGreedy;
  reference_config.execution_schedule =
      ExecutionSchedule::IndependentSources;
  FullLatticeIndex reference(makeTriangleWorkload(), reference_config);
  reference.build();

  std::vector<VariableId> expected_rho;
  for (const auto scope_policy : {
           ScopeOrderPolicy::DfsLexicographic,
           ScopeOrderPolicy::Alphabetical,
           ScopeOrderPolicy::Random}) {
    LatticeBuildConfig config;
    config.enable_derive = false;
    config.enable_value_partition_join = false;
    config.variable_order_policy =
        VariableOrderPolicy::PartialWorkloadGreedy;
    config.scope_order_policy = scope_policy;
    config.scope_order_seed = 73;
    config.execution_schedule = ExecutionSchedule::BoundedScopeCache;
    config.result_cache_capacity = 4;
    config.result_cache_eviction_policy =
        ResultCacheEvictionPolicy::Belady;
    FullLatticeIndex index(makeTriangleWorkload(), config);
    index.build();

    if (expected_rho.empty()) {
      expected_rho = index.plan().variable_order;
    } else {
      assert(index.plan().variable_order == expected_rho);
    }
    assert(index.result().exact_counts == reference.result().exact_counts);
    assert(index.result().value_counts == reference.result().value_counts);
    assert(index.result().cache_stats.peak_resident_results <= 4);
    assert(index.result().cache_stats.resident_results == 0);
    assert(index.result().cache_stats.result_capacity_overflows == 0);
  }

  for (const auto policy : {
           ResultCacheEvictionPolicy::Belady,
           ResultCacheEvictionPolicy::Lru}) {
    LatticeBuildConfig config;
    config.enable_derive = false;
    config.enable_value_partition_join = false;
    config.variable_order_policy =
        VariableOrderPolicy::PartialWorkloadGreedy;
    config.scope_order_policy = ScopeOrderPolicy::DfsLexicographic;
    config.execution_schedule = ExecutionSchedule::BoundedScopeCache;
    config.result_cache_capacity = 1;
    config.result_cache_eviction_policy = policy;
    FullLatticeIndex index(makeTriangleWorkload(), config);
    index.build();

    assert(index.result().exact_counts == reference.result().exact_counts);
    assert(index.result().value_counts == reference.result().value_counts);
    assert(index.result().cache_stats.peak_resident_results <= 1);
    assert(index.result().cache_stats.result_capacity_evictions > 0);
    assert(index.result().cache_stats.recomputations > 0);
    assert(index.result().cache_stats.result_capacity_overflows == 0);
  }
}

}  // namespace

int main() {
  testGeneralizedAdjustmentSets();
  testPartialWorkloadOrdersPreferSharedCore();
  testPruningKeepsTheUnprunedVariableOrder();
  testFullLatticeCountsAndPlan();
  testInjectivityPrunesConflictingInstances();
  testOnlineDeriveCanReplaceExtend();
  testStreamingSelectiveDeriveMaterialization();
  testValuePartitionStillUsesExactNodeEquality();
  testExecutionSchedulesProduceIdenticalCubes();
  testScopeOrderPoliciesOnlyReorderCompleteScopes();
  testPrunedRandomScopeOrderFiltersUnprunedSchedule();
  testBoundedResultCacheOrdersAndEviction();
  return 0;
}
