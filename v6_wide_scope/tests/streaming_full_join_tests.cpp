#include "causal_index_v5/BoundaryJoinIndex.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace causal_index_v4;

int main() {
  LatticeBuildConfig cfg;
  cfg.enable_derive = false;
  cfg.enable_value_partition_join = false;
  cfg.variable_order_policy = VariableOrderPolicy::Alphabetical;
  cfg.execution_schedule = ExecutionSchedule::IndependentSources;
  for (bool empty : {false, true}) {
    std::vector<VariableInput> data;
    for (const auto& id : {"C", "A", "B"}) {
      VariableInput input{id, {"P"}, {}};
      for (std::size_t k = 0; k < 3; ++k)
        input.instances.push_back({k + 1, "all", {{"P", "p"}}, k == 2 ? "v2" : "v1"});
      data.push_back(std::move(input));
    }
    if (empty) data[1].instances.clear();
    for (bool fused : {false, true}) {
      cfg.independent_leaf_pushdown = fused;
      std::size_t calls = 0, pairs = 0, probes = 0;
      cfg.binary_join_callback = [&] { ++calls; };
      cfg.join_work_callback = [&](auto rows, auto matches) { probes += rows; pairs += matches; };
      FullLatticeIndex dense(data, cfg);
      dense.build();
      const auto old_calls = calls, old_pairs = pairs, old_probes = probes;
      calls = pairs = probes = 0;
      FullLatticeIndex streamed(data, cfg);
      std::size_t cubes = 0;
      streamed.streamIndependentCubes([&](Mask mask, auto&& counts) {
        ++cubes;
        assert(counts == dense.result().value_counts[mask]);
      });
      assert(cubes == 7);
      assert(calls == old_calls && pairs == old_pairs && probes == old_probes);
      assert(streamed.result().emitted_rows == dense.result().emitted_rows);
      assert(streamed.result().value_counts.empty());
      assert(streamed.plan().estimates.totals.empty());

      causal_index_v5::BoundaryBuildConfig boundary;
      boundary.include_all_reachable_pairs = false;
      boundary.tree_executor = cfg;
      const std::vector<causal_index_v5::ParentTreeSpec> trees{{"global", {"C", "A", "B"}}};
      auto old_index = causal_index_v5::MaterializedBoundaryIndex::build(data, {}, trees, boundary);
      boundary.stream_all_subset_cubes = true;
      auto new_index = causal_index_v5::MaterializedBoundaryIndex::build(data, {}, trees, boundary);
      assert(old_index.cubes().size() == new_index.cubes().size());
      for (std::size_t i = 0; i < old_index.cubes().size(); ++i) {
        const auto& a = old_index.cubes()[i];
        const auto& b = new_index.cubes()[i];
        assert(a.tree_id == b.tree_id && a.scope == b.scope && a.counts == b.counts);
      }
      const auto suffix = std::to_string(empty) + "_" + std::to_string(fused);
      const auto base = std::filesystem::temp_directory_path() /
          ("streaming_full_join_" + suffix);
      const auto spool_path = base.string() + ".spool";
      const auto memory_path = base.string() + ".memory.bin";
      const auto external_path = base.string() + ".external.bin";
      causal_index_v5::ExternalCubeStorage storage;
      storage.path = spool_path;
      std::ofstream spool(spool_path, std::ios::binary | std::ios::trunc);
      assert(spool);
      boundary.streamed_cube_sink = [&](causal_index_v5::CubeTable&& cube) {
        storage.cell_count += cube.counts.size();
        causal_index_v5::MaterializedBoundaryIndex::writeCubeRecord(spool, cube);
        ++storage.table_count;
      };
      auto spooled_index =
          causal_index_v5::MaterializedBoundaryIndex::build(data, {}, trees, boundary);
      assert(spooled_index.cubes().empty());
      spool.close();
      storage.payload_bytes = std::filesystem::file_size(spool_path);
      const auto memory_stats = new_index.saveWithStats(memory_path);
      const auto external_stats =
          spooled_index.saveWithExternalCubes(external_path, storage);
      assert(memory_stats.total_bytes == external_stats.total_bytes);
      assert(memory_stats.cube_bytes == external_stats.cube_bytes);
      auto loaded =
          causal_index_v5::MaterializedBoundaryIndex::load(external_path);
      assert(loaded.cubes().size() == old_index.cubes().size());
      for (const auto& expected : old_index.cubes()) {
        const auto& actual = loaded.cube(expected.tree_id, expected.scope);
        assert(expected.counts == actual.counts);
      }
      std::filesystem::remove(spool_path);
      std::filesystem::remove(memory_path);
      std::filesystem::remove(external_path);
    }
  }

  {
    std::vector<VariableInput> data;
    for (const auto& id : {"A", "B", "C"}) {
      VariableInput input{id, {"P"}, {}};
      input.instances.push_back({1, "all", {{"P", "p"}}, std::string(id) + "1"});
      input.instances.push_back({2, "all", {{"P", "p"}}, std::string(id) + "2"});
      data.push_back(std::move(input));
    }
    const std::vector<causal_index_v5::CausalEdge> edges{{"A", "B"}, {"B", "C"}};
    const std::vector<causal_index_v5::ParentTreeSpec> trees{
        {"left", {"A", "B"}}, {"right", {"B", "C"}}};
    causal_index_v5::BoundaryBuildConfig boundary;
    boundary.cube_policy = causal_index_v5::LocalCubePolicy::ExplicitScopes;
    boundary.explicit_cube_scopes = {
        {"left", {{"A", "B"}}}, {"right", {{"B", "C"}}}};
    boundary.tree_executor.enable_derive = false;
    boundary.tree_executor.enable_value_partition_join = false;
    boundary.tree_executor.variable_order_policy = VariableOrderPolicy::Alphabetical;

    const auto memory_index =
        causal_index_v5::MaterializedBoundaryIndex::build(data, edges, trees, boundary);
    assert(!memory_index.boundaryViews().empty());
    const auto base = std::filesystem::temp_directory_path() /
        "streaming_parent_only";
    const auto spool_path = base.string() + ".spool";
    const auto memory_path = base.string() + ".memory.bin";
    const auto external_path = base.string() + ".external.bin";
    causal_index_v5::ExternalCubeStorage storage;
    storage.path = spool_path;
    std::ofstream spool(spool_path, std::ios::binary | std::ios::trunc);
    assert(spool);
    boundary.streamed_cube_sink = [&](causal_index_v5::CubeTable&& cube) {
      storage.cell_count += cube.counts.size();
      causal_index_v5::MaterializedBoundaryIndex::writeCubeRecord(spool, cube);
      ++storage.table_count;
    };
    auto streamed_index =
        causal_index_v5::MaterializedBoundaryIndex::build(data, edges, trees, boundary);
    assert(streamed_index.cubes().empty());
    assert(streamed_index.boundaryViews().size() ==
           memory_index.boundaryViews().size());
    spool.close();
    storage.payload_bytes = std::filesystem::file_size(spool_path);
    const auto memory_stats = memory_index.saveWithStats(memory_path);
    const auto external_stats =
        streamed_index.saveWithExternalCubes(external_path, storage);
    assert(memory_stats.total_bytes == external_stats.total_bytes);
    assert(memory_stats.cube_bytes == external_stats.cube_bytes);
    const auto loaded =
        causal_index_v5::MaterializedBoundaryIndex::load(external_path);
    for (const auto& expected : memory_index.cubes()) {
      assert(loaded.cube(expected.tree_id, expected.scope).counts ==
             expected.counts);
    }
    const auto query_id =
        causal_index_v5::BoundaryRequirementPlanner::reachablePairQueryId("A", "C");
    const auto expected_result = memory_index.execute(query_id);
    const auto actual_result = loaded.execute(query_id);
    assert(actual_result.exact_binding_count == expected_result.exact_binding_count);
    assert(actual_result.value_counts == expected_result.value_counts);
    std::filesystem::remove(spool_path);
    std::filesystem::remove(memory_path);
    std::filesystem::remove(external_path);
  }

  cfg.binary_join_callback = {};
  cfg.join_work_callback = {};
  cfg.independent_leaf_pushdown = true;
  std::vector<VariableInput> wide;
  for (int i = 0; i < 50; ++i)
    wide.push_back({"V" + std::to_string(i), {"P"}, {{1, "all", {{"P", "p"}}, "v"}}});
  FullLatticeIndex index(wide, cfg);
  struct Stop {};
  std::size_t completed = 0;
  try {
    index.streamIndependentCubes([&](Mask mask, auto&& counts) {
      assert(mask == ++completed);
      assert(counts.size() == 1 && counts.begin()->second == 1);
      if (completed == 128) throw Stop{};
    });
  } catch (const Stop&) {}
  assert(completed == 128);
  assert(index.plan().estimates.totals.empty());
  assert(index.plan().target_masks.empty());
  assert(index.result().task_order.empty());
  assert(index.result().exact_counts.empty());
  assert(index.result().value_counts.empty());
  std::cout << "FullJoin equivalence and 50-variable incremental startup passed\n";
}
