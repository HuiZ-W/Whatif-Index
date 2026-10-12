#include "causal_index_v4/FullLatticeIndex.h"

#include <cassert>
#include <iostream>
#include <random>

using namespace causal_index_v4;
using Cubes = std::map<Mask, std::map<std::vector<std::string>, Count>>;

void check(const std::vector<VariableInput>& data, bool require_rollup = false) {
  LatticeBuildConfig cfg;
  cfg.enable_derive = false;
  cfg.enable_value_partition_join = false;
  cfg.execution_schedule = ExecutionSchedule::IndependentSources;
  cfg.variable_order_policy = VariableOrderPolicy::Alphabetical;
  cfg.independent_leaf_pushdown = true;
  Cubes reference;
  FullLatticeIndex full(data, cfg);
  full.streamIndependentCubes([&](Mask m, auto&& counts) { reference.emplace(m, std::move(counts)); });
  std::size_t direct_pairs = 0;
  for (std::size_t budget : {std::size_t{0}, std::size_t{512 * 1024 * 1024}, std::size_t{2048}}) {
    cfg.rollup_cache_bytes = budget;
    RollupStats stats;
    cfg.rollup_callback = [&](const auto& value) { stats = value; };
    std::size_t pairs = 0;
    cfg.join_work_callback = [&](auto, auto work) { pairs += work; };
    FullLatticeIndex rollup(data, cfg);
    Cubes actual;
    rollup.streamRollupCubes([&](Mask m, auto&& counts) { actual.emplace(m, std::move(counts)); });
    assert(actual == reference);
    assert(stats.peak_cache_bytes <= budget);
    assert(rollup.result().exact_counts.empty());

    if (budget == 0) {
      assert(stats.used == 0);
      direct_pairs = pairs;
    } else if (require_rollup && budget > 2048) {
      assert(stats.used > 0 && stats.projected_rows > 0 && stats.residual_rows > 0);
      assert(pairs < direct_pairs);
    }
  }
}

int main() {
  // AXZ supplies only key k0; k1..k9 must still be joined. Multiple A
  // witnesses must not multiply XZ, and equal-valued X instances stay distinct.
  std::vector<VariableInput> data{{"X", {"K"}, {}}, {"A", {"K", "D"}, {}}, {"Z", {"K"}, {}}};
  for (std::size_t i = 0; i < 10; ++i) {
    data[0].instances.push_back({i+1, "all", {{"K", "k"+std::to_string(i)}}, "same"});
    data[2].instances.push_back({i+1, "all", {{"K", "k"+std::to_string(i)}}, "same"});
  }
  data[0].instances.push_back({11, "all", {{"K", "k0"}}, "same"});
  data[1].instances = {{1,"all",{{"K","k0"},{"D","d0"}},"a"},
                       {2,"all",{{"K","k0"},{"D","d1"}},"a"}};
  check(data, true);
  data[1].instances.clear();
  check(data);
  data[1].pattern_roles = {"D"};
  data[1].instances = {{1,"all",{{"D","d0"}},"a"},
                       {2,"all",{{"D","d1"}},"a"}};
  check(data);  // Empty shared key: certified full coverage, no residual join.
  // Unsafe keys and injectivity conflicts are covered by direct residual joins.
  std::mt19937 rng(20260915);
  for (int trial = 0; trial < 80; ++trial) {
    std::vector<VariableInput> inputs;
    for (int v = 0; v < 4; ++v) {
      VariableInput input{std::string(1, 'A'+v), {"K", std::string(1, 'P'+v)}, {}};
      for (std::size_t i = 0; i < 5; ++i) {
        const auto key = rng()%3;
        const auto private_node = rng()%6;
        const auto node = private_node < 3 ? "k"+std::to_string(private_node) : "p"+std::to_string(private_node);
        if (node == "k"+std::to_string(key)) continue;
        input.instances.push_back({i+1,"all",{{"K","k"+std::to_string(key)},{input.pattern_roles[1],node}},"v"+std::to_string(i%2)});
      }
      inputs.push_back(std::move(input));
    }
    check(inputs);
  }
  std::cout << "FullJoin rollup counts, duplicates, residual work and cache bounds passed\n";
}
