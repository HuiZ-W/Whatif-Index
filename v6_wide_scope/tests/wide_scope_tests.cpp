#include "causal_index_v5/CausalAdjustment.h"
#include "causal_index_v4/CausalAdjustment.h"
#include "causal_index_v5/AdjustmentScopePlanner.h"
#include "causal_index_v5/ParentTreePlanner.h"
#include <cassert>
#include <cstdio>

using namespace causal_index_v5;

int main() {
  std::vector<std::string> ids;
  for (int i = 0; i < 130; ++i) ids.push_back("V" + std::to_string(i));
  const std::vector<std::string> active{ids[0], ids[63], ids[64], ids[129]};
  const std::vector<std::pair<std::string, std::string>> edges{
      {active[1], active[0]}, {active[1], active[2]},
      {active[0], active[2]}, {active[2], active[3]}};
  CausalDag wide(ids, edges);
  causal_index_v4::CausalDag reference(active, edges);
  auto names = [&](unsigned mask) {
    std::vector<std::string> out;
    for (unsigned i = 0; i < active.size(); ++i)
      if (mask & (1U << i)) out.push_back(active[i]);
    return out;
  };
  // Embed the same DAG across machine-word boundaries and compare all sets.
  for (unsigned x = 1; x < 16; ++x) {
    for (unsigned y = 1; y < 16; ++y) {
      if (x & y) continue;
      auto wx = wide.maskOf(names(x)), wy = wide.maskOf(names(y));
      assert(wide.reaches(wx, wy) == reference.reaches(x, y));
      assert(wide.variablesOf(wide.causalPathClosure(wx, wy)) ==
             reference.variablesOf(reference.causalPathClosure(x, y)));
      for (unsigned a = 0; a < 16; ++a) {
        if (a & (x | y)) continue;
        auto wa = wide.maskOf(names(a));
        assert(wide.isValidAdjustment(wx, wy, wa) ==
               reference.isValidAdjustment(x, y, a));
        auto got = wide.analyzeAdjustmentWithin(wx, wy, wa);
        auto want = reference.analyzeAdjustmentWithin(x, y, a);
        assert(got.adjustment_exists == want.adjustment_exists);
        assert(wide.variablesOf(got.adjustment) == reference.variablesOf(want.adjustment));
        assert(wide.variablesOf(wide.reduceConditioningUnderIntervention(wx, wy, wa)) ==
               reference.variablesOf(reference.reduceConditioningUnderIntervention(x, y, a)));
      }
    }
  }
  assert(causalMaskCount(wide.maskOf(active)) == 4);
  assert((wide.maskOf({ids[129]}) & wide.maskOf({ids[1]})) == 0);

  std::vector<VariableInput> variables;
  std::vector<std::vector<VariableId>> atomic;
  std::vector<ParentTreeSpec> trees;
  for (int i = 0; i < 100; ++i) {
    variables.push_back({ids[i], {"P", "R" + ids[i]},
        {{static_cast<InstanceId>(i + 1), "all",
          {{"P", "p1"}, {"R" + ids[i], "n" + std::to_string(i)}}, "v"}}});
    atomic.push_back(i == 99 ? std::vector<VariableId>{ids[64], ids[99]}
                            : std::vector<VariableId>{ids[i]});
    trees.push_back({"T" + std::to_string(i), {ids[i]}});
  }
  const std::vector<CausalEdge> causal{{ids[64], ids[99]}};
  const auto planned = ParentTreePlanner::plan(variables, causal, atomic, 3);
  assert(planned.stats.reachable_causal_pairs == 1);
  const auto scopes = AdjustmentScopePlanner::compileSingleDo(
      variables, causal, planned.trees, atomic);
  assert(scopes.retained_queries == 1);
  assert(AdjustmentScopePlanner::compilePairwise(
      variables, causal, planned.trees, atomic).retained_queries == 1);

  BoundaryBuildConfig config;
  config.include_all_reachable_pairs = false;
  config.materialize_variable_attachments = true;
  config.max_all_subset_width = 1;
  config.tree_executor.enable_anchor_partition = false;
  config.tree_executor.enable_value_partition_join = false;
  auto index = MaterializedBoundaryIndex::build(variables, causal, trees, config);
  DirectedQuerySpec query;
  query.id = "high-bit-query";
  query.treatments = {ids[64]};
  query.outcomes = {ids[99]};
  assert(index.execute(query).exact_binding_count == 1);
  const std::string path = "/tmp/causal-index-v6-wide-test.bin";
  index.save(path);
  assert(MaterializedBoundaryIndex::load(path).execute(query).exact_binding_count == 1);
  std::remove(path.c_str());
}
