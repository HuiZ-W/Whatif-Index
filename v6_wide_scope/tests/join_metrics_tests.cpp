#include "causal_index_v4/FullLatticeIndex.h"
#include <cassert>
#include <map>
using namespace causal_index_v4;
int main() {
  std::vector<VariableInput> inputs;
  for (int i=0;i<3;++i) {
    auto id=std::string(1, 'A'+i);
    inputs.push_back({id, {"P"}, {{static_cast<InstanceId>(i+1), "all", {{"P","p"}}, id}}});
  }
  for (bool empty : {false, true}) {
    auto data=inputs;
    if (empty) data[2].instances[0].roles[0].node="different";
    std::vector<std::map<std::vector<std::string>,Count>> reference;
    for (bool reuse : {true,false}) {
      for (bool fused : {false,true}) {
        std::size_t calls=0, pairs=0, probes=0;
        LatticeBuildConfig cfg;
        cfg.enable_derive=false;
        cfg.store_all_supports=false;
        cfg.store_value_counts=true;
        cfg.enable_value_partition_join=false;
        cfg.variable_order_policy=VariableOrderPolicy::Alphabetical;
        cfg.execution_schedule=reuse ? ExecutionSchedule::DfsPrefixTree : ExecutionSchedule::IndependentSources;
        cfg.independent_leaf_pushdown=fused;
        cfg.binary_join_callback=[&]{++calls;};
        cfg.join_work_callback=[&](std::size_t rows, std::size_t matches){probes+=rows; pairs+=matches;};
        FullLatticeIndex index(data,cfg);
        index.build();
        assert(calls==(reuse ? 4 : 5));
        assert(pairs==(empty ? (reuse ? 1 : 2) : (reuse ? 4 : 5)));
        assert(probes==calls);
        if (reference.empty()) reference=index.result().value_counts;
        else assert(reference==index.result().value_counts);
      }
    }
  }
  for (std::size_t size : {2, 4}) {
    std::vector<VariableInput> data;
    for (int i=0; i<2; ++i) {
      VariableInput input;
      input.id=std::string(1, 'A'+i);
      input.pattern_roles={"P"};
      for (std::size_t k=0; k<size; ++k)
        input.instances.push_back({k+1, "all", {{"P", "p"}}, std::to_string(k)});
      data.push_back(input);
    }
    for (bool fused : {false,true}) {
      std::size_t calls=0, pairs=0, probes=0;
      LatticeBuildConfig cfg;
      cfg.enable_derive=false;
      cfg.store_all_supports=false;
      cfg.store_value_counts=true;
      cfg.enable_value_partition_join=false;
      cfg.execution_schedule=ExecutionSchedule::IndependentSources;
      cfg.independent_leaf_pushdown=fused;
      cfg.binary_join_callback=[&]{++calls;};
      cfg.join_work_callback=[&](std::size_t rows, std::size_t matches){probes+=rows; pairs+=matches;};
      FullLatticeIndex index(data,cfg);
      index.build();
      assert(calls==1);
      assert(probes==size);
      assert(pairs==size*size);
    }
  }
}
