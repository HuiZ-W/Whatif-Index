#include "whatif/SharedConstruction.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <chrono>
#include <random>

using namespace whatif;
using Scope = std::vector<VariableId>;
using Counts = std::map<Scope, ValueCountMap>;

static ValueCountMap counts(const Scope &scope, const std::vector<BoundaryRow> &rows) {
  ValueCountMap result;
  for (const auto &row : rows) {
    ValueTuple values;
    for (const auto &v : scope) for (const auto &b : row.bindings)
      if (b.variable == v) values.push_back(b.raw_value);
    result[values] += row.multiplicity;
  }
  return result;
}

// Independent exhaustive oracle: no executor helpers, cached supports or keys.
static ValueCountMap oracle(const std::vector<VariableInput> &inputs, const Scope &scope) {
  ValueCountMap result;
  std::function<void(std::size_t, std::map<std::string, std::string>, ValueTuple)> visit;
  visit = [&](std::size_t i, auto mapping, auto values) {
    if (i == scope.size()) { ++result[values]; return; }
    const auto &input = *std::find_if(inputs.begin(), inputs.end(), [&](const auto &v) { return v.id == scope[i]; });
    for (const auto &row : input.instances) {
      auto next = mapping;
      bool valid = true;
      for (const auto &r : row.roles) {
        auto p = next.find(r.role);
        if (p != next.end() && p->second != r.node) valid = false;
        for (const auto &[role, node] : next) if (role != r.role && node == r.node) valid = false;
        next[r.role] = r.node;
      }
      if (valid) { auto v = values; v.push_back(row.value); visit(i + 1, next, v); }
    }
  };
  visit(0, {}, {});
  return result;
}

static void recoveryParts() {
  std::vector<VariableInput> inputs{{"W", {"x", "y", "w"}, {}}, {"X", {"x"}, {}}, {"Y", {"y"}, {}}};
  for (int k = 0; k < 8; ++k) {
    inputs[1].pattern_roles.push_back("px" + std::to_string(k));
    inputs[2].pattern_roles.push_back("py" + std::to_string(k));
  }
  for (int i = 0; i < 5; ++i) for (int side = 1; side <= 2; ++side) {
    core::MatchInstance row;
    row.id = i; row.value = "same";
    for (const auto &role : inputs[side].pattern_roles) row.roles.push_back({role, role + std::to_string(i)});
    inputs[side].instances.push_back(row);
  }
  for (int i = 0; i < 4; ++i) for (int copy = 0; copy < 2; ++copy)
    inputs[0].instances.push_back({static_cast<InstanceId>(i*2+copy), "", {
        {"x", "x"+std::to_string(i)}, {"y", "y"+std::to_string(i)},
        {"w", "w"+std::to_string(i*2+copy)}}, "same"});
  BoundaryBuildConfig config;
  config.tree_executor.variable_order_policy = core::VariableOrderPolicy::Alphabetical;
  config.tree_executor.enable_derive = true;
  Counts outputs;
  auto stats = constructSharedScopes(inputs, {{"W","X","Y"}, {"X","Y"}}, config,
      [&](const auto &scope, const auto &rows) { outputs[scope] = counts(scope, rows); });
  assert(outputs.at({"X","Y"}) == oracle(inputs, {"X","Y"}));
  assert(outputs.at({"X","Y"}).at({"same","same"}) == 25);
  assert(stats.recovery_used == 1 && stats.recovered_rows == 4);
  assert((stats.completion_rows == std::array<std::size_t,4>{12,4,4,1}));
  assert(stats.donor_peak_bytes <= config.tree_executor.rollup_cache_bytes);
  config.tree_executor.rollup_cache_bytes = 0;
  auto no_cache = constructSharedScopes(inputs, {{"W","X","Y"}, {"X","Y"}}, config,
      [&](const auto &scope, const auto &rows) { assert(outputs.at(scope) == counts(scope, rows)); });
  assert(no_cache.recovery_used == 0 && no_cache.donor_peak_bytes == 0);
  config.tree_executor.execution_schedule = core::ExecutionSchedule::IndependentSources;
  config.tree_executor.rollup_cache_bytes = 1024*1024;
  auto independent = constructSharedScopes(inputs, {{"W","X","Y"}, {"X","Y"}}, config,
      [&](const auto &scope, const auto &rows) { assert(outputs.at(scope) == counts(scope, rows)); });
  assert(independent.recovery_used == 0);
  VariableInput z{"Z",{"z"},{}};
  for(int k=0;k<8;++k) z.pattern_roles.push_back("pz"+std::to_string(k));
  for(int i=0;i<3;++i) {
    core::MatchInstance row; row.id=i; row.value="same";
    for(const auto &r:z.pattern_roles) row.roles.push_back({r,r+std::to_string(i)});
    z.instances.push_back(row);
  }
  inputs.push_back(z);
  config.tree_executor.execution_schedule=core::ExecutionSchedule::DfsPrefixTree;
  auto larger=constructSharedScopes(inputs,{{"W","X","Y","Z"},{"X","Y","Z"}},config,
      [&](const auto &scope,const auto &rows) { assert(counts(scope,rows)==oracle(inputs,scope)); });
  assert(larger.recovery_used>=2); // XY prefix and then the (XY)-Z binary join.
}

static void sharedTreesAndWideScopes() {
  std::vector<VariableInput> inputs;
  for (const auto &id : {"A","B","C","D"}) inputs.push_back({id, {"p"}, {{1,"",{{"p","p1"}},"1"}, {2,"",{{"p","p2"}},"2"}}});
  BoundaryBuildConfig config;
  config.include_all_reachable_pairs = false;
  config.tree_executor.variable_order_policy = core::VariableOrderPolicy::Alphabetical;
  const auto index = MaterializedBoundaryIndex::build(inputs, {}, {{"T1",{"A","B","C"}}, {"T2",{"A","B","D"}}}, config);
  assert(index.cubes().size() == 11); // 7+7 minus A, B, AB.
  assert(&index.cube("T1",{"A","B"}) == &index.cube("T2",{"A","B"}));
  assert(index.constructionStats().prefix_extensions == 11);
  BoundaryPlan boundary_plan;
  boundary_plan.requirements = {{"left","T1",{"A","B"},kBoundarySourceSide,{"q"}},
                               {"right","T2",{"A","B"},kBoundaryTargetSide,{"q"}}};
  BoundaryQueryPlan query;
  query.query_id="q"; query.global_scope={"A","B"}; query.view_ids={"left","right"};
  boundary_plan.queries.push_back(query);
  auto shared_config=config; shared_config.precompiled_boundary_plan=boundary_plan;
  const auto shared=MaterializedBoundaryIndex::build(inputs,{},{{"T1",{"A","B","C"}}, {"T2",{"A","B","D"}}},shared_config);
  assert(shared.boundaryViews().size()==1 && shared.plan().requirements.size()==1);
  assert(shared.plan().queries.front().view_ids==std::vector<std::string>{"left"});
  assert(shared.execute("q").value_counts==oracle(inputs,{"A","B"}));
  auto local_config=config;
  local_config.cube_policy=LocalCubePolicy::ExplicitScopes;
  query.view_ids.clear(); boundary_plan.requirements.clear();
  boundary_plan.queries={query}; local_config.precompiled_boundary_plan=boundary_plan;
  const auto local=MaterializedBoundaryIndex::build(inputs,{},{{"T1",{"A","B","C","D"}}},local_config);
  assert(local.cube("T1",{"A","B"}).counts==oracle(inputs,{"A","B"}));
  BoundaryBuildConfig greedy = config;
  greedy.tree_executor.variable_order_policy = core::VariableOrderPolicy::PartialWorkloadGreedy;
  std::vector<Scope> task_scopes{{"A","B"},{"A","C"},{"A","B","D"},{"B","D"}};
  auto order_stats = constructSharedScopes(inputs,task_scopes,greedy,[](const auto &,const auto &){});
  std::set<std::string> placed;
  for (const auto &chosen : order_stats.variable_order) {
    std::pair<std::size_t,std::string> best{std::numeric_limits<std::size_t>::max(),""};
    for (const auto &input : inputs) if (!placed.count(input.id)) {
      std::set<Scope> prefixes;
      for (const auto &s : task_scopes) if (std::find(s.begin(),s.end(),input.id)!=s.end()) {
        Scope p; for (const auto &v:s) if (placed.count(v)) p.push_back(v); prefixes.insert(p);
      }
      // All joins in this fixture have exactly two rows, so the singleton
      // statistics estimate equals the exact M_t score in the paper.
      std::size_t score=0; for (const auto &p:prefixes) score+=p.empty()?2:4;
      best=std::min(best,std::make_pair(score,input.id));
    }
    assert(chosen==best.second); placed.insert(chosen);
  }
  const auto file = std::filesystem::temp_directory_path() /
      ("shared-index-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
  index.save(file.string());
  auto loaded = MaterializedBoundaryIndex::load(file.string());
  assert(&loaded.cube("T1",{"A","B"}) == &loaded.cube("T2",{"A","B"}));
  assert(loaded.cube("T2",{"A","B"}).counts == oracle(inputs,{"A","B"}));
  std::filesystem::remove(file);
  inputs.clear();
  std::vector<Scope> scopes;
  for (int i=0;i<80;++i) { auto id="V"+std::to_string(i); inputs.push_back({id,{"p"},{{1,"",{{"p","p1"}},"1"}}}); scopes.push_back({id}); }
  auto stats = constructSharedScopes(inputs, scopes, config, [](const auto &, const auto &rows) { assert(rows.size()==1); });
  assert(stats.unique_scopes == 80 && stats.prefix_extensions == 80);
}

static void randomizedOracle() {
  std::mt19937 random(71829);
  for (int round=0;round<40;++round) {
    std::vector<VariableInput> input{{"A",{"a","b"},{}},{"B",{"b","c"},{}},{"C",{"c","d"},{}},{"D",{"d","a"},{}}};
    for (auto &v : input) for (int i=0;i<5;++i) {
      core::MatchInstance row; row.id=i; row.value=std::to_string(random()%2);
      for (const auto &role : v.pattern_roles) row.roles.push_back({role,role+std::to_string(random()%3)});
      v.instances.push_back(row);
    }
    if (round%5==0) input[2].instances.clear();
    std::vector<Scope> scopes;
    for (unsigned mask=1;mask<16;++mask) { Scope s; for (unsigned j=0;j<4;++j) if(mask&(1u<<j)) s.push_back(input[j].id); scopes.push_back(s); }
    Counts expected; for (const auto &s:scopes) expected[s]=oracle(input,s);
    for (int mode=0;mode<3;++mode) {
      BoundaryBuildConfig config;
      config.tree_executor.variable_order_policy = mode==1 ? core::VariableOrderPolicy::Random : core::VariableOrderPolicy::PartialWorkloadGreedy;
      config.tree_executor.variable_order_seed=round;
      config.tree_executor.scope_order_seed=round+13;
      config.tree_executor.scope_order_policy=mode==1 ? core::ScopeOrderPolicy::Random : core::ScopeOrderPolicy::DfsLexicographic;
      if(mode==2) config.tree_executor.execution_schedule=core::ExecutionSchedule::IndependentSources;
      auto stats=constructSharedScopes(input, scopes, config, [&](const auto &s,const auto &r) { assert(counts(s,r)==expected.at(s)); });
      auto again=constructSharedScopes(input, scopes, config, [](const auto &,const auto &) {});
      assert(stats.variable_order==again.variable_order && stats.construction_order==again.construction_order);
    }
  }
}

static void largerTargetAndConstraints() {
  std::vector<VariableInput> input{{"A",{"a","b"},{}},{"B",{"a"},{}},{"C",{"b"},{}},{"D",{"d"},{}}};
  for (int i=0;i<4;++i) {
    input[0].instances.push_back({static_cast<InstanceId>(i),"",{{"a","a"+std::to_string(i)},{"b","b"+std::to_string(i)}},"a"});
    input[1].instances.push_back({static_cast<InstanceId>(i),"",{{"a","a"+std::to_string(i)}},"b"});
    input[2].instances.push_back({static_cast<InstanceId>(i),"",{{"b","b"+std::to_string(i)}},"c"});
    // d deliberately collides with another role's node; global injectivity
    // must be checked even when the two patterns share no role name.
    input[3].instances.push_back({static_cast<InstanceId>(i),"",{{"d",i==0?"a0":"d"+std::to_string(i)}},"d"});
  }
  BoundaryBuildConfig config;
  config.tree_executor.variable_order_policy=core::VariableOrderPolicy::Alphabetical;
  std::size_t work=0;
  config.tree_executor.join_work_callback=[&](auto, auto pairs) { work+=pairs; };
  constructSharedScopes(input,{{"A","D"}},config,[](const auto &,const auto &){});
  assert(work==15); // 16 same-key candidates minus one injectivity collision.
  constructSharedScopes(input,{{"A","B","C","D"},{"B","C","D"}},config,
      [&](const auto &scope,const auto &rows) { assert(counts(scope,rows)==oracle(input,scope)); });
  auto invalid=input; invalid.push_back(input.front());
  bool rejected=false;
  try { constructSharedScopes(invalid,{{"A"}},config,[](const auto &,const auto &){}); }
  catch(const std::runtime_error &) { rejected=true; }
  assert(rejected);
}
int main() { recoveryParts(); sharedTreesAndWideScopes(); randomizedOracle(); largerTargetAndConstraints(); }
