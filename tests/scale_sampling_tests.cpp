#define main benchmark_entry_for_sampling_test
#include "../benchmarks/boundary_benchmark.cpp"
#undef main
#include <cassert>

int main() {
  const auto dir = std::filesystem::temp_directory_path() /
      ("scale-sampling-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(dir);
  const auto path = dir / "assignments_raw.tsv";
  {
    std::ofstream out(path);
    for (int i=0;i<100;++i) out << "P=p" << i << "\tA\t" << i << "\n";
    for (int i=0;i<7;++i) out << "P=p" << i << "\tB\t" << i << "\n";
    out << "P=single\tC\t0\n";
  }
  std::map<std::string,std::set<InstanceId>> previous;
  for (std::size_t scale : {20,40,60,80,100}) {
    const auto input = loadVariables(path,0,scale);
    assert(input.size()==3);
    assert(input[0].instances.size()==scale);
    assert(input[1].instances.size()==std::max<std::size_t>(1,7*scale/100));
    assert(input[2].instances.size()==1);
    const auto again=loadVariables(path,0,scale);
    for (std::size_t v=0;v<input.size();++v) {
      std::set<InstanceId> ids, repeated;
      for (auto &r: input[v].instances) ids.insert(r.id);
      for (auto &r: again[v].instances) repeated.insert(r.id);
      assert(ids==repeated);
      for (auto id:previous[input[v].id]) assert(ids.count(id));
      previous[input[v].id]=ids;
    }
  }
  // A variable with no surviving matches is still part of the fixed DAG.
  {
    std::ofstream manifest(dir / "variable_manifest.csv");
    manifest << "variable_id,roles\nA,P\nB,P\nC,P\nEmpty,P\n";
  }
  const auto with_empty=loadVariables(path,0);
  assert(with_empty.size()==4);
  assert(with_empty.back().id=="Empty" && with_empty.back().instances.empty());
  assert(with_empty.back().pattern_roles==std::vector<RoleId>{"P"});
  std::filesystem::remove_all(dir);
}
