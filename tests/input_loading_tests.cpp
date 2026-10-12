#define main benchmark_entry_for_input_test
#include "../benchmarks/boundary_benchmark.cpp"
#undef main
#include <cassert>

int main() {
  const auto dir = std::filesystem::temp_directory_path() /
      ("input-loading-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(dir);
  const auto path = dir / "assignments_raw.tsv";
  {
    std::ofstream out(path);
    for (int i=0;i<100;++i) out << "P=p" << i << "\tA\t" << i << "\n";
    for (int i=0;i<7;++i) out << "P=p" << i << "\tB\t" << i << "\n";
    out << "P=single\tC\t0\n";
  }
  const auto input = loadVariables(path);
  assert(input.size()==3);
  assert(input[0].instances.size()==100);
  assert(input[1].instances.size()==7);
  assert(input[2].instances.size()==1);
  for (std::size_t i=0;i<100;++i)
    assert(input[0].instances[i].value==std::to_string(i));
  // A variable with no surviving matches is still part of the fixed DAG.
  {
    std::ofstream manifest(dir / "variable_manifest.csv");
    manifest << "variable_id,roles\nA,P\nB,P\nC,P\nEmpty,P\n";
  }
  const auto with_empty=loadVariables(path);
  assert(with_empty.size()==4);
  assert(with_empty.back().id=="Empty" && with_empty.back().instances.empty());
  assert(with_empty.back().pattern_roles==std::vector<RoleId>{"P"});
  std::filesystem::remove_all(dir);
}
