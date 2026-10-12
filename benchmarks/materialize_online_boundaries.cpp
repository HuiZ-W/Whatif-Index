#include "whatif/BoundaryJoinIndex.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
  try {
    if (argc != 4) throw std::runtime_error("Usage: whatif_materialize_boundaries input.bin scopes.tsv output.bin");
    if (std::filesystem::exists(argv[3])) throw std::runtime_error("Output already exists");
    auto index = whatif::MaterializedBoundaryIndex::load(argv[1]);
    std::ifstream input(argv[2]);
    if (!input) throw std::runtime_error("Cannot open scopes file");
    std::vector<whatif::ExactScopeSpec> queries;
    std::string line;
    while (std::getline(input, line)) {
      std::istringstream fields(line);
      whatif::ExactScopeSpec query;
      std::string variable;
      query.id = std::to_string(queries.size());
      while (fields >> variable) query.scope.push_back(variable);
      if (!query.scope.empty()) queries.push_back(std::move(query));
    }
    const auto start = std::chrono::steady_clock::now();
    const auto before = index.boundaryViews().size();
    index.materializeExactBoundaryViews(queries);
    const auto stats = index.saveWithStats(argv[3]);
    std::cout << "added_views=" << index.boundaryViews().size() - before
              << "\nindex_total_bytes=" << stats.total_bytes
              << "\noffline_materialization_ms="
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
              << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
