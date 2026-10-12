#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "whatif/PatternManifest.h"

using boost::property_tree::ptree;
namespace {
using Deadline = std::chrono::steady_clock::time_point;
struct Timeout : std::runtime_error {
  Timeout() : std::runtime_error("online join timeout") {}
};
struct Row {
  std::map<std::string, std::string> bindings;
  std::map<std::string, std::string> values;
};
struct Metrics {
  std::uint64_t calls = 0, probes = 0, candidates = 0, output = 0;
};
void check(Deadline deadline) {
  if (std::chrono::steady_clock::now() >= deadline) throw Timeout();
}
void emit(const ptree &value) {
  boost::property_tree::write_json(std::cout, value, false);
  std::cout.flush();
}
ptree strings(const std::vector<std::string> &values) {
  ptree result;
  for (const auto &v : values) { ptree item; item.put_value(v); result.push_back({"", item}); }
  return result;
}
bool compatible(const Row &a, const Row &b) {
  for (const auto &left : a.bindings)
    for (const auto &right : b.bindings) {
      if (left.first == right.first && left.second != right.second) return false;
      if (left.first != right.first && left.second == right.second) return false;
    }
  for (const auto &left : a.values)
    for (const auto &right : b.values)
      if (left.first == right.first && left.second != right.second) return false;
  return true;
}
// Deduplicate by complete object bindings, not by observed attribute values.
std::vector<Row> distinctRows(const std::vector<Row> &rows, Deadline deadline) {
  std::map<std::map<std::string, std::string>, std::map<std::string, std::string>> seen;
  std::vector<Row> result;
  for (const auto &row : rows) {
    check(deadline);
    const auto entry = seen.emplace(row.bindings, row.values);
    if (entry.second) result.push_back(row);
    else if (entry.first->second != row.values) throw std::runtime_error("conflicting mapping values");
  }
  return result;
}
class OnlineJoinEngine {
 public:
  explicit OnlineJoinEngine(const std::string &directory) {
    const auto schemas = whatif::loadPatternSchemas(std::filesystem::path(directory) / "variable_manifest.csv");
    for (const auto &[v, roles] : schemas) relations_[v];
    std::ifstream input(directory + "/assignments_raw.tsv");
    if (!input) throw std::runtime_error("cannot open assignments_raw.tsv");
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty() || line.front() == '#') continue;
      const auto first = line.find('\t');
      const auto second = first == std::string::npos ? first : line.find('\t', first + 1);
      if (second == std::string::npos) throw std::runtime_error("invalid assignment row");
      const auto variable = line.substr(first + 1, second - first - 1);
      Row row;
      std::istringstream binding(line.substr(0, first));
      std::string part;
      std::set<std::string> nodes;
      while (std::getline(binding, part, '|')) {
        const auto equal = part.find('=');
        if (equal == std::string::npos) throw std::runtime_error("invalid role binding");
        const auto role = part.substr(0, equal), node = part.substr(equal + 1);
        if (!row.bindings.emplace(role, node).second || !nodes.insert(node).second)
          throw std::runtime_error("non-injective or duplicate binding");
      }
      if (row.bindings.empty()) throw std::runtime_error("empty binding");
      if (!schemas.empty()) {
        std::vector<std::string> actual;
        for (const auto &[role,node] : row.bindings) actual.push_back(role);
        const auto s=schemas.find(variable);
        if(s==schemas.end() || actual!=s->second) throw std::runtime_error("assignment/manifest schema mismatch");
      }
      row.values.emplace(variable, line.substr(second + 1));
      relations_[variable].push_back(row);
      ++records_;
    }
  }
  ptree ready() const {
    ptree result;
    result.put("status", "ready"); result.put("method", "OnlineJoin");
    result.put("variables", relations_.size()); result.put("records", records_);
    result.put("population", "sampled_vertex_binding_union");
    result.put("join_indexes", "none_persisted"); result.put("join_algorithm", "nested_loop");
    result.put("join_order", "X_Y_Z_request_order");
    result.put("input", "precomputed_match_bindings");
    result.put("row_representation", "string_maps_v1");
    result.put("integer_encoding", false);
    result.put("intermediate_deduplication", "complete_binding_ordered_map");
    return result;
  }
  ptree query(const ptree &request) const {
    std::vector<std::string> scope{request.get<std::string>("X"), request.get<std::string>("Y")};
    for (const auto &v : request.get_child("Z")) scope.push_back(v.second.get_value<std::string>());
    if (std::set<std::string>(scope.begin(), scope.end()).size() != scope.size())
      throw std::runtime_error("repeated query variable");
    const auto count_width = scope.size();
    if (const auto full = request.get_child_optional("match_scope")) {
      std::vector<std::string> specified;
      for (const auto &v : *full) specified.push_back(v.second.get_value<std::string>());
      if (std::set<std::string>(specified.begin(), specified.end()).size() != specified.size())
        throw std::runtime_error("duplicate match variable");
      for (const auto &v : scope)
        if (std::find(specified.begin(), specified.end(), v) == specified.end())
          throw std::runtime_error("match scope misses causal variable");
      for (const auto &v : specified)
        if (std::find(scope.begin(), scope.end(), v) == scope.end()) scope.push_back(v);
    }
    const auto timeoutMs = request.get<double>("timeout_ms", 0);
    const auto deadline = timeoutMs <= 0 ? Deadline::max() : std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double, std::milli>(timeoutMs));
    Metrics metrics;
    for (const auto &v : scope)
      if (!relations_.count(v)) throw std::runtime_error("unknown variable " + v);
    auto current = distinctRows(relations_.at(scope.front()), deadline);
    for (std::size_t i = 1; i < scope.size() && !current.empty(); ++i) {
      std::vector<Row> next;
      ++metrics.calls;
      for (const auto &left : current) {
        ++metrics.probes;
        for (const auto &right : relations_.at(scope[i])) {
          check(deadline); ++metrics.candidates;
          if (!compatible(left, right)) continue;
          Row merged = left;
          merged.bindings.insert(right.bindings.begin(), right.bindings.end());
          merged.values.insert(right.values.begin(), right.values.end());
          next.push_back(merged); ++metrics.output;
        }
      }
      current = distinctRows(next, deadline);
    }
    struct Counts { std::uint64_t nz = 0, nxz = 0, nxyz = 0; };
    std::map<std::vector<std::string>, Counts> strata;
    for (const auto &row : current) {
      check(deadline);
      std::vector<std::string> z;
      for (std::size_t i = 2; i < count_width; ++i) z.push_back(row.values.at(scope[i]));
      auto &counts = strata[z]; ++counts.nz;
      if (row.values.at(scope[0]) == request.get<std::string>("x")) {
        ++counts.nxz;
        if (row.values.at(scope[1]) == request.get<std::string>("y")) ++counts.nxyz;
      }
    }
    bool supported = !current.empty();
    long double answer = 0;
    for (const auto &s : strata) {
      if (!s.second.nxz) supported = false;
      else answer += static_cast<long double>(s.second.nz) / current.size() * s.second.nxyz / s.second.nxz;
    }
    ptree response;
    response.put("status", supported ? "ok" : "insufficient_support");
    response.put("N", current.size());
    if (supported) response.put("answer", static_cast<double>(answer));
    response.put("join_calls", metrics.calls); response.put("join_build_rows", 0);
    response.put("join_probe_rows", metrics.probes); response.put("join_candidate_pairs", metrics.candidates);
    response.put("join_output_rows", metrics.output);
    if (request.get<bool>("include_counts", false)) {
      ptree rows;
      for (const auto &s : strata) {
        ptree item; item.add_child("z", strings(s.first));
        item.put("nz", s.second.nz); item.put("nxz", s.second.nxz); item.put("nxyz", s.second.nxyz);
        rows.push_back({"", item});
      }
      response.add_child("strata", rows);
    }
    return response;
  }
 private:
  std::map<std::string, std::vector<Row>> relations_;
  std::uint64_t records_ = 0;
};
}
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: whatif_online_join dataset_directory");
    const auto load_start = std::chrono::steady_clock::now();
    const OnlineJoinEngine engine(argv[1]);
    auto ready = engine.ready();
    ready.put("load_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count());
    emit(ready);
    std::string line;
    while (std::getline(std::cin, line)) {
      ptree response;
      const auto start = std::chrono::steady_clock::now();
      try {
        std::istringstream stream(line); ptree request;
        boost::property_tree::read_json(stream, request); response = engine.query(request);
      } catch (const Timeout &) { response.put("status", "timeout"); }
        catch (const std::exception &error) { response.put("status", "error"); response.put("error", error.what()); }
      response.put("execution_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      emit(response);
    }
  } catch (const std::exception &error) {
    ptree response; response.put("status", "input_unavailable"); response.put("error", error.what());
    emit(response); return 1;
  }
}
