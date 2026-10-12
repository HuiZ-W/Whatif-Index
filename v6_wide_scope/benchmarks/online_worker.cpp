#include "causal_index_v5/BoundaryJoinIndex.h"
#include "causal_index_v5/CausalAdjustment.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace causal_index_v5;
using boost::property_tree::ptree;

namespace {
ptree strings(const std::vector<std::string> &values) {
  ptree result;
  for (const auto &value : values) { ptree item; item.put_value(value); result.push_back({"", item}); }
  return result;
}
void emit(const ptree &value) {
  boost::property_tree::write_json(std::cout, value, false);
  std::cout.flush();
}
CausalDag loadDag(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open causal.tsv");
  std::vector<std::string> variables;
  std::vector<std::pair<std::string, std::string>> edges;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    std::string child, parents, parent;
    std::getline(row, child, '\t'); std::getline(row, parents);
    if (!parents.empty() && parents.back() == '\r') parents.pop_back();
    variables.push_back(child);
    std::istringstream list(parents);
    while (std::getline(list, parent, ',')) if (!parent.empty()) edges.emplace_back(parent, child);
  }
  return CausalDag(std::move(variables), edges);
}
}

int main(int argc, char **argv) {
  try {
    if (argc != 4) throw std::runtime_error("Usage: online_worker snapshot causal.tsv FullCube|Ours");
    const std::string method = argv[3];
    if (method != "FullCube" && method != "Ours") throw std::runtime_error("unknown method");
    const auto index = MaterializedBoundaryIndex::load(argv[1]);
    const auto dag = loadDag(argv[2]);
    ptree ready;
    ready.put("status", "ready"); ready.put("method", method);
    ready.put("scope_lookup", "hash");
    ready.put("boundary_planner", "coverage_first_v2");
    std::size_t multi_views = 0;
    for (const auto &view : index.boundaryViews()) multi_views += view.scope.size() > 1;
    ready.put("multivariable_views", multi_views);
    ready.put("cubes", index.cubes().size()); ready.put("views", index.boundaryViews().size());
    std::set<std::string> variables;
    for (const auto &tree : index.parentTrees()) variables.insert(tree.variables.begin(), tree.variables.end());
    ready.add_child("variables", strings({variables.begin(), variables.end()}));
    ptree parent_trees;
    for (const auto &tree : index.parentTrees()) {
      ptree item;
      item.put("id", tree.id);
      item.add_child("variables", strings(tree.variables));
      parent_trees.push_back({"", item});
    }
    ready.add_child("parent_trees", parent_trees);
    emit(ready);
    std::string line;
    while (std::getline(std::cin, line)) {
      ptree response;
      try {
        std::istringstream stream(line);
        ptree request;
        boost::property_tree::read_json(stream, request);
        if (request.get<std::string>("operation", "") == "enumerate_cube_adjustments") {
          std::set<std::set<std::string>> scopes;
          for (const auto &cube : index.cubes()) {
            if (cube.scope.size() >= 3 && cube.scope.size() <= 5)
              scopes.emplace(cube.scope.begin(), cube.scope.end());
          }
          ptree candidates;
          for (const auto &scope : scopes) {
            for (const auto &x : scope) for (const auto &y : scope) {
              if (x == y) continue;
              std::vector<std::string> z;
              for (const auto &v : scope) if (v != x && v != y) z.push_back(v);
              if (!dag.isValidAdjustment(dag.maskOf({x}), dag.maskOf({y}), dag.maskOf(z))) continue;
              ptree candidate;
              candidate.put("X", x); candidate.put("Y", y);
              candidate.add_child("Z", strings(z));
              candidates.push_back({"", candidate});
            }
          }
          response.put("status", "enumerated");
          response.add_child("candidates", candidates);
          emit(response);
          continue;
        }
        const auto x = request.get<std::string>("X"), y = request.get<std::string>("Y");
        const auto xv = request.get<std::string>("x"), yv = request.get<std::string>("y");
        std::vector<std::string> z;
        for (const auto &item : request.get_child("Z")) z.push_back(item.second.get_value<std::string>());
        const auto xm = dag.maskOf({x}), ym = dag.maskOf({y}), zm = dag.maskOf(z);
        if ((xm & ym) != 0 || (xm & zm) != 0 || (ym & zm) != 0 || causalMaskCount(zm) != z.size() || !dag.isValidAdjustment(xm, ym, zm))
          throw std::runtime_error("invalid_adjustment");
        std::vector<std::string> scope{x, y}; scope.insert(scope.end(), z.begin(), z.end());
        const auto output_scope = scope;
        if (const auto full = request.get_child_optional("match_scope")) {
          std::set<std::string> seen;
          for (const auto &v : *full) {
            const auto id = v.second.get_value<std::string>();
            if (!seen.insert(id).second) throw std::runtime_error("duplicate match variable");
            if (std::find(scope.begin(), scope.end(), id) == scope.end()) scope.push_back(id);
          }
          for (const auto &v : output_scope)
            if (!seen.count(v)) throw std::runtime_error("match scope misses causal variable");
        }
        OnlineScopeStats stats;
        const auto start = std::chrono::steady_clock::now();
        const bool plan_only = request.get<bool>("plan_only", false);
        const auto result = index.executeExactScope({request.get<std::string>("query_id", "online"), scope}, output_scope, false, method == "FullCube", &stats, plan_only);
        response.put("eligible_multivariable_views", stats.eligible_multivariable_views);
        response.put("selected_multivariable_views", stats.selected_multivariable_views);
        response.put("selected_attachment_views", stats.selected_attachment_views);
        if (plan_only) {
          response.put("status", "planned");
          response.put("cube_hit", stats.cube_hit);
          response.add_child("component_ids", strings(stats.component_ids));
          response.add_child("view_ids", strings(stats.view_ids));
          response.put("component_count", stats.component_ids.size());
          emit(response);
          continue;
        }
        struct Counts { Count nz = 0, nxz = 0, nxyz = 0; };
        std::map<ValueTuple, Counts> strata;
        for (const auto &entry : result.value_counts) {
          ValueTuple key(entry.first.begin() + 2, entry.first.end());
          auto &counts = strata[key]; counts.nz += entry.second;
          if (entry.first[0] == xv) {
            counts.nxz += entry.second;
            if (entry.first[1] == yv) counts.nxyz += entry.second;
          }
        }
        bool supported = result.exact_binding_count != 0;
        long double answer = 0;
        ptree records;
        for (const auto &entry : strata) {
          const auto &c = entry.second;
          if (c.nxz == 0) supported = false;
          else answer += static_cast<long double>(c.nz) / result.exact_binding_count * c.nxyz / c.nxz;
          ptree record; record.add_child("z", strings(entry.first));
          record.put("nz", c.nz); record.put("nxz", c.nxz); record.put("nxyz", c.nxyz);
          records.push_back({"", record});
        }
        response.put("status", supported ? "ok" : "insufficient_support");
        response.put("N", result.exact_binding_count);
        if (supported) response.put("answer", answer);
        // Strata are returned only for correctness validation, not formal timing.
        if (request.get<bool>("include_counts", false)) response.add_child("strata", records);
        response.put("cube_hit", stats.cube_hit);
        response.add_child("component_ids", strings(stats.component_ids));
        response.add_child("view_ids", strings(stats.view_ids));
        response.put("component_count", stats.component_ids.size());
        response.put("join_calls", stats.join_calls);
        response.put("join_probe_rows", stats.join_probe_rows);
        response.put("join_work_pairs", stats.join_work_pairs);
        response.put("execution_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      } catch (const std::exception &error) {
        const std::string message = error.what();
        response.put("status", message.find("not covered by saved views") != std::string::npos || message.find("index_unavailable") != std::string::npos ? "index_unavailable" : "error");
        response.put("error", message);
      }
      emit(response);
    }
  } catch (const std::exception &error) {
    ptree failure; failure.put("status", "index_unavailable"); failure.put("error", error.what()); emit(failure);
    return 1;
  }
}
