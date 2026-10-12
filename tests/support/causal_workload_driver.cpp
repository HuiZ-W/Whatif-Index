#include "reference/CausalAdjustment.h"
#include "whatif/core/FullLatticeIndex.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace whatif::core;
using whatif::testing::CausalDag;
using whatif::testing::CausalMask;

namespace {

struct ParsedLine {
  std::string sample;
  std::string variable;
  std::string value;
};

struct NormalInstance {
  InstanceId id = 0;
  std::string value;
  std::vector<RoleBinding> roles;
  std::unordered_map<RoleId, NodeId> role_to_node;
};

struct NormalVariable {
  VariableId id;
  std::vector<RoleId> roles;
  std::vector<NormalInstance> instances;
};

struct NormalRow {
  std::vector<InstanceId> instance_ids;
  std::vector<std::string> values;
  std::unordered_map<RoleId, NodeId> role_to_node;
  std::unordered_set<NodeId> used_nodes;
};

using ValueKey = std::vector<std::string>;
using ValueCountMap = std::map<ValueKey, Count>;

std::vector<std::string> split(const std::string& text, char delim) {
  std::vector<std::string> out;
  std::string item;
  std::stringstream ss(text);
  while (std::getline(ss, item, delim)) {
    out.push_back(item);
  }
  return out;
}

bool parseTsvLine(const std::string& line, ParsedLine& out) {
  if (line.empty() || line[0] == '#') {
    return false;
  }
  const auto cols = split(line, '\t');
  if (cols.size() < 3) {
    return false;
  }
  out.sample = cols[0];
  out.variable = cols[1];
  out.value = cols[2];
  return true;
}

std::vector<RoleBinding> parseSampleRoles(const std::string& sample) {
  std::vector<RoleBinding> roles;
  for (const auto& part : split(sample, '|')) {
    const auto eq = part.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    roles.push_back({part.substr(0, eq), part.substr(eq + 1)});
  }
  std::sort(roles.begin(), roles.end(), [](const auto& left, const auto& right) {
    return left.role < right.role;
  });
  return roles;
}

std::map<std::string, std::size_t> countVariables(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open assignments file: " + path);
  }
  std::map<std::string, std::size_t> counts;
  std::string line;
  ParsedLine parsed;
  while (std::getline(in, line)) {
    if (parseTsvLine(line, parsed)) {
      counts[parsed.variable] += 1;
    }
  }
  return counts;
}

std::vector<std::string> chooseSmallestVariables(const std::map<std::string, std::size_t>& counts,
                                                 std::size_t n) {
  std::vector<std::pair<std::string, std::size_t>> items(counts.begin(), counts.end());
  std::sort(items.begin(), items.end(), [](const auto& left, const auto& right) {
    if (left.second != right.second) {
      return left.second < right.second;
    }
    return left.first < right.first;
  });
  if (items.size() < n) {
    n = items.size();
  }
  std::vector<std::string> out;
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(items[i].first);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<VariableInput> loadSelectedVariables(const std::string& path,
                                                 const std::vector<std::string>& selected) {
  std::set<std::string> selected_set(selected.begin(), selected.end());
  std::map<std::string, std::vector<MatchInstance>> by_variable;
  std::map<std::string, std::set<RoleId>> roles_by_variable;

  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open assignments file: " + path);
  }
  std::string line;
  ParsedLine parsed;
  InstanceId id = 1;
  while (std::getline(in, line)) {
    if (!parseTsvLine(line, parsed) || selected_set.count(parsed.variable) == 0) {
      continue;
    }
    auto roles = parseSampleRoles(parsed.sample);
    for (const auto& role : roles) {
      roles_by_variable[parsed.variable].insert(role.role);
    }
    by_variable[parsed.variable].push_back({id++, parsed.value, std::move(roles), parsed.value});
  }

  std::vector<VariableInput> variables;
  for (const auto& variable : selected) {
    auto role_set = roles_by_variable[variable];
    variables.push_back({variable,
                         std::vector<RoleId>(role_set.begin(), role_set.end()),
                         std::move(by_variable[variable])});
  }
  return variables;
}


void sampleInstancesRandom(std::vector<VariableInput>& variables,
                           std::size_t min_instances_per_variable,
                           std::size_t max_instances_per_variable,
                           std::uint64_t seed) {
  if (max_instances_per_variable == 0) {
    return;
  }
  if (min_instances_per_variable > max_instances_per_variable) {
    throw std::runtime_error("instance sample minimum exceeds maximum");
  }
  for (std::size_t i = 0; i < variables.size(); ++i) {
    auto& instances = variables[i].instances;
    std::mt19937_64 rng(seed + 0x9e3779b97f4a7c15ULL * (i + 1));
    std::uniform_int_distribution<std::size_t> target_distribution(
        min_instances_per_variable, max_instances_per_variable);
    const std::size_t target = target_distribution(rng);
    if (instances.size() <= target) {
      continue;
    }
    std::shuffle(instances.begin(), instances.end(), rng);
    instances.resize(target);
    std::sort(instances.begin(), instances.end(), [](const auto& left, const auto& right) {
      return left.id < right.id;
    });
  }
}

void sampleInstancesLogScale(std::vector<VariableInput>& variables,
                             std::size_t min_exponent,
                             std::size_t max_exponent,
                             std::uint64_t seed,
                             const std::string& scale_policy,
                             const std::map<std::string, std::size_t>& workload_coverage) {
  if (min_exponent > max_exponent) {
    throw std::runtime_error("log-scale sample minimum exponent exceeds maximum");
  }
  std::vector<std::size_t> levels;
  std::size_t power = 1;
  for (std::size_t exponent = 0; exponent <= max_exponent; ++exponent) {
    if (exponent >= min_exponent) {
      levels.push_back(power);
    }
    if (exponent != max_exponent) {
      if (power > std::numeric_limits<std::size_t>::max() / 10) {
        throw std::runtime_error("log-scale sample exponent is too large");
      }
      power *= 10;
    }
  }
  if (levels.empty() || variables.empty()) {
    return;
  }

  std::vector<std::size_t> targets;
  targets.reserve(variables.size());
  for (std::size_t i = 0; i < variables.size(); ++i) {
    targets.push_back(levels[i % levels.size()]);
  }
  std::sort(targets.begin(), targets.end(), std::greater<std::size_t>());

  std::vector<std::size_t> remaining;
  remaining.reserve(variables.size());
  for (std::size_t i = 0; i < variables.size(); ++i) {
    remaining.push_back(i);
  }

  auto coverage = [&](std::size_t variable) {
    const auto found = workload_coverage.find(variables[variable].id);
    return found == workload_coverage.end() ? std::size_t{0} : found->second;
  };

  for (const auto target : targets) {
    std::vector<std::size_t> feasible;
    for (const auto variable : remaining) {
      if (variables[variable].instances.size() >= target) {
        feasible.push_back(variable);
      }
    }
    if (feasible.empty()) {
      throw std::runtime_error(
          "insufficient instances for log-scale sample target " +
          std::to_string(target));
    }
    std::sort(feasible.begin(), feasible.end(), [&](std::size_t left,
                                                     std::size_t right) {
      if (scale_policy == "coverage-high" || scale_policy == "coverage-low") {
        const auto left_coverage = coverage(left);
        const auto right_coverage = coverage(right);
        if (left_coverage != right_coverage) {
          return scale_policy == "coverage-high"
              ? left_coverage > right_coverage
              : left_coverage < right_coverage;
        }
      }
      const auto left_size = variables[left].instances.size();
      const auto right_size = variables[right].instances.size();
      if (left_size != right_size) {
        return left_size > right_size;
      }
      return variables[left].id < variables[right].id;
    });

    const auto variable_index = feasible.front();
    auto& instances = variables[variable_index].instances;
    std::mt19937_64 rng(seed + 0x9e3779b97f4a7c15ULL * (variable_index + 1));
    if (instances.size() > target) {
      std::shuffle(instances.begin(), instances.end(), rng);
      instances.resize(target);
      std::sort(instances.begin(), instances.end(), [](const auto& left, const auto& right) {
        return left.id < right.id;
      });
    }
    remaining.erase(std::find(remaining.begin(), remaining.end(), variable_index));
  }
}


struct ParentFactorSpec {
  std::string id;
  std::string child;
  std::vector<std::string> parents;
  std::vector<std::string> scope;
};

struct SourceOutputSpec {
  std::vector<std::string> source_scope;
  std::vector<std::vector<std::string>> output_scopes;
};

struct ParentClusterSpec {
  std::string id;
  std::vector<std::string> scope;
  std::vector<ParentFactorSpec> factors;
  std::vector<std::vector<std::string>> target_scopes;
  std::vector<SourceOutputSpec> source_outputs;
};

struct AdjustmentQuerySpec {
  std::vector<std::string> treatments;
  std::string outcome;
  std::vector<std::string> adjustment;
  std::vector<std::string> required_scope;
};

struct ParentWorkloadSpec {
  std::vector<ParentFactorSpec> factors;
  std::vector<ParentClusterSpec> clusters;
  std::vector<std::vector<std::string>> target_scopes;
  std::vector<SourceOutputSpec> source_outputs;
  std::vector<AdjustmentQuerySpec> adjustment_queries;
  std::size_t skipped_factors = 0;
  std::size_t oversize_atomic_factors = 0;
  std::size_t adjustment_query_candidates = 0;
  std::size_t multi_treatment_queries = 0;
  std::size_t relevant_adjustment_queries = 0;
  std::size_t unadjusted_queries = 0;
  std::size_t adjusted_queries = 0;
  std::size_t trivial_queries_pruned = 0;
  std::size_t adjustment_unavailable_queries = 0;
  std::size_t adjustment_scopes_not_parent_covered = 0;
  std::size_t query_clusters_created = 0;
  std::size_t tree_lattice_scope_candidates = 0;
  std::size_t tree_lattice_scopes_retained = 0;
  std::size_t tree_lattice_scopes_pruned = 0;
};

std::vector<std::string> sortedUniqueStrings(std::vector<std::string> values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

std::vector<std::string> setUnionStrings(const std::vector<std::string>& left,
                                         const std::vector<std::string>& right) {
  std::vector<std::string> out;
  std::set_union(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(out));
  return out;
}

std::size_t intersectionSizeStrings(const std::vector<std::string>& left,
                                    const std::vector<std::string>& right) {
  std::size_t count = 0;
  auto l = left.begin();
  auto r = right.begin();
  while (l != left.end() && r != right.end()) {
    if (*l < *r) {
      ++l;
    } else if (*r < *l) {
      ++r;
    } else {
      ++count;
      ++l;
      ++r;
    }
  }
  return count;
}

bool includesStrings(const std::vector<std::string>& superset,
                     const std::vector<std::string>& subset) {
  return std::includes(superset.begin(), superset.end(), subset.begin(), subset.end());
}

std::string scopeSignature(const std::vector<std::string>& scope) {
  std::string out;
  for (const auto& item : scope) {
    out += item;
    out += '\x1f';
  }
  return out;
}

std::string defaultCausalPathForAssignments(const std::string& assignments_path) {
  const auto slash = assignments_path.find_last_of('/');
  if (slash == std::string::npos) {
    return "causal.tsv";
  }
  return assignments_path.substr(0, slash + 1) + "causal.tsv";
}

std::string snapshotFileForAssignments(const std::string& assignments_path,
                                       const std::string& filename) {
  const auto slash = assignments_path.find_last_of('/');
  if (slash == std::string::npos) {
    return filename;
  }
  return assignments_path.substr(0, slash + 1) + filename;
}

std::string roleLabel(const std::string& role) {
  const auto hash = role.find('#');
  return hash == std::string::npos ? role : role.substr(0, hash);
}

std::unordered_map<std::string, std::vector<std::string>> loadPartitionAttributes(
    const std::string& path,
    const std::set<std::string>& selected_variables) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open pattern schema: " + path);
  }
  std::map<std::string, std::set<std::string>> attributes;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto cols = split(line, '\t');
    if (cols.size() <= 8 || cols[0] != "pattern" ||
        selected_variables.count(cols[2]) == 0 || cols[7].empty() || cols[8].empty()) {
      continue;
    }
    attributes[roleLabel(cols[7])].insert(cols[8]);
  }
  std::unordered_map<std::string, std::vector<std::string>> out;
  for (const auto& item : attributes) {
    out.emplace(item.first, std::vector<std::string>(item.second.begin(), item.second.end()));
  }
  return out;
}

std::shared_ptr<const NodeAttributeStore> loadNodeAttributes(
    const std::string& path,
    const std::unordered_map<std::string, std::vector<std::string>>& attributes_by_label,
    std::size_t& loaded_values) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open PG nodes: " + path);
  }
  auto out = std::make_shared<NodeAttributeStore>();
  loaded_values = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto cols = split(line, '\t');
    if (cols.size() < 3) {
      continue;
    }
    const auto wanted = attributes_by_label.find(cols[1]);
    if (wanted == attributes_by_label.end()) {
      continue;
    }
    const std::set<std::string> wanted_set(wanted->second.begin(), wanted->second.end());
    const auto values = split(cols[2], ',');
    NodeAttributeMap selected;
    for (std::size_t i = 0; i + 1 < values.size(); i += 2) {
      if (wanted_set.count(values[i]) != 0) {
        selected[values[i]] = values[i + 1];
      }
    }
    if (!selected.empty()) {
      loaded_values += selected.size();
      out->emplace(cols[0], std::move(selected));
    }
  }
  return out;
}

std::vector<ParentFactorSpec> loadAtomicParentFactors(const std::string& causal_path,
                                                       const std::set<std::string>& selected,
                                                       std::size_t max_cluster_size,
                                                       std::size_t& skipped,
                                                       std::size_t& oversize) {
  std::ifstream in(causal_path);
  if (!in) {
    throw std::runtime_error("failed to open causal file: " + causal_path);
  }
  std::vector<ParentFactorSpec> factors;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto cols = split(line, '\t');
    if (cols.empty()) {
      continue;
    }
    const std::string child = cols[0];
    std::vector<std::string> parents;
    std::vector<std::string> scope{child};
    if (cols.size() > 1 && !cols[1].empty()) {
      for (const auto& parent : split(cols[1], ',')) {
        if (!parent.empty()) {
          parents.push_back(parent);
          scope.push_back(parent);
        }
      }
    }
    parents = sortedUniqueStrings(std::move(parents));
    scope = sortedUniqueStrings(std::move(scope));
    bool usable = !scope.empty();
    for (const auto& variable : scope) {
      if (selected.count(variable) == 0) {
        usable = false;
        break;
      }
    }
    if (!usable) {
      skipped += 1;
      continue;
    }
    if (scope.size() > max_cluster_size) {
      oversize += 1;
    }
    factors.push_back({"F_" + child, child, std::move(parents), std::move(scope)});
  }
  return factors;
}

std::vector<ParentClusterSpec> mergeParentFactors(const std::vector<ParentFactorSpec>& factors,
                                                  std::size_t max_cluster_size) {
  struct WorkingCluster {
    std::vector<std::string> scope;
    std::vector<ParentFactorSpec> factors;
  };
  struct Candidate {
    std::size_t left = 0;
    std::size_t right = 0;
    std::vector<std::string> union_scope;
    bool subset_merge = false;
    std::size_t covered_count = 0;
    std::size_t overlap_size = 0;
    std::size_t added_size = 0;
  };
  auto better = [](const Candidate& candidate, const Candidate& best) {
    return std::make_tuple(candidate.subset_merge,
                           candidate.covered_count,
                           candidate.overlap_size,
                           -static_cast<long long>(candidate.union_scope.size()),
                           -static_cast<long long>(candidate.added_size),
                           -static_cast<long long>(candidate.left),
                           -static_cast<long long>(candidate.right)) >
           std::make_tuple(best.subset_merge,
                           best.covered_count,
                           best.overlap_size,
                           -static_cast<long long>(best.union_scope.size()),
                           -static_cast<long long>(best.added_size),
                           -static_cast<long long>(best.left),
                           -static_cast<long long>(best.right));
  };

  std::vector<WorkingCluster> clusters;
  std::map<std::string, std::size_t> by_scope;
  for (const auto& factor : factors) {
    const auto key = scopeSignature(factor.scope);
    auto found = by_scope.find(key);
    if (found == by_scope.end()) {
      by_scope[key] = clusters.size();
      clusters.push_back({factor.scope, {factor}});
    } else {
      clusters[found->second].factors.push_back(factor);
    }
  }

  while (true) {
    bool has_candidate = false;
    Candidate best;
    for (std::size_t i = 0; i < clusters.size(); ++i) {
      for (std::size_t j = i + 1; j < clusters.size(); ++j) {
        auto union_scope = setUnionStrings(clusters[i].scope, clusters[j].scope);
        if (union_scope.size() > max_cluster_size) {
          continue;
        }
        Candidate candidate;
        candidate.left = i;
        candidate.right = j;
        candidate.union_scope = std::move(union_scope);
        candidate.subset_merge = includesStrings(clusters[i].scope, clusters[j].scope) ||
                                 includesStrings(clusters[j].scope, clusters[i].scope);
        candidate.covered_count = clusters[i].factors.size() + clusters[j].factors.size();
        candidate.overlap_size = intersectionSizeStrings(clusters[i].scope, clusters[j].scope);
        candidate.added_size = candidate.union_scope.size() -
                               std::max(clusters[i].scope.size(), clusters[j].scope.size());
        if (!has_candidate || better(candidate, best)) {
          best = std::move(candidate);
          has_candidate = true;
        }
      }
    }
    if (!has_candidate) {
      break;
    }
    WorkingCluster merged;
    merged.scope = std::move(best.union_scope);
    merged.factors = std::move(clusters[best.left].factors);
    merged.factors.insert(merged.factors.end(), clusters[best.right].factors.begin(),
                          clusters[best.right].factors.end());
    clusters[best.left] = std::move(merged);
    clusters.erase(clusters.begin() + static_cast<std::ptrdiff_t>(best.right));
  }

  std::sort(clusters.begin(), clusters.end(), [](const auto& left, const auto& right) {
    if (left.scope != right.scope) {
      return left.scope < right.scope;
    }
    return left.factors.size() < right.factors.size();
  });

  std::vector<ParentClusterSpec> out;
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    std::sort(clusters[i].factors.begin(), clusters[i].factors.end(), [](const auto& left, const auto& right) {
      return left.id < right.id;
    });
    out.push_back({"mpc" + std::to_string(i), std::move(clusters[i].scope),
                   std::move(clusters[i].factors), {}, {}});
  }
  return out;
}

CausalDag loadCausalDag(const std::string& causal_path,
                        const std::vector<std::string>& selected) {
  const std::set<std::string> selected_set(selected.begin(), selected.end());
  std::ifstream in(causal_path);
  if (!in) {
    throw std::runtime_error("failed to open causal file: " + causal_path);
  }
  std::vector<std::pair<std::string, std::string>> edges;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto cols = split(line, '\t');
    if (cols.empty()) {
      continue;
    }
    const std::string& child = cols[0];
    if (selected_set.count(child) == 0 || cols.size() < 2 || cols[1].empty()) {
      continue;
    }
    for (const auto& parent_name : split(cols[1], ',')) {
      if (selected_set.count(parent_name) != 0) {
        edges.emplace_back(parent_name, child);
      }
    }
  }
  return CausalDag(selected, edges);
}

ParentWorkloadSpec buildParentWorkload(const std::string& causal_path,
                                       const std::vector<std::string>& selected,
                                       std::size_t max_cluster_size,
                                       std::size_t max_tree_depth,
                                       bool enable_adjustment_pruning) {
  (void)max_tree_depth;
  ParentWorkloadSpec out;
  std::set<std::string> selected_set(selected.begin(), selected.end());
  out.factors = loadAtomicParentFactors(causal_path, selected_set, max_cluster_size,
                                        out.skipped_factors, out.oversize_atomic_factors);
  out.clusters = mergeParentFactors(out.factors, max_cluster_size);

  std::set<std::string> emitted_outputs;
  std::map<std::string, std::map<std::string, std::size_t>>
      source_output_positions;
  auto add_source_output = [&](ParentClusterSpec &cluster,
                               const std::vector<std::string> &source,
                               const std::vector<std::string> &output) {
    if (source.empty() || output.empty()) {
      return false;
    }
    const auto output_key = scopeSignature(output);
    if (emitted_outputs.count(output_key) != 0) {
      return false;
    }
    emitted_outputs.insert(output_key);
    const auto source_key = scopeSignature(source);
    auto &positions = source_output_positions[cluster.id];
    const auto found = positions.find(source_key);
    if (found == positions.end()) {
      const auto position = cluster.source_outputs.size();
      positions.emplace(source_key, position);
      cluster.source_outputs.push_back({source, {output}});
    } else {
      cluster.source_outputs[found->second].output_scopes.push_back(output);
    }
    cluster.target_scopes.push_back(output);
    out.target_scopes.push_back(output);
    return true;
  };

  auto find_containing_cluster =
      [&](const std::vector<std::string>& query_scope) -> ParentClusterSpec* {
    std::size_t best_index = out.clusters.size();
    std::pair<std::size_t, std::string> best_score;
    for (std::size_t i = 0; i < out.clusters.size(); ++i) {
      const auto union_scope = setUnionStrings(out.clusters[i].scope, query_scope);
      if (union_scope.size() != out.clusters[i].scope.size()) {
        continue;
      }
      const auto score =
          std::make_pair(out.clusters[i].scope.size(), out.clusters[i].id);
      if (best_index == out.clusters.size() || score < best_score) {
        best_index = i;
        best_score = score;
      }
    }
    if (best_index == out.clusters.size()) {
      return nullptr;
    }
    return &out.clusters[best_index];
  };

  // Freeze the merged parent-tree scopes before query scopes are routed back
  // into clusters. Workload generation is based on the parent decomposition,
  // not on clusters enlarged by earlier queries.
  std::vector<std::vector<std::string>> parent_tree_scopes;
  for (const auto& cluster : out.clusters) {
    parent_tree_scopes.push_back(cluster.scope);
  }

  // Atomic parent factors and singleton marginals are the non-prunable base of
  // the causal index.
  for (auto& cluster : out.clusters) {
    for (const auto& factor : cluster.factors) {
      add_source_output(cluster, factor.scope, factor.scope);
    }
  }
  for (const auto& variable : selected) {
    std::vector<std::string> singleton{variable};
    if (emitted_outputs.count(scopeSignature(singleton)) == 0) {
      auto* cluster = find_containing_cluster(singleton);
      if (cluster == nullptr) {
        throw std::logic_error(
            "parent decomposition does not cover singleton scope");
      }
      add_source_output(*cluster, singleton, singleton);
    }
  }

  if (enable_adjustment_pruning) {
    const auto dag = loadCausalDag(causal_path, selected);
    std::set<std::pair<CausalMask, CausalMask>> seen_queries;
    std::map<std::string, std::vector<std::string>> required_query_scopes;

    // Enumerate do(T)->Y over each parent tree. T is any non-empty set and Y is
    // one outcome outside T. This includes pair queries as the |T|=1 case and
    // multi-variable interventions without turning them into independent pairs.
    for (std::size_t tree_index = 0; tree_index < parent_tree_scopes.size();
         ++tree_index) {
      const auto &tree_scope = parent_tree_scopes[tree_index];
      const CausalMask tree_mask = dag.maskOf(tree_scope);
      CausalMask outcome_bits = tree_mask;
      while (outcome_bits != 0) {
        const CausalMask outcome = outcome_bits & (~outcome_bits + 1);
        outcome_bits ^= outcome;
        const CausalMask available_treatments = tree_mask & ~outcome;
        for (CausalMask treatments = available_treatments; treatments != 0;
             treatments = (treatments - 1) & available_treatments) {
          if (!seen_queries.emplace(treatments, outcome).second) {
            continue;
          }
          out.adjustment_query_candidates += 1;
          if (__builtin_popcountll(treatments) > 1) {
            out.multi_treatment_queries += 1;
          }

          const auto analysis =
              dag.analyzeAdjustmentWithin(treatments, outcome, tree_mask);
          if (!analysis.has_proper_causal_path &&
              !analysis.observationally_connected) {
            out.trivial_queries_pruned += 1;
            continue;
          }
          if (!analysis.adjustment_exists) {
            if (analysis.has_proper_causal_path) {
              out.adjustment_unavailable_queries += 1;
            } else {
              out.trivial_queries_pruned += 1;
            }
            continue;
          }

          // A d-connected sibling query is retained with its non-empty
          // adjustment set. A structurally disconnected no-effect query was
          // already removed above.
          out.relevant_adjustment_queries += 1;
          if (analysis.adjustment == 0) {
            out.unadjusted_queries += 1;
          } else {
            out.adjusted_queries += 1;
          }
          const CausalMask required_mask =
              treatments | outcome | analysis.adjustment;
          auto required_scope = dag.variablesOf(required_mask);
          if ((required_mask & ~tree_mask) != 0 ||
              find_containing_cluster(required_scope) == nullptr) {
            out.adjustment_scopes_not_parent_covered += 1;
            continue;
          }
          required_query_scopes.emplace(scopeSignature(required_scope),
                                        required_scope);
          out.adjustment_queries.push_back(
              {dag.variablesOf(treatments), dag.variablesOf(outcome).front(),
               dag.variablesOf(analysis.adjustment),
               std::move(required_scope)});
        }
      }
    }

    // Pure pruning only keeps scopes already present in the original parent
    // decomposition. It never enlarges a parent cluster or creates a query-only
    // cluster; the unpruned plan therefore remains valid after filtering.
    std::vector<std::vector<std::string>> ordered_required_scopes;
    for (const auto &item : required_query_scopes) {
      ordered_required_scopes.push_back(item.second);
    }
    std::sort(ordered_required_scopes.begin(), ordered_required_scopes.end(),
              [](const auto &left, const auto &right) {
                if (left.size() != right.size()) {
                  return left.size() > right.size();
                }
                return left < right;
              });
    for (const auto &query_scope : ordered_required_scopes) {
      if (emitted_outputs.count(scopeSignature(query_scope)) == 0) {
        auto* cluster = find_containing_cluster(query_scope);
        if (cluster == nullptr) {
          throw std::logic_error(
              "covered adjustment scope lost its parent cluster");
        }
        add_source_output(*cluster, query_scope, query_scope);
      }
    }
  }

  // Report pruning against the complete lattice local to each original parent
  // tree. Prefix-closure tasks are generated later by FullLatticeIndex and are
  // not counted as materialized query outputs here.
  for (std::size_t tree_index = 0; tree_index < parent_tree_scopes.size(); ++tree_index) {
    const auto& tree_scope = parent_tree_scopes[tree_index];
    if (tree_scope.size() >= sizeof(CausalMask) * 8) {
      throw std::runtime_error("parent cluster is too wide to enumerate as a lattice");
    }
    const CausalMask all = (CausalMask{1} << tree_scope.size()) - 1;
    for (CausalMask local = 1; local <= all; ++local) {
      out.tree_lattice_scope_candidates += 1;
      std::vector<std::string> scope;
      for (std::size_t i = 0; i < tree_scope.size(); ++i) {
        if ((local & (CausalMask{1} << i)) != 0) {
          scope.push_back(tree_scope[i]);
        }
      }
      if (!enable_adjustment_pruning &&
          emitted_outputs.count(scopeSignature(scope)) == 0) {
        add_source_output(out.clusters[tree_index], scope, scope);
      }
      if (emitted_outputs.count(scopeSignature(scope)) != 0) {
        out.tree_lattice_scopes_retained += 1;
      } else {
        out.tree_lattice_scopes_pruned += 1;
      }
    }
  }

  std::sort(out.target_scopes.begin(), out.target_scopes.end());
  out.target_scopes.erase(std::unique(out.target_scopes.begin(), out.target_scopes.end()),
                          out.target_scopes.end());
  out.source_outputs.clear();
  for (const auto& cluster : out.clusters) {
    out.source_outputs.insert(out.source_outputs.end(), cluster.source_outputs.begin(), cluster.source_outputs.end());
  }
  for (auto& cluster : out.clusters) {
    std::sort(cluster.target_scopes.begin(), cluster.target_scopes.end());
    cluster.target_scopes.erase(std::unique(cluster.target_scopes.begin(), cluster.target_scopes.end()),
                                cluster.target_scopes.end());
    std::sort(cluster.source_outputs.begin(), cluster.source_outputs.end(), [](const auto& left, const auto& right) {
      return left.source_scope < right.source_scope;
    });
    for (auto& item : cluster.source_outputs) {
      std::sort(item.output_scopes.begin(), item.output_scopes.end());
      item.output_scopes.erase(std::unique(item.output_scopes.begin(), item.output_scopes.end()),
                               item.output_scopes.end());
      for (const auto& output_scope : item.output_scopes) {
        if (item.source_scope != output_scope) {
          throw std::logic_error(
              "causal parent workload must not project a superset join into a smaller output scope");
        }
      }
    }
  }
  return out;
}

std::vector<Mask> masksForScopes(const std::vector<VariableInput>& variables,
                                 const std::vector<std::vector<std::string>>& scopes) {
  std::map<std::string, std::size_t> index;
  for (std::size_t i = 0; i < variables.size(); ++i) {
    index[variables[i].id] = i;
  }
  std::set<Mask> masks;
  for (const auto& scope : scopes) {
    Mask mask = 0;
    for (const auto& variable : scope) {
      auto found = index.find(variable);
      if (found == index.end()) {
        throw std::runtime_error("scope contains unknown selected variable: " + variable);
      }
      mask |= Mask{1} << found->second;
    }
    if (mask != 0) {
      masks.insert(mask);
    }
  }
  return std::vector<Mask>(masks.begin(), masks.end());
}

std::vector<SourceOutputScope> toSourceOutputScopes(const std::vector<SourceOutputSpec>& specs) {
  std::vector<SourceOutputScope> out;
  out.reserve(specs.size());
  for (const auto& spec : specs) {
    out.push_back({spec.source_scope, spec.output_scopes});
  }
  return out;
}

Mask maskForScope(const std::vector<VariableInput>& variables, const std::vector<std::string>& scope) {
  std::map<std::string, std::size_t> index;
  for (std::size_t i = 0; i < variables.size(); ++i) {
    index[variables[i].id] = i;
  }
  Mask mask = 0;
  for (const auto& variable : scope) {
    auto found = index.find(variable);
    if (found == index.end()) {
      throw std::runtime_error("scope contains unknown selected variable: " + variable);
    }
    mask |= Mask{1} << found->second;
  }
  return mask;
}

std::map<Mask, std::vector<Mask>> outputMasksBySource(
    const std::vector<VariableInput>& variables,
    const std::vector<SourceOutputSpec>& specs) {
  std::map<Mask, std::vector<Mask>> out;
  for (const auto& spec : specs) {
    const Mask source = maskForScope(variables, spec.source_scope);
    auto& outputs = out[source];
    const auto& output_scopes = spec.output_scopes.empty()
        ? std::vector<std::vector<std::string>>{spec.source_scope}
        : spec.output_scopes;
    for (const auto& output_scope : output_scopes) {
      const Mask output = maskForScope(variables, output_scope);
      if ((output & ~source) != 0) {
        throw std::runtime_error("output scope must be subset of source scope");
      }
      outputs.push_back(output);
    }
  }
  for (auto& item : out) {
    std::sort(item.second.begin(), item.second.end());
    item.second.erase(std::unique(item.second.begin(), item.second.end()), item.second.end());
  }
  return out;
}

std::vector<NormalVariable> toNormalVariables(const std::vector<VariableInput>& variables) {
  std::vector<NormalVariable> out;
  for (const auto& variable : variables) {
    NormalVariable normal;
    normal.id = variable.id;
    normal.roles = variable.pattern_roles;
    for (const auto& instance : variable.instances) {
      NormalInstance row;
      row.id = instance.id;
      row.value = instance.value.empty() ? instance.bucket : instance.value;
      row.roles = instance.roles;
      for (const auto& binding : instance.roles) {
        row.role_to_node[binding.role] = binding.node;
      }
      normal.instances.push_back(std::move(row));
    }
    out.push_back(std::move(normal));
  }
  return out;
}

std::vector<VariableInput> variablesForScope(
    const std::vector<VariableInput>& variables,
    const std::vector<std::string>& scope) {
  const std::set<std::string> requested(scope.begin(), scope.end());
  std::vector<VariableInput> out;
  for (const auto& variable : variables) {
    if (requested.count(variable.id) != 0) {
      out.push_back(variable);
    }
  }
  if (out.size() != requested.size()) {
    throw std::runtime_error("cluster scope contains an unknown selected variable");
  }
  return out;
}

Mask translateMask(const std::vector<VariableInput>& local_variables,
                   const std::vector<VariableInput>& global_variables,
                   Mask local_mask) {
  std::vector<std::string> scope;
  for (std::size_t i = 0; i < local_variables.size(); ++i) {
    if ((local_mask & (Mask{1} << i)) != 0) {
      scope.push_back(local_variables[i].id);
    }
  }
  return maskForScope(global_variables, scope);
}

bool compatible(const NormalRow& row, const NormalInstance& instance) {
  for (const auto& binding : instance.roles) {
    auto found = row.role_to_node.find(binding.role);
    if (found != row.role_to_node.end()) {
      if (found->second != binding.node) {
        return false;
      }
      continue;
    }
    if (row.used_nodes.count(binding.node) != 0) {
      return false;
    }
  }
  return true;
}

NormalRow mergeRow(const NormalRow& row, std::size_t variable, const NormalInstance& instance) {
  NormalRow out = row;
  out.instance_ids[variable] = instance.id;
  out.values[variable] = instance.value;
  for (const auto& binding : instance.roles) {
    out.role_to_node.emplace(binding.role, binding.node);
    out.used_nodes.insert(binding.node);
  }
  return out;
}

struct NormalJoinResult {
  Count rows = 0;
  ValueCountMap value_counts;
};

ValueKey valueKeyForRow(const NormalRow& row, Mask mask) {
  ValueKey key;
  for (std::size_t v = 0; v < row.values.size(); ++v) {
    if ((mask & (Mask{1} << v)) != 0) {
      key.push_back(row.values[v]);
    }
  }
  return key;
}

std::size_t valueTrieNodeCount(const ValueCountMap& counts) {
  std::set<ValueKey> prefixes;
  prefixes.insert({});
  for (const auto& item : counts) {
    ValueKey prefix;
    for (const auto& value : item.first) {
      prefix.push_back(value);
      prefixes.insert(prefix);
    }
  }
  return prefixes.size();
}

NormalJoinResult normalIndependentJoinForMask(const std::vector<NormalVariable>& variables, Mask mask) {
  const std::size_t n = variables.size();
  std::vector<NormalRow> rows(1);
  rows.front().instance_ids.assign(n, kNoInstance);
  rows.front().values.assign(n, "__missing__");
  for (std::size_t v = 0; v < n; ++v) {
    if ((mask & (Mask{1} << v)) == 0) {
      continue;
    }
    std::vector<NormalRow> next;
    for (const auto& row : rows) {
      for (const auto& instance : variables[v].instances) {
        if (compatible(row, instance)) {
          next.push_back(mergeRow(row, v, instance));
        }
      }
    }
    rows = std::move(next);
    if (rows.empty()) {
      break;
    }
  }

  NormalJoinResult result;
  result.rows = rows.size();
  for (const auto& row : rows) {
    result.value_counts[valueKeyForRow(row, mask)] += 1;
  }
  return result;
}

std::vector<std::size_t> instanceCountOrderForMask(
    const std::vector<NormalVariable>& variables,
    Mask mask) {
  std::vector<std::size_t> order;
  for (std::size_t variable = 0; variable < variables.size(); ++variable) {
    if ((mask & (Mask{1} << variable)) != 0) {
      order.push_back(variable);
    }
  }
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    const auto left_size = variables[left].instances.size();
    const auto right_size = variables[right].instances.size();
    if (left_size != right_size) {
      return left_size < right_size;
    }
    return variables[left].id < variables[right].id;
  });
  return order;
}

std::map<Mask, NormalJoinResult> normalIndependentJoinForSourceOutputs(
    const std::vector<NormalVariable>& variables,
    Mask source_mask,
    const std::vector<Mask>& output_masks,
    bool instance_count_greedy = false,
    const std::vector<std::size_t>* fixed_order = nullptr) {
  const std::size_t n = variables.size();
  std::vector<NormalRow> rows(1);
  rows.front().instance_ids.assign(n, kNoInstance);
  rows.front().values.assign(n, "__missing__");
  std::vector<std::size_t> order;
  if (instance_count_greedy) {
    order = instanceCountOrderForMask(variables, source_mask);
  } else if (fixed_order != nullptr) {
    for (const auto variable : *fixed_order) {
      if ((source_mask & (Mask{1} << variable)) != 0) {
        order.push_back(variable);
      }
    }
  } else {
    for (std::size_t v = 0; v < n; ++v) {
      if ((source_mask & (Mask{1} << v)) != 0) {
        order.push_back(v);
      }
    }
  }
  for (const auto v : order) {
    std::vector<NormalRow> next;
    for (const auto& row : rows) {
      for (const auto& instance : variables[v].instances) {
        if (compatible(row, instance)) {
          next.push_back(mergeRow(row, v, instance));
        }
      }
    }
    rows = std::move(next);
    if (rows.empty()) {
      break;
    }
  }

  std::map<Mask, NormalJoinResult> out;
  for (const auto output_mask : output_masks) {
    if ((output_mask & ~source_mask) != 0) {
      throw std::runtime_error("baseline output mask must be subset of source mask");
    }
    auto& result = out[output_mask];
    result.rows = rows.size();
    for (const auto& row : rows) {
      result.value_counts[valueKeyForRow(row, output_mask)] += 1;
    }
  }
  return out;
}


std::vector<Count> normalIndependentJoinCountsForMasks(
    const std::vector<NormalVariable>& variables,
    const std::vector<Mask>& masks,
    std::size_t worker_threads,
    std::size_t progress_every = 0,
    const std::vector<VariableInput>* display_variables = nullptr,
    const std::string& progress_label = "",
    std::vector<ValueCountMap>* value_counts = nullptr,
    std::vector<std::size_t>* value_trie_nodes = nullptr) {
  const std::size_t n = variables.size();
  const Mask all = (Mask{1} << n) - 1;
  std::vector<Count> counts(static_cast<std::size_t>(all + 1), 0);
  if (value_counts != nullptr) {
    value_counts->assign(static_cast<std::size_t>(all + 1), {});
  }
  if (value_trie_nodes != nullptr) {
    value_trie_nodes->assign(static_cast<std::size_t>(all + 1), 0);
  }
  if (masks.empty()) {
    return counts;
  }
  const std::size_t requested_threads = std::max<std::size_t>(1, worker_threads);
  const std::size_t worker_count = std::min<std::size_t>(requested_threads, masks.size());

  std::size_t completed = 0;
  std::mutex progress_mutex;
  auto describe_mask = [&](Mask mask) {
    std::string out;
    if (display_variables == nullptr) {
      return out;
    }
    bool first = true;
    for (std::size_t v = 0; v < display_variables->size(); ++v) {
      if ((mask & (Mask{1} << v)) == 0) {
        continue;
      }
      if (!first) {
        out += ',';
      }
      first = false;
      out += (*display_variables)[v].id;
    }
    return out;
  };
  auto report_baseline_progress = [&](Mask mask, Count rows) {
    if (progress_every == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(progress_mutex);
    completed += 1;
    const bool should_report = completed == 1 || completed == masks.size() ||
        completed % progress_every == 0;
    if (!should_report) {
      return;
    }
    const double pct = masks.empty() ? 100.0
        : 100.0 * static_cast<double>(completed) / static_cast<double>(masks.size());
    const std::time_t now = std::time(nullptr);
    std::cout << "baseline_progress"
              << "\ttime=" << now;
    if (!progress_label.empty()) {
      std::cout << "\tcluster=" << progress_label;
    }
    std::cout << "\tcompleted=" << completed << '/' << masks.size()
              << "\tpct=" << pct
              << "\tmask=" << mask
              << "\trows=" << rows;
    const auto vars = describe_mask(mask);
    if (!vars.empty()) {
      std::cout << "\tvars=" << vars;
    }
    std::cout << '\n';
    std::cout.flush();
  };

  auto run_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const Mask mask = masks[i];
      const auto joined = normalIndependentJoinForMask(variables, mask);
      counts[static_cast<std::size_t>(mask)] = joined.rows;
      if (value_counts != nullptr) {
        (*value_counts)[static_cast<std::size_t>(mask)] = joined.value_counts;
      }
      if (value_trie_nodes != nullptr) {
        (*value_trie_nodes)[static_cast<std::size_t>(mask)] = valueTrieNodeCount(joined.value_counts);
      }
      report_baseline_progress(mask, joined.rows);
    }
  };

  if (worker_count <= 1) {
    run_range(0, masks.size());
    return counts;
  }

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    const std::size_t begin = masks.size() * worker / worker_count;
    const std::size_t end = masks.size() * (worker + 1) / worker_count;
    workers.emplace_back([&, begin, end] { run_range(begin, end); });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  return counts;
}

std::vector<Count> normalIndependentJoinCounts(const std::vector<NormalVariable>& variables,
                                               std::size_t worker_threads) {
  const std::size_t n = variables.size();
  const Mask all = (Mask{1} << n) - 1;
  std::vector<Mask> masks;
  masks.reserve(static_cast<std::size_t>(all));
  for (Mask mask = 1; mask <= all; ++mask) {
    masks.push_back(mask);
  }
  return normalIndependentJoinCountsForMasks(variables, masks, worker_threads);
}

template <typename Fn>
double timeMillis(Fn&& fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1]
      : "experiments/ldbc_snb_sf003_person_matchsupport50_dryrun/index_query_only_snapshot/assignments_raw.tsv";
  const std::size_t n = argc > 2 ? static_cast<std::size_t>(std::stoul(argv[2])) : 4;
  const std::string mode = argc > 3 ? argv[3] : "compare";
  const std::size_t derive_budget = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 0;
  const std::size_t worker_threads = argc > 5 ? static_cast<std::size_t>(std::stoull(argv[5])) : 1;
  const std::size_t progress_every = argc > 6 ? static_cast<std::size_t>(std::stoull(argv[6])) : 0;
  const std::size_t parent_k = argc > 7 ? static_cast<std::size_t>(std::stoull(argv[7])) : 8;
  auto is_number = [](const std::string& value) {
    return !value.empty() &&
        std::all_of(value.begin(), value.end(), [](unsigned char c) {
          return std::isdigit(c) != 0;
        });
  };
  std::size_t parent_max_tree_depth = parent_k;
  std::size_t min_instances_per_variable = 0;
  std::size_t max_instances_per_variable = 0;
  bool log_scale_sample = false;
  std::size_t log_scale_min_exponent = 0;
  std::size_t log_scale_max_exponent = 0;
  std::string log_scale_policy = "capacity";
  constexpr std::uint64_t sample_seed = 20260716ULL;
  std::size_t next_arg = 8;
  if (argc > static_cast<int>(next_arg)) {
    const std::string maybe_depth = argv[next_arg];
    if (is_number(maybe_depth)) {
      parent_max_tree_depth = static_cast<std::size_t>(std::stoull(maybe_depth));
      next_arg += 1;
    }
  }
  if (argc > static_cast<int>(next_arg)) {
    const std::string maybe_sample = argv[next_arg];
    if (maybe_sample.rfind("log10:", 0) == 0) {
      const auto fields = split(maybe_sample, ':');
      if ((fields.size() != 3 && fields.size() != 4) ||
          !is_number(fields[1]) || !is_number(fields[2])) {
        throw std::runtime_error(
            "log-scale sample must use log10:min_exp:max_exp[:scale_policy]");
      }
      log_scale_sample = true;
      log_scale_min_exponent = static_cast<std::size_t>(std::stoull(fields[1]));
      log_scale_max_exponent = static_cast<std::size_t>(std::stoull(fields[2]));
      if (fields.size() == 4) {
        log_scale_policy = fields[3];
      }
      if (log_scale_policy != "capacity" &&
          log_scale_policy != "coverage-high" &&
          log_scale_policy != "coverage-low") {
        throw std::runtime_error(
            "log-scale policy must be capacity, coverage-high, or coverage-low");
      }
      next_arg += 1;
    } else if (is_number(maybe_sample)) {
      max_instances_per_variable = static_cast<std::size_t>(std::stoull(maybe_sample));
      min_instances_per_variable = max_instances_per_variable;
      next_arg += 1;
    } else {
      const auto delimiter = maybe_sample.find_first_of("-:");
      if (delimiter != std::string::npos &&
          is_number(maybe_sample.substr(0, delimiter)) &&
          is_number(maybe_sample.substr(delimiter + 1))) {
        min_instances_per_variable =
            static_cast<std::size_t>(std::stoull(maybe_sample.substr(0, delimiter)));
        max_instances_per_variable =
            static_cast<std::size_t>(std::stoull(maybe_sample.substr(delimiter + 1)));
        next_arg += 1;
      }
    }
  }
  std::string causal_path = defaultCausalPathForAssignments(path);
  if (argc > static_cast<int>(next_arg)) {
    causal_path = argv[next_arg++];
  }
  const bool enable_value_partition = argc > static_cast<int>(next_arg)
      ? std::string(argv[next_arg++]) != "0"
      : true;
  const std::string order_policy = argc > static_cast<int>(next_arg)
      ? std::string(argv[next_arg++])
      : "cost-model";
  if (order_policy != "alphabetical" && order_policy != "cost-model" &&
      order_policy != "random" &&
      order_policy != "scope-alphabetical" &&
      order_policy != "scope-random" &&
      order_policy != "instance-count" &&
      order_policy != "join-size-greedy" &&
      order_policy != "instance-count-cache" &&
      order_policy != "instance-count-independent" &&
      order_policy != "partial-dp" && order_policy != "partial-greedy" &&
      order_policy != "partial-greedy-cache" &&
      order_policy != "partial-greedy-independent") {
    throw std::runtime_error(
        "unsupported order/schedule policy");
  }
  const bool collect_partition_traces = argc > static_cast<int>(next_arg)
      ? std::string(argv[next_arg++]) != "0"
      : false;
  const bool enable_adjustment_pruning = argc > static_cast<int>(next_arg)
      ? std::string(argv[next_arg++]) != "0"
      : true;
  const bool plan_order_from_unpruned_lattice =
      argc > static_cast<int>(next_arg)
          ? std::string(argv[next_arg++]) != "0"
          : false;
  const std::size_t result_cache_capacity =
      argc > static_cast<int>(next_arg)
          ? static_cast<std::size_t>(std::stoull(argv[next_arg++]))
          : 0;
  ResultCacheEvictionPolicy result_cache_eviction_policy =
      ResultCacheEvictionPolicy::Belady;
  if (argc > static_cast<int>(next_arg)) {
    const std::string policy = argv[next_arg++];
    if (policy == "lru") {
      result_cache_eviction_policy = ResultCacheEvictionPolicy::Lru;
    } else if (policy != "belady") {
      throw std::runtime_error(
          "result cache eviction policy must be belady or lru");
    }
  }
  const bool sample_only = mode == "sample-only" || mode == "parent-sample-only";
  const bool parent_workload = mode.rfind("parent-", 0) == 0;
  const bool parent_greedy_baseline =
      mode == "parent-greedy-baseline-only" ||
      mode == "parent-greedy-compare";
  const bool run_baseline = !sample_only &&
      mode != "index-only" && mode != "parent-index-only";
  const bool baseline_only = mode == "baseline-only" ||
      mode == "parent-baseline-only" || mode == "parent-greedy-baseline-only";
  const bool run_index = !sample_only &&
      !baseline_only;

  auto parsed_order_policy = [&] {
    if (order_policy == "scope-alphabetical" ||
        order_policy == "scope-random") {
      return VariableOrderPolicy::PartialWorkloadGreedy;
    }
    if (order_policy == "alphabetical") {
      return VariableOrderPolicy::Alphabetical;
    }
    if (order_policy.rfind("instance-count", 0) == 0) {
      return VariableOrderPolicy::InstanceCountAscending;
    }
    if (order_policy == "random") {
      return VariableOrderPolicy::Random;
    }
    if (order_policy == "partial-dp") {
      return VariableOrderPolicy::PartialWorkloadDp;
    }
    if (order_policy.rfind("partial-greedy", 0) == 0) {
      return VariableOrderPolicy::PartialWorkloadGreedy;
    }
    return VariableOrderPolicy::CostModel;
  };
  auto parsed_scope_order_policy = [&] {
    if (order_policy == "scope-alphabetical") {
      return ScopeOrderPolicy::Alphabetical;
    }
    if (order_policy == "scope-random") {
      return ScopeOrderPolicy::Random;
    }
    return ScopeOrderPolicy::DfsLexicographic;
  };
  auto parsed_execution_schedule = [&] {
    if (result_cache_capacity > 0 &&
        (order_policy == "partial-greedy" ||
         order_policy == "scope-alphabetical" ||
         order_policy == "scope-random")) {
      return ExecutionSchedule::BoundedScopeCache;
    }
    if (order_policy == "scope-alphabetical" ||
        order_policy == "scope-random") {
      return ExecutionSchedule::ScopeOrderedWorkingPath;
    }
    if (order_policy == "join-size-greedy") {
      return ExecutionSchedule::JoinSizeGreedyWorkingPath;
    }
    if (order_policy.size() >= 6 &&
        order_policy.compare(order_policy.size() - 6, 6, "-cache") == 0) {
      return ExecutionSchedule::CacheAwareGreedy;
    }
    if (order_policy.size() >= 12 &&
        order_policy.compare(order_policy.size() - 12, 12, "-independent") == 0) {
      return ExecutionSchedule::IndependentSources;
    }
    return ExecutionSchedule::DfsPrefixTree;
  };

  std::map<std::string, std::size_t> counts;
  const double count_ms = timeMillis([&] { counts = countVariables(path); });
  auto selected = chooseSmallestVariables(counts, n);
  auto variables = loadSelectedVariables(path, selected);
  std::sort(variables.begin(), variables.end(), [](const auto& left, const auto& right) {
    return left.id < right.id;
  });
  selected.clear();
  selected.reserve(variables.size());
  for (const auto& variable : variables) selected.push_back(variable.id);

  ParentWorkloadSpec sampling_parent_work;
  ParentWorkloadSpec parent_work;
  double parent_workload_plan_ms = 0.0;
  if (parent_workload) {
    parent_workload_plan_ms = timeMillis([&] {
      if (log_scale_sample) {
      // Keep runtime log-scale sampling comparable between pruning variants by
      // deriving coverage from the same adjustment-aware workload.
      sampling_parent_work = buildParentWorkload(
          causal_path, selected, parent_k, parent_max_tree_depth, true);
    }
      if (enable_adjustment_pruning && log_scale_sample) {
      parent_work = sampling_parent_work;
    } else {
      parent_work = buildParentWorkload(causal_path, selected, parent_k,
                                        parent_max_tree_depth,
                                        enable_adjustment_pruning);
    }
      if (!log_scale_sample) {
      // Pre-materialized assignments need no second sampling workload.
      sampling_parent_work = parent_work;
    }
    });
  }
  std::map<std::string, std::size_t> workload_coverage;
  std::set<std::vector<std::string>> unique_source_scopes;
  for (const auto& cluster : sampling_parent_work.clusters) {
    for (const auto& source_output : cluster.source_outputs) {
      unique_source_scopes.insert(source_output.source_scope);
    }
  }
  for (const auto& source : unique_source_scopes) {
    for (const auto& variable : source) {
      workload_coverage[variable] += 1;
    }
  }
  if (log_scale_sample) {
    sampleInstancesLogScale(variables, log_scale_min_exponent,
                            log_scale_max_exponent, sample_seed,
                            log_scale_policy, workload_coverage);
  } else {
    sampleInstancesRandom(variables, min_instances_per_variable,
                          max_instances_per_variable, sample_seed);
  }
  std::vector<NormalVariable> normal_variables;
  if (!parent_workload) {
    normal_variables = toNormalVariables(variables);
  }
  std::unordered_map<std::string, std::vector<std::string>> partition_attributes;
  std::shared_ptr<const NodeAttributeStore> node_attributes;
  std::size_t loaded_node_attribute_values = 0;
  if (run_index && enable_value_partition) {
    partition_attributes = loadPartitionAttributes(
        snapshotFileForAssignments(path, "pattern_schema.tsv"),
        std::set<std::string>(selected.begin(), selected.end()));
    node_attributes = loadNodeAttributes(
        snapshotFileForAssignments(path, "pg_nodes.tsv"), partition_attributes,
        loaded_node_attribute_values);
  }

  std::vector<Mask> compare_masks;
  if (parent_workload) {
    compare_masks = masksForScopes(variables, parent_work.target_scopes);
  } else {
    const Mask all = (Mask{1} << variables.size()) - 1;
    for (Mask mask = 1; mask <= all; ++mask) {
      compare_masks.push_back(mask);
    }
  }

  std::cout << "dataset=" << path << '\n';
  std::cout << "selected_variables=" << selected.size() << '\n';
  std::cout << "count_scan_ms=" << count_ms << '\n';
  std::cout << "worker_threads=" << worker_threads << '\n';
  std::cout << "progress_every_tasks=" << progress_every << '\n';
  std::cout << "min_instances_per_variable=" << min_instances_per_variable << '\n';
  std::cout << "max_instances_per_variable=" << max_instances_per_variable << '\n';
  std::cout << "instance_sample_mode="
            << (log_scale_sample ? "balanced-log10" : "uniform-range") << '\n';
  if (log_scale_sample) {
    std::cout << "instance_sample_log10_min_exponent="
              << log_scale_min_exponent << '\n';
    std::cout << "instance_sample_log10_max_exponent="
              << log_scale_max_exponent << '\n';
    std::cout << "instance_scale_policy=" << log_scale_policy << '\n';
  }
  std::cout << "variable_order_policy="
            << ((order_policy == "scope-alphabetical" ||
                 order_policy == "scope-random")
                    ? "partial-greedy"
                    : order_policy)
            << '\n';
  std::cout << "scope_order_policy="
            << (parsed_scope_order_policy() == ScopeOrderPolicy::Alphabetical
                    ? "alphabetical"
                    : parsed_scope_order_policy() == ScopeOrderPolicy::Random
                          ? "random"
                          : "dfs-lexicographic")
            << '\n';
  std::cout << "result_cache_capacity=" << result_cache_capacity << '\n';
  std::cout << "result_cache_eviction_policy="
            << (result_cache_eviction_policy ==
                        ResultCacheEvictionPolicy::Belady
                    ? "belady" : "lru") << '\n';
  std::cout << "execution_schedule="
            << (parsed_execution_schedule() ==
                        ExecutionSchedule::ScopeOrderedWorkingPath
                    ? "scope-ordered-working-path"
                    : parsed_execution_schedule() ==
                        ExecutionSchedule::JoinSizeGreedyWorkingPath
                    ? "join-size-greedy-working-path"
                    : parsed_execution_schedule() == ExecutionSchedule::BoundedScopeCache
                    ? "bounded-scope-cache"
                    : parsed_execution_schedule() == ExecutionSchedule::CacheAwareGreedy
                    ? "cache-aware-greedy"
                    : parsed_execution_schedule() == ExecutionSchedule::IndependentSources
                          ? "independent-sources"
                          : "dfs-prefix-tree")
            << '\n';
  std::cout << "collect_partition_traces="
            << (collect_partition_traces ? 1 : 0) << '\n';
  std::cout << "adjustment_pruning=" << (enable_adjustment_pruning ? 1 : 0) << '\n';
  std::cout << "plan_order_from_unpruned_lattice="
            << (plan_order_from_unpruned_lattice ? 1 : 0) << '\n';
  std::cout << "value_partition_join="
            << (run_index && enable_value_partition ? 1 : 0) << '\n';
  std::cout << "value_partition_attribute_labels=" << partition_attributes.size() << '\n';
  std::cout << "value_partition_nodes_loaded="
            << (node_attributes ? node_attributes->size() : 0) << '\n';
  std::cout << "value_partition_attribute_values_loaded=" << loaded_node_attribute_values << '\n';
  if (max_instances_per_variable > 0 || log_scale_sample) {
    std::cout << "instance_sample_seed=" << sample_seed << '\n';
  }
  if (parent_workload) {
    std::cout << "parent_workload_plan_ms=" << parent_workload_plan_ms << '\n';
    std::cout << "causal_path=" << causal_path << '\n';
    std::cout << "parent_max_cluster_size=" << parent_k << '\n';
    std::cout << "parent_max_tree_depth=" << parent_max_tree_depth << '\n';
    std::cout << "legacy_parent_max_tree_depth_ignored=1\n";
    std::cout << "atomic_parent_factors=" << parent_work.factors.size() << '\n';
    std::cout << "skipped_parent_factors=" << parent_work.skipped_factors << '\n';
    std::cout << "oversize_atomic_parent_factors=" << parent_work.oversize_atomic_factors << '\n';
    std::cout << "merged_parent_clusters=" << parent_work.clusters.size() << '\n';
    std::cout << "parent_target_scopes=" << parent_work.target_scopes.size() << '\n';
    std::cout << "parent_source_scopes=" << parent_work.source_outputs.size() << '\n';
    std::cout << "adjustment_query_candidates="
              << parent_work.adjustment_query_candidates << '\n';
    std::cout << "multi_treatment_queries="
              << parent_work.multi_treatment_queries << '\n';
    std::cout << "relevant_adjustment_queries="
              << parent_work.relevant_adjustment_queries << '\n';
    std::cout << "unadjusted_queries=" << parent_work.unadjusted_queries << '\n';
    std::cout << "adjusted_queries=" << parent_work.adjusted_queries << '\n';
    std::cout << "trivial_queries_pruned="
              << parent_work.trivial_queries_pruned << '\n';
    std::cout << "adjustment_unavailable_queries="
              << parent_work.adjustment_unavailable_queries << '\n';
    std::cout << "adjustment_scopes_not_parent_covered="
              << parent_work.adjustment_scopes_not_parent_covered << '\n';
    std::cout << "query_clusters_created=" << parent_work.query_clusters_created << '\n';
    std::cout << "tree_lattice_scope_candidates="
              << parent_work.tree_lattice_scope_candidates << '\n';
    std::cout << "tree_lattice_scopes_retained="
              << parent_work.tree_lattice_scopes_retained << '\n';
    std::cout << "tree_lattice_scopes_pruned="
              << parent_work.tree_lattice_scopes_pruned << '\n';
    std::cout << "projected_causal_output_scopes=0\n";
    for (const auto& cluster : parent_work.clusters) {
      std::cout << "cluster\t" << cluster.id << "\twidth=" << cluster.scope.size()
                << "\tfactors=" << cluster.factors.size() << "\tscope=";
      for (std::size_t i = 0; i < cluster.scope.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << cluster.scope[i];
      }
      std::cout << '\n';
    }
  }
  for (const auto& variable : variables) {
    const auto coverage_found = workload_coverage.find(variable.id);
    const auto coverage = coverage_found == workload_coverage.end()
        ? std::size_t{0}
        : coverage_found->second;
    std::cout << "var\t" << variable.id
              << "\tinstances=" << variable.instances.size()
              << "\tworkload_coverage=" << coverage << '\n';
  }
  if (sample_only) {
    return 0;
  }

  auto total_value_cells = [](const std::vector<ValueCountMap>& maps) {
    std::size_t total = 0;
    for (const auto& map : maps) {
      total += map.size();
    }
    return total;
  };
  auto total_trie_nodes = [](const std::vector<std::size_t>& nodes) {
    std::size_t total = 0;
    for (const auto value : nodes) {
      total += value;
    }
    return total;
  };
  auto add_stats = [](RuntimeCacheStats& dst, const RuntimeCacheStats& src) {
    dst.hits += src.hits;
    dst.non_root_hits += src.non_root_hits;
    dst.misses += src.misses;
    dst.inserts += src.inserts;
    dst.evictions += src.evictions;
    dst.recomputations += src.recomputations;
    dst.peak_resident_tuples = std::max(dst.peak_resident_tuples, src.peak_resident_tuples);
    dst.path_budget_overflows += src.path_budget_overflows;
    dst.peak_resident_results =
        std::max(dst.peak_resident_results, src.peak_resident_results);
    dst.result_capacity_evictions += src.result_capacity_evictions;
    dst.result_capacity_overflows += src.result_capacity_overflows;
    dst.working_path_tasks += src.working_path_tasks;
    dst.working_path_reuse_transitions +=
        src.working_path_reuse_transitions;
    dst.working_path_reused_prefix_levels +=
        src.working_path_reused_prefix_levels;
    dst.working_path_join_steps += src.working_path_join_steps;
    dst.working_path_rebuilt_steps += src.working_path_rebuilt_steps;
    dst.trie_nodes_visited += src.trie_nodes_visited;
    dst.trie_subtrees_skipped += src.trie_subtrees_skipped;
    dst.live_candidates_emitted += src.live_candidates_emitted;
    dst.derive_candidates_considered += src.derive_candidates_considered;
    dst.derive_used += src.derive_used;
    dst.derive_materialized_masks += src.derive_materialized_masks;
    dst.derive_materialized_tuples += src.derive_materialized_tuples;
    dst.anchor_buckets_used += src.anchor_buckets_used;
    dst.anchor_candidates_scanned += src.anchor_candidates_scanned;
    dst.intersection_buckets_built += src.intersection_buckets_built;
    dst.intersection_bucket_rows += src.intersection_bucket_rows;
    dst.intersection_candidate_queries += src.intersection_candidate_queries;
    dst.value_partition_joins += src.value_partition_joins;
    dst.value_partition_features_selected += src.value_partition_features_selected;
    dst.value_partition_buckets_built += src.value_partition_buckets_built;
    dst.value_partition_parent_rows += src.value_partition_parent_rows;
    dst.value_partition_candidate_rows += src.value_partition_candidate_rows;
    dst.value_partition_pair_checks += src.value_partition_pair_checks;
    dst.value_partition_compatible_pairs += src.value_partition_compatible_pairs;
    for (const auto& item : src.value_partition_feature_uses) {
      dst.value_partition_feature_uses[item.first] += item.second;
    }
    dst.grouped_prefix_hits += src.grouped_prefix_hits;
    dst.grouped_prefix_inserts += src.grouped_prefix_inserts;
    dst.leaf_cube_pushdowns += src.leaf_cube_pushdowns;
    dst.leaf_cube_rows += src.leaf_cube_rows;
  };

  std::vector<Count> baseline_counts;
  std::vector<ValueCountMap> baseline_value_counts;
  std::vector<std::size_t> baseline_value_trie_nodes;
  std::map<Mask, Count> parent_baseline_counts;
  std::map<Mask, ValueCountMap> parent_baseline_value_counts;
  std::map<Mask, std::size_t> parent_baseline_value_trie_nodes;
  double normal_ms = 0.0;
  if (run_baseline) {
    normal_ms = timeMillis([&] {
      if (parent_workload) {
        for (const auto& cluster : parent_work.clusters) {
          const auto cluster_variables = variablesForScope(variables, cluster.scope);
          const auto cluster_normal_variables = toNormalVariables(cluster_variables);
          const auto source_outputs =
              outputMasksBySource(cluster_variables, cluster.source_outputs);
          std::vector<std::size_t> planned_order;
          if (!parent_greedy_baseline) {
            LatticeBuildConfig planner_config;
            planner_config.enable_derive = false;
            planner_config.variable_order_policy = parsed_order_policy();
            planner_config.source_output_scopes =
                toSourceOutputScopes(cluster.source_outputs);
            FullLatticeIndex planner(cluster_variables, planner_config);
            planner.planOnly();
            planned_order = planner.plan().variable_order_indices;
          }
          std::cout << "baseline_cluster_start\t" << cluster.id
                    << "\twidth=" << cluster.scope.size()
                    << "\tsources=" << source_outputs.size()
                    << "\toutputs=" << cluster.target_scopes.size()
                    << "\tbaseline_order="
                    << (parent_greedy_baseline
                            ? "instance_count_per_source"
                            : order_policy + "_shared_order_per_source")
                    << '\n';
          std::cout.flush();
          std::size_t completed_sources = 0;
          for (const auto& item : source_outputs) {
            const auto projected = normalIndependentJoinForSourceOutputs(
                cluster_normal_variables, item.first, item.second,
                parent_greedy_baseline,
                parent_greedy_baseline ? nullptr : &planned_order);
            completed_sources += 1;
            Count source_rows = 0;
            for (const auto& out_item : projected) {
              const auto mask =
                  translateMask(cluster_variables, variables, out_item.first);
              parent_baseline_counts[mask] = out_item.second.rows;
              parent_baseline_value_counts[mask] = out_item.second.value_counts;
              parent_baseline_value_trie_nodes[mask] =
                  valueTrieNodeCount(out_item.second.value_counts);
              source_rows = out_item.second.rows;
            }
            if (progress_every > 0 &&
                (completed_sources == 1 || completed_sources == source_outputs.size() ||
                 completed_sources % progress_every == 0)) {
              const double pct = 100.0 * static_cast<double>(completed_sources) /
                                 static_cast<double>(std::max<std::size_t>(1, source_outputs.size()));
              const std::time_t now = std::time(nullptr);
              std::cout << "baseline_source_progress"
                        << "\ttime=" << now
                        << "\tcluster=" << cluster.id
                        << "\tcompleted=" << completed_sources << '/' << source_outputs.size()
                        << "\tpct=" << pct
                        << "\tsource=" << item.first
                        << "\toutputs=" << item.second.size()
                        << "\trows=" << source_rows << '\n';
              std::cout.flush();
            }
          }
          std::cout << "baseline_cluster_done\t" << cluster.id << '\n';
          std::cout.flush();
        }
      } else {
        baseline_counts = normalIndependentJoinCountsForMasks(
            normal_variables, compare_masks, worker_threads, progress_every, &variables, "",
            &baseline_value_counts, &baseline_value_trie_nodes);
      }
    });
  }

  std::vector<Count> index_counts;
  std::vector<ValueCountMap> index_value_counts;
  std::vector<std::size_t> index_value_trie_nodes;
  std::map<Mask, Count> parent_index_counts;
  std::map<Mask, ValueCountMap> parent_index_value_counts;
  std::map<Mask, std::size_t> parent_index_value_trie_nodes;
  RuntimeCacheStats stats;
  Count index_emitted_rows = 0;
  Count all_variables_count = 0;
  double index_ms = 0.0;
  if (run_index) {
    index_ms = timeMillis([&] {
      if (parent_workload) {
        for (const auto& cluster : parent_work.clusters) {
          const auto cluster_variables = variablesForScope(variables, cluster.scope);
          LatticeBuildConfig config;
          config.cache_budget_tuples = 0;
          config.result_cache_capacity = result_cache_capacity;
          config.result_cache_eviction_policy =
              result_cache_eviction_policy;
          config.enable_derive = derive_budget > 0;
          config.derive_materialization_budget_tuples = derive_budget;
          config.worker_threads = worker_threads;
          config.enable_value_partition_join = enable_value_partition;
          config.collect_partition_traces = collect_partition_traces;
          config.variable_order_policy = parsed_order_policy();
          config.plan_order_from_unpruned_lattice =
              plan_order_from_unpruned_lattice;
          config.scope_order_policy = parsed_scope_order_policy();
          config.scope_order_seed = sample_seed;
          config.execution_schedule = parsed_execution_schedule();
          config.node_attributes = node_attributes;
          config.value_partition_attributes_by_label = partition_attributes;
          config.source_output_scopes = toSourceOutputScopes(cluster.source_outputs);
          config.progress_every_tasks = progress_every;
          if (progress_every > 0) {
            config.progress_callback = [&](const ProgressEvent& event) {
              const double pct = event.total_tasks == 0
                  ? 100.0
                  : 100.0 * static_cast<double>(event.completed_tasks) /
                        static_cast<double>(event.total_tasks);
              const std::time_t now = std::time(nullptr);
              std::cout << "progress"
                        << "\ttime=" << now
                        << "\tcluster=" << cluster.id
                        << "\tcompleted=" << event.completed_tasks << '/' << event.total_tasks
                        << "\tpct=" << pct
                        << "\tmask=" << event.mask
                        << "\tparent=" << event.parent_mask
                        << "\tvar_index=" << event.variable
                        << "\tvar=" << cluster_variables[event.variable].id
                        << "\tdepth=" << event.depth
                        << "\trows=" << event.rows
                        << "\tresident=" << event.resident_tuples
                        << "\tpeak_resident=" << event.peak_resident_tuples
                        << "\tresident_results=" << event.resident_results
                        << "\tpeak_resident_results=" << event.peak_resident_results
                        << "\temitted=" << event.emitted_rows
                        << '\n';
              std::cout.flush();
            };
          }
          std::cout << "index_cluster_start\t" << cluster.id
                    << "\twidth=" << cluster.scope.size()
                    << "\tsources=" << cluster.source_outputs.size()
                    << "\toutputs=" << cluster.target_scopes.size() << '\n';
          std::cout.flush();
          FullLatticeIndex index(cluster_variables, config);
          index.build();
          if (collect_partition_traces) {
            auto print_histogram = [](const auto& histogram) {
              for (std::size_t i = 0; i < histogram.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << histogram[i];
              }
            };
            for (const auto& trace : index.result().partition_traces) {
              std::cout << "partition_trace"
                        << "\tcluster=" << cluster.id
                        << "\tmode=" << trace.mode
                        << "\tparent_mask=" << trace.parent_mask
                        << "\tchild_mask=" << trace.child_mask
                        << "\tvariable=" << trace.variable
                        << "\tparent_rows=" << trace.parent_rows
                        << "\tcandidate_rows=" << trace.candidate_rows
                        << "\tjoin_roles=" << trace.join_roles
                        << "\tparent_partition_buckets="
                        << trace.parent_partition_buckets
                        << "\tcandidate_partition_buckets="
                        << trace.candidate_partition_buckets
                        << "\tbuckets=" << trace.bucket_count
                        << "\tunmatched_parent_rows=" << trace.unmatched_parent_rows
                        << "\tunmatched_candidate_rows=" << trace.unmatched_candidate_rows
                        << "\tcartesian_pairs=" << trace.cartesian_pairs
                        << "\tpartition_pairs=" << trace.partition_pairs
                        << "\tmax_bucket_pairs=" << trace.max_bucket_pairs
                        << "\tparent_bucket_sum=" << trace.parent_bucket_sum
                        << "\tparent_bucket_square_sum="
                        << trace.parent_bucket_square_sum
                        << "\tparent_bucket_xlogx_sum="
                        << trace.parent_bucket_xlogx_sum
                        << "\tcandidate_bucket_sum=" << trace.candidate_bucket_sum
                        << "\tcandidate_bucket_square_sum="
                        << trace.candidate_bucket_square_sum
                        << "\tcandidate_bucket_xlogx_sum="
                        << trace.candidate_bucket_xlogx_sum
                        << "\tpair_work_square_sum=" << trace.pair_work_square_sum
                        << "\tpair_work_xlogx_sum=" << trace.pair_work_xlogx_sum
                        << "\tmax_parent_bucket=" << trace.max_parent_bucket
                        << "\tmax_candidate_bucket=" << trace.max_candidate_bucket
                        << "\tcompatible_pairs=" << trace.compatible_pairs
                        << "\toutput_rows=" << trace.output_rows
                        << "\tfeatures=";
              for (std::size_t i = 0; i < trace.value_features.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << trace.value_features[i];
              }
              std::cout << "\tparent_hist=";
              print_histogram(trace.parent_bucket_hist);
              std::cout << "\tcandidate_hist=";
              print_histogram(trace.candidate_bucket_hist);
              std::cout << "\tpair_hist=";
              print_histogram(trace.pair_work_hist);
              std::cout << '\n';
            }
          }
          for (const auto& scope : cluster.target_scopes) {
            const auto local_mask = maskForScope(cluster_variables, scope);
            const auto global_mask = maskForScope(variables, scope);
            const auto local_index = static_cast<std::size_t>(local_mask);
            parent_index_counts[global_mask] =
                index.result().exact_counts[local_index];
            parent_index_value_counts[global_mask] =
                index.result().value_counts[local_index];
            parent_index_value_trie_nodes[global_mask] =
                index.result().value_trie_nodes[local_index];
          }
          add_stats(stats, index.result().cache_stats);
          index_emitted_rows += index.result().emitted_rows;
          const auto& plan = index.plan();
          std::cout << "index_cluster_done\t" << cluster.id
                    << "\ttask_order_size=" << plan.task_order.size()
                    << "\ttarget_mask_count=" << plan.target_masks.size()
                    << "\tpeak_resident=" << index.result().cache_stats.peak_resident_tuples
                    << "\tpeak_resident_results="
                    << index.result().cache_stats.peak_resident_results
                    << "\temitted=" << index.result().emitted_rows << '\n';
          std::cout << "cluster_variable_order\t" << cluster.id;
          for (const auto& id : plan.variable_order) {
            std::cout << '\t' << id;
          }
          std::cout << '\n';
          std::cout.flush();
        }
      } else {
        LatticeBuildConfig config;
        config.cache_budget_tuples = 0;
        config.result_cache_capacity = result_cache_capacity;
        config.result_cache_eviction_policy =
            result_cache_eviction_policy;
        config.enable_derive = derive_budget > 0;
        config.derive_materialization_budget_tuples = derive_budget;
        config.worker_threads = worker_threads;
        config.enable_value_partition_join = enable_value_partition;
        config.collect_partition_traces = collect_partition_traces;
        config.variable_order_policy = parsed_order_policy();
        config.plan_order_from_unpruned_lattice =
            plan_order_from_unpruned_lattice;
        config.scope_order_policy = parsed_scope_order_policy();
        config.scope_order_seed = sample_seed;
        config.execution_schedule = parsed_execution_schedule();
        config.node_attributes = node_attributes;
        config.value_partition_attributes_by_label = partition_attributes;
        config.progress_every_tasks = progress_every;
        if (progress_every > 0) {
          config.progress_callback = [&](const ProgressEvent& event) {
            const double pct = event.total_tasks == 0
                ? 100.0
                : 100.0 * static_cast<double>(event.completed_tasks) /
                      static_cast<double>(event.total_tasks);
            const std::time_t now = std::time(nullptr);
            std::cout << "progress"
                      << "\ttime=" << now
                      << "\tcompleted=" << event.completed_tasks << '/' << event.total_tasks
                      << "\tpct=" << pct
                      << "\tmask=" << event.mask
                      << "\tparent=" << event.parent_mask
                      << "\tvar_index=" << event.variable
                      << "\tvar=" << variables[event.variable].id
                      << "\tdepth=" << event.depth
                      << "\trows=" << event.rows
                      << "\tresident=" << event.resident_tuples
                      << "\tpeak_resident=" << event.peak_resident_tuples
                      << "\tresident_results=" << event.resident_results
                      << "\tpeak_resident_results=" << event.peak_resident_results
                      << "\temitted=" << event.emitted_rows
                      << '\n';
            std::cout.flush();
          };
        }
        FullLatticeIndex index(variables, config);
        index.build();
        const auto& plan = index.plan();
        std::cout << "variable_order";
        for (const auto& id : plan.variable_order) {
          std::cout << '\t' << id;
        }
        std::cout << '\n';
        std::cout << "task_order_size=" << plan.task_order.size() << '\n';
        std::cout << "target_mask_count=" << plan.target_masks.size() << '\n';
        index_counts = index.result().exact_counts;
        index_value_counts = index.result().value_counts;
        index_value_trie_nodes = index.result().value_trie_nodes;
        stats = index.result().cache_stats;
        index_emitted_rows = index.result().emitted_rows;
        all_variables_count = index.exactCount(index.allVariablesMask());
      }
    });
  }

  bool counts_match = true;
  if (run_baseline && run_index) {
    for (const Mask mask : compare_masks) {
      if (parent_workload) {
        const auto baseline_found = parent_baseline_counts.find(mask);
        const auto index_found = parent_index_counts.find(mask);
        const auto baseline_count = baseline_found == parent_baseline_counts.end()
            ? Count{0}
            : baseline_found->second;
        const auto index_count = index_found == parent_index_counts.end()
            ? Count{0}
            : index_found->second;
        if (baseline_count != index_count) {
          counts_match = false;
          std::cout << "mismatch_mask=" << mask << " baseline=" << baseline_count
                    << " index=" << index_count << '\n';
        }
        const auto baseline_values = parent_baseline_value_counts.find(mask);
        const auto index_values = parent_index_value_counts.find(mask);
        const ValueCountMap empty_values;
        const auto& baseline_map = baseline_values == parent_baseline_value_counts.end()
            ? empty_values
            : baseline_values->second;
        const auto& index_map = index_values == parent_index_value_counts.end()
            ? empty_values
            : index_values->second;
        if (baseline_map != index_map) {
          counts_match = false;
          std::cout << "mismatch_value_cube_mask=" << mask
                    << " baseline_cells=" << baseline_map.size()
                    << " index_cells=" << index_map.size() << '\n';
        }
      } else {
        const auto i = static_cast<std::size_t>(mask);
        if (baseline_counts[i] != index_counts[i]) {
          counts_match = false;
          std::cout << "mismatch_mask=" << i << " baseline=" << baseline_counts[i]
                    << " index=" << index_counts[i] << '\n';
        }
        if (!baseline_value_counts.empty() && !index_value_counts.empty() &&
            baseline_value_counts[i] != index_value_counts[i]) {
          counts_match = false;
          std::cout << "mismatch_value_cube_mask=" << i
                    << " baseline_cells=" << baseline_value_counts[i].size()
                    << " index_cells=" << index_value_counts[i].size() << '\n';
        }
      }
    }
  }

  Count baseline_total = 0;
  if (parent_workload) {
    for (const auto& item : parent_baseline_counts) {
      baseline_total += item.second;
    }
  } else {
    for (const auto value : baseline_counts) {
      baseline_total += value;
    }
  }

  auto total_sparse_value_cells = [](const std::map<Mask, ValueCountMap>& maps) {
    std::size_t total = 0;
    for (const auto& item : maps) total += item.second.size();
    return total;
  };
  auto total_sparse_trie_nodes = [](const std::map<Mask, std::size_t>& nodes) {
    std::size_t total = 0;
    for (const auto& item : nodes) total += item.second;
    return total;
  };

  std::cout << "subsets=" << ((Mask{1} << variables.size()) - 1) << '\n';
  std::cout << "compare_masks=" << compare_masks.size() << '\n';
  std::cout << "mode=" << mode << '\n';
  std::cout << "derive_materialization_budget_tuples=" << derive_budget << '\n';
  std::cout << "baseline_normal_join_ms=" << (run_baseline ? normal_ms : -1.0) << '\n';
  std::cout << "prefix_reuse_build_ms=" << (run_index ? index_ms : -1.0) << '\n';
  std::cout << "speedup=" << (run_baseline && run_index && index_ms > 0.0 ? normal_ms / index_ms : -1.0) << '\n';
  std::cout << "counts_match=" << (run_baseline && run_index ? (counts_match ? "yes" : "no") : "not_checked") << '\n';
  std::cout << "baseline_total_output_rows=" << baseline_total << '\n';
  std::cout << "baseline_value_cube_cells="
            << (run_baseline
                    ? (parent_workload
                           ? total_sparse_value_cells(parent_baseline_value_counts)
                           : total_value_cells(baseline_value_counts))
                    : 0)
            << '\n';
  std::cout << "baseline_value_trie_nodes="
            << (run_baseline
                    ? (parent_workload
                           ? total_sparse_trie_nodes(parent_baseline_value_trie_nodes)
                           : total_trie_nodes(baseline_value_trie_nodes))
                    : 0)
            << '\n';
  std::cout << "index_emitted_rows=" << (run_index ? index_emitted_rows : 0) << '\n';
  std::cout << "index_value_cube_cells="
            << (run_index
                    ? (parent_workload
                           ? total_sparse_value_cells(parent_index_value_counts)
                           : total_value_cells(index_value_counts))
                    : 0)
            << '\n';
  std::cout << "index_value_trie_nodes="
            << (run_index
                    ? (parent_workload
                           ? total_sparse_trie_nodes(parent_index_value_trie_nodes)
                           : total_trie_nodes(index_value_trie_nodes))
                    : 0)
            << '\n';
  std::cout << "all_variables_count=" << (run_index ? all_variables_count : 0) << '\n';
  std::cout << "cache_hits=" << stats.hits << '\n';
  std::cout << "cache_non_root_hits=" << stats.non_root_hits << '\n';
  std::cout << "cache_misses=" << stats.misses << '\n';
  std::cout << "cache_recomputations=" << stats.recomputations << '\n';
  std::cout << "cache_evictions=" << stats.evictions << '\n';
  std::cout << "cache_peak_resident_results="
            << stats.peak_resident_results << '\n';
  std::cout << "cache_result_capacity_evictions="
            << stats.result_capacity_evictions << '\n';
  std::cout << "cache_result_capacity_overflows="
            << stats.result_capacity_overflows << '\n';
  std::cout << "working_path_tasks=" << stats.working_path_tasks << '\n';
  std::cout << "working_path_reuse_transitions="
            << stats.working_path_reuse_transitions << '\n';
  std::cout << "working_path_reused_prefix_levels="
            << stats.working_path_reused_prefix_levels << '\n';
  std::cout << "working_path_join_steps="
            << stats.working_path_join_steps << '\n';
  std::cout << "working_path_rebuilt_steps="
            << stats.working_path_rebuilt_steps << '\n';
  std::cout << "trie_nodes_visited=" << stats.trie_nodes_visited << '\n';
  std::cout << "trie_subtrees_skipped=" << stats.trie_subtrees_skipped << '\n';
  std::cout << "live_candidates_emitted=" << stats.live_candidates_emitted << '\n';
  std::cout << "derive_materialized_masks=" << stats.derive_materialized_masks << '\n';
  std::cout << "derive_materialized_tuples=" << stats.derive_materialized_tuples << '\n';
  std::cout << "derive_candidates_considered=" << stats.derive_candidates_considered << '\n';
  std::cout << "derive_used=" << stats.derive_used << '\n';
  std::cout << "anchor_buckets_used=" << stats.anchor_buckets_used << '\n';
  std::cout << "anchor_candidates_scanned=" << stats.anchor_candidates_scanned << '\n';
  std::cout << "intersection_buckets_built=" << stats.intersection_buckets_built << std::endl;
  std::cout << "intersection_bucket_rows=" << stats.intersection_bucket_rows << std::endl;
  std::cout << "intersection_candidate_queries=" << stats.intersection_candidate_queries << std::endl;
  std::cout << "value_partition_joins=" << stats.value_partition_joins << std::endl;
  std::cout << "value_partition_features_selected="
            << stats.value_partition_features_selected << std::endl;
  std::cout << "value_partition_buckets_built="
            << stats.value_partition_buckets_built << std::endl;
  std::cout << "value_partition_parent_rows="
            << stats.value_partition_parent_rows << std::endl;
  std::cout << "value_partition_candidate_rows="
            << stats.value_partition_candidate_rows << std::endl;
  std::cout << "value_partition_pair_checks="
            << stats.value_partition_pair_checks << std::endl;
  std::cout << "value_partition_compatible_pairs="
            << stats.value_partition_compatible_pairs << std::endl;
  for (const auto& item : stats.value_partition_feature_uses) {
    std::cout << "value_partition_feature\t" << item.first << "\tuses=" << item.second
              << std::endl;
  }
  std::cout << "grouped_prefix_hits=" << stats.grouped_prefix_hits << std::endl;
  std::cout << "grouped_prefix_inserts=" << stats.grouped_prefix_inserts << std::endl;
  std::cout << "leaf_cube_pushdowns=" << stats.leaf_cube_pushdowns << std::endl;
  std::cout << "leaf_cube_rows=" << stats.leaf_cube_rows << std::endl;
  return counts_match ? 0 : 2;
}
