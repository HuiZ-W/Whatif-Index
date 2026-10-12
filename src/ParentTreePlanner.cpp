#include "whatif/ParentTreePlanner.h"

#include "whatif/CausalAdjustment.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace whatif {
namespace {

using Scope = std::vector<VariableId>;

std::string scopeKey(const Scope &scope) {
  std::string key;
  for (const auto &variable : scope) {
    key += std::to_string(variable.size());
    key.push_back(':');
    key += variable;
    key.push_back('|');
  }
  return key;
}

Scope canonicalScope(const Scope &scope,
                     const std::unordered_map<VariableId, std::size_t> &rank) {
  Scope result = scope;
  std::sort(result.begin(), result.end(),
            [&](const auto &left, const auto &right) {
              return rank.at(left) < rank.at(right);
            });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool containsScope(const Scope &container, const Scope &candidate) {
  const std::unordered_set<VariableId> members(container.begin(),
                                               container.end());
  return std::all_of(candidate.begin(), candidate.end(),
                     [&](const auto &v) { return members.count(v) != 0; });
}

Scope scopeUnion(const Scope &left, const Scope &right,
                 const std::unordered_map<VariableId, std::size_t> &rank) {
  Scope result = left;
  result.insert(result.end(), right.begin(), right.end());
  return canonicalScope(result, rank);
}

std::size_t intersectionSize(const Scope &left, const Scope &right) {
  const std::unordered_set<VariableId> members(left.begin(), left.end());
  return static_cast<std::size_t>(
      std::count_if(right.begin(), right.end(), [&](const auto &variable) {
        return members.count(variable) != 0;
      }));
}

std::vector<Scope> maximalScopes(const std::map<std::string, Scope> &unique) {
  std::vector<Scope> ordered;
  ordered.reserve(unique.size());
  for (const auto &entry : unique) {
    ordered.push_back(entry.second);
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto &left,
                                               const auto &right) {
    if (left.size() != right.size()) {
      return left.size() > right.size();
    }
    return scopeKey(left) < scopeKey(right);
  });

  std::vector<Scope> maximal;
  for (const auto &scope : ordered) {
    if (std::none_of(maximal.begin(), maximal.end(), [&](const auto &larger) {
          return containsScope(larger, scope);
        })) {
      maximal.push_back(scope);
    }
  }
  return maximal;
}

} // namespace

ParentTreePlan ParentTreePlanner::plan(
    const std::vector<VariableInput> &variables,
    const std::vector<CausalEdge> &causal_edges,
    const std::vector<std::vector<VariableId>> &atomic_parent_scopes,
    std::size_t maximum_width) {
  if (variables.empty()) {
    throw std::runtime_error("Parent Tree planner needs causal variables");
  }
  if (atomic_parent_scopes.empty()) {
    throw std::runtime_error("Parent Tree planner needs atomic parent factors");
  }
  if (maximum_width == 0) {
    throw std::runtime_error("maximum Parent Tree width must be positive");
  }

  ParentTreePlan result;
  result.stats.atomic_factor_scopes = atomic_parent_scopes.size();

  std::unordered_map<VariableId, std::size_t> rank;
  std::unordered_map<VariableId, const VariableInput *> input_by_id;
  for (std::size_t i = 0; i < variables.size(); ++i) {
    if (variables[i].id.empty() || !rank.emplace(variables[i].id, i).second) {
      throw std::runtime_error("duplicate or empty causal variable ID");
    }
    input_by_id.emplace(variables[i].id, &variables[i]);
  }

  std::set<std::pair<VariableId, VariableId>> edge_set;
  for (const auto &edge : causal_edges) {
    if (rank.count(edge.parent) == 0 || rank.count(edge.child) == 0) {
      throw std::runtime_error("causal edge references an unknown variable");
    }
    if (edge.parent == edge.child) {
      throw std::runtime_error("causal DAG cannot contain a self edge");
    }
    edge_set.emplace(edge.parent, edge.child);
  }

  std::map<std::string, Scope> unique_atomic;
  for (const auto &raw_scope : atomic_parent_scopes) {
    if (raw_scope.empty()) {
      throw std::runtime_error("atomic parent factor cannot be empty");
    }
    for (const auto &variable : raw_scope) {
      if (rank.count(variable) == 0) {
        throw std::runtime_error("atomic parent factor references unknown " +
                                 variable);
      }
    }
    auto scope = canonicalScope(raw_scope, rank);
    unique_atomic.emplace(scopeKey(scope), std::move(scope));
  }
  result.stats.unique_atomic_factor_scopes = unique_atomic.size();

  std::vector<VariableId> variable_ids;
  variable_ids.reserve(variables.size());
  for (const auto &variable : variables) {
    variable_ids.push_back(variable.id);
  }
  std::vector<std::pair<std::string, std::string>> dag_edges;
  dag_edges.reserve(edge_set.size());
  for (const auto &edge : edge_set) {
    dag_edges.push_back(edge);
  }
  const whatif::CausalDag dag(variable_ids, dag_edges);
  const whatif::CausalMask all =
      (whatif::CausalMask{1} << variables.size()) - 1;
  std::map<std::string, Scope> unique_query_scopes;
  std::map<std::string, Scope> oversized_query_scopes;
  auto add_query_scope = [&](whatif::CausalMask mask) {
    auto scope = canonicalScope(dag.variablesOf(mask), rank);
    auto &destination = scope.size() <= maximum_width
                            ? unique_query_scopes
                            : oversized_query_scopes;
    destination.emplace(scopeKey(scope), std::move(scope));
  };
  for (std::size_t treatment = 0; treatment < variables.size(); ++treatment) {
    const auto treatment_mask =
        whatif::CausalMask{1} << treatment;
    for (std::size_t outcome = 0; outcome < variables.size(); ++outcome) {
      if (treatment == outcome) {
        continue;
      }
      const auto outcome_mask = whatif::CausalMask{1} << outcome;
      if (!dag.reaches(treatment_mask, outcome_mask)) {
        continue;
      }
      result.stats.reachable_causal_pairs += 1;
      const auto path = dag.causalPathClosure(treatment_mask, outcome_mask);
      add_query_scope(path);

      const auto analysis = dag.analyzeAdjustmentWithin(
          treatment_mask, outcome_mask,
          all & ~(treatment_mask | outcome_mask));
      if (!analysis.adjustment_exists) {
        continue;
      }
      result.stats.identifiable_query_scopes += 1;
      add_query_scope(path | analysis.adjustment);
    }
  }
  result.stats.localizable_query_scopes = unique_query_scopes.size();
  result.stats.oversized_query_scopes = oversized_query_scopes.size();

  auto shared_roles = [&](const VariableId &left, const VariableId &right) {
    const auto &left_roles = input_by_id.at(left)->pattern_roles;
    const std::unordered_set<RoleId> roles(left_roles.begin(),
                                           left_roles.end());
    return static_cast<std::size_t>(std::count_if(
        input_by_id.at(right)->pattern_roles.begin(),
        input_by_id.at(right)->pattern_roles.end(),
        [&](const auto &role) { return roles.count(role) != 0; }));
  };
  auto causally_adjacent = [&](const VariableId &left,
                               const VariableId &right) {
    return edge_set.count({left, right}) != 0 ||
           edge_set.count({right, left}) != 0;
  };

  struct VariableCostStats {
    double instances = 0.0;
    std::size_t role_count = 0;
    std::unordered_map<RoleId, double> role_distinct;
  };
  std::unordered_map<VariableId, VariableCostStats> variable_cost_stats;
  constexpr std::size_t kDistinctSampleLimit = 4096;
  for (const auto &variable : variables) {
    VariableCostStats stats;
    stats.instances = static_cast<double>(variable.instances.size());
    stats.role_count = variable.pattern_roles.size();
    const std::size_t sampled =
        std::min(kDistinctSampleLimit, variable.instances.size());
    std::unordered_map<RoleId, std::unordered_set<std::size_t>> observed;
    for (std::size_t sample = 0; sample < sampled; ++sample) {
      const std::size_t index = sample * variable.instances.size() / sampled;
      for (const auto &binding : variable.instances[index].roles) {
        observed[binding.role].insert(std::hash<NodeId>{}(binding.node));
      }
    }
    for (const auto &role : variable.pattern_roles) {
      const double distinct = static_cast<double>(observed[role].size());
      const double scaled = sampled == 0
                                ? 1.0
                                : distinct * stats.instances /
                                      static_cast<double>(sampled);
      stats.role_distinct[role] =
          std::max(1.0, std::min(std::max(1.0, stats.instances), scaled));
    }
    variable_cost_stats.emplace(variable.id, std::move(stats));
  }

  auto mask_of_scope = [&](const Scope &scope) {
    whatif::CausalMask mask = 0;
    for (const auto &variable : scope) {
      mask |= whatif::CausalMask{1} << rank.at(variable);
    }
    return mask;
  };
  auto scope_of_mask = [&](whatif::CausalMask mask) {
    Scope scope;
    for (std::size_t i = 0; i < variables.size(); ++i) {
      if ((mask & (whatif::CausalMask{1} << i)) != 0) {
        scope.push_back(variables[i].id);
      }
    }
    return scope;
  };

  std::unordered_map<std::string, double> cardinality_cache;
  std::function<double(const Scope &)> estimated_cardinality =
      [&](const Scope &raw_scope) {
        const auto scope = canonicalScope(raw_scope, rank);
        const auto key = scopeKey(scope);
        const auto cached = cardinality_cache.find(key);
        if (cached != cardinality_cache.end()) {
          return cached->second;
        }
        if (scope.empty()) {
          return 1.0;
        }

        struct RoleOccurrence {
          VariableId variable;
          RoleId role;
          double distinct = 1.0;
        };
        std::vector<RoleOccurrence> occurrences;
        std::unordered_map<RoleId, std::vector<double>> role_distinct;
        long double log_estimate = 0.0L;
        for (const auto &variable : scope) {
          const auto &stats = variable_cost_stats.at(variable);
          if (stats.instances == 0.0) {
            cardinality_cache.emplace(key, 0.0);
            return 0.0;
          }
          log_estimate += std::log(stats.instances);
          for (const auto &role : input_by_id.at(variable)->pattern_roles) {
            const auto found = stats.role_distinct.find(role);
            const double distinct =
                found == stats.role_distinct.end() ? 1.0 : found->second;
            role_distinct[role].push_back(distinct);
            occurrences.push_back({variable, role, distinct});
          }
        }
        for (const auto &[role, distinct_values] : role_distinct) {
          (void)role;
          if (distinct_values.size() <= 1) {
            continue;
          }
          const double distinct =
              *std::max_element(distinct_values.begin(), distinct_values.end());
          log_estimate -= static_cast<long double>(distinct_values.size() - 1) *
                          std::log(std::max(1.0, distinct));
        }
        for (std::size_t left = 0; left < occurrences.size(); ++left) {
          for (std::size_t right = left + 1; right < occurrences.size();
               ++right) {
            if (occurrences[left].variable == occurrences[right].variable ||
                occurrences[left].role == occurrences[right].role) {
              continue;
            }
            const double distinct =
                std::max(occurrences[left].distinct,
                         occurrences[right].distinct);
            const double correction = 1.0 - 1.0 / std::max(1.0, distinct);
            if (correction <= 0.0) {
              cardinality_cache.emplace(key, 0.0);
              return 0.0;
            }
            log_estimate += std::log(correction);
          }
        }
        constexpr long double kMaximumEstimate = 1.0e18L;
        const long double estimate =
            log_estimate >= std::log(kMaximumEstimate)
                ? kMaximumEstimate
                : std::max(0.0L, std::exp(log_estimate));
        const double result_value = static_cast<double>(estimate);
        cardinality_cache.emplace(key, result_value);
        return result_value;
      };

  auto edge_cost = [&](whatif::CausalMask parent,
                       whatif::CausalMask child) {
    const auto parent_scope = scope_of_mask(parent);
    const auto child_scope = scope_of_mask(child);
    const double parent_size = parent == 0 ? 1.0
                                           : estimated_cardinality(parent_scope);
    const double child_size = estimated_cardinality(child_scope);
    std::unordered_set<RoleId> child_roles;
    for (const auto &variable : child_scope) {
      const auto &roles = input_by_id.at(variable)->pattern_roles;
      child_roles.insert(roles.begin(), roles.end());
    }
    const double width = static_cast<double>(child_scope.size());
    const double role_width = static_cast<double>(child_roles.size());
    double cost = parent_size +
                  child_size * (1.0 + 0.08 * width + 0.02 * role_width) +
                  0.05 * child_size *
                      std::log2(std::max(2.0, child_size));

    const auto added = child & ~parent;
    if (parent != 0 && added != 0 && (added & (added - 1)) == 0) {
      const std::size_t added_index =
          static_cast<std::size_t>(boost::multiprecision::lsb(added));
      std::unordered_set<RoleId> parent_roles;
      for (const auto &variable : parent_scope) {
        const auto &roles = input_by_id.at(variable)->pattern_roles;
        parent_roles.insert(roles.begin(), roles.end());
      }
      std::size_t shared = 0;
      for (const auto &role : variables[added_index].pattern_roles) {
        shared += parent_roles.count(role);
      }
      if (shared == 0) {
        cost += 1000.0 * child_size;
      } else if (shared == 1) {
        cost += 8.0 * child_size;
      } else {
        cost += 4.0 * child_size / static_cast<double>(shared);
      }
    }
    return std::min(1.0e19, cost);
  };

  std::unordered_map<std::string, std::vector<Scope>> tree_target_cache;
  std::function<const std::vector<Scope> &(const Scope &)> tree_targets =
      [&](const Scope &raw_tree) -> const std::vector<Scope> & {
        const auto tree = canonicalScope(raw_tree, rank);
        const auto key = scopeKey(tree);
        const auto cached = tree_target_cache.find(key);
        if (cached != tree_target_cache.end()) {
          return cached->second;
        }
        std::map<std::string, Scope> targets;
        auto add_if_contained = [&](const Scope &scope) {
          if (containsScope(tree, scope)) {
            const auto canonical = canonicalScope(scope, rank);
            targets.emplace(scopeKey(canonical), canonical);
          }
        };
        for (const auto &variable : tree) {
          add_if_contained({variable});
        }
        for (const auto &[key_unused, scope] : unique_atomic) {
          (void)key_unused;
          add_if_contained(scope);
        }
        for (const auto &[key_unused, scope] : unique_query_scopes) {
          (void)key_unused;
          add_if_contained(scope);
        }
        for (const auto &[key_unused, scope] : oversized_query_scopes) {
          (void)key_unused;
          add_if_contained(scope);
        }

        if (tree.size() < 20) {
          const std::uint64_t limit = std::uint64_t{1} << tree.size();
          for (std::uint64_t local = 1; local < limit; ++local) {
            if (__builtin_popcountll(local) < 2) {
              continue;
            }
            Scope candidate;
            whatif::CausalMask candidate_mask = 0;
            for (std::size_t bit = 0; bit < tree.size(); ++bit) {
              if ((local & (std::uint64_t{1} << bit)) == 0) {
                continue;
              }
              candidate.push_back(tree[bit]);
              candidate_mask |= whatif::CausalMask{1}
                                << rank.at(tree[bit]);
            }
            bool valid = false;
            for (std::size_t treatment = 0;
                 treatment < variables.size() && !valid; ++treatment) {
              const auto treatment_mask =
                  whatif::CausalMask{1} << treatment;
              if ((candidate_mask & treatment_mask) == 0) {
                continue;
              }
              for (std::size_t outcome = 0; outcome < variables.size();
                   ++outcome) {
                const auto outcome_mask =
                    whatif::CausalMask{1} << outcome;
                if (treatment == outcome ||
                    (candidate_mask & outcome_mask) == 0 ||
                    !dag.reaches(treatment_mask, outcome_mask)) {
                  continue;
                }
                const auto path =
                    dag.causalPathClosure(treatment_mask, outcome_mask);
                if ((path & ~candidate_mask) != 0) {
                  continue;
                }
                const auto adjustment = candidate_mask & ~path;
                if (dag.reduceConditioningUnderIntervention(
                        treatment_mask, outcome_mask, adjustment) ==
                        adjustment &&
                    dag.isValidAdjustment(treatment_mask, outcome_mask,
                                          adjustment)) {
                  valid = true;
                  break;
                }
              }
            }
            if (valid) {
              add_if_contained(candidate);
            }
          }
        }

        std::vector<Scope> ordered_targets;
        ordered_targets.reserve(targets.size());
        for (auto &[target_key, target] : targets) {
          (void)target_key;
          ordered_targets.push_back(std::move(target));
        }
        return tree_target_cache.emplace(key, std::move(ordered_targets))
            .first->second;
      };

  auto choose_cost_order = [&](const Scope &tree,
                               const std::vector<Scope> &targets) {
    std::vector<whatif::CausalMask> target_masks;
    target_masks.reserve(targets.size());
    for (const auto &target : targets) {
      target_masks.push_back(mask_of_scope(target));
    }
    std::vector<std::size_t> members;
    members.reserve(tree.size());
    for (const auto &variable : tree) {
      members.push_back(rank.at(variable));
    }
    auto incremental = [&](whatif::CausalMask placed,
                           std::size_t candidate, bool amortized) {
      const auto bit = whatif::CausalMask{1} << candidate;
      std::map<whatif::CausalMask, std::size_t> groups;
      for (const auto &target : target_masks) {
        if ((target & bit) != 0) {
          groups[target & placed] += 1;
        }
      }
      double cost = 0.0;
      for (const auto &[parent, coverage] : groups) {
        const double edge = edge_cost(parent, parent | bit);
        cost += amortized
                    ? edge / static_cast<double>(std::max<std::size_t>(
                                 1, coverage))
                    : edge;
      }
      return cost;
    };

    std::vector<std::size_t> order;
    whatif::CausalMask placed = 0;
    while (order.size() < members.size()) {
      std::optional<std::size_t> best;
      double best_score = std::numeric_limits<double>::infinity();
      double best_mass = std::numeric_limits<double>::infinity();
      for (const auto candidate : members) {
        const auto bit = whatif::CausalMask{1} << candidate;
        if ((placed & bit) != 0) {
          continue;
        }
        const double current = incremental(placed, candidate, true);
        double lookahead = 0.0;
        std::size_t future_count = 0;
        for (const auto future : members) {
          const auto future_bit = whatif::CausalMask{1} << future;
          if (future == candidate || (placed & future_bit) != 0) {
            continue;
          }
          lookahead += incremental(placed | bit, future, true);
          future_count += 1;
        }
        if (future_count != 0) {
          lookahead /= static_cast<double>(future_count);
        }
        const double score = current + lookahead;
        const double mass = incremental(placed, candidate, false);
        if (!best || score < best_score - 1e-9 ||
            (std::abs(score - best_score) < 1e-9 &&
             (mass < best_mass - 1e-9 ||
              (std::abs(mass - best_mass) < 1e-9 &&
               variables[candidate].id < variables[*best].id)))) {
          best = candidate;
          best_score = score;
          best_mass = mass;
        }
      }
      if (!best) {
        throw std::logic_error("failed to choose Parent Tree cost order");
      }
      order.push_back(*best);
      placed |= whatif::CausalMask{1} << *best;
    }
    return order;
  };

  std::unordered_map<std::string, double> tree_cost_cache;
  std::function<double(const Scope &)> estimated_tree_cost =
      [&](const Scope &raw_tree) {
        const auto tree = canonicalScope(raw_tree, rank);
        const auto key = scopeKey(tree);
        const auto cached = tree_cost_cache.find(key);
        if (cached != tree_cost_cache.end()) {
          return cached->second;
        }
        const auto &targets = tree_targets(tree);
        const auto order = choose_cost_order(tree, targets);
        std::vector<std::size_t> order_rank(variables.size(), variables.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
          order_rank[order[i]] = i;
        }
        std::set<whatif::CausalMask> prefix_edges;
        for (const auto &target : targets) {
          std::vector<std::size_t> ordered;
          for (const auto &variable : target) {
            ordered.push_back(rank.at(variable));
          }
          std::sort(ordered.begin(), ordered.end(), [&](auto left, auto right) {
            return order_rank[left] < order_rank[right];
          });
          whatif::CausalMask prefix = 0;
          for (const auto variable : ordered) {
            prefix |= whatif::CausalMask{1} << variable;
            prefix_edges.insert(prefix);
          }
        }
        double cost = 0.0;
        for (const auto &child : prefix_edges) {
          std::size_t last = variables.size();
          for (std::size_t variable = 0; variable < variables.size();
               ++variable) {
            if ((child & (whatif::CausalMask{1} << variable)) != 0 &&
                (last == variables.size() ||
                 order_rank[variable] > order_rank[last])) {
              last = variable;
            }
          }
          if (last == variables.size()) {
            continue;
          }
          const auto parent =
              child & ~(whatif::CausalMask{1} << last);
          cost += edge_cost(parent, child);
        }
        for (const auto &variable : tree) {
          const auto &stats = variable_cost_stats.at(variable);
          cost += stats.instances *
                  (0.25 + 0.05 * static_cast<double>(stats.role_count));
        }
        cost = std::min(1.0e20, cost);
        tree_cost_cache.emplace(key, cost);
        return cost;
      };

  // Perform logical subset absorption before enforcing the physical width.
  // This implements Ci subset Cj independently of K.
  std::vector<Scope> logical_clusters;
  logical_clusters.reserve(unique_atomic.size());
  for (const auto &entry : unique_atomic) {
    logical_clusters.push_back(entry.second);
    if (entry.second.size() > maximum_width) {
      result.stats.oversized_atomic_factors += 1;
    }
  }
  while (true) {
    bool absorbed = false;
    for (std::size_t subset = 0; subset < logical_clusters.size() && !absorbed;
         ++subset) {
      for (std::size_t superset = 0; superset < logical_clusters.size();
           ++superset) {
        if (subset == superset ||
            logical_clusters[subset].size() >
                logical_clusters[superset].size() ||
            !containsScope(logical_clusters[superset],
                           logical_clusters[subset])) {
          continue;
        }
        logical_clusters.erase(logical_clusters.begin() +
                               static_cast<std::ptrdiff_t>(subset));
        result.stats.subset_absorptions += 1;
        absorbed = true;
        break;
      }
    }
    if (!absorbed) {
      break;
    }
  }

  std::vector<Scope> clusters;
  std::vector<std::size_t> covered_factors;
  for (const auto &scope : logical_clusters) {
    clusters.push_back(scope);
    covered_factors.push_back(1);
    if (scope.size() > maximum_width) {
      result.stats.preserved_oversized_trees += 1;
    }
  }

  // Query scopes are physical co-location requirements, not extra factors.
  // Keeping only inclusion-maximal seeds preserves coverage while avoiding a
  // separate Tree for every shorter path prefix.
  auto query_seeds = maximalScopes(unique_query_scopes);
  result.stats.maximal_query_seeds = query_seeds.size();
  for (auto &scope : query_seeds) {
    clusters.push_back(std::move(scope));
    covered_factors.push_back(0);
  }

  // A contained scope adds no physical Tree. Oversized atomic factors are
  // indivisible containers and may absorb their local query seeds, but they
  // never participate in bounded unions.
  while (true) {
    bool absorbed = false;
    for (std::size_t subset = 0; subset < clusters.size() && !absorbed;
         ++subset) {
      for (std::size_t superset = 0; superset < clusters.size(); ++superset) {
        if (subset == superset ||
            clusters[subset].size() > clusters[superset].size() ||
            !containsScope(clusters[superset], clusters[subset])) {
          continue;
        }
        covered_factors[superset] += covered_factors[subset];
        clusters.erase(clusters.begin() + static_cast<std::ptrdiff_t>(subset));
        covered_factors.erase(covered_factors.begin() +
                              static_cast<std::ptrdiff_t>(subset));
        result.stats.subset_absorptions += 1;
        absorbed = true;
        break;
      }
    }
    if (!absorbed) {
      break;
    }
  }

  struct MergeCandidate {
    std::size_t left = 0;
    std::size_t right = 0;
    Scope combined;
    double before_cost = 0.0;
    double after_cost = 0.0;
    double gain = 0.0;
    double relative_gain = 0.0;
    std::size_t covered = 0;
    std::size_t overlap = 0;
    std::size_t role_overlap = 0;
    std::size_t causal_affinity = 0;
  };
  for (const auto &cluster : clusters) {
    result.stats.estimated_initial_tree_cost += estimated_tree_cost(cluster);
  }
  while (true) {
    bool found = false;
    MergeCandidate best;
    for (std::size_t left = 0; left < clusters.size(); ++left) {
      for (std::size_t right = left + 1; right < clusters.size(); ++right) {
        if (clusters[left].size() > maximum_width ||
            clusters[right].size() > maximum_width) {
          continue;
        }
        auto combined = scopeUnion(clusters[left], clusters[right], rank);
        if (combined.size() > maximum_width) {
          continue;
        }
        result.stats.cost_merge_candidates += 1;
        MergeCandidate candidate;
        candidate.left = left;
        candidate.right = right;
        candidate.combined = std::move(combined);
        candidate.before_cost = estimated_tree_cost(clusters[left]) +
                                estimated_tree_cost(clusters[right]);
        candidate.after_cost = estimated_tree_cost(candidate.combined);
        candidate.gain = candidate.before_cost - candidate.after_cost;
        candidate.relative_gain =
            candidate.before_cost <= 0.0
                ? 0.0
                : candidate.gain / candidate.before_cost;
        const double tolerance =
            1e-9 * std::max(1.0, candidate.before_cost);
        if (candidate.gain <= tolerance) {
          result.stats.cost_rejected_merges += 1;
          continue;
        }
        candidate.covered = covered_factors[left] + covered_factors[right];
        candidate.overlap = intersectionSize(clusters[left], clusters[right]);
        for (const auto &l : clusters[left]) {
          for (const auto &r : clusters[right]) {
            if (l == r) {
              continue;
            }
            candidate.role_overlap += shared_roles(l, r);
            candidate.causal_affinity += causally_adjacent(l, r);
          }
        }
        const auto score = [](const MergeCandidate &value) {
          return std::make_tuple(value.gain, value.relative_gain,
                                 value.overlap, value.role_overlap,
                                 value.causal_affinity, value.covered,
                                 -static_cast<long long>(value.combined.size()),
                                 -static_cast<long long>(value.left),
                                 -static_cast<long long>(value.right));
        };
        if (!found || score(candidate) > score(best)) {
          found = true;
          best = std::move(candidate);
        }
      }
    }
    if (!found) {
      break;
    }
    clusters[best.left] = std::move(best.combined);
    covered_factors[best.left] += covered_factors[best.right];
    clusters.erase(clusters.begin() + static_cast<std::ptrdiff_t>(best.right));
    covered_factors.erase(covered_factors.begin() +
                          static_cast<std::ptrdiff_t>(best.right));
    result.stats.bounded_merges += 1;
  }
  for (const auto &cluster : clusters) {
    result.stats.estimated_final_tree_cost += estimated_tree_cost(cluster);
  }

  std::sort(clusters.begin(), clusters.end(),
            [&](const auto &left, const auto &right) {
              if (left.size() != right.size()) {
                return left.size() < right.size();
              }
              return scopeKey(left) < scopeKey(right);
            });
  result.trees.reserve(clusters.size());
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    if (clusters[i].empty()) {
      throw std::logic_error("Parent Tree planner produced an empty Tree");
    }
    if (clusters[i].size() > maximum_width) {
      const auto atomic = unique_atomic.find(scopeKey(clusters[i]));
      if (atomic == unique_atomic.end() ||
          atomic->second.size() <= maximum_width) {
        throw std::logic_error(
            "only an original oversized parent factor may exceed K");
      }
    }
    result.trees.push_back({"mpc" + std::to_string(i), std::move(clusters[i])});
  }

  result.stats.localizable_query_scopes = 0;
  result.stats.oversized_query_scopes = 0;
  result.stats.covered_localizable_query_scopes = 0;
  auto classify_query_scopes = [&](const std::map<std::string, Scope> &scopes) {
    for (const auto &entry : scopes) {
      const bool covered = std::any_of(
          result.trees.begin(), result.trees.end(), [&](const auto &tree) {
            return containsScope(tree.variables, entry.second);
          });
      if (covered) {
        result.stats.localizable_query_scopes += 1;
        result.stats.covered_localizable_query_scopes += 1;
      } else {
        result.stats.oversized_query_scopes += 1;
      }
    }
  };
  classify_query_scopes(unique_query_scopes);
  classify_query_scopes(oversized_query_scopes);

  std::map<std::string, Scope> boundary_scopes;
  for (const auto &entry : oversized_query_scopes) {
    const bool local = std::any_of(
        result.trees.begin(), result.trees.end(), [&](const auto &tree) {
          return containsScope(tree.variables, entry.second);
        });
    if (!local) {
      boundary_scopes.emplace(entry);
    }
  }
  for (const auto &entry : unique_atomic) {
    const bool local = std::any_of(
        result.trees.begin(), result.trees.end(), [&](const auto &tree) {
          return containsScope(tree.variables, entry.second);
        });
    if (!local) {
      boundary_scopes.emplace(entry.first, entry.second);
    }
  }
  for (const auto &[key_unused, scope] : boundary_scopes) {
    (void)key_unused;
    const double width = static_cast<double>(scope.size());
    result.stats.estimated_boundary_cost +=
        estimated_cardinality(scope) * (1.0 + 0.15 * width);
  }
  result.stats.estimated_total_cost =
      result.stats.estimated_final_tree_cost +
      result.stats.estimated_boundary_cost;
  return result;
}

} // namespace whatif
