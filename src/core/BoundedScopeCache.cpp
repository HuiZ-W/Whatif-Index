#include "whatif/core/FullLatticeIndex.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace whatif::core {

void FullLatticeIndex::executeBoundedScopeCache() {
  if (config_.result_cache_capacity == 0) {
    throw std::runtime_error(
        "bounded scope cache requires result_cache_capacity > 0");
  }
  if (config_.cache_budget_tuples != 0) {
    throw std::runtime_error(
        "bounded scope cache cannot be combined with cache_budget_tuples");
  }
  if (plan_.scope_order.size() != plan_.target_masks.size()) {
    throw std::runtime_error(
        "bounded scope cache requires one schedule entry per source");
  }

  const std::size_t no_future = std::numeric_limits<std::size_t>::max();
  std::vector<std::vector<Mask>> paths;
  paths.reserve(plan_.scope_order.size());
  std::unordered_map<Mask, std::vector<std::size_t>> use_positions;
  for (std::size_t task = 0; task < plan_.scope_order.size(); ++task) {
    paths.push_back(prefixPath(plan_.scope_order[task]));
    for (const Mask prefix : paths.back()) {
      use_positions[prefix].push_back(task);
    }
  }

  auto next_use = [&](Mask mask, std::size_t after_task) {
    const auto found = use_positions.find(mask);
    if (found == use_positions.end()) {
      return no_future;
    }
    const auto next = std::upper_bound(
        found->second.begin(), found->second.end(), after_task);
    return next == found->second.end() ? no_future : *next;
  };

  auto choose_victim =
      [&](const std::vector<std::pair<Mask, std::size_t>>& entries,
          std::size_t after_task,
          std::optional<Mask> protected_mask) -> std::optional<Mask> {
    std::optional<Mask> victim;
    std::size_t victim_last_used = 0;
    std::size_t victim_next_use = 0;
    for (const auto& entry : entries) {
      const Mask mask = entry.first;
      const std::size_t last_used = entry.second;
      if (mask == 0 || (protected_mask && mask == *protected_mask)) {
        continue;
      }
      const std::size_t candidate_next_use = next_use(mask, after_task);
      bool replace = !victim;
      if (!replace &&
          config_.result_cache_eviction_policy ==
              ResultCacheEvictionPolicy::Lru) {
        replace =
            last_used < victim_last_used ||
            (last_used == victim_last_used && mask < *victim);
      } else if (!replace) {
        replace =
            candidate_next_use > victim_next_use ||
            (candidate_next_use == victim_next_use &&
             (last_used < victim_last_used ||
              (last_used == victim_last_used && mask < *victim)));
      }
      if (replace) {
        victim = mask;
        victim_last_used = last_used;
        victim_next_use = candidate_next_use;
      }
    }
    return victim;
  };

  const bool leaf_pushdown_allowed =
      !config_.store_all_supports && !config_.store_participation &&
      !config_.store_derive_results && !config_.collect_partition_traces;

  // Simulate the equal-slot cache first so progress reports know the exact
  // number of joins, including suffixes rebuilt after an eviction.
  std::unordered_map<Mask, std::size_t> simulated_cache;
  std::size_t simulated_clock = 0;
  std::vector<Mask> scheduled_builds;
  for (std::size_t task = 0; task < paths.size(); ++task) {
    const auto& path = paths[task];
    std::size_t reuse_depth = 0;
    for (std::size_t depth = 0; depth < path.size(); ++depth) {
      if (simulated_cache.count(path[depth]) != 0) {
        reuse_depth = depth + 1;
      }
    }
    if (reuse_depth != 0) {
      simulated_cache[path[reuse_depth - 1]] = ++simulated_clock;
    }

    for (std::size_t depth = reuse_depth; depth < path.size(); ++depth) {
      const Mask child = path[depth];
      const bool final = depth + 1 == path.size();
      const bool can_push_leaf =
          final && leaf_pushdown_allowed &&
          !isDerivePinned(child) && !requiresBindingOutput(child) &&
          next_use(child, task) == no_future;
      scheduled_builds.push_back(child);
      if (can_push_leaf) {
        continue;
      }
      simulated_cache[child] = ++simulated_clock;
      while (simulated_cache.size() > config_.result_cache_capacity) {
        std::vector<std::pair<Mask, std::size_t>> entries(
            simulated_cache.begin(), simulated_cache.end());
        const auto victim = choose_victim(entries, task, child);
        if (!victim) {
          throw std::runtime_error(
              "bounded cache simulation cannot select a victim");
        }
        simulated_cache.erase(*victim);
      }
    }

    for (auto it = simulated_cache.begin(); it != simulated_cache.end();) {
      if (next_use(it->first, task) == no_future) {
        it = simulated_cache.erase(it);
      } else {
        ++it;
      }
    }
  }
  result_.task_order = std::move(scheduled_builds);

  auto resident_result_count = [&] {
    return resident_.size() - (resident_.count(0) == 0 ? 0 : 1);
  };
  auto update_result_cache_stats = [&] {
    const std::size_t count = resident_result_count();
    result_.cache_stats.resident_results = count;
    result_.cache_stats.peak_resident_results =
        std::max(result_.cache_stats.peak_resident_results, count);
    if (count > config_.result_cache_capacity) {
      result_.cache_stats.result_capacity_overflows += 1;
    }
  };
  auto erase_resident = [&](Mask mask, bool capacity_eviction) {
    const auto found = resident_.find(mask);
    if (found == resident_.end() || mask == 0) {
      return;
    }
    result_.cache_stats.resident_tuples -= found->second.rows.row_count;
    resident_.erase(found);
    if (capacity_eviction) {
      result_.cache_stats.evictions += 1;
      result_.cache_stats.result_capacity_evictions += 1;
    }
  };
  auto enforce_capacity = [&](std::size_t task,
                              std::optional<Mask> protected_mask) {
    while (resident_result_count() > config_.result_cache_capacity) {
      std::vector<std::pair<Mask, std::size_t>> entries;
      entries.reserve(resident_.size());
      for (const auto& entry : resident_) {
        if (entry.first != 0 && entry.second.active_pin_count == 0) {
          entries.push_back({entry.first, entry.second.last_used});
        }
      }
      const auto victim = choose_victim(entries, task, protected_mask);
      if (!victim) {
        result_.cache_stats.result_capacity_overflows += 1;
        break;
      }
      erase_resident(*victim, true);
    }
    update_result_cache_stats();
  };
  auto release_dead = [&](std::size_t task) {
    std::vector<Mask> dead;
    for (const auto& entry : resident_) {
      if (entry.first != 0 && entry.second.active_pin_count == 0 &&
          next_use(entry.first, task) == no_future) {
        dead.push_back(entry.first);
      }
    }
    for (const Mask mask : dead) {
      erase_resident(mask, false);
    }
    update_result_cache_stats();
  };

  std::unordered_set<Mask> previously_built;
  update_result_cache_stats();

  for (std::size_t task = 0; task < plan_.scope_order.size(); ++task) {
    const Mask source = plan_.scope_order[task];
    const auto& path = paths[task];
    result_.cache_stats.working_path_tasks += 1;

    std::size_t reuse_depth = 0;
    for (std::size_t depth = 0; depth < path.size(); ++depth) {
      if (resident_.count(path[depth]) != 0) {
        reuse_depth = depth + 1;
      }
    }
    if (reuse_depth != 0) {
      auto& entry = resident_.at(path[reuse_depth - 1]);
      entry.last_used = ++cache_clock_;
      result_.cache_stats.hits += 1;
      result_.cache_stats.non_root_hits += 1;
      result_.cache_stats.working_path_reuse_transitions += 1;
      result_.cache_stats.working_path_reused_prefix_levels += reuse_depth;
    }

    if (reuse_depth == path.size()) {
      aggregateSourceOutputs(source, resident_.at(source).rows);
      release_dead(task);
      enforce_capacity(task, std::nullopt);
      continue;
    }

    Mask parent = reuse_depth == 0 ? 0 : path[reuse_depth - 1];
    if (resident_.count(parent) == 0) {
      throw std::runtime_error("bounded cache parent is not resident");
    }
    pinActive(parent);
    bool parent_is_pinned = true;

    for (std::size_t depth = reuse_depth; depth < path.size(); ++depth) {
      const Mask child = path[depth];
      const Mask added = child & ~parent;
      if (added == 0 || (added & (added - 1)) != 0) {
        throw std::runtime_error(
            "bounded cache path must add exactly one variable");
      }
      const std::size_t variable =
          static_cast<std::size_t>(__builtin_ctzll(added));
      const bool final = depth + 1 == path.size();
      const bool can_push_leaf =
          final && leaf_pushdown_allowed && !isDerivePinned(child) &&
          !requiresBindingOutput(child) && next_use(child, task) == no_future;

      result_.cache_stats.misses += 1;
      result_.cache_stats.working_path_join_steps += 1;
      if (!previously_built.insert(child).second) {
        result_.cache_stats.recomputations += 1;
        result_.cache_stats.working_path_rebuilt_steps += 1;
      }

      auto parent_found = resident_.find(parent);
      if (parent_found == resident_.end()) {
        throw std::runtime_error(
            "bounded cache evicted the active parent");
      }

      if (can_push_leaf) {
        const auto& outputs =
            output_masks_by_source_[static_cast<std::size_t>(child)];
        const Count weight = aggregateLeafValueCounts(
            parent_found->second.rows, parent, variable, child, outputs);
        for (const Mask output : outputs) {
          result_.exact_counts[static_cast<std::size_t>(output)] = weight;
        }
        result_.emitted_rows += weight;
        reportProgress(child, parent, variable, depth + 1, weight);
        unpinActive(parent);
        parent_is_pinned = false;
        break;
      }

      const bool capture_participation =
          config_.store_participation && parent != 0;
      std::vector<std::uint8_t> has_partner;
      std::optional<DeriveResult> derive_choice;
      if (config_.enable_derive && parent != 0) {
        derive_choice =
            bestOnlineDerive(child, parent, parent_found->second.rows);
      }

      PackedSupport child_support;
      if (derive_choice &&
          derive_choice->derive_cost <= derive_choice->extend_cost) {
        derive_choice->used = true;
        child_support = packSupport(derive_choice->rows);
        if (config_.store_derive_results) {
          result_.derive_results.push_back(*derive_choice);
        }
        result_.cache_stats.derive_used += 1;
        if (capture_participation) {
          has_partner.assign(parent_found->second.rows.row_count,
                             child_support.row_count == 0 ? 0 : 1);
        }
      } else {
        child_support =
            parent == 0
                ? singletonSupport(variable)
                : extendSupport(parent_found->second.rows, parent, variable,
                                capture_participation ? &has_partner
                                                      : nullptr);
        if (derive_choice && config_.store_derive_results) {
          result_.derive_results.push_back(*derive_choice);
        }
      }

      if (capture_participation) {
        result_.participation.push_back(
            {parent, variable, std::move(has_partner)});
      }
      const Count weight = supportWeight(child_support);
      result_.emitted_rows += weight;
      if (final) {
        aggregateSourceOutputs(source, child_support);
      }
      reportProgress(child, parent, variable, depth + 1, weight);

      if (config_.store_all_supports || isDerivePinned(child)) {
        const bool first_materialization =
            result_.supports[static_cast<std::size_t>(child)].empty();
        result_.supports[static_cast<std::size_t>(child)] =
            unpackSupport(child_support);
        if (!config_.store_all_supports && first_materialization) {
          result_.cache_stats.derive_materialized_masks += 1;
          result_.cache_stats.derive_materialized_tuples +=
              result_.supports[static_cast<std::size_t>(child)].size();
        }
      }

      putResident(child, std::move(child_support));
      pinActive(child);
      unpinActive(parent);
      enforce_capacity(task, child);
      parent = child;
    }

    if (parent_is_pinned) {
      unpinActive(parent);
    }
    release_dead(task);
    enforce_capacity(task, std::nullopt);
  }

  std::vector<Mask> remaining;
  for (const auto& entry : resident_) {
    if (entry.first != 0) {
      remaining.push_back(entry.first);
    }
  }
  for (const Mask mask : remaining) {
    erase_resident(mask, false);
  }
  update_result_cache_stats();
}

}  // namespace whatif::core
