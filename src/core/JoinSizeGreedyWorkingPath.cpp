#include "whatif/core/FullLatticeIndex.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace whatif::core {

void FullLatticeIndex::executeJoinSizeGreedyWorkingPath() {
  struct ScheduledTask {
    Mask source = 0;
    std::size_t reuse_depth = 0;
    std::vector<Mask> path;
  };

  auto greedy_path_from = [&](Mask prefix, Mask source) {
    std::vector<Mask> suffix;
    while (prefix != source) {
      std::optional<std::size_t> best_variable;
      double best_size = std::numeric_limits<double>::infinity();
      for (std::size_t variable = 0; variable < data_.size(); ++variable) {
        const Mask bit = Mask{1} << variable;
        if ((source & bit) == 0 || (prefix & bit) != 0) {
          continue;
        }
        const Mask child = prefix | bit;
        const double size = plan_.estimates.totals[child];
        if (!best_variable || size < best_size - 1e-9 ||
            (std::abs(size - best_size) < 1e-9 &&
             data_[variable].id < data_[*best_variable].id)) {
          best_variable = variable;
          best_size = size;
        }
      }
      if (!best_variable) {
        throw std::runtime_error("join-size greedy could not extend task");
      }
      prefix |= Mask{1} << *best_variable;
      suffix.push_back(prefix);
    }
    return suffix;
  };

  std::set<Mask> remaining(plan_.target_masks.begin(), plan_.target_masks.end());
  std::vector<Mask> current_path;
  std::vector<ScheduledTask> schedule;
  std::unordered_set<Mask> previously_built;
  result_.task_order.clear();

  while (!remaining.empty()) {
    Mask selected = 0;
    std::size_t selected_reuse_depth = 0;
    double selected_cost = std::numeric_limits<double>::infinity();
    std::vector<Mask> selected_path;

    for (const Mask source : remaining) {
      std::size_t reuse_depth = 0;
      while (reuse_depth < current_path.size() &&
             (current_path[reuse_depth] & ~source) == 0) {
        reuse_depth += 1;
      }
      const Mask reusable_prefix =
          reuse_depth == 0 ? 0 : current_path[reuse_depth - 1];
      auto suffix = greedy_path_from(reusable_prefix, source);
      double cost = 0.0;
      for (const Mask child : suffix) {
        cost += plan_.estimates.totals[child];
      }
      if (selected == 0 || cost < selected_cost - 1e-9 ||
          (std::abs(cost - selected_cost) < 1e-9 &&
           (reuse_depth > selected_reuse_depth ||
            (reuse_depth == selected_reuse_depth && source < selected)))) {
        selected = source;
        selected_reuse_depth = reuse_depth;
        selected_cost = cost;
        selected_path.assign(
            current_path.begin(),
            current_path.begin() + static_cast<std::ptrdiff_t>(reuse_depth));
        selected_path.insert(selected_path.end(), suffix.begin(), suffix.end());
      }
    }

    schedule.push_back({selected, selected_reuse_depth, selected_path});
    result_.cache_stats.working_path_tasks += 1;
    if (selected_reuse_depth != 0) {
      result_.cache_stats.working_path_reuse_transitions += 1;
      result_.cache_stats.working_path_reused_prefix_levels +=
          selected_reuse_depth;
    }
    for (auto it = selected_path.begin() +
                   static_cast<std::ptrdiff_t>(selected_reuse_depth);
         it != selected_path.end(); ++it) {
      result_.cache_stats.working_path_join_steps += 1;
      if (!previously_built.insert(*it).second) {
        result_.cache_stats.working_path_rebuilt_steps += 1;
      }
    }
    result_.task_order.insert(
        result_.task_order.end(),
        selected_path.begin() +
            static_cast<std::ptrdiff_t>(selected_reuse_depth),
        selected_path.end());
    current_path = std::move(selected_path);
    remaining.erase(selected);
  }

  current_path.clear();
  for (const auto& task : schedule) {
    while (current_path.size() > task.reuse_depth) {
      releaseResident(current_path.back());
      current_path.pop_back();
    }

    Mask parent = current_path.empty() ? 0 : current_path.back();
    if (task.reuse_depth == task.path.size()) {
      const auto found = resident_.find(parent);
      if (found == resident_.end()) {
        throw std::runtime_error("live greedy prefix is not resident");
      }
      aggregateSourceOutputs(task.source, found->second.rows);
      continue;
    }

    for (std::size_t depth = task.reuse_depth; depth < task.path.size();
         ++depth) {
      const Mask child = task.path[depth];
      const Mask added = child & ~parent;
      if (added == 0 || (added & (added - 1)) != 0) {
        throw std::runtime_error(
            "join-size greedy path must add one variable");
      }
      const std::size_t variable =
          static_cast<std::size_t>(__builtin_ctzll(added));
      const auto parent_found = resident_.find(parent);
      if (parent_found == resident_.end()) {
        throw std::runtime_error(
            "join-size greedy parent left working path");
      }
      PackedSupport child_support =
          parent == 0
              ? singletonSupport(variable)
              : extendSupport(parent_found->second.rows, parent, variable,
                              nullptr);
      const Count weight = supportWeight(child_support);
      result_.emitted_rows += weight;
      if (child == task.source) {
        aggregateSourceOutputs(task.source, child_support);
      }
      reportProgress(child, parent, variable, depth + 1, weight);
      putResident(child, std::move(child_support));
      current_path.push_back(child);
      parent = child;
    }
  }

  while (!current_path.empty()) {
    releaseResident(current_path.back());
    current_path.pop_back();
  }
}

}  // namespace whatif::core
