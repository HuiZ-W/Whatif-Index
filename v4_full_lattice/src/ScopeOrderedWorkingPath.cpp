#include "causal_index_v4/FullLatticeIndex.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace causal_index_v4 {
namespace {

std::size_t commonPrefixLength(const std::vector<Mask>& left,
                               const std::vector<Mask>& right) {
  std::size_t depth = 0;
  while (depth < left.size() && depth < right.size() &&
         left[depth] == right[depth]) {
    ++depth;
  }
  return depth;
}

}  // namespace

void FullLatticeIndex::executeScopeOrderedWorkingPath() {
  struct ScheduledScope {
    Mask source = 0;
    std::size_t reuse_depth = 0;
    std::size_t retain_depth = 0;
    std::vector<Mask> path;
  };

  if (plan_.scope_order.size() != plan_.target_masks.size()) {
    throw std::runtime_error(
        "scope-ordered execution requires one schedule entry per source");
  }

  std::vector<std::vector<Mask>> paths;
  paths.reserve(plan_.scope_order.size());
  for (const Mask source : plan_.scope_order) {
    paths.push_back(prefixPath(source));
  }

  std::vector<ScheduledScope> schedule;
  schedule.reserve(plan_.scope_order.size());
  std::unordered_set<Mask> previously_built;
  result_.task_order.clear();

  for (std::size_t i = 0; i < plan_.scope_order.size(); ++i) {
    const std::size_t reuse_depth =
        i == 0 ? 0 : commonPrefixLength(paths[i - 1], paths[i]);
    const std::size_t retain_depth =
        i + 1 == paths.size() ? 0
                              : commonPrefixLength(paths[i], paths[i + 1]);
    schedule.push_back(
        {plan_.scope_order[i], reuse_depth, retain_depth, paths[i]});

    result_.cache_stats.working_path_tasks += 1;
    if (reuse_depth != 0) {
      result_.cache_stats.working_path_reuse_transitions += 1;
      result_.cache_stats.working_path_reused_prefix_levels += reuse_depth;
    }
    for (std::size_t depth = reuse_depth; depth < paths[i].size(); ++depth) {
      const Mask child = paths[i][depth];
      result_.cache_stats.working_path_join_steps += 1;
      if (!previously_built.insert(child).second) {
        result_.cache_stats.working_path_rebuilt_steps += 1;
      }
      result_.task_order.push_back(child);
    }
  }

  std::vector<Mask> current_path;
  for (const auto& task : schedule) {
    if (current_path.size() != task.reuse_depth) {
      throw std::runtime_error(
          "scope-ordered working path does not match planned reuse depth");
    }

    if (task.reuse_depth == task.path.size()) {
      const auto found = resident_.find(task.source);
      if (found == resident_.end()) {
        throw std::runtime_error("reusable scope prefix is not resident");
      }
      aggregateSourceOutputs(task.source, found->second.rows);
    } else {
      Mask parent = current_path.empty() ? 0 : current_path.back();
      for (std::size_t depth = task.reuse_depth; depth < task.path.size();
           ++depth) {
        const Mask child = task.path[depth];
        const Mask added = child & ~parent;
        if (added == 0 || (added & (added - 1)) != 0) {
          throw std::runtime_error(
              "scope-ordered path must add exactly one variable");
        }
        const std::size_t variable =
            static_cast<std::size_t>(__builtin_ctzll(added));
        const auto parent_found = resident_.find(parent);
        if (parent_found == resident_.end()) {
          throw std::runtime_error(
              "scope-ordered parent left the current working path");
        }

        const bool final = depth + 1 == task.path.size();
        const bool final_needed_by_next =
            final && task.retain_depth == task.path.size();
        const bool capture_participation =
            config_.store_participation && parent != 0;
        const bool can_push_leaf =
            final && !final_needed_by_next && !capture_participation &&
            !config_.store_all_supports && !isDerivePinned(child) &&
            !config_.store_derive_results &&
            !config_.collect_partition_traces &&
            !requiresBindingOutput(child);

        if (can_push_leaf) {
          const auto& outputs =
              output_masks_by_source_[static_cast<std::size_t>(child)];
          const Count weight = aggregateLeafValueCounts(
              parent_found->second.rows, parent, variable, child, outputs);
          for (const auto output : outputs) {
            result_.exact_counts[static_cast<std::size_t>(output)] = weight;
          }
          result_.emitted_rows += weight;
          reportProgress(child, parent, variable, depth + 1, weight);
          continue;
        }

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
            has_partner.assign(
                parent_found->second.rows.row_count,
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
          aggregateSourceOutputs(task.source, child_support);
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
        current_path.push_back(child);
        parent = child;
      }
    }

    while (current_path.size() > task.retain_depth) {
      releaseResident(current_path.back());
      current_path.pop_back();
    }
  }

  while (!current_path.empty()) {
    releaseResident(current_path.back());
    current_path.pop_back();
  }
}

}  // namespace causal_index_v4
