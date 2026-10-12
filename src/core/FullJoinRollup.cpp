#include "whatif/core/FullLatticeIndex.h"

#include <algorithm>
#include <deque>
#include <numeric>
#include <set>
#include <stdexcept>

namespace whatif::core {

void FullLatticeIndex::streamRollupCubes(const std::function<void(
    Mask, std::map<std::vector<std::string>, Count>&&)>& emit) {
  if (!emit || config_.execution_schedule != ExecutionSchedule::IndependentSources ||
      config_.variable_order_policy != VariableOrderPolicy::Alphabetical ||
      config_.enable_value_partition_join || !config_.target_scopes.empty() ||
      !config_.source_output_scopes.empty() || !config_.binding_output_scopes.empty() ||
      config_.store_all_supports || config_.store_participation ||
      config_.store_derive_results || config_.collect_partition_traces) {
    throw std::runtime_error("rollup requires independent alphabetical full cubes without auxiliary outputs");
  }
  config_.preserve_instance_identity = true;
  plan_ = {};
  result_ = {};
  normalizeInputs();
  buildPostingsAndStats();
  const auto n = data_.size(), r = global_roles_.size();
  auto& order = plan_.variable_order_indices;
  order.resize(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](auto a, auto b) { return data_[a].id < data_[b].id; });
  rank_of_variable_.resize(n);
  for (std::size_t rank = 0; rank < n; ++rank) {
    rank_of_variable_[order[rank]] = rank;
    plan_.variable_order.push_back(data_[order[rank]].id);
  }
  RollupStats stats;
  std::map<Mask, PackedSupport> donors;
  std::deque<Mask> fifo;
  std::size_t cache_bytes = 0;
  const auto bytes = [](const PackedSupport& s) {
    return sizeof(PackedSupport) + 128 + s.instance_ids.capacity() * sizeof(InstanceId) +
        (s.value_codes.capacity() + s.role_node_codes.capacity()) * sizeof(std::uint32_t) +
        s.counts.capacity() * sizeof(Count);
  };
  const auto keyOf = [&](const PackedSupport& s, std::size_t row,
                         const std::vector<std::size_t>& roles) {
    std::string key;
    for (auto role : roles) {
      const auto code = s.role_node_codes[row * r + role];
      key.append(reinterpret_cast<const char*>(&code), sizeof(code));
    }
    return key;
  };
  const auto append = [&](PackedSupport& out, const PackedSupport& in,
                          std::size_t row, Mask mask,
                          const std::vector<bool>& kept_roles) {
    if (rowWeight(in, row) != 1)
      throw std::runtime_error("rollup donor lost instance identities");
    for (std::size_t v = 0; v < n; ++v) {
      out.instance_ids.push_back(mask & (Mask{1} << v) ? in.instance_ids[row*n+v] : kNoInstance);
      out.value_codes.push_back(mask & (Mask{1} << v) ? in.value_codes[row*n+v] : 0);
    }
    for (std::size_t role = 0; role < r; ++role)
      out.role_node_codes.push_back(kept_roles[role] ? in.role_node_codes[row*r+role] : 0);
    out.counts.push_back(1);
    ++out.row_count;
  };

  const auto materialize = [&](Mask target) {
    std::vector<bool> kept_roles(r, false);
    for (const auto& role : rolesInMask(target)) kept_roles[global_role_index_.at(role)] = true;
    auto chosen = donors.end();
    std::size_t donor_variable = 0;
    if (__builtin_popcountll(target) >= 2) {
      for (std::size_t v = 0; v < n; ++v) {
        if (target & (Mask{1} << v)) continue;
        auto found = donors.find(target | (Mask{1} << v));
        if (found != donors.end() && (chosen == donors.end() ||
            found->second.row_count < chosen->second.row_count)) {
          chosen = found;
          donor_variable = v;
        }
      }
    }
    std::vector<std::size_t> key_roles;
    std::unordered_set<std::string> safe_keys;
    if (chosen != donors.end()) {
      for (const auto& role : sharedRoles(target, donor_variable))
        key_roles.push_back(global_role_index_.at(role));
      const std::set<std::size_t> shared(key_roles.begin(), key_roles.end());
      const auto private_roles = std::count(kept_roles.begin(), kept_roles.end(), true) - key_roles.size();
      const auto threshold = private_roles + 1;
      std::unordered_map<std::string, std::vector<std::set<std::uint32_t>>> witnesses;
      // p target-private vertices can block at most p disjoint witnesses.
      // Keys with p+1 witnesses therefore have complete donor coverage.
      for (const auto& instance : data_[donor_variable].instances) {
        ++stats.witness_rows_scanned;
        std::string key;
        for (auto role : key_roles) {
          const auto code = std::find_if(instance.role_node_codes.begin(), instance.role_node_codes.end(),
                                        [&](const auto& item) { return item.first == role; })->second;
          key.append(reinterpret_cast<const char*>(&code), sizeof(code));
        }
        if (safe_keys.count(key)) continue;
        std::set<std::uint32_t> private_nodes;
        for (const auto& item : instance.role_node_codes)
          if (!shared.count(item.first)) private_nodes.insert(item.second);
        auto& sets = witnesses[key];
        bool disjoint = true;
        for (const auto& old : sets)
          for (auto node : private_nodes) if (old.count(node)) disjoint = false;
        if (disjoint) sets.push_back(std::move(private_nodes));
        if (sets.size() >= threshold) safe_keys.insert(key);
      }
    }
    PackedSupport output;
    if (!safe_keys.empty()) {
      ++stats.used;
      std::unordered_set<std::string> seen;
      const auto& source = chosen->second;
      for (std::size_t row = 0; row < source.row_count; ++row) {
        ++stats.donor_rows_scanned;
        if (!safe_keys.count(keyOf(source, row, key_roles))) continue;
        std::string identity;
        for (std::size_t v = 0; v < n; ++v) if (target & (Mask{1} << v)) {
          const auto id = source.instance_ids[row*n+v];
          identity.append(reinterpret_cast<const char*>(&id), sizeof(id));
        }
        if (seen.insert(identity).second) append(output, source, row, target, kept_roles);
      }
      stats.projected_rows += output.row_count;
    }

    auto support = rootSupport();
    Mask prefix = 0;
    bool filtered = safe_keys.empty();
    std::vector<bool> bound_roles(r, false);
    const auto excludeCovered = [&] {
      if (filtered || !std::all_of(key_roles.begin(), key_roles.end(),
                                   [&](auto role) { return bound_roles[role]; })) return;
      PackedSupport residual;
      for (std::size_t row = 0; row < support.row_count; ++row)
        if (!safe_keys.count(keyOf(support, row, key_roles)))
          append(residual, support, row, prefix, bound_roles);
      support = std::move(residual);
      filtered = true;
    };
    excludeCovered();
    for (auto variable : order) {
      const Mask bit = Mask{1} << variable;
      if (!(target & bit)) continue;
      // Empty residuals need no further join; no target cube is pruned.
      if (support.row_count == 0) break;
      support = prefix == 0 ? singletonSupport(variable)
                            : extendSupport(support, prefix, variable, nullptr);
      prefix |= bit;
      for (const auto& role : data_[variable].pattern_roles) bound_roles[global_role_index_.at(role)] = true;
      excludeCovered();
    }
    if (!safe_keys.empty()) {
      stats.residual_rows += support.row_count;
      for (std::size_t row = 0; row < support.row_count; ++row)
        append(output, support, row, target, kept_roles);
    } else {
      output = std::move(support);
    }
    std::map<std::vector<std::string>, Count> counts;
    for (std::size_t row = 0; row < output.row_count; ++row)
      counts[valueKey(output, row, target)] += rowWeight(output, row);
    const auto cube_rows = supportWeight(output);
    result_.emitted_rows += cube_rows;
    emit(target, std::move(counts));
    const auto size = bytes(output);
    if (__builtin_popcountll(target) >= 3 && output.row_count > 0 && size <= config_.rollup_cache_bytes) {
      while (cache_bytes > config_.rollup_cache_bytes - size && !fifo.empty()) {
        cache_bytes -= bytes(donors.at(fifo.front()));
        donors.erase(fifo.front());
        fifo.pop_front();
      }
      cache_bytes += size;
      fifo.push_back(target);
      donors.emplace(target, std::move(output));
      stats.peak_cache_bytes = std::max(stats.peak_cache_bytes, cache_bytes);
    }
    ++result_.completed_tasks;
    const auto every = std::max<std::size_t>(1, config_.progress_every_tasks);
    if (result_.completed_tasks == 1 || result_.completed_tasks % every == 0 ||
        result_.completed_tasks == allVariablesMask()) {
      if (config_.progress_callback) {
        ProgressEvent event;
        event.completed_tasks = result_.completed_tasks;
        event.total_tasks = allVariablesMask();
        event.mask = target;
        event.depth = __builtin_popcountll(target);
        event.rows = cube_rows;
        event.emitted_rows = result_.emitted_rows;
        config_.progress_callback(event);
      }
      if (config_.rollup_callback) config_.rollup_callback(stats);
    }
  };
  // DFS emits earlier subtrees before later ones, making superset donors
  // available. Each target still starts its own join, without prefix reuse.
  std::function<void(Mask, std::size_t)> visit = [&](Mask prefix, std::size_t first) {
    for (std::size_t rank = first; rank < n; ++rank) {
      const Mask target = prefix | (Mask{1} << order[rank]);
      materialize(target);
      visit(target, rank + 1);
    }
  };
  visit(0, 0);
}

}  // namespace whatif::core
