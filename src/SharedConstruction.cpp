#include "whatif/SharedConstruction.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace whatif {
namespace {
using Scope = std::vector<VariableId>;
using Rows = std::vector<BoundaryRow>;
using Support = std::shared_ptr<const Rows>;
using Identity = std::vector<InstanceId>;

Scope canonical(Scope scope) {
  std::sort(scope.begin(), scope.end());
  if (std::adjacent_find(scope.begin(), scope.end()) != scope.end())
    throw std::runtime_error("shared scope repeats a variable");
  return scope;
}
bool contains(const Scope &a, const Scope &b) {
  return std::includes(a.begin(), a.end(), b.begin(), b.end());
}
Identity identity(const BoundaryRow &row, const Scope &scope) {
  Identity key;
  for (const auto &v : scope) {
    auto p = std::lower_bound(row.bindings.begin(), row.bindings.end(), v,
        [](const auto &b, const auto &id) { return b.variable < id; });
    if (p == row.bindings.end() || p->variable != v)
      throw std::logic_error("missing variable in shared mapping");
    key.push_back(p->instance_id);
  }
  return key;
}
bool compatible(const BoundaryRow &a, const BoundaryRow &b) {
  for (const auto &x : a.role_nodes) for (const auto &y : b.role_nodes)
    if ((x.role == y.role) != (x.node == y.node)) return false;
  return true;
}
BoundaryRow merge(const BoundaryRow &a, const BoundaryRow &b) {
  BoundaryRow out = a;
  out.bindings.insert(out.bindings.end(), b.bindings.begin(), b.bindings.end());
  std::sort(out.bindings.begin(), out.bindings.end(),
      [](const auto &x, const auto &y) { return x.variable < y.variable; });
  for (const auto &r : b.role_nodes)
    if (std::none_of(out.role_nodes.begin(), out.role_nodes.end(),
        [&](const auto &s) { return s.role == r.role; })) out.role_nodes.push_back(r);
  std::sort(out.role_nodes.begin(), out.role_nodes.end(),
      [](const auto &x, const auto &y) { return x.role < y.role; });
  return out;
}
std::string nodeKey(const BoundaryRow &row, const Scope &roles) {
  std::string key;
  for (const auto &role : roles) {
    auto p = std::lower_bound(row.role_nodes.begin(), row.role_nodes.end(), role,
        [](const auto &r, const auto &v) { return r.role < v; });
    if (p == row.role_nodes.end() || p->role != role)
      throw std::logic_error("missing role in shared mapping");
    key += std::to_string(p->node.size()) + ":" + p->node;
  }
  return key;
}
std::size_t bytes(const Rows &rows) {
  std::size_t total = sizeof(Rows) + rows.capacity() * sizeof(BoundaryRow);
  for (const auto &row : rows) {
    total += row.bindings.capacity() * sizeof(VariableBindingCell);
    total += row.role_nodes.capacity() * sizeof(RoleBinding);
    for (const auto &b : row.bindings) total += b.variable.capacity() + b.raw_value.capacity();
    for (const auto &r : row.role_nodes) total += r.role.capacity() + r.node.capacity();
  }
  return total;
}

class Executor {
  const BoundaryBuildConfig &config_;
  const core::LatticeBuildConfig &options_;
  std::map<VariableId, Support> inputs_;
  std::map<VariableId, Scope> roles_;
  std::map<RoleId, std::size_t> ndv_;
  std::map<Scope, double> estimates_;
  std::map<VariableId, std::size_t> rank_;
  struct Donor { Support rows; std::size_t charge; };
  std::map<Scope, Donor> donors_;
  std::deque<Scope> fifo_;
  std::size_t donor_bytes_ = 0;
  SharedConstructionStats stats_;

  Scope roles(const Scope &scope) const {
    std::set<RoleId> result;
    for (const auto &v : scope) result.insert(roles_.at(v).begin(), roles_.at(v).end());
    return {result.begin(), result.end()};
  }
  double estimate(const Scope &scope) {
    if (scope.empty()) return 1;
    auto old = estimates_.find(scope);
    if (old != estimates_.end()) return old->second;
    // Fixed singleton statistics; no joins are run to choose an order.
    long double count = 1;
    std::set<RoleId> seen;
    for (const auto &v : scope) {
      count *= inputs_.at(v)->size();
      for (const auto &r : roles_.at(v)) {
        if (seen.count(r)) count /= std::max<std::size_t>(1, ndv_.at(r));
        seen.insert(r);
      }
      count = std::min(count, static_cast<long double>(1e100));
    }
    return estimates_.emplace(scope, static_cast<double>(count)).first->second;
  }
  void order(const std::vector<Scope> &targets) {
    Scope remaining;
    for (const auto &[v, rows] : inputs_) remaining.push_back(v);
    if (options_.variable_order_policy == core::VariableOrderPolicy::Random) {
      std::mt19937_64 random(options_.variable_order_seed);
      std::shuffle(remaining.begin(), remaining.end(), random);
      stats_.variable_order = remaining;
    } else if (options_.variable_order_policy == core::VariableOrderPolicy::Alphabetical) {
      stats_.variable_order = remaining;
    } else {
      std::set<VariableId> placed;
      while (!remaining.empty()) {
        auto best = remaining.begin();
        long double best_score = std::numeric_limits<long double>::infinity();
        for (auto candidate = remaining.begin(); candidate != remaining.end(); ++candidate) {
          std::set<Scope> prefixes;
          for (const auto &s : targets) if (std::binary_search(s.begin(), s.end(), *candidate)) {
            Scope prefix;
            for (const auto &v : s) if (placed.count(v)) prefix.push_back(v);
            prefixes.insert(std::move(prefix));
          }
          long double score = 0;
          for (auto prefix : prefixes) {
            if (!prefix.empty()) score += estimate(prefix);
            prefix.push_back(*candidate);
            score += estimate(canonical(std::move(prefix)));
          }
          if (score < best_score) { best = candidate; best_score = score; }
        }
        stats_.variable_order.push_back(*best);
        placed.insert(*best);
        remaining.erase(best);
      }
    }
    for (std::size_t i = 0; i < stats_.variable_order.size(); ++i)
      rank_[stats_.variable_order[i]] = i;
  }
  void cache(const Scope &scope, Support rows) {
    if (!options_.enable_derive || options_.execution_schedule == core::ExecutionSchedule::IndependentSources ||
        scope.size() < 3 || donors_.count(scope)) return;
    const auto charge = bytes(*rows);
    const auto budget = options_.rollup_cache_bytes;
    if (charge > budget) return;
    while (!fifo_.empty() && donor_bytes_ > budget - charge) {
      auto p = donors_.find(fifo_.front());
      donor_bytes_ -= p->second.charge;
      donors_.erase(p); fifo_.pop_front();
    }
    donors_.emplace(scope, Donor{std::move(rows), charge});
    fifo_.push_back(scope); donor_bytes_ += charge;
    stats_.donor_peak_bytes = std::max(stats_.donor_peak_bytes, donor_bytes_);
  }
  Support extend(const Scope &prefix, const Support &left, const VariableId &v) {
    ++stats_.prefix_extensions;
    if (prefix.empty()) return inputs_.at(v);
    const auto &right = *inputs_.at(v);
    Scope target = prefix; target.push_back(v); target = canonical(std::move(target));
    const auto left_roles = roles(prefix), target_roles = roles(target);
    Scope shared;
    std::set_intersection(left_roles.begin(), left_roles.end(), roles_.at(v).begin(),
        roles_.at(v).end(), std::back_inserter(shared));
    std::unordered_map<std::string, std::vector<const BoundaryRow *>> buckets;
    for (const auto &r : right) buckets[nodeKey(r, shared)].push_back(&r);
    std::vector<const std::vector<const BoundaryRow *> *> matches;
    long double candidates = 0;
    for (const auto &l : *left) {
      auto p = buckets.find(nodeKey(l, shared));
      matches.push_back(p == buckets.end() ? nullptr : &p->second);
      if (p != buckets.end()) candidates += p->second.size();
    }
    Rows recovered;
    std::set<Identity> extracted, present_left;
    std::set<InstanceId> present_right;
    bool use_recovery = false;
    // Estimate elementary comparison/copy work, including donor extraction and
    // deduplication. Projection is attempted only for a potentially cheaper donor.
    const long double pair_cost = 1 + left_roles.size() * roles_.at(v).size() +
        target.size() + target_roles.size();
    const Donor *donor = nullptr;
    for (const auto &[scope, d] : donors_) if (scope.size() > target.size() && contains(scope, target)) {
      if (!donor || d.rows->size() < donor->rows->size()) donor = &d;
    }
    if (donor && candidates > 0 && !donor->rows->empty()) {
      const long double extract_cost = donor->rows->size() *
          (target.size() + target_roles.size() + std::log2(donor->rows->size() + 1.0));
      const long double saving_upper = std::min(candidates, static_cast<long double>(donor->rows->size())) * pair_cost;
      if (extract_cost + candidates < saving_upper) {
        ++stats_.recovery_attempts;
        for (const auto &row : *donor->rows) {
          const auto key = identity(row, target);
          if (!extracted.insert(key).second) continue;
          BoundaryRow projected;
          for (const auto &b : row.bindings)
            if (std::binary_search(target.begin(), target.end(), b.variable)) projected.bindings.push_back(b);
          for (const auto &r : row.role_nodes)
            if (std::binary_search(target_roles.begin(), target_roles.end(), r.role)) projected.role_nodes.push_back(r);
          present_left.insert(identity(projected, prefix));
          present_right.insert(identity(projected, {v}).front());
          recovered.push_back(std::move(projected));
        }
        const long double recovery_cost = extract_cost + candidates +
            std::max<long double>(0, candidates - recovered.size()) * pair_cost;
        use_recovery = recovery_cost < candidates * pair_cost;
      }
    }
    if (!use_recovery) { recovered.clear(); extracted.clear(); }
    else { ++stats_.recovery_used; stats_.recovered_rows += recovered.size(); }
    auto out = std::make_shared<Rows>(std::move(recovered));
    std::size_t work = 0;
    for (std::size_t i = 0; i < left->size(); ++i) {
      if (!matches[i]) continue;
      const auto &l = (*left)[i];
      const auto left_key = identity(l, prefix);
      for (const auto *r : *matches[i]) {
        // Exact identity, not causal values, determines whether this pair was
        // extracted. DD pairs not in the extracted relation still run the SJ.
        Identity key = left_key;
        key.insert(key.begin() + (std::lower_bound(target.begin(), target.end(), v) - target.begin()),
                   r->bindings.front().instance_id);
        if (use_recovery && extracted.count(key)) continue;
        if (!compatible(l, *r)) continue;
        // Match the existing executor's metric: compatible physical pairs,
        // excluding recovered rows and before any output compaction.
        if (work == std::numeric_limits<std::size_t>::max()) throw std::overflow_error("join work overflow");
        ++work;
        out->push_back(merge(l, *r));
        if (use_recovery) {
          const bool dl = present_left.count(left_key), dr = present_right.count(r->bindings.front().instance_id);
          ++stats_.completion_rows[(dl ? 0 : 2) + (dr ? 0 : 1)];
        }
      }
    }
    if (options_.binary_join_callback) options_.binary_join_callback();
    if (options_.join_work_callback) options_.join_work_callback(left->size(), work);
    return out;
  }
public:
  Executor(const std::vector<VariableInput> &variables, const BoundaryBuildConfig &config)
      : config_(config), options_(config.tree_executor) {
    std::map<RoleId, std::set<NodeId>> distinct;
    for (const auto &input : variables) {
      if (inputs_.count(input.id)) throw std::runtime_error("duplicate pattern variable");
      auto schema = canonical(input.pattern_roles);
      roles_[input.id] = schema;
      auto rows = std::make_shared<Rows>();
      std::set<InstanceId> ids;
      for (const auto &instance : input.instances) {
        if (instance.id == core::kNoInstance || !ids.insert(instance.id).second)
          throw std::runtime_error("duplicate or reserved instance ID");
        BoundaryRow row;
        row.bindings.push_back({input.id, instance.id, instance.value.empty() ? instance.bucket : instance.value});
        row.role_nodes = instance.roles;
        std::sort(row.role_nodes.begin(), row.role_nodes.end(), [](const auto &a, const auto &b) { return a.role < b.role; });
        Scope actual;
        std::set<NodeId> nodes;
        for (const auto &r : row.role_nodes) { actual.push_back(r.role); nodes.insert(r.node); distinct[r.role].insert(r.node); }
        if (actual != schema || nodes.size() != actual.size())
          throw std::runtime_error("invalid pattern mapping schema or injectivity");
        rows->push_back(std::move(row));
      }
      for (const auto &r : schema) distinct[r];
      inputs_[input.id] = std::move(rows);
    }
    for (const auto &[r, nodes] : distinct) ndv_[r] = nodes.size();
  }
  SharedConstructionStats run(std::vector<Scope> targets,
      const std::function<void(const Scope &, const Rows &)> &emit) {
    std::set<Scope> unique;
    for (auto &scope : targets) {
      if (scope.empty()) throw std::runtime_error("empty construction scope");
      for (const auto &v : scope) if (!inputs_.count(v)) throw std::runtime_error("unknown construction variable " + v);
      unique.insert(canonical(std::move(scope)));
    }
    targets.assign(unique.begin(), unique.end());
    auto ordering = config_.ordering_scopes.empty() ? targets : config_.ordering_scopes;
    for (auto &s : ordering) s = canonical(std::move(s));
    order(ordering);
    auto path = [&](Scope s) {
      std::sort(s.begin(), s.end(), [&](const auto &a, const auto &b) { return rank_.at(a) < rank_.at(b); });
      return s;
    };
    std::sort(targets.begin(), targets.end(), [&](const auto &a, const auto &b) {
      const auto pa = path(a), pb = path(b);
      return std::lexicographical_compare(pa.begin(), pa.end(), pb.begin(), pb.end(),
          [&](const auto &x, const auto &y) { return rank_.at(x) < rank_.at(y); });
    });
    if (options_.scope_order_policy == core::ScopeOrderPolicy::Random &&
        options_.execution_schedule != core::ExecutionSchedule::IndependentSources) {
      std::mt19937_64 random(options_.scope_order_seed);
      std::shuffle(targets.begin(), targets.end(), random);
    }
    stats_.unique_scopes = targets.size();
    Scope previous;
    std::vector<Support> active{std::make_shared<Rows>(1)};
    for (const auto &target : targets) {
      const auto next = path(target);
      std::size_t common = 0;
      if (options_.execution_schedule != core::ExecutionSchedule::IndependentSources)
        while (common < previous.size() && common < next.size() && previous[common] == next[common]) ++common;
      active.resize(common + 1);
      Scope prefix(next.begin(), next.begin() + common);
      for (std::size_t i = common; i < next.size(); ++i) {
        auto support = extend(canonical(prefix), active.back(), next[i]);
        prefix.push_back(next[i]);
        active.push_back(support);
        cache(canonical(prefix), std::move(support));
      }
      emit(target, *active.back());
      stats_.construction_order.push_back(target);
      if (options_.progress_callback && options_.progress_every_tasks &&
          (stats_.construction_order.size() % options_.progress_every_tasks == 0 ||
           stats_.construction_order.size() == targets.size())) {
        core::ProgressEvent event;
        event.completed_tasks = stats_.construction_order.size();
        event.total_tasks = targets.size();
        event.depth = next.size();
        event.rows = active.back()->size();
        for (const auto &support : active) event.resident_tuples += support->size();
        event.resident_results = active.size();
        options_.progress_callback(event);
      }
      previous = next;
    }
    return stats_;
  }
};
} // namespace

SharedConstructionStats constructSharedScopes(const std::vector<VariableInput> &variables,
    std::vector<Scope> scopes, const BoundaryBuildConfig &config,
    const std::function<void(const Scope &, const Rows &)> &emit) {
  return Executor(variables, config).run(std::move(scopes), emit);
}
} // namespace whatif
