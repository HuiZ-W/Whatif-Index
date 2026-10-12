#include "causal_index_v4/FullLatticeIndex.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace causal_index_v4 {
namespace {

std::string joinKey(const std::vector<std::string>& parts) {
  std::string out;
  for (const auto& part : parts) {
    out += part;
    out += '\x1f';
  }
  return out;
}
void sortUniqueIds(std::vector<std::size_t>& values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

std::string roleLabel(const RoleId& role) {
  const auto hash = role.find('#');
  if (hash != std::string::npos) {
    return role.substr(0, hash);
  }
  const auto underscore = role.find_last_of('_');
  if (underscore != std::string::npos && underscore + 1 < role.size() &&
      std::all_of(role.begin() + static_cast<std::ptrdiff_t>(underscore + 1), role.end(),
                  [](unsigned char c) { return c >= '0' && c <= '9'; })) {
    return role.substr(0, underscore);
  }
  return role;
}

std::size_t partitionHistogramBin(long double value) {
  if (value <= 0.0L) {
    return 0;
  }
  const auto bin = static_cast<std::size_t>(1.0L + std::floor(std::log2(value)));
  return std::min(bin, kPartitionHistogramBins - 1);
}

void recordPartitionBucket(JoinPartitionTrace& trace,
                           std::size_t parent_rows,
                           std::size_t candidate_rows) {
  const long double pair_work = static_cast<long double>(parent_rows) *
                                static_cast<long double>(candidate_rows);
  const auto parent_size = static_cast<long double>(parent_rows);
  const auto candidate_size = static_cast<long double>(candidate_rows);
  trace.bucket_count += 1;
  trace.partition_pairs += pair_work;
  trace.max_bucket_pairs = std::max(trace.max_bucket_pairs, pair_work);
  trace.max_parent_bucket = std::max(trace.max_parent_bucket, parent_rows);
  trace.max_candidate_bucket = std::max(trace.max_candidate_bucket, candidate_rows);
  trace.parent_bucket_sum += parent_size;
  trace.parent_bucket_square_sum += parent_size * parent_size;
  trace.parent_bucket_xlogx_sum += parent_size <= 0.0L
      ? 0.0L : parent_size * std::log(parent_size);
  trace.candidate_bucket_sum += candidate_size;
  trace.candidate_bucket_square_sum += candidate_size * candidate_size;
  trace.candidate_bucket_xlogx_sum += candidate_size <= 0.0L
      ? 0.0L : candidate_size * std::log(candidate_size);
  trace.pair_work_square_sum += pair_work * pair_work;
  trace.pair_work_xlogx_sum += pair_work <= 0.0L
      ? 0.0L : pair_work * std::log(pair_work);
  trace.parent_bucket_hist[partitionHistogramBin(parent_rows)] += 1;
  trace.candidate_bucket_hist[partitionHistogramBin(candidate_rows)] += 1;
  trace.pair_work_hist[partitionHistogramBin(pair_work)] += 1;
}

}  // namespace

FullLatticeIndex::FullLatticeIndex(std::vector<VariableInput> variables,
                                   LatticeBuildConfig config)
    : variables_(std::move(variables)), config_(config) {}

Mask FullLatticeIndex::allVariablesMask() const {
  if (variables_.size() >= sizeof(Mask) * 8) {
    throw std::runtime_error("full lattice mask supports at most 63 variables");
  }
  return (Mask{1} << variables_.size()) - 1;
}

Mask FullLatticeIndex::maskOf(const std::vector<VariableId>& variable_ids) const {
  Mask mask = 0;
  for (const auto& id : variable_ids) {
    auto found = variable_index_.find(id);
    if (found == variable_index_.end()) {
      throw std::runtime_error("unknown variable: " + id);
    }
    mask |= Mask{1} << found->second;
  }
  return mask;
}

std::vector<VariableId> FullLatticeIndex::variablesOf(Mask mask) const {
  std::vector<VariableId> out;
  for (std::size_t i = 0; i < variables_.size(); ++i) {
    if ((mask & (Mask{1} << i)) != 0) {
      out.push_back(variables_[i].id);
    }
  }
  return out;
}

void FullLatticeIndex::planOnly() {
  normalizeInputs();
  buildPostingsAndStats();
  buildEstimateTable();
  planVariableOrder();
  planTaskOrder();
  planMaterialization();
  planDeriveMaterialization();
}

void FullLatticeIndex::build() {
  planOnly();
  executeFullLattice();
}

Count FullLatticeIndex::exactCount(Mask mask) const {
  if (mask >= result_.exact_counts.size()) {
    return 0;
  }
  return result_.exact_counts[mask];
}

double FullLatticeIndex::estimatedCount(Mask mask) const {
  if (mask >= plan_.estimates.totals.size()) {
    return 0.0;
  }
  return plan_.estimates.totals[mask];
}

const std::vector<TupleRow>& FullLatticeIndex::support(Mask mask) const {
  if (!config_.store_all_supports) {
    throw std::runtime_error("support tables were not retained; set store_all_supports=true for debug access");
  }
  if (mask >= result_.supports.size()) {
    throw std::runtime_error("mask support is out of range");
  }
  return result_.supports[mask];
}

void FullLatticeIndex::normalizeInputs() {
  if (variables_.empty()) {
    throw std::runtime_error("full lattice index needs at least one variable");
  }
  if (variables_.size() >= sizeof(Mask) * 8) {
    throw std::runtime_error("full lattice index supports at most 63 variables");
  }

  variable_index_.clear();
  global_roles_.clear();
  global_role_index_.clear();
  anchor_role_index_.reset();
  anchor_role_.clear();
  node_values_.clear();
  node_code_.clear();
  node_values_.push_back(NodeId{});  // code 0 means unbound.
  value_values_.clear();
  value_code_.clear();
  node_attribute_codes_.clear();
  value_values_.push_back(std::string{});  // code 0 means unbound/missing.
  data_.clear();
  data_.reserve(variables_.size());

  auto role_index = [&](const RoleId& role) {
    auto found = global_role_index_.find(role);
    if (found != global_role_index_.end()) {
      return found->second;
    }
    const std::size_t index = global_roles_.size();
    global_roles_.push_back(role);
    global_role_index_[role] = index;
    return index;
  };

  auto node_code = [&](const NodeId& node) {
    auto found = node_code_.find(node);
    if (found != node_code_.end()) {
      return found->second;
    }
    if (node_values_.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      throw std::runtime_error("too many distinct node ids for packed support");
    }
    const auto code = static_cast<std::uint32_t>(node_values_.size());
    node_values_.push_back(node);
    node_code_[node] = code;
    return code;
  };

  auto value_code = [&](const std::string& value) {
    auto found = value_code_.find(value);
    if (found != value_code_.end()) {
      return found->second;
    }
    if (value_values_.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      throw std::runtime_error("too many distinct values for packed support");
    }
    const auto code = static_cast<std::uint32_t>(value_values_.size());
    value_values_.push_back(value);
    value_code_[value] = code;
    return code;
  };

  for (std::size_t i = 0; i < variables_.size(); ++i) {
    const auto& input = variables_[i];
    if (input.id.empty()) {
      throw std::runtime_error("variable id cannot be empty");
    }
    if (!variable_index_.emplace(input.id, i).second) {
      throw std::runtime_error("duplicate variable id: " + input.id);
    }

    VariableData variable;
    variable.id = input.id;
    variable.pattern_roles = input.pattern_roles;
    std::set<RoleId> pattern_roles(input.pattern_roles.begin(), input.pattern_roles.end());
    if (pattern_roles.size() != input.pattern_roles.size()) {
      throw std::runtime_error("duplicate pattern role in variable: " + input.id);
    }
    for (const auto& role : input.pattern_roles) {
      role_index(role);
    }

    std::unordered_set<InstanceId> seen_instances;
    for (const auto& instance : input.instances) {
      if (!seen_instances.insert(instance.id).second) {
        throw std::runtime_error("duplicate instance id in variable: " + input.id);
      }
      InstanceRecord record;
      record.id = instance.id;
      record.bucket = instance.bucket.empty() ? BucketId{"__default__"} : instance.bucket;
      record.value = instance.value.empty() ? record.bucket : instance.value;
      record.value_code = value_code(record.value);
      record.roles = instance.roles;
      std::set<NodeId> local_nodes;
      for (const auto& binding : instance.roles) {
        if (pattern_roles.count(binding.role) == 0) {
          throw std::runtime_error("instance role is not in pattern role set: " + binding.role);
        }
        if (!record.role_to_node.emplace(binding.role, binding.node).second) {
          throw std::runtime_error("duplicate role binding in instance");
        }
        if (!local_nodes.insert(binding.node).second) {
          throw std::runtime_error("injectivity violation inside a single instance");
        }
        const auto r_index = role_index(binding.role);
        const auto n_code = node_code(binding.node);
        record.role_node_codes.push_back({r_index, n_code});
        record.nodes.push_back(binding.node);
        record.node_codes.push_back(n_code);
      }
      if (record.role_to_node.size() != pattern_roles.size()) {
        throw std::runtime_error("instance does not bind every pattern role for variable: " + input.id);
      }
      std::sort(record.roles.begin(), record.roles.end(), [](const auto& left, const auto& right) {
        return left.role < right.role;
      });
      std::sort(record.role_node_codes.begin(), record.role_node_codes.end());
      std::sort(record.nodes.begin(), record.nodes.end());
      std::sort(record.node_codes.begin(), record.node_codes.end());
      variable.instances.push_back(std::move(record));
    }
    data_.push_back(std::move(variable));
  }

  if (config_.node_attributes) {
    std::set<std::string> attributes;
    for (const auto& item : config_.value_partition_attributes_by_label) {
      attributes.insert(item.second.begin(), item.second.end());
    }
    for (const auto& attribute : attributes) {
      auto& codes = node_attribute_codes_[attribute];
      codes.assign(node_values_.size(), 0);
      for (std::size_t code = 1; code < node_values_.size(); ++code) {
        const auto node = config_.node_attributes->find(node_values_[code]);
        if (node == config_.node_attributes->end()) {
          continue;
        }
        const auto value = node->second.find(attribute);
        if (value != node->second.end()) {
          codes[code] = value_code(value->second);
        }
      }
    }
  }
}

void FullLatticeIndex::buildPostingsAndStats() {
  chooseAnchorRole();
  buckets_.clear();
  bucket_index_.clear();
  role_bucket_distinct_counts_.clear();

  std::unordered_map<std::string, std::unordered_set<NodeId>> role_bucket_nodes;

  for (auto& variable : data_) {
    variable.role_node_posting.clear();
    variable.bucket_counts.clear();
    variable.anchor_posting.clear();
    variable.has_anchor_role = false;
    if (anchor_role_index_) {
      variable.has_anchor_role = std::find(variable.pattern_roles.begin(), variable.pattern_roles.end(),
                                           anchor_role_) != variable.pattern_roles.end();
    }
    for (std::size_t local = 0; local < variable.instances.size(); ++local) {
      const auto& instance = variable.instances[local];
      if (bucket_index_.emplace(instance.bucket, bucket_index_.size()).second) {
        buckets_.push_back(instance.bucket);
      }
      variable.bucket_counts[instance.bucket] += 1;
      for (const auto& binding : instance.roles) {
        variable.role_node_posting[roleKey(binding.role, binding.node)].push_back(local);
        if (anchor_role_index_ && binding.role == anchor_role_) {
          auto code_found = node_code_.find(binding.node);
          if (code_found != node_code_.end()) {
            variable.anchor_posting[code_found->second].push_back(local);
          }
        }
        role_bucket_nodes[joinKey({binding.role, instance.bucket})].insert(binding.node);
      }
    }
    for (auto& item : variable.role_node_posting) {
      sortUniqueIds(item.second);
    }
    for (auto& item : variable.anchor_posting) {
      sortUniqueIds(item.second);
    }
  }

  for (const auto& item : role_bucket_nodes) {
    role_bucket_distinct_counts_[item.first] = item.second.size();
  }
  std::sort(buckets_.begin(), buckets_.end());
  bucket_index_.clear();
  for (std::size_t i = 0; i < buckets_.size(); ++i) {
    bucket_index_[buckets_[i]] = i;
  }
  buildVariableTries();
}

void FullLatticeIndex::chooseAnchorRole() {
  anchor_role_index_.reset();
  anchor_role_.clear();
  if (!config_.enable_anchor_partition || data_.empty()) {
    return;
  }

  std::unordered_map<RoleId, std::size_t> role_counts;
  for (const auto& variable : data_) {
    std::set<RoleId> unique_roles(variable.pattern_roles.begin(), variable.pattern_roles.end());
    for (const auto& role : unique_roles) {
      role_counts[role] += 1;
    }
  }

  std::vector<RoleId> common_roles;
  for (const auto& item : role_counts) {
    if (item.second == data_.size()) {
      common_roles.push_back(item.first);
    }
  }
  if (common_roles.empty()) {
    return;
  }
  std::sort(common_roles.begin(), common_roles.end());

  auto use_role = [&](const RoleId& role) {
    auto found = global_role_index_.find(role);
    if (found == global_role_index_.end()) {
      return false;
    }
    anchor_role_ = role;
    anchor_role_index_ = found->second;
    return true;
  };

  if (!config_.anchor_role.empty()) {
    if (std::binary_search(common_roles.begin(), common_roles.end(), config_.anchor_role) &&
        use_role(config_.anchor_role)) {
      return;
    }
    throw std::runtime_error("configured anchor_role is not common to all variables: " +
                             config_.anchor_role);
  }

  auto prefers_fixed_view = [](const RoleId& role) {
    return role.find("#0") != std::string::npos || role.find("_0") != std::string::npos;
  };
  auto preferred = std::find_if(common_roles.begin(), common_roles.end(), prefers_fixed_view);
  if (preferred != common_roles.end()) {
    use_role(*preferred);
    return;
  }
  use_role(common_roles.front());
}

void FullLatticeIndex::buildVariableTries() {
  for (std::size_t variable = 0; variable < data_.size(); ++variable) {
    std::vector<std::size_t> all_instances;
    all_instances.reserve(data_[variable].instances.size());
    for (std::size_t i = 0; i < data_[variable].instances.size(); ++i) {
      all_instances.push_back(i);
    }
    data_[variable].trie = buildTrieForInstances(variable, all_instances);
    data_[variable].anchor_tries.clear();
    if (anchor_role_index_ && data_[variable].has_anchor_role) {
      for (const auto& item : data_[variable].anchor_posting) {
        data_[variable].anchor_tries.emplace(item.first, buildTrieForInstances(variable, item.second));
      }
    }
  }
}

FullLatticeIndex::VariableTrie FullLatticeIndex::buildTrieForInstances(
    std::size_t variable,
    const std::vector<std::size_t>& instances) const {
  VariableTrie trie;
  trie.role_order = data_[variable].pattern_roles;
  std::sort(trie.role_order.begin(), trie.role_order.end());
  trie.nodes.push_back({});
  for (const auto instance_index : instances) {
    insertTrieInstance(variable, trie, instance_index);
  }
  finalizeTrie(trie);
  return trie;
}

void FullLatticeIndex::insertTrieInstance(
    std::size_t variable,
    VariableTrie& trie,
    std::size_t instance_index) const {
  std::size_t node = 0;
  trie.nodes[node].subtree_count += 1;
  const auto& instance = data_[variable].instances[instance_index];
  for (const auto& role : trie.role_order) {
    const auto global_role = global_role_index_.find(role);
    if (global_role == global_role_index_.end()) {
      throw std::runtime_error("trie build missing global role: " + role);
    }
    std::uint32_t value = 0;
    for (const auto& binding : instance.role_node_codes) {
      if (binding.first == global_role->second) {
        value = binding.second;
        break;
      }
    }
    if (value == 0) {
      throw std::runtime_error("trie build missing role binding for role: " + role);
    }
    auto found = trie.nodes[node].build_child.find(value);
    if (found == trie.nodes[node].build_child.end()) {
      const std::size_t next = trie.nodes.size();
      trie.nodes[node].build_child[value] = next;
      trie.nodes.push_back({});
      node = next;
    } else {
      node = found->second;
    }
    trie.nodes[node].subtree_count += 1;
  }
  trie.nodes[node].build_instances.push_back(instance_index);
}

void FullLatticeIndex::finalizeTrie(VariableTrie& trie) const {
  trie.edges.clear();
  trie.leaf_instances.clear();
  for (auto& node : trie.nodes) {
    std::vector<std::pair<std::uint32_t, std::size_t>> children(node.build_child.begin(), node.build_child.end());
    std::sort(children.begin(), children.end(), [](const auto& left, const auto& right) {
      return left.first < right.first;
    });
    node.edge_begin = trie.edges.size();
    node.edge_count = children.size();
    for (const auto& child : children) {
      trie.edges.push_back({child.first, child.second, trie.nodes[child.second].subtree_count});
    }
    node.build_child.clear();

    std::sort(node.build_instances.begin(), node.build_instances.end());
    node.instance_begin = trie.leaf_instances.size();
    node.instance_count = node.build_instances.size();
    trie.leaf_instances.insert(trie.leaf_instances.end(),
                               node.build_instances.begin(),
                               node.build_instances.end());
    node.build_instances.clear();
  }
}

void FullLatticeIndex::buildEstimateTable() {
  const Mask all = allVariablesMask();
  const std::size_t mask_count = static_cast<std::size_t>(all + 1);
  const std::size_t bucket_count = buckets_.empty() ? 1 : buckets_.size();
  plan_.estimates.buckets = buckets_.empty() ? std::vector<BucketId>{"__default__"} : buckets_;
  plan_.estimates.bucket_index.clear();
  for (std::size_t i = 0; i < plan_.estimates.buckets.size(); ++i) {
    plan_.estimates.bucket_index[plan_.estimates.buckets[i]] = i;
  }
  plan_.estimates.by_mask_bucket.assign(mask_count, std::vector<double>(bucket_count, 0.0));
  plan_.estimates.totals.assign(mask_count, 0.0);

  for (Mask mask = 1; mask <= all; ++mask) {
    for (std::size_t b = 0; b < bucket_count; ++b) {
      const auto& bucket = plan_.estimates.buckets[b];
      double product = 1.0;
      bool any_zero = false;
      std::unordered_map<RoleId, std::size_t> role_multiplicity;

      for (std::size_t v = 0; v < data_.size(); ++v) {
        if ((mask & (Mask{1} << v)) == 0) {
          continue;
        }
        const auto count_found = data_[v].bucket_counts.find(bucket);
        const double count = count_found == data_[v].bucket_counts.end() ? 0.0 :
                             static_cast<double>(count_found->second);
        if (count == 0.0) {
          any_zero = true;
          break;
        }
        product *= count;
        for (const auto& role : data_[v].pattern_roles) {
          role_multiplicity[role] += 1;
        }
      }
      if (any_zero) {
        continue;
      }

      double denominator = 1.0;
      for (const auto& item : role_multiplicity) {
        if (item.second <= 1) {
          continue;
        }
        const auto distinct_found = role_bucket_distinct_counts_.find(joinKey({item.first, bucket}));
        const double distinct = distinct_found == role_bucket_distinct_counts_.end()
                                    ? 1.0
                                    : static_cast<double>(std::max<std::size_t>(1, distinct_found->second));
        denominator *= std::pow(distinct, static_cast<int>(item.second - 1));
      }

      struct RoleOccurrence {
        std::size_t variable = 0;
        RoleId role;
      };
      std::vector<RoleOccurrence> role_occurrences;
      for (std::size_t v = 0; v < data_.size(); ++v) {
        if ((mask & (Mask{1} << v)) == 0) {
          continue;
        }
        for (const auto& role : data_[v].pattern_roles) {
          role_occurrences.push_back({v, role});
        }
      }
      double injectivity_factor = 1.0;
      for (std::size_t i = 0; i < role_occurrences.size(); ++i) {
        for (std::size_t j = i + 1; j < role_occurrences.size(); ++j) {
          if (role_occurrences[i].variable == role_occurrences[j].variable ||
              role_occurrences[i].role == role_occurrences[j].role) {
            continue;
          }
          const auto left_found = role_bucket_distinct_counts_.find(joinKey({role_occurrences[i].role, bucket}));
          const auto right_found = role_bucket_distinct_counts_.find(joinKey({role_occurrences[j].role, bucket}));
          const double left_distinct = left_found == role_bucket_distinct_counts_.end()
                                           ? 1.0
                                           : static_cast<double>(std::max<std::size_t>(1, left_found->second));
          const double right_distinct = right_found == role_bucket_distinct_counts_.end()
                                            ? 1.0
                                            : static_cast<double>(std::max<std::size_t>(1, right_found->second));
          const double distinct = std::max(1.0, std::max(left_distinct, right_distinct));
          injectivity_factor *= std::max(0.0, 1.0 - 1.0 / distinct);
        }
      }
      const double estimate = product * injectivity_factor / denominator;
      plan_.estimates.by_mask_bucket[mask][b] = estimate;
      plan_.estimates.totals[mask] += estimate;
    }
  }
}

Mask FullLatticeIndex::scopeToMask(const std::vector<VariableId>& scope, const char* context) const {
  Mask mask = 0;
  for (const auto& variable_id : scope) {
    auto found = variable_index_.find(variable_id);
    if (found == variable_index_.end()) {
      throw std::runtime_error(std::string(context) + " contains unknown variable: " + variable_id);
    }
    mask |= Mask{1} << found->second;
  }
  return mask;
}

std::vector<Mask> FullLatticeIndex::configuredSourceMasks() const {
  std::set<Mask> sources;
  const bool has_explicit_sources = !config_.source_output_scopes.empty() ||
      !config_.target_scopes.empty() || !config_.binding_output_scopes.empty();
  if (!has_explicit_sources) {
    for (Mask mask = 1; mask <= allVariablesMask(); ++mask) {
      sources.insert(mask);
    }
  }
  if (!config_.source_output_scopes.empty()) {
    for (const auto& item : config_.source_output_scopes) {
      const Mask source = scopeToMask(item.source_scope, "source_scope");
      if (source != 0) {
        sources.insert(source);
      }
    }
  }
  if (!config_.target_scopes.empty()) {
    for (const auto& scope : config_.target_scopes) {
      const Mask target = scopeToMask(scope, "target scope");
      if (target != 0) {
        sources.insert(target);
      }
    }
  }
  for (const auto& scope : config_.binding_output_scopes) {
    const Mask target = scopeToMask(scope, "binding output scope");
    if (target != 0) {
      sources.insert(target);
    }
  }
  return std::vector<Mask>(sources.begin(), sources.end());
}

void FullLatticeIndex::prepareSourceOutputMasks() {
  const Mask all = allVariablesMask();
  output_masks_by_source_.assign(static_cast<std::size_t>(all + 1), {});
  output_masks_.clear();
  binding_output_masks_.clear();
  for (const auto& scope : config_.binding_output_scopes) {
    const Mask mask = scopeToMask(scope, "binding output scope");
    if (mask != 0) {
      binding_output_masks_.insert(mask);
    }
  }

  auto add_output = [&](Mask source, Mask output) {
    if (source == 0 || output == 0) {
      return;
    }
    if ((output & ~source) != 0) {
      throw std::runtime_error("output scope must be a subset of its source scope");
    }
    auto& outputs = output_masks_by_source_[static_cast<std::size_t>(source)];
    if (std::find(outputs.begin(), outputs.end(), output) == outputs.end()) {
      outputs.push_back(output);
      output_masks_.insert(output);
    }
  };

  if (!config_.source_output_scopes.empty()) {
    for (const auto& item : config_.source_output_scopes) {
      const Mask source = scopeToMask(item.source_scope, "source_scope");
      if (source == 0) {
        continue;
      }
      if (item.output_scopes.empty()) {
        add_output(source, source);
        continue;
      }
      for (const auto& output_scope : item.output_scopes) {
        add_output(source, scopeToMask(output_scope, "output_scope"));
      }
    }
  }
  if (!config_.target_scopes.empty()) {
    for (const auto& scope : config_.target_scopes) {
      const Mask source = scopeToMask(scope, "target scope");
      add_output(source, source);
    }
  }
  if (config_.source_output_scopes.empty() && config_.target_scopes.empty()) {
    for (const auto source : configuredSourceMasks()) {
      add_output(source, source);
    }
  }
}

void FullLatticeIndex::planVariableOrder() {
  const std::size_t n = data_.size();
  plan_.variable_order.clear();
  plan_.variable_order_indices.clear();
  rank_of_variable_.assign(n, std::numeric_limits<std::size_t>::max());

  Mask active = 0;
  std::vector<Mask> target_masks_for_cost;
  if (config_.plan_order_from_unpruned_lattice) {
    const Mask all = allVariablesMask();
    target_masks_for_cost.reserve(static_cast<std::size_t>(all));
    for (Mask mask = 1; mask <= all; ++mask) {
      target_masks_for_cost.push_back(mask);
    }
  } else {
    target_masks_for_cost = configuredSourceMasks();
  }
  for (const auto scope_mask : target_masks_for_cost) {
    active |= scope_mask;
  }

  auto roles_for_mask = [&](Mask mask) {
    std::set<RoleId> roles;
    for (std::size_t v = 0; v < n; ++v) {
      if ((mask & (Mask{1} << v)) == 0) {
        continue;
      }
      roles.insert(data_[v].pattern_roles.begin(), data_[v].pattern_roles.end());
    }
    return roles;
  };

  struct KeyStats {
    std::size_t distinct = 0;
    double average_bucket = 0.0;
    std::size_t max_bucket = 0;
  };

  std::unordered_map<std::string, KeyStats> key_stats_cache;
  auto candidate_key_stats = [&](std::size_t candidate, const std::vector<RoleId>& key_roles) {
    KeyStats stats;
    if (key_roles.empty()) {
      return stats;
    }
    std::vector<std::string> cache_parts;
    cache_parts.reserve(key_roles.size() + 1);
    cache_parts.push_back(std::to_string(candidate));
    cache_parts.insert(cache_parts.end(), key_roles.begin(), key_roles.end());
    const auto cache_key = joinKey(cache_parts);
    const auto cached = key_stats_cache.find(cache_key);
    if (cached != key_stats_cache.end()) {
      return cached->second;
    }
    std::unordered_map<std::string, std::size_t> counts;
    counts.reserve(data_[candidate].instances.size());
    for (const auto& instance : data_[candidate].instances) {
      std::vector<std::string> key_parts;
      key_parts.reserve(key_roles.size());
      for (const auto& role : key_roles) {
        const auto found = instance.role_to_node.find(role);
        key_parts.push_back(found == instance.role_to_node.end() ? std::string{} : found->second);
      }
      counts[joinKey(key_parts)] += 1;
    }
    stats.distinct = counts.size();
    if (!counts.empty()) {
      stats.average_bucket = static_cast<double>(data_[candidate].instances.size()) /
                             static_cast<double>(counts.size());
      for (const auto& item : counts) {
        stats.max_bucket = std::max(stats.max_bucket, item.second);
      }
    }
    key_stats_cache.emplace(cache_key, stats);
    return stats;
  };

  auto edge_score = [&](Mask parent, Mask child) {
    const double cardinality = plan_.estimates.totals[child];
    const double scope_width = static_cast<double>(__builtin_popcountll(child));
    const double role_width = static_cast<double>(rolesInMask(child).size());
    const double materialization_cost = cardinality * (1.0 + 0.08 * scope_width + 0.02 * role_width);
    const double compact_cost = cardinality * std::log2(std::max(2.0, cardinality));
    double score = cardinality + materialization_cost + 0.05 * compact_cost;

    const Mask added = child & ~parent;
    if (parent != 0 && added != 0 && (added & (added - 1)) == 0) {
      const std::size_t candidate = static_cast<std::size_t>(__builtin_ctzll(added));
      const auto prefix_roles = roles_for_mask(parent);
      std::vector<RoleId> shared_roles;
      for (const auto& role : data_[candidate].pattern_roles) {
        if (prefix_roles.count(role) != 0) {
          shared_roles.push_back(role);
        }
      }
      std::sort(shared_roles.begin(), shared_roles.end());
      shared_roles.erase(std::unique(shared_roles.begin(), shared_roles.end()), shared_roles.end());

      const auto key_stats = candidate_key_stats(candidate, shared_roles);
      const bool anchor_only = shared_roles.size() == 1 && anchor_role_index_ && shared_roles.front() == anchor_role_;
      const double missing_join_penalty = shared_roles.empty() ? 1000.0 : 1.0;
      const double coarse_key_penalty = shared_roles.empty()
                                            ? 1000.0
                                            : std::pow(3.0, static_cast<double>(std::max<int>(0, 3 - static_cast<int>(shared_roles.size()))));
      const double anchor_penalty = anchor_only ? 8.0 : 1.0;
      const double bucket_penalty = 1.0 + std::log2(std::max(1.0, key_stats.average_bucket));
      const double max_bucket_penalty = 1.0 + 0.05 * std::log2(std::max(1.0, static_cast<double>(key_stats.max_bucket)));
      const double distinct_bonus = 1.0 + std::log2(std::max(2.0, static_cast<double>(key_stats.distinct)));
      score += 4.0 * cardinality * missing_join_penalty * coarse_key_penalty * anchor_penalty *
               bucket_penalty * max_bucket_penalty / distinct_bonus;
    }
    return score;
  };

  std::vector<std::size_t> active_variables;
  for (std::size_t candidate = 0; candidate < n; ++candidate) {
    if ((active & (Mask{1} << candidate)) != 0) {
      active_variables.push_back(candidate);
    }
  }
  const std::size_t active_count = active_variables.size();

  std::sort(target_masks_for_cost.begin(), target_masks_for_cost.end());
  target_masks_for_cost.erase(
      std::unique(target_masks_for_cost.begin(), target_masks_for_cost.end()),
      target_masks_for_cost.end());

  auto append_inactive_variables = [&] {
    for (std::size_t candidate = 0; candidate < n; ++candidate) {
      if ((active & (Mask{1} << candidate)) != 0) {
        continue;
      }
      rank_of_variable_[candidate] = plan_.variable_order_indices.size();
      plan_.variable_order_indices.push_back(candidate);
      plan_.variable_order.push_back(data_[candidate].id);
    }
  };

  auto append_active_order = [&](const std::vector<std::size_t>& order) {
    for (const auto variable : order) {
      rank_of_variable_[variable] = plan_.variable_order_indices.size();
      plan_.variable_order_indices.push_back(variable);
      plan_.variable_order.push_back(data_[variable].id);
    }
    append_inactive_variables();
  };

  auto workload_groups = [&](Mask placed, std::size_t candidate) {
    std::map<Mask, std::size_t> groups;
    const Mask bit = Mask{1} << candidate;
    for (const Mask source : target_masks_for_cost) {
      if ((source & bit) != 0) {
        groups[source & placed] += 1;
      }
    }
    return groups;
  };

  auto workload_incremental_cost = [&](Mask placed,
                                       std::size_t candidate,
                                       bool amortize_by_coverage) {
    const Mask bit = Mask{1} << candidate;
    double cost = 0.0;
    for (const auto& item : workload_groups(placed, candidate)) {
      const Mask child = item.first | bit;
      const double edge = plan_.estimates.totals[item.first] +
                          plan_.estimates.totals[child];
      cost += amortize_by_coverage
          ? edge / static_cast<double>(std::max<std::size_t>(1, item.second))
          : edge;
    }
    return cost;
  };

  if (config_.variable_order_policy == VariableOrderPolicy::Alphabetical) {
    std::sort(active_variables.begin(), active_variables.end(),
              [&](std::size_t left, std::size_t right) {
                return data_[left].id < data_[right].id;
              });
    append_active_order(active_variables);
    return;
  }

  if (config_.variable_order_policy == VariableOrderPolicy::Random) {
    std::mt19937_64 rng(config_.variable_order_seed);
    std::shuffle(active_variables.begin(), active_variables.end(), rng);
    append_active_order(active_variables);
    return;
  }

  if (config_.variable_order_policy ==
      VariableOrderPolicy::InstanceCountAscending) {
    std::sort(active_variables.begin(), active_variables.end(),
              [&](std::size_t left, std::size_t right) {
                const auto left_size = data_[left].instances.size();
                const auto right_size = data_[right].instances.size();
                if (left_size != right_size) {
                  return left_size < right_size;
                }
                return data_[left].id < data_[right].id;
              });
    append_active_order(active_variables);
    return;
  }

  if (config_.variable_order_policy == VariableOrderPolicy::PartialWorkloadDp) {
    const std::size_t state_count = plan_.estimates.totals.size();
    std::vector<double> best_cost(state_count,
                                  std::numeric_limits<double>::infinity());
    std::vector<std::size_t> previous_variable(state_count, n);
    best_cost[0] = 0.0;
    for (Mask placed = 0; placed < state_count; ++placed) {
      if ((placed & ~active) != 0 || !std::isfinite(best_cost[placed])) {
        continue;
      }
      for (const auto candidate : active_variables) {
        const Mask bit = Mask{1} << candidate;
        if ((placed & bit) != 0) {
          continue;
        }
        const Mask next = placed | bit;
        const double trial = best_cost[placed] +
            workload_incremental_cost(placed, candidate, false);
        const bool better = trial < best_cost[next] - 1e-9;
        const bool tied = std::abs(trial - best_cost[next]) < 1e-9;
        if (better ||
            (tied && (previous_variable[next] == n ||
                      data_[candidate].id < data_[previous_variable[next]].id))) {
          best_cost[next] = trial;
          previous_variable[next] = candidate;
        }
      }
    }
    if (!std::isfinite(best_cost[active])) {
      throw std::runtime_error("failed to plan partial-workload DP order");
    }
    std::vector<std::size_t> reverse_order;
    for (Mask state = active; state != 0;) {
      const auto variable = previous_variable[state];
      if (variable == n) {
        throw std::runtime_error("partial-workload DP predecessor is missing");
      }
      reverse_order.push_back(variable);
      state &= ~(Mask{1} << variable);
    }
    std::reverse(reverse_order.begin(), reverse_order.end());
    append_active_order(reverse_order);
    return;
  }

  if (config_.variable_order_policy ==
      VariableOrderPolicy::PartialWorkloadGreedy) {
    std::vector<std::size_t> order;
    order.reserve(active_count);
    Mask placed = 0;
    while (order.size() < active_count) {
      std::optional<std::size_t> best;
      double best_score = std::numeric_limits<double>::infinity();
      double best_new_mass = std::numeric_limits<double>::infinity();
      for (const auto candidate : active_variables) {
        const Mask bit = Mask{1} << candidate;
        if ((placed & bit) != 0) {
          continue;
        }
        const double current = workload_incremental_cost(placed, candidate, true);
        double lookahead = 0.0;
        std::size_t lookahead_count = 0;
        for (const auto future : active_variables) {
          const Mask future_bit = Mask{1} << future;
          if (future == candidate || (placed & future_bit) != 0) {
            continue;
          }
          lookahead += workload_incremental_cost(placed | bit, future, true);
          lookahead_count += 1;
        }
        if (lookahead_count != 0) {
          lookahead /= static_cast<double>(lookahead_count);
        }
        const double score = current + lookahead;
        const double new_mass = workload_incremental_cost(placed, candidate, false);
        if (!best || score < best_score - 1e-9 ||
            (std::abs(score - best_score) < 1e-9 &&
             (new_mass < best_new_mass - 1e-9 ||
              (std::abs(new_mass - best_new_mass) < 1e-9 &&
               data_[candidate].id < data_[*best].id)))) {
          best = candidate;
          best_score = score;
          best_new_mass = new_mass;
        }
      }
      if (!best) {
        throw std::runtime_error("failed to plan partial-workload greedy order");
      }
      order.push_back(*best);
      placed |= Mask{1} << *best;
    }
    append_active_order(order);
    return;
  }

  if (target_masks_for_cost.empty()) {
    for (Mask sub = active; sub != 0; sub = (sub - 1) & active) {
      target_masks_for_cost.push_back(sub);
    }
  }
  Mask selected = 0;
  for (std::size_t step = 0; step < active_count; ++step) {
    std::optional<std::size_t> best;
    double best_score = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0; candidate < n; ++candidate) {
      const Mask bit = Mask{1} << candidate;
      if ((active & bit) == 0 || (selected & bit) != 0) {
        continue;
      }
      double score = edge_score(selected, selected | bit);
      for (Mask sub = selected;; sub = (sub - 1) & selected) {
        if (sub != selected) {
          score += 0.10 * edge_score(sub, sub | bit);
        }
        if (sub == 0) {
          break;
        }
      }
      if (!best || score < best_score - 1e-9 ||
          (std::abs(score - best_score) < 1e-9 && data_[candidate].id < data_[*best].id)) {
        best = candidate;
        best_score = score;
      }
    }
    if (!best) {
      throw std::runtime_error("failed to choose greedy variable order");
    }
    plan_.variable_order_indices.push_back(*best);
    plan_.variable_order.push_back(data_[*best].id);
    rank_of_variable_[*best] = step;
    selected |= Mask{1} << *best;
  }

  for (std::size_t candidate = 0; candidate < n; ++candidate) {
    if ((selected & (Mask{1} << candidate)) != 0) {
      continue;
    }
    rank_of_variable_[candidate] = plan_.variable_order_indices.size();
    plan_.variable_order_indices.push_back(candidate);
    plan_.variable_order.push_back(data_[candidate].id);
  }
}

void FullLatticeIndex::planTaskOrder() {
  plan_.task_order.clear();
  plan_.target_masks.clear();
  plan_.scope_order.clear();
  planned_task_masks_.clear();
  const std::size_t n = data_.size();

  prepareSourceOutputMasks();
  std::unordered_set<Mask> required;
  std::set<Mask> unique_sources;
  for (const auto source : configuredSourceMasks()) {
    if (source == 0) {
      continue;
    }
    unique_sources.insert(source);
    Mask prefix = 0;
    for (std::size_t rank = 0; rank < n; ++rank) {
      const auto variable = plan_.variable_order_indices[rank];
      if ((source & (Mask{1} << variable)) == 0) {
        continue;
      }
      prefix |= Mask{1} << variable;
      required.insert(prefix);
    }
  }
  plan_.target_masks.assign(unique_sources.begin(), unique_sources.end());

  std::function<void(Mask, std::size_t)> visit = [&](Mask prefix, std::size_t next_rank) {
    for (std::size_t rank = next_rank; rank < n; ++rank) {
      const auto variable = plan_.variable_order_indices[rank];
      const Mask child = prefix | (Mask{1} << variable);
      if (required.count(child) == 0) {
        continue;
      }
      plan_.task_order.push_back(child);
      planned_task_masks_.insert(child);
      if (unique_sources.count(child) != 0) {
        plan_.scope_order.push_back(child);
      }
      visit(child, rank + 1);
    }
  };
  visit(0, 0);

  if (config_.scope_order_policy == ScopeOrderPolicy::Alphabetical) {
    auto canonical_scope = [&](Mask mask) {
      std::vector<VariableId> ids;
      for (std::size_t variable = 0; variable < n; ++variable) {
        if ((mask & (Mask{1} << variable)) != 0) {
          ids.push_back(data_[variable].id);
        }
      }
      std::sort(ids.begin(), ids.end());
      return ids;
    };
    std::sort(plan_.scope_order.begin(), plan_.scope_order.end(),
              [&](Mask left, Mask right) {
                return canonical_scope(left) < canonical_scope(right);
              });
  } else if (config_.scope_order_policy == ScopeOrderPolicy::Random) {
    std::mt19937_64 rng(config_.scope_order_seed);
    if (!config_.scope_order_from_unpruned_lattice) {
      std::shuffle(plan_.scope_order.begin(), plan_.scope_order.end(), rng);
    } else {
      std::vector<Mask> unpruned_scope_order;
      std::function<void(Mask, std::size_t)> visit_unpruned =
          [&](Mask prefix, std::size_t next_rank) {
            for (std::size_t rank = next_rank; rank < n; ++rank) {
              const auto variable = plan_.variable_order_indices[rank];
              const Mask child = prefix | (Mask{1} << variable);
              unpruned_scope_order.push_back(child);
              visit_unpruned(child, rank + 1);
            }
          };
      visit_unpruned(0, 0);
      std::shuffle(unpruned_scope_order.begin(),
                   unpruned_scope_order.end(), rng);
      plan_.scope_order.clear();
      for (const Mask source : unpruned_scope_order) {
        if (unique_sources.count(source) != 0) {
          plan_.scope_order.push_back(source);
        }
      }
    }
  }
}

void FullLatticeIndex::planMaterialization() {
  plan_.materialization.clear();
  plan_.pinned_masks.clear();
  plan_.estimated_cost = 0.0;
  for (const Mask mask : plan_.task_order) {
    const Mask parent = parentMask(mask);
    const double parent_cost = plan_.estimates.totals[parent];
    const double output_cardinality = plan_.estimates.totals[mask];
    const double scope_width = static_cast<double>(__builtin_popcountll(mask));
    const double role_width = static_cast<double>(rolesInMask(mask).size());
    const double materialization_cost = output_cardinality * (1.0 + 0.08 * scope_width + 0.02 * role_width);
    const double compact_cost = output_cardinality * std::log2(std::max(2.0, output_cardinality));
    plan_.estimated_cost += parent_cost + materialization_cost + 0.05 * compact_cost;

    MaterializedPrefix prefix;
    prefix.mask = mask;
    prefix.estimated_size = output_cardinality;
    prefix.estimated_edge_cost = parent_cost + materialization_cost + 0.05 * compact_cost;
    std::size_t reuse_count = 0;
    for (const Mask other : plan_.task_order) {
      if (other != mask && (other & mask) == mask) {
        reuse_count += 1;
      }
    }
    prefix.reuse_count = reuse_count;
    const double size = std::max(1.0, prefix.estimated_size);
    prefix.score = prefix.reuse_count == 0
                       ? 0.0
                       : static_cast<double>(prefix.reuse_count) * prefix.estimated_edge_cost / size;
    plan_.materialization.push_back(prefix);
  }

  std::sort(plan_.materialization.begin(), plan_.materialization.end(),
            [](const auto& left, const auto& right) {
              if (std::abs(left.score - right.score) > 1e-9) {
                return left.score > right.score;
              }
              if (std::abs(left.estimated_size - right.estimated_size) > 1e-9) {
                return left.estimated_size < right.estimated_size;
              }
              return left.mask < right.mask;
            });

  if (config_.cache_budget_tuples == 0) {
    return;
  }
  double remaining = static_cast<double>(config_.cache_budget_tuples);
  for (auto& item : plan_.materialization) {
    const double size = std::max(1.0, item.estimated_size);
    if (item.score <= 0.0 || size > remaining) {
      continue;
    }
    item.pinned = true;
    plan_.pinned_masks.push_back(item.mask);
    remaining -= size;
  }
}

void FullLatticeIndex::planDeriveMaterialization() {
  plan_.derive_pinned_masks.clear();
  if (!config_.enable_derive || config_.store_all_supports ||
      config_.derive_materialization_budget_tuples == 0) {
    return;
  }

  const Mask all = allVariablesMask();
  std::vector<std::size_t> task_pos(static_cast<std::size_t>(all + 1),
                                    std::numeric_limits<std::size_t>::max());
  for (std::size_t i = 0; i < plan_.task_order.size(); ++i) {
    task_pos[static_cast<std::size_t>(plan_.task_order[i])] = i;
  }

  struct Candidate {
    Mask mask = 0;
    double score = 0.0;
    double size = 0.0;
    std::size_t future_uses = 0;
  };
  std::vector<Candidate> candidates;
  for (const Mask superset : plan_.task_order) {
    double benefit = 0.0;
    std::size_t future_uses = 0;
    for (std::size_t donor = 0; donor < data_.size(); ++donor) {
      const Mask donor_bit = Mask{1} << donor;
      if ((superset & donor_bit) == 0) {
        continue;
      }
      const Mask target = superset & ~donor_bit;
      if (target == 0 || target >= task_pos.size() ||
          task_pos[static_cast<std::size_t>(target)] == std::numeric_limits<std::size_t>::max() ||
          task_pos[static_cast<std::size_t>(superset)] >=
              task_pos[static_cast<std::size_t>(target)]) {
        continue;
      }
      if (sharedRoles(target, donor).empty()) {
        continue;
      }
      benefit += target < plan_.estimates.totals.size() ? plan_.estimates.totals[target] : 1.0;
      future_uses += 1;
    }
    if (future_uses == 0) {
      continue;
    }
    const double size = superset < plan_.estimates.totals.size()
                            ? std::max(1.0, plan_.estimates.totals[superset])
                            : 1.0;
    candidates.push_back({superset, benefit / size, size, future_uses});
  }

  std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
    if (std::abs(left.score - right.score) > 1e-9) {
      return left.score > right.score;
    }
    if (left.future_uses != right.future_uses) {
      return left.future_uses > right.future_uses;
    }
    if (std::abs(left.size - right.size) > 1e-9) {
      return left.size < right.size;
    }
    return left.mask < right.mask;
  });

  double remaining = static_cast<double>(config_.derive_materialization_budget_tuples);
  for (const auto& candidate : candidates) {
    if (candidate.size > remaining) {
      continue;
    }
    plan_.derive_pinned_masks.push_back(candidate.mask);
    remaining -= candidate.size;
  }
}

void FullLatticeIndex::executeFullLattice() {
  const Mask all = allVariablesMask();
  result_ = {};
  resident_.clear();
  cache_clock_ = 0;
  result_.task_order = plan_.task_order;
  result_.exact_counts.assign(static_cast<std::size_t>(all + 1), 0);
  result_.supports.assign(static_cast<std::size_t>(all + 1), {});
  result_.binding_outputs.assign(static_cast<std::size_t>(all + 1), {});
  result_.value_counts.assign(static_cast<std::size_t>(all + 1), {});
  result_.value_trie_nodes.assign(static_cast<std::size_t>(all + 1), 0);

  putResident(0, rootSupport());
  if (config_.execution_schedule ==
      ExecutionSchedule::ScopeOrderedWorkingPath) {
    executeScopeOrderedWorkingPath();
  } else if (config_.execution_schedule ==
             ExecutionSchedule::BoundedScopeCache) {
    executeBoundedScopeCache();
  } else if (config_.execution_schedule ==
      ExecutionSchedule::JoinSizeGreedyWorkingPath) {
    executeJoinSizeGreedyWorkingPath();
  } else if (config_.execution_schedule == ExecutionSchedule::CacheAwareGreedy) {
    executeCacheAwareGreedy();
  } else if (config_.execution_schedule == ExecutionSchedule::IndependentSources) {
    executeIndependentSources();
  } else {
    executeDfs(0, 0, 0);
  }
}

std::vector<Mask> FullLatticeIndex::prefixPath(Mask source) const {
  std::vector<Mask> path;
  Mask prefix = 0;
  for (const auto variable : plan_.variable_order_indices) {
    const Mask bit = Mask{1} << variable;
    if ((source & bit) == 0) {
      continue;
    }
    prefix |= bit;
    path.push_back(prefix);
  }
  return path;
}

bool FullLatticeIndex::isPrefixOfSource(Mask prefix, Mask source) const {
  if (prefix == 0) {
    return true;
  }
  const auto path = prefixPath(source);
  return std::find(path.begin(), path.end(), prefix) != path.end();
}

void FullLatticeIndex::aggregateSourceOutputs(
    Mask source,
    const PackedSupport& support) {
  const Count weight = supportWeight(support);
  for (const auto output :
       output_masks_by_source_[static_cast<std::size_t>(source)]) {
    result_.exact_counts[static_cast<std::size_t>(output)] = weight;
    aggregateProjectedValueCounts(source, output, support);
  }
  materializeBindingOutput(source, support);
}

bool FullLatticeIndex::requiresBindingOutput(Mask mask) const {
  return binding_output_masks_.count(mask) != 0;
}

void FullLatticeIndex::materializeBindingOutput(
    Mask source,
    const PackedSupport& support) {
  if (!requiresBindingOutput(source)) {
    return;
  }
  const std::size_t n = data_.size();
  const std::size_t role_count = global_roles_.size();
  auto& output = result_.binding_outputs[static_cast<std::size_t>(source)];
  output.clear();
  output.reserve(support.row_count);
  for (std::size_t row = 0; row < support.row_count; ++row) {
    MaterializedBindingRow item;
    const auto instance_begin = support.instance_ids.begin() +
        static_cast<std::ptrdiff_t>(row * n);
    item.instance_ids.assign(instance_begin,
                             instance_begin + static_cast<std::ptrdiff_t>(n));
    item.values.assign(n, {});
    for (std::size_t variable = 0; variable < n; ++variable) {
      item.values[variable] = valueForCode(
          support.value_codes[row * n + variable]);
    }
    for (std::size_t role = 0; role < role_count; ++role) {
      const auto code = support.role_node_codes[row * role_count + role];
      if (code != 0) {
        item.role_nodes.push_back({global_roles_[role], node_values_[code]});
      }
    }
    item.multiplicity = rowWeight(support, row);
    output.push_back(std::move(item));
  }
}

void FullLatticeIndex::executeCacheAwareGreedy() {
  std::set<Mask> remaining(plan_.target_masks.begin(), plan_.target_masks.end());
  while (!remaining.empty()) {
    Mask selected = 0;
    std::size_t selected_reuse_depth = 0;
    double selected_remaining_cost = std::numeric_limits<double>::infinity();

    for (const auto source : remaining) {
      const auto path = prefixPath(source);
      std::size_t reuse_depth = 0;
      for (std::size_t i = 0; i < path.size(); ++i) {
        if (resident_.count(path[i]) != 0) {
          reuse_depth = i + 1;
        }
      }
      double remaining_cost = 0.0;
      Mask parent = reuse_depth == 0 ? 0 : path[reuse_depth - 1];
      for (std::size_t i = reuse_depth; i < path.size(); ++i) {
        const Mask child = path[i];
        remaining_cost += plan_.estimates.totals[parent] +
                          plan_.estimates.totals[child];
        parent = child;
      }
      if (selected == 0 ||
          remaining_cost < selected_remaining_cost - 1e-9 ||
          (std::abs(remaining_cost - selected_remaining_cost) < 1e-9 &&
           (reuse_depth > selected_reuse_depth ||
            (reuse_depth == selected_reuse_depth && source < selected)))) {
        selected = source;
        selected_reuse_depth = reuse_depth;
        selected_remaining_cost = remaining_cost;
      }
    }

    const auto path = prefixPath(selected);
    if (selected_reuse_depth == path.size()) {
      aggregateSourceOutputs(selected, ensureResident(selected));
    } else {
      Mask parent = selected_reuse_depth == 0 ? 0 : path[selected_reuse_depth - 1];
      for (std::size_t i = selected_reuse_depth; i < path.size(); ++i) {
        const Mask child = path[i];
        const Mask added = child & ~parent;
        const std::size_t variable =
            static_cast<std::size_t>(__builtin_ctzll(added));
        const bool final = i + 1 == path.size();

        pinActive(parent);
        const auto& parent_support = ensureResident(parent);
        bool needed_by_future_source = false;
        for (const auto other : remaining) {
          if (other != selected && isPrefixOfSource(child, other)) {
            needed_by_future_source = true;
            break;
          }
        }
        const bool can_push_leaf = final && !needed_by_future_source &&
            !config_.store_all_supports && !config_.store_participation &&
            !isDerivePinned(child) && !config_.store_derive_results &&
            !config_.collect_partition_traces &&
            !requiresBindingOutput(child);
        if (can_push_leaf) {
          const auto& outputs =
              output_masks_by_source_[static_cast<std::size_t>(child)];
          const Count weight = aggregateLeafValueCounts(
              parent_support, parent, variable, child, outputs);
          for (const auto output : outputs) {
            result_.exact_counts[static_cast<std::size_t>(output)] = weight;
          }
          result_.emitted_rows += weight;
          reportProgress(child, parent, variable, i + 1, weight);
          unpinActive(parent);
          parent = child;
          continue;
        }

        PackedSupport child_support =
            parent == 0 ? singletonSupport(variable)
                        : extendSupport(parent_support, parent, variable, nullptr);
        const Count weight = supportWeight(child_support);
        result_.emitted_rows += weight;
        if (final) {
          aggregateSourceOutputs(child, child_support);
        }
        reportProgress(child, parent, variable, i + 1, weight);
        putResident(child, std::move(child_support));
        unpinActive(parent);
        parent = child;
      }
    }

    remaining.erase(selected);
    std::vector<Mask> dead_prefixes;
    for (const auto& item : resident_) {
      if (item.first == 0) {
        continue;
      }
      bool useful = false;
      for (const auto source : remaining) {
        if (isPrefixOfSource(item.first, source)) {
          useful = true;
          break;
        }
      }
      if (!useful) {
        dead_prefixes.push_back(item.first);
      }
    }
    for (const auto prefix : dead_prefixes) {
      releaseResident(prefix);
    }
  }
}

void FullLatticeIndex::streamIndependentCubes(const std::function<void(
    Mask, std::map<std::vector<std::string>, Count>&&)>& emit) {
  if (!emit || config_.execution_schedule != ExecutionSchedule::IndependentSources ||
      config_.variable_order_policy != VariableOrderPolicy::Alphabetical ||
      config_.enable_derive || config_.enable_value_partition_join ||
      config_.store_all_supports || config_.store_participation ||
      config_.store_derive_results || config_.collect_partition_traces ||
      !config_.store_value_counts || !config_.target_scopes.empty() ||
      !config_.source_output_scopes.empty() || !config_.binding_output_scopes.empty()) {
    throw std::runtime_error("streamIndependentCubes requires independent alphabetical full cubes without auxiliary outputs");
  }
  plan_ = {};
  result_ = {};
  normalizeInputs();
  buildPostingsAndStats();
  for (std::size_t i = 0; i < data_.size(); ++i)
    plan_.variable_order_indices.push_back(i);
  std::sort(plan_.variable_order_indices.begin(), plan_.variable_order_indices.end(),
            [&](std::size_t a, std::size_t b) { return data_[a].id < data_[b].id; });
  rank_of_variable_.resize(data_.size());
  for (std::size_t rank = 0; rank < data_.size(); ++rank) {
    const auto variable = plan_.variable_order_indices[rank];
    rank_of_variable_[variable] = rank;
    plan_.variable_order.push_back(data_[variable].id);
  }

  // Only the current cube's path and support are transient. Completed cubes
  // go straight to the consumer, never into mask-indexed result arrays.
  const Mask all = allVariablesMask();
  const auto total = static_cast<std::size_t>(all);
  for (Mask source = 1; source <= all; ++source) {
    auto support = rootSupport();
    Mask parent = 0;
    const auto path = prefixPath(source);
    std::map<std::vector<std::string>, Count> counts;
    Count weight = 0;
    for (const Mask child : path) {
      const auto variable = static_cast<std::size_t>(__builtin_ctzll(child & ~parent));
      if (child == source && config_.independent_leaf_pushdown) {
        weight = aggregateLeafValueCounts(support, parent, variable, child, {source}, &counts);
      } else {
        support = parent == 0 ? singletonSupport(variable)
                              : extendSupport(support, parent, variable, nullptr);
        weight = supportWeight(support);
        if (child == source) {
          for (std::size_t row = 0; row < support.row_count; ++row)
            counts[valueKey(support, row, source)] += rowWeight(support, row);
        }
      }
      result_.emitted_rows += weight;
      parent = child;
    }
    emit(source, std::move(counts));
    // Streaming progress counts completed cubes, not stored prefix tasks.
    ++result_.completed_tasks;
    const auto every = std::max<std::size_t>(1, config_.progress_every_tasks);
    if (config_.progress_callback &&
        (source == 1 || source == all || source % every == 0)) {
      ProgressEvent event;
      event.completed_tasks = result_.completed_tasks;
      event.total_tasks = total;
      event.mask = source;
      event.depth = path.size();
      event.rows = weight;
      event.emitted_rows = result_.emitted_rows;
      config_.progress_callback(event);
    }
  }
}

void FullLatticeIndex::executeIndependentSources() {
  result_.task_order.clear();
  for (const auto source : plan_.target_masks) {
    const auto path = prefixPath(source);
    result_.task_order.insert(result_.task_order.end(), path.begin(), path.end());
  }

  for (const auto source : plan_.target_masks) {
    PackedSupport support = rootSupport();
    Mask parent = 0;
    const auto path = prefixPath(source);
    for (std::size_t i = 0; i < path.size(); ++i) {
      const Mask child = path[i];
      const Mask added = child & ~parent;
      const std::size_t variable =
          static_cast<std::size_t>(__builtin_ctzll(added));
      const bool push_leaf = config_.independent_leaf_pushdown &&
          i + 1 == path.size() && !config_.store_all_supports &&
          !config_.store_participation && !config_.store_derive_results &&
          !config_.collect_partition_traces && !requiresBindingOutput(child);
      if (push_leaf) {
        const auto& outputs = output_masks_by_source_[static_cast<std::size_t>(child)];
        const Count weight = aggregateLeafValueCounts(support, parent, variable, child, outputs);
        for (const auto output : outputs)
          result_.exact_counts[static_cast<std::size_t>(output)] = weight;
        result_.emitted_rows += weight;
        reportProgress(child, parent, variable, i + 1, weight);
        parent = child;
        continue;
      }
      PackedSupport child_support =
          parent == 0 ? singletonSupport(variable)
                      : extendSupport(support, parent, variable, nullptr);
      const Count weight = supportWeight(child_support);
      result_.emitted_rows += weight;
      reportProgress(child, parent, variable, i + 1, weight);
      support = std::move(child_support);
      parent = child;
    }
    if (!(config_.independent_leaf_pushdown && !config_.store_all_supports &&
          !config_.store_participation && !config_.store_derive_results &&
          !config_.collect_partition_traces && !requiresBindingOutput(source))) {
      aggregateSourceOutputs(source, support);
    }
  }
}

void FullLatticeIndex::executeDfs(Mask parent_mask,
                                  std::size_t last_order_position,
                                  std::size_t depth) {
  const std::size_t n = data_.size();
  for (std::size_t rank = last_order_position; rank < n; ++rank) {
    const std::size_t variable = plan_.variable_order_indices[rank];
    const Mask child = parent_mask | (Mask{1} << variable);
    if (planned_task_masks_.count(child) == 0) {
      continue;
    }
    bool has_planned_child = false;
    for (std::size_t next_rank = rank + 1; next_rank < n; ++next_rank) {
      const std::size_t next_variable = plan_.variable_order_indices[next_rank];
      if (planned_task_masks_.count(child | (Mask{1} << next_variable)) != 0) {
        has_planned_child = true;
        break;
      }
    }

    pinActive(parent_mask);
    const auto& parent_support = ensureResident(parent_mask);

    std::vector<std::uint8_t> has_partner;
    const bool capture_participation = config_.store_participation && parent_mask != 0;
    const bool can_push_leaf_cube = !has_planned_child && !capture_participation &&
        !config_.store_all_supports && !isDerivePinned(child) &&
        !config_.store_derive_results && !requiresBindingOutput(child);
    const bool use_leaf_cube = can_push_leaf_cube && !config_.collect_partition_traces;
    if (use_leaf_cube) {
      const auto& output_masks = output_masks_by_source_[static_cast<std::size_t>(child)];
      const Count child_weight = aggregateLeafValueCounts(parent_support, parent_mask, variable, child,
                                                          output_masks);
      for (const auto output : output_masks) {
        result_.exact_counts[static_cast<std::size_t>(output)] = child_weight;
      }
      result_.emitted_rows += child_weight;
      reportProgress(child, parent_mask, variable, depth + 1, child_weight);
      unpinActive(parent_mask);
      continue;
    }

    PackedSupport child_support;
    std::optional<DeriveResult> derive_choice;
    if (config_.enable_derive && parent_mask != 0) {
      derive_choice = bestOnlineDerive(child, parent_mask, parent_support);
    }

    if (derive_choice && derive_choice->derive_cost <= derive_choice->extend_cost) {
      derive_choice->used = true;
      child_support = packSupport(derive_choice->rows);
      if (config_.store_derive_results) {
        result_.derive_results.push_back(*derive_choice);
      }
      result_.cache_stats.derive_used += 1;
      if (capture_participation) {
        has_partner.assign(parent_support.row_count, child_support.row_count == 0 ? 0 : 1);
      }
    } else {
      child_support = parent_mask == 0
                          ? singletonSupport(variable)
                          : extendSupport(parent_support, parent_mask, variable,
                                          capture_participation ? &has_partner : nullptr);
      if (derive_choice && config_.store_derive_results) {
        result_.derive_results.push_back(*derive_choice);
      }
    }

    if (capture_participation) {
      result_.participation.push_back({parent_mask, variable, std::move(has_partner)});
    }
    const Count child_weight = supportWeight(child_support);
    const auto& output_masks = output_masks_by_source_[static_cast<std::size_t>(child)];
    for (const auto output : output_masks) {
      result_.exact_counts[static_cast<std::size_t>(output)] = child_weight;
      aggregateProjectedValueCounts(child, output, child_support);
    }
    materializeBindingOutput(child, child_support);
    result_.emitted_rows += child_weight;
    reportProgress(child, parent_mask, variable, depth + 1, child_weight);
    if (config_.store_all_supports || isDerivePinned(child)) {
      const bool first_materialization = result_.supports[child].empty();
      result_.supports[child] = unpackSupport(child_support);
      if (!config_.store_all_supports && first_materialization) {
        result_.cache_stats.derive_materialized_masks += 1;
        result_.cache_stats.derive_materialized_tuples += result_.supports[child].size();
      }
      putResident(child, std::move(child_support));
    } else {
      putResident(child, std::move(child_support));
    }
    unpinActive(parent_mask);
    executeDfs(child, rank + 1, depth + 1);
    if (!config_.store_all_supports) {
      releaseResident(child);
    }
  }
}

void FullLatticeIndex::aggregateValueCounts(Mask mask, const PackedSupport& support) {
  aggregateProjectedValueCounts(mask, mask, support);
}

void FullLatticeIndex::aggregateProjectedValueCounts(Mask source_mask,
                                                     Mask output_mask,
                                                     const PackedSupport& support) {
  if (!config_.store_value_counts) {
    return;
  }
  if ((output_mask & ~source_mask) != 0) {
    throw std::runtime_error("projected output mask must be a subset of source mask");
  }
  auto& counts = result_.value_counts[static_cast<std::size_t>(output_mask)];
  for (std::size_t row = 0; row < support.row_count; ++row) {
    counts[valueKey(support, row, output_mask)] += rowWeight(support, row);
  }
  result_.value_trie_nodes[static_cast<std::size_t>(output_mask)] = valueTrieNodeCount(counts);
  if (config_.streamed_value_count_sink) {
    config_.streamed_value_count_sink(output_mask, std::move(counts));
    counts.clear();
  }
}

Count FullLatticeIndex::aggregateLeafValueCounts(const PackedSupport& parent_support,
                                                Mask parent_mask,
                                                std::size_t variable,
                                                Mask child_mask,
                                                const std::vector<Mask>& output_masks,
                                                std::map<std::vector<std::string>, Count>* streamed_counts) {
  const std::size_t n = data_.size();
  Count total = 0;
  RuntimeCacheStats stats;

  bool has_partition_attribute = false;
  if (config_.enable_value_partition_join && config_.node_attributes &&
      parent_support.row_count >= config_.value_partition_min_parent_rows) {
    for (const auto& role : sharedRoles(parent_mask, variable)) {
      const auto found = config_.value_partition_attributes_by_label.find(roleLabel(role));
      if (found != config_.value_partition_attributes_by_label.end() && !found->second.empty()) {
        has_partition_attribute = true;
        break;
      }
    }
  }
  if (has_partition_attribute) {
    auto leaf_support = extendSupport(parent_support, parent_mask, variable, nullptr);
    total = supportWeight(leaf_support);
    for (const auto output_mask : output_masks) {
      aggregateProjectedValueCounts(child_mask, output_mask, leaf_support);
    }
    result_.cache_stats.leaf_cube_pushdowns += 1;
    result_.cache_stats.leaf_cube_rows += static_cast<std::size_t>(total);
    return total;
  }

  if (parent_mask != 0 && config_.binary_join_callback)
    config_.binary_join_callback();
  for (std::size_t row = 0; row < parent_support.row_count; ++row) {
    auto candidates = candidateInstances(parent_support, row, variable, stats);
    const Count weight = rowWeight(parent_support, row);
    for (const auto local : candidates) {
      const auto& candidate = data_[variable].instances[local];
      total += weight;
      if (!config_.store_value_counts) {
        continue;
      }
      for (const auto output_mask : output_masks) {
        if ((output_mask & ~child_mask) != 0) {
          throw std::runtime_error("projected output mask must be a subset of leaf source mask");
        }
        auto& counts = streamed_counts ? *streamed_counts
            : result_.value_counts[static_cast<std::size_t>(output_mask)];
        std::vector<std::string> key;
        key.reserve(static_cast<std::size_t>(__builtin_popcountll(output_mask)));
        for (std::size_t v = 0; v < n; ++v) {
          if ((output_mask & (Mask{1} << v)) == 0) {
            continue;
          }
          const auto code = v == variable
              ? candidate.value_code
              : parent_support.value_codes[row * n + v];
          key.push_back(valueForCode(code));
        }
        counts[key] += weight;
      }
    }
  }

  if (config_.store_value_counts && !streamed_counts) {
    for (const auto output_mask : output_masks) {
      auto& counts = result_.value_counts[static_cast<std::size_t>(output_mask)];
      result_.value_trie_nodes[static_cast<std::size_t>(output_mask)] = valueTrieNodeCount(counts);
      if (config_.streamed_value_count_sink) {
        config_.streamed_value_count_sink(output_mask, std::move(counts));
        counts.clear();
      }
    }
  }
  result_.cache_stats.trie_nodes_visited += stats.trie_nodes_visited;
  result_.cache_stats.trie_subtrees_skipped += stats.trie_subtrees_skipped;
  result_.cache_stats.live_candidates_emitted += stats.live_candidates_emitted;
  result_.cache_stats.anchor_buckets_used += stats.anchor_buckets_used;
  result_.cache_stats.anchor_candidates_scanned += stats.anchor_candidates_scanned;
  result_.cache_stats.leaf_cube_pushdowns += 1;
  result_.cache_stats.leaf_cube_rows += static_cast<std::size_t>(total);
  if (parent_mask != 0 && config_.join_work_callback)
    config_.join_work_callback(parent_support.row_count, stats.live_candidates_emitted);
  return total;
}

Count FullLatticeIndex::rowWeight(const PackedSupport& support, std::size_t row) const {
  if (support.counts.empty()) {
    return 1;
  }
  return support.counts[row];
}

Count FullLatticeIndex::supportWeight(const PackedSupport& support) const {
  if (support.counts.empty()) {
    return static_cast<Count>(support.row_count);
  }
  Count total = 0;
  for (const auto value : support.counts) {
    total += value;
  }
  return total;
}

std::string FullLatticeIndex::compactRowKey(const PackedSupport& support, std::size_t row) const {
  const std::size_t n = data_.size();
  const std::size_t r = global_roles_.size();
  std::string key;
  key.reserve((n + r) * sizeof(std::uint32_t) +
              (config_.binding_output_scopes.empty()
                   ? 1
                   : n * sizeof(InstanceId) + 2));
  if (!config_.binding_output_scopes.empty()) {
    const auto* instance_begin = support.instance_ids.data() + row * n;
    key.append(reinterpret_cast<const char*>(instance_begin),
               n * sizeof(InstanceId));
    key.push_back('\x1d');
  }
  const auto* value_begin = support.value_codes.data() + row * n;
  const auto* role_begin = support.role_node_codes.data() + row * r;
  key.append(reinterpret_cast<const char*>(value_begin), n * sizeof(std::uint32_t));
  key.push_back('\x1e');
  key.append(reinterpret_cast<const char*>(role_begin), r * sizeof(std::uint32_t));
  return key;
}

FullLatticeIndex::PackedSupport FullLatticeIndex::compactSupport(PackedSupport support) const {
  if (config_.preserve_instance_identity) return support;
  if (support.row_count <= 1) {
    if (support.counts.empty() && support.row_count == 1) {
      support.counts.assign(1, 1);
    }
    return support;
  }

  const std::size_t n = data_.size();
  const std::size_t r = global_roles_.size();
  struct Entry {
    std::size_t row = 0;
    Count count = 0;
  };
  std::unordered_map<std::string, Entry> groups;
  groups.reserve(support.row_count * 2);
  std::vector<std::string> order;
  order.reserve(support.row_count);
  for (std::size_t row = 0; row < support.row_count; ++row) {
    auto key = compactRowKey(support, row);
    const Count weight = rowWeight(support, row);
    auto found = groups.find(key);
    if (found == groups.end()) {
      order.push_back(key);
      groups.emplace(std::move(key), Entry{row, weight});
    } else {
      found->second.count += weight;
    }
  }
  if (groups.size() == support.row_count && !support.counts.empty()) {
    return support;
  }

  PackedSupport out;
  out.row_count = order.size();
  out.instance_ids.reserve(out.row_count * n);
  out.value_codes.reserve(out.row_count * n);
  out.role_node_codes.reserve(out.row_count * r);
  out.counts.reserve(out.row_count);
  for (const auto& key : order) {
    const auto& entry = groups.at(key);
    const std::size_t row = entry.row;
    const auto instance_begin = support.instance_ids.begin() + static_cast<std::ptrdiff_t>(row * n);
    const auto value_begin = support.value_codes.begin() + static_cast<std::ptrdiff_t>(row * n);
    const auto role_begin = support.role_node_codes.begin() + static_cast<std::ptrdiff_t>(row * r);
    out.instance_ids.insert(out.instance_ids.end(), instance_begin,
                            instance_begin + static_cast<std::ptrdiff_t>(n));
    out.value_codes.insert(out.value_codes.end(), value_begin,
                           value_begin + static_cast<std::ptrdiff_t>(n));
    out.role_node_codes.insert(out.role_node_codes.end(), role_begin,
                               role_begin + static_cast<std::ptrdiff_t>(r));
    out.counts.push_back(entry.count);
  }
  return out;
}

std::vector<std::string> FullLatticeIndex::valueKey(const PackedSupport& support,
                                                    std::size_t row,
                                                    Mask mask) const {
  std::vector<std::string> key;
  key.reserve(data_.size());
  const std::size_t n = data_.size();
  for (std::size_t v = 0; v < n; ++v) {
    if ((mask & (Mask{1} << v)) == 0) {
      continue;
    }
    key.push_back(valueForCode(support.value_codes[row * n + v]));
  }
  return key;
}

std::string FullLatticeIndex::valueForCode(std::uint32_t code) const {
  if (code == 0 || code >= value_values_.size()) {
    return "__missing__";
  }
  return value_values_[code];
}

std::size_t FullLatticeIndex::valueTrieNodeCount(
    const std::map<std::vector<std::string>, Count>& counts) {
  std::set<std::vector<std::string>> prefixes;
  prefixes.insert({});
  for (const auto& item : counts) {
    std::vector<std::string> prefix;
    for (const auto& value : item.first) {
      prefix.push_back(value);
      prefixes.insert(prefix);
    }
  }
  return prefixes.size();
}

void FullLatticeIndex::reportProgress(Mask child,
                                      Mask parent_mask,
                                      std::size_t variable,
                                      std::size_t depth,
                                      Count rows) {
  result_.completed_tasks += 1;
  if (!config_.progress_callback) {
    return;
  }
  const std::size_t every = config_.progress_every_tasks == 0 ? 1 : config_.progress_every_tasks;
  const bool should_report = result_.completed_tasks == 1 ||
      result_.completed_tasks == result_.task_order.size() ||
      result_.completed_tasks % every == 0;
  if (!should_report) {
    return;
  }
  ProgressEvent event;
  event.completed_tasks = result_.completed_tasks;
  event.total_tasks = result_.task_order.size();
  event.mask = child;
  event.parent_mask = parent_mask;
  event.variable = variable;
  event.depth = depth;
  event.rows = rows;
  event.resident_tuples = result_.cache_stats.resident_tuples;
  event.peak_resident_tuples = result_.cache_stats.peak_resident_tuples;
  event.resident_results = result_.cache_stats.resident_results;
  event.peak_resident_results = result_.cache_stats.peak_resident_results;
  event.emitted_rows = result_.emitted_rows;
  config_.progress_callback(event);
}

FullLatticeIndex::PackedSupport FullLatticeIndex::rootSupport() const {
  PackedSupport out;
  out.row_count = 1;
  out.instance_ids.assign(data_.size(), kNoInstance);
  out.value_codes.assign(data_.size(), 0);
  out.role_node_codes.assign(global_roles_.size(), 0);
  out.counts.assign(1, 1);
  return out;
}

FullLatticeIndex::PackedSupport FullLatticeIndex::singletonSupport(std::size_t variable) const {
  PackedSupport out;
  out.row_count = data_[variable].instances.size();
  const std::size_t n = data_.size();
  const std::size_t r = global_roles_.size();
  out.instance_ids.assign(out.row_count * n, kNoInstance);
  out.value_codes.assign(out.row_count * n, 0);
  out.role_node_codes.assign(out.row_count * r, 0);
  out.counts.assign(out.row_count, 1);
  for (std::size_t row = 0; row < out.row_count; ++row) {
    const auto& instance = data_[variable].instances[row];
    out.instance_ids[row * n + variable] = instance.id;
    out.value_codes[row * n + variable] = instance.value_code;
    for (const auto& binding : instance.role_node_codes) {
      out.role_node_codes[row * r + binding.first] = binding.second;
    }
  }
  return compactSupport(std::move(out));
}

FullLatticeIndex::PackedSupport FullLatticeIndex::extendSupport(
    const PackedSupport& parent_support,
    Mask parent_mask,
    std::size_t variable,
    std::vector<std::uint8_t>* participation) {
  if (parent_mask != 0 && config_.binary_join_callback)
    config_.binary_join_callback();
  if (participation != nullptr) {
    participation->assign(parent_support.row_count, 0);
  }

  const std::size_t n = data_.size();
  const std::size_t r = global_roles_.size();

  std::vector<std::uint8_t> parent_has_role(r, 0);
  for (std::size_t v = 0; v < data_.size(); ++v) {
    if ((parent_mask & (Mask{1} << v)) == 0) {
      continue;
    }
    for (const auto& role : data_[v].pattern_roles) {
      const auto found = global_role_index_.find(role);
      if (found != global_role_index_.end()) {
        parent_has_role[found->second] = 1;
      }
    }
  }

  std::vector<std::size_t> join_role_indices;
  for (const auto& role : data_[variable].pattern_roles) {
    const auto found = global_role_index_.find(role);
    if (found != global_role_index_.end() && parent_has_role[found->second]) {
      join_role_indices.push_back(found->second);
    }
  }
  std::sort(join_role_indices.begin(), join_role_indices.end());
  join_role_indices.erase(std::unique(join_role_indices.begin(), join_role_indices.end()),
                          join_role_indices.end());

  JoinPartitionTrace partition_trace;
  if (config_.collect_partition_traces) {
    partition_trace.parent_mask = parent_mask;
    partition_trace.child_mask = parent_mask | (Mask{1} << variable);
    partition_trace.variable = data_[variable].id;
    partition_trace.parent_rows = parent_support.row_count;
    partition_trace.candidate_rows = data_[variable].instances.size();
    partition_trace.join_roles = join_role_indices.size();
    partition_trace.cartesian_pairs =
        static_cast<long double>(parent_support.row_count) *
        static_cast<long double>(data_[variable].instances.size());
  }
  auto finish_partition = [&](const std::string& mode,
                              PackedSupport output,
                              std::size_t compatible_pairs) {
    if (parent_mask != 0 && config_.join_work_callback)
      config_.join_work_callback(parent_support.row_count, compatible_pairs);
    auto compacted = compactSupport(std::move(output));
    if (config_.collect_partition_traces) {
      partition_trace.mode = mode;
      partition_trace.compatible_pairs = compatible_pairs;
      partition_trace.output_rows = compacted.row_count;
      result_.partition_traces.push_back(std::move(partition_trace));
    }
    return compacted;
  };

  struct ValuePartitionFeature {
    std::size_t role_index = 0;
    std::string attribute;
  };

  std::vector<ValuePartitionFeature> candidate_features;
  if (config_.enable_value_partition_join && config_.node_attributes &&
      parent_support.row_count >= config_.value_partition_min_parent_rows &&
      !join_role_indices.empty()) {
    for (const auto role_index : join_role_indices) {
      const auto attributes = config_.value_partition_attributes_by_label.find(
          roleLabel(global_roles_[role_index]));
      if (attributes == config_.value_partition_attributes_by_label.end()) {
        continue;
      }
      for (const auto& attribute : attributes->second) {
        if (node_attribute_codes_.count(attribute) != 0) {
          candidate_features.push_back({role_index, attribute});
        }
      }
    }
  }

  auto instance_role_code = [](const InstanceRecord& instance, std::size_t role_index) {
    const auto found = std::lower_bound(
        instance.role_node_codes.begin(), instance.role_node_codes.end(), role_index,
        [](const auto& binding, std::size_t role) { return binding.first < role; });
    return found != instance.role_node_codes.end() && found->first == role_index
        ? found->second
        : std::uint32_t{0};
  };

  auto attribute_code = [&](const ValuePartitionFeature& feature, std::uint32_t node_code) {
    const auto values = node_attribute_codes_.find(feature.attribute);
    if (values == node_attribute_codes_.end() || node_code >= values->second.size()) {
      return std::uint32_t{0};
    }
    return values->second[node_code];
  };

  auto append_code = [](std::string& key, std::uint32_t code) {
    key.append(reinterpret_cast<const char*>(&code), sizeof(code));
  };

  auto support_value_signature = [&](std::size_t row_index,
                                     const std::vector<ValuePartitionFeature>& features) {
    std::string key;
    key.reserve(features.size() * sizeof(std::uint32_t));
    const auto offset = row_index * r;
    for (const auto& feature : features) {
      append_code(key, attribute_code(feature,
                                      parent_support.role_node_codes[offset + feature.role_index]));
    }
    return key;
  };

  auto instance_value_signature = [&](const InstanceRecord& instance,
                                      const std::vector<ValuePartitionFeature>& features) {
    std::string key;
    key.reserve(features.size() * sizeof(std::uint32_t));
    for (const auto& feature : features) {
      append_code(key, attribute_code(feature, instance_role_code(instance, feature.role_index)));
    }
    return key;
  };

  auto support_exact_signature = [&](std::size_t row_index) {
    std::string key;
    key.reserve(join_role_indices.size() * sizeof(std::uint32_t));
    const auto offset = row_index * r;
    for (const auto role_index : join_role_indices) {
      append_code(key, parent_support.role_node_codes[offset + role_index]);
    }
    return key;
  };

  auto instance_exact_signature = [&](const InstanceRecord& instance) {
    std::string key;
    key.reserve(join_role_indices.size() * sizeof(std::uint32_t));
    for (const auto role_index : join_role_indices) {
      append_code(key, instance_role_code(instance, role_index));
    }
    return key;
  };

  auto profile_exact_buckets = [&] {
    struct BucketCounts {
      std::size_t parent_rows = 0;
      std::size_t candidate_rows = 0;
    };
    std::unordered_map<std::string, BucketCounts> buckets;
    const auto reserve_hint = std::min<std::size_t>(
        parent_support.row_count + data_[variable].instances.size(), 1U << 20);
    buckets.reserve(std::max<std::size_t>(1, reserve_hint));
    for (std::size_t row_index = 0; row_index < parent_support.row_count; ++row_index) {
      buckets[support_exact_signature(row_index)].parent_rows += 1;
    }
    for (const auto& instance : data_[variable].instances) {
      buckets[instance_exact_signature(instance)].candidate_rows += 1;
    }
    for (const auto& item : buckets) {
      const auto& counts = item.second;
      if (counts.parent_rows != 0) {
        partition_trace.parent_partition_buckets += 1;
      }
      if (counts.candidate_rows != 0) {
        partition_trace.candidate_partition_buckets += 1;
      }
      if (counts.parent_rows == 0) {
        partition_trace.unmatched_candidate_rows += counts.candidate_rows;
      } else if (counts.candidate_rows == 0) {
        partition_trace.unmatched_parent_rows += counts.parent_rows;
      } else {
        recordPartitionBucket(partition_trace, counts.parent_rows, counts.candidate_rows);
      }
    }
  };

  struct PartitionEstimate {
    long double pair_cost = 0.0L;
    std::size_t left_buckets = 0;
    std::size_t right_buckets = 0;
  };

  const auto sample_limit = std::max<std::size_t>(1, config_.value_partition_selection_sample_rows);
  const auto left_sample_size = std::min(parent_support.row_count, sample_limit);
  const auto right_sample_size = std::min(data_[variable].instances.size(), sample_limit);
  auto estimate_partition = [&](const std::vector<ValuePartitionFeature>& features) {
    std::unordered_map<std::string, std::size_t> left;
    std::unordered_map<std::string, std::size_t> right;
    left.reserve(std::max<std::size_t>(1, left_sample_size * 2));
    right.reserve(std::max<std::size_t>(1, right_sample_size * 2));
    for (std::size_t i = 0; i < left_sample_size; ++i) {
      const auto row = i * parent_support.row_count / left_sample_size;
      left[support_value_signature(row, features)] += 1;
    }
    for (std::size_t i = 0; i < right_sample_size; ++i) {
      const auto local = i * data_[variable].instances.size() / right_sample_size;
      right[instance_value_signature(data_[variable].instances[local], features)] += 1;
    }
    PartitionEstimate estimate;
    estimate.left_buckets = left.size();
    estimate.right_buckets = right.size();
    for (const auto& item : left) {
      const auto found = right.find(item.first);
      if (found != right.end()) {
        estimate.pair_cost += static_cast<long double>(item.second) *
                              static_cast<long double>(found->second);
      }
    }
    return estimate;
  };

  std::vector<ValuePartitionFeature> selected_features;
  if (!candidate_features.empty() && left_sample_size != 0 && right_sample_size != 0) {
    PartitionEstimate current = estimate_partition(selected_features);
    std::vector<std::uint8_t> selected(candidate_features.size(), 0);
    while (selected_features.size() < config_.value_partition_max_features) {
      std::optional<std::size_t> best;
      PartitionEstimate best_estimate;
      long double best_net_gain = 0.0L;
      for (std::size_t i = 0; i < candidate_features.size(); ++i) {
        if (selected[i] != 0) {
          continue;
        }
        auto trial_features = selected_features;
        trial_features.push_back(candidate_features[i]);
        const auto trial = estimate_partition(trial_features);
        if (std::max(trial.left_buckets, trial.right_buckets) >
            config_.value_partition_max_buckets) {
          continue;
        }
        const long double gross_gain = current.pair_cost - trial.pair_cost;
        const long double relative_gain = current.pair_cost <= 0.0L
            ? 0.0L
            : gross_gain / current.pair_cost;
        if (relative_gain < config_.value_partition_min_pair_reduction) {
          continue;
        }
        const auto current_buckets = current.left_buckets + current.right_buckets;
        const auto trial_buckets = trial.left_buckets + trial.right_buckets;
        const long double added_buckets = trial_buckets > current_buckets
            ? static_cast<long double>(trial_buckets - current_buckets)
            : 0.0L;
        const long double net_gain = gross_gain -
            static_cast<long double>(config_.value_partition_bucket_overhead) * added_buckets;
        if (net_gain > best_net_gain) {
          best = i;
          best_estimate = trial;
          best_net_gain = net_gain;
        }
      }
      if (!best) {
        break;
      }
      selected[*best] = 1;
      selected_features.push_back(candidate_features[*best]);
      current = best_estimate;
    }
  }

  const bool use_value_partition = !selected_features.empty();
  const bool use_intersection_buckets = !use_value_partition &&
      parent_support.row_count >= 16384 && join_role_indices.size() >= 2;
  if (config_.collect_partition_traces && !use_value_partition) {
    profile_exact_buckets();
  }
  if (!use_value_partition && !use_intersection_buckets) {
    const std::size_t requested_threads = std::max<std::size_t>(1, config_.worker_threads);
    const bool use_parallel = requested_threads > 1 &&
        parent_support.row_count >= config_.parallel_min_parent_rows;
    const std::size_t worker_count = use_parallel
        ? std::min<std::size_t>(requested_threads, parent_support.row_count)
        : 1;

    struct RowChunkResult {
      FullLatticeIndex::PackedSupport rows;
      RuntimeCacheStats stats;
    };

    auto build_row_chunk = [&](std::size_t begin, std::size_t end, RowChunkResult& chunk) {
      chunk.rows.instance_ids.reserve((end - begin) * n);
      chunk.rows.value_codes.reserve((end - begin) * n);
      chunk.rows.role_node_codes.reserve((end - begin) * r);
      chunk.rows.counts.reserve(end - begin);
      std::unordered_map<std::string, std::size_t> grouped_rows;
      grouped_rows.reserve(std::max<std::size_t>(1, (end - begin) * 2));

      auto append_grouped = [&](std::size_t row_index, const InstanceRecord& candidate) {
        if (config_.preserve_instance_identity) {
          appendMergedTuple(chunk.rows, parent_support, row_index, variable, candidate);
          return;
        }
        std::vector<std::uint32_t> merged_roles(
            parent_support.role_node_codes.begin() + static_cast<std::ptrdiff_t>(row_index * r),
            parent_support.role_node_codes.begin() + static_cast<std::ptrdiff_t>((row_index + 1) * r));
        for (const auto& binding : candidate.role_node_codes) {
          merged_roles[binding.first] = binding.second;
        }

        std::string key;
        key.reserve((n + r) * sizeof(std::uint32_t) +
                    (config_.binding_output_scopes.empty()
                         ? 1
                         : n * sizeof(InstanceId) + 2));
        if (!config_.binding_output_scopes.empty()) {
          for (std::size_t v = 0; v < n; ++v) {
            const auto id = v == variable
                ? candidate.id
                : parent_support.instance_ids[row_index * n + v];
            key.append(reinterpret_cast<const char*>(&id), sizeof(id));
          }
          key.push_back('\x1d');
        }
        for (std::size_t v = 0; v < n; ++v) {
          const auto code = v == variable
              ? candidate.value_code
              : parent_support.value_codes[row_index * n + v];
          key.append(reinterpret_cast<const char*>(&code), sizeof(code));
        }
        key.push_back('\x1e');
        key.append(reinterpret_cast<const char*>(merged_roles.data()), r * sizeof(std::uint32_t));

        const Count weight = rowWeight(parent_support, row_index);
        auto found = grouped_rows.find(key);
        if (found != grouped_rows.end()) {
          chunk.rows.counts[found->second] += weight;
          chunk.stats.grouped_prefix_hits += 1;
          return;
        }

        const auto parent_instance_begin = parent_support.instance_ids.begin() +
            static_cast<std::ptrdiff_t>(row_index * n);
        chunk.rows.instance_ids.insert(chunk.rows.instance_ids.end(), parent_instance_begin,
                                       parent_instance_begin + static_cast<std::ptrdiff_t>(n));
        chunk.rows.instance_ids[chunk.rows.row_count * n + variable] = candidate.id;
        for (std::size_t v = 0; v < n; ++v) {
          const auto code = v == variable
              ? candidate.value_code
              : parent_support.value_codes[row_index * n + v];
          chunk.rows.value_codes.push_back(code);
        }
        chunk.rows.role_node_codes.insert(chunk.rows.role_node_codes.end(),
                                          merged_roles.begin(), merged_roles.end());
        chunk.rows.counts.push_back(weight);
        grouped_rows.emplace(std::move(key), chunk.rows.row_count);
        chunk.rows.row_count += 1;
        chunk.stats.grouped_prefix_inserts += 1;
      };

      for (std::size_t row_index = begin; row_index < end; ++row_index) {
        auto candidates = candidateInstances(parent_support, row_index, variable, chunk.stats);
        bool any = false;
        for (const auto local : candidates) {
          const auto& candidate = data_[variable].instances[local];
          any = true;
          append_grouped(row_index, candidate);
        }
        if (any && participation != nullptr) {
          (*participation)[row_index] = 1;
        }
      }
    };

    std::vector<RowChunkResult> chunks(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&, worker] {
        const std::size_t begin =
            parent_support.row_count * worker / worker_count;
        const std::size_t end =
            parent_support.row_count * (worker + 1) / worker_count;
        build_row_chunk(begin, end, chunks[worker]);
      });
    }
    for (auto& worker : workers) worker.join();

    std::size_t total_rows = 0;
    std::size_t compatible_pairs = 0;
    for (const auto& chunk : chunks) {
      total_rows += chunk.rows.row_count;
      compatible_pairs += chunk.stats.live_candidates_emitted;
      result_.cache_stats.trie_nodes_visited += chunk.stats.trie_nodes_visited;
      result_.cache_stats.trie_subtrees_skipped += chunk.stats.trie_subtrees_skipped;
      result_.cache_stats.live_candidates_emitted += chunk.stats.live_candidates_emitted;
      result_.cache_stats.anchor_buckets_used += chunk.stats.anchor_buckets_used;
      result_.cache_stats.anchor_candidates_scanned += chunk.stats.anchor_candidates_scanned;
      result_.cache_stats.grouped_prefix_hits += chunk.stats.grouped_prefix_hits;
      result_.cache_stats.grouped_prefix_inserts += chunk.stats.grouped_prefix_inserts;
    }

    PackedSupport out;
    out.row_count = total_rows;
    out.instance_ids.reserve(total_rows * n);
    out.value_codes.reserve(total_rows * n);
    out.role_node_codes.reserve(total_rows * r);
    out.counts.reserve(total_rows);
    for (auto& chunk : chunks) {
      out.instance_ids.insert(out.instance_ids.end(),
                              std::make_move_iterator(chunk.rows.instance_ids.begin()),
                              std::make_move_iterator(chunk.rows.instance_ids.end()));
      out.value_codes.insert(out.value_codes.end(),
                             std::make_move_iterator(chunk.rows.value_codes.begin()),
                             std::make_move_iterator(chunk.rows.value_codes.end()));
      out.role_node_codes.insert(out.role_node_codes.end(),
                                 std::make_move_iterator(chunk.rows.role_node_codes.begin()),
                                 std::make_move_iterator(chunk.rows.role_node_codes.end()));
      out.counts.insert(out.counts.end(),
                        std::make_move_iterator(chunk.rows.counts.begin()),
                        std::make_move_iterator(chunk.rows.counts.end()));
    }
    return finish_partition("exact-trie", std::move(out), compatible_pairs);
  }

  if (use_value_partition) {
    struct ValueJoinBucket {
      std::vector<std::size_t> candidate_rows;
      std::vector<std::size_t> parent_rows;
    };


    std::vector<ValueJoinBucket> value_buckets;
    std::unordered_map<std::string, std::size_t> value_bucket_index;
    value_bucket_index.reserve(std::min<std::size_t>(
        config_.value_partition_max_buckets,
        std::max<std::size_t>(1, data_[variable].instances.size())) * 2);
    for (std::size_t local = 0; local < data_[variable].instances.size(); ++local) {
      const auto& instance = data_[variable].instances[local];
      auto value_key = instance_value_signature(instance, selected_features);
      auto found = value_bucket_index.find(value_key);
      if (found == value_bucket_index.end()) {
        const auto next = value_buckets.size();
        value_buckets.push_back({});
        found = value_bucket_index.emplace(std::move(value_key), next).first;
      }
      value_buckets[found->second].candidate_rows.push_back(local);
    }
    std::unordered_set<std::string> parent_partition_keys;
    if (config_.collect_partition_traces) {
      parent_partition_keys.reserve(std::max<std::size_t>(1, std::min(
          parent_support.row_count, config_.value_partition_max_buckets)));
    }
    for (std::size_t row_index = 0; row_index < parent_support.row_count; ++row_index) {
      auto value_key = support_value_signature(row_index, selected_features);
      if (config_.collect_partition_traces) {
        parent_partition_keys.insert(value_key);
      }
      const auto found = value_bucket_index.find(value_key);
      if (found != value_bucket_index.end()) {
        value_buckets[found->second].parent_rows.push_back(row_index);
      } else if (config_.collect_partition_traces) {
        partition_trace.unmatched_parent_rows += 1;
      }
    }

    std::vector<std::size_t> active_buckets;
    active_buckets.reserve(value_buckets.size());
    for (std::size_t i = 0; i < value_buckets.size(); ++i) {
      if (!value_buckets[i].parent_rows.empty()) {
        active_buckets.push_back(i);
      }
    }

    result_.cache_stats.value_partition_joins += 1;
    result_.cache_stats.value_partition_features_selected += selected_features.size();
    result_.cache_stats.value_partition_buckets_built += active_buckets.size();
    result_.cache_stats.value_partition_parent_rows += parent_support.row_count;
    result_.cache_stats.value_partition_candidate_rows += data_[variable].instances.size();
    for (const auto& feature : selected_features) {
      result_.cache_stats.value_partition_feature_uses[
          global_roles_[feature.role_index] + "." + feature.attribute] += 1;
      if (config_.collect_partition_traces) {
        partition_trace.value_features.push_back(
            global_roles_[feature.role_index] + "." + feature.attribute);
      }
    }
    if (config_.collect_partition_traces) {
      partition_trace.parent_partition_buckets = parent_partition_keys.size();
      partition_trace.candidate_partition_buckets = value_buckets.size();
      for (const auto bucket_index : active_buckets) {
        const auto& bucket = value_buckets[bucket_index];
        recordPartitionBucket(partition_trace, bucket.parent_rows.size(),
                              bucket.candidate_rows.size());
      }
      for (const auto& bucket : value_buckets) {
        if (bucket.parent_rows.empty()) {
          partition_trace.unmatched_candidate_rows += bucket.candidate_rows.size();
        }
      }
    }

    const std::size_t requested_threads = std::max<std::size_t>(1, config_.worker_threads);
    const bool use_parallel = requested_threads > 1 &&
        parent_support.row_count >= config_.parallel_min_parent_rows && active_buckets.size() > 1;
    const std::size_t worker_count = use_parallel
        ? std::min<std::size_t>(requested_threads, active_buckets.size())
        : 1;

    struct ValueChunkResult {
      FullLatticeIndex::PackedSupport rows;
      RuntimeCacheStats stats;
    };

    auto build_row_bounds = [&](std::size_t row_index,
                                std::vector<std::uint32_t>& bound_role_codes,
                                std::vector<std::uint32_t>& forbidden_node_codes) {
      bound_role_codes.assign(r, 0);
      forbidden_node_codes.clear();
      forbidden_node_codes.reserve(r);
      const auto offset = row_index * r;
      for (std::size_t role = 0; role < r; ++role) {
        const auto code = parent_support.role_node_codes[offset + role];
        bound_role_codes[role] = code;
        if (code != 0) {
          forbidden_node_codes.push_back(code);
        }
      }
      std::sort(forbidden_node_codes.begin(), forbidden_node_codes.end());
      forbidden_node_codes.erase(
          std::unique(forbidden_node_codes.begin(), forbidden_node_codes.end()),
          forbidden_node_codes.end());
    };

    auto build_value_chunk = [&](std::size_t begin, std::size_t end,
                                 ValueChunkResult& chunk) {
      std::size_t parent_rows_in_chunk = 0;
      for (std::size_t i = begin; i < end; ++i) {
        parent_rows_in_chunk += value_buckets[active_buckets[i]].parent_rows.size();
      }
      chunk.rows.instance_ids.reserve(parent_rows_in_chunk * n);
      chunk.rows.value_codes.reserve(parent_rows_in_chunk * n);
      chunk.rows.role_node_codes.reserve(parent_rows_in_chunk * r);
      chunk.rows.counts.reserve(parent_rows_in_chunk);
      std::unordered_map<std::string, std::size_t> grouped_rows;
      grouped_rows.reserve(std::max<std::size_t>(1, parent_rows_in_chunk * 2));

      auto append_grouped = [&](std::size_t row_index, const InstanceRecord& candidate) {
        if (config_.preserve_instance_identity) {
          appendMergedTuple(chunk.rows, parent_support, row_index, variable, candidate);
          return;
        }
        std::vector<std::uint32_t> merged_roles(
            parent_support.role_node_codes.begin() + static_cast<std::ptrdiff_t>(row_index * r),
            parent_support.role_node_codes.begin() + static_cast<std::ptrdiff_t>((row_index + 1) * r));
        for (const auto& binding : candidate.role_node_codes) {
          merged_roles[binding.first] = binding.second;
        }

        std::string key;
        key.reserve((n + r) * sizeof(std::uint32_t) +
                    (config_.binding_output_scopes.empty()
                         ? 1
                         : n * sizeof(InstanceId) + 2));
        if (!config_.binding_output_scopes.empty()) {
          for (std::size_t v = 0; v < n; ++v) {
            const auto id = v == variable
                ? candidate.id
                : parent_support.instance_ids[row_index * n + v];
            key.append(reinterpret_cast<const char*>(&id), sizeof(id));
          }
          key.push_back('\x1d');
        }
        for (std::size_t v = 0; v < n; ++v) {
          const auto code = v == variable
              ? candidate.value_code
              : parent_support.value_codes[row_index * n + v];
          append_code(key, code);
        }
        key.push_back('\x1e');
        key.append(reinterpret_cast<const char*>(merged_roles.data()),
                   r * sizeof(std::uint32_t));

        const Count weight = rowWeight(parent_support, row_index);
        const auto found = grouped_rows.find(key);
        if (found != grouped_rows.end()) {
          chunk.rows.counts[found->second] += weight;
          chunk.stats.grouped_prefix_hits += 1;
          return;
        }

        const auto parent_instance_begin = parent_support.instance_ids.begin() +
            static_cast<std::ptrdiff_t>(row_index * n);
        chunk.rows.instance_ids.insert(chunk.rows.instance_ids.end(), parent_instance_begin,
                                       parent_instance_begin + static_cast<std::ptrdiff_t>(n));
        chunk.rows.instance_ids[chunk.rows.row_count * n + variable] = candidate.id;
        for (std::size_t v = 0; v < n; ++v) {
          chunk.rows.value_codes.push_back(v == variable
              ? candidate.value_code
              : parent_support.value_codes[row_index * n + v]);
        }
        chunk.rows.role_node_codes.insert(chunk.rows.role_node_codes.end(),
                                          merged_roles.begin(), merged_roles.end());
        chunk.rows.counts.push_back(weight);
        grouped_rows.emplace(std::move(key), chunk.rows.row_count);
        chunk.rows.row_count += 1;
        chunk.stats.grouped_prefix_inserts += 1;
      };

      std::vector<std::uint32_t> bound_role_codes;
      std::vector<std::uint32_t> forbidden_node_codes;
      for (std::size_t i = begin; i < end; ++i) {
        const auto& bucket = value_buckets[active_buckets[i]];
        for (const auto row_index : bucket.parent_rows) {
          build_row_bounds(row_index, bound_role_codes, forbidden_node_codes);
          bool any = false;
          for (const auto local : bucket.candidate_rows) {
            chunk.stats.value_partition_pair_checks += 1;
            const auto& candidate = data_[variable].instances[local];
            if (!instanceCompatible(candidate, bound_role_codes, forbidden_node_codes)) {
              continue;
            }
            chunk.stats.value_partition_compatible_pairs += 1;
            chunk.stats.live_candidates_emitted += 1;
            any = true;
            append_grouped(row_index, candidate);
          }
          if (any && participation != nullptr) {
            (*participation)[row_index] = 1;
          }
        }
      }
    };

    std::vector<ValueChunkResult> chunks(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&, worker] {
        const std::size_t begin =
            active_buckets.size() * worker / worker_count;
        const std::size_t end =
            active_buckets.size() * (worker + 1) / worker_count;
        build_value_chunk(begin, end, chunks[worker]);
      });
    }
    for (auto& worker : workers) worker.join();

    std::size_t total_rows = 0;
    std::size_t compatible_pairs = 0;
    for (const auto& chunk : chunks) {
      total_rows += chunk.rows.row_count;
      compatible_pairs += chunk.stats.value_partition_compatible_pairs;
      result_.cache_stats.live_candidates_emitted += chunk.stats.live_candidates_emitted;
      result_.cache_stats.value_partition_pair_checks +=
          chunk.stats.value_partition_pair_checks;
      result_.cache_stats.value_partition_compatible_pairs +=
          chunk.stats.value_partition_compatible_pairs;
      result_.cache_stats.grouped_prefix_hits += chunk.stats.grouped_prefix_hits;
      result_.cache_stats.grouped_prefix_inserts += chunk.stats.grouped_prefix_inserts;
    }

    PackedSupport out;
    out.row_count = total_rows;
    out.instance_ids.reserve(total_rows * n);
    out.value_codes.reserve(total_rows * n);
    out.role_node_codes.reserve(total_rows * r);
    out.counts.reserve(total_rows);
    for (auto& chunk : chunks) {
      out.instance_ids.insert(out.instance_ids.end(),
                              std::make_move_iterator(chunk.rows.instance_ids.begin()),
                              std::make_move_iterator(chunk.rows.instance_ids.end()));
      out.value_codes.insert(out.value_codes.end(),
                             std::make_move_iterator(chunk.rows.value_codes.begin()),
                             std::make_move_iterator(chunk.rows.value_codes.end()));
      out.role_node_codes.insert(out.role_node_codes.end(),
                                 std::make_move_iterator(chunk.rows.role_node_codes.begin()),
                                 std::make_move_iterator(chunk.rows.role_node_codes.end()));
      out.counts.insert(out.counts.end(),
                        std::make_move_iterator(chunk.rows.counts.begin()),
                        std::make_move_iterator(chunk.rows.counts.end()));
    }
    return finish_partition("value", std::move(out), compatible_pairs);
  }

  struct PrefixBucket {
    std::vector<std::size_t> rows;
    std::vector<std::uint32_t> bound_role_codes;
    std::vector<std::uint32_t> common_forbidden_node_codes;
  };

  std::vector<PrefixBucket> buckets;
  std::unordered_map<std::string, std::size_t> bucket_index;
  buckets.reserve(parent_support.row_count == 0 ? 1 : parent_support.row_count);
  bucket_index.reserve(parent_support.row_count == 0 ? 1 : parent_support.row_count * 2);
  for (std::size_t row_index = 0; row_index < parent_support.row_count; ++row_index) {
    std::string key;
    key.reserve(join_role_indices.size() * sizeof(std::uint32_t));
    const auto offset = row_index * r;
    for (const auto role_index : join_role_indices) {
      const auto code = parent_support.role_node_codes[offset + role_index];
      key.append(reinterpret_cast<const char*>(&code), sizeof(code));
    }

    auto found = bucket_index.find(key);
    if (found == bucket_index.end()) {
      PrefixBucket bucket;
      bucket.bound_role_codes.assign(r, 0);
      for (const auto role_index : join_role_indices) {
        const auto code = parent_support.role_node_codes[offset + role_index];
        bucket.bound_role_codes[role_index] = code;
        if (code != 0) {
          bucket.common_forbidden_node_codes.push_back(code);
        }
      }
      std::sort(bucket.common_forbidden_node_codes.begin(),
                bucket.common_forbidden_node_codes.end());
      bucket.common_forbidden_node_codes.erase(
          std::unique(bucket.common_forbidden_node_codes.begin(),
                      bucket.common_forbidden_node_codes.end()),
          bucket.common_forbidden_node_codes.end());
      bucket.rows.push_back(row_index);
      const auto next = buckets.size();
      buckets.push_back(std::move(bucket));
      bucket_index.emplace(std::move(key), next);
    } else {
      buckets[found->second].rows.push_back(row_index);
    }
  }

  result_.cache_stats.intersection_buckets_built += buckets.size();
  result_.cache_stats.intersection_bucket_rows += parent_support.row_count;

  const std::size_t requested_threads = std::max<std::size_t>(1, config_.worker_threads);
  const bool use_parallel = requested_threads > 1 &&
      parent_support.row_count >= config_.parallel_min_parent_rows && buckets.size() > 1;
  const std::size_t worker_count = use_parallel
      ? std::min<std::size_t>(requested_threads, buckets.size())
      : 1;

  struct ChunkResult {
    FullLatticeIndex::PackedSupport rows;
    RuntimeCacheStats stats;
    std::size_t compatible_pairs = 0;
  };

  auto build_row_bounds = [&](std::size_t row_index,
                              std::vector<std::uint32_t>& bound_role_codes,
                              std::vector<std::uint32_t>& forbidden_node_codes) {
    bound_role_codes.assign(r, 0);
    forbidden_node_codes.clear();
    forbidden_node_codes.reserve(r);
    const auto offset = row_index * r;
    for (std::size_t role = 0; role < r; ++role) {
      const auto code = parent_support.role_node_codes[offset + role];
      bound_role_codes[role] = code;
      if (code != 0) {
        forbidden_node_codes.push_back(code);
      }
    }
    std::sort(forbidden_node_codes.begin(), forbidden_node_codes.end());
    forbidden_node_codes.erase(std::unique(forbidden_node_codes.begin(), forbidden_node_codes.end()),
                               forbidden_node_codes.end());
  };

  auto build_chunk = [&](std::size_t begin, std::size_t end, ChunkResult& chunk) {
    std::size_t parent_rows_in_chunk = 0;
    for (std::size_t bucket = begin; bucket < end; ++bucket) {
      parent_rows_in_chunk += buckets[bucket].rows.size();
    }
    chunk.rows.instance_ids.reserve(parent_rows_in_chunk * n);
    chunk.rows.value_codes.reserve(parent_rows_in_chunk * n);
    chunk.rows.role_node_codes.reserve(parent_rows_in_chunk * r);
    chunk.rows.counts.reserve(parent_rows_in_chunk);

    std::unordered_map<std::string, std::size_t> grouped_rows;
    grouped_rows.reserve(std::max<std::size_t>(1, parent_rows_in_chunk * 2));

    auto append_grouped = [&](std::size_t row_index, const InstanceRecord& candidate) {
      if (config_.preserve_instance_identity) {
        appendMergedTuple(chunk.rows, parent_support, row_index, variable, candidate);
        return;
      }
      std::vector<std::uint32_t> merged_roles(
          parent_support.role_node_codes.begin() + static_cast<std::ptrdiff_t>(row_index * r),
          parent_support.role_node_codes.begin() + static_cast<std::ptrdiff_t>((row_index + 1) * r));
      for (const auto& binding : candidate.role_node_codes) {
        merged_roles[binding.first] = binding.second;
      }

      std::string key;
      key.reserve((n + r) * sizeof(std::uint32_t) +
                  (config_.binding_output_scopes.empty()
                       ? 1
                       : n * sizeof(InstanceId) + 2));
      if (!config_.binding_output_scopes.empty()) {
        for (std::size_t v = 0; v < n; ++v) {
          const auto id = v == variable
              ? candidate.id
              : parent_support.instance_ids[row_index * n + v];
          key.append(reinterpret_cast<const char*>(&id), sizeof(id));
        }
        key.push_back('\x1d');
      }
      for (std::size_t v = 0; v < n; ++v) {
        const auto code = v == variable
            ? candidate.value_code
            : parent_support.value_codes[row_index * n + v];
        key.append(reinterpret_cast<const char*>(&code), sizeof(code));
      }
      key.push_back('');
      key.append(reinterpret_cast<const char*>(merged_roles.data()), r * sizeof(std::uint32_t));

      const Count weight = rowWeight(parent_support, row_index);
      auto found = grouped_rows.find(key);
      if (found != grouped_rows.end()) {
        chunk.rows.counts[found->second] += weight;
        chunk.stats.grouped_prefix_hits += 1;
        return;
      }

      const auto parent_instance_begin = parent_support.instance_ids.begin() +
          static_cast<std::ptrdiff_t>(row_index * n);
      chunk.rows.instance_ids.insert(chunk.rows.instance_ids.end(), parent_instance_begin,
                                     parent_instance_begin + static_cast<std::ptrdiff_t>(n));
      chunk.rows.instance_ids[chunk.rows.row_count * n + variable] = candidate.id;

      for (std::size_t v = 0; v < n; ++v) {
        const auto code = v == variable
            ? candidate.value_code
            : parent_support.value_codes[row_index * n + v];
        chunk.rows.value_codes.push_back(code);
      }
      chunk.rows.role_node_codes.insert(chunk.rows.role_node_codes.end(),
                                        merged_roles.begin(), merged_roles.end());
      chunk.rows.counts.push_back(weight);
      grouped_rows.emplace(std::move(key), chunk.rows.row_count);
      chunk.rows.row_count += 1;
      chunk.stats.grouped_prefix_inserts += 1;
    };

    std::vector<std::uint32_t> row_bound_role_codes;
    std::vector<std::uint32_t> row_forbidden_node_codes;
    for (std::size_t bucket_index = begin; bucket_index < end; ++bucket_index) {
      const auto& bucket = buckets[bucket_index];
      if (bucket.rows.size() == 1) {
        const auto row_index = bucket.rows.front();
        auto candidates = candidateInstances(parent_support, row_index, variable, chunk.stats);
        chunk.compatible_pairs += candidates.size();
        bool any = false;
        for (const auto local : candidates) {
          const auto& candidate = data_[variable].instances[local];
          any = true;
          append_grouped(row_index, candidate);
        }
        if (any && participation != nullptr) {
          (*participation)[row_index] = 1;
        }
        continue;
      }

      auto candidates = candidateInstancesForBounds(variable,
                                                    bucket.bound_role_codes,
                                                    bucket.common_forbidden_node_codes,
                                                    chunk.stats);
      chunk.stats.intersection_candidate_queries += 1;
      for (const auto row_index : bucket.rows) {
        build_row_bounds(row_index, row_bound_role_codes, row_forbidden_node_codes);
        bool any = false;
        for (const auto local : candidates) {
          const auto& candidate = data_[variable].instances[local];
          if (!instanceCompatible(candidate, row_bound_role_codes, row_forbidden_node_codes)) {
            continue;
          }
          chunk.compatible_pairs += 1;
          any = true;
          append_grouped(row_index, candidate);
        }
        if (any && participation != nullptr) {
          (*participation)[row_index] = 1;
        }
      }
    }
  };

  std::vector<ChunkResult> chunks(worker_count);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker] {
      const std::size_t begin = buckets.size() * worker / worker_count;
      const std::size_t end = buckets.size() * (worker + 1) / worker_count;
      build_chunk(begin, end, chunks[worker]);
    });
  }
  for (auto& worker : workers) worker.join();

  std::size_t total_rows = 0;
  std::size_t compatible_pairs = 0;
  for (const auto& chunk : chunks) {
    total_rows += chunk.rows.row_count;
    compatible_pairs += chunk.compatible_pairs;
    result_.cache_stats.trie_nodes_visited += chunk.stats.trie_nodes_visited;
    result_.cache_stats.trie_subtrees_skipped += chunk.stats.trie_subtrees_skipped;
    result_.cache_stats.live_candidates_emitted += chunk.stats.live_candidates_emitted;
    result_.cache_stats.anchor_buckets_used += chunk.stats.anchor_buckets_used;
    result_.cache_stats.anchor_candidates_scanned += chunk.stats.anchor_candidates_scanned;
    result_.cache_stats.intersection_candidate_queries += chunk.stats.intersection_candidate_queries;
    result_.cache_stats.grouped_prefix_hits += chunk.stats.grouped_prefix_hits;
    result_.cache_stats.grouped_prefix_inserts += chunk.stats.grouped_prefix_inserts;
  }

  PackedSupport out;
  out.row_count = total_rows;
  out.instance_ids.reserve(total_rows * n);
  out.value_codes.reserve(total_rows * n);
  out.role_node_codes.reserve(total_rows * r);
  out.counts.reserve(total_rows);
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    out.instance_ids.insert(out.instance_ids.end(),
                            std::make_move_iterator(chunks[worker].rows.instance_ids.begin()),
                            std::make_move_iterator(chunks[worker].rows.instance_ids.end()));
    out.value_codes.insert(out.value_codes.end(),
                           std::make_move_iterator(chunks[worker].rows.value_codes.begin()),
                           std::make_move_iterator(chunks[worker].rows.value_codes.end()));
    out.role_node_codes.insert(out.role_node_codes.end(),
                               std::make_move_iterator(chunks[worker].rows.role_node_codes.begin()),
                               std::make_move_iterator(chunks[worker].rows.role_node_codes.end()));
    out.counts.insert(out.counts.end(),
                      std::make_move_iterator(chunks[worker].rows.counts.begin()),
                      std::make_move_iterator(chunks[worker].rows.counts.end()));
  }

  return finish_partition("exact-intersection", std::move(out), compatible_pairs);
}

std::vector<std::size_t> FullLatticeIndex::candidateInstances(const PackedSupport& support,
                                                              std::size_t row_index,
                                                              std::size_t variable,
                                                              RuntimeCacheStats& stats) const {
  const std::size_t r = global_roles_.size();
  std::vector<std::uint32_t> bound_role_codes(r, 0);
  std::vector<std::uint32_t> forbidden_node_codes;
  forbidden_node_codes.reserve(r);
  const auto offset = row_index * r;
  for (std::size_t role = 0; role < r; ++role) {
    const auto code = support.role_node_codes[offset + role];
    bound_role_codes[role] = code;
    if (code != 0) {
      forbidden_node_codes.push_back(code);
    }
  }
  std::sort(forbidden_node_codes.begin(), forbidden_node_codes.end());
  forbidden_node_codes.erase(std::unique(forbidden_node_codes.begin(), forbidden_node_codes.end()),
                             forbidden_node_codes.end());
  return candidateInstancesForBounds(variable, bound_role_codes, forbidden_node_codes, stats);
}

std::vector<std::size_t> FullLatticeIndex::candidateInstancesForBounds(
    std::size_t variable,
    const std::vector<std::uint32_t>& bound_role_codes,
    const std::vector<std::uint32_t>& forbidden_node_codes,
    RuntimeCacheStats& stats) const {
  std::vector<std::size_t> out;
  const auto& variable_data = data_[variable];
  if (anchor_role_index_ && variable_data.has_anchor_role) {
    const auto anchor_code = bound_role_codes[*anchor_role_index_];
    if (anchor_code != 0) {
      stats.anchor_buckets_used += 1;
      auto bucket_trie = variable_data.anchor_tries.find(anchor_code);
      if (bucket_trie == variable_data.anchor_tries.end()) {
        return out;
      }
      stats.anchor_candidates_scanned += bucket_trie->second.nodes.empty()
          ? 0
          : bucket_trie->second.nodes.front().subtree_count;
      queryTrie(variable, bucket_trie->second, 0, 0, bound_role_codes,
                forbidden_node_codes, out, stats);
      stats.live_candidates_emitted += out.size();
      return out;
    }
  }

  queryTrie(variable, variable_data.trie, 0, 0, bound_role_codes, forbidden_node_codes, out, stats);
  stats.live_candidates_emitted += out.size();
  return out;
}

bool FullLatticeIndex::instanceCompatible(
    const InstanceRecord& instance,
    const std::vector<std::uint32_t>& bound_role_codes,
    const std::vector<std::uint32_t>& forbidden_node_codes) const {
  for (const auto& binding : instance.role_node_codes) {
    const auto bound = bound_role_codes[binding.first];
    if (bound != 0) {
      if (bound != binding.second) {
        return false;
      }
      continue;
    }
    if (std::binary_search(forbidden_node_codes.begin(), forbidden_node_codes.end(), binding.second)) {
      return false;
    }
  }
  return true;
}

void FullLatticeIndex::queryTrie(std::size_t variable,
                                 const VariableTrie& trie,
                                 std::size_t trie_node,
                                 std::size_t depth,
                                 const std::vector<std::uint32_t>& bound_role_codes,
                                 const std::vector<std::uint32_t>& forbidden_node_codes,
                                 std::vector<std::size_t>& out,
                                 RuntimeCacheStats& stats) const {
  stats.trie_nodes_visited += 1;
  const auto& node = trie.nodes[trie_node];
  if (depth == trie.role_order.size()) {
    const auto begin = trie.leaf_instances.begin() + static_cast<std::ptrdiff_t>(node.instance_begin);
    const auto end = begin + static_cast<std::ptrdiff_t>(node.instance_count);
    out.insert(out.end(), begin, end);
    return;
  }

  const auto edge_begin = trie.edges.begin() + static_cast<std::ptrdiff_t>(node.edge_begin);
  const auto edge_end = edge_begin + static_cast<std::ptrdiff_t>(node.edge_count);
  const auto& role = trie.role_order[depth];
  const auto role_found = global_role_index_.find(role);
  if (role_found == global_role_index_.end()) {
    return;
  }
  const auto bound = bound_role_codes[role_found->second];
  if (bound != 0) {
    const auto edge = std::lower_bound(edge_begin, edge_end, bound,
                                       [](const TrieEdge& item, std::uint32_t value) {
                                         return item.value < value;
                                       });
    if (edge == edge_end || edge->value != bound) {
      return;
    }
    queryTrie(variable, trie, edge->child, depth + 1, bound_role_codes, forbidden_node_codes, out, stats);
    return;
  }

  for (auto edge = edge_begin; edge != edge_end; ++edge) {
    if (std::binary_search(forbidden_node_codes.begin(), forbidden_node_codes.end(), edge->value)) {
      stats.trie_subtrees_skipped += edge->subtree_count;
      continue;
    }
    queryTrie(variable, trie, edge->child, depth + 1, bound_role_codes, forbidden_node_codes, out, stats);
  }
}

void FullLatticeIndex::appendMergedTuple(PackedSupport& out,
                                         const PackedSupport& parent,
                                         std::size_t parent_row,
                                         std::size_t variable,
                                         const InstanceRecord& candidate) const {
  const std::size_t n = data_.size();
  const std::size_t r = global_roles_.size();
  const auto parent_instance_begin = parent.instance_ids.begin() +
      static_cast<std::ptrdiff_t>(parent_row * n);
  out.instance_ids.insert(out.instance_ids.end(), parent_instance_begin,
                          parent_instance_begin + static_cast<std::ptrdiff_t>(n));
  out.instance_ids[out.row_count * n + variable] = candidate.id;

  const auto parent_value_begin = parent.value_codes.begin() +
      static_cast<std::ptrdiff_t>(parent_row * n);
  out.value_codes.insert(out.value_codes.end(), parent_value_begin,
                         parent_value_begin + static_cast<std::ptrdiff_t>(n));
  out.value_codes[out.row_count * n + variable] = candidate.value_code;

  const auto parent_role_begin = parent.role_node_codes.begin() +
      static_cast<std::ptrdiff_t>(parent_row * r);
  out.role_node_codes.insert(out.role_node_codes.end(), parent_role_begin,
                             parent_role_begin + static_cast<std::ptrdiff_t>(r));
  for (const auto& binding : candidate.role_node_codes) {
    out.role_node_codes[out.row_count * r + binding.first] = binding.second;
  }
  out.counts.push_back(rowWeight(parent, parent_row));
  out.row_count += 1;
}

FullLatticeIndex::PackedSupport FullLatticeIndex::packSupport(const std::vector<TupleRow>& rows) const {
  PackedSupport out;
  out.row_count = rows.size();
  const std::size_t n = data_.size();
  const std::size_t r = global_roles_.size();
  out.instance_ids.assign(out.row_count * n, kNoInstance);
  out.value_codes.assign(out.row_count * n, 0);
  out.role_node_codes.assign(out.row_count * r, 0);
  out.counts.assign(out.row_count, 1);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    for (std::size_t v = 0; v < std::min(n, rows[row].instance_ids.size()); ++v) {
      out.instance_ids[row * n + v] = rows[row].instance_ids[v];
      if (rows[row].instance_ids[v] != kNoInstance) {
        for (const auto& instance : data_[v].instances) {
          if (instance.id == rows[row].instance_ids[v]) {
            out.value_codes[row * n + v] = instance.value_code;
            break;
          }
        }
      }
    }
    for (const auto& binding : rows[row].role_nodes) {
      auto role_found = global_role_index_.find(binding.role);
      auto node_found = node_code_.find(binding.node);
      if (role_found != global_role_index_.end() && node_found != node_code_.end()) {
        out.role_node_codes[row * r + role_found->second] = node_found->second;
      }
    }
  }
  return compactSupport(std::move(out));
}

std::vector<TupleRow> FullLatticeIndex::unpackSupport(const PackedSupport& support) const {
  std::vector<TupleRow> out;
  out.reserve(support.row_count);
  const std::size_t n = data_.size();
  const std::size_t r = global_roles_.size();
  for (std::size_t row = 0; row < support.row_count; ++row) {
    TupleRow tuple;
    tuple.instance_ids.assign(support.instance_ids.begin() + static_cast<std::ptrdiff_t>(row * n),
                              support.instance_ids.begin() + static_cast<std::ptrdiff_t>((row + 1) * n));
    for (std::size_t role = 0; role < r; ++role) {
      const auto code = support.role_node_codes[row * r + role];
      if (code != 0) {
        tuple.role_nodes.push_back({global_roles_[role], node_values_[code]});
      }
    }
    out.push_back(std::move(tuple));
  }
  return out;
}

DeriveResult FullLatticeIndex::deriveFromSuperset(Mask target, std::size_t donor_variable) const {
  const Mask superset = target | (Mask{1} << donor_variable);
  if (superset >= result_.supports.size() || result_.supports[superset].empty()) {
    throw std::runtime_error("deriveFromSuperset requires a materialized superset support");
  }
  return deriveFromSupersetDirect(target, donor_variable);
}

DeriveResult FullLatticeIndex::deriveFromSupersetDirect(Mask target, std::size_t donor_variable) const {
  if ((target & (Mask{1} << donor_variable)) != 0) {
    throw std::runtime_error("derive donor variable must not already be in target");
  }
  const Mask superset = target | (Mask{1} << donor_variable);
  if (superset >= result_.supports.size()) {
    throw std::runtime_error("derive superset is out of range");
  }

  DeriveResult out;
  out.target = target;
  out.donor_variable = donor_variable;
  out.superset = superset;
  out.key_roles = sharedRoles(target, donor_variable);

  std::set<RoleId> key_role_set(out.key_roles.begin(), out.key_roles.end());
  std::size_t private_role_count = 0;
  for (const auto& role : rolesInMask(target)) {
    if (key_role_set.count(role) == 0) {
      private_role_count += 1;
    }
  }
  const std::size_t threshold = private_role_count + 1;

  std::unordered_map<std::string, std::vector<std::set<NodeId>>> witness_sets;
  for (const auto& instance : data_[donor_variable].instances) {
    const auto key = deriveKey(instance, out.key_roles);
    std::set<NodeId> private_nodes;
    for (const auto& binding : instance.roles) {
      if (key_role_set.count(binding.role) == 0) {
        private_nodes.insert(binding.node);
      }
    }
    bool disjoint = true;
    for (const auto& existing : witness_sets[key]) {
      for (const auto& node : private_nodes) {
        if (existing.count(node) != 0) {
          disjoint = false;
          break;
        }
      }
      if (!disjoint) {
        break;
      }
    }
    if (disjoint) {
      witness_sets[key].push_back(std::move(private_nodes));
    }
  }

  std::unordered_set<std::string> safe_keys;
  for (const auto& item : witness_sets) {
    if (item.second.size() >= threshold) {
      safe_keys.insert(item.first);
    }
  }
  out.safe_key_count = safe_keys.size();

  const auto target_roles_vec = rolesInMask(target);
  const std::set<RoleId> target_roles(target_roles_vec.begin(), target_roles_vec.end());
  std::unordered_map<std::string, TupleRow> dedup;
  for (const auto& row : result_.supports[superset]) {
    const auto key = deriveKey(row, out.key_roles);
    if (safe_keys.count(key) == 0) {
      continue;
    }
    TupleRow projected;
    projected.instance_ids.assign(data_.size(), kNoInstance);
    for (std::size_t v = 0; v < data_.size(); ++v) {
      if ((target & (Mask{1} << v)) != 0) {
        projected.instance_ids[v] = row.instance_ids[v];
      }
    }
    projected.role_nodes = row.role_nodes;
    projected.role_nodes.erase(std::remove_if(projected.role_nodes.begin(), projected.role_nodes.end(),
                                              [&](const auto& binding) {
                                                return target_roles.count(binding.role) == 0;
                                              }),
                               projected.role_nodes.end());
    dedup.emplace(tupleKey(projected, target), std::move(projected));
  }
  out.projected_rows = dedup.size();

  const auto complement = const_cast<FullLatticeIndex*>(this)->computeSupportDirectExcludingKeys(
      target, out.key_roles, safe_keys);
  for (const auto& row : complement) {
    dedup.emplace(tupleKey(row, target), row);
  }
  out.complement_rows = complement.size();
  out.derive_cost = static_cast<double>(result_.supports[superset].size() + out.complement_rows + out.projected_rows);
  out.extend_cost = target < plan_.estimates.totals.size() ? plan_.estimates.totals[target] : 0.0;

  out.rows.reserve(dedup.size());
  for (auto& item : dedup) {
    out.rows.push_back(std::move(item.second));
  }
  std::sort(out.rows.begin(), out.rows.end(), [&](const auto& left, const auto& right) {
    return tupleKey(left, target) < tupleKey(right, target);
  });
  return out;
}

std::vector<RoleId> FullLatticeIndex::rolesInMask(Mask mask) const {
  std::set<RoleId> roles;
  for (std::size_t v = 0; v < data_.size(); ++v) {
    if ((mask & (Mask{1} << v)) == 0) {
      continue;
    }
    roles.insert(data_[v].pattern_roles.begin(), data_[v].pattern_roles.end());
  }
  return {roles.begin(), roles.end()};
}

std::vector<RoleId> FullLatticeIndex::sharedRoles(Mask mask, std::size_t variable) const {
  const auto mask_roles = rolesInMask(mask);
  std::set<RoleId> mask_set(mask_roles.begin(), mask_roles.end());
  std::vector<RoleId> out;
  for (const auto& role : data_[variable].pattern_roles) {
    if (mask_set.count(role) != 0) {
      out.push_back(role);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::size_t FullLatticeIndex::maxRankPosition(Mask mask) const {
  std::size_t best = 0;
  bool found_any = false;
  for (std::size_t v = 0; v < data_.size(); ++v) {
    if ((mask & (Mask{1} << v)) == 0) {
      continue;
    }
    best = found_any ? std::max(best, rank_of_variable_[v]) : rank_of_variable_[v];
    found_any = true;
  }
  return found_any ? best : data_.size();
}

Mask FullLatticeIndex::parentMask(Mask mask) const {
  if (mask == 0) {
    return 0;
  }
  std::size_t max_rank = 0;
  std::size_t max_variable = 0;
  bool found_any = false;
  for (std::size_t v = 0; v < data_.size(); ++v) {
    if ((mask & (Mask{1} << v)) == 0) {
      continue;
    }
    if (!found_any || rank_of_variable_[v] > max_rank) {
      max_rank = rank_of_variable_[v];
      max_variable = v;
      found_any = true;
    }
  }
  return mask & ~(Mask{1} << max_variable);
}

std::string FullLatticeIndex::tupleKey(const TupleRow& row, Mask mask) const {
  std::vector<std::string> parts;
  for (std::size_t v = 0; v < data_.size(); ++v) {
    if ((mask & (Mask{1} << v)) != 0) {
      parts.push_back(std::to_string(row.instance_ids[v]));
    }
  }
  return joinKey(parts);
}

std::string FullLatticeIndex::roleKey(const RoleId& role, const NodeId& node) const {
  return joinKey({role, node});
}

std::string FullLatticeIndex::deriveKey(const TupleRow& row,
                                        const std::vector<RoleId>& roles) const {
  std::unordered_map<RoleId, NodeId> bound;
  for (const auto& binding : row.role_nodes) {
    bound[binding.role] = binding.node;
  }
  std::vector<std::string> parts;
  for (const auto& role : roles) {
    auto found = bound.find(role);
    parts.push_back(found == bound.end() ? std::string{} : found->second);
  }
  return joinKey(parts);
}

std::string FullLatticeIndex::deriveKey(const InstanceRecord& instance,
                                        const std::vector<RoleId>& roles) const {
  std::vector<std::string> parts;
  for (const auto& role : roles) {
    auto found = instance.role_to_node.find(role);
    parts.push_back(found == instance.role_to_node.end() ? std::string{} : found->second);
  }
  return joinKey(parts);
}


FullLatticeIndex::PackedSupport FullLatticeIndex::computeSupportDirect(Mask mask) {
  if (mask == 0) {
    return rootSupport();
  }
  const Mask parent = parentMask(mask);
  const auto variable = maxRankVariable(mask);
  auto parent_support = computeSupportDirect(parent);
  return parent == 0 ? singletonSupport(variable)
                     : extendSupport(parent_support, parent, variable, nullptr);
}

std::vector<TupleRow> FullLatticeIndex::computeSupportDirectExcludingKeys(
    Mask mask,
    const std::vector<RoleId>& key_roles,
    const std::unordered_set<std::string>& blocked_keys) {
  auto rows = unpackSupport(computeSupportDirect(mask));
  if (!blocked_keys.empty()) {
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const auto& row) {
                 return blocked_keys.count(deriveKey(row, key_roles)) != 0;
               }),
               rows.end());
  }
  return rows;
}

const FullLatticeIndex::PackedSupport& FullLatticeIndex::ensureResident(Mask mask) {
  auto found = resident_.find(mask);
  if (found != resident_.end()) {
    found->second.last_used = ++cache_clock_;
    result_.cache_stats.hits += 1;
    if (mask != 0) {
      result_.cache_stats.non_root_hits += 1;
    }
    return found->second.rows;
  }
  result_.cache_stats.misses += 1;
  result_.cache_stats.recomputations += 1;
  ResidentSupport entry;
  entry.rows = computeSupportDirect(mask);
  entry.last_used = ++cache_clock_;
  entry.permanent_pin = isPermanentPinned(mask) || mask == 0;
  entry.active_pin_count = 1;
  result_.cache_stats.resident_tuples += entry.rows.row_count;
  result_.cache_stats.peak_resident_tuples = std::max(result_.cache_stats.peak_resident_tuples,
                                                      result_.cache_stats.resident_tuples);
  result_.cache_stats.inserts += 1;
  auto inserted = resident_.emplace(mask, std::move(entry)).first;
  evictIfNeeded();
  return inserted->second.rows;
}

void FullLatticeIndex::putResident(Mask mask, PackedSupport rows) {
  auto existing = resident_.find(mask);
  if (existing != resident_.end()) {
    result_.cache_stats.resident_tuples -= existing->second.rows.row_count;
    existing->second.rows = std::move(rows);
    existing->second.last_used = ++cache_clock_;
    existing->second.permanent_pin = isPermanentPinned(mask);
    result_.cache_stats.resident_tuples += existing->second.rows.row_count;
    result_.cache_stats.peak_resident_tuples = std::max(result_.cache_stats.peak_resident_tuples,
                                                        result_.cache_stats.resident_tuples);
    evictIfNeeded();
    return;
  }
  ResidentSupport entry;
  entry.rows = std::move(rows);
  entry.last_used = ++cache_clock_;
  entry.permanent_pin = isPermanentPinned(mask) || mask == 0;
  result_.cache_stats.resident_tuples += entry.rows.row_count;
  result_.cache_stats.peak_resident_tuples = std::max(result_.cache_stats.peak_resident_tuples,
                                                      result_.cache_stats.resident_tuples);
  result_.cache_stats.inserts += 1;
  resident_[mask] = std::move(entry);
  evictIfNeeded();
}

void FullLatticeIndex::releaseResident(Mask mask) {
  if (mask == 0) {
    return;
  }
  auto found = resident_.find(mask);
  if (found == resident_.end()) {
    return;
  }
  if (found->second.active_pin_count != 0 || found->second.permanent_pin) {
    return;
  }
  result_.cache_stats.resident_tuples -= found->second.rows.row_count;
  resident_.erase(found);
}

void FullLatticeIndex::evictIfNeeded() {
  if (config_.cache_budget_tuples == 0) {
    return;
  }
  while (result_.cache_stats.resident_tuples > config_.cache_budget_tuples) {
    auto victim = resident_.end();
    double victim_score = std::numeric_limits<double>::infinity();
    for (auto it = resident_.begin(); it != resident_.end(); ++it) {
      if (it->first == 0 || it->second.permanent_pin || it->second.active_pin_count != 0) {
        continue;
      }
      const double size = static_cast<double>(std::max<std::size_t>(1, it->second.rows.row_count));
      const double estimate = it->first < plan_.estimates.totals.size()
                                  ? plan_.estimates.totals[it->first]
                                  : size;
      const double score = estimate / size;
      if (victim == resident_.end() || score < victim_score ||
          (std::abs(score - victim_score) < 1e-9 && it->second.last_used < victim->second.last_used)) {
        victim = it;
        victim_score = score;
      }
    }
    if (victim == resident_.end()) {
      result_.cache_stats.path_budget_overflows += 1;
      break;
    }
    result_.cache_stats.resident_tuples -= victim->second.rows.row_count;
    resident_.erase(victim);
    result_.cache_stats.evictions += 1;
  }
}

void FullLatticeIndex::pinActive(Mask mask) {
  auto found = resident_.find(mask);
  if (found != resident_.end()) {
    found->second.active_pin_count += 1;
  }
}

void FullLatticeIndex::unpinActive(Mask mask) {
  auto found = resident_.find(mask);
  if (found != resident_.end() && found->second.active_pin_count > 0) {
    found->second.active_pin_count -= 1;
  }
  evictIfNeeded();
}

bool FullLatticeIndex::isPermanentPinned(Mask mask) const {
  return std::find(plan_.pinned_masks.begin(), plan_.pinned_masks.end(), mask) != plan_.pinned_masks.end();
}

bool FullLatticeIndex::isDerivePinned(Mask mask) const {
  return std::find(plan_.derive_pinned_masks.begin(), plan_.derive_pinned_masks.end(), mask) !=
         plan_.derive_pinned_masks.end();
}

std::size_t FullLatticeIndex::maxRankVariable(Mask mask) const {
  std::size_t max_rank = 0;
  std::size_t max_variable = 0;
  bool found_any = false;
  for (std::size_t v = 0; v < data_.size(); ++v) {
    if ((mask & (Mask{1} << v)) == 0) {
      continue;
    }
    if (!found_any || rank_of_variable_[v] > max_rank) {
      max_rank = rank_of_variable_[v];
      max_variable = v;
      found_any = true;
    }
  }
  if (!found_any) {
    throw std::runtime_error("empty mask has no max-rank variable");
  }
  return max_variable;
}

std::optional<DeriveResult> FullLatticeIndex::bestOnlineDerive(
    Mask target,
    Mask parent,
    const PackedSupport& parent_support) const {
  std::optional<DeriveResult> best;
  const double extend_cost = static_cast<double>(parent_support.row_count +
                                                 (target < plan_.estimates.totals.size()
                                                      ? plan_.estimates.totals[target]
                                                      : 0.0));
  for (std::size_t donor = 0; donor < data_.size(); ++donor) {
    if ((target & (Mask{1} << donor)) != 0) {
      continue;
    }
    const Mask superset = target | (Mask{1} << donor);
    if (superset >= result_.supports.size() || result_.supports[superset].empty()) {
      continue;
    }
    auto candidate = deriveFromSupersetDirect(target, donor);
    candidate.extend_cost = extend_cost;
    const_cast<FullLatticeIndex*>(this)->result_.cache_stats.derive_candidates_considered += 1;
    if (!best || candidate.derive_cost < best->derive_cost ||
        (std::abs(candidate.derive_cost - best->derive_cost) < 1e-9 && donor < best->donor_variable)) {
      best = std::move(candidate);
    }
  }
  (void)parent;
  return best;
}

std::string maskToString(Mask mask, const std::vector<VariableId>& variables) {
  std::ostringstream out;
  out << "{";
  bool first = true;
  for (std::size_t i = 0; i < variables.size(); ++i) {
    if ((mask & (Mask{1} << i)) == 0) {
      continue;
    }
    if (!first) {
      out << ",";
    }
    first = false;
    out << variables[i];
  }
  out << "}";
  return out.str();
}

}  // namespace causal_index_v4
