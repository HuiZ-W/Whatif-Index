#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace causal_index_v4 {

using VariableId = std::string;
using RoleId = std::string;
using NodeId = std::string;
using BucketId = std::string;
using InstanceId = std::uint64_t;
using Mask = std::uint64_t;
using Count = std::uint64_t;
using NodeAttributeMap = std::unordered_map<std::string, std::string>;
using NodeAttributeStore = std::unordered_map<NodeId, NodeAttributeMap>;

inline constexpr InstanceId kNoInstance = std::numeric_limits<InstanceId>::max();

struct RoleBinding {
  RoleId role;
  NodeId node;
};

struct MatchInstance {
  InstanceId id = 0;
  BucketId bucket;
  std::vector<RoleBinding> roles;
  std::string value;
};

struct VariableInput {
  VariableId id;
  std::vector<RoleId> pattern_roles;
  std::vector<MatchInstance> instances;
};

struct ProgressEvent {
  std::size_t completed_tasks = 0;
  std::size_t total_tasks = 0;
  Mask mask = 0;
  Mask parent_mask = 0;
  std::size_t variable = 0;
  std::size_t depth = 0;
  Count rows = 0;
  std::size_t resident_tuples = 0;
  std::size_t peak_resident_tuples = 0;
  std::size_t resident_results = 0;
  std::size_t peak_resident_results = 0;
  Count emitted_rows = 0;
};

struct RollupStats {
  std::size_t used = 0;
  std::size_t projected_rows = 0;
  std::size_t residual_rows = 0;
  std::size_t donor_rows_scanned = 0;
  std::size_t witness_rows_scanned = 0;
  std::size_t peak_cache_bytes = 0;
};

struct SourceOutputScope {
  // The complete PG-match variable set to join.
  std::vector<VariableId> source_scope;
  // Value/count cubes to emit from the source support. Each output scope must
  // be a subset of source_scope. Empty means emit source_scope itself.
  std::vector<std::vector<VariableId>> output_scopes;
};

enum class VariableOrderPolicy {
  Alphabetical,
  CostModel,
  InstanceCountAscending,
  Random,
  PartialWorkloadDp,
  PartialWorkloadGreedy,
};

enum class ExecutionSchedule {
  DfsPrefixTree,
  ScopeOrderedWorkingPath,
  BoundedScopeCache,
  JoinSizeGreedyWorkingPath,
  CacheAwareGreedy,
  IndependentSources,
};

enum class ResultCacheEvictionPolicy {
  Belady,
  Lru,
};

enum class ScopeOrderPolicy {
  DfsLexicographic,
  Alphabetical,
  Random,
};

struct LatticeBuildConfig {
  // Counts binary join operator invocations, including fused leaf aggregation.
  // Singleton reads are excluded. Called on the executor's coordinating thread.
  std::function<void()> binary_join_callback;
  // Physical parent rows and compatible pairs before output compaction.
  // Compressed multiplicities and reused results are not expanded or recounted.
  std::function<void(std::size_t, std::size_t)> join_work_callback;
  bool independent_leaf_pushdown = false;
  // 0 means no extra resident-cache budget. The executor still keeps the
  // current DFS path working set, matching the final prefix-reuse design.
  std::size_t cache_budget_tuples = 0;
  // Maximum number of non-root support results retained for reuse by
  // BoundedScopeCache after each extension. The root is excluded. Computing a
  // child necessarily creates a transient parent/child overlap before the
  // victim is evicted. 0 disables bounded-result execution.
  std::size_t result_cache_capacity = 0;
  ResultCacheEvictionPolicy result_cache_eviction_policy =
      ResultCacheEvictionPolicy::Belady;
  bool enable_derive = true;
  // Optional streaming derive budget. When positive, the planner keeps only a
  // bounded set of future-useful superset supports for derive, instead of all
  // subset supports. The unit is support tuples, using estimates at plan time.
  std::size_t derive_materialization_budget_tuples = 0;
  // Debug mode: retain every subset support table in result_.supports.
  // The production/default path stores exact counts only and releases support
  // rows after leaving their DFS subtree.
  bool store_all_supports = false;
  // Debug/provenance mode: retain per-edge parent participation bitmaps.
  // Production count-only streaming keeps this off because the bitmaps grow
  // with the number of lattice edges.
  bool store_participation = false;
  // Debug mode: retain full derive result rows. Runtime derive can still use
  // rows transiently without storing them in the final result.
  bool store_derive_results = false;
  // Store final value-assignment factor counts for each target mask. The key is
  // the raw/bucket value tuple in mask variable order.
  bool store_value_counts = true;
  // Optional synchronous ownership handoff for a completed value-count cube.
  // The map is removed from LatticeBuildResult immediately after its final
  // aggregation, allowing callers to spool completed cubes to disk.
  std::function<void(Mask, std::map<std::vector<std::string>, Count>&&)>
      streamed_value_count_sink;
  // Exact rollup donors must retain distinct instance identities.
  bool preserve_instance_identity = false;
  bool full_join_rollup = false;
  std::size_t rollup_cache_bytes = 512ULL * 1024 * 1024;
  std::function<void(const RollupStats&)> rollup_callback;
  // First-level anchor partition. When enabled, the executor chooses a common
  // fixed-view role (for example Person#0) and restricts candidate instances to
  // the same anchor bucket before running the normal compatibility checks.
  bool enable_anchor_partition = true;
  // Empty means auto-detect a common role, preferring #0/_0 style roles.
  RoleId anchor_role;
  // Shared node tuples are partitioned once by stable node attributes. Rows in
  // the same value partition are compared directly with the exact node-equality
  // and injectivity predicates; no second-level exact-node index is built.
  bool enable_value_partition_join = true;
  std::shared_ptr<const NodeAttributeStore> node_attributes;
  // Candidate attributes keyed by node label, e.g. Person -> {gender, speaks}.
  // A small subset is selected greedily for every prefix extension.
  std::unordered_map<std::string, std::vector<std::string>>
      value_partition_attributes_by_label;
  std::size_t value_partition_min_parent_rows = 1024;
  std::size_t value_partition_max_features = 3;
  std::size_t value_partition_max_buckets = 4096;
  std::size_t value_partition_selection_sample_rows = 65536;
  double value_partition_min_pair_reduction = 0.05;
  double value_partition_bucket_overhead = 32.0;
  // Optional profiling only. Default performance runs do not collect per-join
  // bucket histograms.
  bool collect_partition_traces = false;
  // CostModel is the legacy cardinality/materialization heuristic.
  // InstanceCountAscending adds the variable with the fewest inputs.
  // PartialWorkloadDp exactly minimizes the estimated weighted prefix tree for
  // the configured source workload. PartialWorkloadGreedy uses amortized
  // prefix coverage with one-step lookahead over the same workload.
  VariableOrderPolicy variable_order_policy = VariableOrderPolicy::CostModel;
  // Plan rho against the complete local lattice, then apply source pruning
  // only when the DFS task order is generated. This keeps rho fixed between
  // pruning and no-pruning experiments.
  bool plan_order_from_unpruned_lattice = false;
  // Used only by VariableOrderPolicy::Random.
  std::uint64_t variable_order_seed = 20260716ULL;
  // Scope ordering is independent from rho. Alphabetical and Random first
  // arrange complete source scopes, then execute them one by one while keeping
  // only the longest prefix shared with the next scope on the working path.
  ScopeOrderPolicy scope_order_policy = ScopeOrderPolicy::DfsLexicographic;
  std::uint64_t scope_order_seed = 20260716ULL;
  // Generate the scope schedule against the complete local lattice and then
  // remove sources outside the configured workload. This keeps the relative
  // schedule fixed between pruning and no-pruning ablations. It matters for a
  // seeded random schedule: shuffling only the retained scopes would otherwise
  // produce a different relative order after pruning.
  bool scope_order_from_unpruned_lattice = false;
  // DFS constructs the canonical prefix tree recursively. CacheAwareGreedy
  // chooses the next source by reusable cached-prefix depth and remaining
  // estimated cost, without a DFS traversal. IndependentSources retains the
  // same exact-trie probes and variable order but recomputes every source.
  // JoinSizeGreedyWorkingPath greedily minimizes estimated join size both
  // between tasks and inside each task. It keeps only the current root-to-task
  // path, so reuse is limited to a consecutive task's deepest live prefix and
  // does not use the general resident cache.
  ExecutionSchedule execution_schedule = ExecutionSchedule::DfsPrefixTree;

  // Number of worker threads used by independent FullJoin cube tasks and by
  // Rollup's join/witness kernels. 1 keeps deterministic single-thread
  // behavior. Higher values may increase temporary memory because each worker
  // buffers output rows.
  std::size_t worker_threads = 1;
  // Minimum input rows before an individual join or Rollup witness scan uses
  // the worker pool. Tiny operations remain serial to avoid dispatch overhead.
  std::size_t parallel_min_parent_rows = 1024;
  // Empty means full lattice. Non-empty means build only these target scopes
  // plus their prefix paths under the planned variable order.
  std::vector<std::vector<VariableId>> target_scopes;
  // Exact binding-level outputs retained for cross-tree joins. Unlike
  // store_all_supports, this materializes only the listed source scopes. Rows
  // retain stable per-variable instance IDs, raw values, complete role-node
  // bindings, and multiplicity. These scopes are automatically added to the
  // execution workload when they are not already target scopes.
  std::vector<std::vector<VariableId>> binding_output_scopes;
  // Optional source -> output projection workload. When non-empty, target_scopes
  // is ignored for execution planning; each source_scope is joined exactly once
  // and its output_scopes are aggregated from that complete source support.
  std::vector<SourceOutputScope> source_output_scopes;
  // Called after a child support has been counted. The callback is synchronous
  // and should stay lightweight; it is mainly for long full-lattice runs.
  std::size_t progress_every_tasks = 0;
  std::function<void(const ProgressEvent&)> progress_callback;
};

struct TupleRow {
  std::vector<InstanceId> instance_ids;
  std::vector<RoleBinding> role_nodes;
};

struct MaterializedBindingRow {
  // Vectors use FullLatticeIndex::variables() order. kNoInstance and an empty
  // value mark variables outside the materialized scope.
  std::vector<InstanceId> instance_ids;
  std::vector<std::string> values;
  std::vector<RoleBinding> role_nodes;
  Count multiplicity = 1;
};

struct EstimateTable {
  std::vector<BucketId> buckets;
  std::unordered_map<BucketId, std::size_t> bucket_index;
  std::vector<std::vector<double>> by_mask_bucket;
  std::vector<double> totals;
};

struct MaterializedPrefix {
  Mask mask = 0;
  double score = 0.0;
  double estimated_size = 0.0;
  double estimated_edge_cost = 0.0;
  std::size_t reuse_count = 0;
  bool pinned = false;
};

struct LatticePlan {
  std::vector<VariableId> variable_order;
  std::vector<std::size_t> variable_order_indices;
  std::vector<Mask> task_order;
  std::vector<Mask> target_masks;
  // Complete source scopes in their physical visit order. DFS records its
  // canonical preorder; scope-ordered execution records the requested
  // alphabetical or random permutation.
  std::vector<Mask> scope_order;
  EstimateTable estimates;
  std::vector<MaterializedPrefix> materialization;
  std::vector<Mask> pinned_masks;
  std::vector<Mask> derive_pinned_masks;
  double estimated_cost = 0.0;
};

struct ParticipationBitmap {
  Mask parent = 0;
  std::size_t variable = 0;
  std::vector<std::uint8_t> has_partner;
};

struct DeriveResult {
  Mask target = 0;
  std::size_t donor_variable = 0;
  Mask superset = 0;
  std::vector<RoleId> key_roles;
  std::size_t safe_key_count = 0;
  std::size_t projected_rows = 0;
  std::size_t complement_rows = 0;
  double derive_cost = 0.0;
  double extend_cost = 0.0;
  bool used = false;
  std::vector<TupleRow> rows;
};

struct RuntimeCacheStats {
  std::size_t hits = 0;
  std::size_t non_root_hits = 0;
  std::size_t misses = 0;
  std::size_t inserts = 0;
  std::size_t evictions = 0;
  std::size_t recomputations = 0;
  std::size_t resident_tuples = 0;
  std::size_t peak_resident_tuples = 0;
  std::size_t path_budget_overflows = 0;
  std::size_t resident_results = 0;
  std::size_t peak_resident_results = 0;
  std::size_t result_capacity_evictions = 0;
  std::size_t result_capacity_overflows = 0;
  std::size_t working_path_tasks = 0;
  std::size_t working_path_reuse_transitions = 0;
  std::size_t working_path_reused_prefix_levels = 0;
  std::size_t working_path_join_steps = 0;
  std::size_t working_path_rebuilt_steps = 0;
  std::size_t trie_nodes_visited = 0;
  std::size_t trie_subtrees_skipped = 0;
  std::size_t live_candidates_emitted = 0;
  std::size_t derive_candidates_considered = 0;
  std::size_t derive_used = 0;
  std::size_t derive_materialized_masks = 0;
  std::size_t derive_materialized_tuples = 0;
  std::size_t anchor_buckets_used = 0;
  std::size_t anchor_candidates_scanned = 0;
  std::size_t intersection_buckets_built = 0;
  std::size_t intersection_bucket_rows = 0;
  std::size_t intersection_candidate_queries = 0;
  std::size_t value_partition_joins = 0;
  std::size_t value_partition_features_selected = 0;
  std::size_t value_partition_buckets_built = 0;
  std::size_t value_partition_parent_rows = 0;
  std::size_t value_partition_candidate_rows = 0;
  std::size_t value_partition_pair_checks = 0;
  std::size_t value_partition_compatible_pairs = 0;
  std::map<std::string, std::size_t> value_partition_feature_uses;
  std::size_t grouped_prefix_hits = 0;
  std::size_t grouped_prefix_inserts = 0;
  std::size_t leaf_cube_pushdowns = 0;
  std::size_t leaf_cube_rows = 0;
};

inline constexpr std::size_t kPartitionHistogramBins = 32;

struct JoinPartitionTrace {
  Mask parent_mask = 0;
  Mask child_mask = 0;
  VariableId variable;
  std::string mode;
  std::size_t parent_rows = 0;
  std::size_t candidate_rows = 0;
  std::size_t join_roles = 0;
  std::vector<std::string> value_features;
  std::size_t parent_partition_buckets = 0;
  std::size_t candidate_partition_buckets = 0;
  std::size_t bucket_count = 0;
  std::size_t unmatched_parent_rows = 0;
  std::size_t unmatched_candidate_rows = 0;
  long double cartesian_pairs = 0.0L;
  long double partition_pairs = 0.0L;
  long double max_bucket_pairs = 0.0L;
  long double parent_bucket_sum = 0.0L;
  long double parent_bucket_square_sum = 0.0L;
  long double parent_bucket_xlogx_sum = 0.0L;
  long double candidate_bucket_sum = 0.0L;
  long double candidate_bucket_square_sum = 0.0L;
  long double candidate_bucket_xlogx_sum = 0.0L;
  long double pair_work_square_sum = 0.0L;
  long double pair_work_xlogx_sum = 0.0L;
  std::size_t max_parent_bucket = 0;
  std::size_t max_candidate_bucket = 0;
  std::size_t compatible_pairs = 0;
  std::size_t output_rows = 0;
  std::array<std::size_t, kPartitionHistogramBins> parent_bucket_hist{};
  std::array<std::size_t, kPartitionHistogramBins> candidate_bucket_hist{};
  std::array<std::size_t, kPartitionHistogramBins> pair_work_hist{};
};

struct LatticeExecutionResult {
  std::vector<Mask> task_order;
  std::vector<Count> exact_counts;
  std::vector<std::vector<TupleRow>> supports;
  // Indexed by local variable mask. Only masks named by
  // LatticeBuildConfig::binding_output_scopes are populated.
  std::vector<std::vector<MaterializedBindingRow>> binding_outputs;
  std::vector<std::map<std::vector<std::string>, Count>> value_counts;
  std::vector<std::size_t> value_trie_nodes;
  std::vector<JoinPartitionTrace> partition_traces;
  std::size_t completed_tasks = 0;
  std::vector<ParticipationBitmap> participation;
  std::vector<DeriveResult> derive_results;
  RuntimeCacheStats cache_stats;
  Count emitted_rows = 0;
};

class FullLatticeIndex {
 public:
  explicit FullLatticeIndex(std::vector<VariableInput> variables,
                            LatticeBuildConfig config = {});

  const std::vector<VariableInput>& variables() const { return variables_; }
  const LatticePlan& plan() const { return plan_; }
  const LatticeExecutionResult& result() const { return result_; }

  std::size_t variableCount() const { return variables_.size(); }
  Mask allVariablesMask() const;
  Mask maskOf(const std::vector<VariableId>& variable_ids) const;
  std::vector<VariableId> variablesOf(Mask mask) const;

  // Build only the optimizer plan (variable order, DFS task order, and
  // materialization choices), without executing joins. Useful for aligning
  // baselines to the same order as the prefix-reuse executor.
  void planOnly();

  void build();

  // Emit every nonempty cube independently, without dense lattice metadata.
  // The consumer owns materialized outputs; result() retains only counters.
  void streamIndependentCubes(const std::function<void(
      Mask, std::map<std::vector<std::string>, Count>&&)>& emit);
  void streamRollupCubes(const std::function<void(
      Mask, std::map<std::vector<std::string>, Count>&&)>& emit);

  Count exactCount(Mask mask) const;
  double estimatedCount(Mask mask) const;
  const std::vector<TupleRow>& support(Mask mask) const;

  // Recomputes a single derive candidate from an already materialized superset.
  // The result is exact because unsafe keys are completed by direct target rows.
  DeriveResult deriveFromSuperset(Mask target, std::size_t donor_variable) const;

 private:
  struct InstanceRecord {
    InstanceId id = 0;
    BucketId bucket;
    std::string value;
    std::uint32_t value_code = 0;
    std::vector<RoleBinding> roles;
    std::unordered_map<RoleId, NodeId> role_to_node;
    std::vector<std::pair<std::size_t, std::uint32_t>> role_node_codes;
    std::vector<NodeId> nodes;
    std::vector<std::uint32_t> node_codes;
  };

  struct TrieEdge {
    std::uint32_t value = 0;
    std::size_t child = 0;
    std::size_t subtree_count = 0;
  };

  struct TrieNode {
    std::unordered_map<std::uint32_t, std::size_t> build_child;
    std::vector<std::size_t> build_instances;
    std::size_t edge_begin = 0;
    std::size_t edge_count = 0;
    std::size_t instance_begin = 0;
    std::size_t instance_count = 0;
    std::size_t subtree_count = 0;
  };

  struct VariableTrie {
    std::vector<RoleId> role_order;
    std::vector<TrieNode> nodes;
    std::vector<TrieEdge> edges;
    std::vector<std::size_t> leaf_instances;
  };

  struct VariableData {
    VariableId id;
    std::vector<RoleId> pattern_roles;
    std::vector<InstanceRecord> instances;
    std::unordered_map<std::string, std::vector<std::size_t>> role_node_posting;
    std::unordered_map<BucketId, std::size_t> bucket_counts;
    std::unordered_map<std::uint32_t, std::vector<std::size_t>> anchor_posting;
    std::unordered_map<std::uint32_t, VariableTrie> anchor_tries;
    bool has_anchor_role = false;
    VariableTrie trie;
  };

  struct PackedSupport {
    std::size_t row_count = 0;
    // Row-major arrays. instance_ids has row_count * variable_count entries.
    // role_node_codes has row_count * global_role_count entries. Code 0 means
    // the role is currently unbound in that prefix row. value_codes has
    // row_count * variable_count entries, also using 0 as unbound/missing.
    std::vector<InstanceId> instance_ids;
    std::vector<std::uint32_t> value_codes;
    std::vector<std::uint32_t> role_node_codes;
    // Multiplicity represented by this compressed prefix state. A missing/empty
    // counts vector is treated as all ones for backward-compatible debug paths.
    std::vector<Count> counts;
  };

  struct ResidentSupport {
    PackedSupport rows;
    std::size_t last_used = 0;
    bool permanent_pin = false;
    std::size_t active_pin_count = 0;
  };

  void normalizeInputs();
  void buildPostingsAndStats();
  void buildEstimateTable();
  void planVariableOrder();
  void planTaskOrder();
  void planMaterialization();
  void planDeriveMaterialization();
  void executeFullLattice();
  void executeDfs(Mask parent_mask,
                  std::size_t last_order_position,
                  std::size_t depth);
  void executeScopeOrderedWorkingPath();
  void executeBoundedScopeCache();
  void executeJoinSizeGreedyWorkingPath();
  void executeCacheAwareGreedy();
  void executeIndependentSources();
  std::vector<Mask> prefixPath(Mask source) const;
  bool isPrefixOfSource(Mask prefix, Mask source) const;
  void aggregateSourceOutputs(Mask source, const PackedSupport& support);
  void materializeBindingOutput(Mask source, const PackedSupport& support);
  bool requiresBindingOutput(Mask mask) const;
  Mask scopeToMask(const std::vector<VariableId>& scope, const char* context) const;
  void prepareSourceOutputMasks();
  std::vector<Mask> configuredSourceMasks() const;
  void aggregateValueCounts(Mask mask, const PackedSupport& support);
  void aggregateProjectedValueCounts(Mask source_mask,
                                     Mask output_mask,
                                     const PackedSupport& support);
  Count aggregateLeafValueCounts(const PackedSupport& parent_support,
                                 Mask parent_mask,
                                 std::size_t variable,
                                 Mask child_mask,
                                 const std::vector<Mask>& output_masks,
                                 std::map<std::vector<std::string>, Count>* streamed_counts = nullptr);
  Count supportWeight(const PackedSupport& support) const;
  Count rowWeight(const PackedSupport& support, std::size_t row) const;
  PackedSupport compactSupport(PackedSupport support) const;
  std::string compactRowKey(const PackedSupport& support, std::size_t row) const;
  std::vector<std::string> valueKey(const PackedSupport& support, std::size_t row, Mask mask) const;
  std::string valueForCode(std::uint32_t code) const;
  static std::size_t valueTrieNodeCount(const std::map<std::vector<std::string>, Count>& counts);
  void reportProgress(Mask child,
                      Mask parent_mask,
                      std::size_t variable,
                      std::size_t depth,
                      Count rows);
  PackedSupport rootSupport() const;
  PackedSupport singletonSupport(std::size_t variable) const;
  PackedSupport extendSupport(const PackedSupport& parent_support,
                              Mask parent_mask,
                              std::size_t variable,
                              std::vector<std::uint8_t>* participation);
  std::vector<std::size_t> candidateInstances(const PackedSupport& support,
                                              std::size_t row_index,
                                              std::size_t variable,
                                              RuntimeCacheStats& stats) const;
  std::vector<std::size_t> candidateInstancesForBounds(
      std::size_t variable,
      const std::vector<std::uint32_t>& bound_role_codes,
      const std::vector<std::uint32_t>& forbidden_node_codes,
      RuntimeCacheStats& stats) const;
  bool instanceCompatible(const InstanceRecord& instance,
                          const std::vector<std::uint32_t>& bound_role_codes,
                          const std::vector<std::uint32_t>& forbidden_node_codes) const;
  void queryTrie(std::size_t variable,
                 const VariableTrie& trie,
                 std::size_t trie_node,
                 std::size_t depth,
                 const std::vector<std::uint32_t>& bound_role_codes,
                 const std::vector<std::uint32_t>& forbidden_node_codes,
                 std::vector<std::size_t>& out,
                 RuntimeCacheStats& stats) const;
  void appendMergedTuple(PackedSupport& out,
                         const PackedSupport& parent,
                         std::size_t parent_row,
                         std::size_t variable,
                         const InstanceRecord& candidate) const;
  PackedSupport packSupport(const std::vector<TupleRow>& rows) const;
  std::vector<TupleRow> unpackSupport(const PackedSupport& support) const;

  std::vector<RoleId> rolesInMask(Mask mask) const;
  std::vector<RoleId> sharedRoles(Mask mask, std::size_t variable) const;
  std::size_t maxRankPosition(Mask mask) const;
  Mask parentMask(Mask mask) const;
  std::string tupleKey(const TupleRow& row, Mask mask) const;
  std::string roleKey(const RoleId& role, const NodeId& node) const;
  std::string deriveKey(const TupleRow& row, const std::vector<RoleId>& roles) const;
  std::string deriveKey(const InstanceRecord& instance, const std::vector<RoleId>& roles) const;

  void chooseAnchorRole();
  void buildVariableTries();
  VariableTrie buildTrieForInstances(std::size_t variable, const std::vector<std::size_t>& instances) const;
  void insertTrieInstance(std::size_t variable, VariableTrie& trie, std::size_t instance_index) const;
  void finalizeTrie(VariableTrie& trie) const;
  PackedSupport computeSupportDirect(Mask mask);
  std::vector<TupleRow> computeSupportDirectExcludingKeys(
      Mask mask,
      const std::vector<RoleId>& key_roles,
      const std::unordered_set<std::string>& blocked_keys);
  const PackedSupport& ensureResident(Mask mask);
  void putResident(Mask mask, PackedSupport rows);
  void releaseResident(Mask mask);
  void evictIfNeeded();
  void pinActive(Mask mask);
  void unpinActive(Mask mask);
  bool isPermanentPinned(Mask mask) const;
  bool isDerivePinned(Mask mask) const;
  std::size_t maxRankVariable(Mask mask) const;
  std::optional<DeriveResult> bestOnlineDerive(Mask target,
                                               Mask parent,
                                               const PackedSupport& parent_support) const;
  DeriveResult deriveFromSupersetDirect(Mask target, std::size_t donor_variable) const;

  std::vector<VariableInput> variables_;
  LatticeBuildConfig config_;
  std::vector<VariableData> data_;
  std::unordered_map<VariableId, std::size_t> variable_index_;
  std::vector<RoleId> global_roles_;
  std::unordered_map<RoleId, std::size_t> global_role_index_;
  std::optional<std::size_t> anchor_role_index_;
  RoleId anchor_role_;
  std::vector<NodeId> node_values_;
  std::unordered_map<NodeId, std::uint32_t> node_code_;
  std::vector<std::string> value_values_;
  std::unordered_map<std::string, std::uint32_t> value_code_;
  std::unordered_map<std::string, std::vector<std::uint32_t>> node_attribute_codes_;
  std::unordered_set<Mask> planned_task_masks_;
  std::vector<std::vector<Mask>> output_masks_by_source_;
  std::unordered_set<Mask> output_masks_;
  std::unordered_set<Mask> binding_output_masks_;
  std::unordered_map<BucketId, std::size_t> bucket_index_;
  std::vector<BucketId> buckets_;
  std::unordered_map<std::string, std::size_t> role_bucket_distinct_counts_;
  std::vector<std::size_t> rank_of_variable_;
  LatticePlan plan_;
  LatticeExecutionResult result_;
  std::unordered_map<Mask, ResidentSupport> resident_;
  std::size_t cache_clock_ = 0;
};

std::string maskToString(Mask mask, const std::vector<VariableId>& variables);

}  // namespace causal_index_v4
