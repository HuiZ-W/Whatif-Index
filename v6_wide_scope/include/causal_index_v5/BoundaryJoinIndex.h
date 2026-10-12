#pragma once

#include "causal_index_v4/FullLatticeIndex.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace causal_index_v5 {

using VariableId = causal_index_v4::VariableId;
using RoleId = causal_index_v4::RoleId;
using NodeId = causal_index_v4::NodeId;
using InstanceId = causal_index_v4::InstanceId;
using Count = causal_index_v4::Count;
using VariableInput = causal_index_v4::VariableInput;
using RoleBinding = causal_index_v4::RoleBinding;
using ValueTuple = std::vector<std::string>;
using ValueCountMap = std::map<ValueTuple, Count>;

struct CausalEdge {
  VariableId parent;
  VariableId child;
};

struct ParentTreeSpec {
  std::string id;
  // Stable local order used by the Tree-local cube and its Boundary rows.
  std::vector<VariableId> variables;
};

struct DirectedQuerySpec {
  std::string id;
  std::vector<VariableId> treatments;
  std::vector<VariableId> outcomes;
  // Observed variables in a conditional interventional query, e.g. Z in
  // P(Y | do(X=x), Z=z). They are evidence, not additional interventions.
  std::vector<VariableId> conditioning_variables;
  // Adjustment variables or other variables required by an upstream causal
  // planner. They are added before projection onto Parent Trees. A variable
  // may occur in both conditioning_variables and required_variables.
  std::vector<VariableId> required_variables;
};

struct ExactScopeSpec {
  std::string id;
  std::vector<VariableId> scope;
};

enum BoundaryDirection : std::uint8_t {
  kBoundaryNone = 0,
  kBoundarySourceSide = 1,
  kBoundaryTargetSide = 2,
  kBoundaryTransit = 4,
};

struct BoundaryRequirement {
  std::string view_id;
  std::string tree_id;
  std::vector<VariableId> scope;
  std::uint8_t direction_flags = kBoundaryNone;
  std::vector<std::string> query_ids;
};

struct BoundaryQueryPlan {
  std::string query_id;
  std::vector<VariableId> treatments;
  std::vector<VariableId> outcomes;
  std::vector<VariableId> conditioning_variables;
  std::vector<VariableId> required_variables;
  std::vector<VariableId> global_scope;
  std::vector<std::string> view_ids;
};

struct BoundaryPlan {
  std::vector<BoundaryRequirement> requirements;
  std::vector<BoundaryQueryPlan> queries;
};

class BoundaryRequirementPlanner {
public:
  BoundaryRequirementPlanner(std::vector<VariableInput> variables,
                             std::vector<CausalEdge> causal_edges,
                             std::vector<ParentTreeSpec> parent_trees);

  // include_all_reachable_pairs adds one query for every ordered pair linked
  // by a directed causal path. Explicit queries can add adjustment-expanded
  // workloads and are deduplicated by query ID.
  BoundaryPlan plan(const std::vector<DirectedQuerySpec> &explicit_queries,
                    bool include_all_reachable_pairs,
                    bool prune_redundant_views = true,
                    const std::vector<ExactScopeSpec> &exact_scopes = {}) const;

  static std::string reachablePairQueryId(const VariableId &treatment,
                                          const VariableId &outcome);
  static std::string exactScopeQueryId(const std::vector<VariableId> &scope);

private:
  std::vector<VariableInput> variables_;
  std::vector<CausalEdge> causal_edges_;
  std::vector<ParentTreeSpec> parent_trees_;
};

struct VariableBindingCell {
  VariableId variable;
  InstanceId instance_id = causal_index_v4::kNoInstance;
  std::string raw_value;
};

struct BoundaryRow {
  std::vector<VariableBindingCell> bindings;
  std::vector<RoleBinding> role_nodes;
  Count multiplicity = 1;
};

struct BoundaryView {
  std::string id;
  std::string tree_id;
  std::vector<VariableId> scope;
  std::vector<BoundaryRow> rows;
};

struct CubeTable {
  std::string tree_id;
  std::vector<VariableId> scope;
  ValueCountMap counts;
};

enum class LocalCubePolicy {
  AllSubsets,
  ExplicitScopes,
};

struct BoundaryBuildConfig {
  // Query-independent directed prefixes/suffixes incident to tree boundaries.
  bool materialize_structural_boundaries = false;
  LocalCubePolicy cube_policy = LocalCubePolicy::AllSubsets;
  // Independent full-cube baseline: enumerate and retain cubes incrementally.
  bool stream_all_subset_cubes = false;
  // Optional sink for incrementally produced full cubes. When set, each cube
  // is moved to the sink instead of being retained in cubes_.
  std::function<void(CubeTable &&)> streamed_cube_sink;
  // Used only by ExplicitScopes. Keys are ParentTreeSpec::id values.
  std::unordered_map<std::string, std::vector<std::vector<VariableId>>>
      explicit_cube_scopes;
  // Protects callers from accidentally requesting an impractical local cube.
  std::size_t max_all_subset_width = 16;
  bool include_all_reachable_pairs = true;
  // False materializes every Parent Tree projection touched by a cross-tree
  // query. It is intended for the Boundary-pruning ablation.
  bool prune_redundant_boundary_views = true;
  std::vector<DirectedQuerySpec> query_workload;
  // Optional output of an upstream workload compiler. Supplying it avoids
  // rerunning BoundaryRequirementPlanner during physical index construction.
  std::optional<BoundaryPlan> precompiled_boundary_plan;
  // Exact scopes introduced by physical Parent Tree decomposition.
  std::vector<ExactScopeSpec> exact_scope_workload;
  // Materialize one binding-aware singleton view per causal variable. These
  // attachment views make arbitrary future cross-Tree adjustment queries
  // answerable without rematching the property graph.
  bool materialize_variable_attachments = false;
  causal_index_v4::LatticeBuildConfig tree_executor;
};

struct BoundaryJoinResult {
  std::string query_id;
  std::vector<VariableId> global_scope;
  std::vector<VariableId> output_scope;
  Count exact_binding_count = 0;
  ValueCountMap value_counts;
  // Populated only when retain_rows=true.
  std::vector<BoundaryRow> rows;
};

struct OnlineScopeStats {
  bool cube_hit = false;
  std::size_t eligible_multivariable_views = 0;
  std::size_t selected_multivariable_views = 0;
  std::size_t selected_attachment_views = 0;
  std::vector<std::string> component_ids;
  std::vector<std::string> view_ids;
  std::uint64_t join_calls = 0;
  std::uint64_t join_probe_rows = 0;
  std::uint64_t join_work_pairs = 0;
};

struct SnapshotStorageStats {
  std::uint64_t total_bytes = 0;
  std::uint64_t metadata_bytes = 0;
  std::uint64_t cube_bytes = 0;
  std::uint64_t boundary_bytes = 0;
};

struct ExternalCubeStorage {
  std::string path;
  std::size_t table_count = 0;
  std::size_t cell_count = 0;
  std::uint64_t payload_bytes = 0;
};

class MaterializedBoundaryIndex {
public:
  static MaterializedBoundaryIndex
  build(std::vector<VariableInput> variables,
        std::vector<CausalEdge> causal_edges,
        std::vector<ParentTreeSpec> parent_trees,
        BoundaryBuildConfig config = {});

  static MaterializedBoundaryIndex load(const std::string &path);
  void save(const std::string &path) const;
  SnapshotStorageStats saveWithStats(const std::string &path) const;
  SnapshotStorageStats
  saveWithExternalCubes(const std::string &path,
                        const ExternalCubeStorage &storage) const;
  static void writeCubeRecord(std::ostream &out, const CubeTable &cube);
  // Offline-only: retain binding-aware tree projections for an explicit workload.
  void materializeExactBoundaryViews(const std::vector<ExactScopeSpec> &queries);

  const BoundaryPlan &plan() const { return plan_; }
  const std::vector<ParentTreeSpec> &parentTrees() const {
    return parent_trees_;
  }
  const std::vector<CubeTable> &cubes() const { return cubes_; }
  const std::vector<BoundaryView> &boundaryViews() const {
    return boundary_views_;
  }

  const CubeTable &cube(const std::string &tree_id,
                        const std::vector<VariableId> &scope) const;
  const BoundaryView &boundaryView(const std::string &view_id) const;
  const BoundaryQueryPlan &queryPlan(const std::string &query_id) const;

  BoundaryJoinResult execute(const std::string &query_id,
                             std::vector<VariableId> output_scope = {},
                             bool retain_rows = false) const;

  // Canonicalizes and validates one single-treatment/single-outcome query,
  // then answers it from prebuilt Cubes and attachment/Boundary views only.
  // conditioning_variables may contain any number of adjustment variables.
  BoundaryJoinResult execute(const DirectedQuerySpec &query,
                             std::vector<VariableId> output_scope = {},
                             bool retain_rows = false) const;

  static std::string attachmentViewId(const VariableId &variable);

  // Fixed observation population: never expands the scope or reduces Z.
  // Causal validity is checked by the caller. Cube-only forbids online joins.
  BoundaryJoinResult executeExactScope(
      const ExactScopeSpec &query, std::vector<VariableId> output_scope = {},
      bool retain_rows = false, bool cube_only = false,
      OnlineScopeStats *stats = nullptr, bool plan_only = false) const;

private:
  struct ScopeHash {
    std::size_t operator()(const std::vector<VariableId> &scope) const {
      std::size_t hash = 0;
      for (const auto &variable : scope) {
        hash ^= std::hash<VariableId>{}(variable) + 0x9e3779b9U +
                (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };
  void rebuildCubeLookup();
  std::unordered_map<std::vector<VariableId>, std::size_t, ScopeHash> cube_lookup_;
  SnapshotStorageStats
  saveWithStatsImpl(const std::string &path,
                    const ExternalCubeStorage *external_cubes) const;

  std::vector<CausalEdge> causal_edges_;
  std::vector<ParentTreeSpec> parent_trees_;
  BoundaryPlan plan_;
  std::vector<CubeTable> cubes_;
  std::vector<BoundaryView> boundary_views_;
};

} // namespace causal_index_v5
