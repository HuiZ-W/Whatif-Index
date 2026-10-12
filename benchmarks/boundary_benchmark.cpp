#include "whatif/AdjustmentScopePlanner.h"
#include "whatif/BoundaryJoinIndex.h"
#include "whatif/ParentTreePlanner.h"
#include "whatif/PatternManifest.h"
#include <boost/property_tree/json_parser.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace whatif;

namespace {

constexpr std::uint64_t kSampleSeed = 20260716ULL;

struct CausalInput {
  std::vector<CausalEdge> edges;
  std::vector<std::vector<VariableId>> atomic_scopes;
};

std::vector<std::string> split(const std::string &text, char delimiter) {
  std::vector<std::string> result;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    result.push_back(item);
  }
  return result;
}

std::vector<std::string> sortedUnique(std::vector<std::string> values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

std::string scopeKey(const std::vector<std::string> &scope) {
  std::string result;
  for (const auto &variable : scope) {
    result += variable;
    result.push_back('\x1f');
  }
  return result;
}

std::vector<RoleBinding> parseRoles(const std::string &sample) {
  std::vector<RoleBinding> roles;
  for (const auto &part : split(sample, '|')) {
    const auto equals = part.find('=');
    if (equals == std::string::npos) {
      continue;
    }
    roles.push_back({part.substr(0, equals), part.substr(equals + 1)});
  }
  std::sort(roles.begin(), roles.end(),
            [](const auto &left, const auto &right) {
              return left.role < right.role;
            });
  return roles;
}

std::vector<VariableInput> loadVariables(const std::filesystem::path &path,
                                         std::size_t instance_cap,
                                         std::size_t scale_percent = 100) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open assignments: " + path.string());
  }

  std::map<VariableId, std::vector<whatif::core::MatchInstance>> instances;
  std::map<VariableId, std::set<RoleId>> roles;
  const auto schemas = loadPatternSchemas(path.parent_path() / "variable_manifest.csv");
  for (const auto &[v, schema] : schemas) {
    instances[v];
    roles[v].insert(schema.begin(), schema.end());
  }
  InstanceId next_id = 1;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto columns = split(line, '\t');
    if (columns.size() < 3) {
      continue;
    }
    auto parsed_roles = parseRoles(columns[0]);
    if (!schemas.empty()) {
      std::vector<RoleId> actual;
      for (const auto &r : parsed_roles) actual.push_back(r.role);
      const auto s = schemas.find(columns[1]);
      if (s == schemas.end() || actual != s->second) throw std::runtime_error("assignment/manifest schema mismatch");
    }
    for (const auto &binding : parsed_roles) {
      roles[columns[1]].insert(binding.role);
    }
    instances[columns[1]].push_back(
        {next_id++, columns[2], std::move(parsed_roles), columns[2]});
  }

  std::vector<VariableInput> variables;
  variables.reserve(instances.size());
  std::size_t variable_index = 0;
  for (auto &entry : instances) {
    auto &rows = entry.second;
    const auto scaled_size = std::max<std::size_t>(
        1, (rows.size() / 100) * scale_percent +
               (rows.size() % 100) * scale_percent / 100);
    const auto keep = instance_cap == 0 ? scaled_size : std::min(scaled_size, instance_cap);
    if (rows.size() > keep) {
      std::mt19937_64 random(kSampleSeed +
                             0x9e3779b97f4a7c15ULL * (variable_index + 1));
      std::shuffle(rows.begin(), rows.end(), random);
      rows.resize(keep);
      std::sort(rows.begin(), rows.end(),
                [](const auto &left, const auto &right) {
                  return left.id < right.id;
                });
    }
    const auto &role_set = roles[entry.first];
    variables.push_back(
        {entry.first, {role_set.begin(), role_set.end()}, std::move(rows)});
    ++variable_index;
  }
  return variables;
}

CausalInput loadCausalInput(const std::filesystem::path &path,
                            const std::vector<VariableInput> &variables) {
  const std::set<VariableId> selected = [&] {
    std::set<VariableId> result;
    for (const auto &variable : variables) {
      result.insert(variable.id);
    }
    return result;
  }();

  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open causal DAG: " + path.string());
  }

  CausalInput result;
  std::set<std::pair<VariableId, VariableId>> edges;
  std::set<std::string> scopes;
  std::set<VariableId> covered_children;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto columns = split(line, '\t');
    if (columns.empty() || selected.count(columns[0]) == 0) {
      continue;
    }
    std::vector<VariableId> scope{columns[0]};
    if (columns.size() > 1 && !columns[1].empty()) {
      for (const auto &parent : split(columns[1], ',')) {
        if (!parent.empty() && selected.count(parent) != 0) {
          scope.push_back(parent);
          edges.emplace(parent, columns[0]);
        }
      }
    }
    scope = sortedUnique(std::move(scope));
    if (scopes.insert(scopeKey(scope)).second) {
      result.atomic_scopes.push_back(std::move(scope));
    }
    covered_children.insert(columns[0]);
  }

  for (const auto &variable : variables) {
    if (covered_children.count(variable.id) == 0) {
      result.atomic_scopes.push_back({variable.id});
    }
  }
  for (const auto &edge : edges) {
    result.edges.push_back({edge.first, edge.second});
  }
  std::sort(result.atomic_scopes.begin(), result.atomic_scopes.end());
  return result;
}

std::size_t totalInstances(const std::vector<VariableInput> &variables) {
  std::size_t result = 0;
  for (const auto &variable : variables) {
    result += variable.instances.size();
  }
  return result;
}

std::size_t maximumTreeWidth(const std::vector<ParentTreeSpec> &trees) {
  std::size_t result = 0;
  for (const auto &tree : trees) {
    result = std::max(result, tree.variables.size());
  }
  return result;
}

template <typename Function> double measureMilliseconds(Function &&function) {
  const auto start = std::chrono::steady_clock::now();
  function();
  const auto finish = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

void printUsage(const char *program) {
  std::cerr << "usage: " << program
            << " DATASET_DIR METHOD SNAPSHOT [WIDTH_BOUND=6] [INSTANCE_CAP=0] "
               "[SCOPE_ORDER_SEED=20260716]\n"
            << "METHOD: FullCube | FullCube+ | FaCT | FaCT-O | FaCT-P | FaCT-R | FaCT-ORP\n";
}

} // namespace

int main(int argc, char **argv) {
  std::atomic<std::uint64_t> build_join_count{0};
  std::atomic<std::uint64_t> join_work_pairs{0};
  std::atomic<std::uint64_t> join_probe_rows{0};
  std::mutex metric_output_mutex;
  std::filesystem::path temporary_cube_spool;
  try {
    if (argc < 4 || argc > 7) {
      printUsage(argv[0]);
      return 64;
    }

    const std::filesystem::path dataset = argv[1];
    const std::string method = argv[2];
    if (method != "FullCube" && method != "FullCube+" &&
        method != "FaCT" && method != "FaCT-O" &&
        method != "FaCT-P" && method != "FaCT-R" && method != "FaCT-ORP") {
      throw std::runtime_error("unknown construction method: " + method);
    }
    const std::filesystem::path snapshot = argv[3];
    const std::size_t parent_k =
        argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 6;
    const std::size_t instance_cap =
        argc > 5 ? static_cast<std::size_t>(std::stoull(argv[5])) : 0;
    const std::uint64_t scope_order_seed =
        argc > 6 ? static_cast<std::uint64_t>(std::stoull(argv[6]))
                 : kSampleSeed;
    if (parent_k == 0) {
      throw std::runtime_error("WIDTH_BOUND must be positive");
    }
    const auto scale_text = std::getenv("CAUSAL_INDEX_SCALE_PERCENT");
    if (scale_text && std::string(scale_text) != "100")
      throw std::runtime_error("row subsampling is not a PG scale experiment; use scripts/prepare_induced_scale.py");
    if (std::filesystem::exists(dataset.parent_path() / "INCOMPLETE"))
      throw std::runtime_error("induced scale preparation is incomplete");
    std::size_t scale_percent = 100;
    if (std::filesystem::exists(dataset / "scale_manifest.json")) {
      boost::property_tree::ptree scale;
      boost::property_tree::read_json((dataset / "scale_manifest.json").string(), scale);
      scale_percent = scale.get<std::size_t>("percent");
      if (scale_percent < 1 || scale_percent > 100) throw std::runtime_error("invalid scale manifest percent");
    }

    std::vector<VariableInput> variables;
    CausalInput causal;
    double load_ms = measureMilliseconds([&] {
      variables = loadVariables(dataset / "assignments_raw.tsv", instance_cap);
      causal = loadCausalInput(dataset / "causal.tsv", variables);
    });
    if (variables.empty()) {
      throw std::runtime_error("dataset contains no causal-variable instances");
    }
    const std::size_t variable_count = variables.size();
    const std::size_t input_instances = totalInstances(variables);

    std::vector<ParentTreeSpec> trees;
    std::optional<ParentTreePlan> parent_tree_plan;
    std::optional<AdjustmentScopePlan> adjustment_scope_plan;
    double adjustment_plan_ms = 0.0;
    double parent_plan_ms = 0.0;
    BoundaryBuildConfig config;
    config.tree_executor.binary_join_callback = [&] {
      const auto count =
          build_join_count.fetch_add(1, std::memory_order_relaxed) + 1;
      if (count % 1000 == 0) {
        std::lock_guard<std::mutex> lock(metric_output_mutex);
        std::cout << "build_join_count_partial=" << count << std::endl;
      }
    };
    config.tree_executor.independent_leaf_pushdown = true;
    config.tree_executor.join_work_callback = [&](std::size_t rows, std::size_t pairs) {
      auto checked_atomic_add = [](std::atomic<std::uint64_t> &target,
                                   std::size_t increment) {
        if (increment > std::numeric_limits<std::uint64_t>::max()) {
          throw std::overflow_error("join workload counter overflow");
        }
        const auto amount = static_cast<std::uint64_t>(increment);
        auto current = target.load(std::memory_order_relaxed);
        while (true) {
          if (amount > std::numeric_limits<std::uint64_t>::max() - current) {
            throw std::overflow_error("join workload counter overflow");
          }
          if (target.compare_exchange_weak(
                  current, current + amount, std::memory_order_relaxed)) {
            break;
          }
        }
      };
      checked_atomic_add(join_probe_rows, rows);
      checked_atomic_add(join_work_pairs, pairs);
      if (build_join_count.load(std::memory_order_relaxed) % 1000 == 0) {
        std::lock_guard<std::mutex> lock(metric_output_mutex);
        std::cout << "join_work_pairs_partial="
                  << join_work_pairs.load(std::memory_order_relaxed) << '\n'
                  << "join_probe_rows_partial="
                  << join_probe_rows.load(std::memory_order_relaxed)
                  << std::endl;
      }
    };
    config.cube_policy = LocalCubePolicy::AllSubsets;
    config.tree_executor.worker_threads = 1;
    config.tree_executor.enable_value_partition_join = false;
    config.tree_executor.enable_derive = false;
    config.tree_executor.store_all_supports = false;
    config.tree_executor.store_value_counts = true;
    config.tree_executor.progress_every_tasks = 1000;
    config.tree_executor.progress_callback =
        [&](const whatif::core::ProgressEvent &event) {
          std::lock_guard<std::mutex> lock(metric_output_mutex);
          std::cout << "progress" << "\tcompleted=" << event.completed_tasks
                    << '/' << event.total_tasks << "\tdepth=" << event.depth
                    << "\trows=" << event.rows
                    << "\tresident=" << event.resident_tuples
                    << "\tpeak_resident=" << event.peak_resident_tuples
                    << "\temitted=" << event.emitted_rows << '\n';
          std::cout.flush();
        };

    std::vector<CausalEdge> build_edges = causal.edges;
    if (method == "FullCube" || method == "FullCube+") {
      config.stream_all_subset_cubes = true;
      config.tree_executor.full_join_rollup = method == "FullCube+";
      config.tree_executor.rollup_callback = [&](const whatif::core::RollupStats& s) {
        std::lock_guard<std::mutex> lock(metric_output_mutex);
        std::cout << "rollup_used=" << s.used << '\n'
                  << "rollup_projected_rows=" << s.projected_rows << '\n'
                  << "rollup_residual_rows=" << s.residual_rows << '\n'
                  << "rollup_donor_rows_scanned=" << s.donor_rows_scanned << '\n'
                  << "rollup_witness_rows_scanned=" << s.witness_rows_scanned << '\n'
                  << "rollup_peak_cache_bytes=" << s.peak_cache_bytes << std::endl;
      };
      if (config.tree_executor.full_join_rollup)
        std::cout << "rollup_policy=safe_key_projection_plus_residual_join\n"
                  << "rollup_cache_budget_bytes=" << config.tree_executor.rollup_cache_bytes << '\n'
                  << "scope_schedule=alphabetical_dfs_independent_sources" << std::endl;
      std::cout << "full_join_materialization=incremental_all_subsets\n"
                << "progress_unit=completed_cubes" << std::endl;
      std::vector<VariableId> all_variables;
      for (const auto &variable : variables) {
        all_variables.push_back(variable.id);
      }
      trees.push_back({"global", std::move(all_variables)});
      build_edges.clear();
      config.include_all_reachable_pairs = false;
      config.max_all_subset_width = variables.size();
      config.tree_executor.variable_order_policy =
          whatif::core::VariableOrderPolicy::Alphabetical;
      config.tree_executor.execution_schedule =
          whatif::core::ExecutionSchedule::IndependentSources;
    } else {
      parent_plan_ms = measureMilliseconds([&] {
        parent_tree_plan = ParentTreePlanner::plan(
            variables, causal.edges, causal.atomic_scopes, parent_k);
      });
      trees = parent_tree_plan->trees;
      config.include_all_reachable_pairs = true;
      config.max_all_subset_width = maximumTreeWidth(trees);
      auto configure_adjustment_workload = [&] {
        adjustment_plan_ms = measureMilliseconds([&] {
          adjustment_scope_plan = AdjustmentScopePlanner::compileSingleDo(
              variables, causal.edges, trees, causal.atomic_scopes, {}, true);
        });
        config.include_all_reachable_pairs = false;
        config.materialize_variable_attachments = true;
        config.materialize_structural_boundaries =
            std::getenv("CAUSAL_INDEX_STRUCTURAL_BOUNDARIES") != nullptr;
        if (config.materialize_structural_boundaries)
          std::cout << "boundary_policy=query_independent_directed_tree_boundaries\n";
        config.query_workload = adjustment_scope_plan->query_workload;
        config.precompiled_boundary_plan =
            std::move(adjustment_scope_plan->boundary_plan);
        config.tree_executor.variable_order_policy =
            whatif::core::VariableOrderPolicy::PartialWorkloadGreedy;
        config.tree_executor.enable_derive = method != "FaCT-R" && method != "FaCT-ORP";
        // Use FaCT's retained targets for the order in the pruning ablation;
        // only the executed/materialized scopes change in FaCT-P.
        for (const auto &[tree_id, scopes] : adjustment_scope_plan->explicit_cube_scopes)
          config.ordering_scopes.insert(config.ordering_scopes.end(), scopes.begin(), scopes.end());
      };

      {
        const bool prune_local_scopes =
            method == "FaCT" || method == "FaCT-R" || method == "FaCT-O";
        const bool random_scope_order = method == "FaCT-O";
        configure_adjustment_workload();
        if (method == "FaCT-ORP") {
          // Keep the same decomposition and boundary workload, but construct
          // every local scope independently using a fixed variable order.
          config.tree_executor.variable_order_policy =
              whatif::core::VariableOrderPolicy::Alphabetical;
          config.tree_executor.plan_order_from_unpruned_lattice = false;
        }
        if (method == "FaCT-R" || method == "FaCT-ORP") {
          config.tree_executor.execution_schedule =
              whatif::core::ExecutionSchedule::IndependentSources;
        } else if (random_scope_order) {
          config.tree_executor.variable_order_policy = whatif::core::VariableOrderPolicy::Random;
          config.tree_executor.variable_order_seed = scope_order_seed;
          config.tree_executor.execution_schedule =
              whatif::core::ExecutionSchedule::ScopeOrderedWorkingPath;
          config.tree_executor.scope_order_policy =
              whatif::core::ScopeOrderPolicy::Random;
          config.tree_executor.scope_order_seed = scope_order_seed;
          config.tree_executor.scope_order_from_unpruned_lattice = true;
        } else {
          config.tree_executor.execution_schedule =
              whatif::core::ExecutionSchedule::DfsPrefixTree;
        }
        if (prune_local_scopes) {
          config.cube_policy = LocalCubePolicy::ExplicitScopes;
          config.explicit_cube_scopes =
              adjustment_scope_plan->explicit_cube_scopes;
        }
      }
    }

    const bool dump_scopes = std::getenv("CAUSAL_INDEX_DUMP_SCOPES") != nullptr;
    const bool plan_only = std::getenv("CAUSAL_INDEX_PLAN_ONLY") != nullptr;
    if (dump_scopes) {
      for (const auto &tree : trees) {
        std::cout << "plan_tree=" << tree.id;
        for (const auto &variable : tree.variables) {
          std::cout << '\t' << variable;
        }
        std::cout << '\n';
      }
      if (adjustment_scope_plan) {
        for (const auto &[tree_id, scopes] :
             adjustment_scope_plan->explicit_cube_scopes) {
          for (const auto &scope : scopes) {
            std::cout << "plan_scope=" << tree_id;
            for (const auto &variable : scope) {
              std::cout << '\t' << variable;
            }
            std::cout << '\n';
          }
        }
        for (const auto &scope : adjustment_scope_plan->exact_scope_workload) {
          std::cout << "plan_exact_scope=" << scope.id;
          for (const auto &variable : scope.scope) {
            std::cout << '\t' << variable;
          }
          std::cout << '\n';
        }
      }
    }
    if (plan_only) {
      std::cout << "plan_only=1\n";
      std::cout << "parent_trees=" << trees.size() << '\n';
      std::cout << "localizable_query_scopes="
                << (parent_tree_plan
                        ? parent_tree_plan->stats.localizable_query_scopes
                        : 0)
                << '\n';
      std::cout << "covered_localizable_query_scopes="
                << (parent_tree_plan
                        ? parent_tree_plan->stats
                              .covered_localizable_query_scopes
                        : 0)
                << '\n';
      std::cout << "adjustment_retained_local_scopes="
                << (adjustment_scope_plan
                        ? adjustment_scope_plan->retained_local_cube_scopes
                        : 0)
                << '\n';
      std::cout << "cost_merge_candidates="
                << (parent_tree_plan
                        ? parent_tree_plan->stats.cost_merge_candidates
                        : 0)
                << '\n';
      std::cout << "cost_rejected_merges="
                << (parent_tree_plan
                        ? parent_tree_plan->stats.cost_rejected_merges
                        : 0)
                << '\n';
      std::cout << "estimated_initial_tree_cost="
                << (parent_tree_plan
                        ? parent_tree_plan->stats.estimated_initial_tree_cost
                        : 0.0)
                << '\n';
      std::cout << "estimated_final_tree_cost="
                << (parent_tree_plan
                        ? parent_tree_plan->stats.estimated_final_tree_cost
                        : 0.0)
                << '\n';
      std::cout << "estimated_boundary_cost="
                << (parent_tree_plan
                        ? parent_tree_plan->stats.estimated_boundary_cost
                        : 0.0)
                << '\n';
      std::cout << "estimated_total_cost="
                << (parent_tree_plan
                        ? parent_tree_plan->stats.estimated_total_cost
                        : 0.0)
                << '\n';
      return 0;
    }

    if (!snapshot.parent_path().empty()) {
      std::filesystem::create_directories(snapshot.parent_path());
    }
    const bool stream_cubes_to_disk =
        method == "FullCube" || method == "FullCube+";
    ExternalCubeStorage external_cubes;
    std::ofstream cube_spool;
    double cube_spool_write_ms = 0.0;
    if (stream_cubes_to_disk) {
      external_cubes.path = snapshot.string() + ".cube-spool.tmp";
      temporary_cube_spool = external_cubes.path;
      cube_spool.open(external_cubes.path,
                      std::ios::binary | std::ios::trunc);
      if (!cube_spool) {
        throw std::runtime_error("cannot open cube spool for writing: " +
                                 external_cubes.path);
      }
      config.streamed_cube_sink = [&](CubeTable &&cube) {
        const auto start = std::chrono::steady_clock::now();
        external_cubes.cell_count += cube.counts.size();
        MaterializedBoundaryIndex::writeCubeRecord(cube_spool, cube);
        ++external_cubes.table_count;
        const auto finish = std::chrono::steady_clock::now();
        cube_spool_write_ms +=
            std::chrono::duration<double, std::milli>(finish - start).count();
      };
    }

    MaterializedBoundaryIndex index;
    const double build_ms = measureMilliseconds([&] {
      index = MaterializedBoundaryIndex::build(std::move(variables),
                                               std::move(build_edges), trees,
                                               std::move(config));
    });
    if (stream_cubes_to_disk) {
      cube_spool.flush();
      if (!cube_spool) {
        throw std::runtime_error("failed to finish cube spool");
      }
      cube_spool.close();
      external_cubes.payload_bytes =
          std::filesystem::file_size(external_cubes.path);
    }
    std::cout << "build_join_count="
              << build_join_count.load(std::memory_order_relaxed) << std::endl;
    std::cout << "join_work_pairs="
              << join_work_pairs.load(std::memory_order_relaxed) << '\n'
              << "join_probe_rows="
              << join_probe_rows.load(std::memory_order_relaxed) << std::endl;
    std::cout << "scale_percent=" << scale_percent << '\n';
    const double construction_ms =
        parent_plan_ms + adjustment_plan_ms + build_ms;
    std::cout << "construction_ms=" << std::setprecision(12)
              << construction_ms << '\n';

    SnapshotStorageStats storage;
    const double snapshot_ms = measureMilliseconds([&] {
      storage = stream_cubes_to_disk
          ? index.saveWithExternalCubes(snapshot, external_cubes)
          : index.saveWithStats(snapshot);
    });
    if (stream_cubes_to_disk) {
      std::filesystem::remove(external_cubes.path);
      temporary_cube_spool.clear();
    }

    std::size_t cube_cells = external_cubes.cell_count;
    if (!stream_cubes_to_disk) {
      for (const auto &cube : index.cubes()) {
        cube_cells += cube.counts.size();
      }
    }
    std::size_t boundary_rows = 0;
    std::size_t attachment_views = 0;
    std::size_t attachment_rows = 0;
    Count boundary_weight = 0;
    for (const auto &view : index.boundaryViews()) {
      boundary_rows += view.rows.size();
      if (view.tree_id == "@attachment") {
        attachment_views += 1;
        attachment_rows += view.rows.size();
      }
      for (const auto &row : view.rows) {
        if (row.multiplicity >
            std::numeric_limits<Count>::max() - boundary_weight) {
          throw std::overflow_error("Boundary weight metric overflow");
        }
        boundary_weight += row.multiplicity;
      }
    }
    std::size_t cross_tree_queries = 0;
    for (const auto &query : index.plan().queries) {
      if (!query.view_ids.empty()) {
        ++cross_tree_queries;
      }
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "dataset=" << dataset.string() << '\n';
    std::cout << "method=" << method << '\n';
    const auto &shared = index.constructionStats();
    std::cout << "shared_unique_scopes=" << shared.unique_scopes << '\n'
              << "shared_prefix_extensions=" << shared.prefix_extensions << '\n'
              << "recovery_attempts=" << shared.recovery_attempts << '\n'
              << "recovery_used=" << shared.recovery_used << '\n'
              << "recovered_rows=" << shared.recovered_rows << '\n'
              << "donor_peak_bytes=" << shared.donor_peak_bytes << '\n';
    std::cout << "shared_variable_order";
    for (const auto &v : shared.variable_order) std::cout << '\t' << v;
    std::cout << '\n';
    std::cout << "scope_order_seed=" << scope_order_seed << '\n';
    std::cout << "worker_threads=1\n";
    std::cout << "cube_storage="
              << (stream_cubes_to_disk ? "disk-spool" : "memory") << '\n';
    std::cout << "variables=" << variable_count << '\n';
    std::cout << "input_instances=" << input_instances << '\n';
    std::cout << "causal_edges=" << causal.edges.size() << '\n';
    std::cout << "atomic_parent_factors=" << causal.atomic_scopes.size()
              << '\n';
    std::cout << "oversized_atomic_parent_factors="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.oversized_atomic_factors
                      : 0)
              << '\n';
    std::cout << "reachable_causal_pairs="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.reachable_causal_pairs
                      : 0)
              << '\n';
    std::cout << "identifiable_query_scopes="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.identifiable_query_scopes
                      : 0)
              << '\n';
    std::cout << "localizable_query_scopes="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.localizable_query_scopes
                      : 0)
              << '\n';
    std::cout << "oversized_query_scopes="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.oversized_query_scopes
                      : 0)
              << '\n';
    std::cout << "maximal_query_seeds="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.maximal_query_seeds
                      : 0)
              << '\n';
    std::cout << "covered_localizable_query_scopes="
              << (parent_tree_plan
                      ? parent_tree_plan->stats
                            .covered_localizable_query_scopes
                      : 0)
              << '\n';
    std::cout << "preserved_oversized_parent_trees="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.preserved_oversized_trees
                      : 0)
              << '\n';
    std::cout << "cost_merge_candidates="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.cost_merge_candidates
                      : 0)
              << '\n';
    std::cout << "cost_rejected_merges="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.cost_rejected_merges
                      : 0)
              << '\n';
    std::cout << "estimated_initial_tree_cost="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.estimated_initial_tree_cost
                      : 0.0)
              << '\n';
    std::cout << "estimated_final_tree_cost="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.estimated_final_tree_cost
                      : 0.0)
              << '\n';
    std::cout << "estimated_boundary_cost="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.estimated_boundary_cost
                      : 0.0)
              << '\n';
    std::cout << "estimated_total_cost="
              << (parent_tree_plan
                      ? parent_tree_plan->stats.estimated_total_cost
                      : 0.0)
              << '\n';
    std::cout << "parent_trees=" << index.parentTrees().size() << '\n';
    std::cout << "max_parent_tree_width="
              << maximumTreeWidth(index.parentTrees()) << '\n';
    std::cout << "query_plans=" << index.plan().queries.size() << '\n';
    std::cout << "cross_tree_query_plans=" << cross_tree_queries << '\n';
    std::cout << "boundary_requirements=" << index.plan().requirements.size()
              << '\n';
    std::cout << "cube_tables="
              << (stream_cubes_to_disk ? external_cubes.table_count
                                       : index.cubes().size())
              << '\n';
    std::cout << "cube_cells=" << cube_cells << '\n';
    std::cout << "boundary_views=" << index.boundaryViews().size() << '\n';
    std::cout << "attachment_views=" << attachment_views << '\n';
    std::cout << "attachment_rows=" << attachment_rows << '\n';
    std::cout << "boundary_rows=" << boundary_rows << '\n';
    std::cout << "boundary_weight=" << boundary_weight << '\n';
    std::cout << "adjustment_candidate_queries="
              << (adjustment_scope_plan
                      ? adjustment_scope_plan->candidate_queries
                      : 0)
              << '\n';
    std::cout << "adjustment_pruning_semantics="
              << (adjustment_scope_plan
                      ? "single-X-single-Y-multi-Z-dsep-reduction"
                      : "none")
              << '\n';
    std::cout << "adjustment_retained_queries="
              << (adjustment_scope_plan
                      ? adjustment_scope_plan->retained_queries
                      : 0)
              << '\n';
    std::cout << "adjustment_adjusted_queries="
              << (adjustment_scope_plan
                      ? adjustment_scope_plan->adjusted_queries
                      : 0)
              << '\n';
    std::cout << "adjustment_unadjusted_queries="
              << (adjustment_scope_plan
                      ? adjustment_scope_plan->unadjusted_queries
                      : 0)
              << '\n';
    std::cout << "adjustment_trivial_queries_pruned="
              << (adjustment_scope_plan
                      ? adjustment_scope_plan->trivial_queries_pruned
                      : 0)
              << '\n';
    std::cout << "adjustment_query_shape_pruned="
              << (adjustment_scope_plan
                      ? adjustment_scope_plan->unsupported_query_shapes_pruned
                      : 0)
              << '\n';
    std::cout
        << "redundant_condition_variables_removed="
        << (adjustment_scope_plan
                ? adjustment_scope_plan->redundant_condition_variables_removed
                : 0)
        << '\n';
    std::cout << "adjustment_full_local_scopes="
              << (adjustment_scope_plan
                      ? adjustment_scope_plan->full_local_cube_scopes
                      : 0)
              << '\n';
    std::cout
        << "adjustment_valid_local_scopes="
        << (adjustment_scope_plan
                ? adjustment_scope_plan->valid_adjustment_local_cube_scopes
                : 0)
        << '\n';
    std::cout << "adjustment_retained_local_scopes="
              << (adjustment_scope_plan
                      ? adjustment_scope_plan->retained_local_cube_scopes
                      : 0)
              << '\n';
    std::cout << "adjustment_pruned_local_scopes="
              << (adjustment_scope_plan
                      ? adjustment_scope_plan->pruned_local_cube_scopes
                      : 0)
              << '\n';
    std::cout << "load_ms=" << load_ms << '\n';
    std::cout << "adjustment_plan_ms=" << adjustment_plan_ms << '\n';
    std::cout << "build_ms=" << build_ms << '\n';
    std::cout << "total_build_ms=" << adjustment_plan_ms + build_ms << '\n';
    std::cout << "cube_spool_write_ms=" << cube_spool_write_ms << '\n';
    std::cout << "snapshot_write_ms=" << snapshot_ms << '\n';
    std::cout << "index_total_bytes=" << storage.total_bytes << '\n';
    std::cout << "index_metadata_bytes=" << storage.metadata_bytes << '\n';
    std::cout << "index_cube_bytes=" << storage.cube_bytes << '\n';
    std::cout << "index_boundary_bytes=" << storage.boundary_bytes << '\n';
    std::cout << "snapshot=" << snapshot.string() << '\n';
    return 0;
  } catch (const std::exception &error) {
    if (!temporary_cube_spool.empty()) {
      std::error_code ignored;
      std::filesystem::remove(temporary_cube_spool, ignored);
    }
    std::cerr << "build_join_count_partial="
              << build_join_count.load(std::memory_order_relaxed) << '\n';
    std::cerr << "join_work_pairs_partial="
              << join_work_pairs.load(std::memory_order_relaxed) << '\n'
              << "join_probe_rows_partial="
              << join_probe_rows.load(std::memory_order_relaxed) << '\n';
    std::cerr << "error=" << error.what() << '\n';
    return 2;
  }
}
