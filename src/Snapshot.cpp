#include "whatif/BoundaryJoinIndex.h"

#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace whatif {
namespace {

// Keep the on-disk identifier unchanged so existing experimental snapshots load.
constexpr std::array<char, 8> kMagic{{'C', 'B', 'J', 'I', 'D', 'X', '5', '\0'}};
constexpr std::uint32_t kVersion = 2;
constexpr std::uint64_t kMaxCollectionSize = 1'000'000'000ULL;
constexpr std::uint64_t kMaxStringSize = 1ULL << 30;

template <typename T> void writePod(std::ostream &out, const T &value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "snapshot POD must be trivially copyable");
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
  if (!out) {
    throw std::runtime_error("failed to write Boundary snapshot");
  }
}

template <typename T> T readPod(std::istream &in) {
  static_assert(std::is_trivially_copyable<T>::value,
                "snapshot POD must be trivially copyable");
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error("truncated Boundary snapshot");
  }
  return value;
}

void writeSize(std::ostream &out, std::size_t size) {
  writePod(out, static_cast<std::uint64_t>(size));
}

std::size_t readSize(std::istream &in,
                     std::uint64_t maximum = kMaxCollectionSize) {
  const auto size = readPod<std::uint64_t>(in);
  if (size > maximum || size > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("invalid Boundary snapshot collection size");
  }
  return static_cast<std::size_t>(size);
}

void writeString(std::ostream &out, const std::string &value) {
  writeSize(out, value.size());
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!out) {
    throw std::runtime_error("failed to write Boundary snapshot string");
  }
}

std::string readString(std::istream &in) {
  const auto size = readSize(in, kMaxStringSize);
  std::string value(size, '\0');
  in.read(value.data(), static_cast<std::streamsize>(size));
  if (!in) {
    throw std::runtime_error("truncated Boundary snapshot string");
  }
  return value;
}

void writeStrings(std::ostream &out, const std::vector<std::string> &values) {
  writeSize(out, values.size());
  for (const auto &value : values) {
    writeString(out, value);
  }
}

void writeCube(std::ostream &out, const CubeTable &cube) {
  writeString(out, cube.tree_id);
  writeStrings(out, cube.scope);
  writeSize(out, cube.counts.size());
  for (const auto &entry : cube.counts) {
    writeStrings(out, entry.first);
    writePod(out, entry.second);
  }
}

std::vector<std::string> readStrings(std::istream &in) {
  const auto size = readSize(in);
  std::vector<std::string> values;
  values.reserve(size);
  for (std::size_t i = 0; i < size; ++i) {
    values.push_back(readString(in));
  }
  return values;
}

} // namespace

void MaterializedBoundaryIndex::save(const std::string &path) const {
  (void)saveWithStats(path);
}

SnapshotStorageStats
MaterializedBoundaryIndex::saveWithStats(const std::string &path) const {
  return saveWithStatsImpl(path, nullptr);
}

void MaterializedBoundaryIndex::writeCubeRecord(std::ostream &out,
                                                const CubeTable &cube) {
  writeCube(out, cube);
}

SnapshotStorageStats MaterializedBoundaryIndex::saveWithExternalCubes(
    const std::string &path, const ExternalCubeStorage &storage) const {
  if (storage.path.empty()) {
    throw std::runtime_error("external cube spool path is empty");
  }
  if (!cubes_.empty()) {
    throw std::runtime_error(
        "cannot combine retained cubes with an external cube spool");
  }
  return saveWithStatsImpl(path, &storage);
}

SnapshotStorageStats MaterializedBoundaryIndex::saveWithStatsImpl(
    const std::string &path,
    const ExternalCubeStorage *external_cubes) const {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open Boundary snapshot for writing: " +
                             path);
  }
  out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  writePod(out, kVersion);

  writeSize(out, causal_edges_.size());
  for (const auto &edge : causal_edges_) {
    writeString(out, edge.parent);
    writeString(out, edge.child);
  }

  writeSize(out, parent_trees_.size());
  for (const auto &tree : parent_trees_) {
    writeString(out, tree.id);
    writeStrings(out, tree.variables);
  }

  writeSize(out, plan_.requirements.size());
  for (const auto &requirement : plan_.requirements) {
    writeString(out, requirement.view_id);
    writeString(out, requirement.tree_id);
    writeStrings(out, requirement.scope);
    writePod(out, requirement.direction_flags);
    writeStrings(out, requirement.query_ids);
  }

  writeSize(out, plan_.queries.size());
  for (const auto &query : plan_.queries) {
    writeString(out, query.query_id);
    writeStrings(out, query.treatments);
    writeStrings(out, query.outcomes);
    writeStrings(out, query.conditioning_variables);
    writeStrings(out, query.required_variables);
    writeStrings(out, query.global_scope);
    writeStrings(out, query.view_ids);
  }

  const auto metadata_end = out.tellp();
  if (metadata_end < 0) {
    throw std::runtime_error("failed to measure Boundary snapshot metadata");
  }

  if (external_cubes) {
    writeSize(out, external_cubes->table_count);
    std::ifstream spool(external_cubes->path, std::ios::binary);
    if (!spool) {
      throw std::runtime_error("cannot open external cube spool: " +
                               external_cubes->path);
    }
    const auto payload_begin = out.tellp();
    if (payload_begin < 0) {
      throw std::runtime_error("failed to measure cube spool start");
    }
    out << spool.rdbuf();
    const auto payload_end = out.tellp();
    if (spool.bad() || !out || payload_end < payload_begin) {
      throw std::runtime_error("failed to copy external cube spool");
    }
    const auto copied = static_cast<std::uint64_t>(payload_end - payload_begin);
    if (copied != external_cubes->payload_bytes) {
      throw std::runtime_error("external cube spool size changed while saving");
    }
  } else {
    writeSize(out, cubes_.size());
    for (const auto &cube : cubes_) {
      writeCube(out, cube);
    }
  }

  const auto cubes_end = out.tellp();
  if (cubes_end < 0) {
    throw std::runtime_error("failed to measure Boundary snapshot cubes");
  }

  writeSize(out, boundary_views_.size());
  for (const auto &view : boundary_views_) {
    writeString(out, view.id);
    writeString(out, view.tree_id);
    writeStrings(out, view.scope);
    writeSize(out, view.rows.size());
    for (const auto &row : view.rows) {
      writeSize(out, row.bindings.size());
      for (const auto &binding : row.bindings) {
        writeString(out, binding.variable);
        writePod(out, binding.instance_id);
        writeString(out, binding.raw_value);
      }
      writeSize(out, row.role_nodes.size());
      for (const auto &role : row.role_nodes) {
        writeString(out, role.role);
        writeString(out, role.node);
      }
      writePod(out, row.multiplicity);
    }
  }
  out.flush();
  if (!out) {
    throw std::runtime_error("failed to finish Boundary snapshot: " + path);
  }

  const auto total_end = out.tellp();
  if (total_end < 0) {
    throw std::runtime_error("failed to measure Boundary snapshot size");
  }
  SnapshotStorageStats stats;
  stats.metadata_bytes = static_cast<std::uint64_t>(metadata_end);
  stats.cube_bytes = static_cast<std::uint64_t>(cubes_end - metadata_end);
  stats.boundary_bytes = static_cast<std::uint64_t>(total_end - cubes_end);
  stats.total_bytes = static_cast<std::uint64_t>(total_end);
  return stats;
}

MaterializedBoundaryIndex
MaterializedBoundaryIndex::load(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open Boundary snapshot: " + path);
  }
  std::array<char, 8> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!in || magic != kMagic) {
    throw std::runtime_error("invalid Boundary snapshot magic");
  }
  const auto version = readPod<std::uint32_t>(in);
  if (version != 1 && version != kVersion) {
    throw std::runtime_error("unsupported Boundary snapshot version");
  }

  MaterializedBoundaryIndex index;
  const auto edge_count = readSize(in);
  index.causal_edges_.reserve(edge_count);
  for (std::size_t i = 0; i < edge_count; ++i) {
    index.causal_edges_.push_back({readString(in), readString(in)});
  }

  const auto tree_count = readSize(in);
  index.parent_trees_.reserve(tree_count);
  for (std::size_t i = 0; i < tree_count; ++i) {
    ParentTreeSpec tree;
    tree.id = readString(in);
    tree.variables = readStrings(in);
    index.parent_trees_.push_back(std::move(tree));
  }

  const auto requirement_count = readSize(in);
  index.plan_.requirements.reserve(requirement_count);
  for (std::size_t i = 0; i < requirement_count; ++i) {
    BoundaryRequirement requirement;
    requirement.view_id = readString(in);
    requirement.tree_id = readString(in);
    requirement.scope = readStrings(in);
    requirement.direction_flags = readPod<std::uint8_t>(in);
    requirement.query_ids = readStrings(in);
    index.plan_.requirements.push_back(std::move(requirement));
  }

  const auto query_count = readSize(in);
  index.plan_.queries.reserve(query_count);
  for (std::size_t i = 0; i < query_count; ++i) {
    BoundaryQueryPlan query;
    query.query_id = readString(in);
    query.treatments = readStrings(in);
    query.outcomes = readStrings(in);
    if (version >= 2) {
      query.conditioning_variables = readStrings(in);
      query.required_variables = readStrings(in);
    }
    query.global_scope = readStrings(in);
    query.view_ids = readStrings(in);
    index.plan_.queries.push_back(std::move(query));
  }

  const auto cube_count = readSize(in);
  index.cubes_.reserve(cube_count);
  for (std::size_t i = 0; i < cube_count; ++i) {
    CubeTable cube;
    cube.tree_id = readString(in);
    cube.scope = readStrings(in);
    const auto entry_count = readSize(in);
    for (std::size_t entry = 0; entry < entry_count; ++entry) {
      auto key = readStrings(in);
      const auto value = readPod<Count>(in);
      if (!cube.counts.emplace(std::move(key), value).second) {
        throw std::runtime_error("duplicate cube key in Boundary snapshot");
      }
    }
    index.cubes_.push_back(std::move(cube));
  }

  const auto view_count = readSize(in);
  index.boundary_views_.reserve(view_count);
  for (std::size_t i = 0; i < view_count; ++i) {
    BoundaryView view;
    view.id = readString(in);
    view.tree_id = readString(in);
    view.scope = readStrings(in);
    const auto row_count = readSize(in);
    view.rows.reserve(row_count);
    for (std::size_t row_index = 0; row_index < row_count; ++row_index) {
      BoundaryRow row;
      const auto binding_count = readSize(in);
      row.bindings.reserve(binding_count);
      for (std::size_t binding = 0; binding < binding_count; ++binding) {
        row.bindings.push_back(
            {readString(in), readPod<InstanceId>(in), readString(in)});
      }
      const auto role_count = readSize(in);
      row.role_nodes.reserve(role_count);
      for (std::size_t role = 0; role < role_count; ++role) {
        row.role_nodes.push_back({readString(in), readString(in)});
      }
      row.multiplicity = readPod<Count>(in);
      if (row.multiplicity == 0) {
        throw std::runtime_error(
            "Boundary snapshot contains zero multiplicity");
      }
      view.rows.push_back(std::move(row));
    }
    index.boundary_views_.push_back(std::move(view));
  }

  if (in.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("Boundary snapshot has trailing bytes");
  }
  index.rebuildCubeLookup();
  return index;
}

} // namespace whatif
