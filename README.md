# Whatif-Index

C++17 implementation of a cube-based index for what-if queries on property-graph pattern matches. The index materializes local frequency cuboids and mapping-preserving boundary views. Queries use a local cuboid when available, or join saved views before evaluating the adjustment formula.

## Included components

- `v4_full_lattice/`: exact subgraph-join executor, prefix reuse, ordering, and full-join/rollup support.
- `v6_wide_scope/`: decomposition, adjustment-scope planning, local cuboids, boundary views, snapshot persistence, and online workers. Global causal scopes use Boost.Multiprecision; local cube masks remain bounded-width.
- `baselines/sqlite/`: SQLite joins over materialized variable-match tables, plus correctness tests and a portable command-line entry point. This is the adapted relational baseline called **CaRL-PG** in the experiments, not the original CaRL implementation.
- `examples/tiny/`: a synthetic input with a known answer.

Directory and C++ namespace version identifiers are retained to avoid changing implementation interfaces. `causal_index_v5` is also used by the wide-scope implementation.

## Requirements

- Linux, CMake 3.16 or later, and a C++17 compiler.
- Boost headers (Multiprecision and PropertyTree); no separately linked Boost library is needed.
- Python 3 with the standard-library `sqlite3` module for the SQLite baseline.

On Debian/Ubuntu, the relevant packages are `build-essential`, `cmake`, `libboost-dev`, and `python3`.

## Build and test

Run from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s baselines/sqlite -p 'test_*.py' -v
```

The C++ tests cover boundary joins, snapshot round trips, wide causal scopes, rollup completion, streaming construction, sampling, and join metrics. Test assertions remain enabled in Release builds.

The initial release was checked in a clean Linux build with GCC 13.3.0 and Boost 1.83.0: all six CTest tests and all ten SQLite tests passed. Both quick-start queries returned the expected answer. GCC emitted a warning in Boost.Multiprecision headers during the build; it did not prevent compilation or test completion.

## Quick start

Build a snapshot from the synthetic materialized matches:

```sh
mkdir -p results
./build/v6_wide_scope/causal_index_v5_boundary_benchmark \
  examples/tiny boundary-adjustment-prune results/tiny.bin 6
```

Evaluate the supplied query:

```sh
./build/v6_wide_scope/causal_index_online_worker \
  results/tiny.bin examples/tiny/causal.tsv Ours \
  < examples/tiny/query.json
```

The worker emits a readiness record followed by the query response. The example has four complete mappings and returns `P(Y=1 | do(X=1)) = 1`.

Run the equivalent SQLite baseline:

```sh
python3 baselines/sqlite/run.py prepare examples/tiny results/tiny.sqlite
python3 baselines/sqlite/run.py query results/tiny.sqlite examples/tiny/query.json
```

Preparation refuses to overwrite an existing SQLite database. Use a fresh output path when rerunning it.

## Input contract

The construction program starts from **materialized per-variable pattern matches**, not raw property-graph files. The upstream matcher must enforce each variable's pattern and use consistent canonical role names across variables.

`assignments_raw.tsv` has three tab-separated fields, without a header:

```text
role1=node1|role2=node2    variable    value
```

The spaces above stand for tabs. Values and IDs must not contain the format's separators. Each row supplies the full node-role mapping for one variable occurrence. Shared roles are joined by node identity; equal variable values are not join keys. Distinct roles must satisfy the implementation's global injectivity constraint. Required bridge relations must be represented in the input and included in the matching scope.

`causal.tsv` contains a variable and its comma-separated parents, separated by a tab. A root variable has an empty parent field. SQLite additionally requires `variable_manifest.csv`, with columns `variable_id,roles`; multiple roles are separated by `|`.

The released SQLite adapter assumes a unique node-role tuple per variable with a deterministic value. It rejects duplicate tuples. This adapter does not encode distinct parallel-edge identities; do not use it to claim equivalence for edge-distinct match populations without extending the representation.

## Query interface

The online worker accepts one JSON object per input line:

```json
{"X":"X","Y":"Y","Z":[],"x":"1","y":"1","match_scope":["X","Y"]}
```

`X` and `Y` identify the treatment and outcome, `Z` lists adjustment variables, and lowercase `x` and `y` are their requested values. `match_scope` includes all variables/bridge relations needed for the complete matching context. The response reports a scalar probability for the requested outcome value. Submit additional outcome values to obtain a distribution. `include_counts:true` adds validation counts.

The C++ worker validates the adjustment set. The SQLite adapter evaluates the supplied scope and counts; callers must supply a causally valid query. Empty populations and missing treatment support must be handled as non-answer statuses, not zero probabilities.

## Construction variants

The benchmark interface is:

```text
DATASET_DIR METHOD SNAPSHOT [PARENT_K=6] [INSTANCE_CAP=0] [SCOPE_ORDER_SEED=20260716]
```

| Purpose | METHOD |
| --- | --- |
| FullCube | `full-join` |
| FullCube with rollup completion | `full-join-rollup` |
| Decomposition without reuse or pruning | `parent-no-reuse` |
| Proposed construction | `boundary-adjustment-prune` |
| Random-order ablation | `boundary-adjustment-random-prune` |
| No-pruning ablation | `boundary-adjustment-no-prune` |
| No-reuse ablation | `boundary-adjustment-prune-no-reuse` |

`K` controls planner merging; it does not split an oversized atomic parent scope. `INSTANCE_CAP=0` applies no additional instance cap. Construction reports loading, planning, construction and snapshot metrics separately. `build_join_count` counts calls, whereas `join_work_pairs` counts candidate pair work; these are different measurements.

`CAUSAL_INDEX_SCALE_PERCENT` optionally subsamples each variable's input rows. This is instance subsampling, **not** induced-subgraph sampling. Preserve the intended population and measurement protocol when comparing experiments.

## Release scope and reproducibility

This source release includes the construction/query core and synthetic tests. It does not include the raw benchmark datasets, materialized benchmark inputs, prebuilt indexes, cDAG-discovery/LLM pipeline, Neo4j deployment scripts, or server-specific experiment queues. Full paper-scale reproduction requires those inputs and the exact workload/protocol; the quick start is a functional example, not a reproduction of the reported timings.

Compare methods on identical match populations and queries. Record SQLite settings, warm-up, repetitions, concurrency, timing boundaries, and failures. Database loading/index creation is separate from online query execution. Do not silently discard unsupported or timed-out queries when reporting means.

## Publication and anonymity

This repository is hosted under a personal GitHub account and is therefore not an anonymous review artifact. A separate anonymized distribution is needed for double-blind review. No source-server paths or private deployment configuration are required by the released commands.

No project-wide license is declared in this initial release; licensing remains to be selected by the repository owner. Preserve any third-party attribution already present in source files.
