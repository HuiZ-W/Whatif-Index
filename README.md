# Whatif-Index

C++17 implementation of **FaCT**, a cube-based index for what-if queries on property-graph pattern matches. The index materializes local frequency cuboids and mapping-preserving boundary views. **FaCTQuery** uses a local cuboid when available, or joins saved views before evaluating the adjustment formula.

## Included components

- `src/` and `include/whatif/`: one index implementation, including decomposition, adjustment-scope planning, local cuboids, boundary views, and snapshot persistence. The `core/` subdirectories contain its shared subgraph-join, prefix-reuse, ordering, and rollup execution primitives.
- `benchmarks/`: construction and online-query entry points for the proposed method, baselines, and ablations.
- `tests/`: regression tests and test-only reference code. The fixed-width causal model in `tests/reference/` is a comparison oracle, never linked into the production programs.
- `baselines/sqlite/`: SQLite joins over materialized variable-match tables, plus correctness tests and a portable command-line entry point. This is the adapted relational baseline called **CaRL-PG** in the experiments, not the original CaRL implementation.
- `examples/tiny/`: a synthetic input with a known answer.

The public C++ namespace is `whatif`; shared execution primitives use `whatif::core`.
Global causal scopes use Boost.Multiprecision; local cube masks remain bounded-width.
The construction methods use the `METHOD` argument of `whatif_build`; online methods use the entry points below. Snapshot format identifiers are unchanged so previously generated indexes remain readable.

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

The test suite includes the index tests, shared-executor regression tests, and three causal-workload fixtures. Test-only reference code can be excluded with `-DBUILD_TESTING=OFF`.

The test suite includes 14 CTest checks and 10 SQLite tests. Tests compare all seven construction methods with direct joins, including cyclic motifs, cross-tree queries and empty populations. Additional tests cover global prefix sharing, snapshot references to shared cuboids, all four residual join partitions, donor deduplication, more than 64 global variables, and induced-subgraph sampling. The existing Boost.Multiprecision compiler warning does not prevent compilation or these tests from passing.

## Quick start

Build a snapshot from the synthetic materialized matches:

```sh
mkdir -p results
./build/whatif_build \
  examples/tiny FaCT results/tiny.bin 6
```

Evaluate the supplied query:

```sh
./build/whatif_query \
  results/tiny.bin examples/tiny/causal.tsv FaCTQuery \
  < examples/tiny/query.json
```

The worker emits a readiness record followed by the query response. The example has four complete mappings and returns `P(Y=1 | do(X=1)) = 1`.

For **FullCubeQuery**, pass `FullCubeQuery` as the final argument to `whatif_query`, with a snapshot built by `FullCube` or `FullCube+`. For **OnlineJoin**, run `./build/whatif_online_join examples/tiny < examples/tiny/query.json`; it joins the supplied pattern matches without a cube index.

For example, build and query the full-cube baseline:

```sh
./build/whatif_build examples/tiny FullCube results/full.bin 6
./build/whatif_query results/full.bin examples/tiny/causal.tsv FullCubeQuery < examples/tiny/query.json
```

Run **CaRL-PG**, the SQLite baseline:

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
DATASET_DIR METHOD SNAPSHOT [WIDTH_BOUND=6] [INSTANCE_CAP=0] [SCOPE_ORDER_SEED=20260716]
```

| METHOD (paper name) | Construction policy |
| --- | --- |
| `FullCube` | Full cube, independently joining each scope |
| `FullCube+` | Full cube with rollup projection and residual-join completion |
| `FaCT` | Proposed construction with ordering, reuse, and adjustment-set pruning |
| `FaCT-O` | Random variable and construction orders, with reuse and pruning retained |
| `FaCT-P` | Proposed ordering and reuse, without adjustment-set pruning |
| `FaCT-R` | Proposed ordering and pruning, with scopes joined independently |
| `FaCT-ORP` | Same decomposition and boundary workload, with fixed alphabetical order, independent scope joins, and no adjustment-set scope pruning |

Method names are case-sensitive; old command-line aliases are not accepted. All FaCT variants use the same Parent Trees and boundary workload. `FaCT-P` uses FaCT's ordering reference scopes but materializes all local subsets; it disables adjustment-set scope pruning, not boundary-view redundancy elimination. `FaCT-R` and `FaCT-ORP` disable both prefix reuse and recover-and-complete.

FaCT deduplicates target scopes across Parent Trees, chooses a global variable order using the paper's estimated prefix-mass score, and traverses their shared sparse trie. A variable ID identifies one fixed pattern, output expression, and mapping schema; different pattern definitions must use different IDs. Shared cuboids have one physical owner and remain accessible from every containing Parent Tree after saving and loading.

Recover-and-complete projects a cached superset's full mappings, deduplicates them by match identity, and executes residual joins for all four input partitions, including missing pairs within the two participating input sets. It is selected only when its estimated extraction, deduplication and completion cost is below a direct join. Donor payloads have a 512 MiB accounted cache budget by default; active trie paths and final outputs are outside that budget. The library setting `tree_executor.rollup_cache_bytes` controls this budget. Construction logs report `recovery_used`, `recovered_rows`, `donor_peak_bytes`, and the chosen variable order.

Run the construction methods on the same input:

```sh
for method in FullCube 'FullCube+' FaCT FaCT-O FaCT-P FaCT-R FaCT-ORP; do
  ./build/whatif_build examples/tiny "$method" "results/$method.bin" 6 0 20260716
done
```

`WIDTH_BOUND` is the paper's width bound `b`; it controls planner merging and does not split an oversized atomic parent scope. `INSTANCE_CAP=0` applies no additional instance cap. Construction reports loading, planning, construction and snapshot metrics separately. `build_join_count` counts calls, whereas `join_work_pairs` counts compatible physical pairs before output compaction, excluding recovered results; these are different measurements.

### Induced-subgraph scale inputs

Scale experiments select vertices from the original PG, not rows independently from each variable. Supply the complete original vertex universe (including vertices absent from all pattern matches) as a headerless file with one canonical node ID per line. For fixed positive patterns with all matched vertices retained in each mapping, restriction to a selected vertex set is exactly equivalent to rematching those patterns on the induced subgraph:

```sh
python3 scripts/prepare_induced_scale.py DATASET_DIR VERTICES.tsv SCALE_OUTPUT \
  --percent 20 40 60 80 100 --seed 20260716 --positive-patterns
./build/whatif_build SCALE_OUTPUT/20pct FaCT results/scale20.bin 6
```

The script writes nested vertex samples, filtered matches, unchanged causal/schema files, and per-scale manifests. It refuses to overwrite an existing destination. Variables with zero surviving matches remain in `variable_manifest.csv` and in the causal DAG. An `INCOMPLETE` marker indicates an interrupted preparation and must not be benchmarked. Use the same query file and scale input directory for every method. The script keeps a vertex ranking in memory, so its memory requirement is linear in the original vertex count.

For negation, shortest-path, degree, or graph-dependent aggregation patterns, regenerate matches on each induced PG upstream instead of passing `--positive-patterns`. The old row-subsampling environment option is rejected unless its value is `100`. `INSTANCE_CAP` is a debug cap and must remain zero for formal scale experiments.

## Release scope and reproducibility

This source release includes the construction/query core and synthetic tests. It does not include the raw benchmark datasets, materialized benchmark inputs, prebuilt indexes, cDAG-discovery/LLM pipeline, Neo4j deployment scripts, or server-specific experiment queues. Full paper-scale reproduction requires those inputs and the exact workload/protocol; the quick start is a functional example, not a reproduction of the reported timings.

Compare methods on identical match populations and queries. Record SQLite settings, warm-up, repetitions, concurrency, timing boundaries, and failures. Database loading/index creation is separate from online query execution. Do not silently discard unsupported or timed-out queries when reporting means.

## Publication and anonymity

This repository is hosted under a personal GitHub account and is therefore not an anonymous review artifact. A separate anonymized distribution is needed for double-blind review. No source-server paths or private deployment configuration are required by the released commands.

No project-wide license is declared in this initial release; licensing remains to be selected by the repository owner. Preserve any third-party attribution already present in source files.
