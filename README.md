# Whatif-Index

C++17 implementation of FaCT for what-if queries over property-graph pattern matches.

## Installation

Requires Linux, CMake 3.16+, a C++17 compiler, Boost headers, and Python 3 with SQLite.

```sh
sudo apt-get install build-essential cmake libboost-dev python3
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s baselines/sqlite -p 'test_*.py' -v
```

## Datasets

We use eight graph datasets. Click each dataset name to access its source or generator. The sizes below refer to the property graphs used in our experiments.

| Dataset | Nodes | Edges |
| --- | ---: | ---: |
| [Star Wars](https://github.com/neo4j-graph-examples/star-wars) | 260 | 611 |
| [WWC2019](https://github.com/neo4j-graph-examples/wwc2019) | 2,486 | 16,174 |
| [StackOverflow](https://github.com/neo4j-graph-examples/stackoverflow) | 6,193 | 11,540 |
| [FinCEN](https://github.com/neo4j-graph-examples/fincen) | 8,608 | 77,253 |
| [ICIJ OffshoreLeaks](https://offshoreleaks.icij.org/pages/database) | 1,908,466 | 3,124,710 |
| [DBpedia](https://www.dbpedia.org/resources/knowledge-graphs/) | 8,970,120 | 67,526,281 |
| [Freebase](https://developers.google.com/freebase) | 29,119,948 | 111,773,402 |
| [LDBC-SNB SF50](https://ldbcouncil.org/benchmarks/snb/) | 158,913,479 | 1,022,901,700 |

## Data format

Each dataset directory contains materialized per-variable pattern matches and a causal DAG:

| File | Format |
| --- | --- |
| `assignments_raw.tsv` | No header; three tab-separated fields: `role1=node1\|role2=node2`, variable ID, value |
| `causal.tsv` | No header; variable ID and comma-separated parents, separated by a tab; roots have an empty parent field |
| `variable_manifest.csv` | Header `variable_id,roles`; roles separated by `\|`; required for SQLite and scale preparation, and to retain variables with no matches |

One variable ID identifies one fixed pattern and value expression. Each assignment supplies its complete node-role mapping. Shared roles join by node identity, and distinct roles must map to distinct nodes. Include required connector patterns in the input and query scope. IDs and values must not contain field separators.

SQLite requires a unique node-role tuple per variable with a deterministic value. This input format does not encode parallel-edge identities.

Queries use one JSON object per line:

```json
{"X":"X","Y":"Y","Z":[],"x":"1","y":"1","match_scope":["X","Y"]}
```

`X` and `Y` are treatment and outcome variables; `Z` is a valid adjustment set; `x` and `y` are their requested values. `match_scope` contains all variables and connectors needed for the joint match. Add `"include_counts":true` to return validation counts. Empty populations or missing treatment support return `insufficient_support`.

## Running

### Index construction

```text
./build/whatif_build DATASET_DIR METHOD SNAPSHOT [WIDTH_BOUND=6] [SEED=20260716]
```

`WIDTH_BOUND` is the paper's width bound `b`; `SEED` controls the random orders in `FaCT-O`.

| METHOD | Construction |
| --- | --- |
| `FullCube` | Independently join every scope of the full cube |
| `FullCube+` | Full cube with rollup and residual-join completion |
| `FaCT` | Proposed ordering, reuse, and adjustment-set pruning |
| `FaCT-O` | Random ordering |
| `FaCT-R` | No prefix reuse or recover-and-complete |
| `FaCT-P` | No adjustment-set pruning |
| `FaCT-ORP` | Alphabetical ordering, no reuse, and no adjustment-set pruning |

All FaCT variants use the same decomposition. To run all methods on the example:

```sh
mkdir -p results
for method in FullCube 'FullCube+' FaCT FaCT-O FaCT-R FaCT-P FaCT-ORP; do
  ./build/whatif_build examples/tiny "$method" "results/$method.bin" 6 20260716
done
```

### Online queries

```sh
# FaCTQuery
./build/whatif_query results/FaCT.bin examples/tiny/causal.tsv FaCTQuery < examples/tiny/query.json

# FullCubeQuery (also accepts a FullCube+ snapshot)
./build/whatif_query results/FullCube.bin examples/tiny/causal.tsv FullCubeQuery < examples/tiny/query.json

# OnlineJoin
./build/whatif_online_join examples/tiny < examples/tiny/query.json

# CaRL-PG (SQLite)
python3 baselines/sqlite/run.py prepare examples/tiny results/tiny.sqlite
python3 baselines/sqlite/run.py query results/tiny.sqlite examples/tiny/query.json
```

The C++ workers emit a readiness record followed by query responses. The example returns `P(Y=1 | do(X=1)) = 1`. SQLite takes one query file and refuses to overwrite an existing database.

### Scale inputs

Provide all original PG node IDs in `VERTICES.tsv`, one per line without a header:

```sh
python3 scripts/prepare_induced_scale.py DATASET_DIR VERTICES.tsv SCALE_OUTPUT \
  --percent 20 40 60 80 100 --seed 20260716 --positive-patterns
./build/whatif_build SCALE_OUTPUT/20pct FaCT results/scale20.bin 6
```

This creates nested vertex-induced samples by retaining complete matches whose nodes are all selected. It applies to fixed positive patterns; rematch the induced PG upstream for negation, shortest paths, degree predicates, or graph-dependent aggregates. Use a new output directory; an `INCOMPLETE` marker means preparation did not finish.
