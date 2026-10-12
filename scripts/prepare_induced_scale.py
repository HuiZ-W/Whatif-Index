"""Prepare nested induced-PG match populations for fixed positive graph patterns.

For a positive pattern, restricting its complete matches to selected vertices is
exactly matching that pattern on the induced subgraph. This does NOT hold for
negation, shortest paths, degree predicates, or graph-dependent aggregates.
"""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import shutil


def prepare(source, vertices, destination, percentages, seed, positive_patterns=False):
    source, vertices, destination = map(Path, (source, vertices, destination))
    if not positive_patterns:
        raise ValueError('Confirm fixed positive patterns with --positive-patterns; otherwise rematch each induced PG upstream')
    if destination.exists():
        raise ValueError('Refusing to overwrite destination')
    percentages = sorted(set(percentages))
    if not percentages or any(p < 1 or p > 100 for p in percentages):
        raise ValueError('Percentages must be integers in [1,100]')
    universe = set()
    with vertices.open() as stream:
        for line in stream:
            node = line.rstrip('\r\n')
            if not node or '\t' in node or node in universe:
                raise ValueError('vertices file must contain one unique canonical node ID per line, without a header')
            universe.add(node)
    if not universe:
        raise ValueError('Empty vertex universe')
    # A stable pseudorandom permutation makes the samples nested and independent
    # of input-file ordering. The supplied universe includes unmatched PG nodes.
    def key(node):
        return hashlib.sha256((str(seed) + '\0' + node).encode()).digest(), node
    ordered = sorted(universe, key=key)
    ranks = {node: rank for rank, node in enumerate(ordered)}
    with (source / 'variable_manifest.csv').open(newline='') as stream:
        specs = {r['variable_id']: set(r['roles'].split('|')) for r in csv.DictReader(stream)}
    if not specs:
        raise ValueError('Missing variable schemas')
    destination.mkdir(parents=True)
    marker = destination / 'INCOMPLETE'
    marker.write_text('Preparation has not completed. Do not benchmark these inputs.\n')
    cutoffs = {p: len(ordered) * p // 100 for p in percentages}
    counts = {p: {v: 0 for v in specs} for p in percentages}
    source_digest = hashlib.sha256()
    with contextlib.ExitStack() as stack:
        outputs = {}
        for p in percentages:
            folder = destination / f'{p}pct'
            folder.mkdir()
            shutil.copyfile(source / 'causal.tsv', folder / 'causal.tsv')
            shutil.copyfile(source / 'variable_manifest.csv', folder / 'variable_manifest.csv')
            if (source / 'patterns.tsv').exists():
                shutil.copyfile(source / 'patterns.tsv', folder / 'patterns.tsv')
            with (folder / 'selected_vertices.tsv').open('w') as stream:
                stream.writelines(node + '\n' for node in ordered[:cutoffs[p]])
            outputs[p] = stack.enter_context((folder / 'assignments_raw.tsv').open('wb'))
        with (source / 'assignments_raw.tsv').open('rb') as stream:
            for raw in stream:
                source_digest.update(raw)
                line = raw.decode().rstrip('\r\n')
                if not line or line.startswith('#'):
                    continue
                binding, variable, value = line.split('\t')
                pairs = [s.split('=', 1) for s in binding.split('|')]
                mapping = dict(pairs)
                if variable not in specs or len(mapping) != len(pairs) or set(mapping) != specs[variable]:
                    raise ValueError('Pattern mapping does not match its declared schema')
                if len(set(mapping.values())) != len(mapping):
                    raise ValueError('Non-injective input mapping')
                if not mapping or any(node not in ranks for node in mapping.values()):
                    raise ValueError('The vertex universe must include every matched vertex')
                last = max(ranks[node] for node in mapping.values())
                for p in percentages:
                    if last < cutoffs[p]:
                        outputs[p].write(raw)
                        counts[p][variable] += 1
    for p in percentages:
        manifest = {'sampling': 'induced_vertices_positive_patterns', 'percent': p,
                    'seed': seed, 'total_vertices': len(ordered), 'selected_vertices': cutoffs[p],
                    'variable_rows': counts[p], 'source_assignments_sha256': source_digest.hexdigest()}
        (destination / f'{p}pct' / 'scale_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    marker.unlink()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('vertices', type=Path, help='All original PG node IDs, one per line; no header')
    parser.add_argument('destination', type=Path)
    parser.add_argument('--percent', type=int, nargs='+', default=[20, 40, 60, 80, 100])
    parser.add_argument('--seed', type=int, default=20260716)
    parser.add_argument('--positive-patterns', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.vertices, args.destination, args.percent,
                             args.seed, args.positive_patterns), indent=2))


if __name__ == '__main__':
    main()
