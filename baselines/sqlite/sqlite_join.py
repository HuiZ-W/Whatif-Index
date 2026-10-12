"""Relational joins over per-variable sampled object bindings, never values."""
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import time

from sampled_query import finish


def ident(value):
    return '"' + value.replace('"', '""') + '"'


def connect(path, readonly=False, settings='configured'):
    if settings not in ('configured', 'default', 'default_no_statement_cache'):
        raise ValueError('Unknown SQLite settings')
    if settings in ('default', 'default_no_statement_cache'):
        options = {'cached_statements': 0} if settings == 'default_no_statement_cache' else {}
        return sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=' + ('ro' if readonly else 'rwc'), uri=True, **options)
    db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=' + ('ro' if readonly else 'rwc'),
                         uri=True, cached_statements=0)
    db.execute('PRAGMA threads=0')
    db.execute('PRAGMA cache_size=-262144')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA mmap_size=0')
    return db


def schema(source):
    with (Path(source) / 'variable_manifest.csv').open(newline='') as f:
        specs = {r['variable_id']: r for r in csv.DictReader(f)}
    roles = sorted({role for s in specs.values() for role in s['roles'].split('|')})
    columns = {r: 'r' + str(i) for i, r in enumerate(roles)}
    tables = {v: 'matches_' + str(i) for i, v in enumerate(sorted(specs))}
    return specs, columns, tables


def prepare(source, path, expected_sha, progress=None):
    source, path = Path(source), Path(path)
    if path.exists():
        raise ValueError('Refusing to overwrite database')
    start = time.perf_counter()
    specs, columns, tables = schema(source)
    db = connect(path)
    try:
        for v, spec in specs.items():
            cols = [columns[r] for r in spec['roles'].split('|')]
            db.execute('CREATE TABLE ' + ident(tables[v]) + ' (' +
                       ','.join(ident(c) + ' TEXT NOT NULL' for c in cols) +
                       ',value TEXT NOT NULL, PRIMARY KEY (' + ','.join(map(ident, cols)) + ')) WITHOUT ROWID')
        db.commit()
        schema_ms = (time.perf_counter() - start) * 1000
        t = time.perf_counter()
        digest = hashlib.sha256()
        batches = collections.defaultdict(list)
        counts = collections.Counter()
        def flush():
            for v, rows in batches.items():
                if rows:
                    db.executemany('INSERT INTO ' + ident(tables[v]) + ' VALUES (' +
                                   ','.join('?' for _ in rows[0]) + ')', rows)
                    rows.clear()
        with (source / 'assignments_raw.tsv').open('rb') as f:
            for raw in f:
                digest.update(raw)
                line = raw.decode().rstrip('\r\n')
                if not line or line.startswith('#'):
                    continue
                binding, v, value = line.split('\t')
                pairs = [p.split('=', 1) for p in binding.split('|')]
                mapping = dict(pairs)
                roles = specs[v]['roles'].split('|')
                if len(mapping) != len(pairs) or set(mapping) != set(roles) or len(set(mapping.values())) != len(mapping):
                    raise ValueError('Invalid role binding: ' + v)
                batches[v].append([mapping[r] for r in roles] + [value])
                counts[v] += 1
                if sum(map(len, batches.values())) >= 10000:
                    flush()
                    if progress:
                        progress(sum(counts.values()))
        flush()
        if digest.hexdigest() != expected_sha:
            raise ValueError('Population digest mismatch')
        db.commit()
        import_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        indexes = []
        for v, spec in specs.items():
            own = set(spec['roles'].split('|'))
            keys = {tuple([r]) for r in own}
            keys |= {tuple(sorted(own & set(other['roles'].split('|')))) for other in specs.values()}
            for i, key in enumerate(sorted(keys)):
                if not key:
                    continue
                name = 'idx_' + tables[v] + '_' + str(i)
                sql = 'CREATE INDEX ' + ident(name) + ' ON ' + ident(tables[v]) + '(' + ','.join(ident(columns[r]) for r in key) + ')'
                db.execute(sql)
                indexes.append(dict(name=name, variable=v, roles=key, sql=sql))
        db.commit()
        index_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        db.execute('ANALYZE')
        db.commit()
        analyze_ms = (time.perf_counter() - t) * 1000
        metadata = dict(source=str(source), assignment_sha256=expected_sha,
                        records=sum(counts.values()), counts=dict(counts), specs=specs,
                        columns=columns, tables=tables, indexes=indexes,
                        schema_ms=schema_ms, import_ms=import_ms, index_ms=index_ms,
                        analyze_ms=analyze_ms, sqlite_version=sqlite3.sqlite_version,
                        raw_edge_identity_available=False, population='sampled_vertex_binding_union',
                        pragmas={p: db.execute('PRAGMA ' + p).fetchone()[0]
                                 for p in ['threads', 'cache_size', 'temp_store', 'mmap_size', 'automatic_index']},
                        compile_options=[r[0] for r in db.execute('PRAGMA compile_options')])
        db.execute('CREATE TABLE metadata (payload TEXT NOT NULL)')
        db.execute('INSERT INTO metadata VALUES (?)', (json.dumps(metadata),))
        db.commit()
    finally:
        db.close()
    metadata['offline_total_ms'] = (time.perf_counter() - start) * 1000
    metadata['database_bytes'] = path.stat().st_size
    return metadata


class SQLiteJoin:
    def __init__(self, path, query_assignment_sha=None, settings='configured'):
        self.db = connect(path, readonly=True, settings=settings)
        self.meta = json.loads(self.db.execute('SELECT payload FROM metadata').fetchone()[0])
        self.query_assignment_sha = query_assignment_sha or self.meta['assignment_sha256']

    def sql(self, q):
        causal = [q['X'], q['Y'], *q['Z']]
        full = q.get('match_scope', causal)
        if len(set(causal)) != len(causal) or len(set(full)) != len(full) or not set(causal) <= set(full):
            raise ValueError('Invalid full scope')
        if q.get('assignment_sha256', self.query_assignment_sha) != self.query_assignment_sha:
            raise ValueError('Query population mismatch')
        scope = causal + [v for v in full if v not in causal]
        occurrences, conditions, joins = {}, [], []
        for i, v in enumerate(scope):
            alias = 'a' + str(i)
            shared = []
            for role in self.meta['specs'][v]['roles'].split('|'):
                ref = alias + '.' + ident(self.meta['columns'][role])
                if role in occurrences:
                    shared.append(ref + '=' + occurrences[role])
                else:
                    occurrences[role] = ref
            table = ident(self.meta['tables'][v]) + ' AS ' + alias
            joins.append(table if i == 0 else 'JOIN ' + table + ' ON ' + (' AND '.join(shared) if shared else '1=1'))
        # Existing OnlineJoin enforces injectivity across every distinct role.
        conditions = [a + '<>' + b for a, b in itertools.combinations(occurrences.values(), 2)]
        fields = [ref + ' AS b' + str(i) for i, ref in enumerate(occurrences.values())]
        fields += ['a' + str(i) + '.value AS q' + str(i) for i in range(len(scope))]
        inner = 'SELECT DISTINCT ' + ','.join(fields) + ' FROM ' + ' '.join(joins)
        if conditions:
            inner += ' WHERE ' + ' AND '.join(conditions)
        values = ','.join('q' + str(i) for i in range(len(causal)))
        return 'SELECT ' + values + ',COUNT(*) AS frequency FROM (' + inner + ') GROUP BY ' + values

    def query(self, q, include_joint=False):
        start = time.perf_counter()
        sql = self.sql(q)
        generation_ms = (time.perf_counter() - start) * 1000
        counts = collections.defaultdict(lambda: [0, 0, 0])
        joint = []
        sql_start = time.perf_counter()
        cursor = self.db.execute(sql)
        sql_read_ms = (time.perf_counter() - sql_start) * 1000
        while True:
            t = time.perf_counter()
            batch = cursor.fetchmany(4096)
            sql_read_ms += (time.perf_counter() - t) * 1000
            if not batch:
                break
            for row in batch:
                x, y, *tail = row
                z, frequency = tuple(tail[:-1]), tail[-1]
                c = counts[z]
                c[0] += frequency
                if x == q['x']:
                    c[1] += frequency
                    if y == q['y']:
                        c[2] += frequency
                if include_joint:
                    joint.append([list(row[:-1]), frequency])
        result = finish([[list(z), *c] for z, c in counts.items()])
        result.update(latency_ms=(time.perf_counter() - start) * 1000,
                      sql_generation_ms=generation_ms, sql_execute_read_ms=sql_read_ms)
        if include_joint:
            result['joint'] = sorted(joint)
        return result

    def close(self):
        self.db.close()
