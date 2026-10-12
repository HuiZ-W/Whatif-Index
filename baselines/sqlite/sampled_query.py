"""Sampled vertex-binding population reference and matching Cypher adapter.

No raw-graph or persisted cube equivalence is implied by this adapter.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import subprocess
import uuid


class SamplePopulation:
    def __init__(self, source, database, validator, progress=None):
        self.source = Path(source)
        self.validator = str(validator)
        if Path(database).exists():
            raise ValueError('Refusing to overwrite reference database')
        self.db = sqlite3.connect(database)
        with (self.source / 'variable_manifest.csv').open(newline='') as stream:
            self.specs = {r['variable_id']: r for r in csv.DictReader(stream)}
        self.roles = sorted({role for r in self.specs.values() for role in r['roles'].split('|')})
        self.columns = {role: 'r' + str(i) for i, role in enumerate(self.roles)}
        self.sample_id = uuid.uuid4().hex
        self.db.execute('CREATE TABLE records (_var TEXT, _value TEXT, _binding TEXT, ' + ','.join(c + ' TEXT' for c in self.columns.values()) + ', UNIQUE(_var,_binding))')
        digest = hashlib.sha256()
        count = 0
        try:
            with (self.source / 'assignments_raw.tsv').open('rb') as stream:
                for raw in stream:
                    digest.update(raw)
                    line = raw.decode('utf-8').rstrip('\r\n')
                    if not line or line.startswith('#'):
                        continue
                    binding, variable, value = line.split('\t')
                    pairs = [part.split('=', 1) for part in binding.split('|')]
                    roles = dict(pairs)
                    if len(roles) != len(pairs) or len(set(roles.values())) != len(roles):
                        raise ValueError('Non-injective or duplicated role binding')
                    if variable not in self.specs or set(roles) != set(self.specs[variable]['roles'].split('|')):
                        raise ValueError('Manifest/record role mismatch: ' + variable)
                    canonical = json.dumps(sorted(roles.items()), separators=(',', ':'))
                    values = [variable, value, canonical] + [roles.get(r) for r in self.roles]
                    try:
                        self.db.execute('INSERT INTO records VALUES (' + ','.join('?' for _ in values) + ')', values)
                    except sqlite3.IntegrityError as error:
                        raise ValueError('Duplicate binding or conflicting value: ' + variable + ' ' + canonical) from error
                    count += 1
                    if progress and count % 100000 == 0:
                        progress({'phase': 'assignment_audit', 'records_read': count})
            self.db.commit()
            for column in self.columns.values():
                self.db.execute('CREATE INDEX by_' + column + ' ON records(_var,' + column + ')')
        except BaseException:
            self.db.rollback()
            self.db.close()
            raise
        self.metadata = {'sample_id': self.sample_id, 'source': str(self.source.resolve()),
                         'assignment_sha256': digest.hexdigest(), 'records': count,
                         'population': 'sampled_vertex_binding_union', 'raw_edge_identity_available': False,
                         'index_population_equivalence': 'unverified',
                         'definition_sha256': {name: hashlib.sha256((self.source / name).read_bytes()).hexdigest()
                                               for name in ('causal.tsv', 'patterns.tsv', 'variable_manifest.csv')}}
        stat = (self.source / 'assignments_raw.tsv').stat()
        self.metadata['assignment_stat'] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        self.db.execute('CREATE TABLE population_metadata (payload TEXT NOT NULL)')
        self.db.execute('INSERT INTO population_metadata VALUES (?)', (json.dumps(self.metadata),))
        self.db.commit()

    @classmethod
    def open_existing(cls, database, validator):
        obj = cls.__new__(cls)
        obj.db = sqlite3.connect('file:' + str(Path(database).resolve()) + '?mode=ro', uri=True)
        try:
            obj.metadata = json.loads(obj.db.execute('SELECT payload FROM population_metadata').fetchone()[0])
            obj.source = Path(obj.metadata['source'])
            obj.validator = str(validator)
            for name, digest in obj.metadata['definition_sha256'].items():
                if hashlib.sha256((obj.source / name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Source definition changed: ' + name)
            stat = (obj.source / 'assignments_raw.tsv').stat()
            if obj.metadata['assignment_stat'] != {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
                raise ValueError('Source assignment changed')
            with (obj.source / 'variable_manifest.csv').open(newline='') as stream:
                obj.specs = {r['variable_id']: r for r in csv.DictReader(stream)}
            obj.roles = sorted({role for r in obj.specs.values() for role in r['roles'].split('|')})
            obj.columns = {role: 'r' + str(i) for i, role in enumerate(obj.roles)}
            obj.sample_id = obj.metadata['sample_id']
            return obj
        except BaseException:
            obj.db.close()
            raise

    def plan(self, query):
        x, y, z = query['X'], query['Y'], query['Z']
        scope = [x, y] + list(z)
        if 'match_scope' in query:
            full = query['match_scope']
            if len(set(full)) != len(full) or not set(scope) <= set(full):
                raise ValueError('Invalid complete match scope')
            scope.extend(v for v in full if v not in scope)
        if len(set(scope)) != len(scope) or not set(scope) <= self.specs.keys():
            raise ValueError('Invalid query variables')
        validation = subprocess.run([self.validator, str(self.source / 'causal.tsv'), '--check', x, y, *z],
                                    text=True, capture_output=True, timeout=30)
        if validation.returncode != 0 or validation.stdout.strip() != 'valid':
            raise ValueError('Adjustment set rejected by existing DAG validator')
        for variable, value in ((x, query['x']), (y, query['y'])):
            if not isinstance(value, str) or not self.db.execute('SELECT 1 FROM records WHERE _var=? AND _value=? LIMIT 1', (variable, value)).fetchone():
                raise ValueError('Value outside sampled domain: ' + variable)
        aliases = {v: 'a' + str(i) for i, v in enumerate(scope)}
        occurrences = {}
        for v in scope:
            for role in self.specs[v]['roles'].split('|'):
                occurrences.setdefault(role, []).append(aliases[v] + '.' + self.columns[role])
        equalities = [refs[0] + '=' + ref for refs in occurrences.values() for ref in refs[1:]]
        # Identical role schemas imply identical canonical binding keys. This
        # redundant equality enables the unique composite index without
        # dropping any vertex/role consistency checks.
        for left, right in itertools.combinations(scope, 2):
            if set(self.specs[left]['roles'].split('|')) == set(self.specs[right]['roles'].split('|')):
                equalities.append(aliases[left] + '._binding=' + aliases[right] + '._binding')
        equalities += [a[0] + '<>' + b[0] for a, b in itertools.combinations(occurrences.values(), 2)]
        return scope, aliases, occurrences, equalities

    def reference(self, query):
        scope, aliases, occurrences, conditions = self.plan(query)
        conditions += [aliases[v] + '._var=?' for v in scope]
        values = [aliases[v] + '._value AS q' + str(i) for i, v in enumerate(scope)]
        bindings = [refs[0] + ' AS b' + str(i) for i, refs in enumerate(occurrences.values())]
        inner = 'SELECT DISTINCT ' + ','.join(bindings + values) + ' FROM ' + ','.join('records ' + aliases[v] for v in scope) + ' WHERE ' + ' AND '.join(conditions)
        zcols = ['q' + str(i + 2) for i in range(len(query['Z']))]
        sql = 'SELECT ' + (','.join(zcols) + ',' if zcols else '') + 'count(*),sum(CASE WHEN q0=? THEN 1 ELSE 0 END),sum(CASE WHEN q0=? AND q1=? THEN 1 ELSE 0 END) FROM (' + inner + ')'
        if zcols:
            sql += ' GROUP BY ' + ','.join(zcols)
        rows = list(self.db.execute(sql, [query['x'], query['x'], query['y']] + scope))
        strata = [[list(row[:-3]), *row[-3:]] for row in rows if row[-3]]
        return finish(strata)

    def import_neo4j(self, client, batch_size=1000, progress=None):
        client.query('CREATE CONSTRAINT sample_record_identity IF NOT EXISTS FOR (n:SampleRecord) REQUIRE (n._sample,n._var,n._binding) IS UNIQUE')
        client.query('CREATE INDEX sample_record_lookup IF NOT EXISTS FOR (n:SampleRecord) ON (n._sample,n._var)')
        client.query('CREATE CONSTRAINT sample_population_identity IF NOT EXISTS FOR (n:SamplePopulation) REQUIRE n._sample IS UNIQUE')
        client.query('CREATE CONSTRAINT sample_record_ordinal IF NOT EXISTS FOR (n:SampleRecord) REQUIRE (n._sample,n._ordinal) IS UNIQUE')
        for column in self.columns.values():
            client.query(f'CREATE INDEX sample_role_{column} IF NOT EXISTS FOR (n:SampleRecord) ON (n._sample,n._var,n.{column})')
        client.query('CALL db.awaitIndexes(300)')
        ready = client.query('MATCH (n:SamplePopulation {_sample:$sample,ready:true}) RETURN n.records,n.sha256', {'sample': self.sample_id})
        if ready:
            if ready != [[self.metadata['records'], self.metadata['assignment_sha256']]]:
                raise ValueError('Imported population provenance mismatch')
            return
        count, maximum, ordinals = client.query('MATCH (n:SampleRecord {_sample:$sample}) RETURN count(n),coalesce(max(n._ordinal),0),count(n._ordinal)', {'sample': self.sample_id})[0]
        if count != maximum or count != ordinals or count > self.metadata['records']:
            raise ValueError('Cannot resume non-contiguous or legacy import')
        cursor = self.db.execute('SELECT rowid AS _ordinal,* FROM records WHERE rowid>? ORDER BY rowid', (count,))
        names = [item[0] for item in cursor.description]
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            rows = [{**{k: v for k, v in zip(names, row) if v is not None}, '_sample': self.sample_id} for row in batch]
            imported = client.query('UNWIND $rows AS row CREATE (n:SampleRecord) SET n=row RETURN count(n)', {'rows': rows})[0][0]
            if imported != len(rows):
                raise RuntimeError('Incomplete import batch')
            count += imported
            if progress:
                progress({'phase': 'neo4j_import', 'imported_records': count, 'total_records': self.metadata['records']})
        count = client.query('MATCH (n:SampleRecord {_sample:$sample}) RETURN count(n)', {'sample': self.sample_id})[0][0]
        if count != self.metadata['records']:
            raise RuntimeError('Import count mismatch')
        client.query('CREATE (n:SamplePopulation {_sample:$sample, ready:true, records:$records, sha256:$sha})', {'sample': self.sample_id, 'records': count, 'sha': self.metadata['assignment_sha256']})

    def cypher(self, query, include_counts=True):
        scope, aliases, occurrences, conditions = self.plan(query)
        matches = [f'({aliases[v]}:SampleRecord {{_sample:$sample,_var:$v{i}}})' for i, v in enumerate(scope)]
        values = [aliases[v] + '._value AS q' + str(i) for i, v in enumerate(scope)]
        bindings = [refs[0] + ' AS b' + str(i) for i, refs in enumerate(occurrences.values())]
        zcols = ['q' + str(i + 2) for i in range(len(query['Z']))]
        statement = 'MATCH ' + ','.join(matches)
        if conditions:
            statement += ' WHERE ' + ' AND '.join(conditions)
        statement += ' WITH DISTINCT ' + ','.join(bindings + values)
        statement += ' WITH [' + ','.join(zcols) + '] AS z, count(*) AS nz, sum(CASE WHEN q0=$x THEN 1 ELSE 0 END) AS nxz, sum(CASE WHEN q0=$x AND q1=$y THEN 1 ELSE 0 END) AS nxyz'
        statement += ' WITH collect({z:z,nz:nz,nxz:nxz,nxyz:nxyz}) AS strata, sum(nz) AS N RETURN N, CASE WHEN N=0 OR any(s IN strata WHERE s.nxz=0) THEN null ELSE reduce(p=0.0,s IN strata | p + toFloat(s.nz)/N * s.nxyz/s.nxz) END AS answer, strata'
        if not include_counts:
            statement = statement.removesuffix(', strata')
        params = {'sample': self.sample_id, 'x': query['x'], 'y': query['y'], **{'v' + str(i): v for i, v in enumerate(scope)}}
        return statement, params

    def evaluate_neo4j(self, client, query):
        ready = client.query('MATCH (n:SamplePopulation {_sample:$sample,ready:true}) RETURN n.records', {'sample': self.sample_id})
        if ready != [[self.metadata['records']]]:
            raise RuntimeError('Population import not ready')
        statement, params = self.cypher(query)
        rows = client.query(statement, params)
        if len(rows) != 1:
            raise RuntimeError('Unexpected aggregation result')
        n, answer, strata = rows[0]
        result = finish([[s['z'], s['nz'], s['nxz'], s['nxyz']] for s in strata if s['nz']])
        if n != result['N'] or (answer is None) != (result['answer'] is None):
            raise RuntimeError('Server adjustment result inconsistent')
        result['answer'] = answer
        return result


def finish(strata):
    strata = sorted(strata, key=lambda row: tuple(row[0]))
    n = sum(row[1] for row in strata)
    valid = bool(n) and all(row[2] > 0 for row in strata)
    return {'N': n, 'strata': strata, 'status': 'ok' if valid else 'insufficient_support',
            'answer': sum(nz / n * nxyz / nxz for _, nz, nxz, nxyz in strata) if valid else None}
