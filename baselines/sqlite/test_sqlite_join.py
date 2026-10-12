import collections
import csv
import hashlib
import itertools
import tempfile
from pathlib import Path
import unittest

from sqlite_join import SQLiteJoin, prepare


class SQLiteJoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.roles = {'X': ['A', 'B'], 'Y': ['B', 'C'], 'Z': ['C'], 'W': ['C', 'D']}
        with (self.root / 'variable_manifest.csv').open('w') as f:
            w = csv.writer(f); w.writerow(['variable_id', 'roles'])
            w.writerows((v, '|'.join(r)) for v, r in self.roles.items())
        self.rows = {
            'X': [({'A':'1','B':'2'}, 'x'), ({'A':'3','B':'2'}, 'x'), ({'A':'4','B':'2'}, 'other')],
            'Y': [({'B':'2','C':'5'}, 'y'), ({'B':'2','C':'6'}, 'y')],
            'Z': [({'C':'5'}, 'z1'), ({'C':'6'}, 'z2')],
            'W': [({'C':'5','D':'1'}, 'w'), ({'C':'5','D':'7'}, 'w')]}
        lines = ['|'.join(k+'='+v for k,v in mapping.items())+'\t'+var+'\t'+value+'\n'
                 for var, rows in self.rows.items() for mapping,value in rows]
        (self.root/'assignments_raw.tsv').write_text(''.join(lines))
        digest=hashlib.sha256((self.root/'assignments_raw.tsv').read_bytes()).hexdigest()
        prepare(self.root,self.root/'test.sqlite',digest)
        self.engine=SQLiteJoin(self.root/'test.sqlite')
        self.q=dict(X='X',Y='Y',Z=['Z'],x='x',y='y',match_scope=['X','Y','Z','W'])

    def tearDown(self):
        self.engine.close(); self.temp.cleanup()

    def oracle(self,q):
        scope=[q['X'],q['Y'],*q['Z']]
        scope += [v for v in q.get('match_scope',scope) if v not in scope]
        distinct={}
        for rows in itertools.product(*(self.rows[v] for v in scope)):
            mapping={}; valid=True
            for binding,value in rows:
                for role,node in binding.items():
                    if role in mapping and mapping[role]!=node:
                        valid=False
                    mapping[role]=node
            if valid and len(set(mapping.values()))==len(mapping):
                distinct[tuple(sorted(mapping.items()))]=tuple(value for _,value in rows[:2+len(q['Z'])])
        return sorted([[list(key),n] for key,n in collections.Counter(distinct.values()).items()])

    def test_full_mapping_and_bridge(self):
        r=self.engine.query(self.q,True)
        self.assertEqual(r['joint'],self.oracle(self.q))
        self.assertEqual(r['N'],5)
        self.assertEqual(r['answer'],1.0)

    def test_default_connection_settings_and_counts(self):
        import sqlite3
        engine=SQLiteJoin(self.root/'test.sqlite', settings='default')
        reference=sqlite3.connect(self.root/'test.sqlite')
        try:
            for pragma in ('cache_size','temp_store','threads','mmap_size','automatic_index'):
                self.assertEqual(engine.db.execute('PRAGMA '+pragma).fetchone(),
                                 reference.execute('PRAGMA '+pragma).fetchone())
            self.assertEqual(engine.query(self.q,True)['joint'],self.oracle(self.q))
        finally:
            engine.close()
            reference.close()

    def test_identical_values_keep_distinct_mappings(self):
        q={**self.q,'match_scope':['X','Y','Z']}
        r=self.engine.query(q,True)
        self.assertEqual(r['joint'],self.oracle(q))
        self.assertEqual(r['N'],6)

    def test_default_without_statement_cache(self):
        import sqlite3
        from unittest.mock import patch
        with patch('sqlite_join.sqlite3.connect', wraps=sqlite3.connect) as factory:
            engine=SQLiteJoin(self.root/'test.sqlite', settings='default_no_statement_cache')
        reference=sqlite3.connect(self.root/'test.sqlite', cached_statements=0)
        try:
            self.assertEqual(factory.call_args.kwargs['cached_statements'],0)
            for pragma in ('cache_size','temp_store','threads','mmap_size','automatic_index'):
                self.assertEqual(engine.db.execute('PRAGMA '+pragma).fetchone(),
                                 reference.execute('PRAGMA '+pragma).fetchone())
            self.assertEqual(engine.query(self.q,True)['joint'],self.oracle(self.q))
        finally:
            engine.close()
            reference.close()

    def test_insufficient_support(self):
        r=self.engine.query({**self.q,'x':'missing'})
        self.assertEqual(r['status'],'insufficient_support')
        self.assertIsNone(r['answer'])

    def test_invalid_scope(self):
        with self.assertRaises(ValueError):
            self.engine.query({**self.q,'match_scope':['X','Y']})

    def test_no_overwrite(self):
        with self.assertRaises(ValueError):
            prepare(self.root,self.root/'test.sqlite','invalid')

    def test_no_treatment_prefilter(self):
        r=self.engine.query(self.q)
        self.assertEqual(r['strata'],[[['z1'],5,3,3]])

    def test_cross_variable_injectivity(self):
        # W binds D=1, so the X mapping with A=1 must be excluded.
        self.engine.db.close()
        import sqlite3
        db=sqlite3.connect(self.root/'test.sqlite')
        table=self.engine.meta['tables']['W']
        dcol=self.engine.meta['columns']['D']
        db.execute('DELETE FROM "'+table+'" WHERE "'+dcol+'"=?',('7',))
        db.commit();db.close()
        self.engine=SQLiteJoin(self.root/'test.sqlite')
        self.assertEqual(self.engine.query(self.q)['N'],2)

    def test_empty_population(self):
        self.engine.db.close()
        import sqlite3
        db=sqlite3.connect(self.root/'test.sqlite')
        db.execute('DELETE FROM "'+self.engine.meta['tables']['W']+'"')
        db.commit();db.close()
        self.engine=SQLiteJoin(self.root/'test.sqlite')
        r=self.engine.query(self.q)
        self.assertEqual((r['N'],r['status'],r['answer']),(0,'insufficient_support',None))


if __name__=='__main__':
    unittest.main()
