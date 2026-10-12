import importlib.util
import csv
import itertools
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('induced_scale', Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_induced_scale.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InducedScaleTests(unittest.TestCase):
    def test_matches_equal_rematching_induced_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'; source.mkdir()
            nodes = [str(i) for i in range(10)]
            vertices = root / 'vertices.tsv'; vertices.write_text('\n'.join(nodes)+'\n')
            edges = {(str(a), str(b)) for a in range(6) for b in range(6) if a != b and (a+b)%3 != 0}
            def matches(selected):
                lines = set()
                for a,b,c in itertools.permutations(sorted(selected),3):
                    if (a,b) in edges and (b,c) in edges and (c,a) in edges:
                        lines.add(f'a={a}|b={b}|c={c}\tTriangle\t1\n')
                return lines
            (source / 'assignments_raw.tsv').write_text(''.join(sorted(matches(set(nodes)))))
            (source / 'causal.tsv').write_text('Triangle\t\nEmpty\tTriangle\n')
            (source / 'variable_manifest.csv').write_text('variable_id,roles\nTriangle,a|b|c\nEmpty,a\n')
            destination = root / 'scales'
            counts = module.prepare(source,vertices,destination,[20,40,60,80,100],123,True)
            previous = set()
            for p in [20,40,60,80,100]:
                folder = destination / f'{p}pct'
                selected = set((folder/'selected_vertices.tsv').read_text().splitlines())
                self.assertEqual(len(selected), len(nodes)*p//100)
                self.assertTrue(previous <= selected)
                self.assertEqual(set((folder/'assignments_raw.tsv').read_text().splitlines(keepends=True)), matches(selected))
                self.assertEqual(counts[p]['Empty'],0)
                self.assertEqual((folder/'causal.tsv').read_bytes(),(source/'causal.tsv').read_bytes())
                self.assertEqual((folder/'variable_manifest.csv').read_bytes(),(source/'variable_manifest.csv').read_bytes())
                previous=selected
            self.assertFalse((destination/'INCOMPLETE').exists())
            with self.assertRaises(ValueError): module.prepare(source,vertices,destination,[100],123,True)
            with self.assertRaises(ValueError): module.prepare(source,vertices,root/'unchecked',[100],123)


if __name__ == '__main__': unittest.main()
