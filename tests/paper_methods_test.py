"""Compare every public construction method with direct joins on cyclic motifs."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

METHODS = ['FullCube', 'FullCube+', 'FaCT', 'FaCT-O', 'FaCT-P', 'FaCT-R', 'FaCT-ORP']


def run(command, requests=None):
    p = subprocess.run(list(map(str, command)), input=requests, text=True, capture_output=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return p.stdout


def normalize(record):
    strata = sorted((tuple(s['z']) if s['z'] else (), int(s['nz']), int(s['nxz']), int(s['nxyz']))
                    for s in record['strata'])
    return (record['status'], int(record['N']), float(record.get('answer', 0)),
            strata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='whatif-paper-methods-') as directory:
        root=Path(directory)
        data=root/'data'; data.mkdir()
        specs={'X':['a','b'], 'Y':['b','c'], 'Z':['a','c'], 'W':['c','d'], 'V':['e','a']}
        (data/'variable_manifest.csv').write_text('variable_id,roles\n'+''.join(f'{v},{"|".join(r)}\n' for v,r in specs.items()))
        (data/'causal.tsv').write_text('V\t\nW\tV\nZ\tW\nX\tZ\nY\tX,Z\n')
        rows=[]
        for v,roles in specs.items():
            for a,b in itertools.product(range(3),repeat=2):
                if v=='Z' and (a+b)%2: continue
                rows.append(f'{roles[0]}={roles[0]}{a}|{roles[1]}={roles[1]}{b}\t{v}\t{(a+b)%2}\n')
        (data/'assignments_raw.tsv').write_text(''.join(rows))
        queries=[{'X':'X','Y':'Y','Z':['Z'],'x':str(x),'y':str(y),'include_counts':True,'match_scope':scope}
                 for scope in [['X','Y','Z'],['X','Y','Z','W','V']] for x,y in itertools.product(range(2),repeat=2)]
        requests=''.join(json.dumps(q)+'\n' for q in queries)
        direct=[json.loads(s) for s in run([args.build_dir/'whatif_online_join',data],requests).splitlines()][1:]
        expected=list(map(normalize,direct))
        orders={}
        for method in METHODS:
            snapshot=root/(method+'.bin')
            output=run([args.build_dir/'whatif_build',data,method,snapshot,2])
            orders[method]=next(s for s in output.splitlines() if s.startswith('shared_variable_order'))
            mode='FullCubeQuery' if method.startswith('FullCube') else 'FaCTQuery'
            actual=[json.loads(s) for s in run([args.build_dir/'whatif_query',snapshot,data/'causal.tsv',mode],requests).splitlines()][1:]
            assert list(map(normalize,actual))==expected,(method,actual,direct)
            if method in ('FaCT-R','FaCT-ORP'): assert 'recovery_used=0' in output.splitlines(),output
        assert orders['FaCT']==orders['FaCT-P']==orders['FaCT-R'],orders
        assert orders['FaCT-ORP']=='shared_variable_order\tV\tW\tX\tY\tZ',orders
        # Empty match populations must not remove variables from the fixed DAG.
        (data/'assignments_raw.tsv').write_text(''.join(s for s in rows if '\tX\t' not in s))
        for method in ('FullCube','FaCT','FaCT-ORP'):
            snapshot=root/(method+'-empty.bin')
            run([args.build_dir/'whatif_build',data,method,snapshot,2])
            mode='FullCubeQuery' if method=='FullCube' else 'FaCTQuery'
            actual=[json.loads(s) for s in run([args.build_dir/'whatif_query',snapshot,data/'causal.tsv',mode],requests).splitlines()][1:]
            assert all(r['status']=='insufficient_support' and int(r['N'])==0 for r in actual),actual
        empty=[json.loads(s) for s in run([args.build_dir/'whatif_online_join',data],requests).splitlines()][1:]
        assert all(r['status']=='insufficient_support' and int(r['N'])==0 for r in empty),empty
    print('All seven methods match direct joins, including empty and cross-tree populations.')


if __name__=='__main__': main()
