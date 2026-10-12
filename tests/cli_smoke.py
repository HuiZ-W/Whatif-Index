"""Check all documented construction policies through the public query interface."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


METHODS = (
    'FullCube', 'FullCube+', 'FaCT', 'FaCT-O', 'FaCT-P', 'FaCT-R',
)


def run(command, request=None):
    result = subprocess.run([str(x) for x in command], input=request,
                            text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed:\n{result.stdout}\n{result.stderr}')
    return result.stdout


def query(worker, snapshot, data, method, request):
    records = [json.loads(line) for line in run(
        [worker, snapshot, data / 'causal.tsv', method], request).splitlines()]
    assert records[0]['status'] == 'ready', records
    assert records[0]['method'] == method, records
    answer = records[-1]
    assert answer['status'] == 'ok', answer
    assert int(answer['N']) == 4, answer
    assert float(answer['answer']) == 1.0, answer
    return {key: answer[key] for key in ('status', 'N', 'answer', 'strata')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    data = Path(__file__).resolve().parents[1] / 'examples' / 'tiny'
    request = (data / 'query.json').read_text()
    worker = args.build_dir / 'whatif_query'
    expected = None
    with tempfile.TemporaryDirectory(prefix='whatif-cli-') as temporary:
        root = Path(temporary)
        for method in METHODS:
            mode = 'FullCubeQuery' if method.startswith('FullCube') else 'FaCTQuery'
            snapshot = root / (method + '.bin')
            output = run([args.build_dir / 'whatif_build', data, method, snapshot, '6'])
            assert f'method={method}' in output.splitlines(), output
            actual = query(worker, snapshot, data, mode, request)
            if expected is None:
                expected = actual
            assert actual == expected, (method, actual, expected)
            print(f'{method}: counts and probability match')
        # Old names must fail, rather than silently selecting another policy.
        for method in ('full-join', 'boundary-adjustment-prune', 'parent-no-reuse',
                       'FaCT-ORP', 'fact', 'unknown'):
            target = root / 'rejected.bin'
            result = subprocess.run(
                [str(args.build_dir / 'whatif_build'), str(data), method, str(target)],
                text=True, capture_output=True, timeout=120)
            assert result.returncode != 0 and 'unknown construction method' in result.stderr, result
            assert not target.exists(), method
        for method in ('FullCube', 'Ours', 'FaCT', 'unknown'):
            result = subprocess.run(
                [str(worker), str(snapshot), str(data / 'causal.tsv'), method],
                text=True, capture_output=True, timeout=120)
            assert result.returncode != 0, result
            error = json.loads(result.stdout)
            assert 'unknown query method' in error['error'], error
        sqlite_cli = data.parents[1] / 'baselines' / 'sqlite' / 'run.py'
        database = root / 'tiny.sqlite'
        run([sys.executable, sqlite_cli, 'prepare', data, database])
        relational = json.loads(run([
            sys.executable, sqlite_cli, 'query', database, data / 'query.json']))
        assert relational['method'] == 'CaRL-PG', relational
        assert relational['status'] == 'ok', relational
        assert int(relational['N']) == 4 and float(relational['answer']) == 1.0, relational
        print('CaRL-PG: counts, probability, and method label match')
        direct = [json.loads(line) for line in run(
            [args.build_dir / 'whatif_online_join', data], request).splitlines()][-1]
        assert direct['status'] == 'ok', direct
        assert int(direct['N']) == 4 and float(direct['answer']) == 1.0, direct
        print('OnlineJoin: counts and probability match')


if __name__ == '__main__':
    main()
