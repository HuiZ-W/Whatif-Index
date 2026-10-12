"""Check all documented construction policies through the public query interface."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


METHODS = (
    'full-join', 'full-join-rollup', 'parent-no-reuse',
    'boundary-adjustment-prune', 'boundary-adjustment-random-prune',
    'boundary-adjustment-no-prune', 'boundary-adjustment-prune-no-reuse',
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
    answer = records[-1]
    assert answer['status'] == 'ok', answer
    assert int(answer['N']) == 4, answer
    assert float(answer['answer']) == 1.0, answer
    return {key: answer[key] for key in ('status', 'N', 'answer', 'strata')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--reference-builder', type=Path)
    parser.add_argument('--reference-query', type=Path)
    args = parser.parse_args()
    assert bool(args.reference_builder) == bool(args.reference_query)
    data = Path(__file__).resolve().parents[1] / 'examples' / 'tiny'
    request = (data / 'query.json').read_text()
    worker = args.build_dir / 'whatif_query'
    expected = None
    with tempfile.TemporaryDirectory(prefix='whatif-cli-') as temporary:
        root = Path(temporary)
        for method in METHODS:
            mode = 'FullCube' if method.startswith('full-join') else 'Ours'
            snapshot = root / (method + '.bin')
            run([args.build_dir / 'whatif_build', data, method, snapshot, '6'])
            actual = query(worker, snapshot, data, mode, request)
            if expected is None:
                expected = actual
            assert actual == expected, (method, actual, expected)
            if args.reference_builder:
                reference_snapshot = root / (method + '-reference.bin')
                run([args.reference_builder, data, method, reference_snapshot, '6'])
                assert query(args.reference_query, reference_snapshot, data, mode, request) == actual
                # Both directions must preserve the existing snapshot format.
                assert query(worker, reference_snapshot, data, mode, request) == actual
                assert query(args.reference_query, snapshot, data, mode, request) == actual
            print(f'{method}: counts and probability match')
        direct = [json.loads(line) for line in run(
            [args.build_dir / 'whatif_online_join', data], request).splitlines()][-1]
        assert direct['status'] == 'ok', direct
        assert int(direct['N']) == 4 and float(direct['answer']) == 1.0, direct
        print('OnlineJoin: counts and probability match')


if __name__ == '__main__':
    main()
