"""CaRL-PG: SQLite match joins and frequency aggregation for do-queries."""
import argparse
import hashlib
import json
from pathlib import Path

from sqlite_join import SQLiteJoin, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("source", type=Path)
    build.add_argument("database", type=Path)
    query = sub.add_parser("query")
    query.add_argument("database", type=Path)
    query.add_argument("request", type=Path)
    query.add_argument("--settings", default="default_no_statement_cache",
                       choices=["configured", "default", "default_no_statement_cache"])
    args = parser.parse_args()
    if args.command == "prepare":
        digest = hashlib.sha256()
        with (args.source / "assignments_raw.tsv").open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        args.database.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(prepare(args.source, args.database, digest.hexdigest()), default=str))
    else:
        engine = SQLiteJoin(args.database, settings=args.settings)
        try:
            result = engine.query(json.loads(args.request.read_text()), True)
            result['method'] = 'CaRL-PG'
            print(json.dumps(result, default=str))
        finally:
            engine.close()


if __name__ == "__main__":
    main()
