from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="CranePilot reproducible research")
    subs = parser.add_subparsers(dest="command", required=True)
    importer = subs.add_parser("import-data")
    importer.add_argument("--source", required=True, type=Path)
    importer.add_argument("--output", default=Path("data/processed"), type=Path)
    benchmark = subs.add_parser("benchmark")
    benchmark.add_argument("--data", default=Path("data/processed"), type=Path)
    benchmark.add_argument("--config", required=True, type=Path)
    benchmark.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "import-data":
        from .data import import_dataset
        result = import_dataset(args.source, args.output)
        print(f"Imported {len(result['scenarios'])} original scenarios")
    elif args.command == "benchmark":
        from .benchmark import run_benchmark
        result = run_benchmark(args.data, args.output, json.loads(args.config.read_text()))
        print(f"Wrote {len(result)} metric rows")


if __name__ == "__main__":
    main()

