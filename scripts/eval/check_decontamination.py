#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rise_evolve.eval.decontamination import check_files
from rise_evolve.io import repo_path, write_json


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Check train files against frozen benchmark text fingerprints.")
    parser.add_argument("--benchmarks", default="data/benchmarks")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ngram-threshold", type=float, default=0.82)
    parser.add_argument("--fail-on", choices=["high", "medium", "none"], default="high")
    args = parser.parse_args(argv)

    train_files = [Path(x) for x in args.train]
    result = check_files(
        train_files,
        benchmarks_dir=repo_path(args.benchmarks) if not Path(args.benchmarks).is_absolute() else Path(args.benchmarks),
        ngram_threshold=args.ngram_threshold,
        limit=args.limit,
    )
    if args.output:
        write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on == "none":
        return 0
    if args.fail_on == "high" and result["severity_counts"].get("high", 0):
        return 1
    if args.fail_on == "medium" and (
        result["severity_counts"].get("high", 0) or result["severity_counts"].get("medium", 0)
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
