#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rise_evolve.io import ensure_dir, iter_jsonl, utc_now


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Aggregate RISE-Critic benchmark scores into a markdown report.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = list(iter_jsonl(Path(args.scores)))
    by_benchmark: Dict[str, List[float]] = collections.defaultdict(list)
    by_category: Dict[str, List[float]] = collections.defaultdict(list)
    failures = collections.Counter()
    for row in rows:
        score = float(row.get("score", 0.0))
        by_benchmark[str(row.get("benchmark", "unknown"))].append(score)
        by_category[str(row.get("category", "unknown"))].append(score)
        failures[row.get("attribution", {}).get("primary_failure", "unknown")] += 1
    lines = [
        "# RISEvolve Eval Report",
        "",
        f"created_at: {utc_now()}",
        f"rows: {len(rows)}",
        f"mean_score: {fmt(mean([float(row.get('score', 0.0)) for row in rows]) if rows else None)}",
        "",
        "## By Benchmark",
        "",
        "| Benchmark | Rows | Mean score |",
        "|---|---:|---:|",
    ]
    for key, values in sorted(by_benchmark.items()):
        lines.append(f"| {key} | {len(values)} | {fmt(mean(values))} |")
    lines.extend(["", "## By Category", "", "| Category | Rows | Mean score |", "|---|---:|---:|"])
    for key, values in sorted(by_category.items()):
        lines.append(f"| {key} | {len(values)} | {fmt(mean(values))} |")
    lines.extend(["", "## Failure Attribution", "", "| Failure | Count |", "|---|---:|"])
    for key, count in failures.most_common():
        lines.append(f"| {key} | {count} |")
    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
