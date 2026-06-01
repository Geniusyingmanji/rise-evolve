#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rise_evolve.io import ensure_dir, read_json, repo_path, write_jsonl


def rise_rows(limit: int | None = None) -> Iterable[Dict[str, Any]]:
    payload = read_json(repo_path("data", "benchmarks", "rise", "raw", "data_total.json"), [])
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        yield {
            "benchmark": "rise",
            "sample_id": str(row.get("index", idx)),
            "category": row.get("category"),
            "instruction": row.get("instruction"),
            "source_image": row.get("image"),
            "reference_text": row.get("reference"),
            "metadata": row,
        }


def grade_rows(limit: int | None = None) -> Iterable[Dict[str, Any]]:
    payload = read_json(repo_path("data", "benchmarks", "grade", "raw", "data.json"), [])
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        yield {
            "benchmark": "grade",
            "sample_id": str(row.get("task_id", idx)),
            "category": row.get("domain") or row.get("sub_task"),
            "instruction": row.get("text"),
            "source_image": row.get("image_path"),
            "reference_text": row.get("gt"),
            "questions": row.get("questions"),
            "metadata": row,
        }


def kris_rows(limit: int | None = None) -> Iterable[Dict[str, Any]]:
    readme = repo_path("data", "benchmarks", "kris", "raw", "README.md")
    if not readme.exists():
        return
    count = 0
    for line_no, line in enumerate(readme.read_text(encoding="utf-8", errors="ignore").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "knowledge" not in line.lower() and "reasoning" not in line.lower() and "KRIS" not in line:
            continue
        if limit is not None and count >= limit:
            break
        count += 1
        yield {
            "benchmark": "kris",
            "sample_id": f"readme:{line_no}",
            "category": "readme_taxonomy",
            "instruction": line,
            "source_image": None,
            "reference_text": None,
            "metadata": {"line_no": line_no},
        }


def collect(benchmark: str, limit: int | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if benchmark in {"rise", "all"}:
        rows.extend(rise_rows(limit))
    if benchmark in {"grade", "all"}:
        rows.extend(grade_rows(limit))
    if benchmark in {"kris", "all"}:
        rows.extend(kris_rows(limit))
    return rows


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Build eval-only benchmark manifests from frozen raw snapshots.")
    parser.add_argument("--benchmark", choices=["rise", "grade", "kris", "all"], default="all")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    output = Path(args.output) if args.output else repo_path("data", "benchmarks", "manifests", f"{args.benchmark}_eval_manifest.jsonl")
    rows = collect(args.benchmark, args.limit)
    ensure_dir(output.parent)
    write_jsonl(output, rows)
    print(json.dumps({"output": str(output), "rows": len(rows), "benchmark": args.benchmark}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
