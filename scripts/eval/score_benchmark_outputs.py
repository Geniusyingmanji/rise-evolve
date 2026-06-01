#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rise_evolve.io import ensure_dir, iter_jsonl, write_json, write_jsonl
from rise_evolve.reward.critic import score_agent_result


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Score benchmark agent outputs with lightweight RISE-Critic.")
    parser.add_argument("--programs", required=True)
    parser.add_argument("--render-metadata", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    render_lookup: Dict[str, str] = {}
    if args.render_metadata:
        for row in iter_jsonl(Path(args.render_metadata)):
            if row.get("candidate_image"):
                render_lookup[str(row.get("sample_id"))] = row["candidate_image"]
    scores: List[Dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(Path(args.programs))):
        if args.limit is not None and idx >= args.limit:
            break
        sample_id = str(row.get("sample_id") or row.get("task_id") or idx)
        if sample_id in render_lookup:
            row = dict(row)
            row["candidate_image"] = render_lookup[sample_id]
        score = score_agent_result(row)
        score.update({"sample_id": sample_id, "benchmark": row.get("benchmark"), "category": row.get("category")})
        scores.append(score)
    output = Path(args.output)
    ensure_dir(output.parent)
    write_jsonl(output, scores)
    summary = {
        "rows": len(scores),
        "mean_score": mean([row["score"] for row in scores]) if scores else None,
        "mean_heads": {},
    }
    if scores:
        for head in scores[0]["heads"]:
            summary["mean_heads"][head] = mean([row["heads"].get(head, 0.0) for row in scores])
    if args.summary_output:
        write_json(Path(args.summary_output), summary)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
