#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rise_evolve.io import ensure_dir, iter_jsonl, repo_path, write_json, write_jsonl
from rise_evolve.reward.critic import score_verifier_item


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_label: Dict[str, List[float]] = collections.defaultdict(list)
    by_task: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    evidence_modes = collections.Counter()
    failure_counts = collections.Counter()
    for row in rows:
        label = str(row.get("label"))
        by_label[label].append(float(row.get("score", 0.0)))
        task_id = row.get("task_id")
        if task_id and label in {"pass", "fail"}:
            by_task[str(task_id)][label] = float(row.get("score", 0.0))
        for mode in row.get("evidence_modes", []):
            evidence_modes[mode] += 1
        failure_counts[row.get("attribution", {}).get("primary_failure", "unknown")] += 1

    comparable = [scores for scores in by_task.values() if "pass" in scores and "fail" in scores]
    pairwise_acc = None
    if comparable:
        pairwise_acc = sum(1 for scores in comparable if scores["pass"] > scores["fail"]) / len(comparable)
    return {
        "rows": len(rows),
        "label_mean_scores": {label: mean(values) for label, values in by_label.items() if values},
        "pairwise_accuracy": pairwise_acc,
        "pairwise_tasks": len(comparable),
        "evidence_modes": dict(evidence_modes),
        "failure_counts": dict(failure_counts),
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run lightweight RISE-Critic on verifier/reward items.")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-programmatic-priors", action="store_true")
    args = parser.parse_args(argv)

    src = Path(args.input) if args.input else repo_path("data", "train_ready", args.version, "reward_items.jsonl")
    output = Path(args.output) if args.output else repo_path("reports", "reward", f"rise_critic_scores_{args.version}.jsonl")
    summary_output = (
        Path(args.summary_output)
        if args.summary_output
        else repo_path("reports", "reward", f"rise_critic_summary_{args.version}.json")
    )
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(iter_jsonl(src)):
        if args.limit is not None and idx >= args.limit:
            break
        rows.append(score_verifier_item(item, use_programmatic_priors=not args.no_programmatic_priors))
    ensure_dir(output.parent)
    write_jsonl(output, rows)
    summary = summarize(rows)
    summary["warning"] = (
        "programmatic_render_prior is a synthetic-data smoke signal; replace with VLM difference-first critic for real RL"
        if not args.no_programmatic_priors
        else "programmatic priors disabled"
    )
    write_json(summary_output, summary)
    print(json.dumps({"output": str(output), "summary_output": str(summary_output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
