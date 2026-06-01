#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rise_evolve.agent.system_prompt import EDIT_PROGRAM_FORMAT_HINT, RISEVOLVE_SYSTEM_PROMPT
from rise_evolve.io import ensure_dir, iter_jsonl, repo_path, write_jsonl


def convert(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_id": row.get("task_id"),
        "task_id": row.get("task_id"),
        "source_image": row.get("source_image"),
        "instruction": row.get("instruction"),
        "messages": [
            {"role": "system", "content": RISEVOLVE_SYSTEM_PROMPT + "\n" + EDIT_PROGRAM_FORMAT_HINT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": row.get("source_image")},
                    {"type": "text", "text": row.get("instruction", "")},
                ],
            },
        ],
        "reward_reference": {
            "expected_target": row.get("expected_target"),
            "atomic_checklist": row.get("atomic_checklist"),
            "difficulty": row.get("difficulty"),
        },
    }


def convert_rows(path: Path, limit: int | None) -> Iterable[Dict[str, Any]]:
    for idx, row in enumerate(iter_jsonl(path)):
        if limit is not None and idx >= limit:
            break
        yield convert(row)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Convert RISEvolve RL prompts to rollout JSONL.")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    src = Path(args.input) if args.input else repo_path("data", "splits", f"rl_prompt_train_{args.version}.jsonl")
    output = Path(args.output) if args.output else repo_path("data", "train_ready", args.version, "rl_prompts.jsonl")
    rows = list(convert_rows(src, args.limit))
    ensure_dir(output.parent)
    write_jsonl(output, rows)
    print(json.dumps({"output": str(output), "rows": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
