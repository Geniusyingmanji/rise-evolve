#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rise_evolve.agent.schemas import (
    build_answer_text,
    extract_source_image_and_instruction,
    render_tool_trace_for_sft,
    validate_edit_program,
)
from rise_evolve.agent.system_prompt import EDIT_PROGRAM_FORMAT_HINT, RISEVOLVE_SYSTEM_PROMPT
from rise_evolve.io import ensure_dir, iter_jsonl, repo_path, write_json


def input_path(version: str, split: str) -> Path:
    if split == "train":
        return repo_path("data", "splits", f"sft_train_{version}.jsonl")
    if split == "val":
        return repo_path("data", "splits", f"sft_val_{version}.jsonl")
    raise ValueError(f"unsupported split: {split}")


def assistant_text(row: Dict[str, Any]) -> str:
    program = row.get("final_edit_program") or row.get("edit_program")
    trace_text = render_tool_trace_for_sft(row)
    answer = build_answer_text(program or {})
    return "\n".join(x for x in [trace_text, answer] if x)


def to_sharegpt(row: Dict[str, Any]) -> Dict[str, Any]:
    source_image, instruction = extract_source_image_and_instruction(row)
    program = row.get("final_edit_program") or row.get("edit_program") or {}
    validation = validate_edit_program(program)
    user_value = (
        "<image>\n"
        + RISEVOLVE_SYSTEM_PROMPT.strip()
        + "\n\n"
        + EDIT_PROGRAM_FORMAT_HINT.strip()
        + "\n\nInstruction: "
        + instruction.strip()
    )
    return {
        "id": row.get("task_id"),
        "images": [source_image] if source_image else [],
        "conversations": [
            {"from": "human", "value": user_value},
            {"from": "gpt", "value": assistant_text(row)},
        ],
        "metadata": {
            "version": row.get("version"),
            "split": row.get("split"),
            "schema_ok": validation.ok,
            "schema_errors": validation.errors,
            "schema_warnings": validation.warnings,
        },
    }


def to_messages(row: Dict[str, Any]) -> Dict[str, Any]:
    source_image, instruction = extract_source_image_and_instruction(row)
    program = row.get("final_edit_program") or row.get("edit_program") or {}
    validation = validate_edit_program(program)
    return {
        "id": row.get("task_id"),
        "images": [source_image] if source_image else [],
        "messages": [
            {"role": "system", "content": RISEVOLVE_SYSTEM_PROMPT + "\n" + EDIT_PROGRAM_FORMAT_HINT},
            {"role": "user", "content": "<image>\n" + instruction.strip()},
            {"role": "assistant", "content": assistant_text(row)},
        ],
        "metadata": {
            "version": row.get("version"),
            "split": row.get("split"),
            "schema_ok": validation.ok,
            "schema_errors": validation.errors,
            "schema_warnings": validation.warnings,
        },
    }


def convert_rows(path: Path, output_format: str, limit: int | None) -> Iterable[Dict[str, Any]]:
    converter = to_sharegpt if output_format == "sharegpt" else to_messages
    for idx, row in enumerate(iter_jsonl(path)):
        if limit is not None and idx >= limit:
            break
        yield converter(row)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Convert RISEvolve trajectories to VLM SFT format.")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--format", choices=["sharegpt", "messages"], default="sharegpt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    src = input_path(args.version, args.split)
    if not src.exists():
        raise FileNotFoundError(src)
    output = Path(args.output) if args.output else repo_path("data", "train_ready", args.version, f"sft_{args.split}_lf.json")
    rows = list(convert_rows(src, args.format, args.limit))
    ensure_dir(output.parent)
    write_json(output, rows)
    schema_ok = sum(1 for row in rows if row.get("metadata", {}).get("schema_ok"))
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "format": args.format,
                "schema_ok": schema_ok,
                "schema_fail": len(rows) - schema_ok,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
