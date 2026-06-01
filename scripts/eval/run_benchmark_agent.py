#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rise_evolve.agent.schemas import validate_edit_program
from rise_evolve.io import ensure_dir, iter_jsonl, utc_now, write_jsonl


def dry_run_program(row: Dict[str, Any]) -> Dict[str, Any]:
    instruction = row.get("instruction") or ""
    reference = row.get("reference_text")
    task_id = f"{row.get('benchmark')}_{row.get('sample_id')}"
    target_desc = reference or "A minimal, reasoning-correct edit that follows the instruction while preserving unrelated content."
    program = {
        "task_id": task_id,
        "created_at": utc_now(),
        "source_scene_graph": {
            "objects": [],
            "editable_region": "region implied by the instruction",
            "preserve_region": "all unrelated source-image content",
        },
        "task_family": row.get("category") or row.get("benchmark"),
        "knowledge_facts": [{"claim": str(reference), "source": "benchmark_reference_text", "used_in_plan": True}]
        if reference
        else [],
        "target_scene_description": target_desc,
        "edit_operations": [
            {
                "op": "transform_or_annotate",
                "target": "instruction-specified region",
                "change": instruction,
                "region_hint": "localize from source image and instruction",
                "preserve": ["background", "layout", "identity", "viewpoint", "lighting"],
            }
        ],
        "reference_images": [],
        "preservation_constraints": [
            "Keep non-target content unchanged.",
            "Avoid adding explanatory text unless the task asks for text.",
        ],
        "negative_constraints": [
            "Do not change the task premise.",
            "Do not solve by replacing the whole image.",
        ],
        "atomic_checklist": {
            "cognitive": [
                {"id": "C1", "question": "Is the reasoning or knowledge target correct?", "weight": 0.35}
            ],
            "visual": [{"id": "V1", "question": "Is the requested edit executed?", "weight": 0.35}],
            "preservation": [
                {"id": "P1", "question": "Are unrelated regions preserved?", "weight": 0.20}
            ],
            "readability": [{"id": "Q1", "question": "Is the result visually clear?", "weight": 0.10}],
        },
        "editor_prompt": f"{instruction} Desired result: {target_desc}. Preserve unrelated visual context.",
        "failure_modes_to_watch": ["knowledge_fail", "region_fail", "over_edit", "under_edit"],
    }
    return program


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run RISEvolve agent on an eval manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.dry_run:
        raise SystemExit("Only --dry-run is implemented locally. Wire model serving here for real eval.")
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    programs: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(Path(args.manifest))):
        if args.limit is not None and idx >= args.limit:
            break
        program = dry_run_program(row)
        validation = validate_edit_program(program)
        sample_id = f"{row.get('benchmark')}_{row.get('sample_id')}"
        programs.append(
            {
                "sample_id": sample_id,
                "task_id": sample_id,
                "benchmark": row.get("benchmark"),
                "category": row.get("category"),
                "source_image": row.get("source_image"),
                "instruction": row.get("instruction"),
                "edit_program": program,
                "validation": validation.to_dict(),
                "run_mode": "dry_run",
            }
        )
        traces.append(
            {
                "sample_id": sample_id,
                "tool_trace": [
                    {"name": "analyze_image", "arguments": {"image": row.get("source_image")}, "result": "dry_run"},
                    {"name": "query_edit_knowledge", "arguments": {"category": row.get("category")}, "result": "dry_run"},
                ],
            }
        )
    write_jsonl(out_dir / "programs.jsonl", programs)
    write_jsonl(out_dir / "tool_traces.jsonl", traces)
    print(json.dumps({"output_dir": str(out_dir), "rows": len(programs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
