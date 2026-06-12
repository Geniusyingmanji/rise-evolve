#!/usr/bin/env python3
"""Independent GPT-5.5 judge for refined task annotations.

For each refined row, asks a fresh judge (no rewrite context) whether the
refinement preserved edit semantics, avoided answer leakage into the
instruction, and avoided inventing visual content. Writes per-row verdicts
and a summary.

Usage:
  python3 scripts/data/judge_refined_pilot.py \
    --tasks data/tasks/tasks_v1.jsonl \
    --refined data/tasks/tasks_v1_refined_pilot.jsonl \
    --output reports/data_quality/refine_pilot_judge.jsonl
"""

import argparse
import concurrent.futures
import json
import sys
import threading
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from refine_tasks_gpt55 import call_gpt55, load_scene_inventory, parse_json_reply  # noqa: E402

JUDGE_PROMPT = """You are auditing a rewritten image-editing training annotation.
The ORIGINAL defines the ground-truth edit (the edited image is already rendered
from it and cannot change). SCENE_INVENTORY is a verified, code-derived list of
exactly what the source image contains and what the teacher render changes; treat
it as ground truth. A rewrite referencing elements from present_elements or the
teacher_change is GROUNDED, not invented. Judge the REWRITE strictly.

ORIGINAL:
{original}

SCENE_INVENTORY (verified ground truth of the rendered images):
{scene}

REWRITE:
{refined}

Answer strict JSON only:
{{
  "same_edit_semantics": true|false,   // rewrite requests exactly the same visual edit on the same object
  "instruction_leaks_answer": true|false,  // rewritten instruction reveals the solved answer/target state in a way the original instruction did not
  "invented_visual_claims": true|false,    // any rewrite field claims visual content not implied by the original fields OR the scene inventory
  "quality_1to5": int,                 // overall annotation quality for planner training
  "notes": "one short sentence"
}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-judge everything instead of skipping ids already in output")
    ap.add_argument("--inventory", nargs="*", default=None,
                    help="extra scene inventory JSON files merged over the v1 default")
    args = ap.parse_args()

    tasks = {}
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                tasks[row["task_id"]] = row

    refined_rows = []
    with open(args.refined) as f:
        for line in f:
            line = line.strip()
            if line:
                refined_rows.append(json.loads(line))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists() and not args.no_resume:
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line).get("task_id"))
        refined_rows = [r for r in refined_rows if r["task_id"] not in done_ids]
    print(f"to_judge={len(refined_rows)} already_done={len(done_ids)}")

    lock = threading.Lock()
    stats = Counter()
    results = []
    inventories = load_scene_inventory(args.inventory)

    def judge(row):
        task = tasks[row["task_id"]]
        scene = inventories.get(task.get("sub_task", ""), {})
        original = {
            "instruction": task.get("instruction"),
            "expected_target": task.get("expected_target"),
            "edit_operations": task.get("edit_operations"),
            "required_knowledge": task.get("required_knowledge"),
        }
        if task.get("ground_truth"):
            original["ground_truth"] = task["ground_truth"]
        if task.get("verifier_spec"):
            original["verifier_spec"] = task["verifier_spec"]
        refined = {
            k: row["refined"].get(k)
            for k in ["instruction", "editor_prompt", "rational_target_description",
                      "reasoning_chain", "edit_operations", "atomic_checklist"]
        }
        prompt = JUDGE_PROMPT.format(
            original=json.dumps(original, ensure_ascii=False, indent=1),
            scene=json.dumps(scene, ensure_ascii=False, indent=1),
            refined=json.dumps(refined, ensure_ascii=False, indent=1),
        )
        try:
            verdict = parse_json_reply(call_gpt55(prompt, effort=args.effort))
        except Exception as exc:  # noqa: BLE001
            verdict = {"error": str(exc)}
        verdict["task_id"] = row["task_id"]
        with lock:
            results.append(verdict)
            with open(out_path, "a") as f:
                f.write(json.dumps(verdict, ensure_ascii=False) + "\n")
            n_done = len(results)
            if n_done % 50 == 0:
                print(f"judged {n_done}/{len(refined_rows)}", flush=True)
        return verdict

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(judge, refined_rows))

    # summary over the FULL output file (including resumed rows)
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            if v.get("error"):
                stats["error"] += 1
            else:
                stats["same_edit"] += int(bool(v.get("same_edit_semantics")))
                stats["leak"] += int(bool(v.get("instruction_leaks_answer")))
                stats["invented"] += int(bool(v.get("invented_visual_claims")))
                stats["q_sum"] += int(v.get("quality_1to5", 0))
                stats["judged"] += 1

    n = max(stats["judged"], 1)
    summary = {
        "judged": stats["judged"],
        "errors": stats["error"],
        "same_edit_semantics_ratio": round(stats["same_edit"] / n, 4),
        "instruction_leak_ratio": round(stats["leak"] / n, 4),
        "invented_claims_ratio": round(stats["invented"] / n, 4),
        "mean_quality": round(stats["q_sum"] / n, 2),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
