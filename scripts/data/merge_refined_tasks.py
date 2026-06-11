#!/usr/bin/env python3
"""Merge GPT-5.5 refined annotations back into task and trajectory files.

Joins refine_tasks_gpt55.py output (and optionally judge_refined_pilot.py
verdicts as a quality filter) with the original tasks/trajectories, producing
a `v1r` refined snapshot:

- tasks: instruction, rational_target_description, edit_operations,
  atomic_checklist (flat weighted list) replaced; refinement metadata added
- trajectories: user instruction text replaced, assistant filler replaced by
  the refined reasoning_chain, final_edit_program fields replaced
  (editor_prompt, edit_operations, target_scene_description, atomic_checklist
  as weighted group dict, source_scene_graph.editable_region from region_hint)

Rows without an accepted refinement (or rejected by the judge) are dropped by
default (--keep-unrefined to keep originals instead).

Usage:
  python3 scripts/data/merge_refined_tasks.py \
    --tasks data/tasks/tasks_v1.jsonl \
    --trajectories data/trajectories/teacher_trajectories_v1.jsonl \
    --refined data/tasks/tasks_v1_refined_full.jsonl \
    --judge reports/data_quality/refine_full_judge.jsonl \
    --out-tasks data/tasks/tasks_v1r.jsonl \
    --out-trajectories data/trajectories/teacher_trajectories_v1r.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

GROUP_BUDGETS = {"visual": 0.35, "cognitive": 0.30, "preservation": 0.25, "readability": 0.10}


def jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def weighted_checklist(groups: dict):
    """Convert {group: [question,...]} to flat list and weighted group dict."""
    active = {g: qs for g, qs in groups.items() if isinstance(qs, list) and qs}
    budget_total = sum(GROUP_BUDGETS.get(g, 0.1) for g in active)
    flat = []
    grouped = {g: [] for g in GROUP_BUDGETS}
    idx = 1
    for group, questions in active.items():
        group_budget = GROUP_BUDGETS.get(group, 0.1) / max(budget_total, 1e-9)
        per_q = group_budget / len(questions)
        for q in questions:
            item = {"id": f"C{idx}", "question": q, "weight": round(per_q, 4)}
            flat.append(item)
            grouped.setdefault(group, []).append(item)
            idx += 1
    return flat, grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--out-tasks", required=True)
    ap.add_argument("--out-trajectories", required=True)
    ap.add_argument("--keep-unrefined", action="store_true")
    args = ap.parse_args()

    refined = {}
    for row in jsonl(args.refined):
        if row.get("ok") and row.get("refined"):
            refined[row["task_id"]] = row

    rejected_by_judge = set()
    if args.judge and Path(args.judge).exists():
        for v in jsonl(args.judge):
            if v.get("error"):
                continue
            if (not v.get("same_edit_semantics")
                    or v.get("instruction_leaks_answer")
                    or v.get("invented_visual_claims")):
                rejected_by_judge.add(v["task_id"])

    stats = Counter()
    out_tasks = Path(args.out_tasks)
    out_traj = Path(args.out_trajectories)
    out_tasks.parent.mkdir(parents=True, exist_ok=True)
    out_traj.parent.mkdir(parents=True, exist_ok=True)

    usable = {}
    for tid, row in refined.items():
        if tid in rejected_by_judge:
            stats["judge_rejected"] += 1
        else:
            usable[tid] = row["refined"]

    with open(out_tasks, "w") as f:
        for task in jsonl(args.tasks):
            tid = task["task_id"]
            ref = usable.get(tid)
            if not ref:
                stats["unrefined"] += 1
                if args.keep_unrefined:
                    f.write(json.dumps(task, ensure_ascii=False) + "\n")
                    stats["tasks_kept_original"] += 1
                continue
            flat, _ = weighted_checklist(ref["atomic_checklist"])
            task["instruction"] = ref["instruction"]
            task["rational_target_description"] = ref["rational_target_description"]
            task["edit_operations"] = ref["edit_operations"]
            task["atomic_checklist"] = flat
            task["refinement"] = {"refined": True, "model": "gpt-5.5", "round": "v1r"}
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
            stats["tasks_refined"] += 1

    with open(out_traj, "w") as f:
        for traj in jsonl(args.trajectories):
            tid = traj["task_id"]
            ref = usable.get(tid)
            if not ref:
                if args.keep_unrefined:
                    f.write(json.dumps(traj, ensure_ascii=False) + "\n")
                continue
            _, grouped = weighted_checklist(ref["atomic_checklist"])

            messages = []
            user_done = False
            for msg in traj.get("messages", []):
                if msg.get("role") == "user" and not user_done:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                part["text"] = ref["instruction"]
                    messages.append(msg)
                    user_done = True
                elif msg.get("role") != "assistant":
                    messages.append(msg)
            messages.append({"role": "assistant", "content": ref["reasoning_chain"]})

            program = traj.get("final_edit_program") or {}
            program["editor_prompt"] = ref["editor_prompt"]
            program["edit_operations"] = ref["edit_operations"]
            program["target_scene_description"] = ref["rational_target_description"]
            program["atomic_checklist"] = grouped
            region_hints = [op.get("region_hint", "") for op in ref["edit_operations"]]
            if region_hints and any(region_hints):
                scene = program.get("source_scene_graph") or {}
                scene["editable_region"] = "; ".join(h for h in region_hints if h)
                program["source_scene_graph"] = scene

            traj["messages"] = messages
            traj["final_edit_program"] = program
            traj["refinement"] = {"refined": True, "model": "gpt-5.5", "round": "v1r"}
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            stats["trajectories_refined"] += 1

    print(json.dumps(dict(stats), indent=2))
    print(f"tasks -> {out_tasks}")
    print(f"trajectories -> {out_traj}")


if __name__ == "__main__":
    main()
