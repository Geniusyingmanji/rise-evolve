#!/usr/bin/env python3
"""Assemble v2 teacher trajectories and SFT/RL splits from refined annotations.

v2 tasks come from build_reasoning_tasks_v2.py (programmatic, verifiable).
Annotations come from refine_tasks_gpt55.py (+ judge filter). This script:

1. joins tasks + accepted refined annotations
2. builds teacher trajectories: user(image + refined instruction) ->
   assistant(reasoning_chain), final_edit_program assembled from refined and
   task fields (weighted grouped checklist, editor_prompt, edit_operations,
   target_scene_description, scene graph with refined editable_region)
3. emits stratified splits per sub_task (seeded):
   sft_train / sft_val / rl_prompt / hard_heldout
   RL rows carry ground_truth + verifier_spec for verifiable rewards.

Usage:
  python3 scripts/data/build_trajectories_and_splits_v2.py \
    --tasks data/tasks/tasks_v2.jsonl \
    --refined data/tasks/tasks_v2_refined.jsonl \
    --judge reports/data_quality/refine_v2_judge.jsonl \
    --version v2
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

GROUP_BUDGETS = {"visual": 0.35, "cognitive": 0.30, "preservation": 0.25, "readability": 0.10}
SPLIT_RATIOS = [("sft_train", 0.60), ("sft_val", 0.05), ("rl_prompt", 0.25), ("hard_heldout", 0.10)]


def jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def weighted_grouped_checklist(groups: dict):
    active = {g: qs for g, qs in groups.items() if isinstance(qs, list) and qs}
    budget_total = sum(GROUP_BUDGETS.get(g, 0.1) for g in active)
    grouped = {g: [] for g in GROUP_BUDGETS}
    idx = 1
    for group, questions in active.items():
        per_q = (GROUP_BUDGETS.get(group, 0.1) / max(budget_total, 1e-9)) / len(questions)
        for q in questions:
            grouped.setdefault(group, []).append(
                {"id": f"C{idx}", "question": q, "weight": round(per_q, 4)})
            idx += 1
    return grouped


def build_trajectory(task: dict, ref: dict) -> dict:
    grouped = weighted_grouped_checklist(ref["atomic_checklist"])
    scene = dict(task.get("source_scene_graph") or {})
    hints = [op.get("region_hint", "") for op in ref["edit_operations"] if op.get("region_hint")]
    if hints:
        scene["editable_region"] = "; ".join(hints)
    program = {
        "task_id": task["task_id"],
        "version": task.get("version", "v2"),
        "split": task.get("split", ""),
        "task_family": task.get("task_family"),
        "source_scene_graph": scene,
        "knowledge_facts": [
            {**k, "used_in_plan": True} if isinstance(k, dict) else {"claim": str(k), "used_in_plan": True}
            for k in (task.get("required_knowledge") or [])
        ],
        "target_scene_description": ref["rational_target_description"],
        "edit_operations": ref["edit_operations"],
        "reference_images": [],
        "preservation_constraints": task.get("preservation_constraints") or [],
        "negative_constraints": task.get("negative_constraints") or [],
        "atomic_checklist": grouped,
        "editor_prompt": ref["editor_prompt"],
        "failure_modes_to_watch": [
            "reasoning result is wrong in the edited image",
            "target region edited but unrelated content drifts",
            "text or symbols rendered unreadably",
        ],
        "created_at": task.get("created_at"),
    }
    return {
        "task_id": task["task_id"],
        "version": task.get("version", "v2"),
        "split": task.get("split", ""),
        "created_at": task.get("created_at"),
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "path": task["source_image"]},
                {"type": "text", "text": ref["instruction"]},
            ]},
            {"role": "assistant", "content": ref["reasoning_chain"]},
        ],
        "tool_evidence_map": {},
        "final_edit_program": program,
        "refinement": {"refined": True, "model": "gpt-5.5", "round": "v2"},
    }


def build_rl_row(task: dict, ref: dict) -> dict:
    return {
        "task_id": task["task_id"],
        "version": task.get("version", "v2"),
        "source_image": task["source_image"],
        "instruction": ref["instruction"] if ref else task["instruction"],
        "task_family": task.get("task_family"),
        "sub_task": task.get("sub_task"),
        "domain": task.get("domain"),
        "difficulty": task.get("difficulty"),
        "atomic_checklist": (ref or {}).get("atomic_checklist") or task.get("atomic_checklist"),
        "preservation_constraints": task.get("preservation_constraints") or [],
        "ground_truth": task.get("ground_truth"),
        "verifier_spec": task.get("verifier_spec"),
        "reward_channel": "verifiable_vqa" if task.get("verifier_spec") else "vlm_judge",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--version", default="v2")
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args()

    refined = {}
    for row in jsonl(args.refined):
        if row.get("ok") and row.get("refined"):
            refined[row["task_id"]] = row["refined"]

    rejected = set()
    if args.judge and Path(args.judge).exists():
        for v in jsonl(args.judge):
            if v.get("error"):
                continue
            if (not v.get("same_edit_semantics") or v.get("instruction_leaks_answer")
                    or v.get("invented_visual_claims")):
                rejected.add(v["task_id"])

    tasks = list(jsonl(args.tasks))
    usable, dropped = [], Counter()
    for t in tasks:
        ref = refined.get(t["task_id"])
        if not ref:
            dropped["no_refinement"] += 1
            continue
        if t["task_id"] in rejected:
            dropped["judge_rejected"] += 1
            continue
        usable.append((t, ref))

    rng = random.Random(args.seed)
    by_sub = defaultdict(list)
    for pair in usable:
        by_sub[pair[0].get("sub_task", "?")].append(pair)

    assignments = defaultdict(list)
    for sub, pairs in sorted(by_sub.items()):
        rng.shuffle(pairs)
        n = len(pairs)
        offset = 0
        for i, (split, ratio) in enumerate(SPLIT_RATIOS):
            count = round(n * ratio) if i < len(SPLIT_RATIOS) - 1 else n - offset
            assignments[split].extend(pairs[offset:offset + count])
            offset += count

    v = args.version
    traj_path = Path(f"data/trajectories/teacher_trajectories_{v}.jsonl")
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    with open(traj_path, "w") as tf:
        for split, pairs in assignments.items():
            for task, ref in pairs:
                task["split"] = split
                traj = build_trajectory(task, ref)
                traj["split"] = split
                tf.write(json.dumps(traj, ensure_ascii=False) + "\n")
                counts["trajectories"] += 1

    for split, pairs in assignments.items():
        if split in ("sft_train", "sft_val"):
            out = split_dir / f"{split}_{v}.jsonl"
            with open(out, "w") as f:
                for task, ref in pairs:
                    traj = build_trajectory(task, ref)
                    traj["split"] = split
                    f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        elif split == "rl_prompt":
            out = split_dir / f"rl_prompt_train_{v}.jsonl"
            with open(out, "w") as f:
                for task, ref in pairs:
                    f.write(json.dumps(build_rl_row(task, ref), ensure_ascii=False) + "\n")
        else:
            out = split_dir / f"hard_heldout_{v}.jsonl"
            with open(out, "w") as f:
                for task, ref in pairs:
                    f.write(json.dumps(build_rl_row(task, ref), ensure_ascii=False) + "\n")
        counts[split] = len(pairs)

    summary = {
        "version": v,
        "total_tasks": len(tasks),
        "usable": len(usable),
        "dropped": dict(dropped),
        "splits": {k: counts[k] for k in ("sft_train", "sft_val", "rl_prompt", "hard_heldout")},
        "per_subtask_usable": {k: len(v_) for k, v_ in sorted(by_sub.items())},
    }
    report = Path(f"reports/data_quality/splits_summary_{v}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
