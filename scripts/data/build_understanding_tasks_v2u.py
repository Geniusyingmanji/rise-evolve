#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRONG_KEYS = {
    "arithmetic_chain": ["negative_result"],
    "balance_scale": ["wrong_value"],
    "fraction_shade": ["wrong_count"],
    "geometry_angle": ["wrong_angle"],
    "graph_path": ["negative_path", "negative_distance"],
    "mirror_reflection": ["negative_dx"],
    "moon_phase": ["wrong_phase"],
    "ph_indicator": ["negative_color"],
    "process_order": ["negative_order"],
    "sequence_pattern": ["wrong_answer"],
    "sorting_step": ["negative_index"],
    "sudoku4": ["wrong_value"],
    "food_chain": ["wrong_index", "negative_index"],
}
STATIC_WRONG = {
    "block_stack_view": "blocks removed but stack not collapsed by gravity",
    "circuit_bulb": "switch closed but bulb still gray (unlit)",
    "clock_arithmetic": "wrong hour-hand position",
}
FAILURE_TYPES = {
    **{k: "wrong_value" for k in (
        "sudoku4", "balance_scale", "arithmetic_chain", "geometry_angle",
        "sequence_pattern", "fraction_shade", "clock_arithmetic", "moon_phase"
    )},
    **{k: "wrong_color" for k in ("ph_indicator", "circuit_bulb")},
    **{k: "wrong_order" for k in ("process_order", "sorting_step")},
    **{k: "wrong_direction" for k in ("mirror_reflection", "food_chain", "graph_path")},
    "block_stack_view": "wrong_region",
}


def rel(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_tasks(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def task_paths(task):
    task_id = task["task_id"]
    source = ROOT / "data" / "images" / "source" / f"{task_id}.png"
    teacher = ROOT / "data" / "renders" / "teacher" / f"{task_id}_teacher.png"
    negative = ROOT / "data" / "renders" / "negative" / f"{task_id}_negative.png"
    return rel(source), rel(teacher), rel(negative)


def check_paths(paths):
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError("missing image paths: " + ", ".join(missing[:10]))


def intended_change(task):
    ops = task.get("edit_operations") or []
    change = (ops[0] or {}).get("change") if ops else ""
    base = task.get("expected_target", "").strip()
    if change:
        return f"{base} {change}".strip()
    return base


def wrong_info(task):
    sub_task = task["sub_task"]
    params = task["verifier_spec"]["programmatic"]["params"]
    if sub_task in STATIC_WRONG:
        return STATIC_WRONG[sub_task]
    parts = []
    for key in WRONG_KEYS[sub_task]:
        if key in params:
            parts.append(f"{key}={params[key]}")
    if not parts:
        raise KeyError(f"no wrong-info key found for {task['task_id']}")
    return ", ".join(parts)


def wrong_sentence(task):
    constraints = task.get("negative_constraints") or []
    suffix = constraints[0].strip() if constraints else ""
    info = wrong_info(task)
    if suffix:
        return f"Negative render shows {info}. {suffix}"
    return f"Negative render shows {info}."


def row(task, kind, mode, images, prompt, answer):
    return {
        "id": f"v2u_{kind}_{task['task_id']}_{mode}",
        "images": images,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))},
        ],
        "metadata": {
            "task_id": task["task_id"],
            "sub_task": task["sub_task"],
            "mode": mode,
            "verifiable": True,
        },
    }


def build_vqa(task):
    _, teacher, _ = task_paths(task)
    checks = task["verifier_spec"]["vqa_checks"]
    questions = [f"{i}. {c['question']}" for i, c in enumerate(checks, 1)]
    prompt = "<image>\nAnswer these questions with strict JSON only:\n" + "\n".join(questions)
    answers = [c["expected_answer"] for c in checks]
    return row(task, "vqa", "teacher", [teacher], prompt, {"answers": answers})


def diff_answer(task, mode):
    intended = intended_change(task)
    if mode == "teacher":
        return {
            "observed_changes": [intended],
            "intended_changes_present": True,
            "unintended_or_wrong_changes": [],
            "missing_changes": [],
        }
    return {
        "observed_changes": [],
        "intended_changes_present": False,
        "unintended_or_wrong_changes": [wrong_sentence(task)],
        "missing_changes": [intended],
    }


def build_diff(task, mode):
    source, teacher, negative = task_paths(task)
    target = teacher if mode == "teacher" else negative
    prompt = (
        "<image>\n<image>\n"
        "Compare the source image to the edited image. Return strict JSON with "
        "observed_changes, intended_changes_present, unintended_or_wrong_changes, "
        "and missing_changes."
    )
    return row(task, "diff", mode, [source, target], prompt, diff_answer(task, mode))


def judge_answer(task, mode):
    intended = intended_change(task)
    if mode == "teacher":
        return {
            "pass": True,
            "failure_type": "none",
            "rationale": f"The teacher render contains the intended change: {intended}",
        }
    return {
        "pass": False,
        "failure_type": FAILURE_TYPES[task["sub_task"]],
        "rationale": f"The negative render fails the intended change. {wrong_sentence(task)}",
    }


def build_judge(task, mode):
    source, teacher, negative = task_paths(task)
    target = teacher if mode == "teacher" else negative
    prompt = (
        "<image>\n<image>\n"
        f"Instruction: {task['instruction']}\n"
        "Judge whether the edited image satisfies the instruction. Return strict JSON "
        "with pass, failure_type, and rationale."
    )
    return row(task, "judge", mode, [source, target], prompt, judge_answer(task, mode))


def split_task_ids(tasks, val_ratio, seed):
    by_subtask = {}
    for task in tasks:
        by_subtask.setdefault(task["sub_task"], []).append(task["task_id"])
    val_ids = set()
    rng = random.Random(seed)
    for task_ids in by_subtask.values():
        ids = list(task_ids)
        rng.shuffle(ids)
        n_val = int(len(ids) * val_ratio + 0.5)
        if val_ratio > 0 and ids and n_val == 0:
            n_val = 1
        val_ids.update(ids[:n_val])
    return val_ids


def validate_rows(rows):
    for out_row in rows:
        check_paths(out_row["images"])
        json.loads(out_row["conversations"][1]["value"])


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for out_row in rows:
            f.write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/tasks/tasks_v2.jsonl")
    parser.add_argument("--out-dir", default="data/understanding")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    tasks = read_tasks(ROOT / args.tasks)
    val_ids = split_task_ids(tasks, args.val_ratio, seed=7)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    built = {"vqa": [], "diff": [], "judge": []}
    for task in tasks:
        built["vqa"].append(build_vqa(task))
        for mode in ("teacher", "negative"):
            built["diff"].append(build_diff(task, mode))
            built["judge"].append(build_judge(task, mode))

    for kind, rows in built.items():
        splits = {"train": [], "val": []}
        for out_row in rows:
            split = "val" if out_row["metadata"]["task_id"] in val_ids else "train"
            splits[split].append(out_row)
        for split, split_rows in splits.items():
            validate_rows(split_rows)
            path = out_dir / f"v2u_{kind}_{split}.jsonl"
            write_jsonl(path, split_rows)
            print(f"{rel(path)}\t{len(split_rows)}")


if __name__ == "__main__":
    main()
