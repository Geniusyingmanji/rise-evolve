#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import Any, Dict, List

from common import repo_path, utc_now, write_json


REQUIRED_TASK_FIELDS = [
    "task_id",
    "benchmark_family",
    "task_family",
    "sub_family",
    "domain",
    "source_image",
    "instruction",
    "expected_target",
    "rational_target_description",
    "edit_operations",
    "preservation_constraints",
    "negative_constraints",
    "atomic_checklist",
    "difficulty",
    "leakage_tags",
]


def read_jsonl(path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
    return rows


def validate(version: str) -> Dict[str, Any]:
    root = repo_path()
    paths = {
        "tasks": repo_path("data", "tasks", f"tasks_{version}.jsonl"),
        "recipes": repo_path("data", "recipes", f"recipes_{version}.jsonl"),
        "trajectories": repo_path("data", "trajectories", f"teacher_trajectories_{version}.jsonl"),
        "programs": repo_path("data", "programs", f"edit_programs_{version}.jsonl"),
        "renders": repo_path("data", "renders", f"render_metadata_{version}.jsonl"),
        "verifier": repo_path("data", "verifier", f"verifier_items_{version}.jsonl"),
        "filters": repo_path("data", "quality", f"filter_scores_{version}.jsonl"),
    }
    missing_files = [str(path.relative_to(root)) for path in paths.values() if not path.exists()]
    if missing_files:
        return {"ok": False, "missing_files": missing_files, "created_at": utc_now()}

    tasks = read_jsonl(paths["tasks"])
    recipes = read_jsonl(paths["recipes"])
    trajectories = read_jsonl(paths["trajectories"])
    programs = read_jsonl(paths["programs"])
    renders = read_jsonl(paths["renders"])
    verifier = read_jsonl(paths["verifier"])
    filters = read_jsonl(paths["filters"])

    errors = []
    task_ids = [task.get("task_id") for task in tasks]
    duplicate_task_ids = [task_id for task_id, c in collections.Counter(task_ids).items() if c > 1]
    if duplicate_task_ids:
        errors.append({"type": "duplicate_task_ids", "items": duplicate_task_ids[:20]})

    task_id_set = set(task_ids)
    for task in tasks:
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in task]
        if missing:
            errors.append({"type": "missing_task_fields", "task_id": task.get("task_id"), "fields": missing})
        source = root / task.get("source_image", "")
        if not source.exists():
            errors.append({"type": "missing_source_image", "task_id": task.get("task_id"), "path": str(source)})
        weights = [x.get("weight", 0) for x in task.get("atomic_checklist", [])]
        if not weights or abs(sum(weights) - 1.0) > 1e-6:
            errors.append({"type": "bad_checklist_weights", "task_id": task.get("task_id"), "sum": sum(weights)})
        if task.get("leakage_tags", {}).get("benchmark_text_exact_match"):
            errors.append({"type": "exact_benchmark_text_match", "task_id": task.get("task_id")})

    for name, rows in [("recipes", recipes), ("trajectories", trajectories), ("programs", programs), ("filters", filters)]:
        unknown = [row.get("task_id") for row in rows if row.get("task_id") not in task_id_set]
        if unknown:
            errors.append({"type": f"unknown_task_id_in_{name}", "items": unknown[:20]})

    missing_render_images = []
    for row in renders:
        image_path = root / row.get("image_path", "")
        if not image_path.exists():
            missing_render_images.append(row.get("image_path"))
    if missing_render_images:
        errors.append({"type": "missing_render_images", "items": missing_render_images[:20]})

    split_counts = dict(collections.Counter(task.get("split") for task in tasks))
    family_counts = dict(collections.Counter(task.get("benchmark_family") for task in tasks))
    domain_counts = dict(collections.Counter(task.get("domain") for task in tasks))
    report = {
        "ok": not errors,
        "created_at": utc_now(),
        "version": version,
        "counts": {
            "tasks": len(tasks),
            "recipes": len(recipes),
            "trajectories": len(trajectories),
            "programs": len(programs),
            "renders": len(renders),
            "verifier_items": len(verifier),
            "filters": len(filters),
        },
        "split_counts": split_counts,
        "family_counts": family_counts,
        "domain_counts": domain_counts,
        "errors": errors[:100],
    }
    write_json(repo_path("reports", "data_quality", f"validation_{version}.json"), report)
    return report


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0")
    args = parser.parse_args(argv)
    report = validate(args.version)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

