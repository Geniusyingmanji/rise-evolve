#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from common import ensure_dir, load_font, repo_path, text_size, utc_now, write_json, write_jsonl


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def stratified_sample(tasks: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_family: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for task in tasks:
        by_family[task["benchmark_family"]].append(task)
    sampled: List[Dict[str, Any]] = []
    for family, rows in sorted(by_family.items()):
        k = max(1, round(sample_size * len(rows) / len(tasks)))
        rng.shuffle(rows)
        sampled.extend(rows[:k])
    rng.shuffle(sampled)
    return sampled[:sample_size]


def make_contact_sheets(version: str, sample: List[Dict[str, Any]], per_sheet: int = 24) -> List[str]:
    out_dir = repo_path("reports", "data_quality", "review_sheets")
    ensure_dir(out_dir)
    source_root = repo_path()
    font = load_font(13)
    font_bold = load_font(15, bold=True)
    paths = []
    thumb = 150
    gap = 10
    label_h = 88
    cols = 2
    rows_per_sheet = max(1, per_sheet // cols)
    panel_w = thumb * 3 + gap * 4
    panel_h = thumb + label_h
    sheet_w = cols * panel_w
    sheet_h = rows_per_sheet * panel_h
    render_lookup = build_render_lookup(version)

    for sheet_idx in range((len(sample) + per_sheet - 1) // per_sheet):
        batch = sample[sheet_idx * per_sheet : (sheet_idx + 1) * per_sheet]
        sheet = Image.new("RGB", (sheet_w, sheet_h), "#f8fafc")
        draw = ImageDraw.Draw(sheet)
        for i, task in enumerate(batch):
            col = i % cols
            row = i // cols
            x0 = col * panel_w
            y0 = row * panel_h
            task_id = task["task_id"]
            image_paths = [
                task["source_image"],
                render_lookup[task_id]["teacher_render"],
                render_lookup[task_id]["negative_render"],
            ]
            labels = ["source", "teacher", "negative"]
            for j, image_path in enumerate(image_paths):
                img = Image.open(source_root / image_path).convert("RGB")
                img.thumbnail((thumb, thumb))
                px = x0 + gap + j * (thumb + gap)
                py = y0 + gap
                draw.rectangle((px - 1, py - 1, px + thumb + 1, py + thumb + 1), outline="#cbd5e1")
                sheet.paste(img, (px, py))
                draw.text((px, py + thumb + 2), labels[j], font=font, fill="#334155")
            text_y = y0 + thumb + 28
            draw.text((x0 + gap, text_y), task_id, font=font_bold, fill="#111827")
            draw.text((x0 + gap, text_y + 19), f"{task['benchmark_family']} / {task['sub_family']}", font=font, fill="#475569")
            draw.text((x0 + gap, text_y + 38), short(task["instruction"], 86), font=font, fill="#334155")
        out_path = out_dir / f"review_sheet_{version}_{sheet_idx:03d}.png"
        sheet.save(out_path)
        paths.append(str(out_path.relative_to(repo_path())))
    return paths


def build_render_lookup(version: str) -> Dict[str, Dict[str, str]]:
    rows = read_jsonl(repo_path("data", "renders", f"render_metadata_{version}.jsonl"))
    lookup: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    for row in rows:
        lookup[row["task_id"]][row["render_type"]] = row["image_path"]
    return lookup


def audit(version: str, sample_size: int, seed: int) -> Dict[str, Any]:
    root = repo_path()
    tasks = read_jsonl(root / "data" / "tasks" / f"tasks_{version}.jsonl")
    render_lookup = build_render_lookup(version)
    source_hashes = set()
    teacher_hashes = set()
    negative_hashes = set()
    missing_renders = []
    for task in tasks:
        source_hashes.add(file_sha(root / task["source_image"]))
        renders = render_lookup.get(task["task_id"], {})
        for render_type, target_set in [
            ("teacher_render", teacher_hashes),
            ("negative_render", negative_hashes),
        ]:
            path = renders.get(render_type)
            if not path:
                missing_renders.append({"task_id": task["task_id"], "render_type": render_type})
                continue
            target_set.add(file_sha(root / path))

    sample = stratified_sample(tasks, sample_size, seed)
    sample_rows = [
        {
            "task_id": task["task_id"],
            "benchmark_family": task["benchmark_family"],
            "task_family": task["task_family"],
            "sub_family": task["sub_family"],
            "domain": task["domain"],
            "split": task["split"],
            "source_image": task["source_image"],
            "teacher_image": render_lookup[task["task_id"]]["teacher_render"],
            "negative_image": render_lookup[task["task_id"]]["negative_render"],
            "instruction": task["instruction"],
            "expected_target": task["expected_target"],
            "atomic_checklist": task["atomic_checklist"],
            "benchmark_alignment": task.get("benchmark_alignment"),
        }
        for task in sample
    ]
    write_jsonl(repo_path("reports", "data_quality", f"review_sample_{version}.jsonl"), sample_rows)
    sheet_paths = make_contact_sheets(version, sample)

    report = {
        "ok": not missing_renders,
        "created_at": utc_now(),
        "version": version,
        "total_tasks": len(tasks),
        "unique_instructions": len(set(task["instruction"] for task in tasks)),
        "unique_source_file_sha": len(source_hashes),
        "unique_teacher_file_sha": len(teacher_hashes),
        "unique_negative_file_sha": len(negative_hashes),
        "benchmark_alignment_coverage": sum(1 for task in tasks if task.get("benchmark_alignment")),
        "exact_benchmark_text_rejects": sum(1 for task in tasks if task.get("leakage_tags", {}).get("benchmark_text_exact_match")),
        "missing_renders": missing_renders[:20],
        "review_sample_path": f"reports/data_quality/review_sample_{version}.jsonl",
        "review_sheet_paths": sheet_paths,
    }
    write_json(repo_path("reports", "data_quality", f"audit_{version}.json"), report)
    return report


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--sample-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=531)
    args = parser.parse_args(argv)
    report = audit(args.version, args.sample_size, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

