#!/usr/bin/env python3
"""Convert accepted v2r teacher trajectories into LLaMA-Factory ShareGPT
multimodal SFT format for training the RISEvolve planner (Qwen2.5-VL).

Reads the append-only source files (safe to read while the harvester appends):
  data/tasks/tasks_<v>.jsonl, data/quality/filter_scores_<v>.jsonl,
  data/trajectories/teacher_trajectories_<v>.jsonl
Builds a frozen snapshot so training is reproducible even as data grows.

Target the planner emits: a short analysis + the region-aware edit program
(JSON), conditioned on the real source image + instruction. (The downstream
image editor consumes the editor_prompt; the planner is what we SFT here.)

Output:
  data/sft_lf/rise_sft_<v>.json         (ShareGPT rows: conversations + images)
  data/sft_lf/dataset_info.json         (LLaMA-Factory dataset registry)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass  # tolerate a partially-written trailing line
    return out


# task_id -> v3 VLM re-annotation (concrete reasoning content), populated by load_reann().
REANN: dict = {}


def reann_quality(row: dict) -> int:
    fields = ("target_scene_description", "analysis", "edit_operations", "knowledge_facts", "editor_prompt")
    return sum(row.get(f) not in (None, "", []) for f in fields)


def reann_is_complete(row: dict | None) -> bool:
    if not row:
        return False
    fields = ("target_scene_description", "analysis", "edit_operations", "editor_prompt")
    return all(row.get(f) not in (None, "", []) for f in fields)


def load_reann(globpat: str) -> int:
    import glob
    n = 0
    for fp in glob.glob(globpat):
        for line in read_jsonl(Path(fp)):
            if line.get("status") == "ok" and line.get("task_id"):
                old = REANN.get(line["task_id"])
                if old is None or reann_quality(line) >= reann_quality(old):
                    REANN[line["task_id"]] = line
                n += 1
    return n


def build_response(task: dict, traj: dict) -> str:
    prog = traj.get("final_edit_program", {}) or {}
    # the first assistant message carries the analysis / thinking
    thought = ""
    for m in traj.get("messages", []):
        if m.get("role") == "assistant" and m.get("content"):
            thought = m["content"]
            break
    # Emit the FULL edit_program (all schema fields). Strip only bookkeeping keys.
    full = {k: v for k, v in prog.items() if k not in ("created_at", "task_id")}

    # v3 merge: if a VLM re-annotation exists, replace the reasoning-bearing fields with the
    # CONCRETE, image-grounded content (target_scene_description/analysis/edit_operations/
    # knowledge_facts/editor_prompt) — the original program had generic boilerplate here that made
    # the SFT planner UNDERPERFORM the raw-instruction baseline on RISE. Keep the other schema
    # fields (source_scene_graph, task_family, preservation/negative_constraints, atomic_checklist,
    # failure_modes_to_watch, reference_images) so the program stays schema-complete.
    ra = REANN.get(task.get("task_id"))
    if ra:
        for f in ("target_scene_description", "edit_operations", "knowledge_facts", "editor_prompt"):
            if ra.get(f) not in (None, "", []):
                full[f] = ra[f]
        if ra.get("analysis"):
            thought = ra["analysis"]
    return (
        f"<analysis>\n{thought}\n</analysis>\n\n"
        f"<edit_program>\n{json.dumps(full, ensure_ascii=False, indent=2)}\n</edit_program>"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2r")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all accepted")
    ap.add_argument("--reann-glob", default=None,
                    help="glob for v3 VLM re-annotation shards (e.g. reports/reann_v3_shard*.jsonl); "
                         "merges concrete reasoning content by task_id")
    ap.add_argument("--require-reann", action="store_true",
                    help="skip accepted rows without a complete ok v3 re-annotation; useful for clean v3 SFT")
    args = ap.parse_args()
    v = args.version
    if args.reann_glob:
        n = load_reann(args.reann_glob)
        print(f"loaded {n} v3 re-annotations from {args.reann_glob}")

    tasks = {t["task_id"]: t for t in read_jsonl(ROOT / "data" / "tasks" / f"tasks_{v}.jsonl")}
    filters = {f["task_id"]: f for f in read_jsonl(ROOT / "data" / "quality" / f"filter_scores_{v}.jsonl")}
    trajs = {t["task_id"]: t for t in read_jsonl(ROOT / "data" / "trajectories" / f"teacher_trajectories_{v}.jsonl")}

    rows = []
    skipped_noimg = 0
    for tid, task in tasks.items():
        if not filters.get(tid, {}).get("labels", {}).get("accept_sft"):
            continue
        traj = trajs.get(tid)
        if not traj:
            continue
        if args.require_reann and not reann_is_complete(REANN.get(tid)):
            continue
        img = ROOT / task["source_image"]
        if not img.exists():
            skipped_noimg += 1
            continue
        instr = task["instruction"]
        rows.append({
            "conversations": [
                {"from": "human", "value": "<image>\n" + instr},
                {"from": "gpt", "value": build_response(task, traj)},
            ],
            "images": [str(img)],
        })
        if args.max_samples and len(rows) >= args.max_samples:
            break

    out_dir = ROOT / "data" / "sft_lf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"rise_sft_{v}.json"
    out_file.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    info_path = out_dir / "dataset_info.json"
    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            info = {}
    info[f"rise_sft_{v}"] = {
        "file_name": f"rise_sft_{v}.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {
            "role_tag": "from", "content_tag": "value",
            "user_tag": "human", "assistant_tag": "gpt",
        },
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} SFT samples -> {out_file.relative_to(ROOT)} "
          f"(skipped_noimg={skipped_noimg}); dataset registered as rise_sft_{v}")


if __name__ == "__main__":
    main()
