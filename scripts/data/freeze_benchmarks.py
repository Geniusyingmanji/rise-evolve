#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from common import normalize_text, repo_path, sha256_text, utc_now, write_json, write_jsonl


BENCHMARK_SOURCES = {
    "rise": {
        "display_name": "RISEBench",
        "url": "https://raw.githubusercontent.com/PhoenixZ810/RISEBench/main/data/data_total.json",
        "raw_file": "data_total.json",
        "schema": ["index", "category", "instruction", "image", "reference"],
        "allowed_use": "evaluation_and_decontamination_only",
    },
    "grade": {
        "display_name": "GRADE",
        "url": "https://huggingface.co/datasets/VisionXLab/GRADE/raw/main/data.json",
        "raw_file": "data.json",
        "schema": ["image_path", "gt", "text", "task_id", "questions", "consistency", "sub_task", "domain"],
        "allowed_use": "evaluation_and_decontamination_only",
    },
    "kris": {
        "display_name": "KRIS-Bench",
        "url": "https://raw.githubusercontent.com/mercurystraw/Kris_Bench/main/README.md",
        "raw_file": "README.md",
        "schema": ["category", "annotation.json", "image", "instruction"],
        "allowed_use": "evaluation_and_decontamination_only",
    },
}


def fetch_url(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "RISEvolve-data-freeze/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def extract_text_items(name: str, raw_bytes: bytes) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if name in {"rise", "grade"}:
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            return items
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if name == "rise":
                for field in ("instruction", "reference"):
                    value = row.get(field)
                    if isinstance(value, str) and value.strip():
                        items.append(
                            {
                                "benchmark": name,
                                "sample_id": str(row.get("index", idx)),
                                "field": field,
                                "text": value,
                            }
                        )
            if name == "grade":
                value = row.get("text")
                if isinstance(value, str) and value.strip():
                    items.append(
                        {
                            "benchmark": name,
                            "sample_id": str(row.get("task_id", idx)),
                            "field": "text",
                            "text": value,
                        }
                    )
                questions = row.get("questions") or []
                for q in questions:
                    if isinstance(q, dict) and isinstance(q.get("question"), str):
                        items.append(
                            {
                                "benchmark": name,
                                "sample_id": str(row.get("task_id", idx)),
                                "field": f"question:{q.get('id', '')}",
                                "text": q["question"],
                            }
                        )
    elif name == "kris":
        text = raw_bytes.decode("utf-8", errors="ignore")
        for line_id, line in enumerate(text.splitlines()):
            line = line.strip()
            if "KRIS" in line or "knowledge" in line.lower() or "reasoning" in line.lower():
                items.append(
                    {
                        "benchmark": name,
                        "sample_id": f"readme:{line_id}",
                        "field": "readme_line",
                        "text": line,
                    }
                )
    return items


def build_fingerprint(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = repo_path("data", "benchmarks")
    text_index = []
    entries = []

    for name, spec in BENCHMARK_SOURCES.items():
        raw_dir = out_root / name / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / spec["raw_file"]
        entry: Dict[str, Any] = {
            "benchmark": spec["display_name"],
            "key": name,
            "source_url": spec["url"],
            "raw_path": str(raw_path.relative_to(repo_path())),
            "schema": spec["schema"],
            "allowed_use": spec["allowed_use"],
            "frozen_at": utc_now(),
            "status": "pending",
        }
        try:
            raw_bytes = fetch_url(spec["url"], timeout=args.timeout)
            raw_path.write_bytes(raw_bytes)
            items = extract_text_items(name, raw_bytes)
            normalized = []
            for item in items:
                norm = normalize_text(item["text"])
                text_index.append(
                    {
                        "benchmark": item["benchmark"],
                        "sample_id": item["sample_id"],
                        "field": item["field"],
                        "normalized_text": norm,
                        "text_hash": sha256_text(norm),
                    }
                )
                normalized.append(norm)
            entry.update(
                {
                    "status": "frozen",
                    "raw_sha256": sha256_text(raw_bytes.decode("utf-8", errors="ignore")),
                    "text_items": len(items),
                    "text_hashes": [sha256_text(x) for x in normalized],
                }
            )
        except Exception as exc:
            entry.update({"status": "unavailable", "error": repr(exc), "text_items": 0, "text_hashes": []})
        entries.append(entry)

    write_jsonl(out_root / "benchmark_text_index.jsonl", text_index)
    fingerprint = {
        "created_at": utc_now(),
        "note": "Benchmark content is frozen for evaluation and decontamination only. Do not use benchmark samples as training data.",
        "benchmarks": entries,
        "text_index_path": "data/benchmarks/benchmark_text_index.jsonl",
        "decontamination": {
            "exact_normalized_text": "reject",
            "semantic_similarity_gt_0_88": "reject_when_embeddings_are_available",
            "image_phash_hamming_le_6": "reject_when_benchmark_images_are_indexed",
        },
    }
    write_json(out_root / "benchmark_fingerprint.json", fingerprint)
    return fingerprint


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    fingerprint = build_fingerprint(args)
    frozen = sum(1 for b in fingerprint["benchmarks"] if b["status"] == "frozen")
    print(f"wrote data/benchmarks/benchmark_fingerprint.json; frozen={frozen}/{len(fingerprint['benchmarks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

