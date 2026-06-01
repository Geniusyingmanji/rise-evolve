#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from common import ensure_dir, normalize_text, repo_path, utc_now, write_json, write_jsonl


TEXT_BLOCKLIST = {
    "nude",
    "naked",
    "nsfw",
    "porn",
    "sexual",
    "lingerie",
    "blood",
    "gore",
    "weapon",
    "gun",
    "bomb",
    "terrorist",
    "attack",
    "explosion",
    "explode",
    "fire",
    "burn",
    "disaster",
    "violent",
    "violence",
    "tsunami",
    "bare chest",
}

SAFE_LICENSE_TOKENS = {
    "cc-by-4.0",
    "cc by 4.0",
    "mit",
    "apache-2.0",
    "apache 2.0",
}

EVAL_SPLIT_TOKENS = {"test", "eval", "evaluation", "dev", "validation", "val", "benchmark"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_cmd(cmd: List[str], log_path: Path, timeout: int | None = None) -> Tuple[int, str]:
    ensure_dir(log_path.parent)
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_path()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed = round(time.time() - started, 3)
    block = {
        "time": utc_now(),
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "output_tail": proc.stdout[-8000:],
    }
    append_jsonl(log_path, block)
    return proc.returncode, proc.stdout


def pair_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    src = ((row.get("source_image") or {}).get("sha256") or "").strip()
    tgt = ((row.get("target_image") or {}).get("sha256") or "").strip()
    inst = normalize_text(str(row.get("instruction") or ""))
    return src, tgt, inst


def build_seen_keys(paths: Iterable[Path]) -> set:
    seen = set()
    for path in paths:
        for row in read_jsonl(path):
            key = pair_key(row)
            if key[0] and key[1] and key[2]:
                seen.add(key)
    return seen


def candidate_existing_paths(prefix: str, include_all_samples: bool) -> List[Path]:
    paths = [
        repo_path("data", "sources", f"real_edit_pairs_candidate_{prefix}.jsonl"),
    ]
    if include_all_samples:
        paths.extend(sorted(repo_path("data", "sources").glob("real_edit_pairs_sample_*.jsonl")))
    return paths


def text_is_safe(row: Dict[str, Any]) -> bool:
    text = json.dumps(
        {
            "instruction": row.get("instruction"),
            "source_caption": row.get("source_caption"),
            "target_caption": row.get("target_caption"),
            "edit_type": row.get("edit_type"),
        },
        ensure_ascii=False,
    ).lower()
    return not any(token in text for token in TEXT_BLOCKLIST)


def image_ok(meta: Dict[str, Any], min_short_side: int, max_aspect: float) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    short = min(width, height)
    long = max(width, height)
    aspect = long / short if short else 999.0
    path = repo_path(str(meta.get("path") or ""))
    if not path.exists():
        reasons.append("missing_image_file")
    if short < min_short_side:
        reasons.append(f"short_side_lt_{min_short_side}")
    if aspect > max_aspect:
        reasons.append(f"aspect_gt_{max_aspect}")
    if int(meta.get("bytes") or 0) < 8192:
        reasons.append("file_too_small")
    return not reasons, reasons


def license_ok(row: Dict[str, Any], allow_research_only: bool) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    license_text = str(row.get("license") or "").lower()
    if "cc-by-nc" in license_text or "non-commercial" in license_text:
        if not allow_research_only:
            reasons.append("non_commercial_license")
        return not reasons, reasons
    if not any(token in license_text for token in SAFE_LICENSE_TOKENS):
        reasons.append("license_not_in_allowlist")
    return not reasons, reasons


def split_ok(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    split = str(row.get("split") or "").lower()
    if any(token == split or token in split for token in EVAL_SPLIT_TOKENS):
        return False, [f"eval_like_split:{split}"]
    return True, []


def filter_pair(
    row: Dict[str, Any],
    seen: set,
    min_short_side: int,
    max_aspect: float,
    allow_research_only: bool,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    key = pair_key(row)
    if not key[0] or not key[1] or not key[2]:
        reasons.append("missing_key_fields")
    elif key in seen:
        reasons.append("duplicate_pair_or_instruction")
    if not text_is_safe(row):
        reasons.append("unsafe_text")
    leakage = row.get("leakage_tags") or {}
    if leakage.get("benchmark_text_exact_match"):
        reasons.append("exact_benchmark_text_match")
    quality = row.get("quality_tags") or {}
    if quality and not quality.get("ok", True):
        reasons.append("upstream_quality_reject")
    for image_key in ["source_image", "target_image"]:
        ok, image_reasons = image_ok(row.get(image_key) or {}, min_short_side, max_aspect)
        if not ok:
            reasons.extend(f"{image_key}:{reason}" for reason in image_reasons)
    ok, license_reasons = license_ok(row, allow_research_only)
    if not ok:
        reasons.extend(license_reasons)
    ok, split_reasons = split_ok(row)
    if not ok:
        reasons.extend(split_reasons)
    return not reasons, reasons


def merge_cycle(
    version: str,
    prefix: str,
    seen: set,
    min_short_side: int,
    max_aspect: float,
    allow_research_only: bool,
) -> Dict[str, Any]:
    sample_path = repo_path("data", "sources", f"real_edit_pairs_sample_{version}.jsonl")
    candidate_path = repo_path("data", "sources", f"real_edit_pairs_candidate_{prefix}.jsonl")
    reject_path = repo_path("data", "sources", f"real_edit_pairs_rejected_{prefix}.jsonl")
    rows = read_jsonl(sample_path)
    accepted: List[Dict[str, Any]] = []
    rejected = 0
    reason_counts: Dict[str, int] = {}
    for row in rows:
        ok, reasons = filter_pair(row, seen, min_short_side, max_aspect, allow_research_only)
        row["long_collection"] = {
            "prefix": prefix,
            "cycle_version": version,
            "checked_at": utc_now(),
            "quality_gate": "accepted" if ok else "rejected",
            "reject_reasons": reasons,
        }
        if ok:
            seen.add(pair_key(row))
            accepted.append(row)
        else:
            rejected += 1
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            append_jsonl(reject_path, row)
    if accepted:
        ensure_dir(candidate_path.parent)
        with candidate_path.open("a", encoding="utf-8") as f:
            for row in accepted:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "version": version,
        "sample_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": rejected,
        "reject_reasons": dict(sorted(reason_counts.items())),
        "candidate_path": str(candidate_path.relative_to(repo_path())),
        "reject_path": str(reject_path.relative_to(repo_path())),
    }


def decontam_candidate(prefix: str, log_path: Path) -> Dict[str, Any]:
    candidate = repo_path("data", "sources", f"real_edit_pairs_candidate_{prefix}.jsonl")
    output = repo_path("reports", "data_sources", f"decontamination_{prefix}.json")
    if not candidate.exists():
        return {"status": "skipped_no_candidate_file"}
    cmd = [
        sys.executable,
        "scripts/eval/check_decontamination.py",
        "--benchmarks",
        "data/benchmarks",
        "--train",
        str(candidate.relative_to(repo_path())),
        "--output",
        str(output.relative_to(repo_path())),
        "--fail-on",
        "none",
    ]
    code, _ = run_cmd(cmd, log_path)
    result = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    result["returncode"] = code
    result["output"] = str(output.relative_to(repo_path()))
    return result


def write_status(prefix: str, status: Dict[str, Any]) -> None:
    write_json(repo_path("reports", "data_sources", f"long_collection_status_{prefix}.json"), status)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a long, quality-gated real image-edit data collection job.")
    parser.add_argument("--prefix", default=None, help="Stable output prefix. Defaults to v2_long_<UTC timestamp>.")
    parser.add_argument("--duration-hours", type=float, default=9.0)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--max-accepted", type=int, default=1000)
    parser.add_argument("--hf-per-source", type=int, default=35)
    parser.add_argument("--wiki-per-query", type=int, default=0)
    parser.add_argument("--include-wikimedia", action="store_true")
    parser.add_argument("--pause-seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--min-short-side", type=int, default=384)
    parser.add_argument("--max-aspect", type=float, default=2.5)
    parser.add_argument("--allow-research-only", action="store_true")
    parser.add_argument("--dedupe-existing", action="store_true", default=True)
    args = parser.parse_args(argv)

    prefix = args.prefix or "v2_long_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_path = repo_path("logs", "data_collection", f"{prefix}.jsonl")
    status: Dict[str, Any] = {
        "prefix": prefix,
        "pid": os.getpid(),
        "started_at": utc_now(),
        "duration_hours": args.duration_hours,
        "max_cycles": args.max_cycles,
        "max_accepted": args.max_accepted,
        "hf_per_source": args.hf_per_source,
        "wiki_per_query": args.wiki_per_query if args.include_wikimedia else 0,
        "quality_gates": {
            "min_short_side": args.min_short_side,
            "max_aspect": args.max_aspect,
            "allow_research_only": args.allow_research_only,
            "dedupe_existing_samples": args.dedupe_existing,
        },
        "cycles": [],
        "accepted_total": 0,
        "state": "running",
        "log": str(log_path.relative_to(repo_path())),
    }
    write_status(prefix, status)
    append_jsonl(log_path, {"time": utc_now(), "event": "start", "status": status})

    seen = build_seen_keys(candidate_existing_paths(prefix, args.dedupe_existing))
    end_time = time.time() + args.duration_hours * 3600
    cycle = 0
    while time.time() < end_time:
        if args.max_cycles is not None and cycle >= args.max_cycles:
            status["state"] = "stopped_max_cycles"
            break
        if status["accepted_total"] >= args.max_accepted:
            status["state"] = "stopped_max_accepted"
            break

        cycle_seed = args.seed + cycle * 7919
        version = f"{prefix}_c{cycle:04d}"
        cmd = [
            sys.executable,
            "scripts/data/collect_real_edit_sources.py",
            "--version",
            version,
            "--hf-per-source",
            str(args.hf_per_source),
            "--randomize",
            "--seed",
            str(cycle_seed),
        ]
        if args.include_wikimedia:
            cmd.extend(["--wiki-per-query", str(args.wiki_per_query)])
        else:
            cmd.append("--skip-wikimedia")

        cycle_status: Dict[str, Any] = {
            "cycle": cycle,
            "version": version,
            "seed": cycle_seed,
            "started_at": utc_now(),
        }
        append_jsonl(log_path, {"time": utc_now(), "event": "cycle_start", "cycle": cycle_status})
        code, _ = run_cmd(cmd, log_path, timeout=3600)
        cycle_status["collect_returncode"] = code
        if code == 0:
            audit_cmd = [
                sys.executable,
                "scripts/data/audit_real_sources.py",
                "--version",
                version,
                "--sheet-limit",
                "30",
            ]
            audit_code, _ = run_cmd(audit_cmd, log_path, timeout=600)
            cycle_status["audit_returncode"] = audit_code
            merge = merge_cycle(
                version=version,
                prefix=prefix,
                seen=seen,
                min_short_side=args.min_short_side,
                max_aspect=args.max_aspect,
                allow_research_only=args.allow_research_only,
            )
            cycle_status["merge"] = merge
            status["accepted_total"] += int(merge["accepted_rows"])
            status["last_decontamination"] = decontam_candidate(prefix, log_path)
        else:
            cycle_status["merge"] = {"accepted_rows": 0, "rejected_rows": 0, "error": "collect_failed"}
        cycle_status["finished_at"] = utc_now()
        status["cycles"].append(cycle_status)
        status["updated_at"] = utc_now()
        write_status(prefix, status)
        append_jsonl(log_path, {"time": utc_now(), "event": "cycle_finish", "cycle": cycle_status})
        cycle += 1

        if time.time() < end_time and status["accepted_total"] < args.max_accepted:
            time.sleep(max(0, args.pause_seconds))

    if status.get("state") == "running":
        status["state"] = "completed_duration"
    status["finished_at"] = utc_now()
    write_status(prefix, status)
    append_jsonl(log_path, {"time": utc_now(), "event": "finish", "status": status})
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
