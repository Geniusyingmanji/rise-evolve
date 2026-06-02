#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw

from common import ensure_dir, load_font, repo_path, stable_hash, utc_now, write_json


DEFAULT_FAILURE_TYPES = {
    "benchmark_leak",
    "split_or_license_risk",
    "not_same_pair",
    "different_image_context",
    "before_after_reversed",
    "wrong_edit",
    "missing_edit",
    "under_edit",
    "over_editing",
    "over_edit",
    "identity_drift",
    "background_drift",
    "wrong_region",
    "instruction_mismatch",
    "artifact_or_low_quality",
    "artifact_quality",
    "ambiguous_instruction",
    "unsafe_or_watermarked",
    "unsafe_content",
    "before_after_order_unclear",
    "reasoning_or_knowledge_fail",
    "judge_uncertain",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def rel_or_abs(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_path(path_text)


def image_path(row: Dict[str, Any], key: str) -> Path:
    meta = row.get(key) or {}
    return rel_or_abs(str(meta.get("path") or ""))


def review_id(row: Dict[str, Any]) -> str:
    source_meta = row.get("source_image") or {}
    target_meta = row.get("target_image") or {}
    payload = {
        "source_id": row.get("source_id"),
        "row_idx": row.get("row_idx"),
        "instruction": row.get("instruction"),
        "source_sha": source_meta.get("sha256"),
        "target_sha": target_meta.get("sha256"),
    }
    return stable_hash(payload, n=20)


def stratified_sample(rows: List[Dict[str, Any]], sample_size: int, seed: int, field: str) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    order = sorted(groups, key=lambda key: (-len(groups[key]), key))
    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < min(sample_size, len(rows)) and order:
        key = order[cursor % len(order)]
        group = groups[key]
        if group:
            selected.append(group.pop())
        order = [name for name in order if groups[name]]
        cursor += 1
    rng.shuffle(selected)
    return selected


def fit_image(path: Path, size: Tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def comparison_image(row: Dict[str, Any], panel_size: int = 640) -> Image.Image:
    header_h = 52
    footer_h = 0
    width = panel_size * 2
    height = header_h + panel_size + footer_h
    canvas = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24, bold=True)
    small_font = load_font(16)
    draw.rectangle((0, 0, width - 1, height - 1), outline="#cbd5e1", width=2)
    draw.text((18, 14), "SOURCE", font=title_font, fill="#111827")
    draw.text((panel_size + 18, 14), "TARGET", font=title_font, fill="#111827")
    source = fit_image(image_path(row, "source_image"), (panel_size, panel_size))
    target = fit_image(image_path(row, "target_image"), (panel_size, panel_size))
    canvas.paste(source, (0, header_h))
    canvas.paste(target, (panel_size, header_h))
    draw.line((panel_size, 0, panel_size, height), fill="#cbd5e1", width=2)
    rid = review_id(row)
    draw.text((width - 230, 20), rid, font=small_font, fill="#64748b")
    return canvas


def image_data_url(img: Image.Image, quality: int = 86) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def make_contact_sheet(rows: List[Dict[str, Any]], out_path: Path, max_items: int = 40) -> None:
    if not rows:
        return
    rows = rows[:max_items]
    thumb = (180, 180)
    label_h = 56
    pair_w = thumb[0] * 2
    cell_h = thumb[1] + label_h
    cols = 2
    rows_n = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * pair_w, rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(12)
    for idx, row in enumerate(rows):
        col = idx % cols
        rr = idx // cols
        x0 = col * pair_w
        y0 = rr * cell_h
        try:
            src = fit_image(image_path(row, "source_image"), thumb)
            tgt = fit_image(image_path(row, "target_image"), thumb)
            sheet.paste(src, (x0, y0))
            sheet.paste(tgt, (x0 + thumb[0], y0))
        except Exception as exc:
            draw.text((x0 + 8, y0 + 8), f"image error: {exc}", font=font, fill="#dc2626")
        label = f"{idx:03d} {row.get('source_id') or 'unknown'} {review_id(row)}"
        instruction = str(row.get("instruction") or "")[:90]
        draw.rectangle((x0, y0, x0 + pair_w - 1, y0 + cell_h - 1), outline="#cbd5e1", width=1)
        draw.text((x0 + 6, y0 + thumb[1] + 6), label, font=font, fill="#111827")
        draw.text((x0 + 6, y0 + thumb[1] + 24), instruction, font=font, fill="#475569")
    ensure_dir(out_path.parent)
    sheet.save(out_path)


def existing_review_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        if row.get("review_id"):
            ids.add(row["review_id"])
    return ids


def detect_openai_compatible(base_url: Optional[str], model: Optional[str], timeout: int) -> Optional[Dict[str, str]]:
    if base_url:
        return {"type": "openai_compatible", "base_url": base_url.rstrip("/"), "model": model or "gpt-4.1-mini"}
    env_base = os.getenv("OPENAI_BASE_URL")
    if env_base:
        return {
            "type": "openai_compatible",
            "base_url": env_base.rstrip("/"),
            "model": model or os.getenv("VLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini",
        }
    local_base = "http://127.0.0.1:18000/v1"
    try:
        response = requests.get(f"{local_base}/models", timeout=min(timeout, 5))
        if response.ok:
            payload = response.json()
            models = [item.get("id") for item in payload.get("data", []) if item.get("id")]
            chosen = model or os.getenv("VLM_MODEL") or ("gpt-5.5" if "gpt-5.5" in models else (models[0] if models else "gpt-5.5"))
            return {"type": "openai_compatible", "base_url": local_base, "model": chosen}
    except Exception:
        pass
    if os.getenv("OPENAI_API_KEY"):
        return {
            "type": "openai_compatible",
            "base_url": "https://api.openai.com/v1",
            "model": model or os.getenv("VLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini",
        }
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if endpoint and deployment and os.getenv("AZURE_OPENAI_API_KEY"):
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2025-04-01-preview"
        return {
            "type": "azure_openai",
            "endpoint": endpoint.rstrip("/"),
            "deployment": deployment,
            "api_version": api_version,
            "model": model or deployment,
        }
    return None


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def coerce_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def gate_decision(vlm: Dict[str, Any]) -> str:
    critical_failures = {
        "benchmark_leak",
        "split_or_license_risk",
        "not_same_pair",
        "different_image_context",
        "before_after_reversed",
        "wrong_edit",
        "missing_edit",
        "wrong_region",
        "instruction_mismatch",
        "unsafe_content",
        "unsafe_or_watermarked",
    }
    failures = {str(item).strip().lower() for item in (vlm.get("failure_types") or [])}
    if failures & critical_failures:
        return "reject"
    same = coerce_score(vlm.get("same_image_context"))
    faithful = coerce_score(vlm.get("edit_faithfulness"))
    preservation = coerce_score(vlm.get("preservation"))
    artifact = coerce_score(vlm.get("artifact_severity"))
    clarity = coerce_score(vlm.get("instruction_clarity"))
    overall = coerce_score(vlm.get("overall"))
    if same is not None and same <= 2:
        return "reject"
    if faithful is not None and faithful <= 2:
        return "reject"
    if clarity is not None and clarity <= 1:
        return "reject"
    if artifact is not None and artifact >= 4:
        return "reject"
    weak_scores = [x for x in [same, faithful, preservation, clarity, overall] if x is not None and x < 4]
    if weak_scores or (artifact is not None and artifact >= 3):
        return "review"
    return "accept"


def review_prompt(row: Dict[str, Any]) -> str:
    failure_list = ", ".join(sorted(DEFAULT_FAILURE_TYPES))
    return f"""You are auditing a real before/after image-editing training pair.

Instruction:
{row.get('instruction') or ''}

Dataset source: {row.get('source_id') or 'unknown'}
Edit type: {row.get('edit_type') or 'unknown'}

The image shows SOURCE on the left and TARGET on the right. Judge whether the target is a valid edited version of the source for the instruction. Do not reward pairs where the target is a different image, a puzzle-like reconstruction, a reference/depth/mask-dependent output, a severe over-edit, or a weak/ambiguous instruction.

Return only JSON with this exact shape:
{{
  "same_image_context": 0-5,
  "edit_faithfulness": 0-5,
  "preservation": 0-5,
  "artifact_severity": 0-5,
  "instruction_clarity": 0-5,
  "overall": 0-5,
  "decision": "accept|review|reject",
  "failure_types": ["one or more of: {failure_list}"],
  "reasons": ["short concrete reasons"]
}}

Scoring: 5 is best except artifact_severity where 0 is best and 5 is severe."""


def call_openai_compatible(provider: Dict[str, str], row: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    img = comparison_image(row)
    payload = {
        "model": provider["model"],
        "messages": [
            {
                "role": "system",
                "content": "You are a strict VLM data-quality critic for image-editing training data. Return compact JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": review_prompt(row)},
                    {"type": "image_url", "image_url": {"url": image_data_url(img)}},
                ],
            },
        ],
        "max_completion_tokens": 600,
    }
    response = requests.post(
        f"{provider['base_url'].rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    text = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    parsed = extract_json_object(text)
    parsed["_raw_response"] = text[:2000]
    return parsed


def call_azure_openai(provider: Dict[str, str], row: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "api-key": os.getenv("AZURE_OPENAI_API_KEY", "")}
    img = comparison_image(row)
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a strict VLM data-quality critic for image-editing training data. Return compact JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": review_prompt(row)},
                    {"type": "image_url", "image_url": {"url": image_data_url(img)}},
                ],
            },
        ],
        "max_completion_tokens": 600,
    }
    url = (
        f"{provider['endpoint']}/openai/deployments/{provider['deployment']}"
        f"/chat/completions?api-version={provider['api_version']}"
    )
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    text = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    parsed = extract_json_object(text)
    parsed["_raw_response"] = text[:2000]
    return parsed


def call_provider(provider: Dict[str, str], row: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    if provider["type"] == "openai_compatible":
        return call_openai_compatible(provider, row, timeout)
    if provider["type"] == "azure_openai":
        return call_azure_openai(provider, row, timeout)
    raise ValueError(f"unsupported provider type: {provider['type']}")


def summarize(review_path: Path, selected_count: int, provider: Optional[Dict[str, str]], sheet_path: Path) -> Dict[str, Any]:
    rows = read_jsonl(review_path) if review_path.exists() else []
    decisions: Dict[str, int] = {}
    by_source: Dict[str, Dict[str, int]] = {}
    failure_types: Dict[str, int] = {}
    score_fields = [
        "same_image_context",
        "edit_faithfulness",
        "preservation",
        "artifact_severity",
        "instruction_clarity",
        "overall",
    ]
    scores: Dict[str, List[float]] = {field: [] for field in score_fields}
    for row in rows:
        decision = row.get("gate_decision") or row.get("vlm_decision") or row.get("status") or "unknown"
        decisions[decision] = decisions.get(decision, 0) + 1
        source = str(row.get("source_id") or "unknown")
        by_source.setdefault(source, {})
        by_source[source][decision] = by_source[source].get(decision, 0) + 1
        for failure in row.get("failure_types") or []:
            failure_types[str(failure)] = failure_types.get(str(failure), 0) + 1
        for field in score_fields:
            value = coerce_score(row.get(field))
            if value is not None:
                scores[field].append(value)
    means = {field: round(statistics.mean(vals), 3) for field, vals in scores.items() if vals}
    return {
        "review_path": str(review_path.relative_to(repo_path())),
        "contact_sheet": str(sheet_path.relative_to(repo_path())) if sheet_path.exists() else None,
        "selected_count": selected_count,
        "reviewed_count": len(rows),
        "provider": provider or {"type": "unavailable"},
        "decisions": dict(sorted(decisions.items())),
        "by_source": by_source,
        "failure_types": dict(sorted(failure_types.items(), key=lambda kv: (-kv[1], kv[0]))),
        "score_means": means,
        "updated_at": utc_now(),
    }


def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": row.get("source_id"),
        "dataset_id": row.get("dataset_id"),
        "split": row.get("split"),
        "row_idx": row.get("row_idx"),
        "instruction": row.get("instruction"),
        "edit_type": row.get("edit_type"),
        "source_image": row.get("source_image"),
        "target_image": row.get("target_image"),
        "license": row.get("license"),
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run VLM spot checks for real image-edit pair candidates.")
    parser.add_argument("--input", required=True, help="Accepted candidate JSONL manifest.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix. Defaults to manifest stem plus sample metadata.")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=6202)
    parser.add_argument("--stratify-by", default="source_id")
    parser.add_argument("--provider", choices=["auto", "off"], default="auto")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, for example http://127.0.0.1:18000/v1.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--sheet-items", type=int, default=40)
    args = parser.parse_args(argv)

    input_path = rel_or_abs(args.input)
    rows = read_jsonl(input_path)
    selected = stratified_sample(rows, args.sample_size, args.seed, args.stratify_by)
    manifest_name = input_path.stem.replace("real_edit_pairs_candidate_", "")
    output_prefix = args.output_prefix or f"{manifest_name}_s{len(selected)}_seed{args.seed}"
    review_path = repo_path("reports", "data_sources", f"vlm_review_{output_prefix}.jsonl")
    summary_path = repo_path("reports", "data_sources", f"vlm_review_summary_{output_prefix}.json")
    sheet_path = repo_path("reports", "data_sources", f"vlm_review_sheet_{output_prefix}.png")

    make_contact_sheet(selected, sheet_path, max_items=args.sheet_items)
    done = existing_review_ids(review_path)
    provider = None if args.provider == "off" else detect_openai_compatible(args.base_url, args.model, args.timeout)

    if provider is None:
        for row in selected:
            rid = review_id(row)
            if rid in done:
                continue
            append_jsonl(
                review_path,
                {
                    "review_id": rid,
                    "status": "provider_unavailable",
                    "reviewed_at": utc_now(),
                    **compact_row(row),
                },
            )
        summary = summarize(review_path, len(selected), provider, sheet_path)
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    for idx, row in enumerate(selected, start=1):
        rid = review_id(row)
        if rid in done:
            continue
        record = {
            "review_id": rid,
            "status": "reviewed",
            "reviewed_at": utc_now(),
            "provider": provider,
            "sample_index": idx,
            **compact_row(row),
        }
        try:
            vlm = call_provider(provider, row, args.timeout)
            failures = vlm.get("failure_types") or []
            if isinstance(failures, str):
                failures = [failures]
            record.update(
                {
                    "same_image_context": coerce_score(vlm.get("same_image_context")),
                    "edit_faithfulness": coerce_score(vlm.get("edit_faithfulness")),
                    "preservation": coerce_score(vlm.get("preservation")),
                    "artifact_severity": coerce_score(vlm.get("artifact_severity")),
                    "instruction_clarity": coerce_score(vlm.get("instruction_clarity")),
                    "overall": coerce_score(vlm.get("overall")),
                    "vlm_decision": str(vlm.get("decision") or "review").lower(),
                    "gate_decision": gate_decision(vlm),
                    "failure_types": failures,
                    "reasons": vlm.get("reasons") or [],
                    "raw_response": vlm.get("_raw_response", ""),
                }
            )
        except Exception as exc:
            record.update({"status": "provider_error", "error": str(exc)[:1000], "gate_decision": "review"})
        append_jsonl(review_path, record)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary = summarize(review_path, len(selected), provider, sheet_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
