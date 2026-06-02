#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from PIL import Image

from common import (
    ROOT,
    average_hash,
    ensure_dir,
    normalize_text,
    repo_path,
    utc_now,
    write_json,
    write_jsonl,
)


USER_AGENT = "RISEvolve-real-data-collector/0.1 (https://github.com/Geniusyingmanji/rise-evolve)"
HF_API = "https://huggingface.co/api/datasets"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_FIRST_ROWS_API = "https://datasets-server.huggingface.co/first-rows"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


CURATED_HF_SOURCES: List[Dict[str, Any]] = [
    {
        "source_id": "magicbrush_train",
        "dataset_id": "osunlp/MagicBrush",
        "url": "https://huggingface.co/datasets/osunlp/MagicBrush",
        "license": "cc-by-4.0",
        "split": "train",
        "config": "default",
        "schema": "magicbrush",
        "status": "usable_train",
        "recommended_use": ["SFT", "reward", "editor_pair_bootstrap"],
        "notes": "Real source/target image-editing pairs with natural-language instructions and masks. Use train split only.",
    },
    {
        "source_id": "imagenhub_filtered",
        "dataset_id": "ImagenHub/Text_Guided_Image_Editing",
        "url": "https://huggingface.co/datasets/ImagenHub/Text_Guided_Image_Editing",
        "license": "cc-by-4.0",
        "split": "filtered",
        "config": "default",
        "schema": "imagenhub",
        "status": "usable_train_like",
        "recommended_use": ["reward", "editor_pair_bootstrap"],
        "notes": "Text-guided editing pairs with captions. Avoid dev split; use filtered/extra only for training candidates.",
    },
    {
        "source_id": "anyedit_train",
        "dataset_id": "Bin1117/AnyEdit",
        "url": "https://huggingface.co/datasets/Bin1117/AnyEdit",
        "license": "cc-by-4.0",
        "split": "train",
        "config": "default",
        "schema": "anyedit",
        "status": "usable_train_with_filtering",
        "recommended_use": ["SFT", "reward", "multimodal_edit_coverage"],
        "notes": "Large multimodal editing dataset. Filter unsafe content, visual-input-dependent rows, and low-quality instructions before training.",
    },
    {
        "source_id": "omniedit_train",
        "dataset_id": "TIGER-Lab/OmniEdit-Filtered-1.2M",
        "url": "https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M",
        "license": "mit",
        "split": "train",
        "config": "default",
        "schema": "omniedit",
        "status": "usable_train_with_filtering",
        "recommended_use": ["reward", "editor_pair_bootstrap", "edit_type_coverage"],
        "notes": "Large filtered source/edited pairs with quality scores. Apply safety and benchmark decontamination filters.",
    },
    {
        "source_id": "anyedit_thinking_train",
        "dataset_id": "Tangc03/anyedit_top2000_thinking",
        "url": "https://huggingface.co/datasets/Tangc03/anyedit_top2000_thinking",
        "license": "apache-2.0",
        "split": "train",
        "config": "default",
        "schema": "anyedit_thinking",
        "status": "reasoning_trace_candidate_not_default_pair_sample",
        "recommended_use": ["reasoning_edit_bootstrap", "critic_rationale_candidate"],
        "notes": "Reasoning-heavy AnyEdit subset with input/output images and thinking traces. Excluded from default long pair sampling until VLM pair-alignment checks verify same-image edit consistency.",
    },
    {
        "source_id": "hq_edit",
        "dataset_id": "UCSC-VLAA/HQ-Edit",
        "url": "https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit",
        "license": "cc-by-nc-4.0",
        "split": "train",
        "config": "default",
        "schema": "hq_edit",
        "status": "research_only_needs_sampling_adapter",
        "recommended_use": ["reward", "editor_pair_bootstrap"],
        "notes": "High-quality image-edit pairs, but non-commercial license. The dataset server currently does not expose rows reliably; use HF parquet shards with a dedicated adapter.",
    },
    {
        "source_id": "omniedit_got",
        "dataset_id": "LucasFang/OmniEdit-GoT",
        "url": "https://huggingface.co/datasets/LucasFang/OmniEdit-GoT",
        "license": "mit",
        "split": "train",
        "config": "default",
        "schema": "omniedit_got_path_metadata",
        "status": "metadata_only_needs_path_adapter",
        "recommended_use": ["region_reasoning", "edit_program_bootstrap"],
        "notes": "Contains grounding-oriented thought traces and image paths, but not direct Image columns through the dataset viewer. Needs a parquet/path adapter before image download.",
    },
    {
        "source_id": "seed_edit_openimages",
        "dataset_id": "AILab-CVC/SEED-Data-Edit-Part1-Openimages",
        "url": "https://huggingface.co/datasets/AILab-CVC/SEED-Data-Edit-Part1-Openimages",
        "license": "cc-by-nc-4.0",
        "split": "annotation_jsonl",
        "config": "raw",
        "schema": "seed_edit_annotations",
        "status": "research_only_metadata_first",
        "recommended_use": ["source_pool", "instruction_seed"],
        "notes": "Large edit annotations over OpenImages. Use metadata first; verify image license and avoid test/eval subsets.",
    },
    {
        "source_id": "seed_edit_unsplash",
        "dataset_id": "AILab-CVC/SEED-Data-Edit-Part1-Unsplash",
        "url": "https://huggingface.co/datasets/AILab-CVC/SEED-Data-Edit-Part1-Unsplash",
        "license": "cc-by-nc-4.0",
        "split": "annotation_jsonl",
        "config": "raw",
        "schema": "seed_edit_annotations",
        "status": "research_only_metadata_first",
        "recommended_use": ["source_pool", "instruction_seed"],
        "notes": "Large edit annotations over Unsplash. Check Unsplash terms and retain provenance.",
    },
    {
        "source_id": "instructpix2pix",
        "dataset_id": "timbrooks/instructpix2pix-clip-filtered",
        "url": "https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered",
        "license": "unspecified_on_hf",
        "split": "train",
        "config": "default",
        "schema": "instructpix2pix",
        "status": "needs_license_review",
        "recommended_use": ["baseline_only"],
        "notes": "Large synthetic edit-pair dataset. Do not mix into primary training until license and generation provenance are reviewed.",
    },
    {
        "source_id": "mit_adobe_fivek",
        "dataset_id": "MIT-Adobe FiveK / HF mirrors",
        "url": "https://data.csail.mit.edu/graphics/fivek/",
        "license": "research_terms_or_other",
        "split": "train_only_if_mirror_has_split",
        "config": "external",
        "schema": "retouching_before_after",
        "status": "photoshop_like_retouching_needs_terms_review",
        "recommended_use": ["photo_retouching", "preservation_reward"],
        "notes": "Classic before/after expert retouching dataset. Useful for Photoshop-like global edits, but not reasoning edits; verify redistribution terms.",
    },
    {
        "source_id": "ppr10k",
        "dataset_id": "PPR10K / MMArt-PPR10k mirrors",
        "url": "https://github.com/csjliang/PPR10K",
        "license": "mirror_reports_apache_2_0_but_verify_upstream",
        "split": "train",
        "config": "external",
        "schema": "portrait_retouching_before_after",
        "status": "photoshop_like_retouching_needs_terms_review",
        "recommended_use": ["portrait_retouching", "visual_quality_reward"],
        "notes": "Portrait retouching before/after data; useful for quality/preservation reward, not RISE/GRADE reasoning.",
    },
    {
        "source_id": "emu_edit_test",
        "dataset_id": "facebook/emu_edit_test_set",
        "url": "https://huggingface.co/datasets/facebook/emu_edit_test_set",
        "license": "unspecified_on_hf",
        "split": "test",
        "config": "default",
        "schema": "eval_only",
        "status": "excluded_eval_only",
        "recommended_use": ["external_eval_only"],
        "notes": "Held-out test set. Do not use for training.",
    },
]


WIKIMEDIA_QUERY_BANK: List[Dict[str, Any]] = [
    {
        "query": "fresh banana photograph",
        "task_family": "temporal_reasoning",
        "sub_family": "aging_decay",
        "instruction": "Edit only the banana to show natural brown and black ripening spots after several days at room temperature, while preserving the plate, table, background, lighting, and viewpoint.",
    },
    {
        "query": "potted sprout plant photograph",
        "task_family": "temporal_reasoning",
        "sub_family": "growth",
        "instruction": "Edit the plant so it appears two weeks older with a taller stem and several more leaves, while keeping the pot, soil, camera angle, and background unchanged.",
    },
    {
        "query": "ice cube melting photograph",
        "task_family": "causal_reasoning",
        "sub_family": "thermal_phase_change",
        "instruction": "Show the ice after warming: make it partially melted with a small puddle of water, while preserving the container, surface, lighting, and camera viewpoint.",
    },
    {
        "query": "slice of bread photograph",
        "task_family": "temporal_reasoning",
        "sub_family": "process_result",
        "instruction": "Edit the bread so it looks evenly toasted golden brown, without changing the plate, table, background, or camera viewpoint.",
    },
    {
        "query": "traffic light photograph",
        "task_family": "factual_order_correction",
        "sub_family": "traffic_signal",
        "instruction": "Change the illuminated traffic signal to green while preserving the traffic light housing, pole, background, weather, and perspective.",
    },
    {
        "query": "red apple on table photograph",
        "task_family": "temporal_reasoning",
        "sub_family": "aging_decay",
        "instruction": "Edit only the apple to show bruising and early decay spots, while preserving the table, shadows, background, and camera viewpoint.",
    },
    {
        "query": "flower bud photograph",
        "task_family": "temporal_reasoning",
        "sub_family": "growth",
        "instruction": "Edit the flower bud into a newly opened flower of the same plant, preserving stem position, leaves, background, lighting, and viewpoint.",
    },
    {
        "query": "river meander aerial photograph",
        "task_family": "discipline_reasoning",
        "sub_family": "geography_river_process",
        "instruction": "Mark the outer bend as erosion and the inner bend as deposition using small clear labels, without changing the river shape, land texture, or map-like viewpoint.",
    },
    {
        "query": "stone arch bridge photograph",
        "task_family": "discipline_reasoning",
        "sub_family": "history_architecture_restoration",
        "instruction": "Restore the visibly missing or damaged stone arch segment in a historically plausible way, preserving the original stone material, lighting, and surrounding scene.",
    },
    {
        "query": "litmus paper chemistry photograph",
        "task_family": "discipline_reasoning",
        "sub_family": "chemistry_indicator",
        "instruction": "If the strip is blue litmus in acid, edit only the litmus strip so it turns red, preserving the container, liquid, background, and camera angle.",
    },
    {
        "query": "light refraction glass water photograph",
        "task_family": "discipline_reasoning",
        "sub_family": "physics_refraction",
        "instruction": "Add a clear ray path that bends at the air-water or air-glass boundary according to refraction, while preserving the original setup and background.",
    },
    {
        "query": "penguin photograph",
        "task_family": "knowledge_reasoning",
        "sub_family": "species_trait",
        "instruction": "Edit the bird to emphasize a plausible penguin appearance with black back, white belly, and orange feet, while keeping the pose, background, and lighting stable.",
    },
    {
        "query": "table setting photograph fork knife plate",
        "task_family": "knowledge_reasoning",
        "sub_family": "social_convention",
        "instruction": "Correct the table setting so the fork is on the left of the plate and the knife is on the right, while preserving the plate, table, lighting, and viewpoint.",
    },
    {
        "query": "sponge compressed photograph",
        "task_family": "causal_reasoning",
        "sub_family": "force_deformation",
        "instruction": "Show the sponge compressed downward by force, with visible flattening only on the sponge and no change to the surface, background, lighting, or viewpoint.",
    },
    {
        "query": "rusty metal object photograph",
        "task_family": "temporal_reasoning",
        "sub_family": "aging_oxidation",
        "instruction": "Edit the metal object to show a less rusted earlier state with cleaner metallic areas, while preserving shape, background, lighting, and viewpoint.",
    },
]


SAFE_TEXT_BLOCKLIST = [
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
]

VISUAL_INPUT_DEPENDENT_PATTERNS = (
    "[v",
    "visual input",
    "reference image",
    "given depth image",
    "given segmentation",
    "given mask",
    "refer to the given",
    "follow the given",
    "watch the given",
    "according to the given",
)

COMMONS_LICENSE_ALLOW = (
    "cc0",
    "public domain",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_id(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "item"


def is_safe_text(text: str) -> bool:
    text = text.lower()
    return not any(term in text for term in SAFE_TEXT_BLOCKLIST)


def is_visual_input_dependent(base: Dict[str, Any]) -> bool:
    text = json.dumps(
        {
            "instruction": base.get("instruction"),
            "input": base.get("input"),
            "output": base.get("output"),
            "edit_type": base.get("edit_type"),
        },
        ensure_ascii=False,
    ).lower()
    if base.get("visual_input_url"):
        return True
    return any(pattern in text for pattern in VISUAL_INPUT_DEPENDENT_PATTERNS)


def load_benchmark_norms() -> set:
    path = repo_path("data", "benchmarks", "benchmark_text_index.jsonl")
    norms = set()
    if not path.exists():
        return norms
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("normalized_text"):
            norms.add(row["normalized_text"])
    return norms


def decontam_status(text: str, benchmark_norms: set) -> Dict[str, Any]:
    norm = normalize_text(text)
    exact = norm in benchmark_norms
    return {
        "normalized_text": norm,
        "benchmark_text_exact_match": exact,
        "status": "rejected_exact_text_match" if exact else "passed_exact_text_check",
    }


def request_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30, retries: int = 3) -> Dict[str, Any]:
    response = None
    for attempt in range(retries):
        response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if response.status_code not in {429, 500, 502, 503, 504}:
            break
        wait = int(response.headers.get("Retry-After") or 0) or (2 + attempt * 3)
        time.sleep(wait)
    assert response is not None
    response.raise_for_status()
    return response.json()


def enrich_hf_catalog() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in CURATED_HF_SOURCES:
        row = dict(source)
        dataset_id = source["dataset_id"]
        if "/" in dataset_id and not dataset_id.startswith("MIT-Adobe"):
            try:
                meta = request_json(f"{HF_API}/{dataset_id}", timeout=20)
                row["hf_downloads"] = meta.get("downloads")
                row["hf_likes"] = meta.get("likes")
                row["hf_tags"] = meta.get("tags", [])
                card = meta.get("cardData") or {}
                row["hf_card_license"] = card.get("license")
                row["hf_size_categories"] = card.get("size_categories")
            except Exception as exc:
                row["hf_error"] = str(exc)
        rows.append(row)
    return rows


def write_catalog_report(catalog: List[Dict[str, Any]]) -> None:
    write_json(repo_path("data", "sources", "real_edit_source_catalog.json"), catalog)
    lines = [
        "# Real Image Editing Source Search",
        "",
        "Generated by `scripts/data/collect_real_edit_sources.py`.",
        "",
        "This catalog separates train-usable sources, research-only sources, license-review sources, and eval-only sources. Benchmark datasets such as RISE/GRADE/KRIS remain evaluation/decontamination-only.",
        "",
        "| Source | License | Status | Use | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in catalog:
        use = ", ".join(row.get("recommended_use", []))
        lines.append(
            f"| `{row['source_id']}` | {row.get('license', '')} | {row.get('status', '')} | {use} | {row.get('notes', '').replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "Photoshop-style before/after data should come from dataset releases such as MIT-Adobe FiveK or PPR10K, not arbitrary tutorial/blog images, unless each image has explicit reusable licensing and provenance.",
        ]
    )
    out = repo_path("reports", "data_sources", "real_edit_source_report.md")
    ensure_dir(out.parent)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hf_rows(dataset_id: str, config: str, split: str, offset: int, length: int) -> List[Dict[str, Any]]:
    try:
        payload = request_json(
            HF_ROWS_API,
            params={"dataset": dataset_id, "config": config, "split": split, "offset": offset, "length": length},
            timeout=80,
        )
    except Exception:
        if offset != 0:
            raise
        payload = request_json(
            HF_FIRST_ROWS_API,
            params={"dataset": dataset_id, "config": config, "split": split},
            timeout=80,
        )
    rows: List[Dict[str, Any]] = []
    for item in payload.get("rows", []):
        row = dict(item["row"])
        row["_row_idx"] = item["row_idx"]
        rows.append(row)
    return rows


def hf_num_examples(dataset_id: str, config: str, split: str) -> Optional[int]:
    try:
        payload = request_json(
            "https://datasets-server.huggingface.co/info",
            params={"dataset": dataset_id, "config": config},
            timeout=30,
        )
    except Exception:
        return None
    splits = ((payload.get("dataset_info") or {}).get("splits") or {})
    info = splits.get(split) or {}
    value = info.get("num_examples")
    return int(value) if isinstance(value, int) and value > 0 else None


def image_url(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        return value.get("src")
    return None


def save_image_from_url(url: str, out_path: Path) -> Dict[str, Any]:
    ensure_dir(out_path.parent)
    response = None
    for attempt in range(4):
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        if response.status_code != 429:
            break
        wait = int(response.headers.get("Retry-After") or 0) or (3 + attempt * 3)
        time.sleep(wait)
    assert response is not None
    response.raise_for_status()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(response.content)
    with Image.open(tmp) as img:
        if out_path.suffix.lower() in {".jpg", ".jpeg"}:
            img = img.convert("RGB")
            img.save(out_path, quality=92)
        else:
            img.save(out_path)
        width, height = img.size
        mode = img.mode
    tmp.unlink(missing_ok=True)
    return {
        "path": str(out_path.relative_to(ROOT)),
        "width": width,
        "height": height,
        "mode": mode,
        "sha256": sha256_file(out_path),
        "ahash": average_hash(out_path),
        "bytes": out_path.stat().st_size,
    }


def normalize_hf_pair(source: Dict[str, Any], row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    schema = source["schema"]
    base: Dict[str, Any] = {
        "source_id": source["source_id"],
        "dataset_id": source["dataset_id"],
        "dataset_url": source["url"],
        "license": source["license"],
        "split": source["split"],
        "row_idx": row.get("_row_idx"),
        "schema": schema,
    }
    if schema in {"magicbrush", "imagenhub"}:
        instruction = row.get("instruction") or ""
        base.update(
            {
                "instruction": instruction,
                "source_image_url": image_url(row.get("source_img")),
                "target_image_url": image_url(row.get("target_img")),
                "mask_image_url": image_url(row.get("mask_img")),
                "source_caption": row.get("source_global_caption"),
                "target_caption": row.get("target_global_caption"),
                "target_local_caption": row.get("target_local_caption"),
                "edit_type": "text_guided_local_edit",
            }
        )
    elif schema == "anyedit":
        instruction = row.get("edit_instruction") or row.get("input") or ""
        base.update(
            {
                "instruction": instruction,
                "source_image_url": image_url(row.get("image_file")),
                "target_image_url": image_url(row.get("edited_file")),
                "visual_input_url": image_url(row.get("visual_input")),
                "edit_type": row.get("edit_type"),
                "input": row.get("input"),
                "output": row.get("output"),
            }
        )
    elif schema == "omniedit":
        prompts = row.get("edited_prompt_list") or []
        if isinstance(prompts, str):
            prompts = [prompts]
        instruction = prompts[-1] if prompts else row.get("task") or ""
        base.update(
            {
                "instruction": instruction,
                "source_image_url": image_url(row.get("src_img")),
                "target_image_url": image_url(row.get("edited_img")),
                "edit_type": row.get("task"),
                "edited_prompt_list": prompts,
                "source_consistency_score_1": row.get("sc_score_1"),
                "source_consistency_score_2": row.get("sc_score_2"),
                "perceptual_quality_score": row.get("pq_score"),
                "overall_score": row.get("o_score"),
            }
        )
    elif schema == "anyedit_thinking":
        instruction = row.get("edit_instruction") or ""
        base.update(
            {
                "instruction": instruction,
                "source_image_url": image_url(row.get("input_image")),
                "target_image_url": image_url(row.get("output_image")),
                "edit_type": row.get("edit_type"),
                "thinking_process": row.get("thinking_process"),
                "prompt_template": row.get("prompt_template"),
                "original_data": row.get("original_data"),
                "batch_number": row.get("batch_number"),
            }
        )
    else:
        return None
    if not base.get("instruction") or not base.get("source_image_url") or not base.get("target_image_url"):
        return None
    if not is_safe_text(json.dumps(base, ensure_ascii=False)):
        return None
    if is_visual_input_dependent(base):
        return None
    return base


def image_quality_gate(meta: Dict[str, Any]) -> Dict[str, Any]:
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    side_min = min(width, height)
    side_max = max(width, height)
    aspect = (side_max / side_min) if side_min else 999
    ok = side_min >= 256 and aspect <= 3.0 and int(meta.get("bytes") or 0) >= 4096
    reasons: List[str] = []
    if side_min < 256:
        reasons.append("short_side_lt_256")
    if aspect > 3.0:
        reasons.append("aspect_gt_3")
    if int(meta.get("bytes") or 0) < 4096:
        reasons.append("file_too_small")
    return {"ok": ok, "short_side": side_min, "aspect_ratio": round(aspect, 3), "reject_reasons": reasons}


def pair_quality_gate(pair: Dict[str, Any]) -> Dict[str, Any]:
    src = image_quality_gate(pair["source_image"])
    tgt = image_quality_gate(pair["target_image"])
    ok = src["ok"] and tgt["ok"]
    return {"ok": ok, "source": src, "target": tgt}


def offset_plan(source: Dict[str, Any], per_source: int, randomize: bool, rng: random.Random) -> List[int]:
    length = min(16, max(8, per_source * 2))
    if not randomize:
        return list(range(0, max(300, per_source * 8), length))
    n = hf_num_examples(source["dataset_id"], source["config"], source["split"])
    if not n:
        return list(range(0, max(300, per_source * 8), length))
    max_offset = max(0, n - length)
    attempts = min(max(per_source * 4, 24), max_offset + 1)
    offsets = sorted({rng.randint(0, max_offset) for _ in range(attempts)})
    if 0 not in offsets:
        offsets.insert(0, 0)
    return offsets


def materialize_hf_samples(
    version: str,
    per_source: int,
    benchmark_norms: set,
    randomize: bool,
    seed: int,
    hf_source_ids: Optional[set] = None,
    exclude_edit_types: Optional[set] = None,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    default_source_ids = {"magicbrush_train", "imagenhub_filtered", "anyedit_train", "omniedit_train"}
    selected_source_ids = hf_source_ids or default_source_ids
    excluded_types = {normalize_text(x) for x in (exclude_edit_types or set()) if x}
    sample_sources = [
        s
        for s in CURATED_HF_SOURCES
        if s["source_id"] in selected_source_ids
    ]
    rng = random.Random(seed)
    seen_pair_keys = set()
    seen_image_sha = set()
    for source in sample_sources:
        got = 0
        for offset in offset_plan(source, per_source, randomize, rng):
            if got >= per_source:
                break
            try:
                rows = hf_rows(source["dataset_id"], source["config"], source["split"], offset=offset, length=min(16, max(8, per_source * 2)))
            except Exception as exc:
                print(f"warning: failed HF rows for {source['source_id']}: {exc}")
                continue
            if not rows:
                continue
            for row in rows:
                pair = normalize_hf_pair(source, row)
                if not pair:
                    continue
                edit_type_norm = normalize_text(str(pair.get("edit_type") or ""))
                if edit_type_norm and edit_type_norm in excluded_types:
                    continue
                sample_id = f"{version}_{source['source_id']}_{pair['row_idx']:06d}"
                pair_key = (source["source_id"], pair["row_idx"], pair["instruction"])
                if pair_key in seen_pair_keys:
                    continue
                seen_pair_keys.add(pair_key)
                out_root = repo_path("data", "real_edits", version, "pairs", source["source_id"], sample_id)
                try:
                    pair["source_image"] = save_image_from_url(pair.pop("source_image_url"), out_root / "source.jpg")
                    pair["target_image"] = save_image_from_url(pair.pop("target_image_url"), out_root / "target.jpg")
                    if pair.get("mask_image_url"):
                        pair["mask_image"] = save_image_from_url(pair.pop("mask_image_url"), out_root / "mask.png")
                    if pair.get("visual_input_url"):
                        pair["visual_input_image"] = save_image_from_url(pair.pop("visual_input_url"), out_root / "visual_input.jpg")
                    pair.pop("mask_image_url", None)
                    pair.pop("visual_input_url", None)
                except Exception as exc:
                    print(f"warning: image download failed for {sample_id}: {exc}")
                    continue
                image_key = (pair["source_image"]["sha256"], pair["target_image"]["sha256"])
                if image_key in seen_image_sha:
                    continue
                seen_image_sha.add(image_key)
                pair["quality_tags"] = pair_quality_gate(pair)
                if not pair["quality_tags"]["ok"]:
                    continue
                pair["sample_id"] = sample_id
                pair["provenance"] = {
                    "type": "hf_dataset_viewer_sample",
                    "downloaded_at": utc_now(),
                    "row_idx": pair["row_idx"],
                    "dataset": source["dataset_id"],
                    "split": source["split"],
                    "randomized_offsets": randomize,
                    "seed": seed,
                }
                pair["leakage_tags"] = decontam_status(pair["instruction"], benchmark_norms)
                if pair["leakage_tags"]["benchmark_text_exact_match"]:
                    continue
                rows_out.append(pair)
                got += 1
                if got >= per_source:
                    break
            time.sleep(0.2)
    write_jsonl(repo_path("data", "sources", f"real_edit_pairs_sample_{version}.jsonl"), rows_out)
    return rows_out


def commons_license_ok(license_name: str) -> bool:
    low = (license_name or "").lower()
    return any(token in low for token in COMMONS_LICENSE_ALLOW)


def commons_search(query: str, limit: int) -> List[Dict[str, Any]]:
    data = request_json(
        COMMONS_API,
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": max(5, limit * 4),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size|sha1",
            "iiurlwidth": 960,
        },
        timeout=30,
    )
    pages = list((data.get("query") or {}).get("pages", {}).values())
    out: List[Dict[str, Any]] = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        license_name = (meta.get("LicenseShortName") or {}).get("value") or ""
        mime = info.get("mime") or ""
        if mime not in {"image/jpeg", "image/png"}:
            continue
        if not commons_license_ok(license_name):
            continue
        out.append(
            {
                "title": page.get("title"),
                "pageid": page.get("pageid"),
                "url": info.get("thumburl") or info.get("url"),
                "original_url": info.get("url"),
                "description_url": info.get("descriptionurl"),
                "mime": mime,
                "width": info.get("thumbwidth") or info.get("width"),
                "height": info.get("thumbheight") or info.get("height"),
                "license": license_name,
                "artist": (meta.get("Artist") or {}).get("value"),
                "credit": (meta.get("Credit") or {}).get("value"),
                "sha1": info.get("sha1"),
            }
        )
        if len(out) >= limit:
            break
    return out


def strip_html(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def materialize_wikimedia_sources(version: str, per_query: int, benchmark_norms: set) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    image_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    seen_sha = set()
    for query_spec in WIKIMEDIA_QUERY_BANK:
        try:
            hits = commons_search(query_spec["query"], per_query)
        except Exception as exc:
            print(f"warning: commons search failed for {query_spec['query']}: {exc}")
            continue
        for idx, hit in enumerate(hits):
            if not hit.get("url"):
                continue
            time.sleep(1.0)
            raw_id = clean_id(f"{query_spec['query']}_{hit.get('pageid')}_{idx}")
            image_id = f"{version}_commons_{raw_id}"
            suffix = ".png" if hit["mime"] == "image/png" else ".jpg"
            out_path = repo_path("data", "real_edits", version, "wikimedia", f"{image_id}{suffix}")
            try:
                image_meta = save_image_from_url(hit["url"], out_path)
            except Exception as exc:
                print(f"warning: commons image download failed for {hit.get('title')}: {exc}")
                continue
            if image_meta["sha256"] in seen_sha:
                out_path.unlink(missing_ok=True)
                continue
            seen_sha.add(image_meta["sha256"])
            image_row = {
                "image_id": image_id,
                "source_id": "wikimedia_commons",
                "query": query_spec["query"],
                "title": hit.get("title"),
                "description_url": hit.get("description_url"),
                "original_url": hit.get("original_url"),
                "license": hit.get("license"),
                "artist": strip_html(hit.get("artist")),
                "credit": strip_html(hit.get("credit")),
                "image": image_meta,
                "task_family_hint": query_spec["task_family"],
                "sub_family_hint": query_spec["sub_family"],
                "manual_visual_check": {
                    "status": "unverified_candidate",
                    "required_before_training": True,
                },
                "provenance": {"type": "wikimedia_commons_api", "downloaded_at": utc_now()},
            }
            task_id = f"{version}_real_seed_{len(task_rows):06d}"
            task_row = {
                "task_id": task_id,
                "source": "wikimedia_real_source_seed",
                "source_image": image_meta["path"],
                "source_image_provenance": {
                    "dataset": "Wikimedia Commons",
                    "description_url": hit.get("description_url"),
                    "license": hit.get("license"),
                    "artist": strip_html(hit.get("artist")),
                },
                "benchmark_family": "RISE_GRADE_KRIS_like",
                "task_family": query_spec["task_family"],
                "sub_family": query_spec["sub_family"],
                "instruction": query_spec["instruction"],
                "expected_target": "To be generated by a strong image editor and verified by checklist-first critic.",
                "render_status": "needs_strong_editor_target",
                "source_manual_visual_check": {
                    "status": "unverified_candidate",
                    "required_before_training": True,
                },
                "leakage_tags": decontam_status(query_spec["instruction"], benchmark_norms),
                "license": hit.get("license"),
            }
            if task_row["leakage_tags"]["benchmark_text_exact_match"]:
                continue
            image_rows.append(image_row)
            task_rows.append(task_row)
        time.sleep(0.2)
    write_jsonl(repo_path("data", "sources", f"wikimedia_source_pool_{version}.jsonl"), image_rows)
    write_jsonl(repo_path("data", "tasks", f"real_seed_prompts_{version}.jsonl"), task_rows)
    return image_rows, task_rows


def summarize(version: str, catalog: List[Dict[str, Any]], pairs: List[Dict[str, Any]], wiki_images: List[Dict[str, Any]], wiki_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [x for x in catalog if x.get("status", "").startswith("usable")]
    summary = {
        "version": version,
        "generated_at": utc_now(),
        "catalog_sources": len(catalog),
        "usable_or_train_like_sources": len(usable),
        "hf_pair_samples": len(pairs),
        "wikimedia_source_images": len(wiki_images),
        "wikimedia_seed_tasks": len(wiki_tasks),
        "hf_pair_sources": sorted({x["source_id"] for x in pairs}),
        "wikimedia_task_families": dict(sorted(_count(x["task_family"] for x in wiki_tasks).items())),
        "outputs": {
            "catalog": "data/sources/real_edit_source_catalog.json",
            "source_report": "reports/data_sources/real_edit_source_report.md",
            "hf_pairs": f"data/sources/real_edit_pairs_sample_{version}.jsonl",
            "wikimedia_pool": f"data/sources/wikimedia_source_pool_{version}.jsonl",
            "wikimedia_tasks": f"data/tasks/real_seed_prompts_{version}.jsonl",
        },
        "policy": [
            "Use train/filtered sources only for training candidates; exclude test/dev benchmark-style splits unless manually approved.",
            "Do not train on RISE/GRADE/KRIS raw images, instructions, references, GT images, or annotations.",
            "Photoshop/tutorial before-after web images require explicit license/provenance; arbitrary blog images are not collected.",
            "Run image/text decontamination and VLM quality filtering before promoting this seed pool into SFT/RL splits.",
        ],
    }
    write_json(repo_path("reports", "data_sources", f"real_collection_{version}.json"), summary)
    return summary


def _count(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def csv_set(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and sample real image-editing data sources.")
    parser.add_argument("--version", default="v2_seed")
    parser.add_argument("--hf-per-source", type=int, default=12)
    parser.add_argument("--wiki-per-query", type=int, default=2)
    parser.add_argument("--randomize", action="store_true", help="Sample random HF row offsets instead of only early rows.")
    parser.add_argument("--seed", type=int, default=531)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-wikimedia", action="store_true")
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument(
        "--hf-source-ids",
        default="",
        help="Comma-separated HF source IDs to sample. Defaults to the standard train/non-eval pair sources.",
    )
    parser.add_argument(
        "--exclude-edit-types",
        default="",
        help="Comma-separated normalized edit_type values to skip before image download, for example tune_transfer.",
    )
    args = parser.parse_args()

    benchmark_norms = load_benchmark_norms()
    catalog = enrich_hf_catalog()
    write_catalog_report(catalog)
    if args.catalog_only:
        print(json.dumps({"catalog_sources": len(catalog), "output": "data/sources/real_edit_source_catalog.json"}, indent=2))
        return
    hf_source_ids = csv_set(args.hf_source_ids)
    valid_source_ids = {source["source_id"] for source in CURATED_HF_SOURCES}
    unknown_source_ids = sorted(hf_source_ids - valid_source_ids)
    if unknown_source_ids:
        parser.error(f"unknown --hf-source-ids: {', '.join(unknown_source_ids)}")
    exclude_edit_types = csv_set(args.exclude_edit_types)
    pairs = (
        []
        if args.skip_hf
        else materialize_hf_samples(
            args.version,
            args.hf_per_source,
            benchmark_norms,
            args.randomize,
            args.seed,
            hf_source_ids=hf_source_ids,
            exclude_edit_types=exclude_edit_types,
        )
    )
    if args.skip_wikimedia or args.wiki_per_query <= 0:
        wiki_images, wiki_tasks = [], []
        write_jsonl(repo_path("data", "sources", f"wikimedia_source_pool_{args.version}.jsonl"), wiki_images)
        write_jsonl(repo_path("data", "tasks", f"real_seed_prompts_{args.version}.jsonl"), wiki_tasks)
    else:
        wiki_images, wiki_tasks = materialize_wikimedia_sources(args.version, args.wiki_per_query, benchmark_norms)
    summary = summarize(args.version, catalog, pairs, wiki_images, wiki_tasks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
