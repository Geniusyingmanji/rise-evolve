#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

from common import ensure_dir, repo_path, write_json


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_font(size: int = 11) -> ImageFont.ImageFont:
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def thumb(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "#f8fafc")
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def make_pair_sheet(rows: List[Dict[str, Any]], output: Path, limit: int = 24) -> None:
    rows = stratified_rows(rows, "source_id", limit)
    if not rows:
        return
    font = load_font(11)
    thumb_size = 150
    label_h = 54
    cols = 3
    gap = 10
    width = cols * thumb_size * 2 + (cols + 1) * gap
    height = ((len(rows) + cols - 1) // cols) * (thumb_size + label_h + gap) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        rr, cc = divmod(i, cols)
        x = gap + cc * (thumb_size * 2 + gap)
        y = gap + rr * (thumb_size + label_h + gap)
        for j, key in enumerate(["source_image", "target_image"]):
            p = Path(row[key]["path"])
            im = thumb(p, thumb_size)
            sheet.paste(im, (x + j * thumb_size, y))
            draw.rectangle((x + j * thumb_size, y, x + (j + 1) * thumb_size - 1, y + thumb_size - 1), outline="#cbd5e1")
            draw.text((x + j * thumb_size + 4, y + 4), "source" if j == 0 else "target", font=font, fill="#111827")
        label = f"{row.get('source_id')} | {row.get('instruction', '')}"[:92]
        draw.text((x, y + thumb_size + 4), label, font=font, fill="#111827")
    ensure_dir(output.parent)
    sheet.save(output)


def make_wikimedia_sheet(rows: List[Dict[str, Any]], output: Path, limit: int = 32) -> None:
    rows = stratified_rows(rows, "task_family_hint", limit)
    if not rows:
        return
    font = load_font(10)
    thumb_size = 140
    label_h = 56
    cols = 4
    gap = 10
    width = cols * thumb_size + (cols + 1) * gap
    height = ((len(rows) + cols - 1) // cols) * (thumb_size + label_h + gap) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        rr, cc = divmod(i, cols)
        x = gap + cc * (thumb_size + gap)
        y = gap + rr * (thumb_size + label_h + gap)
        p = Path(row["image"]["path"])
        im = thumb(p, thumb_size)
        sheet.paste(im, (x, y))
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline="#cbd5e1")
        status = (row.get("manual_visual_check") or {}).get("status", "unverified")
        label = f"{row.get('task_family_hint')} | {row.get('query')}"[:44]
        draw.text((x, y + thumb_size + 4), label, font=font, fill="#111827")
        draw.text((x, y + thumb_size + 19), f"{row.get('license', '')}"[:44], font=font, fill="#475569")
        draw.text((x, y + thumb_size + 34), status[:44], font=font, fill="#b91c1c" if status.startswith("rejected") else "#166534")
    ensure_dir(output.parent)
    sheet.save(output)


def stratified_rows(rows: List[Dict[str, Any]], key: str, limit: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    selected: List[Dict[str, Any]] = []
    while len(selected) < limit and any(groups.values()):
        for group_key in sorted(groups):
            if groups[group_key]:
                selected.append(groups[group_key].pop(0))
                if len(selected) >= limit:
                    break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit real-image source pool manifests.")
    parser.add_argument("--version", default="v2_seed")
    parser.add_argument("--sheet-limit", type=int, default=24)
    args = parser.parse_args()

    pair_path = repo_path("data", "sources", f"real_edit_pairs_sample_{args.version}.jsonl")
    rejected_pair_path = repo_path("data", "sources", f"real_edit_pairs_rejected_{args.version}.jsonl")
    wiki_path = repo_path("data", "sources", f"wikimedia_source_pool_{args.version}.jsonl")
    task_path = repo_path("data", "tasks", f"real_seed_prompts_{args.version}.jsonl")
    pairs = read_jsonl(pair_path)
    rejected_pairs = read_jsonl(rejected_pair_path)
    wiki = read_jsonl(wiki_path)
    tasks = read_jsonl(task_path)

    missing_pair_images = []
    for row in pairs:
        for key in ["source_image", "target_image", "mask_image", "visual_input_image"]:
            if key in row and not Path(row[key]["path"]).exists():
                missing_pair_images.append({"sample_id": row.get("sample_id"), "key": key, "path": row[key]["path"]})
    missing_wiki_images = [row["image"]["path"] for row in wiki if not Path(row["image"]["path"]).exists()]

    pair_sources = collections.Counter(row.get("source_id") for row in pairs)
    pair_licenses = collections.Counter(row.get("license") for row in pairs)
    wiki_licenses = collections.Counter(row.get("license") for row in wiki)
    wiki_status = collections.Counter((row.get("manual_visual_check") or {}).get("status", "unknown") for row in wiki)
    task_status = collections.Counter(row.get("render_status") for row in tasks)

    report = {
        "version": args.version,
        "pair_rows": len(pairs),
        "pair_sources": dict(sorted(pair_sources.items())),
        "pair_licenses": dict(sorted(pair_licenses.items())),
        "pair_unique_source_sha": len({row["source_image"]["sha256"] for row in pairs if "source_image" in row}),
        "pair_unique_target_sha": len({row["target_image"]["sha256"] for row in pairs if "target_image" in row}),
        "pair_unique_instructions": len({row.get("instruction") for row in pairs}),
        "pair_exact_benchmark_text_leaks": sum(1 for row in pairs if (row.get("leakage_tags") or {}).get("benchmark_text_exact_match")),
        "pair_quality_rejects": sum(1 for row in pairs if not (row.get("quality_tags") or {}).get("ok", True)),
        "rejected_pair_rows": len(rejected_pairs),
        "rejected_pair_reasons": dict(
            sorted(
                collections.Counter(
                    (row.get("safety_tags") or {}).get("status", "unknown")
                    for row in rejected_pairs
                ).items()
            )
        ),
        "missing_pair_images": missing_pair_images,
        "wikimedia_rows": len(wiki),
        "wikimedia_licenses": dict(sorted(wiki_licenses.items())),
        "wikimedia_manual_status": dict(sorted(wiki_status.items())),
        "missing_wikimedia_images": missing_wiki_images,
        "real_seed_task_rows": len(tasks),
        "real_seed_task_status": dict(sorted(task_status.items())),
        "real_seed_exact_benchmark_text_leaks": sum(1 for row in tasks if (row.get("leakage_tags") or {}).get("benchmark_text_exact_match")),
    }

    out_dir = repo_path("reports", "data_sources")
    write_json(out_dir / f"real_source_audit_{args.version}.json", report)
    make_pair_sheet(pairs, out_dir / f"real_pair_sheet_{args.version}.png", limit=args.sheet_limit)
    make_wikimedia_sheet(wiki, out_dir / f"wikimedia_source_sheet_{args.version}.png", limit=args.sheet_limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
