#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rise_evolve.io import ensure_dir, iter_jsonl, repo_path, write_jsonl


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Placeholder benchmark renderer adapter.")
    parser.add_argument("--programs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["copy-source", "skip"], default="skip")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    render_dir = out_dir / "renders"
    ensure_dir(render_dir)
    rows = []
    for idx, row in enumerate(iter_jsonl(Path(args.programs))):
        if args.limit is not None and idx >= args.limit:
            break
        source = row.get("source_image")
        sample_id = row.get("sample_id") or row.get("task_id") or str(idx)
        record = {"sample_id": sample_id, "mode": args.mode, "source_image": source, "candidate_image": None, "status": "skipped"}
        if args.mode == "copy-source" and source:
            src_path = repo_path(source) if not Path(source).is_absolute() else Path(source)
            if src_path.exists():
                dst = render_dir / f"{sample_id}.png"
                shutil.copyfile(src_path, dst)
                record.update({"candidate_image": str(dst), "status": "copied_source"})
            else:
                record.update({"status": "missing_source"})
        rows.append(record)
    write_jsonl(out_dir / "render_metadata.jsonl", rows)
    print(json.dumps({"output_dir": str(out_dir), "rows": len(rows), "mode": args.mode}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
