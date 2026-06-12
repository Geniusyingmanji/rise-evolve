#!/usr/bin/env python3
"""VLM render verification for programmatic v2 tasks via the local gpt-5.5 proxy.

For each sampled task, shows the VLM the TEACHER render and runs the task's own
verifier_spec VQA checks (exact-match against expected answers). Then shows the
NEGATIVE render and expects at least one check to FAIL. A generator is healthy
when teacher-pass is high and negative-pass is low — the same checks later serve
as the verifiable RL reward, so this doubles as a reward-channel calibration.

Usage:
  python3 scripts/data/vlm_verify_renders.py \
    --tasks data/tasks/tasks_v2.jsonl --per-subtask 2 \
    --output reports/data_quality/vlm_render_verify_v2.json
"""

import argparse
import base64
import concurrent.futures
import io
import json
import threading
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

API_URL = "http://localhost:4142/v1/responses"
MODEL = "gpt-5.5"


def b64_image(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask_vqa(image_path: str, questions, effort="low", timeout=240):
    qlist = "\n".join(f"{i+1}. {q['question']}" for i, q in enumerate(questions))
    prompt = (
        "Look at the image and answer each question with a SHORT exact answer "
        "(a number, word, or short phrase). Answer strict JSON only: "
        '{"answers": ["...", ...]} in question order.\n' + qlist
    )
    payload = {
        "model": MODEL,
        "reasoning": {"effort": effort},
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image(image_path)}"},
                {"type": "input_text", "text": prompt},
            ],
        }],
    }
    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text = c["text"].strip()
                    if text.startswith("```"):
                        text = text.strip("`").lstrip("json").strip()
                    return json.loads(text)["answers"]
    raise RuntimeError("no output_text")


def norm(s) -> str:
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum() or ch.isspace()).strip()


def tokens(s) -> list:
    """Alphanumeric token sequence, treating ->, commas, brackets, 'and' as separators."""
    text = str(s).lower()
    for sep in ["->", "→", ",", ";", "[", "]", "(", ")", " and ", " then "]:
        text = text.replace(sep, " ")
    return [t for t in text.split() if t.isalnum()]


def match(expected: str, got: str) -> bool:
    e, g = norm(expected), norm(got)
    if e == g or (e and e in g) or (g and g in e):
        return True
    te, tg = tokens(expected), tokens(got)
    return bool(te) and te == tg


def score_render(task, render_path):
    checks = (task.get("verifier_spec") or {}).get("vqa_checks") or []
    if not checks or not Path(render_path).exists():
        return None
    answers = ask_vqa(render_path, checks)
    results = []
    for chk, got in zip(checks, answers):
        results.append({
            "question": chk["question"],
            "expected": chk["expected_answer"],
            "got": got,
            "pass": match(chk["expected_answer"], got),
            "weight": chk.get("weight", 1.0),
        })
    total_w = sum(r["weight"] for r in results) or 1.0
    score = sum(r["weight"] for r in results if r["pass"]) / total_w
    return {"score": round(score, 3), "checks": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--per-subtask", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    by_sub = defaultdict(list)
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_sub[row.get("sub_task", "?")].append(row)

    sampled = []
    for rows in by_sub.values():
        sampled.extend(rows[: args.per_subtask])

    lock = threading.Lock()
    rows_out = []
    stats = Counter()

    def run(task):
        tid = task["task_id"]
        teacher = f"data/renders/teacher/{tid}_teacher.png"
        negative = f"data/renders/negative/{tid}_negative.png"
        entry = {"task_id": tid, "sub_task": task.get("sub_task")}
        try:
            entry["teacher"] = score_render(task, teacher)
            entry["negative"] = score_render(task, negative)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
        with lock:
            rows_out.append(entry)
            if entry.get("error"):
                stats["error"] += 1
            else:
                if entry["teacher"]:
                    stats["teacher_n"] += 1
                    stats["teacher_pass"] += entry["teacher"]["score"] >= 0.99
                if entry["negative"]:
                    stats["neg_n"] += 1
                    stats["neg_reject"] += entry["negative"]["score"] < 0.99
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, sampled))

    per_sub = defaultdict(lambda: {"teacher_pass": 0, "teacher_n": 0, "neg_reject": 0, "neg_n": 0})
    for e in rows_out:
        if e.get("error"):
            continue
        s = per_sub[e["sub_task"]]
        if e.get("teacher"):
            s["teacher_n"] += 1
            s["teacher_pass"] += e["teacher"]["score"] >= 0.99
        if e.get("negative"):
            s["neg_n"] += 1
            s["neg_reject"] += e["negative"]["score"] < 0.99

    report = {
        "sampled": len(sampled),
        "errors": stats["error"],
        "teacher_pass_rate": round(stats["teacher_pass"] / max(stats["teacher_n"], 1), 3),
        "negative_reject_rate": round(stats["neg_reject"] / max(stats["neg_n"], 1), 3),
        "per_subtask": {k: dict(v) for k, v in sorted(per_sub.items())},
        "rows": rows_out,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in
                      ["sampled", "errors", "teacher_pass_rate", "negative_reject_rate"]}, indent=2))
    for k, v in report["per_subtask"].items():
        print(f"  {k}: teacher {v['teacher_pass']}/{v['teacher_n']} neg_reject {v['neg_reject']}/{v['neg_n']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
