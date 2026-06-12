#!/usr/bin/env python3
"""Refine RISEvolve task annotations with GPT-5.5 via the local key-free proxy.

Targets the boilerplate failure modes quantified by audit_sft_diversity.py
(and identified in the v2r SFT postmortem):

- instruction: template glue + monotone phrasing -> diverse surface forms
  (style rotated deterministically per task), SAME semantics
- editor_prompt: 100% instruction echo -> concrete executable editor
  instruction describing the visual target state
- rational_target_description: expected_target echo + fixed template ->
  grounded source->knowledge->change->preserve reasoning
- reasoning_chain: new field, first-person planner reasoning for SFT
  assistant targets
- edit_operations: change==region_hint==target boilerplate -> distinct,
  concrete fields
- atomic_checklist: 100% generic questions -> task-specific binary questions

Hard constraint: the edit semantics are FROZEN (teacher/negative renders
already exist). The model may only rephrase, concretize, and enrich; it must
not change what is edited or claim new visual content. Violations are caught
by post-validation and sent to the failures file.

Usage:
  python3 scripts/data/refine_tasks_gpt55.py \
    --tasks data/tasks/tasks_v1.jsonl \
    --output data/tasks/tasks_v1_refined.jsonl \
    --limit 50 --workers 4
"""

import argparse
import concurrent.futures
import json
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from common import normalize_text  # noqa: E402

API_URL = "http://localhost:4142/v1/responses"
MODEL = "gpt-5.5"
BENCH_INDEX = REPO_ROOT / "data" / "benchmarks" / "benchmark_text_index.jsonl"
SCENE_INVENTORY = REPO_ROOT / "data" / "taxonomy" / "scene_inventory_v1.json"

STYLES = [
    "terse imperative, one short sentence, no preamble",
    "detailed polite user request, 2 sentences, natural everyday wording",
    "question form, e.g. 'Can you show ...?' or 'What would ... look like?'",
    "casual user tone with a small typo-free informal phrasing, 1-2 sentences",
    "professional specification tone, precise and compact",
    "short contextual story setup (max 2 sentences) ending in the edit request",
    "minimal keyword-style request, telegraphic but unambiguous",
]

GLUE_PATTERNS = [
    r"after inspecting the diagram",
    r"keep unrelated regions unchanged",
    r"preserve object identity unless the instruction explicitly changes it",
    r"the target state must be knowledge plausible",
    r"do not alter any unrelated object",
    r"focus on reasoning correctness",
    r"preserve all unrelated visual context",
]

FORBIDDEN_TERMS = ["risebench", "rise-bench", "kris-bench", "krisbench", "kris bench"]

SYSTEM_PROMPT = """You rewrite annotations for an image-editing planner training set.
The edit itself is FROZEN: the target image is already rendered. You must keep the
exact same edit semantics (same object, same change, same preserved content).

Hard rules:
1. Never change WHAT is edited or claim any visual content beyond the given fields.
2. Remove template glue; write natural, varied, self-contained text.
3. Never mention benchmark names, test sets, or evaluation protocols.
4. instruction: rewrite in the requested style. It must still unambiguously request
   the same edit and require the same reasoning to solve. Do NOT leak the answer or
   the expected visual outcome into the instruction if the original instruction did
   not leak it (the user asks a question; the planner must reason out the answer).
5. editor_prompt: a concrete, directly executable instruction for a downstream image
   editor. It DOES state the resolved visual target (this is the planner's output,
   produced after reasoning), names the target region, and lists what must stay
   unchanged. It must not be a copy of the instruction.
6. rational_target_description: 2-4 sentences grounded in the given scene:
   current source state -> the knowledge/reasoning rule that applies -> the exact
   visual change that must appear -> what stays unchanged.
7. reasoning_chain: first-person planner reasoning (3-6 sentences): observe the
   source, recall/apply the needed knowledge, derive the target state, plan the
   region-aware edit. No meta-talk, no mention of "training" or "dataset".
8. edit_operations: same ops as given, but target / region_hint / change must be
   three DISTINCT, concrete strings (target = object phrase; region_hint = where in
   the image; change = what visually changes).
9. atomic_checklist: 4-6 task-specific yes/no questions across the four groups;
   each question must be answerable by looking at the edited image alone and must
   reference concrete objects/states from this task (no generic wording).
10. GROUNDING: when a scene_inventory is provided, the source image contains ONLY
   the listed present_elements, in the listed style. Never describe, reference, or
   imply any element from commonly_assumed_but_absent or anything else not listed.
   Match the stated visual style (these are flat programmatic diagrams, not photos).
   Only state object positions that appear in scene_inventory.layout; use that
   layout wording for region_hint so regions are spatially precise.
Output strict JSON only, no markdown fences."""

OUTPUT_SCHEMA_HINT = """Return JSON with exactly these keys:
{
  "instruction": str,
  "editor_prompt": str,
  "rational_target_description": str,
  "reasoning_chain": str,
  "edit_operations": [{"op": str, "target": str, "region_hint": str, "change": str, "preserve": [str]}],
  "atomic_checklist": {"cognitive": [str], "visual": [str], "preservation": [str], "readability": [str]},
  "risk_flags": [str]
}
Set risk_flags to ["semantic_drift_risk"] if you are not certain the rewrite keeps
the exact same edit semantics, else []."""

_print_lock = threading.Lock()


def load_benchmark_texts():
    texts = set()
    if BENCH_INDEX.exists():
        with open(BENCH_INDEX) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    t = row.get("normalized_text", "")
                    if t:
                        texts.add(t)
    return texts


def call_gpt55(prompt: str, effort: str, max_retries: int = 4, timeout: int = 300) -> str:
    payload = {
        "model": MODEL,
        "reasoning": {"effort": effort},
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
    }
    body = json.dumps(payload).encode()
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                API_URL,
                data=body,
                headers={
                    "Authorization": "Bearer dummy",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            return c.get("text", "")
            raise RuntimeError(f"no output_text in response status={data.get('status')}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(min(2 ** attempt * 3, 30))
    raise RuntimeError(f"gpt-5.5 call failed after {max_retries} retries: {last_err}")


def load_scene_inventory(extra_paths=None):
    merged = {}
    paths = [SCENE_INVENTORY] + [Path(p) for p in (extra_paths or [])]
    for path in paths:
        if path.exists():
            merged.update(json.loads(path.read_text()))
    return merged


def build_prompt(task: dict, style: str, scene_inventory: dict = None) -> str:
    ctx = {
        "task_family": task.get("task_family"),
        "sub_task": task.get("sub_task"),
        "domain": task.get("domain"),
        "knowledge_type": task.get("knowledge_type"),
        "instruction_original": task.get("instruction"),
        "expected_target": task.get("expected_target"),
        "rational_target_description_original": task.get("rational_target_description"),
        "required_knowledge": task.get("required_knowledge"),
        "source_scene_graph": task.get("source_scene_graph"),
        "edit_operations_original": task.get("edit_operations"),
        "preservation_constraints": task.get("preservation_constraints"),
        "negative_constraints": task.get("negative_constraints"),
    }
    # verifiable tasks (v2+) carry exact per-task facts — ground rewrites in them
    if task.get("ground_truth"):
        ctx["ground_truth"] = task["ground_truth"]
    if task.get("verifier_spec"):
        ctx["verifier_spec"] = task["verifier_spec"]
    if scene_inventory:
        ctx["scene_inventory"] = scene_inventory
    return (
        f"Instruction style to use: {style}\n\n"
        f"Task fields (ground truth, semantics frozen):\n"
        f"{json.dumps(ctx, ensure_ascii=False, indent=1)}\n\n"
        f"{OUTPUT_SCHEMA_HINT}"
    )


def parse_json_reply(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def validate_refined(task: dict, refined: dict, bench_texts: set) -> list:
    errors = []
    required = [
        "instruction",
        "editor_prompt",
        "rational_target_description",
        "reasoning_chain",
        "edit_operations",
        "atomic_checklist",
    ]
    for key in required:
        if key not in refined or not refined[key]:
            errors.append(f"missing:{key}")
    if errors:
        return errors

    if refined.get("risk_flags"):
        errors.append("model_flagged:" + ",".join(refined["risk_flags"]))

    norm_fields = {
        "instruction": normalize_text(refined["instruction"]),
        "editor_prompt": normalize_text(refined["editor_prompt"]),
        "rational": normalize_text(refined["rational_target_description"]),
        "reasoning": normalize_text(refined["reasoning_chain"]),
    }

    for name, text in norm_fields.items():
        if not text or len(text.split()) < 4:
            errors.append(f"too_short:{name}")
        for term in FORBIDDEN_TERMS:
            if term.replace("-", " ") in text:
                errors.append(f"forbidden_term:{name}")
        for pat in GLUE_PATTERNS:
            if re.search(pat, text):
                errors.append(f"glue_retained:{name}")
        if text in bench_texts:
            errors.append(f"benchmark_text_match:{name}")

    if norm_fields["editor_prompt"] == norm_fields["instruction"]:
        errors.append("editor_prompt_echoes_instruction")

    for op in refined["edit_operations"]:
        if op.get("target") == op.get("region_hint") == op.get("change"):
            errors.append("edit_op_still_boilerplate")
            break

    checklist = refined["atomic_checklist"]
    if not isinstance(checklist, dict):
        errors.append("checklist_not_dict")
    else:
        total_q = sum(len(v) for v in checklist.values() if isinstance(v, list))
        if total_q < 3 or total_q > 8:
            errors.append(f"checklist_count:{total_q}")

    # semantic anchor: at least one scene object must be referenced by a content word
    objects = (task.get("source_scene_graph") or {}).get("objects") or []
    if objects:
        combined = " ".join(norm_fields.values())
        anchor_tokens = {
            tok
            for obj in objects
            for tok in normalize_text(str(obj)).split()
            if len(tok) > 3
        }
        if anchor_tokens and not any(tok in combined for tok in anchor_tokens):
            errors.append(f"anchor_object_missing:{sorted(anchor_tokens)[:3]}")

    return errors


def style_for(task_id: str) -> str:
    digest = sum(ord(c) for c in task_id)
    return STYLES[digest % len(STYLES)]


def refine_one(task: dict, effort: str, bench_texts: set, inventories: dict) -> dict:
    style = style_for(task["task_id"])
    scene = inventories.get(task.get("sub_task", ""))
    reply = call_gpt55(build_prompt(task, style, scene), effort=effort)
    refined = parse_json_reply(reply)
    errors = validate_refined(task, refined, bench_texts)
    return {
        "task_id": task["task_id"],
        "style": style,
        "grounded": scene is not None,
        "refined": refined,
        "validation_errors": errors,
        "ok": not errors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--failures", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="low", choices=["minimal", "low", "medium", "high", "xhigh"])
    ap.add_argument("--stratify", action="store_true",
                    help="with --limit, sample evenly across sub_task instead of head-of-file")
    ap.add_argument("--inventory", nargs="*", default=None,
                    help="extra scene inventory JSON files merged over the v1 default")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path = Path(args.failures) if args.failures else out_path.with_suffix(".failures.jsonl")

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["task_id"])

    tasks = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    pending = [t for t in tasks if t["task_id"] not in done_ids]
    if args.limit:
        if args.stratify:
            by_sub = {}
            for t in pending:
                by_sub.setdefault(t.get("sub_task", "?"), []).append(t)
            quota = max(1, args.limit // max(len(by_sub), 1))
            sample = []
            for rows in by_sub.values():
                sample.extend(rows[:quota])
            pending = sample[: args.limit]
        else:
            pending = pending[: args.limit]

    print(f"total={len(tasks)} done={len(done_ids)} pending={len(pending)} "
          f"workers={args.workers} effort={args.effort}")
    if not pending:
        return

    bench_texts = load_benchmark_texts()
    inventories = load_scene_inventory(args.inventory)
    print(f"scene inventory entries: {len(inventories)}")
    stats = Counter()
    t0 = time.time()
    write_lock = threading.Lock()

    def worker(task):
        try:
            result = refine_one(task, args.effort, bench_texts, inventories)
        except Exception as exc:  # noqa: BLE001
            result = {"task_id": task["task_id"], "ok": False,
                      "validation_errors": [f"exception:{exc}"], "refined": None}
        with write_lock:
            if result["ok"]:
                with open(out_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                stats["ok"] += 1
            else:
                with open(fail_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                stats["fail"] += 1
            n = stats["ok"] + stats["fail"]
            if n % 20 == 0:
                rate = n / max(time.time() - t0, 1)
                with _print_lock:
                    print(f"[{n}/{len(pending)}] ok={stats['ok']} fail={stats['fail']} "
                          f"{rate:.2f} tasks/s eta={int((len(pending)-n)/max(rate,1e-9)/60)}min",
                          flush=True)
        return result["ok"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, pending))

    print(f"done: ok={stats['ok']} fail={stats['fail']} "
          f"elapsed={int(time.time()-t0)}s -> {out_path}")


if __name__ == "__main__":
    main()
