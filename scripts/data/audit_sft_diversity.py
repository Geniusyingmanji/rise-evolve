#!/usr/bin/env python3
"""Audit diversity and boilerplate in RISEvolve task/program data.

Quantifies the failure modes identified in the v2r postmortem so they can be
tracked before/after refinement:

- task family / sub_task / domain / difficulty distribution skew
- instruction template glue (fixed prefixes/suffixes) and n-gram duplication
- edit_operations boilerplate (change == region_hint == target)
- generic checklist question ratio
- rational_target_description template ratio
- editor_prompt instruction-echo ratio (when a programs file is given)

Usage:
  python3 scripts/data/audit_sft_diversity.py --tasks data/tasks/tasks_v1.jsonl \
    --programs data/programs/edit_programs_v1.jsonl \
    --output reports/data_quality/diversity_audit_v1.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from common import normalize_text  # noqa: E402

GLUE_PATTERNS = [
    r"^after inspecting the diagram",
    r"^study the image first",
    r"^look at the image carefully",
    r"keep unrelated regions unchanged",
    r"preserve object identity unless the instruction explicitly changes it",
    r"the target state must be knowledge plausible",
    r"do not alter any unrelated object",
    r"focus on reasoning correctness",
    r"preserve all unrelated visual context",
]

GENERIC_CHECKLIST = [
    "is the result consistent with the required reasoning or discipline knowledge",
    "are unrelated objects background layout lighting and viewpoint preserved",
    "is the edited image visually clear and free of unrelated additions",
    "does the edit correctly perform this operation",
]

RATIONAL_TEMPLATE = r"this requires [a-z_ ]+ reasoning"


def ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])


def jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def audit_tasks(rows):
    n = len(rows)
    fam = Counter(r.get("task_family", "?") for r in rows)
    sub = Counter(r.get("sub_task", "?") for r in rows)
    dom = Counter(r.get("domain", "?") for r in rows)
    bench = Counter(r.get("benchmark_family", "?") for r in rows)
    diff = Counter(str((r.get("difficulty") or {}).get("level", "?")) for r in rows)

    instr_norm = [normalize_text(r.get("instruction", "")) for r in rows]
    uniq_instr = len(set(instr_norm))

    glue_hits = Counter()
    for s in instr_norm:
        for pat in GLUE_PATTERNS:
            if re.search(pat, s):
                glue_hits[pat] += 1
    any_glue = sum(
        1 for s in instr_norm if any(re.search(p, s) for p in GLUE_PATTERNS)
    )

    gram8 = Counter()
    for s in instr_norm:
        toks = s.split()
        gram8.update(" ".join(g) for g in set(ngrams(toks, 8)))
    dup_grams = {g: c for g, c in gram8.items() if c >= max(20, n // 100)}

    op_boiler = 0
    op_total = 0
    for r in rows:
        for op in r.get("edit_operations", []):
            op_total += 1
            if op.get("change") == op.get("region_hint") == op.get("target"):
                op_boiler += 1

    generic_q = 0
    q_total = 0
    uniq_q = set()
    for r in rows:
        checklist = r.get("atomic_checklist", [])
        if isinstance(checklist, dict):
            checklist = [x for v in checklist.values() for x in v]
        for item in checklist:
            q = normalize_text(item.get("question", ""))
            q_total += 1
            uniq_q.add(q)
            if any(q.startswith(g) or g in q for g in GENERIC_CHECKLIST):
                generic_q += 1

    rat_template = sum(
        1
        for r in rows
        if re.search(RATIONAL_TEMPLATE, normalize_text(r.get("rational_target_description", "")))
    )
    rat_echo = sum(
        1
        for r in rows
        if normalize_text(r.get("rational_target_description", "")).startswith(
            normalize_text(r.get("expected_target", ""))[:60]
        )
        and r.get("expected_target")
    )

    return {
        "num_tasks": n,
        "task_family_distribution": dict(fam.most_common()),
        "sub_task_distribution": dict(sub.most_common()),
        "domain_distribution": dict(dom.most_common()),
        "benchmark_family_distribution": dict(bench.most_common()),
        "difficulty_distribution": dict(diff.most_common()),
        "instruction_unique_ratio": round(uniq_instr / max(n, 1), 4),
        "instruction_glue_any_ratio": round(any_glue / max(n, 1), 4),
        "instruction_glue_pattern_hits": {k: v for k, v in glue_hits.most_common()},
        "instruction_top_repeated_8grams": dict(
            sorted(dup_grams.items(), key=lambda kv: -kv[1])[:25]
        ),
        "edit_op_boilerplate_ratio": round(op_boiler / max(op_total, 1), 4),
        "checklist_generic_ratio": round(generic_q / max(q_total, 1), 4),
        "checklist_unique_questions": len(uniq_q),
        "checklist_total_questions": q_total,
        "rational_template_ratio": round(rat_template / max(n, 1), 4),
        "rational_expected_echo_ratio": round(rat_echo / max(n, 1), 4),
    }


def audit_programs(rows, tasks_by_id):
    n = 0
    echo = 0
    uniq_prompts = set()
    for r in rows:
        prog = r.get("final_edit_program") or r
        prompt = normalize_text(prog.get("editor_prompt", ""))
        if not prompt:
            continue
        n += 1
        uniq_prompts.add(prompt)
        task = tasks_by_id.get(r.get("task_id"))
        if task:
            instr = normalize_text(task.get("instruction", ""))
            if instr and (instr in prompt or prompt in instr):
                echo += 1
    return {
        "num_programs": n,
        "editor_prompt_unique_ratio": round(len(uniq_prompts) / max(n, 1), 4),
        "editor_prompt_instruction_echo_ratio": round(echo / max(n, 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--programs", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tasks = list(jsonl(args.tasks))
    report = {"tasks_file": args.tasks, "task_audit": audit_tasks(tasks)}

    if args.programs:
        tasks_by_id = {t["task_id"]: t for t in tasks}
        report["programs_file"] = args.programs
        report["program_audit"] = audit_programs(list(jsonl(args.programs)), tasks_by_id)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    ta = report["task_audit"]
    print(f"tasks: {ta['num_tasks']}")
    print(f"instruction_unique_ratio: {ta['instruction_unique_ratio']}")
    print(f"instruction_glue_any_ratio: {ta['instruction_glue_any_ratio']}")
    print(f"edit_op_boilerplate_ratio: {ta['edit_op_boilerplate_ratio']}")
    print(f"checklist_generic_ratio: {ta['checklist_generic_ratio']}")
    print(f"rational_template_ratio: {ta['rational_template_ratio']}")
    print(f"rational_expected_echo_ratio: {ta['rational_expected_echo_ratio']}")
    print(f"top families: {list(ta['task_family_distribution'].items())[:5]}")
    if "program_audit" in report:
        pa = report["program_audit"]
        print(f"editor_prompt_echo_ratio: {pa['editor_prompt_instruction_echo_ratio']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
