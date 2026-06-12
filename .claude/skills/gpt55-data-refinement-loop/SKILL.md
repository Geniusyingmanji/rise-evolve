---
name: gpt55-data-refinement-loop
description: Use when training annotations are boilerplate (template glue, field echoes, generic checklists), when a judge mass-rejects refined rows as "invented", when SFT data needs diversity/grounding repair, or when building new verifiable task data with VQA reward checks in this repo.
---

# GPT-5.5 data refinement factory (refine → judge → merge → splits → decontam)

## Overview
Semantics-frozen annotation repair: GPT-5.5 rewrites task annotations for diversity and grounded reasoning while the edit semantics (and existing renders) stay fixed; an independent GPT-5.5 judge filters; merge rebuilds tasks/trajectories/splits; decontamination gates the result. Boilerplate metrics go from 57–100% to 0 when run correctly (v1r: 9,543 rows, judge 98.8% semantic preservation).

## Pipeline commands (working reference)

```bash
python3 scripts/data/audit_sft_diversity.py --tasks TASKS.jsonl --output AUDIT.json   # quantify first
python3 scripts/data/refine_tasks_gpt55.py --tasks TASKS.jsonl --output REFINED.jsonl \
  --inventory data/taxonomy/scene_inventory_v2.json --workers 10 --effort low          # resume-safe
python3 scripts/data/judge_refined_pilot.py --tasks TASKS.jsonl --refined REFINED.jsonl \
  --output JUDGE.jsonl --inventory ... --workers 10                                    # resume-safe
python3 scripts/data/merge_refined_tasks.py ... (v1-style)  OR  build_trajectories_and_splits_v2.py (v2-style)
python3 scripts/eval/check_decontamination.py --benchmarks data/benchmarks --train SPLITS... --fail-on high
```
Run long passes via a script file under `setsid nohup bash script.sh > log &` (inline `nohup bash -c` died silently mid-run once; script files survived hours).

## Grounding rules (each learned from a real mass-failure)

| Rule | Failure it prevents |
|---|---|
| Pilot 40–60 stratified rows FIRST; judge them; only then full run | full-run waste; v1 pilot caught 47.7% hallucination before scale |
| Rewriter must receive a code-verified scene inventory (elements + 9-zone layout + absent-but-assumed list) | rewriter invents scene details (sun, labels, soil) → judge flags ~48% |
| Judge must receive the SAME grounding payload as the rewriter (inventory, ground_truth) | judge false-rejects grounded text (invented_claims jumped 24%→100% on payload mismatch) |
| Per-task randomized facts (`ground_truth`, `verifier_spec`) must go to BOTH rewriter and judge | judge decimated randomized families: sequence_pattern 140/144 rejected, graph_path 117/150 |
| VQA verifier questions must be READING-ONLY + format-pinned ("Answer digits only") | VLM computes the answer instead of reading → negatives falsely pass (geometry_angle) |
| Verify teacher AND negative renders: teacher must pass VQA, negative must fail | silent render bugs (sorting_step teacher never applied the swap; pixel-diff self-test missed it) |
| Compare VQA answers as token sequences (strip `-> , [ ] and`) | format-only failures: 'lit' vs 'on', '[2, 3]' vs '2,3' |
| Judge-rejected rows: don't discard — re-run with richer grounding payload | recoverable rows; family-balance collapse |

## Quality gates (numbers from production runs)
- Judge thresholds: same_edit ≥97%, leak <3%, invented <10% on pilot before scaling.
- Expect ~0.7–0.9 tasks/s at 8–10 workers; ~5% refine validation failures are normal.
- Always re-audit diversity post-merge (all boilerplate ratios must be 0) and re-run decontamination.

## Red flags
- invented_claims suddenly ≫ pilot level → grounding payload mismatch between rewriter and judge.
- One sub_task nearly wiped out in splits summary → per-task facts missing from judge payload.
- Background pass log stops growing → process silently killed; relaunch as setsid script (resume is safe).
