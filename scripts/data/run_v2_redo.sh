#!/usr/bin/env bash
# Redo refinement for v2 tasks dropped by the first pass (ground-truth-grounded).
set -uo pipefail
cd "$(dirname "$0")/../.."

python3 scripts/data/refine_tasks_gpt55.py \
  --tasks /tmp/tasks_v2_redo.jsonl \
  --output data/tasks/tasks_v2_refined_redo.jsonl \
  --inventory data/taxonomy/scene_inventory_v2.json \
  --workers 10 --effort low

python3 scripts/data/judge_refined_pilot.py \
  --tasks data/tasks/tasks_v2.jsonl \
  --refined data/tasks/tasks_v2_refined_redo.jsonl \
  --output reports/data_quality/refine_v2_judge_redo.jsonl \
  --inventory data/taxonomy/scene_inventory_v2.json \
  --workers 10

python3 - <<'PY'
import json
redo_refined = {json.loads(l)["task_id"]: l for l in open("data/tasks/tasks_v2_refined_redo.jsonl")}
with open("data/tasks/tasks_v2_refined_combined.jsonl", "w") as out:
    for l in open("data/tasks/tasks_v2_refined.jsonl"):
        if json.loads(l)["task_id"] in redo_refined:
            continue
        out.write(l)
    for l in redo_refined.values():
        out.write(l)
redo_judge = {json.loads(l)["task_id"]: l for l in open("reports/data_quality/refine_v2_judge_redo.jsonl")}
with open("reports/data_quality/refine_v2_judge_combined.jsonl", "w") as out:
    for l in open("reports/data_quality/refine_v2_judge.jsonl"):
        if json.loads(l)["task_id"] in redo_judge:
            continue
        out.write(l)
    for l in redo_judge.values():
        out.write(l)
print("combined refined + judge files")
PY

python3 scripts/data/build_trajectories_and_splits_v2.py \
  --tasks data/tasks/tasks_v2.jsonl \
  --refined data/tasks/tasks_v2_refined_combined.jsonl \
  --judge reports/data_quality/refine_v2_judge_combined.jsonl \
  --version v2

python3 scripts/eval/check_decontamination.py \
  --benchmarks data/benchmarks \
  --train data/splits/sft_train_v2.jsonl data/splits/sft_val_v2.jsonl data/splits/rl_prompt_train_v2.jsonl \
  --fail-on high

echo "[redo] done"
