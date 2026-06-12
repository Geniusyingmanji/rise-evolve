#!/usr/bin/env bash
# v2 data pipeline: wait for the v1r refinement to release proxy capacity,
# then annotate v2 tasks (GPT-5.5), judge, assemble trajectories/splits,
# and run the decontamination gate.
set -uo pipefail
cd "$(dirname "$0")/../.."

TASKS=data/tasks/tasks_v2.jsonl
REFINED=data/tasks/tasks_v2_refined.jsonl
JUDGE_OUT=reports/data_quality/refine_v2_judge.jsonl
INV=data/taxonomy/scene_inventory_v2.json

echo "[v2] waiting for v1r refinement to finish"
while pgrep -f "refine_tasks_gpt55.py.*tasks_v1_refined_full" > /dev/null; do
  sleep 120
done
echo "[v2] proxy free, annotating $(wc -l < "$TASKS") v2 tasks"

python3 scripts/data/refine_tasks_gpt55.py \
  --tasks "$TASKS" \
  --output "$REFINED" \
  --inventory "$INV" \
  --workers 10 --effort low

echo "[v2] judging"
python3 scripts/data/judge_refined_pilot.py \
  --tasks "$TASKS" \
  --refined "$REFINED" \
  --output "$JUDGE_OUT" \
  --inventory "$INV" \
  --workers 10

echo "[v2] building trajectories and splits"
python3 scripts/data/build_trajectories_and_splits_v2.py \
  --tasks "$TASKS" \
  --refined "$REFINED" \
  --judge "$JUDGE_OUT" \
  --version v2

echo "[v2] diversity audit"
python3 scripts/data/audit_sft_diversity.py \
  --tasks "$TASKS" \
  --output reports/data_quality/diversity_audit_v2_seed.json || true

echo "[v2] decontamination gate"
python3 scripts/eval/check_decontamination.py \
  --benchmarks data/benchmarks \
  --train data/splits/sft_train_v2.jsonl data/splits/sft_val_v2.jsonl data/splits/rl_prompt_train_v2.jsonl \
  --fail-on high

echo "[v2] done"
wc -l data/trajectories/teacher_trajectories_v2.jsonl data/splits/sft_train_v2.jsonl \
  data/splits/sft_val_v2.jsonl data/splits/rl_prompt_train_v2.jsonl data/splits/hard_heldout_v2.jsonl
