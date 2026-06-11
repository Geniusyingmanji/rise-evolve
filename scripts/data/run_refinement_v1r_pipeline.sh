#!/usr/bin/env bash
# Orchestrate the v1r refinement pipeline after refine_tasks_gpt55.py:
# incrementally judge refined rows while refinement runs, then merge,
# rebuild SFT splits, audit diversity, and run the decontamination gate.
set -uo pipefail
cd "$(dirname "$0")/../.."

REFINED=data/tasks/tasks_v1_refined_full.jsonl
JUDGE_OUT=reports/data_quality/refine_full_judge.jsonl
TOTAL_TASKS=$(wc -l < data/tasks/tasks_v1.jsonl)

refine_running() {
  pgrep -f "refine_tasks_gpt55.py.*tasks_v1_refined_full" > /dev/null
}

echo "[pipeline] waiting on refinement ($TOTAL_TASKS tasks total)"
while refine_running; do
  # judge whatever new rows exist, then wait
  python3 scripts/data/judge_refined_pilot.py \
    --tasks data/tasks/tasks_v1.jsonl \
    --refined "$REFINED" \
    --output "$JUDGE_OUT" \
    --workers 6 || true
  sleep 300
done
echo "[pipeline] refinement finished, final judge pass"

python3 scripts/data/judge_refined_pilot.py \
  --tasks data/tasks/tasks_v1.jsonl \
  --refined "$REFINED" \
  --output "$JUDGE_OUT" \
  --workers 10

echo "[pipeline] merging refined tasks + trajectories"
python3 scripts/data/merge_refined_tasks.py \
  --tasks data/tasks/tasks_v1.jsonl \
  --trajectories data/trajectories/teacher_trajectories_v1.jsonl \
  --refined "$REFINED" \
  --judge "$JUDGE_OUT" \
  --out-tasks data/tasks/tasks_v1r.jsonl \
  --out-trajectories data/trajectories/teacher_trajectories_v1r.jsonl

echo "[pipeline] rebuilding SFT splits with refined rows"
python3 scripts/data/merge_refined_tasks.py \
  --tasks data/tasks/tasks_v1.jsonl \
  --trajectories data/splits/sft_train_v1.jsonl \
  --refined "$REFINED" \
  --judge "$JUDGE_OUT" \
  --out-tasks /tmp/_ignore_tasks_train.jsonl \
  --out-trajectories data/splits/sft_train_v1r.jsonl

python3 scripts/data/merge_refined_tasks.py \
  --tasks data/tasks/tasks_v1.jsonl \
  --trajectories data/splits/sft_val_v1.jsonl \
  --refined "$REFINED" \
  --judge "$JUDGE_OUT" \
  --out-tasks /tmp/_ignore_tasks_val.jsonl \
  --out-trajectories data/splits/sft_val_v1r.jsonl

echo "[pipeline] diversity audit on refined tasks"
python3 scripts/data/audit_sft_diversity.py \
  --tasks data/tasks/tasks_v1r.jsonl \
  --output reports/data_quality/diversity_audit_v1r.json

echo "[pipeline] decontamination gate"
python3 scripts/eval/check_decontamination.py \
  --benchmarks data/benchmarks \
  --train data/splits/sft_train_v1r.jsonl data/splits/sft_val_v1r.jsonl \
  --fail-on high

echo "[pipeline] done"
wc -l data/tasks/tasks_v1r.jsonl data/trajectories/teacher_trajectories_v1r.jsonl \
  data/splits/sft_train_v1r.jsonl data/splits/sft_val_v1r.jsonl
