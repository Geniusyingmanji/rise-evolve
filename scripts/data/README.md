# Data Pipeline Scripts

Run from the repository root.

```bash
python3 scripts/data/freeze_benchmarks.py
python3 scripts/data/mine_taxonomy.py
python3 scripts/data/build_pilot_dataset.py --num-tasks 600 --version v0
python3 scripts/data/validate_dataset.py --version v0

# Scaled v1 run
python3 scripts/data/build_pilot_dataset.py --num-tasks 10000 --version v1 --seed 531
python3 scripts/data/validate_dataset.py --version v1
python3 scripts/data/audit_dataset.py --version v1 --sample-size 96
```

Main outputs:

- `data/benchmarks/benchmark_fingerprint.json`
- `data/taxonomy/benchmark_taxonomy.yaml`
- `data/tasks/tasks_v0.jsonl`
- `data/trajectories/teacher_trajectories_v0.jsonl`
- `data/programs/edit_programs_v0.jsonl`
- `data/renders/render_metadata_v0.jsonl`
- `data/splits/*_v0.jsonl`
- `reports/data_quality/summary_v0.md`
- `reports/data_quality/validation_v0.json`

Current v1 outputs:

- `data/tasks/tasks_v1.jsonl`: 10,000 tasks
- `data/trajectories/teacher_trajectories_v1.jsonl`: 10,000 teacher trajectories
- `data/programs/edit_programs_v1.jsonl`: 10,000 edit programs
- `data/renders/render_metadata_v1.jsonl`: 20,000 teacher/negative render records
- `data/verifier/verifier_items_v1.jsonl`: 20,000 verifier items
- `data/splits/sft_train_v1.jsonl`: 7,000 SFT trajectories
- `data/splits/rl_prompt_train_v1.jsonl`: 1,000 RL prompts
- `data/splits/verifier_train_v1.jsonl`: 2,000 verifier items
- `data/splits/ved_memory_train_v1.jsonl`: 300 experience pairs
- `reports/data_quality/summary_v1.md`
- `reports/data_quality/validation_v1.json`
- `reports/data_quality/audit_v1.json`
- `reports/data_quality/review_sample_v1.jsonl`
- `reports/data_quality/review_sheets/review_sheet_v1_*.png`
