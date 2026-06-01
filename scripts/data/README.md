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

# Real-image source discovery and seed collection
python3 scripts/data/collect_real_edit_sources.py \
  --version v2_seed \
  --hf-per-source 12 \
  --wiki-per-query 2
python3 scripts/data/audit_real_sources.py --version v2_seed

# Randomized HF-only expansion example
python3 scripts/data/collect_real_edit_sources.py \
  --version v2_hf150 \
  --hf-per-source 30 \
  --skip-wikimedia \
  --randomize \
  --seed 601
python3 scripts/data/audit_real_sources.py --version v2_hf150 --sheet-limit 30

# Long-running quality-gated collection
python3 scripts/data/run_long_collection.py \
  --prefix v2_long_YYYYMMDD \
  --duration-hours 9 \
  --max-accepted 1000 \
  --hf-per-source 35 \
  --pause-seconds 120
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

Real-image seed outputs:

- `data/sources/real_edit_source_catalog.json`: searched and curated source catalog.
- `reports/data_sources/real_edit_source_report.md`: license/use summary.
- `data/sources/real_edit_pairs_sample_v2_seed.jsonl`: sampled source/target/instruction pairs from training-like public datasets.
- `data/sources/wikimedia_source_pool_v2_seed.jsonl`: licensed Wikimedia source/reference images.
- `data/tasks/real_seed_prompts_v2_seed.jsonl`: real-source candidate edit prompts that still need strong-editor targets.
- `data/real_edits/v2_seed/`: downloaded seed images.
- `reports/data_sources/real_source_audit_*.json`: automatic integrity, diversity, decontamination, and missing-image audit.
- `reports/data_sources/real_pair_sheet_*.png`: stratified source/target review sheet.

Current expanded HF sample:

- `data/sources/real_edit_pairs_sample_v2_hf150.jsonl`: 141 safety-filtered real edit pairs.
- `data/sources/real_edit_pairs_rejected_v2_hf150.jsonl`: 9 text-safety rejected pairs.
- `data/real_edits/v2_hf150/`: downloaded real edit pair images.
- `reports/data_sources/real_source_audit_v2_hf150.json`: audit report.

Long collection outputs:

- `data/sources/real_edit_pairs_candidate_<prefix>.jsonl`: cumulative accepted, quality-gated candidate pairs.
- `data/sources/real_edit_pairs_rejected_<prefix>.jsonl`: cumulative rejected rows with explicit reject reasons.
- `logs/data_collection/<prefix>.jsonl`: per-command and per-cycle execution log.
- `reports/data_sources/long_collection_status_<prefix>.json`: current job status.
- `reports/data_sources/decontamination_<prefix>.json`: text benchmark decontamination report for candidates.

The long runner rejects visual-input-dependent rows such as depth/reference/mask-guided edits because the current RISE/GRADE/KRIS-style training target is single-source-image editing. It also excludes reasoning-trace sources that are not reliable same-image before/after edit pairs.
