# Data Quality Summary v1

- Created at: `2026-05-31T17:12:57Z`
- Total tasks: `10000`
- Source images: `10000`
- Teacher renders: `10000`
- Negative renders: `10000`
- Unique source aHash buckets: `287`
- Unique teacher aHash buckets: `295`
- Unique negative aHash buckets: `248`
- Benchmark fingerprint hash: `8a280d23eb75e5f5`

## Split Counts

- `hard_heldout`: 200
- `rl_prompt_train`: 1000
- `sft_train`: 7000
- `sft_val`: 500
- `ved_memory_train`: 300
- `verifier_train`: 1000

## Benchmark Family

- `GRADE_like`: 3500
- `KRIS_like`: 2500
- `RISE_like`: 4000

## Task Family

- `anomaly_correction`: 625
- `causal_reasoning`: 1000
- `discipline_reasoning`: 3500
- `entity_attribute_edit`: 625
- `logical_reasoning`: 1000
- `multi_element_composition`: 625
- `procedural_knowledge`: 625
- `spatial_reasoning`: 1000
- `temporal_reasoning`: 1000

## Domain

- `biology`: 975
- `chemistry`: 350
- `computer_science`: 350
- `economics`: 350
- `everyday_common_sense`: 1000
- `everyday_physics`: 1000
- `everyday_procedure`: 625
- `geography`: 350
- `geometry`: 1000
- `history`: 350
- `logic`: 1000
- `math`: 350
- `music`: 350
- `physics`: 350
- `practical_knowledge`: 1250
- `sports`: 350

## Gates

- Schema gate: all generated records contain task, recipe, trajectory, edit program, render metadata, verifier item, preference pair, and experience pair.
- Decontamination gate: exact normalized instruction check is run when `data/benchmarks/benchmark_text_index.jsonl` exists.
- Image gate: all source/teacher/negative images are programmatic internal images with average hashes recorded.
- Evidence gate: each task includes curated search queries and knowledge facts used in trajectory evidence maps.
- Current limitation: semantic text similarity and CLIP/DINO image similarity are placeholders until embedding dependencies are installed.

## Output Paths

- `data/tasks/tasks_v1.jsonl`
- `data/trajectories/teacher_trajectories_v1.jsonl`
- `data/programs/edit_programs_v1.jsonl`
- `data/renders/render_metadata_v1.jsonl`
- `data/splits/sft_train_v1.jsonl`
- `data/splits/rl_prompt_train_v1.jsonl`
- `data/splits/verifier_train_v1.jsonl`
- `data/splits/ved_memory_train_v1.jsonl`
