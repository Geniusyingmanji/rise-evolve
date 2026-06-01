# Data Quality Summary v0

- Created at: `2026-05-31T16:50:12Z`
- Total tasks: `600`
- Source images: `600`
- Teacher renders: `600`
- Negative renders: `600`
- Benchmark fingerprint hash: `8a280d23eb75e5f5`

## Split Counts

- `hard_heldout`: 25
- `rl_prompt_train`: 100
- `sft_train`: 300
- `sft_val`: 50
- `ved_memory_train`: 25
- `verifier_train`: 100

## Benchmark Family

- `GRADE_like`: 210
- `KRIS_like`: 150
- `RISE_like`: 240

## Task Family

- `anomaly_correction`: 38
- `causal_reasoning`: 60
- `discipline_reasoning`: 210
- `entity_attribute_edit`: 38
- `logical_reasoning`: 60
- `multi_element_composition`: 37
- `procedural_knowledge`: 37
- `spatial_reasoning`: 60
- `temporal_reasoning`: 60

## Domain

- `biology`: 59
- `chemistry`: 21
- `computer_science`: 21
- `economics`: 21
- `everyday_common_sense`: 60
- `everyday_physics`: 60
- `everyday_procedure`: 37
- `geography`: 21
- `geometry`: 60
- `history`: 21
- `logic`: 60
- `math`: 21
- `music`: 21
- `physics`: 21
- `practical_knowledge`: 75
- `sports`: 21

## Gates

- Schema gate: all generated records contain task, recipe, trajectory, edit program, render metadata, verifier item, preference pair, and experience pair.
- Decontamination gate: exact normalized instruction check is run when `data/benchmarks/benchmark_text_index.jsonl` exists.
- Image gate: all source/teacher/negative images are programmatic internal images with average hashes recorded.
- Evidence gate: each task includes curated search queries and knowledge facts used in trajectory evidence maps.
- Current limitation: semantic text similarity and CLIP/DINO image similarity are placeholders until embedding dependencies are installed.

## Output Paths

- `data/tasks/tasks_v0.jsonl`
- `data/trajectories/teacher_trajectories_v0.jsonl`
- `data/programs/edit_programs_v0.jsonl`
- `data/renders/render_metadata_v0.jsonl`
- `data/splits/sft_train_v0.jsonl`
- `data/splits/rl_prompt_train_v0.jsonl`
- `data/splits/verifier_train_v0.jsonl`
- `data/splits/ved_memory_train_v0.jsonl`
