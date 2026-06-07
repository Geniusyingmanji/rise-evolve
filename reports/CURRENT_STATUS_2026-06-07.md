# RISEvolve Current Status - 2026-06-07

This note records the latest recoverable project state from `GOAL.md`,
`reports/RUNLOG.md`, `reports/EXPERIMENT_LOG.md`, saved JSON reports, and local
checkpoint metadata.

## Executive Conclusion

The v2r SFT run produced a schema-valid planner, but the current trained planner
does not yet support the project claim. In the first honest end-to-end RISE n20
comparison, the prompted base planner beat the SFT planner under the same fixed
editor and independent judge:

| System | RISE n20 accuracy | Instruction reasoning | Appearance consistency | Visual plausibility |
| --- | ---: | ---: | ---: | ---: |
| Prompted Qwen3-VL-8B base | 0.50 | 3.20 | 4.35 | 4.25 |
| v2r full-field SFT | 0.15 | 2.40 | 4.10 | 4.00 |

The root cause is data quality, not merely model format. The v2r training targets
were full-schema and parseable, but the reasoning-bearing fields were mostly
boilerplate: generic target descriptions, instruction echoes, and an over-strong
preservation/localization suffix. This taught the SFT planner to under-edit hard
reasoning prompts.

## Data Status

- v2r annotated tasks: 34,947.
- Accepted for SFT: 31,482, accept rate 90.08%.
- Split counts: SFT train 24,555; SFT val 1,259; RL prompts 3,148; VED 944;
  hard heldout 1,576.
- Text decontamination: pass, 27,703 rows checked against 3,580 benchmark text
  items, 0 hits.
- Main quality gap: severe diversity skew. The report contains only two task
  families, `multi_element_composition` 33,195 and `causal_reasoning` 1,752.
  The run log further identifies the practical edit-type skew as mostly addition
  plus tune-transfer.

## v3 Re-annotation Status

The v3 VLM re-annotation pivot is the right next step and has substantial output
already saved:

- Re-annotation shard lines: 31,252.
- Unique ok task ids: 30,885.
- Accepted SFT ids with an ok v3 re-annotation: 27,829 / 31,482,
  approximately 88.40%.
- Accepted SFT ids without an ok v3 re-annotation: 3,653.
- Failures: 39 `gen_error`, 1 `parse_fail`.
- Duplicate task ids: 227 ids, 337 extra duplicate lines.

The v3 annotations have not yet been merged into a rebuilt SFT snapshot, and no
v3-trained checkpoint exists in the visible workspace.

## Training Status

Latest visible checkpoint:

- `checkpoints/rise_planner_qwen3vl8b_lora_v2r_full`
- 2 epochs, 2,624 / 2,624 steps complete.
- Train loss: 0.05258.
- Runtime: 1:28:38.

The older `checkpoints/rise_planner_qwen3vl8b_lora_v2r` checkpoint is deprecated:
it was trained on the earlier 5-field target bug and should only be treated as a
negative-control baseline.

No RISEvolve training process is currently running, and all eight A800 GPUs were
idle during the recovery check.

## Data Location Note

In the visible workspace, the `data/` directory is absent and `git status` reports
tracked data files as deleted. Reports and checkpoints remain available in the
working repo. A v2r data copy was subsequently located at:

```text
/ky200t/datasets/zhouyan/share/quantaalpha/ymj/rise-evolve/data
```

That copy contains the required v2r task, quality, trajectory, split, and image
files needed by `scripts/train/convert_sft_lf.py`.

## Next Action

1. Make the located v2r data copy available to the working repo.
2. Rebuild the SFT snapshot with:

   ```bash
   python3 scripts/train/convert_sft_lf.py \
     --version v2r \
     --reann-glob 'reports/reann_v3_shard*.jsonl'
   ```

3. For a clean v3 SFT snapshot, use `--require-reann` to exclude accepted rows
   that still fall back to old boilerplate targets.
4. Validate the rebuilt ShareGPT rows for count, image existence, full
   10-field schema, and absence of the old boilerplate target text.
5. Re-run decontamination before launching any v3 SFT training.
