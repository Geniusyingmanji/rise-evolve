# Evaluation Scripts

Minimal benchmark harness:

```bash
python3 scripts/eval/prepare_benchmarks.py --benchmark rise
python3 scripts/eval/run_benchmark_agent.py \
  --manifest data/benchmarks/manifests/rise_eval_manifest.jsonl \
  --output-dir outputs/eval/rise/debug \
  --dry-run
python3 scripts/eval/score_benchmark_outputs.py \
  --programs outputs/eval/rise/debug/programs.jsonl \
  --output outputs/eval/rise/debug/scores_rise_critic.jsonl \
  --summary-output outputs/eval/rise/debug/summary.json
python3 scripts/eval/make_eval_report.py \
  --scores outputs/eval/rise/debug/scores_rise_critic.jsonl \
  --output outputs/eval/rise/debug/report.md
```

Run decontamination gates before training:

```bash
python3 scripts/eval/check_decontamination.py \
  --benchmarks data/benchmarks \
  --train data/splits/sft_train_v1.jsonl \
  --fail-on high
```

`run_benchmark_agent.py --dry-run` is a schema and report-path check. Real benchmark evaluation should replace it with a served checkpoint and a fixed editor backend.
