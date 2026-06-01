# Training Script Stubs

These scripts prepare data and configs for the first RISEvolve training loop.

```bash
python3 scripts/train/convert_sft.py --version v1 --split train
python3 scripts/train/convert_sft.py --version v1 --split val
python3 scripts/train/convert_rl_prompts.py --version v1
```

`convert_sft.py` writes LLaMA-Factory-style ShareGPT records with source-image paths and assistant-only ReAct supervision. `convert_rl_prompts.py` writes rollout prompts for the later verl GRPO worker.

Actual LLaMA-Factory and verl launch commands are intentionally kept in `configs/` until the cluster paths, model paths, and serving backend are fixed.
