# Reward Scripts

Lightweight RISE-Critic pipeline for smoke testing:

```bash
python3 scripts/reward/build_reward_items.py --version v1
python3 scripts/reward/run_rise_critic.py --version v1
python3 -m rise_evolve.reward.server --port 8766
```

The current critic uses schema checks plus programmatic render priors for the synthetic v1 teacher/negative images. This is useful for validating the reward data path and GRPO plumbing. Production RL should replace that prior with VLM difference-first scoring.
