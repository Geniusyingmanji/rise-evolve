# RISEvolve Engineering Plan: Training, Reward, Evaluation, and Decontamination

更新日期：2026-06-01

本文档把已有调研落到可执行工程计划上。目标不是照搬某一篇论文，而是把 GenEvolve 的 agentic training、FIRM/Edit-R1 的 image-editing reward/RL 经验、RewardHarness 的 agentic reward evolution、以及 LLaMA-Factory / verl 这类成熟训练框架组合成适合 RISE/GRADE/KRIS 图像编辑 benchmark 的训练与评测流水线。

## 0. 当前工程状态

已完成：

- 数据构造 pipeline：`scripts/data/`。
- v1 数据：10k tasks、10k teacher trajectories、10k edit programs、10k source images、10k teacher renders、10k negative renders、20k verifier items。
- 数据质检：`validate_dataset.py`、`audit_dataset.py`、review sheets、validation/audit reports。
- 方法文档：`training_plan.md`、`reward_design.md`、`data_pipeline.md`、`survey.md`。

尚未完成：

- SFT 数据转换脚本和 LLaMA-Factory 配置。
- Agent runtime，包括工具协议、schema 校验、rollout 记录。
- RISE-Critic reward server，包括 difference-first report、checklist verification、failure attribution。
- GRPO / Edit-OPD 训练入口，包括 rollout worker、reward bridge、experience memory、token/head-aware loss。
- RISE/GRADE/KRIS benchmark evaluation harness，包括 benchmark snapshot、run/render/score/report/decontamination gate。

结论：目前“数据和训练/评测设计”已经有基础，但“训练代码、训练脚本、benchmark eval 脚本”还需要按本文档落地。

## 1. 开源代码学习结论

### 1.1 GenEvolve

代码：`https://github.com/MeiGen-AI/GenEvolve`

可复用点：

- OpenAI-compatible agent runtime：`genevolve/agent.py` 用统一 client 调用 policy，并维护多轮 messages。
- ReAct 协议：`<think>`、`<tool_call>`、`<answer>`，方便 SFT 和 rollout 复用同一格式。
- 工具抽象：`search`、`image_search`、`query_knowledge` 均是可记录、可回放的 tool observation。
- 最终 answer schema：生成一个结构化 program，而不是直接输出自然语言。
- Evaluation 组织方式：agent 产出 program，独立 generator 渲染，独立 evaluator 打分。

边界：

- 公开仓库主要是 inference/runtime/evaluation，不包含完整训练脚本。
- 它的任务是 text-to-image generation；RISEvolve 必须加入 source image、edit region、preservation constraints、difference-based reward。

RISEvolve 改造：

- 新增 `analyze_image`、`ground_region`、`solve_symbolic`、`verify_edit`。
- 最终输出从 `gen_prompt/reference_images` 改为 `edit_program`。
- 轨迹 reward 不只看最终图，还看 source-image understanding、reasoning correctness、region locality、non-target preservation。

### 1.2 FIRM / Trust Your Critic

代码：`https://github.com/VisionXLab/FIRM-Reward`

可复用点：

- Difference-first reward：先描述 source 与 edited image 的差异，再评价 execution/consistency。
- Reward server 模式：训练进程只请求 reward service，critic 可独立升级。
- 编辑 reward 不能简单加权。FIRM 的 CME 是：

```text
R_CME = Execution * (0.6 + 0.4 * Consistency)
```

- 代码里已支持 multi-score 和 total-score fusion；Qwen-Image-Edit / FLUX Kontext 的 RL 脚本可作为 editor-side baseline。

边界：

- FIRM 主要训练底层 editor/generator，不直接训练 agent tool policy。
- Execution/Consistency 两头不足以覆盖 RISE/GRADE/KRIS 的 reasoning correctness、symbolic correctness、tool credit。

RISEvolve 改造：

- 把 CME 扩展成 cognitive-gated editing reward：

```text
G_task = min(R_exec, R_cog_applicable)

R_image =
  G_task * (
    0.45
    + 0.20 * R_preserve
    + 0.15 * R_region
    + 0.10 * R_quality
    + 0.10 * R_readability
  )

R_agent =
  0.45 * R_program
  + 0.45 * R_image
  + 0.05 * R_tool
  + 0.05 * R_format
```

- 新增 failure attribution，区分 `planner_fail`、`editor_fail`、`knowledge_fail`、`region_fail`、`over_edit`、`under_edit`、`judge_uncertain`。

### 1.3 Edit-R1

代码：`https://github.com/PKU-YuanGroup/Edit-R1`

可复用点：

- DiffusionNFT / RL fine-tuning image editor 的训练组织方式。
- 数据格式可借鉴：`images/`、`train_metadata.jsonl`、`test_metadata.jsonl`，每行绑定 source image、instruction、requirement。
- MLLM reward server 接入方式与训练脚本分离。

RISEvolve 结论：

- 第一阶段不直接训练 diffusion editor，先训练 agent planner。
- 如果 agent 已经能稳定产出正确 program，但 editor 执行能力成为瓶颈，再启动 editor-side RL baseline，并复用 RISE-Critic 作为 reward。

### 1.4 RewardHarness

代码：`https://github.com/TIGER-AI-Lab/RewardHarness`

可复用点：

- 把 reward modeling 视为 context/library evolution，而不是只训练 reward model weights。
- 用少量 preference demos 迭代出 Skills + Tools library，让 frozen VLM sub-agent 在评估时调用。
- 输出 scalar score，可接 GRPO reward normalization。

RISEvolve 改造：

- 把我们 v1 的 verifier pairs 和人工 review 样本变成 reward-demo set。
- 演化的是 `RISE-Critic` 的 checklist/rubric/tools，而不是 agent policy。
- 作为 reward prompt/library 的迭代机制，避免每次 reward 错误都重新训练 critic model。

### 1.5 LLaMA-Factory

用途：

- Stage 1 SFT cold start。
- 目标模型：Qwen3-VL-8B / Qwen2.5-VL-7B。
- 优先 LoRA/QLoRA 验证数据格式和收敛，再考虑全参或更大模型。

需要补的 adapter：

- 把 `data/splits/sft_train_v1.jsonl` 转成 LLaMA-Factory 多模态 conversation 格式。
- Assistant-only loss mask。
- 图像路径标准化。
- tool observations 作为 history，但不对 observation token 算 loss。

### 1.6 verl

用途：

- Stage 3/4 GRPO / Edit-OPD。
- verl 已有 GRPO trainer、group sampling、relative reward、Qwen2.5-VL/Qwen3-VL vision scripts、vLLM/SGLang rollout backend。

需要补的 adapter：

- Prompt dataset：source image + edit instruction + system prompt。
- Rollout worker：执行 agent tools，收集 full trace，生成 edit program。
- Renderer bridge：调用固定 image editor，把 program 渲染成 edited image。
- Reward bridge：调用 RISE-Critic server，返回 scalar reward + reward heads + attribution。
- Loss extension：第一版用普通 GRPO；第二版加 head-aware token credit 和 Edit-OPD reverse-KL/SDL。

## 2. 建议目标目录结构

```text
rise_evolve/
  agent/
    agent.py                  # GenEvolve-style ReAct runtime
    system_prompt.py
    schemas.py                # edit_program/tool_trace/reward schemas
    tools/
      analyze_image.py
      search.py
      image_search.py
      query_edit_knowledge.py
      ground_region.py
      solve_symbolic.py
  reward/
    prompts.py                # difference-first/checklist/failure-attribution prompts
    critic.py                 # RISE-Critic orchestration
    scoring.py                # gated reward fusion + head logging
    attribution.py
    server.py                 # HTTP/gRPC reward service
  training/
    sft_dataset.py
    rollout_worker.py
    reward_bridge.py
    grpo_loss.py
    edit_opd_loss.py
    experience_memory.py
  eval/
    benchmark_loaders.py
    decontamination.py
    run_agent.py
    render_edits.py
    score_outputs.py
    aggregate.py

configs/
  sft_qwen3vl_lora.yaml
  sft_qwen25vl_lora.yaml
  grpo_agent_debug.yaml
  grpo_agent_opd.yaml
  reward_rise_critic.yaml
  eval_rise.yaml
  eval_grade.yaml
  eval_kris.yaml

scripts/
  train/
    convert_sft.py
    convert_rl_prompts.py
    train_sft.sh
    train_grpo_debug.sh
    train_grpo_opd.sh
  reward/
    build_reward_items.py
    serve_rise_critic.py
    run_rise_critic.py
    calibrate_reward.py
  eval/
    prepare_benchmarks.py
    run_benchmark_agent.py
    render_benchmark_edits.py
    score_benchmark_outputs.py
    make_eval_report.py
    check_decontamination.py
```

## 3. 训练流程

### Phase A: 数据适配与锁定

输入：

- `data/splits/sft_train_v1.jsonl`
- `data/splits/sft_val_v1.jsonl`
- `data/splits/rl_prompt_train_v1.jsonl`
- `data/splits/verifier_train_v1.jsonl`
- `data/splits/ved_memory_train_v1.jsonl`

脚本：

- `scripts/train/convert_sft.py`
- `scripts/train/convert_rl_prompts.py`
- `scripts/reward/build_reward_items.py`

产物：

```text
data/train_ready/v1/sft_train_lf.json
data/train_ready/v1/sft_val_lf.json
data/train_ready/v1/rl_prompts.parquet
data/train_ready/v1/reward_items.jsonl
data/train_ready/v1/manifest.json
```

验收：

- SFT train/val 数量与 split 一致。
- 每条样本存在 source image。
- Assistant-only span 可定位。
- edit_program JSON 可解析率 > 99%。
- train/val/rl/verifier/heldout task id 无交叉。
- benchmark exact/minhash/image-phash gate 通过。

### Phase B: Agent runtime

目标：把训练、rollout、benchmark eval 共用同一个 agent runtime。

关键接口：

```python
run_agent(
    source_image: str,
    instruction: str,
    tools: ToolRegistry,
    max_tool_calls: int,
    output_schema: EditProgramSchema,
) -> AgentRun
```

`AgentRun` 必须保存：

- input id、image path、instruction。
- every message / tool call / observation。
- final edit_program。
- parse/repair status。
- token counts、latency、tool counts。

验收：

- dry-run mock tools 可在 100 条样本上稳定产出 schema。
- tool call parse 成功率 > 98%。
- final answer JSON parse 成功率 > 98%。
- 不允许 silent repair 进入训练 reward；repair 只能用于 eval/report 标注。

### Phase C: RISE-Critic reward server

Critic pipeline：

```text
source image + edited image + instruction + edit_program + tool_trace
  -> expected-diff planning
  -> observed-diff report
  -> checklist verification
  -> reward heads
  -> failure attribution
  -> scalar reward
```

Reward schema：

```json
{
  "task_id": "...",
  "score": 0.0,
  "heads": {
    "program": 0.0,
    "cognitive": 0.0,
    "execution": 0.0,
    "preservation": 0.0,
    "region": 0.0,
    "quality": 0.0,
    "readability": 0.0,
    "tool": 0.0,
    "format": 0.0
  },
  "attribution": {
    "primary_failure": "planner_fail|editor_fail|knowledge_fail|region_fail|over_edit|under_edit|judge_uncertain|none",
    "token_credit_hints": []
  },
  "difference_report": {
    "intended": [],
    "missing": [],
    "unintended": [],
    "implied": []
  },
  "checklist_results": []
}
```

第一版实现：

- `R_format/R_tool/R_program` 以规则和 schema 为主。
- `R_image` 用 VLM judge prompt + programmatic checks。
- 对程序生成图，增加模板级 rule checker，给 reward calibration 提供 anchor。

第二版实现：

- 引入 RewardHarness-style rubric evolution：用人工 review / v1 verifier pairs 更新 checklist library。
- 引入 critic ensemble：一个 VLM critic + 一个 rule/CLIP/OCR/checklist critic；分歧大时标注 `judge_uncertain` 并跳过 RL 更新。

验收：

- 在 verifier pairs 上，teacher render 胜过 negative render 的准确率 > 80%。
- 对 unchanged candidate 有显著惩罚，避免 lazy edit shortcut。
- 对 over-edit candidate 有显著惩罚，避免改图发散。
- 每个 reward head 的分布非塌缩，不能 90% 样本都在同一个分数桶。

### Phase D: SFT cold start

框架：LLaMA-Factory。

推荐配置：

```yaml
model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
stage: sft
finetuning_type: lora
template: qwen3_vl
cutoff_len: 32768
learning_rate: 1.0e-5
num_train_epochs: 2
bf16: true
freeze_vision_tower: true
freeze_multi_modal_projector: true
loss_mask: assistant_only
```

训练脚本：

```bash
bash scripts/train/train_sft.sh configs/sft_qwen3vl_lora.yaml
```

验收：

- 100 样本 overfit/smoke loss 正常下降。
- SFT val JSON parse success > 98%。
- hard heldout program sufficiency 高于 prompted baseline。
- 工具调用不出现循环或无效重复。

### Phase E: GRPO agent RL

框架：verl。

Rollout：

```text
prompt -> current policy samples K trajectories
       -> tools execute
       -> final edit_program
       -> fixed editor renders
       -> RISE-Critic scores
       -> group-relative advantage
       -> GRPO update
```

推荐 debug 配置：

```yaml
algorithm: grpo
rollouts_per_prompt: 4
train_batch_size: 8
max_tool_calls: 6
temperature: 0.7
top_p: 0.95
kl_coef: 1.0e-3
lr: 5.0e-7
editor_backend: mock_or_programmatic_first
reward_backend: rise_critic
```

推进顺序：

1. 只用 programmatic renderer 跑 50-100 steps，验证 RL 链路。
2. 接固定开源 editor，跑 200-500 steps，观察 reward heads。
3. 接强 editor / API editor，少量高质量 rollout。
4. 扩大到 v1 RL prompts。

验收：

- group 内 reward 方差正常，不全同分。
- KL 不爆，format success 不下降。
- `R_program`、`R_cognitive`、`R_region` 至少有一个 head 稳定提升。
- unchanged/over-edit 失败率不升高。

### Phase F: Edit-OPD / visual-cognitive experience distillation

创新点：把 OPD 从一般 language/task distillation 改成图像编辑特化的 on-policy visual-cognitive distillation。

流程：

```text
current policy samples K trajectories
  -> render and score
  -> select best/worst when reward gap >= threshold
  -> summarize experience:
       source image cue
       required reasoning fact
       correct edit region
       preservation rule
       observed failure of bad rollout
       critic attribution
  -> teacher branch sees retrieved experience
  -> student branch does not see it
  -> sampled-token reverse-KL / SDL on same on-policy tokens
```

Experience memory item：

```json
{
  "task_signature": "...",
  "image_cues": [],
  "reasoning_rule": "",
  "region_rule": "",
  "preservation_rule": "",
  "bad_rollout_failure": "",
  "critic_attribution": "",
  "best_program_excerpt": "",
  "reward_gap": 0.0
}
```

图像编辑特化 token mask：

- tool-call tokens：由 `R_tool/R_cognitive/R_program` 影响。
- reasoning tokens：由 `R_cognitive/R_program` 影响。
- region tokens：由 `R_region/R_preserve` 影响。
- preservation/negative constraint tokens：由 `R_preserve/over_edit` 影响。
- editor prompt tokens：由 `R_exec/R_quality/R_region` 影响。
- format tokens：由 `R_format` 影响。

验收：

- Edit-OPD 相比 GRPO-only 在 heldout reasoning categories 上提升。
- teacher context 只在训练时使用，推理时不依赖 memory。
- 不能把 benchmark examples 或 benchmark answers 写入 memory。

### Phase G: Editor-side RL baseline

只有在 agent planner 已经稳定但 editor 执行成为主瓶颈时启动。

推荐基线：

- FIRM-Reward / Edit-R1 风格的 Qwen-Image-Edit 或 FLUX Kontext RL。
- 数据用 agent 产出的高置信 program + source image + strong teacher render。
- Reward 仍用 RISE-Critic，但只回传 editor-side reward，不更新 agent。

验收：

- 固定 agent 下，editor-side RL 提升 execution/quality，不牺牲 preservation。
- 与 agent-side RL 分开报告，避免贡献混淆。

## 4. Benchmark 测试流程

原则：benchmark 只用于 eval，不用于训练、prompt evolution、reward evolution 或 hyperparameter search 的训练反馈。

统一流程：

```text
official benchmark snapshot
  -> lock revision/license/hash
  -> build eval-only manifest
  -> decontamination check against train data and memory
  -> run RISEvolve agent
  -> save edit_program + full tool trace
  -> render with fixed editor(s)
  -> score with official evaluator if available
  -> score with RISE-Critic
  -> human spot check
  -> aggregate per benchmark/category/domain
  -> failure attribution report
```

脚本：

- `scripts/eval/prepare_benchmarks.py`
  - 下载或导入 RISE/GRADE/KRIS。
  - 生成 normalized text hash、n-gram/minhash、image pHash、CLIP/DINO embeddings。
  - 产出 eval-only manifest。
- `scripts/eval/run_benchmark_agent.py`
  - 读取 manifest。
  - 调用 frozen checkpoint。
  - 保存 `programs.jsonl` 和 `tool_traces.jsonl`。
- `scripts/eval/render_benchmark_edits.py`
  - 用固定 editor backend 渲染。
  - 记录 editor 版本、seed、resolution、API/model id。
- `scripts/eval/score_benchmark_outputs.py`
  - 优先跑 official evaluator。
  - 同时跑 RISE-Critic，输出 reward heads 和 attribution。
- `scripts/eval/make_eval_report.py`
  - 汇总总体分、分类分、失败类型、样例页。
- `scripts/eval/check_decontamination.py`
  - 训练前和评测前都运行。

输出结构：

```text
outputs/eval/
  rise/{run_id}/
    manifest.json
    programs.jsonl
    tool_traces.jsonl
    renders/
    scores_official.jsonl
    scores_rise_critic.jsonl
    report.md
  grade/{run_id}/...
  kris/{run_id}/...
```

指标：

| Benchmark | 主指标 | 分解指标 |
|---|---|---|
| RISE | official accuracy / judge score | temporal、causal、spatial、logical、preservation、over-edit |
| GRADE | official score / judge score | domain score、knowledge correctness、diagram/text readability |
| KRIS | official score / judge score | factual plausibility、multi-element consistency、procedure/anomaly correctness |
| All | RISE-Critic score | program、cognitive、execution、preservation、region、quality、tool、format |

必须报告的 failure metrics：

- `planner_fail_rate`
- `knowledge_fail_rate`
- `region_fail_rate`
- `editor_fail_rate`
- `over_edit_rate`
- `under_edit_rate`
- `judge_uncertain_rate`
- `json_parse_fail_rate`
- `tool_loop_rate`

## 5. 防数据泄漏设计

### 5.1 目录隔离

```text
data/benchmarks/        # eval-only, read-only
data/train_ready/       # training inputs
data/experience/        # OPD memory, training-only
outputs/eval/           # benchmark outputs, never fed back into training
```

训练脚本必须拒绝读取：

- `data/benchmarks/**`
- `outputs/eval/**`
- 含有 `benchmark_id`、`benchmark_source_path`、`official_answer` 的 training row。

### 5.2 文本去污染

对 benchmark instruction / target / annotation 生成：

- exact normalized hash。
- 5-gram / 8-gram minhash。
- sentence embedding。
- template signature。

训练样本拒绝规则：

- normalized exact match。
- high n-gram overlap。
- high embedding similarity 且核心实体/数值/关系一致。
- 从 benchmark 改写但保留同一答案、同一布局、同一 entity set。

### 5.3 图像去污染

对 benchmark source/target/reference image 生成：

- file sha256。
- perceptual hash。
- CLIP/DINO embedding。
- OCR text hash。

训练样本拒绝规则：

- sha256 或 pHash 近重复。
- CLIP/DINO 高相似且 OCR/scene/object signature 接近。
- benchmark reference image 被 image search 直接选入 training trajectory。

### 5.4 Template/entity split

RISE/GRADE/KRIS 这种 benchmark 的泄漏不只来自原文，也可能来自模板。

必须拆分：

- RISE：object/process/template split，例如不要把同一 “fresh fruit after many days” 模板同时放 train 和 benchmark-near eval。
- GRADE：domain + concept + answer form split，例如光学折射、BST traversal、offside rule 不能用 benchmark 同题型同答案。
- KRIS：entity/procedure/anomaly split，例如同一常识规则、同一交通灯状态、同一餐具摆放规则要隔离。

### 5.5 Reward/OPD memory 防泄漏

禁止：

- 用 benchmark 样本演化 reward prompt/library。
- 把 benchmark eval failure 写入 OPD experience memory。
- 用 benchmark score 做 RL reward 或 early stopping 的训练反馈。

允许：

- 用 benchmark taxonomy 设计 broad categories。
- 用 benchmark official evaluator 做最终 reporting。
- 对 benchmark 结果做只读 error analysis，形成 paper 分析，但不能回灌训练。

### 5.6 CI gate

每次训练前必须运行：

```bash
python3 scripts/eval/check_decontamination.py \
  --benchmarks data/benchmarks \
  --train data/train_ready/v1 \
  --experience data/experience \
  --fail-on high
```

失败条件：

- exact text/image match。
- suspicious benchmark id/path/official answer 字段。
- high-similarity pair 未被人工确认豁免。
- eval output 被引用到 training manifest。

## 6. 第一阶段实施清单

优先级按“能尽快闭环训练和评测”排序：

1. `rise_evolve/agent/schemas.py`：定义 edit_program、tool_trace、reward schema。
2. `scripts/train/convert_sft.py`：v1 trajectory 到 LLaMA-Factory 数据。
3. `configs/sft_qwen3vl_lora.yaml`：SFT cold-start 配置。
4. `rise_evolve/agent/agent.py`：GenEvolve-style runtime skeleton。
5. `rise_evolve/reward/scoring.py`：RISE-Critic gated reward fusion。
6. `rise_evolve/reward/critic.py`：rule/VLM hybrid critic skeleton。
7. `scripts/reward/run_rise_critic.py`：离线评分 verifier pairs。
8. `scripts/eval/prepare_benchmarks.py`：冻结 benchmark manifest 和 fingerprints。
9. `scripts/eval/check_decontamination.py`：训练前 gate。
10. `scripts/eval/run_benchmark_agent.py`：dry-run + real checkpoint eval。
11. `scripts/eval/render_benchmark_edits.py`：固定 editor rendering。
12. `scripts/eval/score_benchmark_outputs.py`：official evaluator + RISE-Critic。
13. `scripts/train/train_sft.sh`：启动 SFT。
14. `scripts/train/train_grpo_debug.sh`：verl GRPO smoke。
15. `scripts/train/train_grpo_opd.sh`：Edit-OPD 第二阶段。

## 7. 我们的贡献表述

建议 paper/contribution 组织为三点：

1. **Reasoning-informed image editing agent**：把图像编辑从单步 prompt following 拆成 source understanding、knowledge/tool use、region-aware planning、editor execution，针对 RISE/GRADE/KRIS 的 reasoning bottleneck。
2. **RISE-Critic**：在 FIRM difference-first reward 基础上加入 cognitive gate、program/image dual heads、region preservation heads、failure attribution，实现适合 agent training 的 editing-aware reward。
3. **Edit-OPD**：把 on-policy distillation 改成图像编辑特化的 visual-cognitive experience distillation，用同 prompt best/worst rollout 的 visual difference、checklist failure、region/preservation rule 给 agent token-level guidance。

配套工程贡献：

- Eval-only benchmark harness。
- Decontamination-first data pipeline。
- Planner/editor failure attribution report。

## 8. 成功判据

短期：

- SFT 模型在 hard heldout 上 schema success > 98%，program sufficiency 高于 prompted baseline。
- RISE-Critic 在 verifier pairs 上 pairwise accuracy > 80%。
- GRPO debug 链路能跑通，并且 reward heads 不塌缩。

中期：

- Agent-side GRPO 在 v1 hard heldout 上提升 cognitive/program/region heads。
- Edit-OPD 相比 GRPO-only 提升 reasoning-heavy categories。
- Benchmark eval harness 能生成可复现报告，包含 run id、checkpoint、editor backend、benchmark revision。

长期：

- 在 RISE/GRADE/KRIS official or frozen benchmark 上稳定提升。
- failure attribution 显示提升来自 planner/reasoning/region，而不是 benchmark leakage 或 judge hacking。
