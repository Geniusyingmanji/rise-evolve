# RISEvolve Training Plan: GRPO + Edit-OPD for Image Editing Agents

更新日期：2026-06-01

目标：训练一个面向 RISEBench / GRADE / KRIS-Bench 的图像编辑规划 agent。训练对象优先是 VLM agent policy，而不是直接训练底层 diffusion editor。Agent 负责源图分析、知识检索、参考图选择、区域规划、checklist 合成和 editor prompt 生成；下游 editor 先保持固定。

## 1. 从 GenEvolve 学到的训练范式

GenEvolve 将开放式图像生成建模为工具编排轨迹：

```text
user request
  -> search / image_search / query_knowledge
  -> prompt-reference program z = (gen_prompt, reference_images)
  -> reference-conditioned generator
  -> image reward + text/program reward
```

公开信息核实：

- 论文：`GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation`, arXiv:2605.21605。
- 代码：`https://github.com/MeiGen-AI/GenEvolve`，公开了 inference runtime、三类工具、8 个 skill markdown、Qwen-Image-Edit/Nano Banana Pro wrapper、evaluation 脚本。
- 模型：`MeiGen-AI/GenEvolve`，Qwen3-VL-8B-based policy。
- 数据：`MeiGen-AI/GenEvolve-Data-Bench`，SFT 9,000 trajectories、RL 3,175 prompts + GT images、Bench 594 prompts。
- 边界：公开 repo 明确说明 full training scripts are not included；训练细节主要来自论文和 released data。

训练方法拆解：

1. SFT cold start：用多轮 ReAct 轨迹训练 Qwen3-VL-8B 的 language-policy 部分，freeze vision tower/projector，只对 assistant tokens 计算 loss。
2. On-policy rollout：每个 prompt 采样 K 条完整轨迹，轨迹包括工具调用、最终 program、rendered image。
3. Dual reward：`R = 0.5 R_img + 0.5 R_text`。`R_img` 是图像侧 KScore-style judge，`R_text` 是 program sufficiency judge。
4. GRPO：用 group-relative advantage 优化 assistant tokens，包括 tool decisions 和 final program tokens。
5. Visual Experience Distillation：同 prompt 的 best/worst rollout 若 reward gap 足够大，则抽取 visual experience bundle。Teacher branch 看到 retrieved experience，student branch 不看；teacher 不另采样，只对同一批 on-policy tokens 重新打分，通过 sampled-token reverse-KL / SDL 给 dense token guidance。
6. 推理时只部署 student，不需要动态 visual experience memory。

关键超参参考：

| 项 | GenEvolve 设置 | RISEvolve 初始建议 |
|---|---:|---:|
| Backbone | Qwen3-VL-8B-Instruct | Qwen3-VL-8B / Qwen2.5-VL-7B |
| SFT cutoff | 32,768 | 32,768 |
| SFT LR | 1e-5 | 1e-5 |
| RL rollouts | 8 prompts × 6 rollouts | 4-8 prompts × 4-6 rollouts |
| Sampling | temp 0.7, top-p 0.95 | temp 0.7, top-p 0.9-0.95 |
| GRPO LR | 1e-6 | 5e-7 - 1e-6 |
| Clip | low 0.20, high 0.28 | same |
| KL | 1e-3 | 1e-3, plus SFT replay |
| SDL coefficient | 2.0 | 0.5-2.0 sweep |
| Experience min gap | 0.20 | 0.15-0.25 |
| SDL token mask | top 10% teacher-student disagreement | decision-token top 10-20% |

## 2. Why Train the Agent First

图像编辑训练有两条路线：

1. 直接 post-train diffusion / flow editor。
2. 固定 editor，训练前置 agent / planner。

RISEvolve 应先走第二条。原因：

- RISE/GRADE/KRIS 的主要瓶颈是“知道该怎么改”，不是局部纹理生成。
- Agent policy 的行动空间是离散 token/tool/program，适合 GRPO、OPD、DPO 等语言策略训练。
- 固定 editor 可以把失败归因拆开：planner wrong、region wrong、knowledge wrong、editor execution wrong。
- 若一开始训练 diffusion editor，reward 稀疏、成本高、保真和指令遵循会互相拉扯。

底层 editor 可以后续作为第二阶段优化对象，优先使用 reward-weighted fine-tuning、DPO/LPO、Flow-OPD，而不是从零开始做 pixel-level PPO。

## 3. RISEvolve Agent Action Space

输入：

```text
source_image + edit_instruction
```

工具：

| Tool | 作用 | 是否进入训练 reward |
|---|---|---|
| `analyze_image(image, focus)` | 源图对象、属性、关系、当前状态、可编辑区域 | 是 |
| `search(queries)` | GRADE/KRIS/RISE 知识和事实检索 | 是 |
| `image_search(query)` | 目标外观、材料、历史外观、图示参考 | 是 |
| `query_edit_knowledge(skill)` | temporal/causal/spatial/logical/discipline/region/preserve skill | 是 |
| `ground_region(image, target)` | bbox/mask/region phrase，服务局部保真 | 是 |
| `solve_symbolic(problem)` | 数独、棋盘、数学、迷宫、逻辑约束 | 是 |
| `verify_edit(...)` | 训练/eval 的 checklist judge，不作为推理必需工具 | 间接 |

最终输出 `edit_program`：

```json
{
  "source_scene_graph": {},
  "task_family": "temporal|causal|spatial|logical|discipline|mixed",
  "knowledge_facts": [],
  "target_scene_description": "",
  "edit_operations": [],
  "reference_images": [],
  "preservation_constraints": [],
  "negative_constraints": [],
  "atomic_checklist": {
    "cognitive": [],
    "visual": [],
    "preservation": [],
    "readability": []
  },
  "editor_prompt": "",
  "failure_modes_to_watch": []
}
```

## 4. Training Stages

### Stage 0: Prompted Baseline and Failure Taxonomy

不训练。用强 VLM planner 生成 `edit_program`，接固定 editor 跑 100-200 个 RISE/GRADE/KRIS-style dev cases。

产出：

- editor upper bound 估计；
- planner fail / region fail / knowledge fail / editor fail / judge fail 分类；
- reward checklist prompt 初版；
- 工具预算和是否需要外部 search 的边界。

### Stage 1: SFT Cold Start

目标：让 agent 稳定学会 ReAct、工具协议、edit program schema、checklist 和区域保真写法。

数据：

- 当前 v1 的 7,000 SFT trajectories 可作为 format bootstrap。
- 后续加入 1K-3K real-image / strong-editor teacher trajectories，避免程序图分布过窄。
- 混入基础编辑数据改造样本，但权重低于 RISE/GRADE/KRIS-style hard cases。

训练配置：

```yaml
framework: LLaMA-Factory
backbone: Qwen3-VL-8B-Instruct
trainable: language_policy_only
freeze_vision_tower: true
freeze_projector: true
cutoff_len: 32768
learning_rate: 1e-5
epochs: 2
precision: bf16
loss_mask: assistant_tokens_only
```

验收：

- JSON parse success > 98%。
- 必选字段完整率 > 98%。
- 工具调用重复率 < 10%。
- 在 dev cases 上比 direct editor/prompted rewrite 有明显 program sufficiency 提升。

### Stage 2: Verifier and Reward Calibration

目标：先把 reward 稳住，再进入 GRPO。

Reward 采用 checklist-first + difference-first，不直接让 VLM 给总分。`Trust Your Critic / FIRM` 证明了一个关键风险：编辑 reward 若用 `0.5 * execution + 0.5 * consistency`，模型会学会几乎不编辑来骗取高 consistency。FIRM 的 CME 用 `Execution * (0.6 + 0.4 * Consistency)` 把 execution 变成高 reward 的必要条件。RISEvolve 需要把这个思想扩展到 reasoning editing。

1. 从 instruction + source analysis + target description 生成 atomic checklist。
2. 对 source/edited 做 difference-first observation，先描述 intended/missing/unintended/implied changes。
3. 逐项判断 edited image 是否满足 checklist。
4. 输出 reward heads 和 failure attribution。

Reward heads：

```text
R_cognitive: 推理/学科/事实/符号结果是否正确
R_visual: 目标变化是否完成，图像是否自然
R_preserve: 非编辑区域、身份、视角、光照、背景是否保留
R_region: edit region 是否准确，是否 over-edit / under-edit
R_tool: 工具调用是否必要、有效、无重复，证据是否被 program 使用
R_format: JSON/schema/checklist/reference binding 是否合规
R_program: 不看 render，仅看 edit program 是否逻辑正确、可执行、证据充分
```

不建议简单 weighted sum。建议使用 gated/base-and-bonus reward：

```text
G_task = min(R_exec, R_cognitive_applicable)

R_image =
  G_task * (
    0.45
    + 0.20 R_preserve
    + 0.15 R_region
    + 0.10 R_quality
    + 0.10 R_readability
  )

R_agent =
  0.45 R_program
  + 0.45 R_image
  + 0.05 R_tool
  + 0.05 R_format
```

训练时对同一 prompt 的 K 个 rollout 做 group 内归一化。日志中必须保留每个 head 的分数，避免一个高分 head 淹没另一个失败 head。

如果判定为 `editor_fail`，即 plan/checklist/region 正确但 editor 没执行出来，不应把全部负反馈压到 reasoning/tool tokens。建议保留较高 `R_program`，主要降低 `editor_prompt`、`region phrase`、`negative_constraints` 相关 token 的优势。详细 reward 方案见 `reward_design.md`。

### Stage 3: Edit-GRPO for Agent Policy

目标：优化工具选择、推理、区域规划和 editor prompt tokens。

Rollout：

```yaml
prompts_per_step: 4-8
rollouts_per_prompt: 4-6
temperature: 0.7
top_p: 0.9-0.95
max_tool_calls: 6-8
max_response_tokens: 12000-20000
editor: fixed Qwen-Image-Edit / FLUX.1 Kontext / GPT-4o-Image backend
```

GRPO：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
L = L_GRPO + beta_kl KL(policy || ref_policy)
```

Token credit 建议分区：

- tool-call tokens：主要受 `R_tool + R_cognitive` 影响；
- reasoning / target description tokens：主要受 `R_cognitive` 影响；
- region / preserve tokens：主要受 `R_region + R_preserve` 影响；
- editor_prompt tokens：主要受 `R_visual + R_preserve` 影响；
- format tokens：主要受 `R_format` 影响。

第一版可以用统一 rollout reward，第二版再做 token mask/head-aware advantage。

结合 reward heads 后，第二版建议：

```text
tool_call tokens               <- R_tool + R_cognitive + R_program
reasoning / knowledge tokens   <- R_cognitive + R_program
region tokens                  <- R_region + R_preserve
preservation / negative tokens <- R_preserve + over_edit penalty
editor_prompt tokens           <- R_exec + R_region + R_quality
format tokens                  <- R_format
```

### Stage 4: Edit-OPD / Visual-Cognitive Experience Distillation

用户提到的 OPD 在近期文献中主要指 On-Policy / Online Policy Distillation。与 DPO/IPO 这类离线偏好优化不同，OPD 用当前 policy 的 on-policy rollout 作为蒸馏路径，teacher 和 student 在同一批 tokens 上比较分布，从而降低 train-inference mismatch。

RISEvolve 的二维图像编辑适配称为 `Edit-OPD`：

```text
current student samples K edit trajectories
  -> render edited images
  -> verifier scores + diagnostics
  -> choose best/worst if reward gap >= threshold
  -> summarize visual-cognitive experience
  -> teacher branch sees retrieved experience
  -> teacher re-scores same student tokens
  -> student minimizes GRPO + lambda_OPD * token-level distillation loss
```

Experience slots：

| Slot | 内容 |
|---|---|
| `M_analyze` | 源图观察、状态判断、关键对象和 region 识别差异 |
| `M_knowledge` | 该搜什么、何时不搜、哪些事实必须进入 plan |
| `M_reference` | 参考图角色、去重、ordinal binding、是否有害 |
| `M_reasoning` | temporal/causal/spatial/logical/discipline 推理差异 |
| `M_region` | edit/preserve 区域、mask/bbox/region phrase 差异 |
| `M_editor` | 当前固定 editor 更容易执行的 prompt 结构 |
| `M_failure` | over-edit、under-edit、hallucination、文字/数量/符号失败规避 |

Distillation mask：

- 只对 assistant tokens。
- 只对有 retrieved experience 的 rollout。
- 优先选择 teacher-student logprob 分歧最大的 top 10-20% decision tokens。
- 不蒸馏 tool observation 和 rendered image。

Objective：

```text
L_RISEvolve = L_GRPO
            + lambda_opd * L_EditOPD
            + beta_kl * KL(policy || ref_policy)
            + gamma_sft * L_SFT_replay
```

建议初值：

```yaml
lambda_opd: [0.5, 1.0, 2.0]
beta_kl: 1e-3
gamma_sft: 0.05-0.20
experience_min_gap: 0.20
experience_buffer: 500-2000 bundles
retrieval_embedder: Qwen3-Embedding-0.6B or bge-m3
```

二维图像编辑相对通用 OPD 的改进点：

1. Region-aware OPD：teacher experience 明确告诉 student 哪些区域该改、哪些区域不能动。
2. Checklist-conditioned OPD：teacher 看到 verifier 失败项，把错误 checklist 转成 action-level guidance。
3. Editor-gap-aware OPD：区分 plan 错误和 editor 执行错误；若 plan 正确但 editor 失败，不把所有责任压到 reasoning tokens。
4. Tool-boundary OPD：蒸馏“该不该 search / image_search / solve_symbolic”的边界，而不是鼓励所有 hard case 都检索。
5. Multi-editor OPD：同一 edit_program 用多个 editor 渲染，teacher 总结 editor-robust prompt 写法。

### Stage 5: Targeted Self-Evolution

把 dev/eval 中的失败 case 聚类，生成新的 hard prompts 和 teacher trajectories：

- RISE temporal：状态变化不自然、过度腐烂/老化、背景漂移。
- RISE causal：物理结果错、受力/热/光学方向错。
- RISE spatial：对象相对位置、遮挡、视角、透视失败。
- RISE logical：符号/棋盘/迷宫/数量失败。
- GRADE：学科事实错、图示不可读、公式/结构错误。
- KRIS：factual/conceptual/procedural 混淆。

每轮只扩 1K-3K hard cases，先 human-audit 100-200 条，再并入 SFT/RL pool。

## 5. RL Algorithm Choice

首选：GRPO + Edit-OPD。

原因：

- Agent policy 是离散 token 策略，和 GenEvolve、AlphaGRPO、ReasonEdit 等近期多模态训练范式一致。
- 多 rollout 同 prompt 天然适配 group-relative advantage。
- 不需要 value model，工程复杂度低于 PPO。
- 可以直接把 final rendered image reward 传回 tool/reasoning/program tokens。
- 与 OPD/SDL 互补：GRPO 告诉模型哪条轨迹更好，OPD 告诉模型具体哪些 token-level 决策应向 privileged teacher 靠拢。

其他算法定位：

| 算法 | 是否适合作为主线 | 用法 |
|---|---|---|
| SFT | 必需 | 格式、工具协议、schema、基本 reasoning |
| DPO/IPO/ORPO | 不建议主线 | 可用于离线 preference pairs warm-up |
| PPO | 暂不优先 | 需要 value model，成本高；除非 GRPO 不稳 |
| GRPO | 主线 | planner/tool/program policy |
| Listwise DPO/LPO/LAIR | 可做 editor 后训练 | 多候选 edited images 的 preference post-training |
| DiffusionOPD / Flow-OPD | 第二阶段 | 统一多个 reward/domain 的 diffusion/flow editor，不是第一阶段 agent 训练 |
| Reward-weighted SFT | 可选 | 对高 reward trajectories 做稳定回放 |

若后续训练底层 editor：

- 对局部编辑优先用 mask/region-aware reward。
- 保留 source fidelity regularization：LPIPS/DINO/SSIM/identity loss。
- 使用 listwise / OPD 而非单一 scalar RL，避免 prompt adherence 与 preservation seesaw。
- 先训 LoRA 或 small adapter，保留 base editor 的通用能力。

## 6. Evaluation and Ablations

Main benchmarks：

- RISEBench：Temporal / Causal / Spatial / Logical。
- GRADE：10 discipline domains。
- KRIS-Bench：Factual / Conceptual / Procedural / anomaly / multi-element。

Baselines：

- direct editor；
- prompted planner + same editor；
- DDA-like planner without search/image_search/memory；
- RePlan-like region planner；
- RISEvolve-SFT；
- RISEvolve-GRPO；
- RISEvolve-GRPO+Edit-OPD。

Ablations：

- no `analyze_image`；
- no `search`；
- no `image_search`；
- no `query_edit_knowledge`；
- no `ground_region`；
- no `solve_symbolic`；
- no checklist；
- no preserve reward；
- no tool reward；
- no Edit-OPD；
- DPO warm-up vs GRPO；
- single editor vs multi-editor rollout。

Diagnostics：

- planner correct / editor fail ratio；
- editor correct despite weak plan ratio；
- tool call rate、usefulness、repeat rate；
- reward head breakdown；
- preserve metrics on non-edit region；
- checklist-human agreement；
- best-worst experience quality audit。

## 7. Immediate Implementation Milestones

1. Convert current v1 trajectories to LLaMA-Factory/VLM SFT format with assistant-only masks.
2. Implement prompted RISEvolve baseline and fixed editor adapter.
3. Implement checklist verifier and reward head logging.
4. Run 100-case dev failure taxonomy.
5. SFT on current v1 + small real-image supplement.
6. Run 50-100 GRPO debug steps with K=4 and no OPD.
7. Add Edit-OPD memory, teacher context patching, token mask, and SDL loss.
8. Scale to 300-800 GRPO+Edit-OPD steps.
9. Run RISE/GRADE/KRIS eval and ablations.

## 8. Expected Failure Modes and Controls

| Risk | Control |
|---|---|
| VLM judge reward hacking | checklist-first scoring, two judges, human audit, reward model cross-check |
| Tool overuse | `R_tool`, no-search allowed for simple cases, query dedup checks |
| Editor bottleneck | report oracle-plan upper bound and planner/editor failure attribution |
| Programmatic data bias | add real-image source pool and strong-editor renders |
| OPD teacher overfits stale memory | reward-gap + recency eviction, retrieval threshold, SFT replay |
| Logical/symbolic rendering weak | route through `solve_symbolic`, specialized text rendering prompts, separate hard split |
