# RISEvolve Reward Design: Difference-First, Gated, Editing-Aware Critic

更新日期：2026-06-01

目标：为 RISEvolve 的 SFT 过滤、RL reward、Edit-OPD experience、failure attribution 提供一套图像编辑特化 reward 方案。核心原则是：图像编辑 reward 不能是单一 VLM 总分，也不能是简单 weighted sum；必须显式处理“该改的是否改对”和“不该改的是否保持”之间的 shortcut。

## 1. Trust Your Critic / FIRM 的关键创新

论文：`Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation`, arXiv:2603.12247。
代码：`https://github.com/VisionXLab/FIRM-Reward`。
模型/数据：HF collection `VisionXLab/firm-reward`，包括 `FIRM-Edit-8B`、`FIRM-Gen-8B`、`FIRM-Bench`、`FIRM-Edit-Consistency`、`FIRM-Gen` 等。

FIRM 的核心判断是：RL 的瓶颈不是 policy optimizer，而是 critic。通用 MLLM 作为 reward model 会 hallucinate、忽略细粒度差异、给出噪声分数，从而误导 online RL。

### 1.1 FIRM-Edit: Difference-First Scoring

FIRM 发现一个现象：MLLM 做“评价者”时容易漏掉细节，但做“差异描述者”时更可靠。因此编辑 reward 不直接让 MLLM 看两张图打分，而是拆成两步：

```text
source image + edited image
  -> MLLM describes obvious and detailed visual differences
  -> evaluator receives source/edited/instruction/difference report
  -> score execution and consistency separately
```

两类 score：

- `Execution`：用户要求的编辑是否被正确执行。
- `Consistency`：未要求改变的对象、区域、布局、身份、背景是否被保持。

这直接对应图像编辑的本质：successful edit = target change + non-target preservation。

### 1.2 FIRM-Gen: Plan-Then-Score

对于文生图，FIRM 不是直接打总分，而是先让 LLM 把 prompt 分解成 checklist，再让 MLLM 按 checklist 检查图像。这和我们对 RISE/GRADE/KRIS 的做法一致：复杂指令必须先拆成可验证项。

```text
generation prompt
  -> LLM produces analysis/checking plan
  -> MLLM checks generated image against each criterion
  -> final score
```

### 1.3 FIRM-Bench and Reward Model Training

FIRM 构造了：

- `FIRM-Edit-370K`：编辑 reward 数据。
- `FIRM-Gen-293K`：生成 reward 数据。
- `FIRM-Bench`：807 个人工标注样本，其中 edit 包括 execution 和 consistency，gen 包括 instruction following。
- `FIRM-Edit-8B` / `FIRM-Gen-8B`：从 Qwen3-VL-8B-Instruct 初始化的 task-specific reward models。

关键点：reward benchmark 的分数分布被刻意平衡，避免 reward model 只学会高分样本。

### 1.4 Base-and-Bonus Reward: CME / QMA

FIRM 最重要的 RL reward 结论是：简单加权会导致 shortcut。

编辑场景中，如果用：

```text
R = 0.5 * Consistency + 0.5 * Execution
```

模型会学会“几乎不编辑”，因为这样 consistency 很高，即使 execution 低，总分也可能不差。FIRM 称这个行为为 lazy / unchanged shortcut。

FIRM 的编辑 reward 是 Consistency-Modulated Execution (CME)：

```text
R_CME = Execution * (0.6 + 0.4 * Consistency)
```

含义：

- Execution 是 base gate：没有完成编辑，就拿不到高分。
- Consistency 是 bonus：只有在完成编辑后，保真才进一步加分。

生成场景对应 Quality-Modulated Alignment (QMA)：

```text
R_QMA = InstructionFollowing * (0.4 + 0.6 * Quality)
```

对 RISEvolve 的启示：reward fusion 要有“必要条件”和“加分项”，不能把互相冲突的指标简单相加。

## 2. 对 RISEvolve 的直接影响

FIRM 主要训练底层 image editor / generator。RISEvolve 优先训练 agent planner，因此 reward 要同时服务两件事：

1. 评价最终 edited image。
2. 给 agent 的工具选择、推理、区域规划、editor prompt token 做 credit assignment。

我们不能照搬二头 `Execution / Consistency`，因为 RISE/GRADE/KRIS 还要求：

- 推理/学科知识正确；
- 源图状态理解正确；
- edit region 准确；
- 非编辑区保真；
- 符号、文字、公式、图示可读；
- 工具调用有必要且被最终 program 使用；
- 区分 planner failure 和 editor execution failure。

因此建议将 FIRM 的 `difference-first + base-and-bonus` 扩展成 `RISE-Critic`。

## 3. RISE-Critic Pipeline

### 3.1 输入

```json
{
  "source_image": "...",
  "edited_image": "...",
  "edit_instruction": "...",
  "edit_program": {
    "target_scene_description": "...",
    "edit_operations": [],
    "preservation_constraints": [],
    "negative_constraints": [],
    "atomic_checklist": {},
    "knowledge_facts": [],
    "editor_prompt": "..."
  },
  "tool_trace": [],
  "optional_region_mask": "...",
  "optional_reference_images": []
}
```

### 3.2 Stage A: Expected-Diff Planning

从 `edit_program` 生成 expected diff：

```text
allowed_changes:
  - target object/region
  - expected new attributes/state/content
  - physically implied consequences

protected_content:
  - objects, background, identity, viewpoint, lighting, layout
  - text/formula/chart elements not targeted

cognitive_targets:
  - required factual / symbolic / discipline result
  - source of knowledge or solver output
```

### 3.3 Stage B: Difference-First Observation

模仿 FIRM-Edit，但更结构化：

```text
source image + edited image
  -> obvious differences
  -> detailed differences
  -> region-localized differences if mask/bbox exists
  -> text/symbol/chart differences
  -> possible over-edit / under-edit / identity drift
```

关键改动：difference report 需要把变化标成四类：

| Type | 含义 |
|---|---|
| intended | 与 edit operation 对齐的变化 |
| missing | 应该出现但没出现的变化 |
| unintended | 不在 instruction/program 中的变化 |
| implied | 指令导致的合理物理/语义后果，不算过度编辑 |

### 3.4 Stage C: Checklist Verification

对每个 atomic checklist 输出二值或 1-5 子分，并保留 rationale：

```json
{
  "cognitive": [{"id": "C1", "pass": true, "score": 1.0, "evidence": "..."}],
  "execution": [{"id": "E1", "pass": false, "score": 0.25, "evidence": "..."}],
  "region": [{"id": "R1", "pass": true, "score": 1.0, "evidence": "..."}],
  "preservation": [{"id": "P1", "pass": false, "score": 0.4, "evidence": "..."}],
  "quality": [{"id": "Q1", "pass": true, "score": 0.8, "evidence": "..."}],
  "readability": [{"id": "T1", "pass": true, "score": 1.0, "evidence": "..."}]
}
```

### 3.5 Stage D: Failure Attribution

输出 failure type：

| Failure | 判定 |
|---|---|
| `planner_fail` | edit_program 的目标、知识、区域或 checklist 本身错 |
| `editor_fail` | edit_program 合理，但 edited image 没执行出来 |
| `over_edit` | 目标变化出现，但未保护区域被改 |
| `under_edit` | 保真好，但目标变化缺失 |
| `region_fail` | 改错对象、错区域、边界污染 |
| `knowledge_fail` | 学科/事实/逻辑结果错 |
| `judge_uncertain` | 两个 critic 或规则冲突，需要人工/跳过 |

这个 attribution 是 RISEvolve 区别于普通 reward model 的关键：它决定哪些 agent tokens 应该被惩罚或蒸馏。

## 4. Reward Heads

推荐 heads：

```text
R_cog      reasoning / factual / symbolic correctness
R_exec     requested edit execution
R_preserve non-target preservation
R_region   edit locality and target-region correctness
R_quality  visual plausibility, readability, artifact control
R_tool     necessary, non-redundant, used evidence/tools
R_format   parseable schema, reference binding, checklist validity
R_program  plan correctness before editor rendering
```

其中：

- `R_program` 只看 agent program，不看 edited image，降低 editor 噪声。
- `R_exec/R_preserve/R_region/R_quality` 看 rendered edited image。
- `R_cog` 同时看 program 和 image；如果图像结果与正确 reasoning 不一致，则低分。
- `R_tool/R_format` 用程序规则和 tool trace 计算，尽量不依赖 VLM。

## 5. Gated Reward Formula

FIRM 的 CME 适合普通编辑，但 RISE/GRADE/KRIS 需要 cognitive gate。建议初版：

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
```

说明：

- 若任务不需要推理/知识，`R_cog_applicable = 1`。
- `R_exec` 是必须项，防止 unchanged shortcut。
- `R_cog` 是 reasoning edit 的必须项，防止视觉上像但逻辑/学科结果错。
- `R_preserve/R_region/R_quality` 是 bonus，但只有任务执行和推理正确时才充分加分。

Agent 训练时再合成：

```text
R_agent =
  0.45 * R_program
  + 0.45 * R_image
  + 0.05 * R_tool
  + 0.05 * R_format
```

如果检测到 `editor_fail`，建议：

```text
R_agent = max(R_agent, 0.6 * R_program)
```

并在 token credit 中降低对 reasoning/tool tokens 的负向更新，主要更新 `editor_prompt`、`negative_constraints`、`region phrase`。

## 6. Token-Level Credit Assignment

普通 GRPO 用同一个 scalar reward 更新所有 assistant tokens。RISEvolve 应记录 token groups：

| Token group | 主要 reward |
|---|---|
| `tool_call` | `R_tool`, `R_cog`, `R_program` |
| `reasoning` | `R_cog`, `R_program` |
| `knowledge_facts` | `R_cog`, evidence-use checks |
| `region` | `R_region`, `R_preserve` |
| `preservation_constraints` | `R_preserve`, `over_edit` penalty |
| `negative_constraints` | `R_preserve`, `R_quality` |
| `editor_prompt` | `R_exec`, `R_quality`, `R_region` |
| `checklist` | verifier consistency and human agreement |
| `format` | `R_format` |

第一版可以只记录 groups，不做复杂 loss。第二版做 head-aware GRPO：

```text
Adv_token =
  sum_h mask(token, head_h) * normalize_group(R_h)
```

## 7. Reward Data Construction

当前 v1 已有 20K verifier items，可以按 FIRM 的经验继续增强：

1. 分数/失败类型要平衡，不能全是高分正例。
2. hard negatives 要覆盖：
   - unchanged shortcut；
   - over-edit；
   - wrong object/region；
   - correct visual but wrong reasoning；
   - correct reasoning but unreadable text/formula；
   - reference identity drift；
   - hallucinated background/object；
   - missing physically implied consequence。
3. 每条 reward item 保存：
   - expected diff；
   - observed diff；
   - checklist judgments；
   - score heads；
   - failure attribution；
   - whether suitable for RL / OPD / verifier SFT。
4. 人工校准 300-500 条，报告 MAE、pairwise accuracy、per-failure recall。

可外接/对比：

- FIRM-Edit-8B：作为 off-the-shelf critic baseline 或 ensemble member。
- EditScore / EditReward / EditHF / HP-Edit：作为 preference/reward baseline。
- Edit-R1 / ReasonEdit：作为 reasoning verifier baseline。
- CoCoEdit / RC-GRPO / Edit-GRPO：作为 region/preservation reward 参考。

## 8. Integration with Edit-OPD

RISE-Critic 的诊断直接进入 Edit-OPD teacher context：

```text
teacher-only context:
  expected diff
  observed diff
  failed checklist
  failure attribution
  reward head breakdown
  best-vs-worst contrast
```

Example:

```text
Failure: under_edit
Failed checklist: target fruit should show mold spots and wrinkling.
Observed diff: fruit color slightly darker, but shape and surface remain fresh.
Guidance: make temporal state explicit in target_scene_description and editor_prompt; preserve plate/counter/background.
Token focus: reasoning, target_scene_description, editor_prompt.
```

This makes Edit-OPD more than generic self-distillation: it distills editing-specific diagnostic knowledge that is unavailable at inference time.

## 9. Recommended Ablations

Reward ablations:

1. VLM direct 1-5 score.
2. FIRM-style `0.5 * execution + 0.5 * consistency`.
3. FIRM CME: `execution * (0.6 + 0.4 * consistency)`.
4. RISE-Critic without cognitive gate.
5. RISE-Critic without difference-first observation.
6. RISE-Critic without editor-gap attribution.
7. RISE-Critic full.

Training ablations:

1. GRPO with scalar reward.
2. GRPO with head logging only.
3. Head-aware GRPO.
4. Head-aware GRPO + Edit-OPD.

Metrics:

- official RISE / GRADE / KRIS scores；
- checklist pass rate；
- non-edit preservation metrics；
- unchanged shortcut rate；
- over-edit rate；
- planner correct / editor fail ratio；
- reward-human MAE；
- pairwise preference accuracy。
