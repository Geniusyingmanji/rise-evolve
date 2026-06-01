# RISEvolve Survey: Agentic Training and Reasoning-Centric Image Editing

调研时间：2026-05-31  
目标：为 `RISEvolve` 的研究定位、训练路线、奖励设计和实验方案提供文献依据。重点关注 agentic training、工具使用训练、图像编辑、推理型图像编辑、奖励模型和 benchmark。

## 1. 关键结论

1. 图像编辑的前沿正在从“单步 instruction-to-image”转向“planner/thinker + editor”的解耦范式。DDA-Thinker、MIRA、RePlan、EditThinker、ThinkRL-Edit 都在证明：固定强 editor，单独训练 reasoning/planning 模块也能显著提升推理型编辑。
2. GenEvolve 的核心价值不是图像生成本身，而是把生成过程建模为 tool-orchestrated trajectory，再用 best-vs-worst experience distillation 提供 token-level 决策监督。这个思想迁移到编辑是合理的，但必须加入源图理解、区域约束、非编辑区保真和可验证推理。
3. RISEBench、GRADE、KRIS-Bench、CompBench、Edit-Compass 共同显示一个趋势：强编辑模型已经能做常规局部编辑，但在 temporal/causal/spatial/logical、知识约束、多对象、多区域、多步任务上仍然不稳。
4. 奖励设计不能只用 VLM 给一个总分。DDA-Thinker、Edit-R1、ThinkRL-Edit、EditReward、RewardHarness 都指向更细粒度的 verifier/checklist/rubric：先拆解约束，再逐条验证。
5. 对 RISEvolve 来说，最有区分度的主张应是：一个可自进化的 reasoning-informed editing agent，显式学习何时分析源图、何时检索知识、何时找参考图、如何生成可执行 edit program、如何用经验修正未来的工具和计划决策。
6. 需要避免 novelty 被 DDA-Thinker/MIRA/RePlan 吃掉。Plan 需要强调 RISEvolve 不只是 thinker，而是 search/image-search/skill/memory/checklist/edit-program 的完整 agentic training pipeline。

## 2. Benchmark 和问题定义

### 2.1 RISEBench

**Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing**，arXiv:2504.02826。

- 任务：Reasoning-Informed viSual Editing。
- 四类 reasoning：Temporal、Causal、Spatial、Logical。
- 评价维度：Instruction Reasoning、Appearance Consistency、Visual Plausibility。
- 公开 arXiv v4 摘要中报告 GPT-4o-Image accuracy 仅 28.8%。如果使用更新 leaderboard 中的 35.9%，实验前需要固定 benchmark 版本和引用来源。
- 启示：RISEvolve 的训练数据 recipe 应直接覆盖四类 reasoning，但不能使用测试集本身训练。

### 2.2 GRADE

**GRADE: Benchmarking Discipline-Informed Reasoning in Image Editing**，arXiv:2603.12264。

- 规模：520 个样本，10 个学科域。
- 评价维度：Discipline Reasoning、Visual Consistency、Logical Readability。
- 目标：考察隐式学科知识约束下的编辑，而非显式“把 A 换成 B”。
- 启示：agent 需要外部知识检索和可引用的推理依据，单纯 prompt rewriting 不够。

### 2.3 KRIS-Bench

**KRIS-Bench: Benchmarking Next-Level Intelligent Image Editing Models**，arXiv:2505.16707。

- 规模：1,267 个实例。
- 知识类型：Factual、Conceptual、Procedural。
- 评价：包含 Knowledge Plausibility。
- 启示：可作为 RISE/GRADE 之外的泛化评测，尤其验证知识型编辑是否过拟合 GRADE。

### 2.4 其他相关 benchmark

| Benchmark | 论文/来源 | 关注点 | 对 RISEvolve 的价值 |
|---|---|---|---|
| EditBench | Imagen Editor and EditBench, arXiv:2212.06909 | inpainting、对象/属性/场景 | 常规编辑基础能力 |
| MagicBrush | arXiv:2306.10012 | 人工标注 instruction-guided editing | SFT 源图和基础编辑数据 |
| ImgEdit-Bench | arXiv:2505.20275 | 单轮、多轮、复杂编辑 | 泛化评估 |
| CompBench | arXiv:2505.12200 | location/appearance/dynamics/objects 复杂组合 | 多约束指令拆解 |
| Edit-Compass | arXiv:2605.13062 | world knowledge、visual reasoning、多图编辑、reward benchmark | reward/rubric 设计参考 |
| GSI-Bench | arXiv:2604.20570 | 3D spatial grounded editing | spatial 子类增强 |
| ByteMorph | arXiv:2506.03107 | 非刚性运动、形变、动态交互 | temporal/causal 数据扩展 |

## 3. Agentic Training 线索

### 3.1 基础范式

- **ReAct**，arXiv:2210.03629：交替生成 reasoning trace 和 action，奠定 tool-use agent 交互格式。
- **Toolformer**，arXiv:2302.04761：用少量 API demo 自监督学习何时调用工具、如何使用结果。
- **ART**，arXiv:2303.09014：自动选择多步 reasoning/tool-use demos，并在生成中暂停执行工具。
- **MM-REACT**，arXiv:2303.11381：ChatGPT + vision experts，多模态 reasoning/action 系统。
- **ViperGPT**，arXiv:2303.08128：生成 Python 程序调用视觉模块，说明“可执行中间程序”能提升视觉推理解释性。

对 RISEvolve 的直接影响：采用 ReAct 式多轮轨迹；工具结果应进入 context；最终输出不只是 prompt，而是可执行 edit program。

### 3.2 自改进和记忆

- **STaR**，arXiv:2203.14465：用正确答案反推 rationale，迭代增强 reasoning。
- **Self-Refine**，arXiv:2303.17651：自反馈-自修正，不更新权重也能提升输出。
- **Reflexion**，arXiv:2303.11366：把失败反馈转成 verbal memory，后续 trial 使用。
- **Voyager**，arXiv:2305.16291：自动 curriculum、skill library、环境反馈驱动长期学习。

对 RISEvolve 的直接影响：VED 的 experience memory 不应只是“更好 prompt 模板”，而应总结可迁移的决策差异：搜索什么、跳过什么、如何定位区域、如何避免 editor 失效。

### 3.3 2025-2026 Agentic RL 前沿

| 方向 | 代表工作 | 可借鉴点 |
|---|---|---|
| step-level credit | StepPO, arXiv:2604.18401 | agent action 更适合作为 step 而非 token；可为工具调用分配奖励 |
| 工具边界 | AKBE, arXiv:2605.26952 | 判断何时需要工具，抑制无效检索 |
| 环境合成 | EnvFactory, arXiv:2605.18703; COVERT, arXiv:2604.09813 | 生成可验证工具环境和扰动反馈 |
| 视觉工具 RL | VISTA-Gym/VISTA-R1, arXiv:2511.19773 | 多模态工具集、可验证反馈、轨迹日志 |
| 轨迹效率 | InfoTree, arXiv:2605.05262 | hard prompt 用树搜索提升 rollout informativeness |
| 数据高效 agent | PC Agent-E, arXiv:2505.13909; STEVE, arXiv:2503.12532 | 少量人类轨迹 + 合成替代动作 + step verification |

对 RISEvolve 的直接影响：在 GRPO 之外加入 `R_tool` 或 penalty，评估“是否必要、是否有效、是否重复”；对 hard prompts 才分配更多 rollout/工具预算。

## 4. 图像编辑基础路线

### 4.1 传统 diffusion editing

- **SDEdit**，arXiv:2108.01073：用输入图加噪再去噪实现编辑，奠定 image-guided diffusion editing。
- **Prompt-to-Prompt**，arXiv:2208.01626：cross-attention 控制文本驱动编辑，强调局部语义绑定。
- **Null-text Inversion**，arXiv:2211.09794：通过 inversion 增强真实图编辑保真。
- **ControlNet**，arXiv:2302.05543：将边缘、深度、pose 等控制信号注入 diffusion。

这些方法解决“怎么改”，但不解决“该改成什么、为什么这样改、哪里不能动”。

### 4.2 Instruction-guided editing 数据和模型

- **InstructPix2Pix**，arXiv:2211.09800：GPT-3 + Stable Diffusion 合成大规模编辑数据，单 forward 编辑。
- **MagicBrush**，arXiv:2306.10012：人工标注 real image editing，包含 single-turn/multi-turn、mask-free/mask-provided。
- **Emu Edit**，arXiv:2311.10089：多任务编辑，覆盖 region/free-form/CV generative tasks。
- **OmniGen**，arXiv:2409.11340：统一图像生成模型，支持编辑、subject-driven、条件生成。
- **ImgEdit**，arXiv:2505.20275：1.2M 高质量编辑 pairs 和 ImgEdit-Bench。

对 RISEvolve 的影响：这些数据适合作为源图和普通编辑能力底座，但推理型指令需要重新合成和过滤。

## 5. Reasoning-Centric Image Editing 前沿

### 5.1 Planner/Thinker + Editor 解耦

| 工作 | 核心机制 | 对 RISEvolve 的启示 |
|---|---|---|
| Unified Thinker, arXiv:2601.03127 | task-agnostic planning core，可接不同 generator | RISEvolve 应显式宣称 planner 可迁移到多 editor |
| ThinkRL-Edit, arXiv:2601.03467 | CoT planning/reflection + checklist reward + unbiased chain preference grouping | 多奖励不能简单加权；用 checklist 降低 VLM 打分方差 |
| EditThinker, arXiv:2512.05965 | Think-while-Edit：critique、refine instruction、repeat | 可作为 test-time refinement baseline |
| DDA-Thinker, arXiv:2604.25477 | 固定 Editor，只优化 Thinker；cognitive-atomic + visual-atomic rewards | RISEvolve 必须加入 plan-level reward，不能只看 edited image |
| MIRA, arXiv:2511.21087 | perception-reasoning-action loop；SFT + GRPO；150K tool-use 数据 | 可对比 iterative agent，但 RISEvolve 更强调搜索/知识/experience |
| RePlan, arXiv:2512.16864 | region-aligned planning + GRPO + attention-region injection | RISEvolve 需要 edit_region/mask/region checklist |
| From Plans to Pixels, arXiv:2605.15181 | planner + orchestrator + outcome rewards，成功轨迹反哺 planner | 和 RISEvolve 目标接近，需强调 VED 和知识工具差异 |

### 5.2 Reasoning-aware editing 数据

- **Reasoning to Edit / Reason50K / ReasonBrain**，arXiv:2507.01908：hypothetical instruction editing，Physical/Temporal/Causal/Story reasoning。
- **Uni-Edit**，arXiv:2605.21487：把 VQA 数据转成 embedded question/nested logic 的 intelligent editing 指令。
- **ScaleEdit-12M**，arXiv:2603.20644：开源 hierarchical multi-agent 数据构造，23 task families，显著提升 RISE/KRIS。
- **DataEvolver**，arXiv:2605.01789：goal-driven loop agents 自动生成、检查、纠正、过滤视觉数据。

对 RISEvolve 的影响：数据构造应从 benchmark recipe 扩展到 VQA/知识库/物理过程/视频过渡，不要只从 MagicBrush 改写。

### 5.3 编辑模型 post-training 和局部保真

| 工作 | 核心机制 | 可用策略 |
|---|---|---|
| EditReward, arXiv:2509.26346 | 200K preference pairs，人类偏好 reward | 用作外部 reward 或数据过滤器 |
| Edit-R1, arXiv:2604.27505 | verifier-based RRM：principle decomposition + CoT + GCPO + GRPO | 构建 RISEvolve 的 `R_program` 和 `R_edit` judge |
| RewardHarness, arXiv:2605.08703 | 少量偏好样例，演化 tools/skills 作为 reward context | 用于低成本 reward skill library |
| ReasonEdit, arXiv:2605.07477 | 可解释编辑评测 + GRPO | 训练解释型 judge 或做诊断报告 |
| Edit-GRPO, arXiv:2605.16951 | locality-preserving policy optimization | 将 edit/preserve 区域分开奖励 |
| RC-GRPO-Editing, arXiv:2604.09386 | 区域约束 GRPO，减少背景扰动 | `edit_region` 不是 optional，应进入 reward |
| CoCoEdit, arXiv:2602.14068 | region regularized RL，像素相似度 + MLLM reward | 非编辑区用 PSNR/SSIM/LPIPS/DINO 指标补充 |
| Rethinking Where to Edit, arXiv:2604.20258 | task-aware localization | 自动 mask/region planning 的参考 |

## 6. 与 GenEvolve 的关系

**GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation**，arXiv:2605.21605。

核心机制：

- 将生成尝试建模为 trajectory：`search/image_search/query_knowledge -> prompt-reference program -> generator -> reward/diagnostics`。
- 多个 rollout 同 prompt 对比 best/worst，将差异总结为 structured visual experience。
- teacher branch 看到 retrieved experience，student branch 不看；通过 SDL/Visual Experience Distillation 给 student token-level dense supervision。
- GRPO 提供 trajectory-level reward，SDL 提供 decision-token guidance。

迁移到图像编辑时必须改动：

- 输入从 text prompt 变成 `(source_image, edit_instruction)`，所以第一工具应是 `analyze_image` 或更强的 scene graph/region parser。
- 输出从 `gen_prompt + refs` 变成 `edit_program`，包括目标描述、区域、保真约束、执行 prompt、参考图、atomic checklist。
- Reward 从 image quality 扩展为 plan correctness、reasoning correctness、non-edit preservation、region locality、visual plausibility。
- Experience slot 必须包含 reasoning 和 preservation 失败，而不只是 prompt/reference 经验。

## 7. RISEvolve 的建议定位

### 7.1 建议 claim

> RISEvolve trains a self-evolving, tool-orchestrated image editing agent that learns to transform reasoning- and knowledge-intensive edit requests into executable, region-aware edit programs. Unlike planner-only approaches, it jointly learns source-image analysis, knowledge retrieval, reference selection, skill activation, atomic verification, and edit-program synthesis through visual-cognitive experience distillation.

中文版本：

> RISEvolve 不是直接训练一个编辑模型，而是训练一个可自进化的编辑规划 agent。它在源图分析、知识检索、参考图选择、技能路由、区域规划和 checklist 验证之间做决策，并把这些决策转成下游 editor 可执行的 edit program。

### 7.2 与强相关工作区分

| 对比对象 | 对方强点 | RISEvolve 区分点 |
|---|---|---|
| DDA-Thinker | dual-atomic reward，固定 editor 下优化 planner | RISEvolve 增加外部知识/图像检索、skill memory、best-worst experience distillation |
| MIRA | 迭代感知-推理-行动，SFT+GRPO | RISEvolve 更面向 RISE/GRADE 的知识和推理检索，不只是分步 atomic edits |
| RePlan | region-aligned planning | RISEvolve 把 region planning 纳入工具和 reward，但还覆盖 temporal/causal/discipline reasoning |
| Edit-R1 | 强 reward verifier | RISEvolve 可用 verifier，但优化对象是 agentic edit program policy |
| RewardHarness | agentic reward context evolution | RISEvolve 演化的是编辑 agent 的决策经验，而不仅是 judge skill library |
| GenEvolve | tool-orchestrated VED for generation | RISEvolve 将 VED 改造成 visual-cognitive editing experience，并加入源图、区域和保真约束 |

## 8. 对 plan 的具体改造建议

1. `edit_program` 增加字段：`source_scene_graph`、`target_scene_description`、`edit_operations`、`edit_region`、`preservation_constraints`、`knowledge_facts`、`reference_roles`、`atomic_checklist`、`negative_constraints`、`editor_prompt`。
2. 奖励从 `0.5 R_edit + 0.5 R_program` 改成多头/分组 reward：`R_cognitive`、`R_visual`、`R_preserve`、`R_tool` 分别算 advantage，再汇总更新，避免 weighted-sum 淹没冲突信号。
3. Teacher trajectory 里强制生成 rational target description，用于 checklist synthesis。这是 DDA-Thinker/Edit-R1 都验证过的低噪声做法。
4. `query_edit_knowledge` 的 skill 不应只有大类，应包含 output recipe：每个 skill 返回“需要观察什么、需要检索什么、如何写 region、如何写 checklist、常见失败”。
5. 增加工具边界控制：重复检索、无效检索、未使用证据都应惩罚；对于无需外部知识的 case，允许不 search。
6. 数据 curriculum：先 SFT 训练格式和基本工具调用，再只用中等难度样本做 RL，过滤过易/过难/teacher-editor gap 太大的样本。
7. 强 baseline 必须包括 DDA-like planner-only、MIRA/RePlan-like iterative/region planner、GPT/Gemini planner + same editor、direct editor。
8. 消融必须证明 agentic 部分有用：no search、no image_search、no analyze_image、no region、no checklist、no VED、weighted reward vs separate rewards。

## 9. 风险判断

1. **Novelty 风险高于原 plan 预期**：DDA-Thinker、MIRA、RePlan、EditThinker 已经覆盖“planner + editor”。应把 novelty 集中到 tool-orchestrated self-evolution、知识检索、experience distillation、editing-specific program/reward。
2. **Editor bottleneck 会限制 logical/math 类任务**：需要在评测中区分 thinker failure 和 editor execution failure，并报告 oracle-plan upper bound。
3. **VLM judge reward hacking**：必须用 checklist + 多 judge + human spot check；可用 EditReward/Edit-R1 类 reward model 交叉验证。
4. **工具调用可能过度**：引入 `R_tool` 和 knowledge-boundary audit，统计每类任务工具调用收益。
5. **成本可能偏高**：优先做 1K-2K 高质量 trajectory + 小规模 GRPO proof，再扩到 10K。

## 10. 推荐阅读优先级

最高优先级：

1. GenEvolve, arXiv:2605.21605
2. RISEBench, arXiv:2504.02826
3. GRADE, arXiv:2603.12264
4. DDA-Thinker, arXiv:2604.25477
5. Edit-R1, arXiv:2604.27505
6. MIRA, arXiv:2511.21087
7. RePlan, arXiv:2512.16864
8. RewardHarness, arXiv:2605.08703
9. Edit-Compass, arXiv:2605.13062
10. ScaleEdit-12M, arXiv:2603.20644

第二优先级：

1. ThinkRL-Edit, arXiv:2601.03467
2. EditThinker, arXiv:2512.05965
3. Reasoning to Edit / Reason50K, arXiv:2507.01908
4. Uni-Edit, arXiv:2605.21487
5. CompBench, arXiv:2505.12200
6. ImgEdit, arXiv:2505.20275
7. EditReward, arXiv:2509.26346
8. CoCoEdit, arXiv:2602.14068
9. GSI-Bench, arXiv:2604.20570
10. KRIS-Bench, arXiv:2505.16707
