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

## 11. 2026-06-01 深入调研：GenEvolve、OPD 和编辑 RL

### 11.1 GenEvolve 训练方法复盘

GenEvolve 的公开代码位于 `https://github.com/MeiGen-AI/GenEvolve`，包含 OpenAI-compatible agent runtime、`search` / `image_search` / `query_knowledge` 三类工具、8 个 generation skill、Qwen-Image-Edit/Nano Banana Pro wrapper 和 benchmark evaluation 脚本。Hugging Face 发布了 Qwen3-VL-8B-based policy `MeiGen-AI/GenEvolve`，以及 `MeiGen-AI/GenEvolve-Data-Bench` 数据：SFT 9,000 trajectories、RL 3,175 prompts + GT images、Bench 594 prompts。需要注意：repo 明确说明 full training scripts are not included，训练实现要根据论文方法复现。

论文中的训练流程可以抽象为：

1. **SFT cold start**：用 teacher 轨迹训练 ReAct 工具协议。只优化 assistant-side trajectory tokens，user prompt 和 tool observations masked。
2. **Prompt-reference program**：最终输出 `z=(g,R)`，其中 `g` 是 generator-facing prompt，`R` 是按 ordinal phrase 引用的 reference images，不输出 URL/ID。
3. **Dual reward + GRPO**：每个 prompt 采样 K 条轨迹，render image 后用 `R = 0.5 R_img + 0.5 R_text` 打分；`R_img` 评估生成图，`R_text` 评估 program sufficiency；GRPO 优化同组相对优势。
4. **Visual Experience Extraction**：同 prompt 的 best/worst 轨迹若 reward gap `Δ >= 0.20`，抽取 search strategy、knowledge activation、reference selection、prompt construction、failure avoidance 五类 experience。
5. **Visual Experience Distillation / SDL**：student 看到普通 context，teacher branch 看到 patched experience context；teacher 不另生成轨迹，只对同一批 on-policy tokens 重新打分，用 sampled-token reverse-KL 做 dense token-level distillation。推理时只部署 student。

对 RISEvolve 的最重要启示：如果把图像编辑也建模为 tool-orchestrated trajectory，训练主目标应该是 agent 的决策 token，而不是直接把所有 reward 压到 diffusion editor。

### 11.2 近期 OPD 线索

用户提到的 OPD 在近期工作中主要指 **On-Policy / Online Policy Distillation**。其共同点是：teacher 和 student 通常共享或相关参数，student 产生 on-policy rollouts，teacher 在 privileged context、specialized expert 或局部输入下对同一轨迹给 token/step-level dense supervision，从而减少纯 scalar RL 的方差和离线偏好训练的 distribution mismatch。

相关工作：

| 工作 | 方向 | 对 RISEvolve 的启示 |
|---|---|---|
| Vision-OPD, arXiv:2605.18740 | MLLM fine-detail perception；crop-conditioned teacher 蒸馏 full-image student | 可用于源图局部证据：teacher 看 crop/region，student 学会 full-image 下关注关键区域 |
| DiffusionOPD, arXiv:2605.15055 | diffusion model 多任务 online policy distillation | 若后续训练 editor，可用 task-specific teacher distill 到统一 editor |
| Flow-OPD, arXiv:2605.08063 | flow matching model 的 OPD，先单 reward teacher，再 on-policy distill unified student | 适合解决编辑中 prompt adherence、preservation、aesthetic 多 reward 的 seesaw |
| GenEvolve SDL, arXiv:2605.21605 | visual experience conditioned teacher 蒸馏普通 student | 最直接可迁移到编辑 agent，改成 visual-cognitive editing experience |

二维图像编辑方向目前更成熟的是 preference/RL post-training 和 reward model，OPD 还没有形成统一 benchmark 标准。因此 RISEvolve 可以把 OPD 具体化为 **Edit-OPD**：best/worst 编辑轨迹产生 visual-cognitive experience；teacher 看到 region/checklist/diagnostics 作为 privileged context；student 在普通 inference context 下学习这些 decision-token 偏好。

### 11.3 图像编辑 RL / Preference 最新进展

| 工作 | 结论 | 可借鉴点 |
|---|---|---|
| HP-Edit, arXiv:2604.19406 | human-preference post-training for image editing；构造 RealPref-50K 和 HP-Scorer，用 reward post-train editor | 可作为 editor-side RLHF 参考；也可借鉴 preference scorer |
| EditHF-1M, arXiv:2603.14916 | 1M editing feedback，29M preference pairs，维度为 visual quality / instruction alignment / attribute preservation | RISEvolve verifier/reward 应显式拆这三维 |
| ReasonEdit, arXiv:2605.07477 | 22K edited images + 113K CoT + 1.3M human judgments；用 GRPO 训练可解释编辑 evaluator | 可用于训练解释型 judge，不一定直接训练 planner |
| Talk2Move, arXiv:2601.02356 | 对 object-level geometric transformation 使用 GRPO 和 object-centric spatial rewards | RISE spatial / region reward 的直接参考 |
| RL-RIG, arXiv:2602.19974 | Generate-Reflect-Edit，Reflection-GRPO 同时训练 VLM actor 和 image editor | 支持 planner + editor 分阶段或联合优化 |
| AlphaGRPO, arXiv:2605.12495 | UMM 上用 decompositional verifiable reward 做 GRPO，自反式生成/编辑 | `DVReward` 思路适合把复杂编辑指令拆成 atomic verifiable questions |
| UniRef-Image-Edit, arXiv:2602.14186 | 多参考图编辑，SFT + RL；解决多参考一致性 | RISEvolve 的 reference roles 和 ordinal binding 可借鉴 |
| ParetoSlider, arXiv:2604.20816 | 多目标 RL，推理时调 prompt adherence vs source fidelity 等 trade-off | 图像编辑 reward 不宜早期 scalarization，需保留多头指标 |
| Diffusion LAIR, arXiv:2605.26491 | listwise reward-aware diffusion alignment，使用同 prompt 多候选分数 | 对底层 editor 后训练比 pairwise DPO 更适合 |
| Diffusion-LPO, arXiv:2510.01540 | listwise preference optimization，覆盖 image editing | 可把 RISEvolve 的多候选 edited images 转成 listwise editor training data |

### 11.4 推荐训练选择

短期最适合 RISEvolve 的 RL 算法是 **GRPO + Edit-OPD**：

- GRPO 负责用 rendered image / program reward 选择更好的完整轨迹。
- Edit-OPD 负责把 best-vs-worst 的差异变成 dense token-level decision guidance。
- DPO/IPO/ORPO 适合作为离线 preference warm-up，但不能替代 on-policy tool exploration。
- PPO 成本更高，需要 value model；除非 GRPO 不稳定，否则不是首选。
- DiffusionOPD/Flow-OPD 更适合第二阶段训练底层 editor，而不是第一阶段训练 agent。

二维图像编辑的 OPD 改造应包含五个特化点：

1. **Region-aware OPD**：teacher 看到 crop/mask/region diagnostics，student 学会全图输入下的区域定位。
2. **Checklist-conditioned OPD**：teacher 看到失败 checklist，将失败项转成 action-level guidance。
3. **Editor-gap-aware OPD**：区分 plan 正确但 editor 失败的情况，避免错误惩罚 reasoning tokens。
4. **Tool-boundary OPD**：蒸馏何时需要 search/image_search/solve_symbolic，抑制工具过度调用。
5. **Multi-editor OPD**：同一 program 用多个 editor 渲染，teacher 总结 editor-robust prompt/reference/region 写法。

详细训练流程见 `training_plan.md`。

## 12. 2026-06-01 Reward 深入调研：FIRM 与编辑 reward 设计

### 12.1 Trust Your Critic / FIRM

**Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation**，arXiv:2603.12247。代码仓库为 `VisionXLab/FIRM-Reward`，开源了 generation/editing RL 代码、reward server、FIRM-Bench 评测脚本，并在 Hugging Face collection `VisionXLab/firm-reward` 发布 `FIRM-Edit-8B`、`FIRM-Gen-8B`、`FIRM-Bench` 和相关数据。

FIRM 的创新点：

1. **Critic-first 视角**：RL 效果受 reward model 上限约束。通用 MLLM 做 critic 会 hallucinate、漏掉细微变化、给噪声分数。
2. **FIRM-Edit difference-first pipeline**：先让 MLLM 描述 source/edited 的 obvious + detailed differences，再让 evaluator 在差异描述、图像对和 instruction 条件下打分。这样比直接打分更接近人类判断。
3. **Execution / Consistency 双头编辑 reward**：execution 评估目标编辑是否完成；consistency 评估未编辑区域是否保持。
4. **FIRM-Gen plan-then-score pipeline**：先将 T2I prompt 拆成 checklist，再按 checklist 评价生成图，降低复杂 prompt 下的 hallucination。
5. **Reward benchmark 和 task-specific reward model**：构造 FIRM-Edit-370K、FIRM-Gen-293K、FIRM-Bench，并训练 FIRM-Edit-8B / FIRM-Gen-8B。
6. **Base-and-Bonus reward fusion**：证明简单 weighted sum 会 reward hacking。编辑中 `0.5*Execution + 0.5*Consistency` 会诱导模型输出几乎不变的图以获取高 consistency。因此提出 `CME = Execution * (0.6 + 0.4 * Consistency)`，把 execution 作为高 reward 的必要条件；生成中对应 `QMA = InstructionFollowing * (0.4 + 0.6 * Quality)`。

对 RISEvolve 的启示：我们的 reward 不能只是多头加权，必须设计 gating。对 reasoning editing 来说，`Execution` 之外还要加入 `Cognitive Correctness` 作为 gate，否则模型可能做出视觉上合理但推理/学科上错误的编辑。

### 12.2 其他 reward/RL 相关工作

| 工作 | reward 设计启示 |
|---|---|
| EditScore, arXiv:2509.23909 | 高保真专用 reward model 是 online RL 能工作的关键；大通用 VLM 不一定给有效学习信号 |
| EditReward, arXiv:2509.26346 | 200K 人类偏好 pairs 可训练 human-aligned reward，也可用于筛选高质量编辑数据 |
| Edit-R1, arXiv:2604.27505 | 从简单 scorer 转向 reasoning verifier；把 instruction 拆成 principles，再逐项验证 |
| RewardHarness, arXiv:2605.08703 | reward modeling 可以做 context/tool/skill evolution，而不一定总是更新 critic 权重 |
| Edit-Compass / EditReward-Compass, arXiv:2605.13062 | reward benchmark 需要模拟 RL 中真实 preference pairs，并覆盖 world knowledge、visual reasoning、多图编辑 |
| ReasonEdit, arXiv:2605.07477 | reward 的解释质量也要训练和评测；CoT/verifier 本身可以通过 GRPO 优化 |
| HP-Edit, arXiv:2604.19406 | 少量人类偏好 + VLM scorer 可形成实用 post-training reward |
| EditHF-1M, arXiv:2603.14916 | 编辑 reward 应拆成 visual quality、instruction alignment、attribute preservation |
| RC-GRPO-Editing, arXiv:2604.09386 | 区域约束能减少背景扰动导致的 noisy GRPO advantage |
| CoCoEdit, arXiv:2602.14068 | MLLM reward 之外应加入非编辑区域 pixel/feature similarity reward |
| Edit-GRPO, arXiv:2605.16951 | edit region 与 preserve region 应使用 region-specific optimization signals |

### 12.3 RISEvolve Reward 建议

建议把 reward 模块命名为 **RISE-Critic**，其核心是：

```text
expected diff planning
  -> difference-first observation
  -> checklist verification
  -> gated/base-and-bonus reward
  -> failure attribution
  -> head-aware token credit
```

推荐 heads：

```text
R_cog      推理/事实/学科/符号正确性
R_exec     请求的目标变化是否执行
R_preserve 非编辑区、身份、视角、背景是否保持
R_region   编辑区域和局部性
R_quality  视觉自然度、artifact、文字/图示可读性
R_tool     工具调用必要性、有效性、证据使用
R_format   schema/reference/checklist 合规
R_program  不看 render 的 plan correctness
```

推荐 gated reward：

```text
G_task = min(R_exec, R_cog_applicable)

R_image = G_task * (
  0.45 + 0.20 R_preserve + 0.15 R_region + 0.10 R_quality + 0.10 R_readability
)

R_agent = 0.45 R_program + 0.45 R_image + 0.05 R_tool + 0.05 R_format
```

如果任务不需要推理，`R_cog_applicable=1`。如果检测到 `editor_fail`，即 program 合理但 editor 未执行，应主要优化 `editor_prompt/region/negative_constraints`，而不是惩罚 reasoning/tool tokens。详细方案见 `reward_design.md`。
