# RISEvolve 改进计划(2026-06-11)

基于:(a) 仓库内部状态(`reports/CURRENT_STATUS_2026-06-07.md`、`survey.md`、`plan.md`、`training_plan.md`、`reward_design.md`);(b) 2026-06-11 三路并行文献调研(benchmark/SOTA、RL+reward+OPD、数据构造)。引用标注:✅ = 调研中直接核实过 arXiv 摘要/repo;◐ = 仅经搜索摘要核实,引用前需重查原文。

---

## 0. 执行摘要

1. **v2r SFT 失败(RISE n20: 0.15 vs prompted base 0.50)的根因诊断被文献完全印证**:boilerplate 轨迹 SFT 教的是"接口"不是"语义",且会压制 base 模型已有的 agent 能力。修复方向不是"更多真实编辑对",而是**少量、推理密集、可验证、任务族平衡的数据 + 从自己模型拒绝采样**。
2. **竞争格局已变**:DDA-Thinker(2604.25477)用同款架构(Qwen3-VL planner + frozen Qwen-Image-Edit-2511)做到 RISE 40.0%(32B)/ 31.9%(8B)、KRIS 79.94/76.99。**我们 8B 规模的硬性目标线是 RISE > 31.9、KRIS > 76.99**。它只用 5k 全合成推理密集 SFT + 1.4k 难度过滤 RFT——配方公开但无代码/数据发布。
3. **GRADE 是最大的空白奖品**:开源最佳仅 2.7% accuracy(Qwen-Edit-2511),**还没有任何 planner 方法发表过 GRADE 数字**,而知识检索 planner 正是 GRADE failure analysis 指向的方案。RISE-Logical 也是全场最弱(开源 ≤15.3%,闭源 ≤37.6%),`solve_symbolic` 是差异化机会。
4. **n20 评测必须废弃**。RISEBench 评测脚本 2026-04-23 修复过 bug 并全量重跑,所有对比必须用 post-fix 脚本、全量 360 或分层 n≥100。
5. 训练路线调整:**SFT 做小不做大 → RFT(拒绝采样微调)先于 GRPO → GRPO 带 2026 稳定性组件 → Edit-OPD 用门控版**。推理端加 adaptive Best-of-N(critic 已有,几乎白拿的分)。

---

## 1. 2026-06-11 竞争格局

### 1.1 榜单现状(post 2026-04-23 RISEBench 修复)

RISEBench(accuracy %,官方 repo):

| 系统 | Temporal | Causal | Spatial | Logical | Overall |
|---|---|---|---|---|---|
| GPT-Image-1.5(闭源第一) | 57.6 | 62.2 | 62.0 | 21.2 | **51.4** |
| Gemini-3-pro-image-preview | 43.5 | 63.3 | 48.0 | 37.6 | 48.3 |
| **DDA-Thinker-32B + QIE-2511** | 45.9 | 50.0 | 47.0 | 15.3 | **40.0** |
| **DDA-Thinker-8B + QIE-2511(我们的对标线)** | — | — | — | — | **31.9** |
| ThinkRL-Edit(RL on editor) | — | — | — | — | 29.7 |
| Unified Thinker(8B thinker) | — | — | — | — | 28.9 |
| Qwen-Image-Edit-2511(最强开源裸 editor) | 21.2 | 18.9 | 31.0 | 4.7 | 19.4 |

GRADE:Nano Banana Pro 46.2%、GPT-Image-2 ~56%(◐ README 行未核实)、**最强开源 Qwen-Edit-2511 仅 2.7%**;无 planner 方法发表过数字。KRIS:Gemini-3-Pro 85.31 > GPT-4o 80.09 ≈ **DDA-Thinker-32B 79.94** > DDA-8B 76.99 > QIE-2511 base 67.14。

结论排序:**decoupled planner+RL ≫ RL-on-editor(Edit-R1 在 GEdit SOTA 但 KRIS 仅 55.98,知识推理不迁移)≫ unified+CoT ≫ training-free agent**。我们的架构选择是对的。

### 1.2 直接竞品(必须进 related work 并差异化)

| 竞品 | 与我们的重叠 | 我们的差异点 |
|---|---|---|
| DDA-Thinker(2604.25477)✅ | 同架构、GRPO、dual-atomic checklist reward;开源 SOTA | 无知识/图像检索、无 symbolic solver、无 experience distillation、无 GRADE 数字、无发布 |
| From Plans to Pixels(2605.15181)✅ | **同 Qwen3-VL-8B planner** + 成功轨迹回流训练(LoRA r=1) | 面向广告/开放编辑,无 RISE/GRADE/KRIS 数字,无知识检索 |
| Unified Thinker(2601.03127)✅ | 8B thinker + frozen editor,SFT+双阶段 GRPO,RISE 28.9 | 其核心批评"外部 planner 不 grounded 于 editor 能力"正打向我们——必须用 editor-gap-aware reward + M_editor experience 正面回应 |
| RE-Edit/EditRefine(2606.05172)✅ | Qwen2.5-VL-7B agent + QIE,SFT→GRPO | 规模小、增益小(+4.5);其 1,000 样本 RE-Edit 可作额外评测 |
| IEA(2606.08016)✅ / ImageEdit-R1(2603.08059)✅ | 工具编排 + RL | 修图向/多 agent 向,非知识推理;IEA 的"边际工具有用性"奖励值得借鉴 |

### 1.3 仍未被占领的差异化点

1. **外部知识检索**(search/image_search):planner 阵营无人做,GRADE(2.7% open)和 KRIS knowledge-plausibility 直接奖励它。
2. **GRADE first-mover**:第一个发表 GRADE planner 数字。
3. **RISE-Logical**:全场最弱,`solve_symbolic` + 程序化渲染指令是明确路径。
4. **Region-aware edit program**:RePlan/GVCoT 做了 grounding 但都没上 RISE/KRIS/GRADE。
5. **Visual-cognitive experience distillation**(Edit-OPD):GenEvolve 思想在编辑域无人落地。

### 1.4 评测卫生(立刻执行)

- RISEBench 一律用 post-2026-04-23 脚本;核实 `gpt_eval.py` judge 是 GPT-4o 还是 GPT-4.1(README 两种说法并存)。
- KRIS judge 固定 GPT-4o(2025-05 snapshot);GRADE judge 文档不明(疑似 Gemini Flash ◐),跑前读 eval 代码确认并冻结版本。
- 所有内部对比从 n20 升级到分层 n≥100 dev split + 最终全量;dev split 固定随机种子并入库。

---

## 2. v2r SFT 失败的文献印证(为什么修法是对的)

| 现象 | 文献 | 结论 |
|---|---|---|
| SFT 后低于 prompted base | "What Do Agents Learn from Trajectory-SFT"(2602.01611)✅ | trajectory-SFT 放大 interface shortcutting;接口扰动下崩溃(30.5→4.5)而 prompted 模型稳定。**诊断法:对 SFT 模型做工具改名/JSON 字段重排扰动测试** |
| 窄域 SFT 压制通用 agent 能力 | "Awakening the Sleeping Agent"(2604.08388)✅ | 能力是被压制不是被擦除;**混入 ~100 条通用 agentic 轨迹即可恢复**(BFCL 0→83.8) |
| teacher 分布轨迹有害 | DAgger revisited(2605.12913)✅、SWiRL(2504.04736 ◐)、SD-Zero(2604.12002 ◐) | 纯 teacher 演示=covariate shift;**从 student 访问的状态采集、由 teacher/critic 重标**,或直接拒绝采样自己的模型 |
| 小而干净 > 大而糙 | DDA-Thinker(2604.25477)✅ | 5k 合成推理密集 + 1.4k difficulty-filtered RFT → RISE 40.0。我们 27k 常规编辑对标注方向本身错配 |
| RL 比 SFT 遗忘少 | 2507.05386 ◐ | 能力获取尽量后移到 RFT/RL 阶段,SFT 只做格式与协议 |

**v3 重标注快照(27,807 行)的处置**:不要直接全量训。先用 EditScore 式 group filter(低 max 分=任务不可达、低 std=无区分度的组剔除)+ 任务族配额重采样,预计只保留一个小的平衡子集(数千行)进入混合;它的主要价值是常规编辑格式底座,不是推理能力来源。

---

## 3. 数据计划 v4

### 3.1 原则(来自所有成功 pipeline 的共性)

(a) **answer-first / 程序可验证构造**:先有正确目标(程序生成/代码级编辑再渲染/视频帧对),再反推 rationale(STaR/REER 式),annotator 以 GT 编辑图为条件写 plan,再过滤"该 plan 是否能复现该编辑";
(b) **checklist/judge 拒绝采样**:二值 checklist(≤6 题)优于区间打分(DDA-Thinker、ThinkRL-Edit 双重验证);
(c) **任务族显式配额平衡**(EditThinker 按任务类型×分数档平衡)——杜绝 95% multi_element_composition 再现。

### 3.2 可直接引入的公开数据(license 已查)

| 数据 | 规模/License | 用途 | 注意 |
|---|---|---|---|
| **UniREdit-Data-100K**(HF maplebb/,2511.01295)✅ | 96k,Apache-2.0 | logical/spatial 弱项直补:数独/迷宫/棋类由 Python 程序生成,目标正确性有保证,CoT 非 boilerplate | 与 UniREditBench 同 pipeline;对 RISE/GRADE/KRIS 无声明重叠,自行图像级去重 |
| **ScaleEdit-12M**(2603.20644)✅ | 12.4M,MIT | 只按任务族子采样 knowledge-infused 切片(几万级),不全量摄入 | 6.73TB,选择性下载 |
| ThinkEdit-140k(2512.05965)✅ | 140k SFT + 27k RL | critique-refine 多轮轨迹带逐步分数,替代 instruction-echo 目标 | **license 未标注 + 含 Pico-Banana(CC BY-NC-ND)切片**,隔离处理;HF viewer schema 报错,下载后先验证 |
| Structured-visuals 1.3M(2510.05091)◐ / FigEdit(2512.00752)◐ | 1.3M / 30.8k | GRADE 学科图表的最佳代理:**代码级改 drawing program 再渲染**,目标精确 | release/license 未核实;即使不用数据,配方必抄 |
| GoT(2503.10639)◐ | ~8M | 带显式空间坐标的 reasoning chain 模板 | license 查 repo |
| Pico-Banana-400K(2510.19808)✅ | 257k SFT + 56k preference | preference pairs 用于 critic/judge 训练 | CC BY-NC-ND:不可商用、衍生物不可再分发,单独隔离 |

### 3.3 自建数据配方(对齐 benchmark 任务族,不碰 benchmark 内容)

| 任务族 | 配方 | 来源文献 |
|---|---|---|
| temporal/causal | **视频首尾帧对**:ChronoEdit 开源了从图像对生成 CoT 编辑 prompt 的 VLM 标注 pipeline,直接复用 | ChronoEdit(2510.04290)✅ |
| logical/symbolic | Python 程序生成谜题图对(UniREdit 配方)+ 我们已有的 v1 程序化生成器升级 | UniREdit ✅ |
| spatial/physical | Kubric/Blender 渲染源图(渲染源避免扩散幻觉)+ 场景图模板指令 | Kubric、BlendFusion ◐ |
| discipline(GRADE 向) | **改代码再渲染**:Vega-Lite/matplotlib/TikZ 图表、物理光路图、化学结构式,deterministic edit functions;算法 solver 映射模板(InternVL-U 配方) | FigEdit ◐、Structured-visuals ◐、InternVL-U ✅ |
| 验证回路 | 每张生成图用 VQA 约束反验证(期望答案不符则重生成) | SafetyPairs(2510.21120)◐ |

### 3.4 轨迹标注(替代 v2r 的 boilerplate 流程)

1. **Answer-first**:annotator 看到 (source, instruction, GT target/可验证规则) 才写 reasoning + edit program;再用"遮住 GT 只给 plan → editor 渲染 → critic 判定是否复现"过滤。
2. **DAgger 化**:一半轨迹从当前 student(prompted base 或最新 checkpoint)rollout 的状态出发,由强模型/critic 重标——直接对抗 covariate shift(2605.12913 ✅)。
3. **grounding-first**:先 ground 全部实体再生成指令(InterCoG ◐),杜绝 instruction echo。
4. **接口形式随机化**:工具名同义变体、JSON 字段顺序扰动进 SFT 数据(2602.01611 ✅)。
5. **通用能力回放**:每个 SFT 混合固定加入数百条通用工具调用/多模态指令轨迹(2604.08388 ✅)。

### 3.5 去污染(P0,训练任何真实数据前)

领域内普遍缺失(DDA-Thinker/EditThinker/UniREdit 均无声明),InternVL-U 是正例。我们执行双重过滤并写进 CI gate:

- pHash 精确/近重复 + **SSCD embedding cosine ≥ 0.95** 语义重复,对照 RISEBench-360、GRADE-520、KRIS-1267(+ 若用 UniREdit 数据则加 UniREditBench-2700);
- 工具直接用 FineVision 开源的 `huggingface/large-scale-image-deduplication`(2510.17269 ◐)。
- 现有 `check_decontamination.py` 只有文本指纹,补图像通道。

---

## 4. 训练计划 v4

### 4.1 SFT(做小、做准、设门)

- 规模:**3k-8k**,任务族配额硬约束;来源 = 3.2 公开数据子集 + 3.3 自建 + v3 快照过滤子集 + 数百条通用回放。
- 配置沿用 `training_plan.md` Stage 1(LoRA/language-only,1e-5,2 epochs),新增:接口随机化、boilerplate 检测器(已有)作准入门。
- **硬性 gate:SFT 模型必须在固定 dev(分层 n≥100,官方 judge)上 ≥ prompted base,且接口扰动测试不崩;不过门就不进 RL,回数据环节。**

### 4.2 先 RFT 再 GRPO(新增阶段,降低风险与成本)

GRPO 之前插入 1-2 轮 **RFT/拒绝采样微调**(DDA-Thinker 的 1.4k difficulty-filtered RFT、From Plans to Pixels 的成功轨迹回流都验证了这条便宜路线):

1. 用当前模型对训练 prompt 池采 K=4-8 个 edit program → 渲染 → critic 打分;
2. 难度过滤:全对(太易)与全错(太难)的 prompt 丢弃,保留中间带;
3. 高分轨迹(优先 critic+规则双通过)做 SFT 回流;失败轨迹经 self-revision(SD-Zero 式,以 critic 反馈为条件修订)后复检再入。

这一步同时产出 GRPO 需要的 difficulty 标签和 reward 分布画像。

### 4.3 Edit-GRPO(吸收 2026 稳定性组件)

在 `training_plan.md` Stage 3 基础上修改:

| 问题 | 采纳方案 | 文献 |
|---|---|---|
| 渲染+评审是昂贵环境步 | **rollout 缓存回放 2-4 次**(fixed editor ⇒ reward 可精确复用)+ 异步 rollout worker(每 worker 持有 editor+critic 实例)+ turn-level IS clipping | Experience Replay(2604.08706)✅、Polar(2605.24220)◐、SO-GRPO(2511.20718)◐ |
| 组内奖励无方差 | **前缀树 rollout**:同一 tool-use 前缀分支多个 edit program,只渲染叶子;按 pass-rate 把组维持在 ~50% 区间 | TRACE(2606.11119)✅、2605.05112 ◐ |
| 工具调用是脆弱行为 | 监控每工具调用率与 all-wrong 子组率;all-wrong 工具子组从共享 thinking 前缀重采样 | AXPO(2605.28774)✅ |
| step 级 credit | 以 (源图, 累积工具证据) 哈希做 anchor-state 分组,免费拿 step-level advantage | GiGPO(2505.10978)◐ |
| 工具奖励被 hack | 不奖励调用次数,奖励**证据使用**(program 引用工具输出)与**边际有用性** | Proof-of-Use(2510.10931)◐、IEA(2606.08016)✅ |

### 4.4 RISE-Critic 修订(在 `reward_design.md` 之上)

保留 difference-first + cognitive-gated CME 主体,新增:

1. **不从零训 critic**:第一版用 FIRM-Edit-8B + EditScore-7B(均开源)ensemble 作 R_exec/R_preserve 来源,自研部分只做 checklist 与 cognitive gate;Edit-R1 的 RL-RRM 7B(EditReward-Bench 78.2%)作交叉验证。
2. **Grid 并排重打分**:N 个候选独立打分后,拼一张 grid 让 VLM 并排重排——恢复组内方差、压 judge 噪声,每组只多 1 次调用(Stable-Layers 2605.30257 ✅)。
3. **Avg@K judge 自集成**(EditScore 实践)。
4. **校准方向:正例高精度**——奖错比漏奖危害大(2604.07666 ✅);更要紧的是**系统性偏差**:按编辑类别审计 critic 准确率,系统性错的类别降权/屏蔽(2603.16140 ◐)。
5. **不平衡组降权**(1-of-8 正确的组 advantage 可被单个 FP 抬高 60%,S-GRPO 2508.05928 ◐)+ 低置信**负向**判定的 advantage 收缩(不对称门控,EGPO 2602.22751 ◐)。
6. 人工校准 300 条,报告 MAE/pairwise acc/per-failure recall(原计划保留)。

### 4.5 Edit-OPD 修订(关键:2026 文献几乎重写了做法)

GenEvolve 原版 SDL 直接搬到多轮 agent 会不稳(SDAR 的核心发现)。修订:

1. **RL 主、蒸馏辅,门控为先**:对 detached teacher-student gap 做 sigmoid 门;**正负不对称**——teacher 背书的 token 加强蒸馏,teacher 否决的 token 软衰减(privileged experience 本身可能错)(SDAR 2605.15155 ✅)。
2. **符号一致性门**:蒸馏方向与 GRPO advantage 符号一致的 token 外推、矛盾的 token 内插(SG-OPD 2606.09304 ✅)。
3. **Privileged context 升级**:除 best-worst 文本经验外,把**渲染结果 + critic 诊断本身**作为 teacher-only context(Visual-SDPO 2606.10334 ✅ 验证了渲染反馈作 privileged context),这正好与 reward_design.md 第 8 节的诊断注入对接。
4. **稀有行为引导**:若某工具(如 image_search)在 on-policy 数据中几乎不出现,抽一部分 rollout 在采样时就注入 experience(guided rollouts),蒸馏 loss 只盖 privileged context 实际抬高了采样 token 概率的位置(EDGE-OPD 2605.23493 ✅)。
5. **健康监控**:(a) Top-K reverse-KL 必须用 stop-grad 变体(2605.11182 ✅);(b) 定期测 privileged context 是否真的移动了 teacher logits——不动则蒸馏项是纯噪声(2604.13016 ✅);(c) 蒸馏 KL 应集中在少数决策 token 上;(d) 只蒸馏跨任务复发的"规则型"经验,不蒸馏 per-image 细节。

### 4.6 推理端:adaptive Best-of-N(白拿的分,论文里如实报算力)

- 平坦 BoN N=8 约 +14%@8×成本(◐);ADE-CoT(2603.00141,CVPR 2026)用难度感知预算 + region/caption 早剪枝 + 机会性停止做到 >2× 加速同质量。我们的 critic region/preservation head 正是其早剪枝信号。
- 最终排序用 Avg@K 自集成 + EditReward 异源 ensemble 对冲系统性偏差。
- 报告两条曲线:single-sample 与 BoN(标注 N 与渲染次数),保证可比性与诚实性。

---

## 5. 阶段计划与验收门(每阶段不过门即回退)

| 阶段 | 内容 | 验收门 | 预估 |
|---|---|---|---|
| **P0 评测地基**(本周) | post-fix RISE 全量 harness + GRADE/KRIS eval 代码核读 + 固定 dev split(RISE 120 / GRADE 100 / KRIS 150 分层);跑 4 条基线:裸 QIE-2511、prompted 8B planner、prompted Gemini/GPT planner(oracle 上界)、(若有)旧 SFT ckpt | 基线表 + failure taxonomy(planner/region/knowledge/editor/judge fail 分布) | 1 周,纯 API+GPU 推理 |
| **P1 去污染+数据 v4** | 图像级去污染 CI;UniREdit/ScaleEdit 切片接入;ChronoEdit/FigEdit 配方自建 temporal+discipline 数据;answer-first 标注 + 任务族配额 | 3k-8k 平衡 SFT 集;100-200 条人工抽检通过;去污染 0 hit | 2-3 周 |
| **P2 SFT v4** | 4.1 配置训练 + 接口扰动测试 | **dev 上 ≥ prompted base**(不达标禁止进 RL);schema>98%,工具重复<10% | 1 周 |
| **P3 RFT 1-2 轮** | 4.2 拒绝采样回流 + difficulty 过滤 | dev 提升显著(配对 bootstrap p<0.05);难度分布画像产出 | 1-2 周 |
| **P4 Edit-GRPO** | 4.3 组件;先 50-100 debug steps(K=4,无 OPD) | reward 上升且 head breakdown 健康;unchanged-shortcut 率不升 | 2-3 周 |
| **P5 +Edit-OPD** | 4.5 门控蒸馏并入同一 loop | 对比 GRPO-only 有额外提升;teacher-logit 位移监控通过 | 与 P4 重叠 1-2 周 |
| **P6 全量评测+消融** | 三 benchmark 全量(单样本 + BoN 两条曲线);消融按 `training_plan.md` §6 + 新增:RFT-only vs +GRPO vs +OPD、grid-rescoring on/off、知识检索 on/off | **8B 目标:RISE > 31.9(DDA-8B)、KRIS > 76.99、GRADE 首个 planner 数字(目标两位数 accuracy)** | 1-2 周 |

**红线(不变)**:benchmark 原图/指令/GT/标注/近改写不入训练;eval-only 目录隔离;每次训练前文本+图像双通道去污染 gate;BoN/test-time 方法在论文中明确披露采样数与算力。

## 6. 风险更新

| 风险 | 变化 | 应对 |
|---|---|---|
| Novelty 被 DDA-Thinker / From Plans to Pixels 进一步压缩 | ↑ | 主张集中到:知识检索 + GRADE first-mover + solve_symbolic/Logical + 门控 Edit-OPD;引用并显式对比全部 §1.2 竞品 |
| "外部 planner 不 grounded 于 editor 能力"(Unified Thinker 的批评) | 新 | editor-gap-aware reward + M_editor experience + RFT 回流(成功轨迹天然 grounded);消融报告 planner-fail/editor-fail 分解 |
| critic 系统性偏差(比随机噪声更危险) | 新认识 | 按编辑类别审计;异源 RM ensemble;正例高精度校准 |
| ThinkEdit/Pico-Banana license | 新 | NC-ND 数据隔离,仅用于非衍生用途(critic 评测);主训练集用 Apache/MIT 来源 |
| GPU/API 预算 | — | RFT-first 路线 + rollout 缓存回放 + 前缀树渲染叶子,把渲染次数压到原方案 1/2-1/3 |

## 7. 本计划相对旧文档的差异一览

- `plan.md`/`training_plan.md` 的 Stage 1 SFT "用 v1 7k + 真实补充" → **改为 3k-8k 推理密集平衡集**;v1 程序化数据降级为格式 bootstrap 的小配比,v2r/v3 真实对标注只保留过滤子集。
- 真实编辑对长采集(run_long_collection)**停止作为 planner SFT 主线**;其产物转用于 critic/preservation 训练与 negative 样本。
- Stage 3 前**新增 RFT 阶段**。
- Edit-OPD 从 GenEvolve 原版 SDL **升级为门控版**(SDAR/SG-OPD/EDGE-OPD/stop-grad Top-K)。
- RISE-Critic 从自研为主 → **开源 RM ensemble 为底 + 自研 cognitive gate/checklist 为壳**;新增 grid 并排重打分。
- 评测从 n20 → 分层 n≥100 dev + 全量 final;新增接口扰动测试与 BoN 双曲线报告。
- 去污染新增图像双通道(pHash + SSCD)并入 CI。
