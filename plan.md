# RISEvolve: Self-Evolving Reasoning-Informed Image Editing Agent

## 1. 项目定位

**核心思路**：将 GenEvolve 的 agent 自进化框架从图像生成迁移到推理密集型图像编辑，在 RISE-Bench 和 GRADE 两个 benchmark 上刷分。

**目标 Benchmark**：
- **RISE-Bench** (NeurIPS 2025 DB Oral): 360 样本，4 类推理编辑（Temporal/Causal/Spatial/Logical），当前最强 GPT-4o-Native 仅 35.9% accuracy
  - 注：RISEBench 公开 arXiv v4 (arXiv:2504.02826) 摘要报告 GPT-4o-Image accuracy 为 28.8%；35.9% 可能来自后续 leaderboard 或不同评测版本。正式实验前需要冻结 benchmark/evaluator 版本并核对引用。
- **GRADE**: 520 样本，10 学科知识编辑（数学/物理/化学/生物/计算机/经济/历史/地理/音乐/体育），当前最强 Nano Banana Pro 仅 46.2%

**核心卖点**：现有编辑模型直接"看指令就改图"，缺乏推理和知识检索能力。RISEvolve 让 agent 先推理、搜索、规划，再输出结构化编辑指令给下游模型执行，本质是**把 reasoning 从编辑模型中解耦出来交给专门的 agent**。

---

## 2. 方法概览

```
输入: (source_image, edit_instruction)
          ↓
   RISEvolve Agent (Qwen3-VL-8B)
     ├─ analyze_image(src)        → 理解源图内容与关键区域
     ├─ search(queries)           → 检索领域知识/推理依据
     ├─ image_search(query)       → 找编辑目标的参考图
     └─ query_edit_knowledge(skill) → 激活编辑推理技能
          ↓
   输出: Edit Program (JSON)
     {
       "reasoning_chain": "推理过程...",
       "edit_prompt": "精炼的编辑指令，引用参考图...",
       "reference_images": [{img_id, note}],
       "edit_region": "optional, 编辑区域描述"
     }
          ↓
   下游编辑模型执行
     - 强: GPT-4o-Image / Gemini Image
     - 开源: Qwen-Image-Edit / FLUX-edit
```

---

## 3. 与 GenEvolve 的关键差异（Novelty）

| 维度 | GenEvolve | RISEvolve (ours) |
|------|-----------|------------------|
| 任务 | 开放式图像生成 | 推理密集型图像编辑 |
| 输入 | 纯文本 prompt | **源图 + 编辑指令**（多模态输入） |
| 新工具 | 无 | **analyze_image**: agent 先理解源图再行动 |
| 新技能 | 生成导向（layout, aesthetic...） | **推理导向**: temporal/causal/spatial/logical/discipline reasoning |
| 输出 | gen_prompt + refs | **edit_prompt + refs + reasoning_chain + edit_region** |
| 奖励 | 生成质量 | **编辑质量**: 推理正确性 + 非编辑区保真 + 视觉合理性 |
| Experience | 搜索/知识/参考选择经验 | + **推理链经验**: 哪种推理策略对哪类任务有效 |

---

## 4. 工具设计

### 4.1 analyze_image(image, focus=None)

Agent 的第一步，理解源图内容。返回结构化描述：
- 主体对象及属性
- 空间布局
- 当前状态（用于时序/因果推理）
- 可编辑区域

```
<tool_call>
{"name": "analyze_image", "arguments": {"focus": "banana condition and surroundings"}}
</tool_call>

→ Observation:
主体: 一根黄色香蕉，表面光滑无斑点，成熟度约70%
位置: 画面中央偏左，放在木质桌面上
状态: 新鲜，未剥皮，轻微弯曲
背景: 米色墙壁，自然光从左侧射入
可编辑区域: 香蕉本体(中心)，桌面(下半部分)
```

**实现方式**: 调用 VLM（Qwen3-VL 自身或外部 Gemini）对源图做结构化分析。

### 4.2 search(queries, top_k=5)

与 GenEvolve 相同，用 Serper API 检索文本知识。

**编辑场景典型用法**:
- GRADE 物理题: `search(["ice refraction index", "light bending through ice"])`
- RISE 时序题: `search(["banana decomposition timeline stages"])`
- GRADE 历史题: `search(["Roman Colosseum original appearance reconstruction"])`

### 4.3 image_search(query, top_k=5)

与 GenEvolve 相同，检索参考图片。

**编辑场景典型用法**:
- 需要知道编辑目标长什么样: `image_search("rotten decomposed banana closeup")`
- 需要学科参考: `image_search("light refraction through glass prism diagram")`

### 4.4 query_edit_knowledge(skill_name)

替换 GenEvolve 的 8 个生成技能，设计编辑推理导向的技能体系：

| 技能名 | 触发条件 | 对应 Benchmark |
|--------|----------|----------------|
| `temporal_reasoning` | 时间变化（老化、生长、腐烂、季节） | RISE-Temporal |
| `causal_reasoning` | 因果关系（施力→变形、加热→融化） | RISE-Causal |
| `spatial_reasoning` | 视角变换、物体重排、3D推理 | RISE-Spatial |
| `logical_reasoning` | 数独、迷宫、棋盘、数学推理 | RISE-Logical |
| `physics_knowledge` | 物理定律（光学、力学、热力学） | GRADE-Physics |
| `chemistry_knowledge` | 化学反应、分子结构 | GRADE-Chemistry |
| `biology_knowledge` | 生物过程、解剖学 | GRADE-Biology |
| `humanities_knowledge` | 历史还原、地理特征、文化元素 | GRADE-History/Geography |
| `visual_consistency` | 保持非编辑区域一致性 | 通用 |
| `edit_region_planning` | 确定需要编辑的精确区域 | 通用 |

每个 skill 返回一个 markdown 指导文档，教 agent 如何针对该类任务构造编辑指令。

---

## 5. 数据构造

### 5.1 Prompt (编辑请求) 构造

#### 来源一：从 Benchmark 反向扩充

RISE-Bench 360 + GRADE 520 = 880 个测试样本。这些不能用于训练，但可以分析其**任务分布和 recipe 模式**，然后用 recipe 控制生成大量同类训练数据。

```python
# Recipe 示例（RISE-Temporal 类型）
recipe = {
    "task_family": "temporal_reasoning",
    "source_image_desc": "a fresh red apple on a kitchen counter",
    "edit_instruction_template": "Show what this {object} will look like after {time_period}",
    "required_knowledge": ["decomposition process", "color change timeline"],
    "difficulty": "medium",
    "expected_visual_change": "browning, wrinkling, mold spots",
}
```

#### 来源二：公开编辑数据集改造

从以下数据集提取 source image，用 GPT-4o/Claude 生成推理密集型编辑指令：

| 数据集 | 用途 | 预估可用量 |
|--------|------|-----------|
| MagicBrush | 通用编辑 source images | ~5,000 |
| InstructPix2Pix | 编辑指令参考 | ~30,000 (筛选推理类) |
| EditBench | 高质量编辑 pairs | ~240 |
| SEED-Bench-2 | 多模态理解 | ~2,000 (转编辑) |

#### 来源三：学科知识驱动生成（针对 GRADE）

用 GPT-4o 针对 10 个学科批量生成：
```
对于{学科}，生成一个需要{学科知识}才能正确编辑的图像编辑任务。
要求：
1. 提供源图描述（用于检索或生成源图）
2. 编辑指令（隐式需要学科知识，不直接说答案）
3. 正确编辑结果的描述
4. 需要的具体知识点
```

**目标总量**: ~10,000 条训练 prompt + ~500 条验证 prompt

### 5.2 Source Image 获取

- 部分来自公开数据集（MagicBrush, EditBench）
- 部分用 FLUX/SDXL 根据 recipe 中的 source_image_desc 生成
- GRADE 学科类：从网上检索真实图片（实验装置、历史建筑等）

### 5.3 Teacher Trajectory 生成

与 GenEvolve 相同的 pipeline，但工具改为编辑导向：

**Teacher 模型**: Gemini 2.5 Pro 或 Claude 4 Sonnet（需要强多模态理解 + 推理能力）

**流程**:
```
1. 给 teacher 模型：system_prompt + source_image + edit_instruction
2. Teacher 按 ReAct 格式输出多轮 tool call：
   Round 1: <think>分析任务</think> + analyze_image(src)
   Round 2: <think>需要什么知识</think> + search(queries)
   Round 3: <think>找参考图</think> + image_search(query)
   Round 4: <think>激活技能</think> + query_edit_knowledge(skill)
   Round 5: <think>综合推理</think> + <answer>{edit_program}</answer>
3. 所有 tool call 真实执行（Serper API + VLM 分析）
4. 收集完整轨迹 τ = (src, x, a₁, o₁, ..., aₜ, oₜ, z)
```

**每条 trajectory 成本估算**:
- Teacher API: ~5 轮对话 × ~2K tokens/轮 ≈ 10K tokens
- Search API: ~2-3 次 Serper 调用 ≈ $0.003
- Image search: ~1-2 次 ≈ $0.002
- 总计: ~$0.02-0.05/条 → 10,000 条 ≈ $200-500

### 5.4 Trajectory 过滤

**阶段一：程序化硬检查**（参照 GenEvolve）
- 轨迹完整性（必须有 analyze_image + 至少一次 search/image_search + answer）
- JSON 格式合法
- edit_prompt 中用 ordinal phrase 引用参考图
- 无 URL 泄露
- reasoning_chain 非空

**阶段二：VLM 质量评审**
用 Gemini/GPT-4o 评审 6 个维度：
1. 源图分析准确性（analyze_image 结果是否正确）
2. 知识检索相关性（搜索的内容是否与任务相关）
3. 参考图选择合理性
4. 推理链正确性（最关键 — 推理过程是否逻辑自洽）
5. 编辑指令忠实性（最终指令是否覆盖了所有必要修改）
6. 训练价值

**目标**: 保留率 ~60-70%，最终得到 ~6,000-7,000 条 SFT 数据

### 5.5 GT 编辑图像生成

用强编辑模型渲染 edit program：
- **首选**: GPT-4o-Image（当前 RISE-Bench 最强）
- **备选**: Gemini Image / Nano Banana Pro

生成后再过滤：
- 编辑是否正确执行
- 非编辑区是否保持一致
- 推理结果是否视觉合理

---

## 6. 训练流程

### 6.1 Stage 1: SFT Cold Start

**目标**: 让 Qwen3-VL-8B 学会工具调用流程和编辑推理的基本模式。

```yaml
# 训练配置（参照 GenEvolve Table 9）
framework: LLaMA-Factory
deepspeed: ZeRO-3
backbone: Qwen3-VL-8B-Instruct
trainable: language_policy_only  # freeze vision encoder + projector
data: ~6,000 training + ~300 validation trajectories
cutoff_len: 32768
optimizer: AdamW
weight_decay: 1e-6
learning_rate: 1e-5
lr_scheduler: cosine
warmup_ratio: 0.02
epochs: 2
hardware: 8-16 GPUs (A100/H100)
micro_batch_size: 2
gradient_accumulation: 1  # 有效 batch = GPU数 × 2
precision: bf16
flash_attention: true
activation_checkpointing: true
loss_mask: assistant_tokens_only  # mask 掉 user prompt 和 tool observation
```

**数据格式**: 标准多轮对话，包含源图（作为图片 token 输入）：
```json
{
  "messages": [
    {"role": "system", "content": "<system_prompt>"},
    {"role": "user", "content": [
      {"type": "image", "image": "source_image.jpg"},
      {"type": "text", "text": "Edit instruction..."}
    ]},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>{\"name\":\"analyze_image\",...}</tool_call>"},
    {"role": "user", "content": "Observation: ..."},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>{\"name\":\"search\",...}</tool_call>"},
    {"role": "user", "content": "Observation: ..."},
    {"role": "assistant", "content": "<think>...</think>\n<answer>{...}</answer>"}
  ]
}
```

**SFT 只训 assistant 部分的 token**，user/system/observation 全部 mask。

### 6.2 Stage 2: Self-Evolution (GRPO + VED)

**目标**: 通过 on-policy 采样和自进化，持续提升 agent 的推理和编辑规划能力。

#### 6.2.1 Rollout 采样

```yaml
framework: verl (rLLM) + SGLang rollout
hardware: 1 node × 8 GPUs, FSDP, parameter+optimizer offload
init: SFT checkpoint

rollout:
  prompts_per_step: 8
  rollouts_per_prompt: 6  # K=6
  temperature: 0.7
  top_p: 0.95
  max_prompt_tokens: 6144
  max_response_tokens: 30000
  max_tool_calls: 11
```

每步采样 8×6=48 条轨迹。每条轨迹：
1. Agent 拿到 (source_image, edit_instruction)
2. 自主决定调用哪些工具、搜索什么、用什么技能
3. 输出 edit_program
4. 下游模型执行编辑，生成 edited_image

#### 6.2.2 奖励函数设计（关键改动点）

```
R = 0.5 × R_edit + 0.5 × R_program
```

**R_edit（编辑质量评分）— 用 VLM Judge 打分**:

针对 RISE-Bench 和 GRADE 的评分维度：

| 维度 | 权重 | 说明 |
|------|------|------|
| Reasoning Correctness | 0.4 | 编辑结果是否体现了正确的推理（时序变化是否合理、因果关系是否正确、学科知识是否准确） |
| Appearance Consistency | 0.3 | 非编辑区域是否保持一致（背景、无关物体、光照风格） |
| Visual Plausibility | 0.2 | 编辑结果是否视觉自然、无 artifact |
| Instruction Following | 0.1 | 是否完成了用户指令的所有要求 |

Judge prompt 示例：
```
给定：
- 源图 (source image)
- 编辑指令 (edit instruction)
- 编辑结果 (edited image)
- [可选] 参考答案描述

请从以下维度评分 (1-5):
1. 推理正确性: 编辑结果是否体现了正确的推理？
2. 外观一致性: 未编辑区域是否保持不变？
3. 视觉合理性: 结果是否自然、无明显瑕疵？
4. 指令遵循: 是否完成了所有编辑要求？
```

**R_program（程序充分性评分）**:

检查 edit_program 是否包含：
- 正确的推理链（reasoning_chain 逻辑合理）
- 充分的知识 grounding（搜索到的事实被使用）
- 参考图被正确引用
- 编辑指令覆盖所有必要修改

5-bin 评分: {0, 0.25, 0.5, 0.75, 1}

**Judge 模型**: Gemini 2.5 Pro（或 GPT-4o）

#### 6.2.3 GRPO 训练

```yaml
grpo:
  advantage: group_relative  # Â_i = (R_i - R̄) / (σ_R + ε)
  learning_rate: 1e-6
  lr_scheduler: cosine
  warmup: none
  clip_ratio_low: 0.20
  clip_ratio_high: 0.28  # 非对称 high clip
  kl_coef: 1e-3
  aggregation: seq-mean-token-sum
```

#### 6.2.4 Visual Experience Distillation (VED/SDL)

**Experience 提取**：

对每组 6 条轨迹，找 best/worst pair：
```
τ⁺ = argmax R(τᵢ), τ⁻ = argmin R(τᵢ)
Δ = R(τ⁺) - R(τ⁻)
```
仅当 Δ ≥ 0.20 时提取 experience bundle。

**6 类编辑经验 slot**（比 GenEvolve 多一类 reasoning）：

| Slot | 内容 |
|------|------|
| M_analyze | 源图分析策略差异：好的轨迹如何更准确地理解源图 |
| M_search | 搜索策略差异：搜了什么关键词、哪些有用 |
| M_ref | 参考图选择差异：选了什么参考、为什么更好 |
| M_reasoning | **推理链差异**：好的轨迹如何构建正确的推理链 |
| M_edit | 编辑指令构造差异：最终指令如何更精确 |
| M_fail | 失败规避：差的轨迹犯了什么错 |

**Experience 存储与检索**:
```yaml
experience_memory:
  summarizer: Gemini 2.5 Pro  # 生成 experience bundle
  buffer_capacity: 500
  eviction: FIFO + reward_gap
  embedder: Qwen3-Embedding-0.6B  # 按 prompt 相似度检索
  min_reward_gap: 0.20
  max_comparisons_per_step: 8
```

**SDL 损失**:
```yaml
sdl:
  coefficient: 2.0  # λ_SDL = 2.0
  importance_ratio_cap: 2.0  # ρ_max
  kl_estimator: k3  # clamp [-10, 10]
  token_selection: top_10_percent  # 只在分歧最大的 10% token 上蒸馏
  mask: assistant_tokens_with_experience  # 仅有 experience 的序列
```

**Teacher-Student 机制**:
- Student: 正常 context = (source_image, edit_instruction, tool history)
- Teacher: 增强 context = 正常 context + Patch(retrieved_experience)
- 共享同一套权重，teacher 的梯度 detach（stop gradient）
- 只在训练时有 teacher 分支，推理时只用 student

#### 6.2.5 联合损失

```
ℒ = ℒ_GRPO + 2.0 × ℒ_SDL
```

每步同时优化，不交替。

---

## 7. 评测

### 7.1 RISE-Bench 评测

```bash
# 1. Agent rollout: 对 360 个测试样本生成 edit program
python scripts/run_agent.py \
  --input rise_bench_test.jsonl \
  --output rise_bench_programs.jsonl

# 2. 下游模型执行编辑
python scripts/generate_edits.py \
  --programs rise_bench_programs.jsonl \
  --backend gpt-4o-image \
  --output rise_bench_results/

# 3. 官方评测
python rise_bench_eval.py --results rise_bench_results/
```

评测指标：
- Overall Score = 0.4×Consistency + 0.4×InsReasoning + 0.2×Plausibility
- Overall Accuracy = 所有维度满分的比例
- 分类报告：Temporal / Causal / Spatial / Logical

### 7.2 GRADE 评测

```bash
# 同样流程，用 GRADE 官方评测
python grade_eval.py --results grade_results/
```

评测指标：
- Accuracy (Strict): 所有推理题正确 + 视觉/可读性满分
- Relax Score: 0.6×Reasoning + 0.3×Consistency + 0.1×Readability
- 分学科报告：Math/Physics/Chemistry/...

---

## 8. 时间线与资源估算

### 8.1 时间线

| 阶段 | 工作内容 | 预估时间 |
|------|----------|----------|
| Week 1 | 工具实现 + system prompt 设计 + skill 文档撰写 | 1 周 |
| Week 2-3 | Recipe 设计 + Teacher trajectory 生成 + 过滤 | 2 周 |
| Week 3 | Source image 收集/生成 | 与 W2 并行 |
| Week 4 | SFT 训练 + 验证 | 1 周 |
| Week 5-6 | GRPO + VED 自进化训练 | 2 周 |
| Week 7 | 评测 + 消融实验 | 1 周 |
| Week 8 | 论文撰写 | 1 周 |
| **总计** | | **~8 周** |

### 8.2 资源估算

| 资源 | 用量 | 成本估算 |
|------|------|----------|
| Teacher trajectory 生成 (Gemini/Claude API) | ~10K 条 × $0.03 | ~$300 |
| GT 编辑图像生成 (GPT-4o-Image) | ~3K 张 × $0.04 | ~$120 |
| RL Judge 调用 (Gemini) | ~500 steps × 48 calls × $0.01 | ~$240 |
| Experience 提取 (Gemini) | ~500 steps × 8 calls × $0.01 | ~$40 |
| SFT 训练 GPU | 8-16 × A100 × ~6h | ~$50-100 |
| RL 训练 GPU | 8 × A100 × ~48h | ~$200-400 |
| Serper API (search) | ~50K 次 | ~$25 |
| **总计** | | **~$1,000-1,200** |

---

## 9. 风险与应对

| 风险 | 概率 | 应对方案 |
|------|------|----------|
| 下游编辑模型执行力不足（agent 推理正确但模型改不出来） | 高 | 用最强模型 GPT-4o-Image 做下游；论文中分析 agent-generator gap |
| RISE Logical 类任务（数独/迷宫）太难，搜索无法帮助 | 高 | 重点投入 Temporal/Causal/Spatial + GRADE 学科题；Logical 作为 limitation |
| Novelty 不足，reviewer 认为只是换了 domain | 中 | 强调编辑特有贡献：analyze_image 工具、reasoning_chain 输出、编辑区域感知、推理经验蒸馏 |
| SFT 数据质量不够（teacher 轨迹质量差） | 中 | 多轮过滤 + 小规模人工抽检 + 迭代优化 recipe |
| RL 训练不稳定 | 低 | 参照 GenEvolve 超参，渐进调试 |

---

## 10. 关键文件结构

```
rise-evolve/
├── plan.md                     # 本文件
├── survey.md                   # agentic training / reasoning editing 调研
├── data_pipeline.md            # RISE/GRADE/KRIS 对齐的数据构造 pipeline
├── genevolve/                   # fork from GenEvolve, 改造
│   ├── system_prompt.py         # 编辑版 system prompt
│   ├── agent.py                 # 加入 analyze_image 工具
│   ├── tools/
│   │   ├── web_search.py        # 复用
│   │   ├── image_search.py      # 复用
│   │   └── image_analyzer.py    # 新增：源图分析
│   ├── knowledge/
│   │   └── skills/              # 10 个编辑推理技能 markdown
│   │       ├── temporal_reasoning.md
│   │       ├── causal_reasoning.md
│   │       ├── spatial_reasoning.md
│   │       ├── logical_reasoning.md
│   │       ├── physics_knowledge.md
│   │       ├── chemistry_knowledge.md
│   │       ├── biology_knowledge.md
│   │       ├── humanities_knowledge.md
│   │       ├── visual_consistency.md
│   │       └── edit_region_planning.md
│   └── generator.py             # 对接编辑模型（非生成模型）
├── data/
│   ├── benchmarks/              # RISE/GRADE/KRIS 快照、指纹、去污染索引
│   ├── taxonomy/                # benchmark-derived taxonomy/checklist templates
│   ├── recipes/                 # prompt 生成 recipe
│   ├── tasks/                   # materialized task JSONL
│   ├── images/                  # source/reference/generated/programmatic images
│   ├── trajectories/            # teacher trajectory
│   ├── programs/                # edit program v2
│   ├── renders/                 # teacher/student/negative renders
│   ├── verifier/                # checklist/reward 校准数据
│   └── splits/                  # sft/rl/verifier/ved/hard heldout split
├── scripts/
│   ├── data/                    # freeze/mine/generate/materialize/filter/split
│   ├── generate_recipes.py      # 生成 recipe
│   ├── run_teacher.py           # 跑 teacher trajectory
│   ├── filter_trajectories.py   # 过滤
│   ├── train_sft.sh             # SFT 训练
│   ├── train_rl.sh              # GRPO+VED 训练
│   ├── run_agent.py             # 推理
│   ├── generate_edits.py        # 下游编辑
│   └── evaluate.py              # 评测
├── configs/
│   ├── sft_config.yaml
│   └── rl_config.yaml
└── evaluation/
    ├── rise_bench/
    └── grade/
```

---

## 11. 文献调研后的 Plan v2 增补

详细调研见 `survey.md`。核心判断是：2026 年的 reasoning-centric image editing 已经出现 DDA-Thinker、MIRA、RePlan、ThinkRL-Edit、EditThinker、Edit-R1、RewardHarness 等强相关工作。RISEvolve 不能只定位为“给 editor 前面加 thinker”，而应明确为 **tool-orchestrated self-evolving editing agent**：学习源图分析、知识检索、参考图选择、技能路由、区域规划、atomic verification 和 edit-program synthesis 的完整决策过程。

### 11.1 更新后的 novelty 定位

| 对比对象 | 已覆盖内容 | RISEvolve 需要强调的差异 |
|----------|------------|--------------------------|
| DDA-Thinker | 固定 editor，优化 thinker；dual-atomic reward | RISEvolve 不只是 planner，还学习 search/image_search/skill/memory 的工具编排，并用 best-worst experience distillation 提供 token-level 决策监督 |
| MIRA | perception-reasoning-action loop；SFT+GRPO | RISEvolve 面向知识和推理密集任务，显式做外部知识 grounding 和参考图 grounding |
| RePlan | region-aligned planning | RISEvolve 将 region planning 纳入 edit program 和 reward，同时覆盖 temporal/causal/logical/discipline reasoning |
| Edit-R1 / EditReward | 编辑 reward/verifier | RISEvolve 可以使用 verifier，但优化对象是 agentic edit-program policy |
| RewardHarness | 自演化 reward skill library | RISEvolve 演化的是编辑 agent 的工具和计划经验，而不只是 judge 上下文 |
| GenEvolve | 生成任务的 tool-orchestrated VED | RISEvolve 改成源图条件下的 visual-cognitive editing experience，增加区域、保真、推理正确性和 editor gap 诊断 |

建议论文主张：

> RISEvolve trains a self-evolving, tool-orchestrated image editing agent that converts reasoning- and knowledge-intensive edit requests into executable, region-aware edit programs. Unlike planner-only approaches, it jointly learns source-image analysis, knowledge retrieval, reference selection, skill activation, atomic verification, and edit-program synthesis through visual-cognitive experience distillation.

### 11.2 Edit Program v2

原来的 `edit_prompt + refs + reasoning_chain + edit_region` 不够区分 plan quality 和 editor execution。建议输出 schema 升级为：

```json
{
  "source_scene_graph": {
    "objects": [],
    "attributes": [],
    "relations": [],
    "current_state": "",
    "uncertain_observations": []
  },
  "task_family": "temporal|causal|spatial|logical|discipline|mixed",
  "knowledge_facts": [
    {"claim": "", "source": "search|skill|model", "used_in_plan": true}
  ],
  "target_scene_description": "理想编辑结果的文字描述，用于 checklist 和 judge",
  "edit_operations": [
    {
      "op": "add|remove|replace|transform|move|style|text|geometry",
      "target": "",
      "region": "",
      "desired_change": "",
      "preserve": []
    }
  ],
  "reference_images": [
    {"img_id": "IMG_001", "role": "identity|material|layout|domain_reference", "note": ""}
  ],
  "preservation_constraints": [
    "背景、光照、相机视角、无关对象和主体身份保持不变"
  ],
  "negative_constraints": [
    "不要新增无关对象；不要改变未指定文字；不要全图风格漂移"
  ],
  "atomic_checklist": {
    "cognitive": [],
    "visual": [],
    "preservation": [],
    "readability": []
  },
  "editor_prompt": "给下游 editor 的精炼可执行指令",
  "failure_modes_to_watch": []
}
```

关键变化：
- `target_scene_description` 作为 DDA-Thinker/Edit-R1 式 checklist synthesis 的锚点，降低 judge 噪声。
- `edit_operations` 把复杂指令拆成 region-aware atomic edits，便于定位 planner failure。
- `atomic_checklist` 同时服务训练 reward、离线过滤和失败诊断。
- `negative_constraints` 专门抑制 over-editing，这是当前编辑模型常见失败。

### 11.3 工具设计 v2

在现有四个工具基础上，建议把工具分成必选和可选两层：

| 工具 | 类型 | 用途 |
|------|------|------|
| `analyze_image(image, focus)` | 必选 | 源图对象、关系、状态、可编辑区域 |
| `search(queries)` | 条件必选 | 学科知识、物理/历史/生物事实、规则 |
| `image_search(query)` | 条件必选 | 获取目标状态、材质、历史外观、图示参考 |
| `query_edit_knowledge(skill)` | 必选/可选 | 激活任务族策略和失败规避 |
| `ground_region(image, target)` | 可选 | 输出 bbox/mask/region phrase，支持 RePlan/RC-GRPO 风格局部保真 |
| `solve_symbolic(problem)` | 可选 | 数独、迷宫、算式、棋盘等 logical 类任务 |
| `verify_edit(source, instruction, program, edited)` | 训练/评测 | checklist-based verifier，返回失败项和诊断 |

工具策略要加入 reward：
- 重复 query、未使用证据、无关检索惩罚。
- 对无需检索的简单 case，允许 agent 不调用 `search`。
- 对 GRADE/历史/科学 case，缺少知识 grounding 直接降低 `R_cognitive`。

### 11.4 奖励函数 v2

建议从单一 weighted sum 改成多头 reward，并分别归一化 advantage，避免 DDA-Thinker/ThinkRL-Edit 指出的 reward fusion bias。

```
R_cognitive: edit_program 是否逻辑正确、知识正确、可执行、checklist 完整
R_visual: edited_image 是否满足目标变化、视觉自然、无 artifact
R_preserve: 非编辑区是否保持一致，结合 VLM + PSNR/SSIM/LPIPS/DINO/mask similarity
R_tool: 工具调用是否必要、有效、无重复，证据是否进入 final program
```

训练时可以实现为：

```
Adv = normalize(R_cognitive) + normalize(R_visual) + normalize(R_preserve) + 0.2 * normalize(R_tool)
```

或更严格地做 separate-GRPO：
- cognitive group 更新 plan/reasoning/tool tokens；
- visual/preserve group 更新 editor_prompt、region、negative_constraints tokens；
- tool group 更新 tool-call decision tokens。

Judge prompt 应先从 `target_scene_description` 生成二值 checklist，再逐项判断，最后汇总分数。不要直接让 VLM 给 1-5 总分。

### 11.5 数据构造 v2

训练数据建议分四层，而不是只从 benchmark recipe 和通用编辑数据改造：

| 数据层 | 来源 | 目标 |
|--------|------|------|
| 基础编辑 | MagicBrush、InstructPix2Pix、ImgEdit | 学会常规 edit program 格式 |
| 推理编辑 | RISEBench recipe、Reason50K、CompBench、ByteMorph | temporal/causal/spatial/logical |
| 知识编辑 | GRADE recipe、KRIS-Bench taxonomy、学科知识库 | discipline/factual/conceptual/procedural |
| agentic hard cases | SFT/RL 中失败样本自扩展 | 针对模型弱点做 curriculum |

每条 teacher trajectory 增加：
1. rational target description；
2. atomic checklist；
3. tool evidence usage map；
4. editor feasibility note：判断这个计划是否是当前 editor 能执行的。

过滤时新增：
- `target_scene_description` 与 `editor_prompt` 是否一致；
- checklist 是否可视觉验证；
- source image 分析是否错误；
- final prompt 是否引入未被知识或参考图支持的新事实；
- 是否存在 teacher reasoning 正确但 editor 明显做不到的样本，进入单独的 hard/editor-gap split。

### 11.6 训练路线 v2

建议把阶段拆得更可控：

| 阶段 | 目标 | 数据量建议 | 产出 |
|------|------|------------|------|
| Stage 0: Prompted agent baseline | 不训练，仅用强 VLM planner 跑 RISE/GRADE 小样本 | 100-200 eval | 确认 editor 上限和 failure taxonomy |
| Stage 1: SFT format cold start | 学会 ReAct、工具调用、edit program v2 | 3K-8K trajectories | 可稳定输出 JSON 和 checklist |
| Stage 2: Verifier/judge calibration | 训练或固化 checklist judge | 1K-5K preference/checklist | 降低 reward 噪声 |
| Stage 3: GRPO on agent | 优化 tool/plan/program tokens | 300-800 steps | 提升 reasoning/edit correctness |
| Stage 4: Visual-Cognitive Experience Distillation | best-worst 经验蒸馏 | 与 Stage 3 联合 | 学到可迁移工具和计划策略 |
| Stage 5: targeted self-evolution | 只对失败族扩数据 | 1K-3K hard cases | 改善 RISE logical/GRADE 弱学科 |

Stage 3/4 的 experience slot 建议调整为：

| Slot | 内容 |
|------|------|
| `M_analyze` | 源图观察和 region 判断差异 |
| `M_knowledge` | 需要检索/无需检索的判断，事实使用情况 |
| `M_reference` | 参考图角色分配和引用方式 |
| `M_reasoning` | temporal/causal/spatial/logical/discipline 推理差异 |
| `M_region` | edit vs preserve 区域规划差异 |
| `M_editor` | 哪些 prompt 写法更适合固定 editor 执行 |
| `M_failure` | over-edit、under-edit、hallucination、symbolic rendering 失败规避 |

### 11.7 实验和消融

必须补的 baseline：
- direct editor：Qwen-Image-Edit / FLUX.1 Kontext / Gemini / GPT-4o-Image；
- prompted planner：GPT-4o/Gemini 生成 edit prompt + same editor；
- DDA-like：无 search/image_search/memory，仅 thinker 输出 plan；
- RePlan-like：region planner + same editor；
- MIRA/EditThinker-like：iterative critique-refine；
- RISEvolve-SFT；
- RISEvolve-GRPO；
- RISEvolve-GRPO+VED。

关键消融：
- no `analyze_image`；
- no `search`；
- no `image_search`；
- no `query_edit_knowledge`；
- no `target_scene_description`；
- no `atomic_checklist`；
- no `edit_region` / no preserve reward；
- weighted-sum reward vs separate/normalized reward；
- no VED；
- no tool penalty；
- different editor backends。

需要报告的诊断：
- thinker correct / editor fail 的比例；
- editor correct despite weak plan 的比例；
- search 使用率、有效率、重复率；
- 每类 reasoning 的 reward breakdown；
- 非编辑区保真指标；
- checklist 与人工评审一致性。

### 11.8 风险更新

| 风险 | 更新判断 | 应对 |
|------|----------|------|
| Novelty 被 planner-only 工作覆盖 | 高 | 主张 tool-orchestrated self-evolution + knowledge/reference grounding + VED，不只说 thinker |
| Judge reward hacking | 高 | checklist、双 judge、少量人工校准、EditReward/Edit-R1 交叉验证 |
| Editor 执行力限制 logical/math | 高 | oracle-plan upper bound、failure attribution、必要时 logical 类走 specialized solver + text rendering editor |
| 工具过度调用 | 中 | `R_tool`、knowledge boundary audit、按任务族限制预算 |
| 数据成本 | 中 | 先做 1K-2K 高质量 proof，验证提升后再扩到 10K |

### 11.9 最近两周执行优先级

1. 固定评测版本：拉取 RISEBench、GRADE、KRIS-Bench，确认 license、数据格式、leaderboard 指标。
2. 先实现 prompted RISEvolve baseline：不训练，用 Gemini/GPT-4o 按 edit program v2 生成计划，接 Qwen-Image-Edit/FLUX.1 Kontext。
3. 在 100 个样本上做 failure taxonomy：planner fail、knowledge fail、region fail、editor fail、judge fail。
4. 写 10 个 skill markdown，每个包含 observation checklist、search recipe、region recipe、editor prompt recipe、common failures。
5. 生成 500 条 teacher trajectories，人工抽检 50 条，验证 schema 和 checklist 是否稳定。
6. 再决定是否进入 3K-8K SFT 数据生成。

### 11.10 数据构造 Pipeline v1

详细方案见 `data_pipeline.md`。核心原则是：RISE/GRADE/KRIS 只用于 official eval、taxonomy mining、rubric abstraction 和 decontamination，不把 benchmark 原图、指令、GT、reference、annotation 或其同义改写放进训练集。

数据流固定为：

```text
benchmark snapshots
  -> benchmark fingerprints
  -> taxonomy/checklist templates
  -> recipe bank
  -> source image pool
  -> materialized tasks
  -> teacher trajectories
  -> edit renders and rollouts
  -> filtering/scoring
  -> SFT/RL/verifier/VED splits
```

首批任务分布：

| Bucket | 占比 | 覆盖重点 |
|--------|------|----------|
| RISE-like | 40% | temporal/causal/spatial/logical，logical 先保底 15% 但优先程序渲染 |
| GRADE-like | 35% | 10 个学科域，借鉴 `questions` rubric 生成 checklist |
| KRIS-like | 25% | factual/conceptual/procedural knowledge，覆盖 anomaly、multi-element、temporal、viewpoint |

Pilot v0 目标：

| 产物 | 数量 |
|------|------|
| recipe candidates | 1,200 |
| source images | 800 |
| materialized tasks | 600 |
| teacher trajectories | 500 |
| accepted SFT | 300 |
| RL prompts | 100 |
| verifier items | 300-500 |
| human audit | 100 tasks |

Full v1 目标是在 pilot 通过后扩到 8K-9K materialized tasks、5K-6K accepted teacher trajectories、1K-1.5K RL prompts、5K-10K verifier items。

实现优先级：

1. `freeze_benchmarks.py`：记录 RISE/GRADE/KRIS 版本、license、evaluator、text/image fingerprints。
2. `mine_taxonomy.py`：只抽象 taxonomy 和 checklist templates，不导出 benchmark 原样样本。
3. `generate_recipes.py`：按 RISE/GRADE/KRIS-derived taxonomy 生成 recipe candidates，并做 text decontamination。
4. `acquire_source_images.py` + `materialize_tasks.py`：绑定新 source image，生成统一 task schema。
5. `run_teacher.py`：生成 ReAct trajectory、tool evidence map、rational target description、edit program v2。
6. `render_edits.py` + `filter_data.py`：生成 teacher/student/negative renders，按 schema、decontamination、evidence、program、render、human audit gates 过滤。
7. `build_splits.py`：按 recipe/entity/source/template 隔离构建 SFT、RL、verifier、VED、hard heldout。

当前 v0 已落地为轻量脚本版：

| 脚本 | 状态 | 作用 |
|------|------|------|
| `scripts/data/freeze_benchmarks.py` | 已实现 | 冻结 RISE/GRADE/KRIS 公开入口，生成 benchmark text fingerprint |
| `scripts/data/mine_taxonomy.py` | 已实现 | 输出 taxonomy、checklist templates、curated knowledge bank |
| `scripts/data/build_pilot_dataset.py` | 已实现 | 生成程序化 source/teacher/negative images、tasks、teacher trajectories、programs、splits |
| `scripts/data/validate_dataset.py` | 已实现 | 检查 JSONL、图片路径、split、checklist 权重、exact decontamination |

v0 产物：

| 产物 | 数量 |
|------|------|
| `data/tasks/tasks_v0.jsonl` | 600 tasks |
| `data/trajectories/teacher_trajectories_v0.jsonl` | 600 teacher trajectories |
| `data/renders/render_metadata_v0.jsonl` | 600 teacher + 600 negative renders |
| `data/splits/sft_train_v0.jsonl` | 300 SFT trajectories |
| `data/splits/rl_prompt_train_v0.jsonl` | 100 RL prompts |
| `data/splits/verifier_train_v0.jsonl` | 200 verifier items |
| `data/splits/ved_memory_train_v0.jsonl` | 25 experience pairs |
| `data/splits/hard_heldout_v0.jsonl` | 25 heldout tasks |

v0 的定位是先打通 agentic data schema 和训练文件格式。它使用程序化图像，因此可验证、可重复、成本低；下一轮 v1 应把 GRADE-like 学科题扩到更多参数化模板，同时接入真实/生成 source image 和强 teacher editor，提升视觉多样性。

v1 已扩到 1w agentic 数据：

| 产物 | 数量 |
|------|------|
| tasks / recipes / edit programs / teacher trajectories | 10,000 each |
| source images / teacher renders / negative renders | 10,000 each |
| verifier items | 20,000 |
| preference pairs / experience pairs | 10,000 / 10,000 |
| SFT train / val | 7,000 / 500 |
| RL prompts | 1,000 |
| verifier train items | 2,000 |
| VED memory pairs | 300 |
| hard heldout | 200 |

v1 分布为 RISE-like 4,000、GRADE-like 3,500、KRIS-like 2,500。`scripts/data/validate_dataset.py --version v1` 已通过，exact benchmark text match 为 0。v1 还加入了 instruction paraphrase、benchmark alignment metadata、task-level visual jitter；当前 instruction 去重约 9.4K/10K，source PNG 精确哈希约 5K 个桶。下一步质量提升应集中在：增加更多 GRADE 参数化生成器、接入真实/生成 source image、加入 CLIP/DINO semantic decontamination、对 200-500 条样本做人工抽检。

已补充审阅产物：`scripts/data/audit_dataset.py --version v1 --sample-size 96` 会输出 `reports/data_quality/audit_v1.json`、`review_sample_v1.jsonl` 和 4 张 source/teacher/negative 三联 contact sheet，便于人工快速抽检知识正确性、区域保真和 negative case 难度。

### 11.11 训练 Plan v3：GRPO + Edit-OPD

详细训练方案已单独整理到 `training_plan.md`。这里记录关键决策：

1. **优先训练 agent，不直接训练 editor**。RISE/GRADE/KRIS 的主要瓶颈是源图理解、知识/推理、区域规划、非编辑区保真和 checklist 验证；这些是离散工具/程序决策，先训练 VLM agent 更稳。
2. **SFT 冷启动使用现有 v1 数据，但必须补 real-image / strong-editor trajectories**。当前 1w 程序化数据适合学 schema、工具协议和 verifier，但视觉分布太窄；进入正式 SFT 前建议加入 1K-3K 高质量真实图或强编辑器渲染轨迹。
3. **主 RL 算法选 GRPO**。每个 prompt 采样 K=4-6 条 edit trajectories，render 后用多头 reward 打分，group-relative advantage 更新 assistant tokens。相比 PPO，GRPO 不需要 value model；相比 DPO，GRPO 能探索工具调用和 on-policy failure。
4. **加入 Edit-OPD / Visual-Cognitive Experience Distillation**。同 prompt best/worst rollout 的差异被总结为 experience；teacher branch 看到 region/checklist/diagnostics/experience，student branch 不看；teacher 对同一批 on-policy tokens 重新打分，给 student dense token-level guidance。
5. **Reward 必须多头记录，不能只保留总分**。推荐 heads：`R_cognitive`、`R_visual`、`R_preserve`、`R_region`、`R_tool`、`R_format`。先做 weighted reward，日志中保留 head breakdown；第二版再做 token-mask/head-aware advantage。
6. **OPD 的二维图像编辑特化**：region-aware、checklist-conditioned、editor-gap-aware、tool-boundary、multi-editor。核心是把“哪里该改/哪里不能动/为什么失败/该不该检索”蒸馏到 agent 的决策 token。

阶段计划：

| 阶段 | 目标 | 验收 |
|------|------|------|
| Stage 0 | Prompted planner + fixed editor baseline | 100-200 dev cases failure taxonomy |
| Stage 1 | SFT cold start | JSON/schema >98%，工具重复率 <10% |
| Stage 2 | Verifier/reward calibration | checklist-human agreement 通过人工抽检 |
| Stage 3 | Edit-GRPO | RISE/GRADE/KRIS dev reward 稳定提升 |
| Stage 4 | Edit-OPD | 对比 GRPO-only 有额外提升，工具和 region error 下降 |
| Stage 5 | Targeted self-evolution | 针对 weak buckets 扩 1K-3K hard cases |

最近实现顺序：

1. 把 `data/trajectories/teacher_trajectories_v1.jsonl` 转成 LLaMA-Factory/Qwen-VL SFT 格式。
2. 实现 prompted RISEvolve baseline 和 editor adapter。
3. 实现 checklist-first verifier，输出 reward head breakdown。
4. 小规模 SFT 后跑 100-case dev eval。
5. 做 K=4、50-100 step 的 GRPO debug，不开 OPD。
6. 接入 experience memory、teacher context patch、SDL/Edit-OPD loss，再扩到 300-800 step。

### 11.12 Reward Plan v2：RISE-Critic

FIRM / Trust Your Critic 的关键结论是：图像编辑 reward 的瓶颈在 critic，且简单加权会 reward hacking。特别是 `0.5 * Execution + 0.5 * Consistency` 会诱导 editor 输出几乎不变的图来获取高 consistency。FIRM 用 `CME = Execution * (0.6 + 0.4 * Consistency)` 把 execution 设为高 reward 的必要条件。

RISEvolve 不能只照搬 FIRM 的二头 reward，因为我们的任务是 reasoning/knowledge-heavy editing agent。建议 reward 模块升级为 **RISE-Critic**：

```text
expected diff planning
  -> difference-first observation
  -> checklist verification
  -> gated/base-and-bonus reward
  -> failure attribution
  -> head-aware token credit
```

核心 heads：

| Head | 用途 |
|------|------|
| `R_cog` | 推理、事实、学科、符号结果正确性 |
| `R_exec` | 目标编辑是否执行 |
| `R_preserve` | 非编辑区域、身份、视角、背景保持 |
| `R_region` | 编辑区域和局部性 |
| `R_quality` | 视觉自然度、artifact、可读性 |
| `R_tool` | 工具调用必要性、有效性、证据使用 |
| `R_format` | schema、reference binding、checklist 合规 |
| `R_program` | 不看 render 的 plan correctness，降低 editor 噪声 |

推荐 gated reward：

```text
G_task = min(R_exec, R_cog_applicable)

R_image = G_task * (
  0.45 + 0.20 R_preserve + 0.15 R_region + 0.10 R_quality + 0.10 R_readability
)

R_agent = 0.45 R_program + 0.45 R_image + 0.05 R_tool + 0.05 R_format
```

关键创新点：

1. **Difference-first critic**：先比较 source/edited 的真实差异，再按 expected diff 和 checklist 打分，减少 VLM 直接打分漏细节。
2. **Cognitive-gated CME**：在 FIRM 的 execution gate 之外加入 reasoning/knowledge gate，防止视觉上像但学科/逻辑错误的编辑得高分。
3. **Editor-gap-aware reward**：区分 `planner_fail` 和 `editor_fail`。若 program 正确但 editor 失败，保留较高 `R_program`，主要更新 editor prompt / region phrase / negative constraints。
4. **Head-aware token credit**：不同 token group 由不同 reward head 更新，避免一个 scalar reward 同时误伤工具、推理、区域和 prompt tokens。
5. **Reward diagnostics feed Edit-OPD**：failed checklist、observed diff、failure attribution 和 head breakdown 进入 teacher-only context，成为 Edit-OPD 的 privileged signal。

详细方案见 `reward_design.md`。下一步实现时，先做 verifier/reward 数据格式扩展：保存 `expected_diff`、`observed_diff`、`score_heads`、`failure_attribution`，再接入 GRPO。
