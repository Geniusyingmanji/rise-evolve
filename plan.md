# RISEvolve: Self-Evolving Reasoning-Informed Image Editing Agent

## 1. 项目定位

**核心思路**：将 GenEvolve 的 agent 自进化框架从图像生成迁移到推理密集型图像编辑，在 RISE-Bench 和 GRADE 两个 benchmark 上刷分。

**目标 Benchmark**：
- **RISE-Bench** (NeurIPS 2025 DB Oral): 360 样本，4 类推理编辑（Temporal/Causal/Spatial/Logical），当前最强 GPT-4o-Native 仅 35.9% accuracy
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
│   ├── recipes/                 # prompt 生成 recipe
│   ├── trajectories/            # teacher trajectory
│   ├── sft/                     # 过滤后 SFT 数据
│   └── rl/                      # RL 训练数据
├── scripts/
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
