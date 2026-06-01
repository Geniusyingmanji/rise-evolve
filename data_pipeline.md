# RISEvolve Data Construction Pipeline

目标：构造一套不泄漏 RISEBench/GRADE/KRIS-Bench 测试样本、但任务分布和评测 rubric 对齐的推理型图像编辑训练数据。数据需要同时服务四类训练或评估产物：

1. **SFT trajectories**：教会 agent 按 ReAct 方式分析源图、检索知识、选择工具、生成 edit program。
2. **RL prompts and rollouts**：给 GRPO/agentic RL 提供 prompt、候选 edit program、渲染图和 verifier reward。
3. **Verifier/reward data**：训练或校准 checklist judge、program judge、image judge。
4. **VED experience pairs**：把 good/bad trajectory 的差异蒸馏成可迁移经验。

## 1. 核心原则

1. **benchmark 只做评测和抽象 taxonomy mining**。RISE/GRADE/KRIS 的原图、指令、GT、reference、annotation 不进入训练集，也不做 paraphrase 后进入训练集。
2. **先生成 recipe，再落到图像和任务**。不要直接让 LLM 模仿 benchmark 样本，而是先抽象出任务族、知识点、视觉变化、检查项，再用新图像和新实体生成任务。
3. **每条数据都要有可验证结构**。最终任务不能只有 instruction，必须包含 expected target、edit operations、preservation constraints、negative constraints、atomic checklist 和 provenance。
4. **区分 planner failure 和 editor failure**。teacher edit program 正确但 editor 渲染失败的数据不一定丢弃，可进入 verifier 或 editor-gap 诊断集合，但不能直接作为 SFT 金数据。
5. **split 按实体、模板、来源和任务族隔离**。避免同一图像、同一实体、同一数学题模板、同一化学反应模板同时出现在 train/val/test-like heldout。

## 2. Benchmark 对齐入口

| Benchmark | 公开 schema/评测要点 | 数据构造用途 | 泄漏边界 |
| --- | --- | --- | --- |
| RISEBench | `index/category/instruction/image/reference`；Temporal/Causal/Spatial/Logical；评测 Instruction Reasoning、Appearance Consistency、Visual Plausibility | 抽象四类 reasoning taxonomy、target-description 风格、atomic checklist | 不使用 benchmark image、instruction、reference；不做同义改写训练 |
| GRADE | `image_path/gt/text/task_id/questions/domain/sub_task/consistency`；10 学科域；rubric questions 带分值 | 抽象学科知识域和 question-based rubric；训练 checklist/verifier 格式 | 不使用 before/gt/text/questions 原样训练；rubric 只抽象成模板 |
| KRIS-Bench | 约 1,267 instances；3 类知识、7 个 reasoning dimensions、22 个任务族；评测 Visual Consistency、Visual Quality、Instruction Following、Knowledge Plausibility | 扩展到 factual/conceptual/procedural knowledge 和更细任务族；做泛化 heldout | 不使用 annotation/image/instruction 原样训练 |

## 3. 统一任务 schema

所有 recipe、task、trajectory、render 都应能追溯到同一个 `task_id`。建议先用 JSONL，后续再转 HuggingFace dataset。

```json
{
  "task_id": "rv2_rise_temporal_decay_000123",
  "source": "synthetic_recipe_v1",
  "split": "sft_train",
  "benchmark_family": "RISE_like",
  "task_family": "temporal_reasoning",
  "sub_family": "aging_decay",
  "knowledge_type": "conceptual",
  "domain": "everyday_physics",
  "sub_task": "future_state_prediction",
  "source_image": "data/images/source/rv2_rise_temporal_decay_000123.png",
  "source_image_provenance": {
    "type": "generated",
    "generator": "image_teacher",
    "prompt_hash": "sha256:..."
  },
  "instruction": "Edit the image to show what the fruit would look like after being left on a kitchen counter for ten days.",
  "expected_target": "The banana should become heavily spotted and partly blackened, with a softer sagging peel, while the plate and table remain unchanged.",
  "rational_target_description": "A ripe banana left at room temperature for ten days typically develops many brown to black spots and a softer peel. The edit should alter only the banana, not the plate or table.",
  "required_knowledge": [
    {
      "claim": "Banana peel darkens and becomes mottled as it ripens and decays at room temperature.",
      "source": "teacher_knowledge_or_search",
      "confidence": 0.87
    }
  ],
  "edit_operations": [
    {
      "op": "appearance_change",
      "target": "banana peel",
      "region_hint": "central yellow banana",
      "change": "add dense brown and black ripening spots; slightly soften contour"
    }
  ],
  "preservation_constraints": [
    "Keep the plate shape, table texture, camera viewpoint, lighting, and background unchanged."
  ],
  "negative_constraints": [
    "Do not add mold unless explicitly requested.",
    "Do not change the banana into a different fruit."
  ],
  "atomic_checklist": [
    {
      "id": "C1",
      "question": "Does the banana show clear brown or black ripening/decay marks?",
      "weight": 0.35
    },
    {
      "id": "C2",
      "question": "Is the plate and table background preserved?",
      "weight": 0.25
    },
    {
      "id": "C3",
      "question": "Is the result visually plausible under the original lighting and perspective?",
      "weight": 0.25
    },
    {
      "id": "C4",
      "question": "Does the image avoid adding unrelated objects or text?",
      "weight": 0.15
    }
  ],
  "difficulty": {
    "level": 2,
    "reason": "Single object temporal state prediction with clear preservation constraints."
  },
  "leakage_tags": {
    "benchmark_text_max_sim": 0.41,
    "benchmark_image_max_sim": 0.18,
    "status": "passed"
  },
  "license": "internal_generated"
}
```

## 4. Stage 0: 冻结 benchmark 快照和指纹

先把 RISE/GRADE/KRIS 的公开版本固定下来，记录 commit、dataset revision、下载时间、license、evaluator 版本。产物只用于评测和 decontamination。

建议产物：

```text
data/benchmarks/rise/raw/
data/benchmarks/grade/raw/
data/benchmarks/kris/raw/
data/benchmarks/*/fingerprints/
data/benchmarks/benchmark_fingerprint.json
```

`benchmark_fingerprint.json` 至少包含：

```json
{
  "benchmark": "RISEBench",
  "version": "github_commit_or_hf_revision",
  "samples": 360,
  "text_hashes": ["sha256:..."],
  "image_phash": ["..."],
  "clip_image_embeddings": "data/benchmarks/rise/fingerprints/clip_image.npy",
  "dino_image_embeddings": "data/benchmarks/rise/fingerprints/dino_image.npy",
  "text_embeddings": "data/benchmarks/rise/fingerprints/text.npy",
  "license": "recorded_from_source",
  "allowed_use": "evaluation_and_decontamination_only"
}
```

去污染规则：

| 类型 | 自动拒绝 | 人工复核 |
| --- | --- | --- |
| Text exact | 归一化 instruction/reference/question 完全匹配 | 无 |
| Text semantic | 与 benchmark instruction/reference embedding cosine `> 0.88` | `0.82-0.88` |
| Image pHash | Hamming distance `<= 6` | `7-12` |
| Image embedding | DINO/CLIP cosine `> 0.92` | `0.86-0.92` |
| Entity/template | 同一著名实体、同一公式题、同一图表结构与 benchmark 高度重合 | 需要人工决定是否换实体或换模板 |

阈值初始偏保守，pilot 后根据误杀率调整。任何被拒绝的数据保留 `reject_reason`，但不进入训练。

## 5. Stage 1: Benchmark-derived taxonomy bank

这一步只挖 taxonomy、rubric 风格和难度轴，不复制 benchmark 样本。

### 5.1 RISE-like taxonomy

| RISE category | 子族 | 典型变化 | 适合数据来源 |
| --- | --- | --- | --- |
| temporal_reasoning | aging_decay, growth, seasonal_change, charging_discharging, settling_drying | 未来状态、老化、腐烂、生长、沉积、干湿变化 | synthetic source, OpenImages/COCO, generated scenes |
| causal_reasoning | force_deformation, thermal_phase_change, fluid_dynamics, chemical_effect, biological_response | 施力变形、加热融化、冷却凝固、液体流动、化学/生物反应结果 | synthetic physical scenes, generated diagrams, public photos |
| spatial_reasoning | viewpoint_change, 3d_rotation, occlusion, relative_position, object_rearrangement | 视角转换、遮挡关系、空间重排、镜像或投影 | rendered 3D/simple graphics, generated scenes |
| logical_reasoning | math_symbolic, board_game, maze_path, rule_based_pattern, counting_constraint | 数字/规则求解、棋盘、迷宫、图案逻辑 | programmatic rendering, SVG/LaTeX/Matplotlib |

### 5.2 GRADE-like taxonomy

GRADE 的价值不只是学科域，而是 `questions` 形式的 rubric。每个 discipline task 都要落到 2-5 个可判定 checklist。

| Domain | 任务例子 | 构造方式 |
| --- | --- | --- |
| math | 替换变量值、补全几何角度、函数图像变化 | 程序生成题目和图像，保留 symbolic answer |
| physics | 光路、力学方向、热学状态、电路变化 | 规则引擎或小型 simulator 生成 source/target |
| chemistry | 分子结构、反应现象、实验装置变化 | RDKit/模板图/生成图结合，知识库校验 |
| biology | 解剖结构、生命周期、细胞过程 | 稳定概念图和真实图结合，强调标签和结构位置 |
| computer science | 数据结构、流程图、代码执行结果 | Graphviz/程序渲染，答案可自动验证 |
| economics | 供需曲线、图表变化、政策冲击 | Matplotlib 生成图表，规则校验 |
| history | 建筑/文物/时代状态修复 | 需要 image_search 或知识检索，人工抽检比例更高 |
| geography | 地貌、地图、气候带、地理特征 | GIS/地图素材/生成图，注意真实地理知识 |
| music | 乐谱、乐器结构、节奏符号 | 程序渲染乐谱或稳定素材 |
| sports | 战术位置、规则状态、器材变化 | 规则模板和示意图优先 |

### 5.3 KRIS-like taxonomy

KRIS 用于补足 GRADE 没覆盖的智能编辑维度，尤其是 knowledge plausibility。

| KRIS 维度 | RISEvolve 子族 | 数据构造重点 |
| --- | --- | --- |
| factual knowledge | entity_attribute_edit, landmark_restore, species_trait | 外部知识 grounding；不能只靠视觉猜测 |
| conceptual knowledge | physical_rule, biological_process, social_convention | 先推理概念，再决定视觉变化 |
| procedural knowledge | multi_step_process, rule_execution, workflow_state | 需要显式步骤和中间状态检查 |
| multi-element composition | relation_preserving_multi_object | 多对象绑定和位置关系 |
| temporal prediction | future_state_prediction | 和 RISE temporal 合并但扩展实体 |
| viewpoint change | view_synthesis_planning | 与 spatial reasoning 合并 |
| anomaly correction | implausibility_detection_and_fix | 先发现不合理处，再编辑 |

taxonomy 产物：

```text
data/taxonomy/benchmark_taxonomy.yaml
data/taxonomy/recipe_templates.yaml
data/taxonomy/checklist_templates.yaml
data/taxonomy/difficulty_rubric.yaml
```

## 6. Stage 2: Recipe generation

Recipe 是抽象任务蓝图，不绑定具体 benchmark 样本。LLM 只看 taxonomy 和少量人工写的模板，不直接看 benchmark 原始样本。

### 6.0 Real-image source expansion

v1 的程序图适合可控 reasoning bootstrap，但不足以覆盖真实图像编辑的纹理、光照、背景和非编辑区保真。v2 起增加真实图像 seed pool，入口脚本为：

```bash
python3 scripts/data/collect_real_edit_sources.py \
  --version v2_seed \
  --hf-per-source 12 \
  --wiki-per-query 2
```

数据源分三类：

| 类别 | 例子 | 用途 | 训练边界 |
| --- | --- | --- | --- |
| Public edit pairs | MagicBrush, ImagenHub filtered, AnyEdit train, OmniEdit train | 直接提供 source/target/instruction，用于 SFT、reward、editor-pair bootstrap | 只用 train/filtered；跑 safety、license、benchmark decontam 和 VLM 质量过滤 |
| Photoshop-like before/after | MIT-Adobe FiveK, PPR10K | 训练全局/局部 retouching、preservation、visual quality reward | 只在许可允许时用；不把任意教程/博客图直接抓进训练 |
| Licensed source/reference images | Wikimedia Commons, OpenImages metadata | 生成 RISE/GRADE/KRIS-like real-source edit prompts，后续用强 editor 生成 target | 保留 URL、license、artist、hash；target 生成后再进质量 gate |

当前 seed 产物：

```text
data/sources/real_edit_source_catalog.json
reports/data_sources/real_edit_source_report.md
data/sources/real_edit_pairs_sample_v2_seed.jsonl
data/sources/wikimedia_source_pool_v2_seed.jsonl
data/tasks/real_seed_prompts_v2_seed.jsonl
data/real_edits/v2_seed/
```

已实现的随机 HF-only 扩展采样：

```bash
python3 scripts/data/collect_real_edit_sources.py \
  --version v2_hf150 \
  --hf-per-source 30 \
  --skip-wikimedia \
  --randomize \
  --seed 601
python3 scripts/data/audit_real_sources.py --version v2_hf150 --sheet-limit 30
```

`v2_hf150` 当前包含 141 条 safety-filtered real edit pairs；另有 9 条灾害/攻击/爆炸/火灾等文本安全命中的样本写入 `data/sources/real_edit_pairs_rejected_v2_hf150.jsonl`，不进入候选 manifest。这个样本池用于调通真实图像 SFT/reward 数据格式，仍不能直接视为最终训练集。

这些数据默认只是 **candidate pool**。进入正式训练前必须额外完成：

1. 与 RISE/GRADE/KRIS 的 text exact/semantic、image pHash/DINO/CLIP 去污染。
2. 许可分层：`cc-by/cc0/mit` 可优先；`cc-by-nc` 只用于 research；未知许可只做分析或 baseline。
3. VLM difference-first 质量过滤：目标变化完成、非编辑区保真、无额外对象、无安全风险。
4. 数据集 split 隔离：同一源图、同一 commons page、同一 HF row、同一实体不能跨 train/val/heldout。

Recipe schema：

```json
{
  "recipe_id": "recipe_temporal_decay_001",
  "benchmark_family": "RISE_like",
  "task_family": "temporal_reasoning",
  "sub_family": "aging_decay",
  "domain": "everyday_objects",
  "source_scene_spec": "A fresh cut apple slice on a white ceramic plate under indoor light.",
  "instruction_template": "Show what this would look like after {time_span} in {environment}.",
  "target_reasoning": "Apple flesh oxidizes and browns; edges dry slightly; plate remains unchanged.",
  "visual_change_spec": [
    "Add brown oxidation on exposed apple flesh.",
    "Slightly dry the cut edges."
  ],
  "preservation_spec": [
    "Keep plate, lighting, camera angle, background unchanged."
  ],
  "decoys": [
    "Do not turn the apple into a whole apple.",
    "Do not add unrelated insects or mold unless time_span is long enough."
  ],
  "required_knowledge": [
    "Oxidation of cut apple flesh causes browning."
  ],
  "checklist_seed": [
    "Correct future state",
    "Original identity preserved",
    "Background preserved",
    "No unrelated changes"
  ],
  "difficulty": 2
}
```

Recipe 生成策略：

1. 每个子族先人工写 5-10 个 seed recipe。
2. LLM 扩展为多实体、多环境、多难度版本。
3. 对每个 recipe 生成 3-5 个 paraphrased instructions，但同一 recipe 只允许落入同一个 split。
4. 生成 negative variants：故意缺少知识、区域不明确、包含冲突约束，用于 verifier 和 failure training。
5. 所有 recipe 先过 decontamination text check，再进入 source image acquisition。

## 7. Stage 3: Source image acquisition

source image 优先级：

1. **程序渲染**：math、physics diagram、chemistry diagram、CS graph、economics chart、music score、board/maze/rule tasks。优点是答案和 checklist 可自动验证。
2. **自生成 source image**：temporal、causal、spatial、多对象、常识场景。要求保存 prompt、seed、模型、生成时间和 license。
3. **公开图像数据集**：MagicBrush、OpenImages、COCO、Visual Genome、可商用或研究许可图像。只作为源图，不直接继承原编辑指令。
4. **检索图片**：历史、地理、真实实体类任务可用 image_search 辅助，但必须保存 URL/license，并优先只作为 reference，不直接放训练源图。

建议目录：

```text
data/images/source/
data/images/reference/
data/images/generated_source/
data/images/programmatic/
data/images/provenance/source_metadata.jsonl
```

source image 过滤：

| Gate | 检查内容 |
| --- | --- |
| Visual clarity | 主体清楚、分辨率足够、无大面积水印或文字污染 |
| Editability | 目标区域可见，编辑不会要求不可观察的背面或超大范围重绘 |
| Safety/license | license 可记录；无隐私敏感或不适合训练内容 |
| Benchmark decontamination | image pHash、CLIP/DINO embedding 不接近 benchmark 图像 |
| Domain suitability | 图像确实支持 recipe 的视觉变化和知识推理 |

## 8. Stage 4: Task materialization

把 recipe 和 source image 绑定成正式 task。

自动步骤：

1. VLM 描述 source image，生成 `source_scene_graph`、主体区域、可编辑区域、背景保留区域。
2. LLM 根据 recipe、source scene 和 knowledge 生成最终 `instruction`。
3. LLM 生成 `rational_target_description`，明确为什么要这么改。
4. 生成 `edit_operations`，每个 op 绑定 target、region_hint、change、preserve。
5. 生成 `atomic_checklist`，权重和必须项分开。
6. 跑 text/image decontamination。

task materialization 需要拒绝：

- source 图像和 recipe 不匹配，例如 recipe 要编辑冰块融化，但图中没有冰块。
- 目标不可见或区域无法定位。
- instruction 过于接近 benchmark 原句。
- expected target 不可验证，或者 checklist 是泛泛的“看起来正确”。
- preservation constraints 和 edit operations 冲突。

## 9. Stage 5: Teacher trajectory generation

Teacher trajectory 是 SFT 的核心，不只是最终答案。要求统一 ReAct 风格。

可用工具：

| Tool | 何时使用 | 产物 |
| --- | --- | --- |
| `analyze_image` | 所有样本默认使用 | source scene graph、目标区域、保留区域 |
| `search` | factual/discipline/temporal/causal 知识不确定时 | knowledge facts、引用片段、置信度 |
| `image_search` | landmark/history/geography/style/reference-sensitive task | reference candidates、reference roles |
| `query_edit_knowledge` | 按 task_family 调取技能库 | checklist 模板、常见失败、区域规划建议 |
| `solve_symbolic` | math/logic/CS/economics/chart/rule tasks | symbolic answer、步骤、可验证结果 |
| `ground_region` | 多对象或局部精细编辑 | bbox/mask/region text |

Trajectory 输出格式：

```json
{
  "task_id": "rv2_rise_temporal_decay_000123",
  "messages": [
    {
      "role": "user",
      "content": [{"type": "image", "path": "..."}, {"type": "text", "text": "..."}]
    },
    {
      "role": "assistant",
      "thought": "I need to identify the editable object and verify the temporal change.",
      "tool_call": {"name": "analyze_image", "arguments": {"image": "..."}}
    },
    {
      "role": "tool",
      "name": "analyze_image",
      "content": {"objects": ["banana", "plate"], "editable_region": "banana"}
    }
  ],
  "tool_evidence_map": [
    {
      "claim": "banana peel darkens after ripening",
      "evidence_tool": "query_edit_knowledge",
      "used_in": ["rational_target_description", "atomic_checklist.C1"]
    }
  ],
  "final_edit_program": {
    "target_scene_description": "...",
    "edit_operations": [],
    "preservation_constraints": [],
    "atomic_checklist": [],
    "editor_prompt": "..."
  }
}
```

Teacher 质量要求：

1. 每个 factual claim 必须能追溯到 source image observation、symbolic solver 或 knowledge/search evidence。
2. 工具调用不能堆砌。无必要搜索的样本应标记 `search_not_needed_reason`。
3. final edit program 必须可被 editor 执行，不能只有抽象推理。
4. checklist 必须和 instruction、expected target、preservation constraints 一一对应。
5. 对 hard task 允许多轮自检和修订，但最终保留 clean trajectory 和 revision diff。

## 10. Stage 6: Rendering and rollout generation

每个 task 建议保存三类图：

| Render 类型 | 用途 | 生成方式 |
| --- | --- | --- |
| `teacher_render` | SFT 质量检查、verifier 正例 | 强 editor 或人工辅助编辑 |
| `student_rollout_render` | RL reward、preference pairs | 当前 agent + open/editor 多采样 |
| `negative_render` | verifier 负例和 VED bad case | 删除关键约束、错误知识、错误区域、过度编辑 |

Editor 选择：

- teacher render 优先使用当前最强可用模型，例如 GPT-Image-1.5、Gemini/Nano Banana Pro、或人工编辑辅助。
- student rollout 使用目标部署 editor 或开源 editor，例如 Qwen-Image-Edit、FLUX.1 Kontext、SeedEdit、Step1X-Edit。
- 对程序渲染类任务，如果 target 可直接渲染，应保存 programmatic ground truth，避免 editor 噪声污染 verifier。

保存 metadata：

```json
{
  "render_id": "rv2_rise_temporal_decay_000123_teacher_0",
  "task_id": "rv2_rise_temporal_decay_000123",
  "render_type": "teacher_render",
  "editor": "gpt_image_1_5",
  "edit_program_hash": "sha256:...",
  "image_path": "data/renders/teacher/rv2_rise_temporal_decay_000123.png",
  "sampling_params": {"n": 1},
  "created_at": "2026-05-31T00:00:00Z"
}
```

## 11. Stage 7: Filtering and scoring

过滤不是一个总分，而是一组 gate。建议每条数据保留所有中间分数。

| Gate | 输入 | 通过条件 | 失败去向 |
| --- | --- | --- | --- |
| Schema gate | task/trajectory/program JSON | 必填字段存在、权重和合法、路径存在 | reject |
| Decontamination gate | task text/source image | 低于阈值 | reject or review |
| Source-image gate | source image + VLM analysis | 目标存在、区域可见、无 license 问题 | reject |
| Evidence gate | trajectory | claim 有 observation/search/solver 支持 | reject trajectory |
| Program gate | edit program | operation 可执行、region/preserve/checklist 一致 | revise or reject |
| Render gate | teacher render | checklist pass、视觉质量达标、保留区域稳定 | accept/revise |
| Editor-gap gate | program + render | program 正确但 editor 失败时标记 | verifier/editor-gap only |
| Human audit | 抽样或高风险样本 | 通过人工 spot check | accept or blacklist template |

建议分数：

```json
{
  "task_id": "...",
  "scores": {
    "schema": 1.0,
    "decontamination": 0.96,
    "source_alignment": 0.91,
    "evidence_grounding": 0.88,
    "program_executability": 0.86,
    "checklist_quality": 0.90,
    "teacher_render_pass": 0.82,
    "preservation": 0.79,
    "visual_quality": 0.84
  },
  "labels": {
    "accept_sft": true,
    "accept_rl": true,
    "accept_verifier": true,
    "editor_gap": false,
    "reject_reason": null
  }
}
```

SFT 接收建议：

- `evidence_grounding >= 0.75`
- `program_executability >= 0.80`
- `checklist_quality >= 0.80`
- `teacher_render_pass >= 0.75`
- 没有 decontamination 红线

RL prompt 接收建议：

- task 本身清楚，但 teacher_render 不一定完美。
- 难度主要集中在 level 2-4，避免过易样本让 reward 无区分度。
- 每个 task 可以采 4-8 个 rollout，构造 GRPO group。

Verifier 接收建议：

- 正负例都要有，并覆盖 instruction following、knowledge plausibility、visual preservation、region correctness。
- 负例不能只是低质量图，还要包含“看起来好但知识错”“目标对但背景被改”“区域错”等 hard negatives。

## 12. Stage 8: Split strategy

建议 split：

| Split | 用途 | 规模 v1 | 隔离原则 |
| --- | --- | --- | --- |
| `sft_train` | SFT trajectories | 5,000-6,000 | recipe/entity/source image 不与 val/heldout 重叠 |
| `sft_val` | SFT early stopping/manual inspect | 300-500 | task_family 均衡 |
| `rl_prompt_train` | GRPO/rollout | 1,000-1,500 prompts | 不含 teacher render 直接监督 |
| `verifier_train` | checklist/reward training | 2,000-5,000 pairs/items | 正负例按 failure type 均衡 |
| `ved_memory_train` | best-worst experience distillation | 500-1,000 pairs | 同 prompt 多 trajectory 对比 |
| `hard_heldout` | 内部泛化测试 | 300-500 | 新 entity、新模板、新 source |
| `benchmark_eval` | RISE/GRADE/KRIS 官方评测 | official only | 永不训练 |

Split 锁定粒度：

1. `recipe_id` 不能跨 split。
2. 同一 source image 或其 augmentation 不能跨 split。
3. 同一 programmatic generator seed family 不能跨 split。
4. 同一著名实体或 landmark 不能跨 split，尤其 history/geography/KRIS factual。
5. 同一数学/物理/化学模板参数变体不能跨 split。

## 13. Dataset artifacts

推荐最终产物：

```text
data/recipes/recipes_v1.jsonl
data/tasks/tasks_v1.jsonl
data/trajectories/teacher_trajectories_v1.jsonl
data/programs/edit_programs_v1.jsonl
data/renders/render_metadata_v1.jsonl
data/preferences/preference_pairs_v1.jsonl
data/experience/experience_pairs_v1.jsonl
data/verifier/verifier_items_v1.jsonl
data/splits/sft_train.jsonl
data/splits/sft_val.jsonl
data/splits/rl_prompt_train.jsonl
data/splits/verifier_train.jsonl
data/splits/ved_memory_train.jsonl
data/splits/hard_heldout.jsonl
reports/data_quality/summary_v1.md
reports/data_quality/leakage_report_v1.json
reports/data_quality/distribution_v1.json
```

每个 artifact 都要包含 `task_id`、`version`、`created_by`、`created_at`、`upstream_hash`，方便复现和回滚。

## 14. 规模规划

### 14.1 Pilot v0

目标是验证 pipeline 和 agent 格式，不追求大规模。

| 阶段 | 数量 |
| --- | --- |
| recipe candidates | 1,200 |
| source images | 800 |
| materialized tasks | 600 |
| teacher trajectories | 500 |
| accepted SFT | 300 |
| RL prompts | 100 |
| verifier items | 300-500 |
| human audit | 100 tasks |

分布：

- RISE-like 40%：temporal 30%、causal 30%、spatial 25%、logical 15%。
- GRADE-like 35%：10 个 domain 尽量均衡，math/physics/chemistry/biology 可略高。
- KRIS-like 25%：factual/conceptual/procedural 均衡，覆盖 anomaly/multi-element/temporal/viewpoint。

### 14.2 Full v1

| 阶段 | 数量 |
| --- | --- |
| recipe candidates | 12,000 |
| source images | 8,000-10,000 |
| materialized tasks | 8,000-9,000 |
| accepted teacher trajectories | 5,000-6,000 |
| RL prompts | 1,000-1,500 |
| verifier items | 5,000-10,000 |
| VED experience pairs | 500-1,000 |
| hard heldout | 300-500 |

扩展原则：

1. 先补 RISE/GRADE/KRIS 中 baseline 失败最多的族。
2. logical/math/diagram 类优先程序渲染，降低 teacher/editor gap。
3. history/geography/entity 类提高 human audit 比例。
4. 每周重算 distribution 和 leakage report，不让数据自然偏向容易编辑的 single-object appearance change。

## 15. 脚本和目录规划

建议新增：

```text
scripts/data/freeze_benchmarks.py
scripts/data/mine_taxonomy.py
scripts/data/generate_recipes.py
scripts/data/acquire_source_images.py
scripts/data/materialize_tasks.py
scripts/data/run_teacher.py
scripts/data/render_edits.py
scripts/data/filter_data.py
scripts/data/build_splits.py
scripts/data/build_verifier_data.py
scripts/data/build_experience_pairs.py
scripts/data/report_data_quality.py
```

目录结构：

```text
data/
  benchmarks/
    rise/
    grade/
    kris/
  taxonomy/
  recipes/
  tasks/
  images/
    source/
    reference/
    generated_source/
    programmatic/
    provenance/
  trajectories/
  programs/
  renders/
    teacher/
    rollout/
    negative/
  preferences/
  experience/
  verifier/
  splits/
reports/
  data_quality/
```

脚本依赖顺序：

```text
freeze_benchmarks
  -> mine_taxonomy
  -> generate_recipes
  -> acquire_source_images
  -> materialize_tasks
  -> run_teacher
  -> render_edits
  -> filter_data
  -> build_splits
  -> build_verifier_data / build_experience_pairs
  -> report_data_quality
```

## 16. Quality dashboard

每次构造数据后输出一份 `reports/data_quality/summary_*.md`。

必须报告：

1. 按 `benchmark_family/task_family/sub_family/domain/knowledge_type/difficulty` 的分布。
2. RISE/GRADE/KRIS taxonomy 覆盖率。
3. source image 来源、license、分辨率、去重统计。
4. decontamination top-k 相似样本和拒绝比例。
5. teacher tool use 统计：平均工具数、search 使用率、solver 使用率、重复工具率。
6. checklist 统计：平均检查项数、权重分布、hard negative 覆盖。
7. filter gate pass rate 和主要 reject reasons。
8. human audit pass rate。
9. accepted SFT/RL/verifier/VED 数量。
10. 与 official RISE/GRADE/KRIS eval 的隔离状态。

## 17. Pilot v0 十天执行表

| Day | 任务 | 产出 |
| --- | --- | --- |
| 1 | 冻结 RISE/GRADE/KRIS 快照，生成 fingerprint | `benchmark_fingerprint.json`、license/evaluator 记录 |
| 2 | 写 taxonomy seed 和 checklist templates | `benchmark_taxonomy.yaml`、`checklist_templates.yaml` |
| 3 | 生成 1,200 recipe candidates，跑 text decontamination | `recipes_v0_candidates.jsonl` |
| 4 | 准备 800 张 source images，程序渲染优先 | `source_metadata.jsonl` |
| 5 | materialize 600 tasks，生成 checklist | `tasks_v0.jsonl` |
| 6-7 | 生成 500 teacher trajectories | `teacher_trajectories_v0.jsonl` |
| 8 | 渲染 teacher/negative samples，跑 filter gates | `filter_scores_v0.jsonl` |
| 9 | 人工抽检 100 条，修正高频失败模板 | `human_audit_v0.csv`、blacklist |
| 10 | 构建 300 SFT、100 RL、300-500 verifier 数据并跑 prompted baseline | `data/splits/*_v0.jsonl`、`summary_v0.md` |

## 18. 首批验收标准

Pilot v0 只有在满足以下条件后再扩到 full v1：

1. decontamination 自动报告无红线样本。
2. accepted SFT 至少 300 条，人工抽检通过率 `>= 80%`。
3. teacher trajectory 中工具调用和最终 edit program 一致，随机抽查 50 条错误率 `< 15%`。
4. checklist 能区分至少 3 类 failure：知识错、区域错、保留失败。
5. prompted RISEvolve baseline 在内部 hard heldout 上优于 direct-editor prompt，至少在 checklist pass rate 上有可见增益。
6. 失败分析能明确下一轮扩数据方向，例如 RISE logical、GRADE chemistry、KRIS factual entity。
