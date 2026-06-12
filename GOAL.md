# RISEvolve / UniEvolve 目标(2026-06-11 设定)

## 总目标

训练自进化的 agentic 图像模型,两阶段推进:

- **Phase 1 — RISEvolve(编辑,当前主线)**:Qwen3-VL-8B planner + 固定 editor,SFT→RFT→Edit-GRPO(+Edit-OPD),在 RISEBench / GRADE / KRIS 上以 8B 规模超过 DDA-Thinker-8B(RISE > 31.9,KRIS > 76.99),GRADE 发表首个 planner 数字。算法规范:`rl_algorithm_spec.md`;评测矩阵:`eval_benchmark_matrix.md`。
- **Phase 2 — UniEvolve(统一,已立项调研)**:同一个 policy 统一理解+生成+编辑的工具编排:分析→路由(生成|编辑|分解)→程序合成→执行→验证→修正,多任务 GRPO + 跨任务经验蒸馏。可行性评估与三大风险:`unified_agentic_model_assessment.md`。先导实验:GenEvolve 9k 生成轨迹混入 SFT 测跨任务迁移。

## 数据目标(三任务,持续优化)

| 任务 | 来源 | 状态/目标 |
|---|---|---|
| 编辑 | v1r(GPT-5.5 去 boilerplate 重写)+ v2(16 族程序可验证)+ 公开推理编辑数据切片 | v1r/v2 流水线运行中;目标 1 万级高质量、任务族平衡、全部过裁判+去污染 |
| 生成 | GenEvolve-Data-Bench(SFT 9k 轨迹 + RL 3,175 prompts,公开)+ 后续自建知识型生成任务 | 引入并适配 schema |
| 理解 | v2 verifier VQA 任务 + difference-first 评审任务(source/teacher/negative 三元组天然产出)+ 失败归因任务 | 构建 v2u 理解数据集 |

质量铁律:answer-first / 程序可验证构造;独立裁判过滤;任务族配额;接口形式随机化;文本+图像双通道去污染;benchmark 内容零入训。

## 指挥结构

- **Fable 5(Claude Code)= 主指挥**:规划、质量门、验证、整合,管理任务清单与文档。
- **codex CLI(免 key gpt-5.5,4142 代理)= 主力执行**:代码生成(生成器/管线)、批量数据标注(/v1/responses 直连)、视觉质检(vision 输入)。注:azure_uami 链路(9876)因资源关闭公网访问不可用,统一走 4142。
- **claude 子 agent = 调研与并行任务**:文献调研、benchmark 核实、跨目录探索。
- 持续机制:每轮数据迭代沉淀可复用脚本(scripts/data/ 即 skill 库);模型级统一模型调研持续滚动,反哺 agent 级设计。

## 当前里程碑清单

1. [进行中] v1r 编辑数据流水线(refine→裁判→merge→去污染)
2. [进行中] v2 可验证编辑数据流水线(2,400 任务标注中)
3. [ ] v2u 理解数据集(VQA + difference-first + 失败归因)
4. [ ] GenEvolve-Data-Bench 引入与 schema 适配
5. [ ] GPU 空闲后:P0 评测地基(post-fix 全量基线 + failure taxonomy)→ SFT v4(过门才进 RL)
6. [ ] 跨任务迁移先导实验(gen 轨迹混入 → 编辑 dev)
