# 统一模型与视觉 Agent 的数据配方调研(2026-06-11)

目的:为 UniEvolve 三任务(理解/生成/编辑)agentic 语料设计提供实证依据。[V]=直接读 PDF 核实,[H]=arXiv HTML 二手提取,引用前建议复核 PDF。

## 1. 关键事实速览

| 模型 | 阶段顺序 | 关键配比(und:gen:edit) | 规模 | 遗忘/回归与缓解 | judge 数据? |
|---|---|---|---|---|---|
| BAGEL [H] | 全联合 4 阶段 | PT 60% 生成对;SFT interleave 20/20/20+30% T2I;und 仅 5-10% | ~5.2T tok | gen-heavy(4g1u)不伤理解;CE:MSE=0.25:1;**编辑 2.64T tok 才涌现,推理编辑 3.61T** | 无 |
| InternVL-U [H] | **先生成→编辑→最后并入理解**(MLLM 先冻结) | SFT und:gen:edit = **1:1:2** | 未披露 | MMMU **−4.3** 回归;靠 S1-2 冻结 MLLM 缓解 | 无 |
| UniGen-1.5 [H] | PT→SFT→edit-align→RL | SFT gen:und:text = **3:4:1(理解过配)** | align 17.7K 三元组;RL 17.4K | **SFT/RL 后理解零回归**;**500 步 17.7K 的 edit-instruction-alignment(预测编辑结果的文字描述)使 RL 增益放大 ~6×(+0.06→+0.38 ImgEdit)** | 仅外部 RM |
| DeepGen 1.0 [V] | align→联合 SFT→RL | 无理解数据,VLM 上 LoRA 保理解 | ~50M | **RL 奖励未覆盖推理任务 → RISE 13.3→10.8 静默回归**;KL 不够,需每步辅助 SFT loss(λ=1e-4)锚定 | 无 |
| Emu3.5 [H] | PT 10T→3T→SFT 150B tok→RL | SFT 含 8.9M 文本 + 3.7M VQA | 13T tok | **从未评理解回归**(隐藏失败模式) | 仅过滤器 |
| Show-o2 [H] | **flow-head 先训(LLM 冻)→全解冻** | S2 und:gen ≈ 9M:16M | 66M 对 | 单阶段联合训退化语言;两阶段免回放 | 无 |
| Qwen-Image-2.0 [V] | 6 阶段分辨率爬坡→SFT→RLHF | gen:edit 9:1→**7:3**;理解全靠冻结 Qwen3-VL 编码器 | 未披露 | 数据飞轮:bad-case 归因路由到 RL/预训练/提示工程三轨 | 5 个任务专用 RM(外部) |
| AlphaGRPO [H] | **RL-only 无冷启动**(BAGEL 上) | 19.5K prompts(39 任务×500,难度 3:5:2) | — | 反思的假阳性矫正:不提升的 refinement 给组内最低奖励 | 分解式验证器(10 语义+8 质量原子问题,r=√(v̄s·v̄q)) |
| GenEvolve [H] | teacher 蒸馏 SFT→GRPO+经验蒸馏 | 仅生成轨迹 | 19,990 prompts→**69.2% 接受率**→8.8K SFT;GT 图过滤存活 73.5%;RL 2,575 | — | 轨迹内(经验槽) |
| GenAgent [H] | SFT 32K→GRPO | 仅生成;**轨迹内 judgment(J)token 被训练** | 32K | 无理解回放 | **有:去掉 pairwise 改进判定奖励 −0.018 GenEval++** |
| IEA [H] | SFT 29K→RL→合成 SFT 202K | edit 88% + **summary(批评式)12%** | 260K | S3 含 **35% 回放** | **有:Image-Summary 任务** |

**Q:judge/批评数据放进理解混合有用吗?** 统一模型一律不放(理解=VQA/caption/OCR/grounding);**agent 论文一律以轨迹内形式放且有效**(GenAgent J tokens、IEA Summary、GenEvolve 经验槽)。→ 我们 v2u 的 diff/judge 任务走的正是被验证的形式。

## 2. 对 UniEvolve 语料的 5 条决定(采纳)

1. **预算 ~22K SFT + ~4K RL + ~4K 回放;编辑过配**:SFT 按 und 5K : gen 8K : edit 9K;理解切片必须保留(UniGen-1.5 零回归 vs InternVL-U −4.3 MMMU 是最干净的天然对照)。
2. **真实工具回路采集 teacher 轨迹,双层过滤,按 2× 目标量采集原始轨迹**(GenEvolve:69.2% 轨迹接受 × 73.5% 图像过滤存活);难题挖掘用 GenAgent 的"3 次生成全失败才保留"。
3. **轨迹 schema 即刻冻结,只训 assistant tokens**:system 定义工具→`<think>`→tool call(JSON)→observation(mask)→judgment tokens→结构化 answer(ordinal 引用,不出 URL);保留部分不终止轨迹保探索;并行准备 AlphaGRPO 式单轮反思变体供 RL 稳定性。
4. **批评作为一等训练任务**:v2u_diff(IEA Summary 强化版)+ v2u_judge 进理解切片;RL 时用分解式可验证奖励 + "是否改进"pairwise 奖励 + 假阳性矫正(已并入 `rl_algorithm_spec.md` §4 思路)。
5. **RL 回归防护(已有实证失败案例)**:奖励覆盖必须横跨三任务(DeepGen 教训);后期 SFT 轮保留 30-35% 回放(IEA);RL 中加辅助 SFT loss 锚(DeepGen);**每个 checkpoint 三能力全评**(Emu3.5 反面教材);RL 前先做 UniGen-1.5 式 edit-instruction-alignment 文本预热——我们的任务自带 target_scene_description,可零成本构建该预热集。

## 3. 直接可复用资产

- GenEvolve-Data-Bench(已下元数据:8,800 SFT + 2,575 RL,Apache-2.0;图像待磁盘清理后下载)
- DeepGen 数据:`huggingface.co/datasets/DeepGenTeam/DeepGen-1.0`(含 reasoning-gen/edit 切片)
- UniGen-1.5 alignment 任务模板:(源图, 编辑指令)→ 编辑结果文字描述,17.7K 即够;我们 v1r/v2 字段可直接转

## 4. 待核事项

GenEvolve 接受数 13,379 vs 13,179 不一致(查 PDF);GenEvolve/GenAgent 的确切 system prompt 序列化在附录/repo;InternVL-U、Qwen-Image-2.0 绝对数据量未披露;Emu3.5 理解回归无数据。
