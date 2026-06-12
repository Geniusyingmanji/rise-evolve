# 生成-编辑-理解统一的 Agentic 图像模型:可行性评估(2026-06-11)

基于两路深度调研:模型级统一(unified backbone)与 agent 级统一(unified agent policy)。结论先行:**可行,且存在一个明确无人占领的交集;推荐 agent 级路线,作为 RISEvolve 编辑里程碑达成后的下一篇工作。**

## 1. 两条路线的证据现状

### 1.1 模型级统一(一个网络同时做理解+生成+编辑)

- **学术规模已被证明可行**:InternVL-U 4B 在生成/编辑上超 BAGEL 14B;DeepGen 1.0 仅 5B/50M 样本,WISE 0.73 开源最佳、UniREditBench 77.5;EMMA 4B 超 BAGEL-7B。胜出配方收敛:**MLLM + diffusion/MMDiT 头 + 解耦表征 + 三阶段训练收尾 GRPO**。
- **协同证据(理解→生成/编辑,可靠复现)**:BAGEL Self-CoT 使 WISE 0.52→0.70、KRIS 56.2→60.2;AlphaGRPO(ICML'26)零编辑训练使 GEdit 6.52→7.08(理解对齐零样本迁移到编辑);自反思推理时增益 +3-4.5%。
- **干扰证据(重要反面)**:ROVER 系统研究显示视觉推理在物理/感知任务上加分、在**符号/抽象任务上减分**;UniCorn 提出 "Conduction Aphasia"——模型能准确批评一张图的错误却生成不出正确的同一场景,理解与生成**不会自动耦合**。
- 对我们的含义:模型级统一需要的训练量(BAGEL 数 T tokens;追平 Qwen-Image-2.0/Seedream 5.0 的生产线)不是学术预算能做的;且其"何时该推理"的问题(ROVER)恰恰是 agent 层的决策问题。

### 1.2 Agent 级统一(一个 VLM policy 编排生成/编辑/理解工具)

调研确认**交集为空**:

| 已存在 | 缺失 |
|---|---|
| 单任务训练的 agent policy:生成(GenEvolve、GenAgent +23.6% GenEval++)、编辑(MIRA、IEA、ImageEdit-R1、JarvisArt)、工作流(ComfyUI-R1) | **跨生成+编辑+理解的单一训练 policy:不存在** |
| 跨生成+编辑的系统:全部 prompted(T2I-Copilot、ComfyMind、Agent Banana、ImAgent)或只训小 DQN 选择器(Image-POSER) | **学习型路由**(generate / edit / decompose / 何时不分解)带结果奖励训练:不存在 |
| 经验机制:GenEvolve 5-slot VED(仅生成域)、EvolveR/AgentEvolver(文本域) | **跨任务经验蒸馏**(编辑轨迹中的保真策略迁移到生成规划等):不存在 |
| 共享权重的自验证 RL:ADPO/V₁(纯文本)、RL-RIG(actor/checker 分离) | **视觉 agent 的 create+verify 同权重共训**:不存在 |
| 文本域多任务 agentic RL:协同大于干扰(To-Mix-or-To-Merge),需 task-advantage 归一化(AgentRL)与课程(Omni-Thinker) | **视觉创作域的跨任务迁移矩阵**(编辑 RL 是否提升生成规划?):无人测过 |

## 2. 推荐定位(暂名 UniEvolve,RISEvolve 的自然延伸)

> 第一个自进化、RL 训练的统一视觉 agent policy:同一个 Qwen3-VL-8B 在分析→路由(生成|编辑|分解)→程序合成→执行→验证→修正的完整循环中行动,以多任务 GRPO + 跨任务经验蒸馏训练,并给出 agent 级跨任务迁移矩阵(UniGen-1.5/Janus-Pro-R1 模型级协同结果的 agent 级对应物)。

关键资产盘点——三类任务的数据/组件我们**已经有雏形**:

| 任务类型 | 数据来源 | 状态 |
|---|---|---|
| 编辑 | v1r(~9.4k 去 boilerplate)+ v2(2,400 可验证)+ 公开推理编辑数据 | 本轮已建成 |
| 生成 | **GenEvolve-Data-Bench 公开**:SFT 9,000 轨迹 + RL 3,175 prompts + GT 图 | 直接可用,工具协议同源 |
| 理解/验证 | v2 verifier_spec 的 VQA 任务本身就是理解任务;RISE-Critic 的 difference-first 评审任务 | 本轮已建成 |
| RL 算法 | `rl_algorithm_spec.md` 全部组件(门控奖励/可验证通道/前缀树/OPD)任务无关,加 task-advantage 归一化即成多任务版 | 已设计 |

新增的核心科学问题(也是论文卖点):
1. **跨任务迁移矩阵**:train{edit} → eval{gen planning},train{gen} → eval{edit planning},train{verify} → eval{both}。
2. **学习型路由**:输入一个请求,policy 决定从空白生成还是基于源图编辑还是分解为多步——Image-POSER 证明该决策可学但只用了 DQN;ImageEdit-R1 证明"何时不分解"非平凡(多轮会复合误差)。
3. **共享权重自验证**:verify 轮与 create 轮同权重共训(ADPO 的梯度隔离配方迁移到视觉),让验证精度本身成为带奖励的训练技能。

## 3. 三大风险与对策

| 风险 | 证据 | 对策 |
|---|---|---|
| 多任务 RL 奖励异质性(生成 judge 分 vs 编辑半可验证 vs 理解全可验证) | Omni-Thinker:朴素混合失败;Imbalanced Gradients:梯度失衡是根因 | AgentRL task-advantage 归一化 + 课程(verifiable 先行,与我们 RL 课程同构);二值 checklist(ThinkRL-Edit) |
| 自验证塌缩/奖励 hack(同权重下"宽容自评") | ADPO 文档化该现象,需梯度隔离;UniRL-Zero 需显式纠错训练 | ADPO 式解耦 advantage + token mask;冻结外部 judge 作锚(FIRM-Edit-8B) |
| 成本与拥挤(渲染型 rollout 贵 1-2 个数量级;GenAgent/ImAgent/Unify-Agent 团队距此一步之遥) | Image-POSER 因此只训 DQN | 我们的前缀树+缓存回放组件正为此设计;先发优势靠完整评测网格(MME-Unify + WISE + RISE/GRADE/KRIS + MMMU)定义比较协议 |

## 4. 执行建议(不打乱当前主线)

1. **现在不转向**:先完成 RISEvolve 编辑里程碑(8B 超 DDA-Thinker-8B:RISE>31.9/KRIS>76.99)——这是统一版的编辑支柱和可信度基础。
2. **零成本先导实验**(GPU 空闲后随 SFT 一起做):把 GenEvolve 公开 9k 生成轨迹混入我们 SFT,测编辑 dev 是否提升(跨任务协同的最便宜检验);反向用我们的编辑数据测 WISE 子集。结果直接决定统一方向的优先级。
3. **数据侧已就绪**:v2 生成器天然产出"理解任务"(VQA 检查),给 verify-轮训练数据零额外成本。
4. 若先导实验显示正迁移 → 立项 UniEvolve 作为第二篇;若负迁移 → 同样是可发表发现(agent 级 ROVER),且印证当前单任务路线正确。

## 5. 关键文献索引

模型级:InternVL-U 2603.09877 · BAGEL 2505.14683 · DeepGen 2602.12205 · EMMA 2512.04810 · AlphaGRPO 2605.12495 · UniGen-1.5 2511.14760 · UniCorn 2601.03193 · ROVER 2511.01163 · Unified-GRPO 2509.09666 · UniRL 2505.23380
Agent 级:GenEvolve 2605.21605 · GenAgent 2601.18543 · Image-POSER 2511.11780 · ImageEdit-R1 2603.08059 · ComfyUI-R1 2506.09790 · JarvisArt 2506.17612 · ImAgent 2511.11483 · Unify-Agent 2603.29620 · AgentEvolver 2511.10395 · EvolveR 2510.16079
多任务 RL:AgentRL 2510.04206 · To-Mix-or-To-Merge 2602.12566 · Omni-Thinker 2507.14783 · Imbalanced Gradients 2510.19178 · ADPO 2601.01483 · V₁ 2603.04304
评测:MME-Unify 2504.03641 · UniEval 2505.10483 · WISE 2503.07265 · ROVER 2511.01163
