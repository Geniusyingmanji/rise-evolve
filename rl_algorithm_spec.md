# Edit-GRPO + Edit-OPD:图像编辑特化 RL 算法规范(v1,2026-06-11)

整合自 `training_plan.md`、`reward_design.md`、`improvement_plan_2026-06-11.md`,吸收 2026-06 文献调研结论,形成单一可实现规范。目标:训练 Qwen3-VL-8B planner(固定下游 editor),在 RISE/GRADE/KRIS 上超过 DDA-Thinker-8B(RISE 31.9 / KRIS 76.99)。

## 0. 为什么通用 GRPO 不适合图像编辑 agent

| 编辑任务特性 | 对 RL 的影响 | 本规范的应对组件 |
|---|---|---|
| 环境步昂贵:每个 rollout = diffusion 渲染 + VLM 评审(10-60s) | rollout 预算是第一约束 | §3 渲染高效组构造(前缀树 + 回放) |
| 奖励来自 VLM judge:方差大 + 系统性偏差 | advantage 噪声、错误学习信号 | §4 噪声鲁棒 advantage + §2 可验证奖励快速通道 |
| 失败三方归因:planner / editor / judge | 把 editor 失败惩罚到推理 token 会教模型"少计划" | §5 editor-gap-aware credit routing |
| 两类 shortcut:不改图骗 consistency;视觉合理但推理错 | 简单加权奖励必被 hack | §1 认知门控奖励 |
| 工具调用是可选动作 | 组内工具子组全错 → advantage 归零,恰好在最需要学的地方没信号 | §3.3 all-wrong 子组重采样 |

## 1. 认知门控奖励(novel:扩展 FIRM CME 到推理编辑)

FIRM 证明 `0.5*Exec + 0.5*Consistency` 诱导"几乎不编辑"。其 CME(`Exec*(0.6+0.4*Cons)`)只防 unchanged-shortcut。推理编辑还有第二类 shortcut:视觉执行了但学科/逻辑结果错。因此双重门:

```text
G_task   = min(R_exec, R_cog_applicable)        # 两个必要条件取 min
R_image  = G_task * (0.45 + 0.20*R_preserve + 0.15*R_region
                     + 0.10*R_quality + 0.10*R_readability)
R_agent  = 0.45*R_program + 0.45*R_image + 0.05*R_tool + 0.05*R_format
```

- `R_cog_applicable`:无需推理的任务恒为 1(退化为 FIRM CME)。
- `R_program` 不看渲染图,只评 plan 的逻辑/知识/可执行性——隔离 editor 噪声。
- `R_tool` 奖励**证据使用**(program 引用了工具输出)与**边际有用性**(IEA 2606.08016),不奖励调用次数(防 Proof-of-Use 式 hack)。

## 2. 可验证奖励快速通道(novel for editing:RLVR 子集)

v2 数据(`build_reasoning_tasks_v2.py`)每个任务携带:

```json
"ground_truth":  {"type": "sudoku_cell", "row": 1, "col": 2, "value": 3},
"verifier_spec": {"vqa_checks": [{"question": "What digit is in the highlighted cell?",
                                   "expected_answer": "3", "weight": 0.6}, ...],
                  "programmatic": {...}}
```

对 logical/symbolic/math/CS 任务族,`R_exec` 与 `R_cog` 由 **VQA 精确匹配**计算:VLM 只回答客观问题("高亮格里是什么数字"),逐项与 `expected_answer` 精确比对——VLM 做"读图器"而非"评分器",噪声远低于 1-5 打分(FIRM difference-first 的同源逻辑推到极致)。这部分等价于 RLVR,**作为 RL 第一阶段先训**:奖励最干净、且正是全场最弱的 RISE-Logical(开源 ≤15.3%)。

训练课程:Stage-RL-1 verifiable 子集 → Stage-RL-2 混入 VLM-judge 任务族(物理/历史/常识),此时 policy 已具备稳定工具/格式行为,judge 噪声的边际伤害更小。

## 3. 渲染高效组构造

每个 prompt 采 K=4-6 条轨迹,但渲染次数 ≠ K:

1. **前缀树 rollout**(TRACE 2606.11119):同一 prompt 的多条轨迹共享 analyze/search 工具前缀,仅在 edit_program 处分叉;只渲染叶子。渲染数下降 ~40-60%,且共享前缀=受控对比,天然提供 turn-level credit。
2. **回放缓存**(2604.08706):`(trajectory, rendered_image, reward_heads)` 缓存复用 2-4 个 update,turn-level importance clipping(SO-GRPO 2511.20718)。editor 固定 ⇒ reward 可精确复用,无需重评。
3. **All-wrong 工具子组重采样**(AXPO 2605.28774):若组内所有调用了工具的轨迹都失败,从共享 thinking 前缀重采样工具调用及后续,恢复梯度信号。
4. **Pass-rate 课程**(2605.05112):按任务历史 pass-rate 维持组在 ~50% 区间;饱和/绝望任务降采样,渲染预算让给信息量最大的组。

## 4. 噪声鲁棒 advantage

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)        # 标准 GRPO 基线
```

之上叠四个修正(全部来自 2026 验证过的做法):

1. **Grid 并排重打分**(Stable-Layers 2605.30257):K 个渲染候选拼一张 grid,judge 一次性并排排序——恢复组内方差、消除孤立打分的分数压缩。每组只 +1 次 judge 调用。
2. **Avg@K judge 自集成**(EditScore 实践):最终排序用 judge 采样均值。
3. **不平衡组降权**(S-GRPO 2508.05928):1-of-K 正确的组,单个假阳性可使 advantage 虚高 60% → 按组平衡度降权。
4. **不对称不确定性门控**(EGPO 2602.22751):低置信的**负向**判定收缩 advantage 幅度;正向不收缩——错罚好行为比漏奖更伤(2604.07666:precision > recall)。

按编辑类别离线审计 judge 系统性偏差(2603.16140),系统性错误的类别屏蔽出 RL 池。

## 5. Editor-gap-aware credit routing(novel;回应 Unified Thinker 的 grounding 批评)

RISE-Critic 输出 failure attribution(`planner_fail / editor_fail / over_edit / under_edit / region_fail / knowledge_fail / judge_uncertain`)。当 `editor_fail`(plan 对、渲染没执行出来):

```text
R_agent       = max(R_agent, 0.6 * R_program)     # 保底,不让 editor 噪声压垮好 plan
A_token(g)    = A_global * route(g, attribution)  # token 组路由
  route(editor_prompt | region | negative_constraints) = 1.0   # 主要更新对象
  route(reasoning | knowledge | tool_call)              = 0.2  # 屏蔽大部分负梯度
```

Token 组由输出 JSON 的字段边界确定(schema 固定,定位零成本)。这使 planner 通过 RL **学会写当前 editor 执行得动的 prompt**——即"grounded in editor capabilities",但无需像 Unified Thinker 那样做像素级 RL。

## 6. Gated Edit-OPD(辅助蒸馏;GenEvolve SDL 的 2026 修正版)

GenEvolve 原版 SDL 直接搬到多轮 agent 不稳(SDAR 2605.15155)。修正:

```text
L = L_GRPO + lambda_opd * L_EditOPD + beta_kl * KL(pi || pi_ref) + gamma * L_SFT_replay
lambda_opd = 0.5-1.0(GRPO 稳定后开启,同一批 on-policy tokens)

L_EditOPD = sum_t  gate_t * KL_rev(pi_teacher(t) || pi_student(t))    # stop-grad top-K
gate_t    = sigmoid(detach(logit_teacher - logit_student) / tau)      # gap 门控
            * asym(t)        # teacher 背书的 token 全权重;teacher 否决的 token 衰减 0.3
            * sign_agree(t)  # 与 GRPO advantage 符号一致 → 1.0;矛盾 → 0.25(SG-OPD)
```

- **Privileged context**(teacher-only):渲染结果图 + critic 诊断(expected/observed diff、失败 checklist、attribution、reward heads)+ 检索到的 best/worst 经验。Visual-SDPO(2606.10334)验证渲染反馈作 privileged context 有效。
- **稀有行为引导**(EDGE-OPD 2605.23493):某工具在 on-policy 数据中近乎不出现时,一小部分 rollout 采样时就注入经验,蒸馏只盖 privileged context 实际抬高概率的 token。
- **健康监控**:privileged context 必须实际移动 teacher logits(2604.13016,可测);蒸馏 KL 应集中于少数决策 token;只蒸馏跨任务复发的规则型经验。

## 7. 完整训练循环伪代码

```text
for step in 1..N:
  prompts = curriculum_sample(pool, target_pass_rate≈0.5)
  for p in prompts:
    tree   = rollout_prefix_tree(policy, p, K)            # 共享工具前缀
    leaves = tree.leaf_programs()
    imgs   = [render_cached(editor, z) for z in leaves]   # 缓存复用
    heads  = [critic(p, z, img) for z, img in zip(leaves, imgs)]
    heads  = grid_rescore(imgs, heads)                    # 并排重排
    R      = [gated_reward(h) for h in heads]             # §1 / §2
    if all_wrong(tool_subgroup): resample_from_prefix()   # §3.3
    A      = group_advantage(R, balance_w, uncert_gate)   # §4
    A_tok  = route_by_attribution(A, heads.attribution)   # §5
    exp    = extract_experience(best, worst) if gap>0.2
    L      = grpo_loss(A_tok) + lambda*opd_loss(exp, heads) + kl + sft_replay
  update(policy);  replay_buffer.age()
```

## 8. 与现有工作的差异

| 对比 | 它做的 | 本算法新增 |
|---|---|---|
| DDA-Thinker(2604.25477) | dual-atomic checklist + GRPO,开源 SOTA | 认知门控(min 而非加和)、editor-gap 路由、OPD、可验证快速通道、渲染高效组 |
| Edit-GRPO(2605.16951)/RC-GRPO | editor 侧 region-aware RL | 我们优化 planner token 策略,region 信号经 critic head 路由到 region token 组 |
| ThinkRL-Edit(2601.03467) | checklist 奖励 + chain preference | 工具编排、failure attribution、OPD、回放/前缀树效率层 |
| GenEvolve SDL(2605.21605) | 生成域 best/worst 经验蒸馏 | 编辑域 privileged 诊断、门控+符号一致性(SDAR/SG-OPD 修正)、editor-gap 感知 |
| FIRM(2603.12247) | CME 防 unchanged-shortcut | 第二重认知门;VQA 精确匹配把 difference-first 推到 RLVR |

## 9. 实现接入点与消融

- 框架:verl GRPO loop;rollout worker 持有 editor+critic 实例,异步队列(Polar 式);reward server = `rise_evolve/reward/server.py` 扩展 heads + attribution + verifier_spec 执行器。
- 第一里程碑(GPU 空闲后):K=4、50-100 步、仅 verifiable 子集(v2 logical)、无 OPD——预期看到 VQA 奖励单调上升且 format 不崩。
- 核心消融:±认知门(vs 加权和)、±editor-gap 路由、±grid 重打分、±前缀树(同渲染预算下对比)、±OPD、verifiable-first vs 混合课程。
- 红线:reward/OPD memory 不得包含任何 benchmark 内容;BoN/重排序在评测报告中如实披露。
