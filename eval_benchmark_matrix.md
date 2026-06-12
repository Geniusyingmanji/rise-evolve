# RISEvolve 评测矩阵 v2(2026-06-11)

响应"图像推理相关 benchmark 也应考虑"的扩展设计。全部条目经 2026-06-11 可用性核实(HF/GitHub/license/judge)。

## Tier 1 主评测(论文主表,不变)

| Benchmark | 规模 | Judge | 注意 |
|---|---|---|---|
| RISEBench | 360 | GPT-4o/4.1(README 两说,跑前核实 `gpt_eval.py`) | 必须用 post-2026-04-23 修复版脚本 |
| GRADE | 520 | 疑似 Gemini Flash(文档不明,跑前读 eval 代码并冻结) | 无 planner 方法发表过数字 = first-mover |
| KRIS-Bench | 1,267 | GPT-4o(2025-05 snapshot,官方 pin) | 无 2026 更新 |

## Tier 2 推理编辑扩展评测(泛化证明 + 廉价开发门)

| Benchmark | 数据 | License | 规模/分类 | Judge | 角色 |
|---|---|---|---|---|---|
| **PaintBench** | HF `PaintBench/PaintBench` | 代码 MIT / 数据 CC-BY-4.0 | test 2,020 + dev 280;20 ops/4 类 | **确定性**(CIE ΔE76 + mIoU),无 judge | **dev-280 作日常回归门**:零 API 成本、零 judge 噪声、抗污染(程序生成可再生);注意 SOTA 仅 17.1% mIoU,小模型有地板效应 |
| **RE-Edit** | HF `Yixuan-Ding-ZJU/RE-Edit` | MIT | 1,000;5 维(物理/环境/文化/因果/指代),中英双语 | Qwen3-VL-30B 主 + GPT-4.1 副 | 主 judge 开源 = GPU 空闲时可**本地复现评测**;判 variant/snapshot 未 pin,用前核对 eval config |
| **UniREditBench** | HF `maplebb/UniREditBench` | HF 卡 Apache-2.0(GH 无 LICENSE) | 2,700;8 维/18 子维,game-world+real | GPT-4.1(pin `gpt-4.1-2025-04-14`)双参考(文本+GT图) | game-world 维度与我们 v2 生成器同类型,极适合验证 logical 提升。**⚠ 污染耦合确认**:与 UniREdit-Data-100K 同 pipeline——若训练用了该数据,报分必须先做图像级去重并在论文披露 |
| WiseEdit | HF `123123chen/WiseEdit-Benchmark` | Apache-2.0(HF 卡) | 1,220;认知三阶段×知识三型 + Complex 子集 | GPT-4o(未 pin 版本) | 知识/认知编辑泛化;判定可复现性弱,作次要 |
| TECCI | HF `google/tecci` | CC-BY-4.0 | 7,550(GGIS 7,020 + IRCS 530) | 人评 + Gemini 3 Flash autorater(**代码未放出**) | 抗污染最强(作者自摄未公开图);自动评测需自行复刻 autorater prompt,暂列观察 |
| InEdit-Bench | HF `SZStrong/InEdit-Bench` | Apache-2.0(HF 卡) | HF 仅 477 行(论文未给总数,需核对) | GPT-4o(未 pin) | 中间逻辑路径(状态转移/动态过程/科学模拟),与 RISE temporal/causal 互补;规模存疑作次要 |

推荐组合:**主表 Tier1 三个 + PaintBench + RE-Edit + UniREditBench(带去重披露)**;WiseEdit/InEdit 进附录,TECCI 待 autorater。

## Tier 3 Planner 能力回归监控(每次 SFT/RL 后必跑)

依据:窄域 SFT 会压制 VLM 通用能力且标准评测不可见(arXiv:2602.01611, 2604.08388)。我们的 planner 是 Qwen3-VL-8B,理解能力 = 源图分析/知识运用的根基。

| Eval | HF id / split | 规模 | 用法 |
|---|---|---|---|
| MathVista | `AI4Math/MathVista` testmini | 1,000 | 图表/几何推理回归(GRADE-math 相关性高) |
| MMMU | `MMMU/MMMU` validation | 900 | 学科知识回归(GRADE 10 域相关) |
| MMStar | `Lin-Chen/MMStar` val | 1,500 | 通用感知/推理(license 未标,仅内部使用) |
| BLINK | `BLINK-Benchmark/BLINK` val | ~1,901(14 配置) | 空间/对应/深度感知(RISE-spatial 相关) |

门限建议:任一回归 >3 个百分点 → 数据混合比/LoRA 配置回炉(对照 2604.08388 的 ~100 条通用轨迹回放修复)。

## 去污染扩展(立即生效的规则)

1. `freeze_benchmarks.py` 待扩展:UniREditBench、RE-Edit、WiseEdit、InEdit-Bench、PaintBench(text+image 双指纹);TECCI 图像未在网上出现过,风险天然低。
2. **硬规则**:引入 UniREdit-Data-100K 训练之前,必须先冻结 UniREditBench 指纹并做 pHash+SSCD 双向去重;其 game-world 与我们 v2 生成器同类但独立实现,v2 数据无耦合。
3. Tier 3 理解评测同样进去污染索引(训练数据不得含其题目)。
4. 所有 judge 版本在 `configs/eval_*.yaml` 中 pin 死并写进论文 reproducibility 段。

## 对当前工作的直接含义

- v2 可验证数据(数独/迷宫/电路/排序…)与 UniREditBench game-world、PaintBench symbolic ops 高度同型——这两个 benchmark 是检验 v2 数据价值的最快信号,且 PaintBench 零成本。
- RE-Edit 的 EditRefine 基线(Qwen2.5-VL-7B agent + QIE)是我们 8B 对比表的又一行。
- Tier 3 回归监控直接回答"SFT 是否伤害了 planner 理解能力"——v2r 失败的另一种早期预警。
