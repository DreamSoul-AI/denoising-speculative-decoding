# 2026-11-01 trajectory 与 observation 联合 mask

2026年10月31日会议的方法没有公式（待负责人定夺事项 B3）。本次按研究计划第一章写下的默认定义 `trajectory-observation-blend-v1` 实现，不等待确认；负责人另定公式时用新协议名，本次实现与结果不改签。RPipe 来源（B2）按“继续固定 v0.3.1”的默认推进，本次没有触及 `src/rpipe/`。

## 改动

- `spec_mtp.masks.ObservationBlendMaskProvider` 包装 ESP 或 EMA provider。`masks()` 返回 \(\hat m + g(z-\hat m)\)，z 为上一验证前向 mask 槽 logits 的 softmax 对输入嵌入表的期望。observation 不写回被包装的状态。输出 hidden provider 被拒绝。
- `build_mask_provider` 读取 `spec.observation.gain`，只接受这一个键，g 必须在 [0,1]。
- `SpecDecoder` 在 prefill 和每个继续的解码轮取出下一轮 mask logits 后调用 provider 的 `observe_mask_logits`（provider 有此方法时）。验证、树、缓存和计数没有改。
- `spec_mtp.analysis.round_split` 按上一轮是否接受草稿给出三组轮数、接受数、提交数和 τ，用来检验“收益只是利用了最后提交 token”（研究计划 A1）。

## 测试

`tests/spec_mtp/test_observation_blend.py`：g=0 与原 provider 的输出和 trace 完全相同；g∈{0.5,1}、T∈{0,1}、ESP/EMA、静态/动态树下输出等于独立 AR，T0 argmax 审计通过；混合代数与不写回；解码器观测次数等于 1 加继续轮数；非法设置和非有限 logits 被拒绝；`round_split` 的手算 trace。实际执行结果见运行记录。

没有接受数或速度的测量，tiny 随机模型上的通过不说明方法有效。

## 测量前的对齐修正

首版把同一组 z 共享给树上所有 owner。深度 d 的 owner 的槽 j 代表第 d+j 层，因此 d>0 时错位。在任何测量之前改为 `masks_by_depth`：槽 j 用 \(z_{d+j}\)，超出 k 时退回 trajectory；解码器有此方法时按 `tree.depths` 取 owner 的 mask。协议名不变，因为此前没有产生测量。

## 拒绝条件化变体

`spec.observation.condition: rejection`（协议 `trajectory-observation-rejection-v1`）。解码器的挂钩由 `masks_by_depth` 改为 `masks_for_tree(tree)`，provider 拿到整棵树后，对有子节点的 owner 的槽 1 用排除子节点 token 后重新归一化的期望嵌入，只需子节点的嵌入，不对整张词表重算。v1 等价于不做排除的同一入口。测试覆盖手算条件化、质量全在子节点时退回 trajectory、非法 condition，以及两种 condition 下 g=0 恒等和与 AR 一致。

## CPU 合成检查入口

`studies/trajectory_observation/structure/pilot.py` 在 CPU 上训练 tiny tied LLaMA 学两种合成二阶马尔可夫语言，只在语言 A 拟合 prompt 上选联合增益，冻结后报告语言 A 隔离 prompt 与语言 B 的 τ、`round_split`、AR 一致和 argmax 审计。输出必须在源码树外，已存在的文件不覆盖。它还没有包装成 Study。结果见 [联合 mask 检查](<../运行记录 Run Record/2026-11-01 联合mask检查.md>)。

同日稍后，Enmao 作废 B2：删除内嵌 `src/rpipe/` 和顶层 `tools/`。仍被测试和 Study 调用的脚本移到 `studies/_shared/structure` 或 `studies/_shared/flow`，测试入口改为 `tests/run.py`。流程实现留在 `spec_mtp.flow` 与 `spec_mtp.structure`，不再 `import rpipe`。上文“没有触及 src/rpipe”只描述联合 mask 实现当时的范围。

`bonus-slot-hard-v1`（`spec.bonus_slot.mode: hard`）只替换各 owner 的槽 0 为上一轮对应行 argmax 的 token 嵌入，其余槽保持 trajectory，不增加前向，不与 observation 混用。真实模型入口仍是 `spec-mtp decode` 的本地目录与外部 SHA256 清单；极小 Llama 的 CPU 冒烟走这条路径。确认判据写在研究计划，测量写在运行记录。

`bonus-slot-topk-v1`（`mode: topk`）把同一槽换成该行 logits 在草稿宽度 `branches[d]` 上重新归一化后的期望嵌入。宽度不是正整数时拒绝。不增加前向，不与 observation 混用。

## 合成干预与小模型对照（论文可用）

比较都是 prompt 级、三棵树接受数差相加后的双侧符号检验，192 对（3 种子 × 2 划分 × 32）。胜必须多于负且 p<0.05 才算通过。τ 是解码阶段提交数除以解码调用数。数字来自同日运行记录，这里不另算。

| **干预** | **比较** | **结果** | **结论** |
| --- | --- | --- | --- |
| observation 联合 v1 | 确认种子 21/22/23，ESP 联合对 ESP 轨迹 | 70 胜 99 负，p=0.031 | 负结果，方向更差 |
| observation top1·拒绝 | 确认种子 21/22/23，ESP 联合·top1·拒绝对 ESP 轨迹 | 74 胜 103 负，p=0.035 | 负结果，与探索方向相反 |
| `bonus-slot-hard-v1` | 探索种子 11/12/13，ESP bonus 对 ESP 轨迹 | 79 胜 99 负 14 平，p=0.154 | 负结果，不跑确认 |
| `bonus-slot-topk-v1` | 探索种子 11/12/13，同一比较 | 78 胜 96 负 18 平，p=0.197 | 负结果，不跑确认 |
| oracle 上界 | 种子 11/12/13，全部槽真实嵌入对轨迹 | k=1/2/3 的 τ 从约 1.33/1.18/1.18 升到约 1.90/2.66/3.36 | 空间存在，接近 k+1 |
| oracle 拆分 | 只给 bonus 之后的槽真实嵌入 | k=2/3 的 τ 与轨迹相同（约 1.19/1.17） | 收益全在 bonus 槽的真实 token |
| Qwen3-0.6B ESP 对 AR | CPU，T=0，k=1 宽度 [4]，4 条 prompt 各 8 token | 4/4 token 相同，argmax 通过，合并 τ=1.333 | 贪心一致；样本太小，不检验 bonus 槽 |
| Qwen3-0.6B ESP 对 AR | 同一树，32 条 prompt 各 16 token | 32/32 token 相同。τ 均值 1.355，95% 区间 [1.302, 1.410]。每 token 前向 0.764，墙钟 0.144 秒；AR 为 1.000 次、0.133 秒 | 贪心一致。前向更少，这条 CPU 墙钟并不更快 |
| `bonus-slot-hard-v1` 真实 | 同上，对 ESP 默认 | τ 均值 1.246 [1.182, 1.320]。5 胜 20 负 7 平，p=0.004，均值差 −0.108 [−0.164, −0.053] | 负结果，显著更低 |
| `bonus-slot-topk-v1` 真实 | 同上，对 ESP 默认 | τ 均值 1.276 [1.209, 1.348]。5 胜 14 负 13 平，p=0.064，均值差 −0.078 [−0.132, −0.028] | 判据未通过；均值更低 |
| 标量卡尔曼，合成探索 | 种子 11/12/13，在线对 ESP 轨迹 | 117 胜 59 负 16 平，p=1.5e-5。冻结 K 对轨迹 76:78。在线对冻结 110:72（p=0.006） | 探索通过。拒绝后轮次 71:113，A1 未排除 |
| 标量卡尔曼，合成确认 | 种子 21/22/23，同一主比较 | 101 胜 83 负 8 平，p=0.210。在线对冻结 96:85（p=0.457） | 未通过。这是 trajectory-only，不否定会议的观测项。不上 Qwen3-0.6B |
| 轨迹加对角 K | 种子 11/12/13，离线对角 K 对 ESP 轨迹 | 91 胜 80 负 21 平，p=0.445。拟合 K 约 −0.01 到 −0.03 | 主比较未通过。不跑确认 |
| 冻结仿射 K | 种子 31/32/33。看过 11/12/13 次要臂后另立的主比较，对 ESP 轨迹 | 101 胜 85 负 6 平，p=0.271。holdout 74:21，cross 27:64。参数 4160，拟合 1504 行 | 未通过。不跑确认，不上 Qwen3-0.6B |
| 混合语言岭仿射 | 种子 41/42/43。看过上一条后才设计。两种语言混合 fit，岭在 fit 内选择 | 129 胜 50 负 13 平，p=3.1e-9。语言 A 60:31，语言 B 69:19。重放误差 fit 与两种评测都约 0.022–0.026 | 探索通过 |
| 混合语言岭仿射，确认 | 种子 21/22/23，同一主比较 | 124 胜 58 负 10 平，p=1.1e-6。语言 A 57:32（p=0.011），语言 B 67:26（p=2.5e-5）。在线对冻结 35:42（p=0.494） | 通过。在线不更高 |
| Qwen3-0.6B 岭仿射 | 32 条未见短 prompt，k=1 宽度 [4]，拟合用另 192 条 | 14 胜 8 负 10 平，p=0.286。τ 均值 1.368 对 1.355。重放误差 fit 3.8e-4，评测 6.7e-4 | 未通过。不改树、不重选岭 |

拒绝条件化的主判据是对 EMA 轨迹，确认 102:71（p=0.022）通过，但对 ESP 轨迹是 88:94。合成负结果都是相对 ESP 轨迹。真实比较的两次符号检验没有做多重校正。

结论：合成设置上，不增加前向的 observation 形式、top1 和两种 bonus 槽嵌入都没有超过 ESP 轨迹。Qwen3-0.6B 上 hard 的 τ 显著更低，topk 的均值也更低。标量卡尔曼只重写轨迹、不用本轮 verify 观测，确认未通过。按 10 月 31 日会议写下的 \(m=\hat m+K(z-\hat m)\)，对角 K 在探索种子上是 91:80（p=0.445），同样未通过。仿射是次要臂，探索 111:70，但不因此改主比较。拟合得到的对角和标量 K 都接近 0。局限：z 取最近提交 token 的输入嵌入，只改槽 0，K 在合成拟合集上估计。确认和 Qwen3-0.6B 都没有跑。此后把仿射另立为新的主比较，探索种子改为 31/32/33：101:85（p=0.271），仍未通过。holdout 与 cross 符号相反，见运行记录。看过该结果后另立混合语言岭仿射，探索种子 41/42/43 为 129:50（p=3.1e-9），确认种子 21/22/23 为 124:58（p=1.1e-6），两种语言都是胜多于负。在线对冻结在两段里都不更高。Qwen3-0.6B 上同一冻结臂是 14:8（p=0.286），未通过。
