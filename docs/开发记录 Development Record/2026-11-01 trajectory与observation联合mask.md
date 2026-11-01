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
| Qwen3-0.6B ESP 对 AR | CPU，T=0，k=1 宽度 [4]，4 条 prompt 各 8 token | 4/4 token 相同，argmax 通过，合并 τ=1.333 | 贪心一致；不是加速，也不检验 bonus 槽 |

拒绝条件化的主判据是对 EMA 轨迹，确认 102:71（p=0.022）通过，但对 ESP 轨迹是 88:94。上表的负结果都是相对 ESP 轨迹。

结论：合成设置上，不增加前向的 observation 形式、top1 和两种 bonus 槽嵌入都没有超过 ESP 轨迹。oracle 说明缺的是 bonus 槽上的真实 token，草稿 argmax 和 top-w 期望嵌入都没有逼近它。再换一种同类嵌入不再做。本机最小完整缓存是 Qwen3-0.6B（不用 1.7B/4B，不下载、不用 GPU）；在这个小规模上，贪心 ESP 与独立 AR 的 token 相同，τ 只说明有接受，不能当成速度或方法收益。
