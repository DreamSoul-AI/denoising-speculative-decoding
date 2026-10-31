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

## CPU 合成检查入口

`studies/trajectory_observation/structure/pilot.py` 在 CPU 上训练 tiny tied LLaMA 学两种合成二阶马尔可夫语言，只在语言 A 拟合 prompt 上选联合增益，冻结后报告语言 A 隔离 prompt 与语言 B 的 τ、`round_split`、AR 一致和 argmax 审计。输出必须在源码树外，已存在的文件不覆盖。它还没有包装成 RPipe Study。结果见 [联合 mask 检查](<../运行记录 Run Record/2026-11-01 联合mask检查.md>)。
