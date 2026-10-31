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
