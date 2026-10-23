# 去噪推测解码
研究免训练多 token 起草与验证：以 ESP 为参照，比较输入嵌入 EMA、速度外推和输出隐藏状态掩码。中文名称为“去噪推测解码”。仓库地址：[DreamSoul-AI/denoising-speculative-decoding](https://github.com/DreamSoul-AI/denoising-speculative-decoding)。

截至2026年10月21日，8B 非官方 Table 1 独立复验已完成，论文目标未复现，关键结果及证据边界见 [通用调研](<docs/去噪推测解码 通用调研.md>)。提交 token 的验证前向 argmax 自检与独立 AR 一致率分别报告，τ 是提交数与解码调用数之比，不能据此认定实际速度收益。

K=[A,b] 仿射残差校正已于2026年10月20日确认作为独立 Study 路线，先 CPU 小规模，再冻结 K 作跨数据集三臂对照。本分支提供独立 Study 的合成微型随机模型工程验证，尚未执行原定160对数据协议，未证明命中率或泛化提升。

# 一、文档入口
| **文档** | **维护内容** |
| --- | --- |
| [协作约定](AGENTS.md) | 资料定位、适用规范及执行边界 |
| [去噪推测解码 通用调研](<docs/去噪推测解码 通用调研.md>) | 技术机制、来源比较、历史结果与证据边界 |
| [去噪推测解码 架构设计](<docs/去噪推测解码 架构设计.md>) | 目标质量场景、职责、接口、状态、部署和验证 |
| [去噪推测解码 代码结构设计](<docs/去噪推测解码 代码结构设计.md>) | 目标 Python 包、文件责任、依赖及测试镜像 |
| [去噪推测解码 研究计划](<docs/去噪推测解码 研究计划.md>) | 唯一中文计划、Study 问题与暂停/恢复条件 |

# 二、当前实现与使用边界
当前安装包为 `denoise-spec-mtp`，科学包 `spec_mtp` 按 `contracts/data/model/decode/masks/tree/audit/analysis/cli` 分工。`python -m spec_mtp` / `spec-mtp` 提供 `decode/audit/compare`；旧 make/launch/process 研究命令明确拒绝并提示使用 `python -m studies._shared.flow.cli`。只有 `studies/` 使用 RPipe 薄适配。

`src/rpipe/` 固定于 `7a15c46b75c5eb951cccd60ed7b9fc3463db90f5`（v0.3.1），既存 Windows torch-first 差异按当前字节保留。第三方 ESP 来源树及 `ORIGIN.json` 保留，不能把它称为作者官方实现。原始 SpecBench 数据保留于 `datasets/spec_bench/question.jsonl`。

旧逐条运行证据、日志、图和报告保留在固定历史/外部运行根，未恢复进 Git。历史原件重放项在 `tests/manifest.json` 逐 selector 列出缺失原件；当前声明、输入拒绝、模型、mask、提交链、来源和聚合合同仍在 CPU 计划内。精确历史重放未执行，不能称全部历史结果已复核。

# 三、开发与验证
工作分支经代码 Draft PR 面向 `dev`，待评审；不合并、不启用自动合并或发布。当前 HEAD 的本地结果、历史红项和未覆盖范围记在 PR 正文，不用旧提交结果代替。

执行离线 CPU 计划：`python tools/run_tests.py --plan cpu-pr --output-root <源码树外的新验证目录>`。静态清单先验证必需项和精确排除，逐参数 setup/call/teardown 写独立记录，缺收集、缺终态、失败、跳过和证据写失败均非零。整批预算420秒，在测试边界检查；子进程测试各有有限超时，单个本地计算不支持强制中断。运行产物不进 Git。

本轮不调整现有 Actions、CI/CD 或设置。远端结果仅作参考，本地 CPU 回归不覆盖 CUDA、真实权重/数据、吞吐或采样分布保证。已有 CUDA 环境保持原样，不下载模型或覆盖 torch。
