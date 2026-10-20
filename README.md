# 去噪推测解码
研究免训练多 token 起草与验证：以 ESP 为参照，比较输入嵌入 EMA、速度外推和输出隐藏状态掩码。中文名称“去噪推测解码”为建议名，待 Enmao 汇总确认。仓库地址：[DreamSoul-AI/denoising-speculative-decoding](https://github.com/DreamSoul-AI/denoising-speculative-decoding)。

截至2026年10月21日，8B 非官方 Table 1 独立复验已完成，论文目标未复现，关键结果及证据边界见 [通用调研](<docs/去噪推测解码 通用调研.md>)。提交 token 的验证前向 argmax 自检与独立 AR 一致率分别报告，τ 是提交数与解码调用数之比，不能据此认定实际速度收益。

K=[A,b] 仿射残差校正已于2026年10月20日确认作为独立 Study 路线，先 CPU 小规模，再冻结 K 作跨数据集三臂对照。当前 dev 无该实现，本轮只形成文档设计，尚未证明命中率或泛化提升。

# 一、文档入口
| **文档** | **维护内容** |
| --- | --- |
| [协作约定](AGENTS.md) | 资料定位、适用规范及执行边界 |
| [去噪推测解码 通用调研](<docs/去噪推测解码 通用调研.md>) | 技术机制、来源比较、历史结果与证据边界 |
| [去噪推测解码 架构设计](<docs/去噪推测解码 架构设计.md>) | 目标质量场景、职责、接口、状态、部署和验证 |
| [去噪推测解码 代码结构设计](<docs/去噪推测解码 代码结构设计.md>) | 目标 Python 包、文件责任、依赖及测试镜像 |
| [去噪推测解码 研究计划](<docs/去噪推测解码 研究计划.md>) | 唯一中文计划、Study 问题与暂停/恢复条件 |

# 二、当前实现与使用边界
当前安装包为 `denoise-spec-mtp`，Python 包为 `spec_mtp`。入口仍是 `python -m spec_mtp` 和 `spec-mtp`。它通过固定的 RPipe 副本执行 Study，当前源码仍采用 `flow/structure` 组织，目标标准 Python 包设计尚未实施。

`src/rpipe/` 固定于 `7a15c46b75c5eb951cccd60ed7b9fc3463db90f5`（v0.3.1），既存 Windows torch-first 差异按当前字节保留。第三方 ESP 来源树及 `ORIGIN.json` 保留，不能把它称为作者官方实现。原始 SpecBench 数据保留于 `datasets/spec_bench/question.jsonl`。

本轮收敛了第一方文档并移除旧报告、图和运行证据快照。旧 Study 声明、分析工具和文档测试仍引用部分旧路径，尚未随设计迁移。新 clone 能阅读本页和四份文档，不能直接重放旧 Study、复算已删除的公开逐条摘要或核验本机原始前向。恢复历史材料须使用通用调研列出的固定 Git 提交和原路径，不能用当前文档替代冻结字节。

# 三、开发与验证
工作分支经 Draft PR 面向 `dev`。本轮仅交付文档设计，等待评审，不重构代码、不合并、不启用自动合并或发布。

现有 `.github/workflows/cpu-checks.yml` 使用 Ubuntu 24.04/Python 3.12，执行 Ruff、mypy、RPipe 来源/入口检查和 `pytest tests/spec_mtp`。该配置存在不代表本轮通过。旧文档索引与双语计划测试不适用于新文档结构，当前是已知未完成项，不能以历史628项通过记录放行本次改动。

后续实施前先核对 [研究计划](<docs/去噪推测解码 研究计划.md>) 和影响范围，再决定离线检查。已有 CUDA 环境保留 torch，不以 CPU 安装命令覆盖。权重优先 ModelScope，历史 GPU、测速和下载命令均不是新的执行授权。
