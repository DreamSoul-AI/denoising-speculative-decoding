**去噪推测解码 开发计划**
更新日期：2026年11月4日。本文只登记工程工作和验收。研究问题、假设和停止规则在 [研究计划](<研究计划 Research Plan 去噪推测解码.md>)。最近的计时分离在 [2026-11-01 计时失败不再改写计数](<开发记录 Development Record/2026-11-01 计时失败不再改写计数.md>)。此前的包拆分和文档重整在 [2026-10-31 工程重构与文档重整](<开发记录 Development Record/2026-10-31 工程重构与文档重整.md>)。

# 一、已经完成、不必再排期
科学包 `spec_mtp` 在 2026-11-03 之前按 CLI、数据、模型、解码、mask、树、审计和分析落在扁平模块。该布局现在是冻结遗留。目标目录改为 `structure` / `flow` 两柱，规则在 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)。Study 通过 `studies/_shared` 调用解码器。K Study 的 `postcommit-final-norm-v1` 夹具能在合成输入上拟合和拒绝错误协议名。τ 的零调用口径已收束到 `prompt_tau`。时钟或设备同步失败时，解码仍返回 token 和接受计数，并且不补写 0 时长。文档入口已按职责分开。

# 二、剩余工程项
2026年11月3日的目录规则已经写入代码结构设计。步骤 0 是文档和 CPU 门禁，已经完成。步骤 1 已把 `contracts.py` 迁到 `structure/api/decode_api.py`，旧路径再导出已删除，不改数学。步骤 2 已把 `tree.py` 拆到 `structure/algorithm/decode/`，旧路径再导出已删除。步骤 3 已把 `masks.py` 拆到 `structure/algorithm/mask/`，顶层再导出已删除。步骤 4 已把 `decode.py` 迁到 `structure/algorithm/decode/loop.py`，顶层再导出已删除。步骤 5–7（data/model/audit/analysis、trajectory_observation、cli）尚未开始。步骤 8 里测试镜像改路径和 `.importlinter` 按两柱重写仍未做。那些步骤只搬位置，不改数学。本计划不因为文档改了就授权后续搬迁。

| **事项** | **原因** | **验收** |
| --- | --- | --- |
| 会议方法的 CPU 合成检查 | 已完成。top1 确认未通过，oracle 空间全在槽 1，停止在合成设置上换形式 | 见运行记录 |
| observation 上的仿射 K | 研究计划把完整 K 与仅 bias 列为对照 | 合成检查有结论后再定 z→Az+b 的拟合数据 |
| 飞书规范正文 | 用户令牌过期时只读了索引和参考仓 | 下次能读取时核对冲突段落，不预先改写成“已符合全文” |
| 下一高信息实验的逐臂墙钟 | 研究计划第九章已写定 `office-8b-traj-wall-v1`。`office_traj.py` 已逐臂记录 `wall_seconds`，缺失不写成 0。外层打印仍不进入比较 | 已跑。接受数 `direction` 为 positive，`wall_direction` 为 shorter，两区间都不含 0。见 [2026-11-04 Qwen3-8B trajectory墙钟](<运行记录 Run Record/2026-11-04 Qwen3-8B trajectory墙钟.md>)。次项 B `office-8b-traj-failset-v1` 已分析，见 [2026-11-04 Qwen3-8B trajectory失败子集](<运行记录 Run Record/2026-11-04 Qwen3-8B trajectory失败子集.md>) |
| 独立 AR 不一致的只读分类 | 研究计划第十章 `office-ar-inconsistency-v1`。已有行含 `ar_equal`、`output_sha256`、`argmax_passed`、`accepted`、`tau`。没有首分叉下标、共同前缀长度或两侧 token。八组与墙钟 jsonl 在运行记录所指目录 | 已完成。`classify_pair` 与 `classify_archive` 在 `office_traj.py`。旧行缺字段保持缺失。结论见开发记录 |
| 首分叉与 fp32 参照 | 研究计划第十章 `office-ar-first-fork-v1`。这一次小规模 GPU 已批准。记录臂相对 BF16 AR 的首分叉；float32 AR 只在常驻权重放得进空闲显存时加载 | 已跑 128 条。共享分叉比例 0.546875，不贴两档。fp32 状态 `not_loaded`。见 [2026-11-04 Qwen3-8B 首分叉](<运行记录 Run Record/2026-11-04 Qwen3-8B 首分叉.md>) |
| BF16 AR 自决定 | 研究计划第十章 `office-ar-bf16-determinism-v1`。这一次小规模 GPU 已批准。在 `office_traj.py` 做双运行、整段路径对逐 token prompt，以及归档下标上的 top1−top2。`ar_generate` 增加默认关闭的 `prompt_tokenwise`。产物在源码树外。不加载 fp32，不跑墙钟按长度 | CPU 测试覆盖三条判定、中间档和归档 qid 并集。正式数字见运行记录 |

不新增插件注册、通用校正框架或第二套测试运行器。

2026年11月3日，同节点预算的额外分支按研究计划里已写定的 `extra-branch-replace-v1` 执行。这次只比较接受数，不测速，不下载模型。运行产物留在源码树外。

# 三、RPipe 来源（B2 作废）
2026年11月1日 Enmao 决定：仓库不再内嵌 `src/rpipe/`，顶层不再有 `tools/`。Study 流程由 `spec_mtp.flow` 与 `spec_mtp.structure` 自己实现，目录以 https://github.com/DreamSoul-AI/RPipe 为结构模板持续对齐。此前“继续固定 v0.3.1”的默认假设作废。来源提交 `7a15c46b75c5eb951cccd60ed7b9fc3463db90f5` 只说明被吸收的那一版流程代码从哪来，不再是本仓的固定副本。

# 四、明确不做
不下载模型，不恢复范数矩阵、配对矩阵和 ESP 长矩阵，不恢复 BF16 相对独立 AR 的算子数值机理排查，不合并 `dev` 或 `main`，不改 Actions 和仓库保护，不把运行产物提交进 Git。`office-ar-first-fork-v1` 的 GPU 已经跑完。`office-ar-bf16-determinism-v1` 的这一次小规模 GPU 已经批准。除此以外的 GPU 测量、GPU 测速和复跑仍须另行批准。墙钟按长度不在这次批准里。
