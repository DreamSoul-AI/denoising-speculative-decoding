**去噪推测解码 开发计划**
更新日期：2026年11月4日。本文只登记工程工作和验收。研究问题、假设和停止规则在 [研究计划](<研究计划 Research Plan 去噪推测解码.md>)。最近的计时分离在 [2026-11-01 计时失败不再改写计数](<开发记录 Development Record/2026-11-01 计时失败不再改写计数.md>)。此前的包拆分和文档重整在 [2026-10-31 工程重构与文档重整](<开发记录 Development Record/2026-10-31 工程重构与文档重整.md>)。

# 一、已经完成、不必再排期
科学包 `spec_mtp` 在 2026-11-03 之前按 CLI、数据、模型、解码、mask、树、审计和分析落在扁平模块。该布局现在是冻结遗留。目标目录改为 `structure` / `flow` 两柱，规则在 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)。Study 通过 `studies/_shared` 调用解码器。K Study 的 `postcommit-final-norm-v1` 夹具能在合成输入上拟合和拒绝错误协议名。τ 的零调用口径已收束到 `prompt_tau`。时钟或设备同步失败时，解码仍返回 token 和接受计数，并且不补写 0 时长。文档入口已按职责分开。

# 二、剩余工程项
2026年11月3日的目录规则已经写入代码结构设计。步骤 0 是文档和 CPU 门禁，已经完成。步骤 1 已把 `contracts.py` 迁到 `structure/api/decode_api.py`，旧路径保留再导出，不改数学。步骤 2 已把 `tree.py` 拆到 `structure/algorithm/decode/`，旧路径保留再导出。步骤 3–8（masks、decode、data/model/audit/analysis、trajectory_observation、cli、清 shim）尚未开始，也未另行批准。那些步骤只搬位置，不改数学。本计划不因为文档改了就授权后续搬迁。

| **事项** | **原因** | **验收** |
| --- | --- | --- |
| 会议方法的 CPU 合成检查 | 已完成。top1 确认未通过，oracle 空间全在槽 1，停止在合成设置上换形式 | 见运行记录；真实模型比较仍需另行批准 |
| observation 上的仿射 K | 研究计划把完整 K 与仅 bias 列为对照 | 合成检查有结论后再定 z→Az+b 的拟合数据 |
| 飞书规范正文 | 用户令牌过期时只读了索引和参考仓 | 下次能读取时核对冲突段落，不预先改写成“已符合全文” |
| 下一高信息实验的逐臂墙钟 | 研究计划第九章已写定 `office-8b-traj-wall-v1`。现有 office 正式行没有逐臂 `wall_seconds`，外层打印把两臂和独立 AR 包在同一次计时里 | 获准后再改 `studies/trajectory_observation/structure/office_traj.py`。本行不授权实现或 GPU |

不新增插件注册、通用校正框架或第二套测试运行器。

2026年11月4日，研究计划第九章写定下一高信息实验。主项是 `office-8b-traj-wall-v1`：在已经跑完的 summarization 切片上，比较 traj 与 full 的逐臂墙钟和接受数是否同向。次项是 `office-8b-traj-failset-v1`：从已归档 jsonl 界定 full 接受数更高的 prompt 子集，不新跑模型。获准实现墙钟时，只在 `structure/office_traj.py` 记下第九章定义的 `wall_seconds`，接受计数保持原口径，不另建脚本，不增加 legacy 条目。本段不批准这次实现，也不批准 GPU 运行。

2026年11月3日，同节点预算的额外分支按研究计划里已写定的 `extra-branch-replace-v1` 执行。这次只比较接受数，不测速，不下载模型。运行产物留在源码树外。

# 三、RPipe 来源（B2 作废）
2026年11月1日 Enmao 决定：仓库不再内嵌 `src/rpipe/`，顶层不再有 `tools/`。Study 流程由 `spec_mtp.flow` 与 `spec_mtp.structure` 自己实现，目录以 https://github.com/DreamSoul-AI/RPipe 为结构模板持续对齐。此前“继续固定 v0.3.1”的默认假设作废。来源提交 `7a15c46b75c5eb951cccd60ed7b9fc3463db90f5` 只说明被吸收的那一版流程代码从哪来，不再是本仓的固定副本。

# 四、明确不做
不下载模型，不恢复范数矩阵、配对矩阵和 ESP 长矩阵，不做尚未另行批准的 GPU 测速，不合并 `dev` 或 `main`，不改 Actions 和仓库保护，不把运行产物提交进 Git。研究计划第九章写定了墙钟和失败子集的协议。写进计划不等于批准实现或 GPU 运行。真实实验仍要另行批准后才进入实现。
