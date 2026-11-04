# 2026-11-04 observation 增益扫描

协议 `office-8b-traj-g-scan-v1` 已写入 [研究计划](<../研究计划 Research Plan 去噪推测解码.md>) 第十二章。这一步只写计划。没有实现，也没有运行。数字留空，等以后的运行记录。

四臂共用 xcat7 的 ESP 草稿 mask，只改 `spec.observation.gain`。g 为 0、0.25、0.5、0.75。g=0 是 trajectory-only。0.25、0.5、0.75 是混合。不测 g=1。

句子是 Dolly `databricks-dolly-15k` train 的 summarization，与 `office-8b-traj-xcat7-v1` 相同。128 条，qid 0–127，没有跳过。换行拼接的 SHA256 是 `f35b56bf668124a609c47dc51f97d40cabd75fdac5e6a3eb0fd133a25665020a`。生成长度固定 256。`stop_on_eos` false。

主指标是接受数，不是任务准确率。每个 g>0 与 g=0 在同一 128 条上配对。双侧符号检验的 p 只作探索，不设 p<0.05 通过线。τ 均值和 bootstrap 区间一并报告。墙钟只作描述，不作为停止规则。

以后的助手名字是 `studies/trajectory_observation/structure/office_traj_gscan.py`。命令留在 `office_traj.py`，选句沿用其中的 xcat7 助手。本记录不新增 Python 文件。
