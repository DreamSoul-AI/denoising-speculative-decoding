**去噪推测解码 开发计划**
更新日期：2026年11月21日。本文只登记工程工作和验收。研究问题、假设和停止规则在 [研究计划](<研究计划 Research Plan 去噪推测解码.md>)。最近的计时分离在 [2026-11-01 计时失败不再改写计数](<开发记录 Development Record/2026-11-01 计时失败不再改写计数.md>)。此前的包拆分和文档重整在 [2026-10-31 工程重构与文档重整](<开发记录 Development Record/2026-10-31 工程重构与文档重整.md>)。

# 一、已经完成、不必再排期
科学包 `spec_mtp` 在 2026-11-03 之前按 CLI、数据、模型、解码、mask、树、审计和分析落在扁平模块。该布局现在是冻结遗留。目标目录改为 `structure` / `flow` 两柱，规则在 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)。Study 通过 `studies/_shared` 调用解码器。K Study 的 `postcommit-final-norm-v1` 夹具能在合成输入上拟合和拒绝错误协议名。τ 的零调用口径已收束到 `prompt_tau`。时钟或设备同步失败时，解码仍返回 token 和接受计数，并且不补写 0 时长。文档入口已按职责分开。

# 二、剩余工程项
2026年11月3日的目录规则已经写入代码结构设计。步骤 0 是文档和 CPU 门禁，已经完成。步骤 1 已把 `contracts.py` 迁到 `structure/api/decode_api.py`，旧路径再导出已删除，不改数学。步骤 2 已把 `tree.py` 拆到 `structure/algorithm/decode/`，旧路径再导出已删除。步骤 3 已把 `masks.py` 拆到 `structure/algorithm/mask/`，顶层再导出已删除。步骤 4 已把 `decode.py` 迁到 `structure/algorithm/decode/loop.py`，顶层再导出已删除。步骤 5–7（data/model/audit/analysis、trajectory_observation、cli）尚未开始。步骤 8 里测试镜像改路径和 `.importlinter` 按两柱重写仍未做。那些步骤只搬位置，不改数学。本计划不因为文档改了就授权后续搬迁。

| **事项** | **原因** | **验收** |
| --- | --- | --- |
| 会议方法的 CPU 合成检查 | 已完成。top1 确认未通过，oracle 空间全在槽 1，停止在合成设置上换形式 | 见运行记录 |
| observation 上的仿射 K | 研究计划把完整 K 与仅 bias 列为对照 | 合成检查有结论后再定 z→Az+b 的拟合数据 |
| 飞书规范正文 | 用户令牌过期时只读了索引和参考仓 | 下次能读取时核对冲突段落，不预先改写成“已符合全文” |
| 下一高信息实验的逐臂墙钟 | 研究计划「结论总表」已写入 `office-8b-traj-wall-v1`。`office_traj.py` 已逐臂记录 `wall_seconds`，缺失不写成 0。外层打印仍不进入比较 | 已跑。接受数 `direction` 为 positive，`wall_direction` 为 shorter，两区间都不含 0。见 [2026-11-04 Qwen3-8B trajectory墙钟](<运行记录 Run Record/2026-11-04 Qwen3-8B trajectory墙钟.md>)。次项 B `office-8b-traj-failset-v1` 已分析，见 [2026-11-04 Qwen3-8B trajectory失败子集](<运行记录 Run Record/2026-11-04 Qwen3-8B trajectory失败子集.md>) |
| 独立 AR 不一致的只读分类 | 研究计划「结论总表」`office-ar-inconsistency-v1`。已有行含 `ar_equal`、`output_sha256`、`argmax_passed`、`accepted`、`tau`。没有首分叉下标、共同前缀长度或两侧 token。八组与墙钟 jsonl 在运行记录所指目录 | 已完成。`classify_pair` 与 `classify_archive` 在 `office_traj.py`。旧行缺字段保持缺失。结论见开发记录 |
| 首分叉与 fp32 参照 | 研究计划「结论总表」`office-ar-first-fork-v1`。助手在 `office_traj_fork.py`，命令仍在 `office_traj.py` | 已跑。见 [2026-11-04 Qwen3-8B 首分叉](<运行记录 Run Record/2026-11-04 Qwen3-8B 首分叉.md>) |
| BF16 AR 自决定 | 研究计划「结论总表」`office-ar-bf16-determinism-v1`。助手在 `office_traj_determinism.py`。`prompt_tokenwise` 在 `loop.py` 的 `ar_generate` | 已跑。见 [2026-11-04 Qwen3-8B BF16自决定](<运行记录 Run Record/2026-11-04 Qwen3-8B BF16自决定.md>) |
| 墙钟按长度 | 研究计划「结论总表」`office-8b-traj-wall-by-length-v1`。助手在 `office_traj_length.py`。`draft_seconds` 与 `verify_seconds` 在 `loop.py` | 已跑。见 [2026-11-04 Qwen3-8B 墙钟按长度](<运行记录 Run Record/2026-11-04 Qwen3-8B 墙钟按长度.md>)。结构步骤 5 本次仍不搬迁 |
| observation 增益扫描 | 研究计划「结论总表」已写入 `office-8b-traj-g-scan-v1`。四臂只改 `spec.observation.gain`，g 为 0、0.25、0.5、0.75。逻辑在 `office_traj.py` 的 `--protocol gscan` | 已跑。三对相对 g=0 的接受数方向都是 negative。见 [2026-11-04 Qwen3-8B observation增益扫描](<运行记录 Run Record/2026-11-04 Qwen3-8B observation增益扫描.md>) |

不新增插件注册、通用校正框架或第二套测试运行器。

2026年11月3日，同节点预算的额外分支按研究计划里已写定的 `extra-branch-replace-v1` 执行。这次只比较接受数，不测速，不下载模型。运行产物留在源码树外。

# 三、RPipe 来源（B2 作废）
2026年11月1日 Enmao 决定：仓库不再内嵌 `src/rpipe/`，顶层不再有 `tools/`。Study 流程由 `spec_mtp.flow` 与 `spec_mtp.structure` 自己实现，目录以 https://github.com/DreamSoul-AI/RPipe 为结构模板持续对齐。此前“继续固定 v0.3.1”的默认假设作废。来源提交 `7a15c46b75c5eb951cccd60ed7b9fc3463db90f5` 只说明被吸收的那一版流程代码从哪来，不再是本仓的固定副本。

# 四、明确不做
不下载模型，不恢复范数矩阵、配对矩阵和 ESP 长矩阵，不恢复 BF16 相对独立 AR 的算子数值机理排查，不合并 `main`，不改 Actions 和仓库保护，不把运行产物提交进 Git。`office-ar-first-fork-v1`、`office-ar-bf16-determinism-v1`、`office-8b-traj-wall-by-length-v1` 和 `office-8b-traj-g-scan-v1` 的 GPU 都已经跑完，数字在对应运行记录。结构步骤 5 本次仍不搬迁。除此以外的 GPU 测量、GPU 测速和复跑仍须另行批准。

# 五、EMA／Holt／轨迹 Kalman 研究的待实施项

2026年11月20日用户将当前研究主线切回状态估计。以下区分当前输入实现与后续工程准备，不表示获准启动新 GPU。研究问题、数学、候选参数和停止条件以[唯一研究计划](<研究计划 Research Plan 去噪推测解码.md>)为准；Jason 的来源收益复核已归档，首阶段不启用其校准。

| **事项** | **所需合同** | **验收与状态** |
| --- | --- | --- |
| 状态与槽位映射核对 | 复用 `masks_for_tree`，以正式前缀位置、owner 深度、slot 编号确定真实预测距离；拒绝分支不改正式状态 | 已实现真实深度／仅槽位距离对照，覆盖纯读、多个提交、EOS／预算截断；CPU 结果见本批运行记录 |
| 最小输入状态模型 | 零阶、同锚点差分、Holt、固定 Q/R 的 Kalman 与对应冻结增益；共同初始化和目标读出 | 输入单位步已实现，数学退化、P 对称半正定、非有限值与稳态失败有检查；稀疏跳步留后续，不沿用历史类名代替验证 |
| 真实接受量输出 | 保留来源截断前计数，新研究另记最终保留 A、C、S、A/C、τ及额外前向 | 已追加 `retained-acceptance-v1` 统计与审计，包含末轮截断及部分失败；本版无额外前向，不回签旧 Run；E1薄接入已直接读取新协议，旧历史汇总保持原义 |
| 输出 hidden 观测账本 | 保存实际位置、forward 与可用时刻；bonus hidden 延迟；各方法共用读出映射 | 没有未来观测泄漏，不能用 token 查表值冒充 hidden；待输入合同完成后实施 |
| 有界离线读取与结果表 | 记录实际版本、模型身份、配置与输入，按 prompt 配对，不调用模型 | 校准与评测隔离、来源拒绝和缺失字段保持未知；E1已实现新协议配对／重审计及CSV／Markdown表；后续独立协议另登记 |

不增加新的通用运行器或第二套 decoder。新增实现按当前两柱规则及设计落位；真正改变接口或责任时先更新对应设计。输入项与E1配对结果导出已实现，输出账本仍待后续。

## 2026年11月20日有界开发批次

本批按约60分钟推进输入路线，隔离分支交付 Draft PR。先做实现与 CPU 合同验证，不启动 GPU、下载或测速。

1. 在 mask 算法目录新增独立输入轨迹策略：共享 prompt 均值、零初始速度，提供零阶、差分 EMA、Holt、固定 Q/R 二状态 Kalman、稳态冻结增益；保留历史 provider 行为。
2. 复用 `masks_for_tree`，默认预测距离为 owner 深度加一基 slot，另设仅 slot 距离作布局对照。读取不得吸收 draft 或改变正式状态；提交由已有 decoder 裁剪后逐 token 调用。
3. 追加裁剪后的 `retained_accepted` 和逐轮保留节点，不改旧 `accepted`。统一输出 A、C、S、A/C、τ，零调用的比值保持未定义。
4. 以手算递推、退化等价、纯读取、树位置和协方差性质作单元验证，再用受控解码检查拒绝、多个接受、bonus、EOS／预算；执行现有有界 CPU 计划。
5. 更新设计、开发／运行记录与入口，提交推送分支并创建面向 dev 的 Draft PR。输出 hidden 账本、稀疏跳步和真实模型收益留后续，不用本次 CPU 检查替代研究实验。

本批代码与验证入口：[输入轨迹状态实现](<开发记录 Development Record/2026-11-20 输入轨迹状态实现.md>)、[输入轨迹CPU验证](<运行记录 Run Record/2026-11-20 输入轨迹CPU验证.md>)。用户11月21日另行授权实验后，新增新协议的按 prompt 汇总、dev 选参与冻结 eval 薄入口，开始 E1；输出观测位置账本继续待办。本次授权和协议登记在研究计划，不接续旧 GPU 授权。
