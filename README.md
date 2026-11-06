# 去噪推测解码
研究免训练多 token 起草与验证：以 ESP 为参照，比较输入嵌入 EMA、速度外推和输出隐藏状态掩码。中文名称为“去噪推测解码”。仓库地址：[DreamSoul-AI/denoising-speculative-decoding](https://github.com/DreamSoul-AI/denoising-speculative-decoding)。

截至2026年10月21日的 8B 非官方 Table 1 独立复验已经完成，论文目标未复现。数字和原件在 [历史证据归位](<docs/运行记录 Run Record/2026-10-31 历史证据归位.md>)。提交 token 的验证前向 argmax 自检与独立 AR 一致率分别报告，τ 是提交数与解码调用数之比，不能据此认定实际速度收益。

当前收束已有 EMA 与具体 Kalman 方案：同8条 writing 的32/256预算均未支持EMA相对ESP的联合收益，256扩展候选已关闭，不自动续512或扫β；旧480条局部接受正证据保留。Kalman observation-only四场景受测合同已通过，多个旧配方收益为负或未确认。当前成果、范围与处置见唯一 [研究计划](<docs/研究计划 Research Plan 去噪推测解码.md>)。`k_affine_residual` 的 `postcommit-final-norm-v1` 仅为静态 K 合成工程夹具，不代表已有 Kalman 变体；160 对协议和真实模型收益未执行。

# 一、文档入口
| **文档** | **维护内容** |
| --- | --- |
| [协作约定](AGENTS.md) | 资料定位、适用规范及执行边界 |
| [通用调研 General Research 去噪推测解码](<docs/通用调研 General Research 去噪推测解码.md>) | 机制、方案、直接来源与当前判断 |
| [文献调研 Literature Review 去噪推测解码](<docs/文献调研 Literature Review 去噪推测解码.md>) | 已读文献、阅读深度和与本项目的关系 |
| [架构设计 Architecture Design 去噪推测解码](<docs/架构设计 Architecture Design 去噪推测解码.md>) | 职责、状态、流程、质量目标和取舍 |
| [代码结构设计 Code Structure Design 去噪推测解码](<docs/代码结构设计 Code Structure Design 去噪推测解码.md>) | 模块责任、接口、依赖和测试镜像 |
| [研究计划 Research Plan 去噪推测解码](<docs/研究计划 Research Plan 去噪推测解码.md>) | 当前成果与结论、重点问题、对照和结案规则 |
| [研究成果归档](<docs/研究成果归档 Research Archive 去噪推测解码.md>) | 已关闭或低后续投入价值的旧支线记录，不承接当前计划 |
| [开发计划 Development Plan 去噪推测解码](<docs/开发计划 Development Plan 去噪推测解码.md>) | 剩余工程、验收和待确认的来源策略 |
| [缺陷记录 Bug Report 去噪推测解码](<docs/缺陷记录 Bug Report 去噪推测解码.md>) | 尚未关闭的问题 |
| [开发记录 Development Record](<docs/开发记录 Development Record/INDEX.md>) | 代码与文档结构调整。目录见 [INDEX](<docs/开发记录 Development Record/INDEX.md>)。 |
| [运行记录 Run Record](<docs/运行记录 Run Record/INDEX.md>) | 实际执行、环境与结果。目录见 [INDEX](<docs/运行记录 Run Record/INDEX.md>)。 |

# 二、当前实现与使用边界
当前安装包为 `denoise-spec-mtp`，研究宿主 `spec_mtp` 按 [RPipe](https://github.com/DreamSoul-AI/RPipe) 的 `structure/` 与 `flow/` 两柱组织。宿主顶层保留包标记、模块入口和稳定控制台薄入口 `cli.py`。合同在 `structure/api/decode_api.py`，数据在 `structure/data/prompts.py`，模型在 `structure/model/hf_causal.py`，解码与 mask 在 `structure/algorithm/`，统计在 `structure/algorithm/counts.py`，观测与审计在 `structure/artifact/observation.py`。原数据、模型、审计和分析顶层模块已删除，调用方直接导入真实路径。

`python -m spec_mtp` / `spec-mtp` 由 `flow/cli.py` 提供 `decode/audit/compare`。Study 使用 `python -m studies._shared.flow.cli`，接入宿主 `flow/study_cli.py` 与默认六阶段。包初始化不加载模型或整套调度器。流程由本仓实现，不再内嵌 `src/rpipe/`。旧流程 API facade 已删除，构建直接使用所属层工厂。九个共用诊断再导出已清理，消费者直接导入真实模块。已登记的 Study 顶层遗留入口清单为空，其余结构与测试镜像差距见代码结构设计与开发计划。

当前主线是按已验证范围完成 Kalman/EMA 方案与旧 Run 结案；工程只处理直接阻塞这些结论或来源保全的具体缺口，不重复派发已完成合同，不等待新算法。正确性、优化和性能验证使用 CUDA，先确认可用并记录实际设备；CPU 只做快速纯合同单元测试。GPU 满载时等待或缩小规模，模型下载不在范围。本任务普通推送现有工作分支，经 Draft PR106 面向 dev 评审，不修改 main/dev 或绕过检查。

第三方 ESP 来源树及 `ORIGIN.json` 保留，不能把它称为作者官方实现。原始 SpecBench 数据保留于 `datasets/spec_bench/question.jsonl`。

科学 CLI 的 `decode` 必须显式提供 `--model-sources <JSON>`：清单位于模型目录外，以相对路径映射该完整、自包含的本地模型目录内全部文件的 SHA256，不缺项、不含目录外成员。输出也放在模型根外；解析后逃逸模型根的符号链接会被拒绝，不直接承诺兼容引用外部 blobs 的 Hub 快照。载模前、运行头及结束记录写出前分块读取核验实际文件；漂移时保留部分观测，不写完成末行。运行头另记实际模型 config、backend、device、dtype 和能力，调用者原 `sources` 标签保持原义。审计带模型来源的新记录时使用 `audit <raw> --sources <标签JSON> --model-sources <原模型清单JSON>`；缺少独立模型清单不能记来源完整。旧无模型来源字段的记录仍只比较原标签，不代表重新验证实际模型。`audit/compare` 不加载模型或重新生成观测。

旧逐条运行证据、日志、图和报告保留在固定历史/外部运行根，未恢复进 Git。历史读端测试从固定 Git 提交 `b8b20b5ada7fd860894114cbf224ce749af430cf` 按需提取原始 blob 到测试临时目录，核对对象哈希与写入字节；缺少 Git 对象直接失败，不下载或伪造原件。当前分析器和校验门读取这些不可变输入，不导入或执行归档源码，不修改原哈希与断言；相关15个参数项已恢复到 CPU 计划，不再排除。此验证只复核保存的历史观测与读端合同，不重新运行历史模型，也不代表全部历史结果已独立复验。

新建 Study 运行使用 `python -m studies._shared.flow.cli run studies/<name> --run-root <外部Study运行根>`，也可先 `make studies/<name> --run-root <外部Study运行根>`，再 `launch <外部Study运行根>`。使用 --run-root 时 Study 路径紧随命令。外部根保存声明与源码白名单快照、study_source.json、runs/shared/docs、index、锁与日志，源码 Study 保持只读；状态、日志、process 和 report 后续直接传该外部根。生成的 PowerShell 启动入口同样走 shared CLI，使用 shared 控制台并传播退出码；调度worker固定当前宿主导入根，并在写运行态前验证来源。重复 launch 跳过已完成Run，部分完成时只派发pending；这是调度恢复，不是checkpoint/RNG恢复。失败默认不自动重试，损坏或与index/config不一致的jobs会拒绝，调整GPU分配或轮次需显式remake。已绑定且逐文件摘要相同的根可继续使用，原 Study 或快照漂移会失败，须用新根与研究协议处理，不覆盖旧运行。原 Study 中的绝对导入仍需要原 checkout；静态快照限单成员16MiB、合计32MiB，不复制权重、旧 Run 或未声明生成物。

launch在派发或等待异常后停止新group与retry，只回收本次成功取得句柄的直接worker：逐一terminate，沿原5秒wait、超时kill及5秒wait尝试退出。再次取消或回收错误按顺序保留在原异常cleanup_errors，清理不依赖console输出，原始异常与既有Result不改。该直接worker的wait返回不代表后代或所有OS I/O已停止；全Run硬超时仍未实现，未知状态不自动重试。

中断组在全部直接child清理尝试后，将退出事实发布为 `scripts/launch-attempts/<attempt_id>/receipt.json`。index/config SHA在该组任何Popen前取得，回执区分terminate/kill请求与wait实际确认退出；未确认returncode不补零，后代、未返回Popen与通用writer停写保持unknown。它不是科学Result或整个launch的停写证明。仅成功发布后原异常带 `cleanup_receipt_path`；失败附cleanup_errors，首错及旧文件保持。正常完成不创建回执。每件最多64KiB，partial.json写入/fsync后发布；别名、碰撞及失败余件不自动修复或覆盖。回执属于运行控制制品，不计入诊断固定份额；总量/自动保留期及OS硬时限未实现。

离线报告使用 `python -m studies._shared.flow.cli report <Study> --out <外部新文件.md>`，输出路径必填，拒绝已有文件、输入目录和源码 checkout；报告检查不回写旧 process 或 gate。`output_space_extension` 的已保存槽位诊断迁入 `python -m studies.output_space_extension.flow.process.run --raw <保存张量目录> --execution <原运行矩阵.json> --repository <本地仓库> --commit <完整提交> --execution-git-sha256 <Git原件SHA256> --device cuda:0 --output-root <新外部目录>`。它只读原件并在CUDA分析保存张量，不载入模型；实际运行矩阵须匹配旧报告哈希，与固定 Git 原件分别记录字节身份并核对 JSON 内容。缺失或漂移直接失败，不自动修换行或重签旧来源。历史读件需要支持 `--no-lazy-fetch` 的本地 Git，当前验证版本为 `2.55.0.windows.3`，不支持时直接失败，不降级联网。followup 读端同样迁入 `studies.output_space_followup.flow.process.run`，参数与槽位读端一致，保存张量实际在 CUDA 分析并登记设备。

smoke 当前微型工程验证使用 `python -m studies.smoke.flow.execute.run --plan <当前计划> --plan-sha256 <实际SHA> --version <新工程版本> --device cuda:0 --output-root <新外部根>`，保留三种variant、四条输入、T0/0.7/1.0和32-token预算。T0完整提交链argmax为门，采样AR一致仅作诊断；`--compare <完整CUDA报告> --compare-sha256 <实际SHA>`要求同目录probe_complete/header/cells与实际SHA闭合，再核协议、模型、输入、设备与随机身份，不拿旧CPU报告冒充CUDA基线。历史CPU图使用 `python -m studies.smoke.flow.write.run --input <原delivery_validation.json> --input-sha256 <实际SHA> --output-root <新外部根>`，只读已保存计数，保留934dac3的CPU身份；可另给固定Git身份核原件，不加载模型。

extension的三个保存证据入口统一为 `python -m studies.output_space_extension.flow.process.run saved <numerics|kernels|operators> --inputs <角色索引JSON> --inputs-sha256 <实际SHA> --device cuda:0 --output-root <新外部根>`。索引保留各报告自己的原tool/plan/common及运行矩阵绑定字段，固定Git成员与运行件分别验SHA；原件缺失严格拒绝，不执行归档代码。CUDA margin与历史CPU参考分开记录；旧模型加载、实时kernel profiler和已停止的BF16机理命令均不提供执行入口。

small_matrix 的分析统一使用 `python -m studies.output_space_small_matrix.flow.process.run <transfers|cross-prompt|trajectories|windows> --raw <原输入> --repository <本地Git仓库> --commit <完整提交> --report-git-sha256 <固定报告SHA> --output-root <新外部目录>`。trajectories 的原输入为原运行 execution JSON，只读 JSON，不接受设备参数。其余三种的原输入为原报告与张量目录，必需 `--device cuda:0`。transfers 另需 `--original <原observer目录> --observer-git-sha256 <固定observer报告SHA>`，windows 另需 `--execution <原运行矩阵JSON> --execution-git-sha256 <固定矩阵SHA>`。实际运行原件、固定 Git 原件与张量分别绑定，新工具来源另记，不运行归档源码或模型。优化模式 `python -O` 会明确拒绝，不能跳过科学断言。原 helper 身份缺失保持未知，不补成当前 helper。

两套历史图使用 `python -m studies.output_space_small_matrix.flow.write.run <acceptance|windows> --input <原报告JSON> --repository <本地Git仓库> --commit <完整提交> --input-git-sha256 <固定报告SHA> --output-root <新外部目录>`，windows 另需 `--scale <原scale报告JSON> --scale-git-sha256 <固定scale报告SHA>`。只渲染已保存的标量，不重算模型或统计。归档 window/scale 的已知后补附注单独登记，共有字段仍须全部相等，原 observer 字节必须匹配原 scale binding，不把归档附注混进原运行件。图片保留原数值、颜色、坐标与标题，完成记录绑定实际输入、固定 Git 来源、当前绘图源码和 PNG 摘要，失败只保留 partial，不回写 Study/docs 或旧 Run。

prerequisite 的 q171 图使用 `python -m studies.output_space_prerequisite.flow.write.run --input <原executionJSON> --repository <本地Git仓库> --commit <完整提交> --input-git-sha256 <固定报告SHA> --output-root <新外部目录>`，同样只读固定历史标量，保留三臂与原轴范围规则，独占新输出并核 PNG 与来源。

input-mask 的 q171 图使用同一组显式参数运行 `python -m studies.input_mask_baseline.flow.write.run`，输入为其原 `v4_execution.json`。它保留两臂 accepted 柱及保存的 cosine sums/counts 均值，零计数不画对应线，使用该报告自己的来源身份。

五方法配对保存报告使用 `python -m studies.qwen3_4b_bf16_five_methods.flow.process.run --input <paired_evidence_v1.json> --repository <本地Git仓库> --commit <完整提交> --input-git-sha256 <固定报告SHA> --output-root <新外部目录>`，绘图改为同 Study 的 `flow.write.run` 并沿用这些参数。process 的可选 `--original-study <原Run所在Study>` 只核报告已记录的六类文件摘要；这是 480×11 保存计数与 WTL 复核，明确 `native_run_revalidated=false`，不重新验收模型、token 或旧 Run。当前消费者直接导入实际 readout/plots，不再动态装载被删除的顶层脚本。

ESP 四格及 Qwen 旧协议分别使用 `python -m studies.esp_table1_bf16_full_argmax.flow.process.run`、`python -m studies.esp_table1_bf16_qwen_protocol.flow.process.run`，必需 `--study-dir <实际输入Study> --execution-root <原执行源码与引用所在根> --output-root <新外部目录>`。独立复验的 `studies.esp_table1_nonofficial_revalidation.flow.process.run` 另需 `--baseline-dir <原基线证据/源码目录> --baseline-runs <原基线Run目录>`，不能拿当前 helper 冒充原聚合器。原生来源锁仍按原合同拒绝漂移，缺少兼容执行源码时不能声称重聚合成功。图可单独用 full_argmax 的 `flow.write.run --report <已保存四格证据JSON> --report-sha256 <实际摘要> --output-root <新外部目录>`。派生报告、图及输入绑定写新根，正式完整证据在绘图与来源复核后最后发布，失败只留 partial。

full 的完整 AR 读端使用 `python -m studies.esp_table1_bf16_full.flow.process.run --study-dir <实际输入Study> --execution-root <原执行源码根> --output-root <新外部目录>`，保留 BC10 auxiliary 和逐格 AR exact 判据。四格已观测与质量通过分别报告，AR 不一致的负结果可完整读出；原运行件、输入或原源缺失仍拒绝，不改用 argmax 门或当前 helper 补签来源。

4B 的 recheck、recheck_v2、holdout、input_control 分别用 `python -m studies.<Study>.flow.process.run --study-dir <旧Study输入> --execution-root <原运行根> --execution <保存execution JSON> --repository <本地Git> --commit <完整提交> --execution-git-sha256 <Git原件SHA256> --output-root <新外部目录>`，共用配对公式但保留各自输入和质量条件，只复核保存计数，不重新验收 native gate。holdout 与 recheck_v2 的 `flow.write.run --bindings <JSON> --repository <本地Git> --output-root <新外部目录>` 生成中文报告和配对图；bindings 的角色值为 `{"path":"实际输入","commit":"完整提交","sha256":"Git原件SHA256"}`，execution 必需，其余历史引用未提供时不生成失效链接。

cross_task 采样路径分析使用 `python -m studies.output_space_cross_task.flow.process.run <first-delta|sampling-expansion|continuation> --inputs <角色输入JSON> --inputs-sha256 <实际SHA256> --output-root <新外部目录>`。输入索引固定 schema=1、kind、repository、commit、reports、files 与 producer_sources，报告分别给实际路径和固定 Git 身份，原工具/计划与张量/选择集分别绑定。后两种数值审计另需 `--device cuda:0`，first-delta 只读保存 JSON。历史 acceptance 图使用该 Study 的 `flow.write.run --execution <JSON> --execution-git-sha256 <Git原件SHA256> --canonical <JSON> --canonical-git-sha256 <Git原件SHA256> --repository <本地Git> --commit <完整提交> --output-root <新外部目录>`，保留四条主图与单独 q241 对照，未运行模型。

cross_task 的其余读端由同一 process 入口分组：`native <export|hypotheses|mre|cross-phi|canonical>` 承接原生证据、假设判读与 canonical 审计，`state <local-extension|state-factors|sampling-owner|sampled-depth>` 承接状态和捕获深度。各组用 `--help` 查看显式角色输入及输出要求；数值命令要求实际可用的 CUDA 设备，MRE 为纯 JSON 工程夹具。历史成员可分别提供完整 commit，缺省沿用索引 commit，原路径按固定角色核对；归档报告与原 tool/plan 分别锁定各自字节。当前 CUDA 概率/margin 与保存的 CPU 参考分开标设备和验证范围。

paired_revalidation 的保存报告用 `python -m studies.qwen3_4b_bf16_paired_revalidation.flow.process.run published --input <paired报告> --prior <five-methods报告> --repository <本地Git> --commit <完整提交> --input-git-sha256 <原件SHA> --prior-git-sha256 <原件SHA> --output-root <新外部根>`。norm_scan 的已归档五端点复用证明用其 `flow.process.run published --input <reused报告> --directions <方向证据> --repository <本地Git> --commit <完整提交> --input-git-sha256 <原件SHA> --directions-git-sha256 <原件SHA> --output-root <新外部根>`。两者可另给 `--execution-root <原执行源码根>` 核保存文件摘要，范围仍为 counter-only，不能当 native 重验或八个新 norm 格已完成。

上述两组的原生旧件读出由同一 process 的 `native --help` 查看，要求显式旧 Study/数据根与新输出根；旧来源锁不兼容仍拒绝。norm 的 `reused --data-root <原执行根> --output-root <新根>` 保留只核复用端点的范围。geometry 的独立读件子进程使用该模块入口与独占子输出，不再按旧顶层分析脚本启动。

四个 output_space_4b 与 ESP preflight/numerics/reduction_check/shape_check 共用 untimed/CUDA 适配，在 recipe 的原来源门通过后注册，导入 recipe 不改注册表或 CUBLAS 配置。ESP 特有 numerics/shape 仍在显式注册时依次覆盖。计时字段的零值仍是禁用占位。旧 Run 与 source lock 保留原身份；声明移除已迁脚本后，新执行仍须按新 version/Run 冻结当前来源。

五方法配对执行与普通/recovery 方向执行使用共享 algorithm/geometry/validation/recipe，Study 根和配对原件根显式绑定。方向公式与原 CPU 随机序列保持，普通版资源不足立即拒绝，recovery 只在既定准入边界等待并把记录写入实际 Run assets，不再写源码根 `.tmp`。配置、旧来源表与执行材料缺失仍拒绝，不因模块归并恢复暂停的研究矩阵。

ESP 后端和局部 Linear 控制统一在 `_shared/structure/esp_controls.py`，三份相同 numerics 与 shape 不再独立维护副本。recipe 在原准入后选择 NumericsEval 或 ShapeEval，成功和异常时恢复原后端状态与局部 forward；微型 CUDA 只验证作用域与 shape 工程合同，不恢复已停止的算子机理研究。

ESP full/Qwen 的 numeric policy、full_argmax 的独立 argmax 观察和非官方复验也使用该共享模块。四个 recipe 通过原准入后显式选择 policy/argmax/revalidation，导入不注册；运行前后核快照，非官方的原模型摘要、480 条输入、计数与逐 prompt 资源门另行保持。audit 独占创建，错误时恢复 selector/generate 并移除 hook，旧审计不追加新执行。

norm、paired repeat 与 continuation 也通过共享部件装配，运行根经 study_source 绑定定位原科学身份，配对矩阵不取临时目录名作 experiment。来源、固定 Git 计划及准入策略分别保留，资源等待后复核再载模；同一 decoder 的诊断逐 prompt 保存独立快照，content 起点随当前 prompt 读取。八格、七格与十一格仍是各自暂停的协议，不因入口可用自动派发。

容量校准入口为 `python -m studies.qwen3_4b_bf16_bias_norm_scan.flow.execute.capacity --model-root <本地模型> --model-sources <完整模型清单> --prompts <冻结manifest> --prompts-sha256 <SHA> --study <冻结study.yaml> --study-sha256 <SHA> --plan <计划> --plan-sha256 <SHA> --version <新版本> --device cuda:0 --output-root <新外部根>`。入口保持 q295/1500 输入/256 输出/template2，要求非优化 Python、真实 CUDA、完整来源校验与模型关闭后最后发布完成标记，不联网取权重。当前只用 tiny CUDA 验工程，未重新测量真实 4B 容量或旧主机进程树峰值。准入共用 resource_admission 的显式 capacity/observer，保留各协议非抢占及 RAM 政策。

trajectory 真实模型专用入口为 `python -m studies.trajectory_observation.flow.execute.office <qwen8b|tree|traj> --help`、`python -m studies.trajectory_observation.flow.execute.fresh_replace --help`、`python -m studies.trajectory_observation.flow.execute.fair_branch --help` 与 `python -m studies.trajectory_observation.flow.execute.preforward_signal --help`。它们要求显式本地模型目录、目录外完整模型SHA清单、本地JSONL、计划文件及其SHA、新外部输出；fair_branch以 `--split specbench|dolly --input <本地JSONL>` 保留原337–464/前128切片，preforward保留SpecBench 81–208及128条校准模板，不再猜模型目录或联网取数据。共用本地输入准入，只读取已解析绑定路径。preforward的summary/fit可留作partial，只有收尾复核及关闭后写出的protocol_complete标记表示运行完整；旧gate字段不产生新研究授权。

bonus-aligned/output-hidden 两个专用协议使用 `python -m studies.trajectory_observation.flow.execute.hybrid_gain <bonus-aligned|output-hidden> --help`，同样必需本地模型目录、完整模型清单、SpecBench JSONL、计划 SHA、CUDA 设备和新外部输出。前者保留 465–560 切片与四臂，后者保留 209–336 切片、32 条 holdout 与六臂，校准各为 128 条模板。共同收集、落盘与会话编排归同一 flow，structure 保留纯校准与统计。逐臂核完整 qid 与 argmax 审计，收尾核来源并关闭模型后才发布 protocol_complete，旧根不续写。

合成 pilot 使用 `python -m studies.trajectory_observation.flow.execute.pilot --config <完整本地JSON> --plan <研究计划> --plan-sha256 <计划SHA> --device cuda:0 --output <新外部目录>`，模式、seeds 及预算由显式冻结 config 给出，互斥模式拒绝，不因 DEFAULT 存在就自动启动 1500 步旧训练。训练、模型前向及拟合在 CUDA，CPU 只承担离散样本准备和序列化。九个旧 structure 命令已归 flow，不留反向 wrapper。当前工程验证使用微型模型、oracle、拟合与入口合同，没有运行默认训练、真实权重或旧研究矩阵。pilot两个producer从train_model返回session后保护所有权，保本函数首错/取消并关闭一次；train_model构造session后的实际tied-storage准入也已保护，拒绝、属性错误或取消均关闭一次并保首错，成功返回不关闭。正常模型固定请求tied embeddings，未发现自然untied路径；受控storage反例不改变borrowed native所有权，也不证明借用显存已释放。HFCausalLM取得hf后的eval/config/vocab/max_positions初始化现有失败保护，关闭半成品session一次、保首错与取消并附close次错；owned仅断开自身模型引用，borrowed保native。pilot删去构造器外重复native.eval，成功路径由构造器调用恰一次；loader返回前、原生训练故障及外部traceback引用不在即时释放保证内。

checkpoint/RNG的独立合同见架构设计CR0–CR5。native `TrainAlgorithm` 将读取、准入与应用分开：先核合同、模型状态、数据游标和RNG，再改活跃训练状态；实际bundle载入前后核SHA，来源进入恢复摘要。`algorithm.config.resume_from` 显式指定文件时，缺失即失败；默认latest不存在仍可新建训练。Data子状态使用v2，冻结sampler/loader generator及其共享关系；旧v1不能自动当exact恢复，原件保留。无完整training_state的旧checkpoint继续明确警告partial。

微型CUDA回归已覆盖零worker标准DataLoader、shuffle、随机transform、Dropout、梯度累积，以及第2步保存后新进程恢复至同一6步预算；逐批样本、模型/优化器/调度器、tracker数值、游标与后续RNG均与连续执行比较。该范围不包含HF Trainer、多worker、分布式、真实模型或跨设备等价；数据身份按元数据核对，尚无数据内容哈希保证。

epoch预算与早停另有独立进程CUDA对照，固定评估分数只用于检验stall与停止状态。已停止或任一预算上限已尽的恢复只还原状态，不再取batch、评估或写checkpoint/tracker。checkpoint以同目录独占临时文件同步后最后替换bundle；提交前错误保留旧bundle，提交后诊断清理失败不否定已提交状态。未发布stage及其别名拒绝读入；只有旧分片目录时，写入前要求显式迁移，不自动补签或覆盖旧原件。

诊断日志以 `assets/logs/events.jsonl` 为稳定锚点，每段到4MiB后另建 `events.000001.jsonl` 等递增段，旧字节不覆盖、不删除。状态与logs命令按全段读取，报告日志链接指末段；旧 `run.log` 保持只读兼容。JSONL逐条有界读取，状态摘要流式归约；缺段、链接别名和损坏尾记录拒绝，完整输出的UTC排序仍占用选中记录内存。宿主托管Logger隔离自身诊断故障，Flow与System复用同一实例；默认直接API仍严格报错，业务Tracker/checkpoint/Result错误不吞。失败操作计数不等于丢失条数，stderr仅安全类别与完整性未知。新空托管日志以events.policy.json冻结64MiB累计配额（含policy与全部诊断段）和实际UTC起点后的7天追加窗口；重开不续期，旧非空/legacy无policy不补签，到限停止追加并报告本实例suppressed_records。14个Study独立创建点已显式采用managed=True，14项构造绑定与配对报告、CUDA smoke故障入口分别验收；完整协议执行范围按实际回执区分。五个保存件绘图flow在owned Figure释放成功后复核输入/来源并发布新completion，保留原异常及figure_cleanup_error。office/fair/fresh/preforward四个模型flow也按本函数首错关闭一次，关闭后复核来源；office/fair/preforward的完成件复用既有publisher，fresh已写summary保留为数据、不新增完成标记。旧文件不删除。物理保留期、跨Run全盘总量/到期清理与OSIO硬超时仍是缺口；最低Python3.10及全部依赖组合尚未实机验证。

显式新外置Study可在study.yaml顶层声明 `diagnostic_budget_bytes: 1048576`：make在派发前为有限Run集合固定分配，只有全部份额与配置/index绑定完成才发布ready。每Study预留8192控制字节、每Run预留16384，剩余平均分配（每Run最少4096、最多64MiB）；例如两Run各503808字节，含policy及全部JSONL。控制单件最多4096，固定stage失败后拒绝重入。声明只适用于新 `--run-root`，不收编旧根；无声明沿用原行为。重开不续7天窗口，失败/取消/崩溃/封存/purge不退款或挪用份额，源或绑定漂移及未完成初始化严格拒绝。它约束协议具名诊断文件的逻辑字节，不保证文件系统块、业务输出、外部写入或硬I/O超时；历史根与未知成员保持拒绝处置。

只读盘点使用 `python -m studies._shared.flow.cli logs <外部Study运行根> --inventory`，输出每个index Run的精确诊断家族JSON；重复run_id或规范Run目录拒绝，避免同一家族重复计量或同名串身份。不加flag仍输出原事件文本。`bytes`计已知JSONL、policy和legacy文本的实际总字节，`capture_bytes`仅计受采集配额约束的JSONL与policy，不包含legacy。无policy保`unbound`，未知配额/起点/期限为`null`；已知追加状态为`within_limits`、`expired`或`capacity`，全部`disposition=retain`，到期不表示任务停止或可删。列举、身份及metadata读取失败不填假零，准入/读取/构造失败返回2、stdout为空，stderr只给安全异常类别。入口不建来源锁或Run，不读诊断正文、不改原字节；不承诺stdout故障时的原子输出，也不提供物理清理、跨Run配额或硬IO超时。

托管Logger构造/追加对进程内写锁只作一次非阻塞准入；锁忙沿既有诊断失败计数与安全应急提示处理，业务继续。严格Logger仍等待并序列化写入。竞争可能形成诊断缺口；构造被拒的实例不会自动补建日志根，后续可保持unbound。该锁准入合同只消除排队，前置callback、当前writer和console仍同步，不能据此证明清理完成或OS I/O已停止；显式owned处置与固定诊断预留各按其独立合同执行。

新、未执行Run可在首次Logger前显式接管诊断家族：`python -m studies._shared.flow.cli logs <外部Study> --retention prepare --run-id <ID>`；已有logs目录、Result、旧根或不可信路径拒绝。随后正常运行沿用原入口。需要处置时先 `logs <外部Study> --retention seal --run-id <ID>` 撤销当前协议内新旧Logger的追加权并取得seal SHA，再由运行管理者审查引用并显式调用 `logs <外部Study> --retention purge --run-id <ID> --seal-sha256 <SHA>`。只清除封存列举且复核相同的JSONL/分段与policy，不删除目录、legacy、未知成员、模型、数据或Result。

`assets/.diagnostic-capture.json`、`assets/.diagnostic-capture.lock`、`assets/.diagnostic-capture.seal.json`、`assets/.diagnostic-capture.purge.json` 为永久控制，不清除或手动重建；原生OS锁只用本机Windows/POSIX后端，无软锁替代。pending/closed/部分失败保持封闭，不自动重试、补建或恢复捕获；文件锁释放不等于业务进程退出，封存也不表示科学完成或日志完整。仅保证当前协作协议内的writer，不覆盖旧版本、外部直接写入或非协作替换；不提供自动到期回收、旧Run收编、动态跨Run退款/重分或OS级硬I/O超时；新Study固定预留见前述声明。默认日志及只读inventory保持原用途，inventory统计诊断内容，不把控制journal当科学完成凭证。

# 三、开发与验证
工作分支经 PR 面向 `dev`，完成检查并按当次明确授权集成，不自动发布。当前 HEAD 的本地结果、历史红项和未覆盖范围记在 PR 正文，不用旧提交结果代替。

共用解码诊断使用 `python -m studies._shared.flow.diagnostics <fingerprint|profile> --config <完整本地JSON> --plan <当前计划> --plan-sha256 <计划SHA> --device cuda:0 --output <新外部目录>`。fingerprint 配置为 `{"variants": null}` 时保留原完整预设矩阵，也可显式给预设列表；profile 要求完整模型、prompt、token、线程、重复与 variant 配置，字段及默认值由该入口的 PROFILE_DEFAULTS 定义。来源、配置、计划与设备绑定到新输出，模型关闭及收尾验证后才写完整标记。fingerprint 的 `--compare <已有报告>` 必须具有匹配工作负载/设备身份及完整内容摘要，旧缺身份报告不当作已验证基线。T0 审计与采样诊断分开，不推广采样保证。

profile 在 CUDA 工作负载上使用 cProfile，component 是主机侧 self-time，保留的 decoder host-clock token rates 也不是端到端 CUDA 吞吐或 kernel 测量。旧 CPU profile 不按新设备身份重解释。本轮只跑微型单 cell 与 profile 合同，未执行完整 fingerprint 矩阵或性能研究。

本机快速单元执行 `python tests/run.py --plan cpu-quick --output-root <外部新验证目录>`，CUDA 工程正确性执行 `python tests/run.py --plan cuda-validation --output-root <另一外部新验证目录>`。CUDA计划不可用即失败，回执保存实际设备、精度和显存；两计划分开报告。原 cpu-pr 计划保留既有用途，本轮不以其全量CPU结果替代CUDA验证。静态清单先验证必需项和精确排除，逐参数 setup/call/teardown 写独立记录，缺收集、缺终态、失败、跳过和证据写失败均非零。预算随所选计划：cpu-quick为120秒、cuda-validation为300秒、cpu-pr为420秒，在测试边界检查；子进程测试各有有限超时，单个本地计算不支持强制中断。运行产物不进 Git。

每次计划在外部输出根冻结有效声明/计划 `manifest.json`、`environment.json`、`selection.json` 和逐参数phase/result，生成中文 `report.md`，最后发布绑定报告及证据SHA的 `summary.json`。报告从落盘记录汇总执行、测量、质量、维度、预算、成本和未验证范围；没有正式summary或其退出码非零就不是成功。缺收集、准入失败、缺阶段及报告写失败均保留失败/未知，不用当前源码清单补回未记录的事实。只读 `render_report(root)` 可核对重建，无额外CLI；旧输入与报告不覆盖，原异常载荷不展入正文。

native shared CLI 的 smoke 测试按架构 NS0–NS4 使用独立 CUDA 验证身份。测试复制原声明到外部新根，显式绑定 system/eval 设备，微型模型在 CUDA 初始化并记录实际输入、权重摘要、dtype/backend和运行种子；CPU只准备固定离散token及做纯IO拒绝检查。T0以完整提交链argmax验收，独立AR与接受数为诊断，旧CPU的27/31/8不作CUDA基线。原CPU Study声明、旧Run和历史图保持原义，不因测试迁移重签。

运行前先将TEMP/TMP设为当次指定的外部任务目录，再将 `PYTHONPYCACHEPREFIX` 设为该目录下可写的缓存子目录；正常启动允许Python读写字节码缓存，所有制品仍留在任务目录。不要同时给全新空前缀和 `PYTHONDONTWRITEBYTECODE=1` 或给每个worker加 `-B`：这会绕开安装目录已有缓存并让独立进程反复编译。PowerShell可用 `Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue` 移除禁写设置；用 `python -c "import sys; print(sys.dont_write_bytecode, sys.pycache_prefix)"` 核实际子解释器配置。原native launch的90秒门限、完整断言和失败退出保持；缓存行为见[Python官方说明](https://docs.python.org/3.13/library/sys.html#sys.pycache_prefix)。

native CLI 验证使用独立测试计划：`python tests/run.py --plan cuda-native-smoke --output-root <另一外部新验证目录>`，单卡、300秒预算，包含11个原smoke数值入口路径及1个报告launch，共12项；不并入原cuda-validation预算。两组GPU验证依次执行，实际设备与来源保存到各自回执，纯配置/来源拒绝仍由cpu-quick选择。该计划不运行真实权重或旧研究矩阵。

typed request的轻量合同归API测试镜像；audit与科学CLI用例依赖的合成解码数值，以及decode生成/AR/选择器、mask、KV、tree/attention合同，使用cuda-validation，纯参数/路径/保存表示检查留CPU快速计划。共享夹具按显式设备创建并核embedding与forward，真实tiny模型由同一工厂恢复RNG/设备并关闭owned资源。日志区分选择器张量、刻意零forward与实际forward；局部CUDA Generator须保持全局状态。科学CLI的728参数本地模型验证使用测试内委托loader移到CUDA，不代表生产CLI增加了设备选项；它不替代真实模型研究或跨设备等价验收。Hypothesis保max_examples=80的原属性规模，单个属性仍算一个pytest参数结果，不加计内部示例。

Run和Study质量门分别从 `studies._shared.flow.summarize.run`、`studies._shared.flow.process.run` 使用 `run_gate`、`study_gate`；保存件验证从 `studies._shared.structure.validation` 使用，不再通过旧 `flow.checks` 转导出。删除当前兼容文件会改变新来源摘要，旧来源锁与Run仍按原身份解释，不补签或删成员绕过校验。

可选CSV Logger按首个有效header固定列序，同键重排仍写入正确列；追加前核header、schema及末尾换行，目标被改换/清空/移除或输入损坏时拒绝，旧行不重写。首次零字节文件可写一个header，新实例沿用有效header。stdout-only仍输出动态指标；它与collect正式CSV、Tracker及诊断JSONL各守原职责，不提供原子追加或并发写保证。alignment保存计数的七参数回归在真实CUDA计算结果上核设备，加入cuda-validation，纯JSON/来源拒绝保持CPU快测。

测量类（numeric或metrics/samples粒度）须在声明给出唯一 `metric_specs`（name/unit/statistic），通过 `record_property('metrics', [...])` 提交同名有限实数、样本/暖机规模、数据规模、环境、测量边界及inline raw_samples。记录中的 `metrics` 保留原user_properties映射，测量列表位于 `metrics.metrics`，原设备诊断可并列。空/缺采集、重复属性、阶段不完整或写失败均阻断；采集后的可变对象不能改写已保存call事实。每参数的 `*.result.json` 在完整阶段后定稿，numeric为 `status=measured`、`verdict=null`，不算质量通过；summary分别记录measurement_completed与quality_passed，两者独立，带有效测量的分类用例可同时计入。当前只支持inline样本，不建立附件或基准框架；微型CUDA内容测量不提供性能结论。

Study声明的 `provenance.project_plan.path` 只有在无commit/hash的当前导航角色时跟随唯一中文研究计划；带commit/path/sha的固定组及 `docs/source_refs.json` 仍按原Git身份读取。historical_sections中的anchor按原版本索引解释，不指向当前中文正文锚点。导航维护不修改历史source_path/commit/blob/hash、旧Run或原科学输入；原件缺失仍拒绝。

本任务不调整现有 Actions、CI/CD 或设置。微型CUDA工程验证只覆盖既定合成夹具，不能替代真实模型研究或采样分布保证；已有真实权重EMA比较另从研究计划与对应运行记录定位。已有 CUDA 环境保持原样，不下载模型或覆盖 torch。

Python配置对象直接从 `spec_mtp.structure.<algorithm|data|model|system>.config` 导入；`RunConfig`/`ExperimentConfig`从 `spec_mtp.structure.control.run_config` 导入，Study构建 `expand_study` 从 `spec_mtp.structure.make.write` 导入。宿主普通父包只标识职责，不再通过再导出加载工厂或运行层；实际符号入口见代码结构设计。YAML空文档仍为空配置，非mapping的false/0/序列等无效输入明确拒绝；写入前失败保留旧配置。

## 研发维护与文档现状

资源准入在普通与优化Python模式下均显式保留完整容量、argmax审计、BF16和单次方向观察拒绝门，原阈值与等待政策保持。配对报告在来源复核、图资源释放及回执构造全部成功后才发布完成凭证；清理失败保留原始异常。资源与报告测试分别镜像 `studies/_shared/structure/` 和 `studies/_shared/flow/write/`，按真实调用层级与成本登记，CPU/Agg验证与CUDA验证分别记录。

Python 3.10+、PyTorch 与固定 Transformers 条件见 [pyproject.toml](pyproject.toml)，当前研究宿主为 `spec_mtp`。目录入口：[产品或库源码](src/spec_mtp/)、[研究声明与适配](studies/)、[测试](tests/)、[当前文档](docs/)。运行与测试按本 README、相关设计和实际配置选择，不把历史命令视为新的实验授权。

文档职责按 AGENTS.md 维护。通用调研不代替文献调研，研究计划不代替开发计划。历史测量在运行记录，结构调整在开发记录。

宿主CPU安全合同运行 `python -m pytest tests/spec_mtp`，真实CUDA合同运行 `python -m pytest tests/cuda/spec_mtp`；两根分别完整镜像源码相对路径，`python -m pytest tests` 仍收集两者。GPU入口先核CUDA可用并记录实际设备，保留设备、RNG、模型与张量断言，不提供无GPU回退。未知方法拒绝等纯结构guard留在CPU根，现有workflow命令不变，也不通过收集过滤或skip选择资源。

登记1063函数，cpu-pr/cpu-quick/cuda-validation/cuda-native-smoke分别907/545/144/12函数，计划ID跟随实际测试位置、target仍指生产职责。宿主CPU安全根1183参数、CUDA根285参数；本轮新增7个纯CPU函数18参数，并增强既有CUDA2参数的退出回执校验，旧参数、断言和计划成员保留。native BF16主树超时已用同源码、同用例和原90秒门限完成缓存配置对照：全新可写缓存及已生成缓存只读复验均完整通过；重复源码编译是本次可复现超时的主要开销，无须改调度或放宽门限。上轮主树失败与隔离通过的环境缓存值未完整冻结，历史差异不反推补签；范围与原失败由运行记录第五十九节保留。实际源码/提交和结果从运行记录定位，旧72项观测模型矩阵继续暂停。

已有Kalman/EMA的P0 CUDA核验按独立参考区分递推、状态时序与研究收益：EMA闭式历史、标量Joseph、逐维/残差与对角增益各自验证；Online/Aligned每轮首commit，Hybrid last为实际末commit（截断时不保证bonus）。已修非有限噪声参数准入与forget=1除零，端点关闭Q/R自适应而P/F仍更新。局部正确性不证明Kalman建模最优或真实模型加速，成果与来源缺口见唯一研究计划。
