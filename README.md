# 去噪推测解码
研究免训练多 token 起草与验证：以 ESP 为参照，比较输入嵌入 EMA、速度外推和输出隐藏状态掩码。中文名称为“去噪推测解码”。仓库地址：[DreamSoul-AI/denoising-speculative-decoding](https://github.com/DreamSoul-AI/denoising-speculative-decoding)。

截至2026年10月21日的 8B 非官方 Table 1 独立复验已经完成，论文目标未复现。数字和原件在 [历史证据归位](<docs/运行记录 Run Record/2026-10-31 历史证据归位.md>)。提交 token 的验证前向 argmax 自检与独立 AR 一致率分别报告，τ 是提交数与解码调用数之比，不能据此认定实际速度收益。

已实现的 K 校正是协议 `postcommit-final-norm-v1` 的合成工程夹具，不是 2026年10月31日会议中的 trajectory 与 observation 联合更新，也不是 Kalman 滤波。160 对数据协议和真实模型收益都还没有执行。历史协议与合成测量入口见 [研究计划](<docs/研究计划 Research Plan 去噪推测解码.md>) 与 [联合 mask 检查](<docs/运行记录 Run Record/2026-11-01 联合mask检查.md>)。

# 一、文档入口
| **文档** | **维护内容** |
| --- | --- |
| [协作约定](AGENTS.md) | 资料定位、适用规范及执行边界 |
| [通用调研 General Research 去噪推测解码](<docs/通用调研 General Research 去噪推测解码.md>) | 机制、方案、直接来源与当前判断 |
| [文献调研 Literature Review 去噪推测解码](<docs/文献调研 Literature Review 去噪推测解码.md>) | 已读文献、阅读深度和与本项目的关系 |
| [架构设计 Architecture Design 去噪推测解码](<docs/架构设计 Architecture Design 去噪推测解码.md>) | 职责、状态、流程、质量目标和取舍 |
| [代码结构设计 Code Structure Design 去噪推测解码](<docs/代码结构设计 Code Structure Design 去噪推测解码.md>) | 模块责任、接口、依赖和测试镜像 |
| [研究计划 Research Plan 去噪推测解码](<docs/研究计划 Research Plan 去噪推测解码.md>) | 研究问题、假设、对照和停止规则 |
| [开发计划 Development Plan 去噪推测解码](<docs/开发计划 Development Plan 去噪推测解码.md>) | 剩余工程、验收和待确认的来源策略 |
| [缺陷记录 Bug Report 去噪推测解码](<docs/缺陷记录 Bug Report 去噪推测解码.md>) | 尚未关闭的问题 |
| [开发记录 Development Record](<docs/开发记录 Development Record/INDEX.md>) | 代码与文档结构调整。目录见 [INDEX](<docs/开发记录 Development Record/INDEX.md>)。 |
| [运行记录 Run Record](<docs/运行记录 Run Record/INDEX.md>) | 实际执行、环境与结果。目录见 [INDEX](<docs/运行记录 Run Record/INDEX.md>)。 |

# 二、当前实现与使用边界
当前安装包为 `denoise-spec-mtp`，研究宿主 `spec_mtp` 按 [RPipe](https://github.com/DreamSoul-AI/RPipe) 的 `structure/` 与 `flow/` 两柱组织。宿主顶层保留包标记、模块入口和稳定控制台薄入口 `cli.py`。合同在 `structure/api/decode_api.py`，数据在 `structure/data/prompts.py`，模型在 `structure/model/hf_causal.py`，解码与 mask 在 `structure/algorithm/`，统计在 `structure/algorithm/counts.py`，观测与审计在 `structure/artifact/observation.py`。原数据、模型、审计和分析顶层模块已删除，调用方直接导入真实路径。

`python -m spec_mtp` / `spec-mtp` 由 `flow/cli.py` 提供 `decode/audit/compare`。Study 使用 `python -m studies._shared.flow.cli`，接入宿主 `flow/study_cli.py` 与默认六阶段。包初始化不加载模型或整套调度器。流程由本仓实现，不再内嵌 `src/rpipe/`。旧流程 API facade 已删除，构建直接使用所属层工厂。九个共用诊断再导出已清理，消费者直接导入真实模块。Study 顶层脚本仍有遗留，见代码结构设计与开发计划。

算法由 Enmao 重新设计，当前进行结构整理与有界工程验证。新算法方向由 Enmao 确定，旧协议作为历史证据。正确性、优化和性能验证优先 CUDA，运行前确认可用并记录实际设备，CPU 只做快速单元测试。GPU 满载时等待或缩小规模，模型下载仍不在当前范围。每轮从 dev 工作分支经 PR 集成，不直接修改 main。

第三方 ESP 来源树及 `ORIGIN.json` 保留，不能把它称为作者官方实现。原始 SpecBench 数据保留于 `datasets/spec_bench/question.jsonl`。

科学 CLI 的 `decode` 必须显式提供 `--model-sources <JSON>`：清单位于模型目录外，以相对路径映射该完整、自包含的本地模型目录内全部文件的 SHA256，不缺项、不含目录外成员。输出也放在模型根外；解析后逃逸模型根的符号链接会被拒绝，不直接承诺兼容引用外部 blobs 的 Hub 快照。载模前、运行头及结束记录写出前分块读取核验实际文件；漂移时保留部分观测，不写完成末行。运行头另记实际模型 config、backend、device、dtype 和能力，调用者原 `sources` 标签保持原义。审计带模型来源的新记录时使用 `audit <raw> --sources <标签JSON> --model-sources <原模型清单JSON>`；缺少独立模型清单不能记来源完整。旧无模型来源字段的记录仍只比较原标签，不代表重新验证实际模型。`audit/compare` 不加载模型或重新生成观测。

旧逐条运行证据、日志、图和报告保留在固定历史/外部运行根，未恢复进 Git。历史读端测试从固定 Git 提交 `b8b20b5ada7fd860894114cbf224ce749af430cf` 按需提取原始 blob 到测试临时目录，核对对象哈希与写入字节；缺少 Git 对象直接失败，不下载或伪造原件。当前分析器和校验门读取这些不可变输入，不导入或执行归档源码，不修改原哈希与断言；相关15个参数项已恢复到 CPU 计划，不再排除。此验证只复核保存的历史观测与读端合同，不重新运行历史模型，也不代表全部历史结果已独立复验。

新建 Study 运行使用 `python -m studies._shared.flow.cli run studies/<name> --run-root <外部Study运行根>`，也可先 `make studies/<name> --run-root <外部Study运行根>`，再 `launch <外部Study运行根>`。使用 --run-root 时 Study 路径紧随命令。外部根保存声明与源码白名单快照、study_source.json、runs/shared/docs、index、锁与日志，源码 Study 保持只读；状态、日志、process 和 report 后续直接传该外部根。生成的 PowerShell 启动入口同样走 shared CLI，使用 shared 控制台并传播退出码；调度worker固定当前宿主导入根，并在写运行态前验证来源。重复 launch 跳过已完成Run，部分完成时只派发pending；这是调度恢复，不是checkpoint/RNG恢复。失败默认不自动重试，损坏或与index/config不一致的jobs会拒绝，调整GPU分配或轮次需显式remake。已绑定且逐文件摘要相同的根可继续使用，原 Study 或快照漂移会失败，须用新根与研究协议处理，不覆盖旧运行。原 Study 中的绝对导入仍需要原 checkout；静态快照限单成员16MiB、合计32MiB，不复制权重、旧 Run 或未声明生成物。

离线报告使用 `python -m studies._shared.flow.cli report <Study> --out <外部新文件.md>`，输出路径必填，拒绝已有文件、输入目录和源码 checkout；报告检查不回写旧 process 或 gate。`output_space_extension` 的已保存槽位诊断迁入 `python -m studies.output_space_extension.flow.process.run --raw <保存张量目录> --execution <原运行矩阵.json> --repository <本地仓库> --commit <完整提交> --execution-git-sha256 <Git原件SHA256> --output-root <新外部目录>`。它只读原件并加载 CPU 张量，不载入模型；实际运行矩阵须匹配旧报告哈希，与固定 Git 原件分别记录字节身份并核对 JSON 内容。缺失或漂移直接失败，不自动修换行或重签旧来源。历史读件需要支持 `--no-lazy-fetch` 的本地 Git，当前验证版本为 `2.55.0.windows.3`，不支持时直接失败，不降级联网。followup 读端同样迁入 `studies.output_space_followup.flow.process.run`，参数与槽位读端一致，保存张量实际在 CUDA 分析并登记设备。

small_matrix 的分析统一使用 `python -m studies.output_space_small_matrix.flow.process.run <transfers|cross-prompt|trajectories|windows> --raw <原输入> --repository <本地Git仓库> --commit <完整提交> --report-git-sha256 <固定报告SHA> --output-root <新外部目录>`。trajectories 的原输入为原运行 execution JSON，只读 JSON，不接受设备参数。其余三种的原输入为原报告与张量目录，必需 `--device cuda:0`。transfers 另需 `--original <原observer目录> --observer-git-sha256 <固定observer报告SHA>`，windows 另需 `--execution <原运行矩阵JSON> --execution-git-sha256 <固定矩阵SHA>`。实际运行原件、固定 Git 原件与张量分别绑定，新工具来源另记，不运行归档源码或模型。优化模式 `python -O` 会明确拒绝，不能跳过科学断言。原 helper 身份缺失保持未知，不补成当前 helper。

两套历史图使用 `python -m studies.output_space_small_matrix.flow.write.run <acceptance|windows> --input <原报告JSON> --repository <本地Git仓库> --commit <完整提交> --input-git-sha256 <固定报告SHA> --output-root <新外部目录>`，windows 另需 `--scale <原scale报告JSON> --scale-git-sha256 <固定scale报告SHA>`。只渲染已保存的标量，不重算模型或统计。归档 window/scale 的已知后补附注单独登记，共有字段仍须全部相等，原 observer 字节必须匹配原 scale binding，不把归档附注混进原运行件。图片保留原数值、颜色、坐标与标题，完成记录绑定实际输入、固定 Git 来源、当前绘图源码和 PNG 摘要，失败只保留 partial，不回写 Study/docs 或旧 Run。

prerequisite 的 q171 图使用 `python -m studies.output_space_prerequisite.flow.write.run --input <原executionJSON> --repository <本地Git仓库> --commit <完整提交> --input-git-sha256 <固定报告SHA> --output-root <新外部目录>`，同样只读固定历史标量，保留三臂与原轴范围规则，独占新输出并核 PNG 与来源。

trajectory 真实模型专用入口为 `python -m studies.trajectory_observation.flow.execute.office <qwen8b|tree|traj> --help`、`python -m studies.trajectory_observation.flow.execute.fresh_replace --help`、`python -m studies.trajectory_observation.flow.execute.fair_branch --help` 与 `python -m studies.trajectory_observation.flow.execute.preforward_signal --help`。它们要求显式本地模型目录、目录外完整模型SHA清单、本地JSONL、计划文件及其SHA、新外部输出；fair_branch以 `--split specbench|dolly --input <本地JSONL>` 保留原337–464/前128切片，preforward保留SpecBench 81–208及128条校准模板，不再猜模型目录或联网取数据。共用本地输入准入，只读取已解析绑定路径。preforward的summary/fit可留作partial，只有收尾复核及关闭后写出的protocol_complete标记表示运行完整；旧gate字段不产生新研究授权。

bonus-aligned/output-hidden 两个专用协议使用 `python -m studies.trajectory_observation.flow.execute.hybrid_gain <bonus-aligned|output-hidden> --help`，同样必需本地模型目录、完整模型清单、SpecBench JSONL、计划 SHA、CUDA 设备和新外部输出。前者保留 465–560 切片与四臂，后者保留 209–336 切片、32 条 holdout 与六臂，校准各为 128 条模板。共同收集、落盘与会话编排归同一 flow，structure 保留纯校准与统计。逐臂核完整 qid 与 argmax 审计，收尾核来源并关闭模型后才发布 protocol_complete，旧根不续写。

合成 pilot 使用 `python -m studies.trajectory_observation.flow.execute.pilot --config <完整本地JSON> --plan <研究计划> --plan-sha256 <计划SHA> --device cuda:0 --output <新外部目录>`，模式、seeds 及预算由显式冻结 config 给出，互斥模式拒绝，不因 DEFAULT 存在就自动启动 1500 步旧训练。训练、模型前向及拟合在 CUDA，CPU 只承担离散样本准备和序列化。九个旧 structure 命令已归 flow，不留反向 wrapper。当前工程验证使用微型模型、oracle、拟合与入口合同，没有运行默认训练、真实权重或旧研究矩阵。

checkpoint/RNG的独立合同见架构设计CR0–CR5。native `TrainAlgorithm` 将读取、准入与应用分开：先核合同、模型状态、数据游标和RNG，再改活跃训练状态；实际bundle载入前后核SHA，来源进入恢复摘要。`algorithm.config.resume_from` 显式指定文件时，缺失即失败；默认latest不存在仍可新建训练。Data子状态使用v2，冻结sampler/loader generator及其共享关系；旧v1不能自动当exact恢复，原件保留。无完整training_state的旧checkpoint继续明确警告partial。

微型CUDA回归已覆盖零worker标准DataLoader、shuffle、随机transform、Dropout、梯度累积，以及第2步保存后新进程恢复至同一6步预算；逐批样本、模型/优化器/调度器、tracker数值、游标与后续RNG均与连续执行比较。该范围不包含HF Trainer、多worker、分布式、真实模型或跨设备等价；数据身份按元数据核对，尚无数据内容哈希保证。

epoch预算与早停另有独立进程CUDA对照，固定评估分数只用于检验stall与停止状态。已停止或任一预算上限已尽的恢复只还原状态，不再取batch、评估或写checkpoint/tracker。checkpoint以同目录独占临时文件同步后最后替换bundle；提交前错误保留旧bundle，提交后诊断清理失败不否定已提交状态。未发布stage及其别名拒绝读入；只有旧分片目录时，写入前要求显式迁移，不自动补签或覆盖旧原件。

诊断日志新写 `assets/logs/events.jsonl`，状态、日志与报告读端兼容旧 `run.log`。应用启动与 Run 身份分开，普通日志不保存异常原文或模型输入。流式文件哈希兼容 Python 3.10 所需 API，但最低版本及全部依赖组合尚未实机验证。

# 三、开发与验证
工作分支经 PR 面向 `dev`，完成检查并按当次明确授权集成，不自动发布。当前 HEAD 的本地结果、历史红项和未覆盖范围记在 PR 正文，不用旧提交结果代替。

共用解码诊断使用 `python -m studies._shared.flow.diagnostics <fingerprint|profile> --config <完整本地JSON> --plan <当前计划> --plan-sha256 <计划SHA> --device cuda:0 --output <新外部目录>`。fingerprint 配置为 `{"variants": null}` 时保留原完整预设矩阵，也可显式给预设列表；profile 要求完整模型、prompt、token、线程、重复与 variant 配置，字段及默认值由该入口的 PROFILE_DEFAULTS 定义。来源、配置、计划与设备绑定到新输出，模型关闭及收尾验证后才写完整标记。fingerprint 的 `--compare <已有报告>` 必须具有匹配工作负载/设备身份及完整内容摘要，旧缺身份报告不当作已验证基线。T0 审计与采样诊断分开，不推广采样保证。

profile 在 CUDA 工作负载上使用 cProfile，component 是主机侧 self-time，保留的 decoder host-clock token rates 也不是端到端 CUDA 吞吐或 kernel 测量。旧 CPU profile 不按新设备身份重解释。本轮只跑微型单 cell 与 profile 合同，未执行完整 fingerprint 矩阵或性能研究。

本机快速单元执行 `python tests/run.py --plan cpu-quick --output-root <外部新验证目录>`，CUDA 工程正确性执行 `python tests/run.py --plan cuda-validation --output-root <另一外部新验证目录>`。CUDA计划不可用即失败，回执保存实际设备、精度和显存；两计划分开报告。原 cpu-pr 计划保留既有用途，本轮不以其全量CPU结果替代CUDA验证。静态清单先验证必需项和精确排除，逐参数 setup/call/teardown 写独立记录，缺收集、缺终态、失败、跳过和证据写失败均非零。整批预算420秒，在测试边界检查；子进程测试各有有限超时，单个本地计算不支持强制中断。运行产物不进 Git。

本轮不调整现有 Actions、CI/CD 或设置。远端结果仅作参考，本地微型CUDA工程验证覆盖既定合成夹具，不覆盖真实权重/数据、吞吐或采样分布保证。已有 CUDA 环境保持原样，不下载模型或覆盖 torch。

## 研发维护与文档现状

Python 3.10+、PyTorch 与固定 Transformers 条件见 [pyproject.toml](pyproject.toml)，当前研究宿主为 `spec_mtp`。目录入口：[产品或库源码](src/spec_mtp/)、[研究声明与适配](studies/)、[测试](tests/)、[当前文档](docs/)。运行与测试按本 README、相关设计和实际配置选择，不把历史命令视为新的实验授权。

文档职责按 AGENTS.md 维护。通用调研不代替文献调研，研究计划不代替开发计划。历史测量在运行记录，结构调整在开发记录。本轮从已关联 DreamSoul 资料库读取研发根约定及相关代码、协作、测试、研究计划、文献、文档和架构规范。本地资料的读取不代表云端同步或最新远端核验。
