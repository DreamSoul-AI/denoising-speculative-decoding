**去噪推测解码 开发计划**
更新日期：2026年11月5日。职责依据 [架构设计](<架构设计 Architecture Design 去噪推测解码.md>) 与 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)，科学问题由 [研究计划](<研究计划 Research Plan 去噪推测解码.md>) 独立维护。

# 一、当前范围
继续 `codex/structure-cleanup-20261105` 并更新 PR 106。远端必需状态阻断合并时继续工程整理，不修改规则或伪造状态。本轮比较基线为上一轮 `8b40874`，算法、默认参数、历史输入和冻结观测合同保持。本轮按用户要求不访问或修改RPipe，只复用此前已读的远端固定 `414ccb7be5d4eb3739ccc09a4774cad7e8e1a9e6` 结构原则，未重新核实其分支现状。

本次先用约30分钟规划，再进入目标模式实施和完整验证，总计预估45–50分钟。规划窗口从2026年11月5日17:04:40至约17:34:40（Asia/Shanghai），期间只读取实现/规范、收集测试与做临时纯IO反例，不修改生产或测试源码。下列任务为本次具体方案，实际结果只从日期运行记录和PR定位。

## 本轮目标、步骤与验收
| **阶段与预计时长** | **具体工作** | **验收** |
| --- | --- | --- |
| 规划1，8分钟 | 核基线、全部checks消费者、38函数/54参数及来源锁；读取CSV真实调用路径 | 区分当前唯一测试消费者、20份原来源锁与4项原字节缺口；不把历史路径当当前导入 |
| 规划2，10分钟 | 逐函数划分validation、Run gate、Study gate、revalidation与CSV合同，确认CPU/GPU成本和夹具归属 | 保原函数、参数和失败判据；47个纯IO参数与7个实际张量参数分开；CSV错列有最小IO反例 |
| 规划3，7分钟 | 独立审查删除影响、CSV追加政策、夹具惰性与净HEAD验收集合 | 不改历史SHA/Run，异常类身份保持；列清数据损坏、缺结果、设备不符和预算失败如何拒绝 |
| 规划4，5分钟 | 冻结当前设计、精确ID映射、元数据及执行顺序 | 源码/测试尚未修改；每项有真实owner、目标文件、验收命令和停止条件 |
| 实施1，6分钟 | 归位checks/CSV及已有revalidation镜像，四个假证据工厂复用现有conftest；同步全部调用方后删除checks转导出层 | 38个原checks函数及54参数不丢；原5个revalidation函数和CSV用例保留；无新生产模块或独立助手文件 |
| 实施2，4分钟 | 在原CSV Logger内修复已复现的列序/追加header问题，补边界与IO拒绝测试，同步清单和正文 | 文件列序固定，同键重排正确，schema或header不合时写前拒绝；stdout原格式保持；错误上抛，旧行不重写 |
| 实施3，6–8分钟 | 静态检查、独立审查、净HEAD CPU快测；确认CUDA可用并顺序执行受影响CUDA与native smoke计划 | 每参数正式结果、实际设备、报告/证据SHA、只读重建和缺件拒绝均有效；CPU不运行7个alignment张量参数 |
| 交付，2分钟 | 普通推送原分支，更新Draft PR106并回读，核main/dev和工作树 | 目标dev、Draft保持；不合并、不强推、不改保护，不提交制品 |

测试按真实职责归位：纯验证合已有 `tests/studies/_shared/structure/test_validation.py`；Run gate归 `tests/studies/_shared/flow/summarize/test_run.py`，Study gate归 `tests/studies/_shared/flow/process/test_run.py`；原revalidation测试迁 `tests/studies/_shared/flow/test_revalidate.py` 并合并原checks中的保存件复核。CSV归 `tests/studies/_shared/structure/test_csv_logger.py`。共享工厂 `saved_decode_config`、`saved_decode_assets`、`saved_decode_study`、`corrupt_saved_argmax` 位于既有 `tests/conftest.py`，只在请求时导入和创建材料，每次返回独立对象，不跨测试文件导入助手。

alignment七参数的主合同是最终gate拒绝不一致保存计数，放summarize镜像，真实collect只构造有效基线。测试先断言CUDA可用、记录设备，在有界CUDA默认设备上下文内执行实际collect和verify/gate，核张量实际设备，退出恢复上下文；不改生产数学或把设备字符串当GPU证据。元数据为integration/content/C3/GPU1，加入cuda-validation并从CPU计划精确排除。其余受影响纯IO及CSV回归进入cpu-quick，按实际调用细分unit/integration和C1/C2，不提高既有预算掩盖超时。

CSV Logger仍负责可选CSV和现有stdout格式，不合并诊断JSONL或Tracker。文件首次列序由有效header确定；每次追加前重读有序header及末尾记录边界，同键顺序变化按固定header写。首次使用的零字节文件可正常写一个header；非空文件的空/重复/损坏header、非UTF8、缺末尾换行及缺键/增键均拒绝。同一实例发现header有序列表变化、目标被清空或移除也拒绝，不自行补header或修复旧行。无文件输出时保留动态字段、step/prefix及四位小数；IO故障原样上抛，不宣称原子追加、多writer或完整旧CSV重验。

CSV镜像保原 `test_appends_rows_under_one_header`，新增七个单一行为函数：固定header重排（同/新实例）、schema准入（缺/增/首空）、坏header/追加边界（10种）、已有零字节文件、step/prefix与UTF8、stdout动态字段（2种）、IO错误（mkdir/open/write）。合计八函数23参数，都是快速CPU IO；新行为先用反例观察失败，再做原模块最小修复。step同名键的现有语义不在本轮扩展，真实调用方未使用该组合。

验收按同一净提交执行，以下输出根必须各自全新且位于本任务授权临时根：

```text
python -m ruff check src studies tests
python -m mypy
python -m mypy --namespace-packages --explicit-package-bases --follow-imports=silent studies/_shared/structure/csv_logger.py
python tests/run.py --plan cpu-quick --output-root <新CPU根>
python tests/run.py --plan cuda-validation --output-root <新CUDA根>
python tests/run.py --plan cuda-native-smoke --output-root <新native-CUDA根>
```

两GPU计划顺序执行，各保既有300秒预算；cpu-quick保120秒。预计44个既有函数改测试ID、七个新增CSV函数使登记总量935→942，实际以AST与清单核对；原54个checks参数及原11个revalidation参数后缀不变。净HEAD还须核所有受影响函数有正式终态、报告SHA与只读重建一致、三份报告链接有效、20份原来源锁字节不变及四原件缺失的拒绝回执。任何必需项失败、跳过、缺件、源漂移或CUDA不可用均阻断交付；负结果不删，预算不足先缩小经说明的范围或等待，不回退CPU充数。

必须保留的失败边界：缺阶段/结果或实验子集不能通过；source/config/输入文件漂移先拒绝，旧成功文件不被预检覆盖；完整argmax为T0门、AR只诊断；四项原运行字节未知继续严格拒绝且不加载张量。当前checks删除会改变新来源摘要，旧来源锁不重新签、不排除被删文件绕过核验。模型、算法和停止的研究协议不因整理获得新执行。

# 二、六类核对与落实
| **类别** | **已有实现条件** | **仍需完成的具体差距** |
| --- | --- | --- |
| 目录结构 | 宿主和Study按structure/flow分工，薄入口调用实际部件；配置和tracker装配复用既有工厂，不另建facade | 冻结顶层legacy保持为空；父包仅标识职责，普通部件使用canonical leaf导入。仍需核其余科学子包公开面与尚未镜像的测试，不推广本次范围 |
| 配置与支持版本 | 文件 SHA256 统一使用 provenance.file_sha256 的有界流式读取，去除 Python 3.11 专属 file_digest 依赖 | Python 3.10 尚未实机运行，test_package 的 tomllib importorskip 会阻断该版本的严格计划，需确定测试兼容策略；不可据模拟缺 API 测试声称全版本支持 |
| 入口脚本 | 共用 launch_once 委托 shared CLI，fingerprint/profile 归统一 diagnostics flow；专用入口包含 fresh、office、fair、pilot/preforward 和 hybrid_gain | 已登记顶层遗留、structure命令及直接盘符清单均为空；旧配置、拼接路径与间接装配仍须逐消费者审查，不据清单归零宣称全项目合规 |
| 日志与输出 | 新诊断 JSONL 与安全单行呈现，旧文本只读；status/logs/report 统一实际路径与时序 | report 已要求外部新目标且只读检查；output_space_extension 离线槽位分析改显式输入/输出并独占创建。新增显式外部 Study 根，白名单快照与原源双校验，全部 runs/shared/docs/index/锁留运行根；轮转/保留及日志故障隔离未完成 |
| 测试 | 新增行为和失败用例逐项登记，归细 target 与层级，静态直接反向导入清单归零；真实 CUDA 计划成本标 C3 并声明单卡及时间估计 | 其余用例仍大量继承 integration，target 较宽；numeric和测量粒度的空/缺/无效结果门及终态已实现，纯测量不计质量通过；批次Markdown报告及同run/plan/HEAD证据核验已实现；附件式原始证据和更细结果协议仍待按实际需求裁剪；read_plan按实际枚举与被选计划核成本/资源语义；显存需求仍待估计，需逐用例复核，不能只以字段存在认定合规 |
| 文档 | 九类文档保持职责，研究问题和工程待办分开；设计与实际差距、日期记录和固定来源分别定位 | 提供固定 Git 原件只读接口，槽位与followup分析已迁入两柱；其余完整分析器仍依赖旧工具/计划映射，原 helper 身份存在未知，不能替换路径冒充同一来源 |

表中当前源码状态与对应实际验证从 [开发记录](<开发记录 Development Record/2026-11-05 宿主结构收束.md>)、[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) 定位。没有把完整原生重放或整个规范的落地写成已完成。

# 三、下一轮优先级与验收
1. 恢复后续：native步数/epoch、早停、终态和已列checkpoint故障已有微型CUDA覆盖，范围与结果见日期运行记录。实际研究恢复仍须复用来源冻结机制绑定代码、模型与数据内容；HF Trainer、多worker及跨设备保持未验证范围，不从当前夹具扩大保证，也不承诺硬件故障时跨库事务回滚或断电持久性。
2. 遗留模块：按消费者归并其余再导出与重复助手，分离 Study 组成和命令。保留来源字节，legacy 只减，CLI help 与单项测试可独立运行
3. 测试合同：后续复核其余共享测试镜像和Python 3.10测试依赖策略；native smoke沿NS0–NS4维护CUDA身份、预算与完整argmax。继续逐项核level/target/成本，保留空结果与报告发布阻断；若新增附件或基准要求，先明确证据与比较合同，不能跳过必需项换取通过
4. 日志可靠性：确定实际容量与保留要求，隔离诊断落盘故障和正式科学终态，保留首错及追加失败，不吞错或让日志代替结果
5. 其它入口与分析器：cross-task 顶层分析命令已归两柱，norm/paired/continuation 及 ESP 执行/读端已归并，smoke与extension六入口收束后，继续复核其余科学子包、旧声明路径与来源合同的可读性，并补未实际重读的native/state全套历史材料；旧CPU证据仍保设备身份，extension已停止机理保持拒绝。配对 published 复核与保存张量分析均不是 native 重验；原 Run/ESP 来源锁不匹配仍拒绝，后续若需跨源码历史重验必须独立完成来源合同，不以新 helper 重签。其他读端遇归档附注也须按真实字段与 binding 定义，不推广窗口特例。默认完整矩阵与旧 profiling 不因入口迁移获得执行授权。

# 四、交付边界
通常每轮20–30分钟；本次按最新要求先约30分钟规划，再实施到合同和验证完成。以当前HEAD的快速单元与有界CUDA验证、静态检查和审查证据更新同一PR。保留平台实际分支保护，不尝试合并，不启用autoMerge、Actions，不修改workflow、设置或规则。

临时材料仅放用户指定 DreamSoul/tmp 下的任务目录，不写 C 盘。不另开新算法，CUDA验证按实际设备与负载安排，不下载、升级依赖或修改main。固定 ESP 来源树、ORIGIN.json、SpecBench 与旧 Run 保持原身份，研究恢复另按唯一计划与当前授权执行。
