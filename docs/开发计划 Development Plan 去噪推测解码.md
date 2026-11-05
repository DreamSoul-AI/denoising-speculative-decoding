**去噪推测解码 开发计划**
更新日期：2026年11月5日。职责依据 [架构设计](<架构设计 Architecture Design 去噪推测解码.md>) 与 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)，科学问题由 [研究计划](<研究计划 Research Plan 去噪推测解码.md>) 独立维护。

# 一、当前范围
继续 `codex/structure-cleanup-20261105` 并更新 PR 106。远端必需状态阻断合并时继续工程整理，不修改规则或伪造状态。本轮比较基线为上一轮 `b60a8bb`，算法、默认参数、历史输入和冻结观测合同保持。远端 RPipe main 重新核对仍为 `414ccb7be5d4eb3739ccc09a4774cad7e8e1a9e6`，只采用已读文件的结构与输入身份原则，不以本机其它 checkout 作参考。

本轮约 30 分钟顺序为：4 分钟核基线、规范、远端与 CUDA 并按静态消费者选组，11 分钟归并五方法配对、ESP 汇总和配套历史图入口，7 分钟同步调用方/测试/文档及审查失败路径，8 分钟验证干净提交并推送。验收保留协议/公式、argmax 与 AR 诊断边界，历史输入独立绑定且不由当前 helper 重签，编排和新输出归 flow，遗留清单只减。CPU 仅做快单测与绘图/IO 合同，数值验证使用 CUDA 并记录设备，不运行研究权重、旧矩阵或新算法。

# 二、六类核对与落实
| **类别** | **本轮处理** | **仍需完成的具体差距** |
| --- | --- | --- |
| 目录结构 | 删除四个 API facade，构建直接调用既有工厂，真实 tracker 装配归 algorithm/tracker.py；删除九个纯诊断再导出；structure 命令清单已空 | 55 个顶层遗留文件尚未迁完，父包 structure/control 仍有间接装配链 |
| 配置与支持版本 | 文件 SHA256 统一使用 provenance.file_sha256 的有界流式读取，去除 Python 3.11 专属 file_digest 依赖 | Python 3.10 尚未实机运行，test_package 的 tomllib importorskip 会阻断该版本的严格计划，需确定测试兼容策略；不可据模拟缺 API 测试声称全版本支持 |
| 入口脚本 | 共用 launch_once 委托 shared CLI，fingerprint/profile 归统一 diagnostics flow；专用入口包含 fresh、office、fair、pilot/preforward 和 hybrid_gain | structure 命令和直接盘符清单均已空，但不能据此声称其余 Study 顶层脚本、旧配置与拼接路径均已整理 |
| 日志与输出 | 新诊断 JSONL 与安全单行呈现，旧文本只读；status/logs/report 统一实际路径与时序 | report 已要求外部新目标且只读检查；output_space_extension 离线槽位分析改显式输入/输出并独占创建。新增显式外部 Study 根，白名单快照与原源双校验，全部 runs/shared/docs/index/锁留运行根；轮转/保留及日志故障隔离未完成 |
| 测试 | 新增行为和失败用例逐项登记，归细 target 与层级，静态直接反向导入清单归零；真实 CUDA 计划成本标 C3 并声明单卡及时间估计 | 其余用例仍大量继承 integration，target 较宽；read_plan 尚未完整核枚举与成本/资源语义，显存需求仍待估计，需逐用例复核，不能只以字段存在认定合规 |
| 文档 | 研究计划补三项既有 Study 的问题/边界/恢复条件，以及新设计接入前的选择与判读；同步当前设计和日期记录 | 提供固定 Git 原件只读接口，槽位与followup分析已迁入两柱；其余完整分析器仍依赖旧工具/计划映射，原 helper 身份存在未知，不能替换路径冒充同一来源 |

表中当前源码状态与对应实际验证从 [开发记录](<开发记录 Development Record/2026-11-05 宿主结构收束.md>)、[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) 定位。没有把完整原生重放或整个规范的落地写成已完成。

# 三、下一轮优先级与验收
1. 恢复后续：native步数/epoch、早停、终态和已列checkpoint故障已有微型CUDA覆盖，范围与结果见日期运行记录。实际研究恢复仍须复用来源冻结机制绑定代码、模型与数据内容；HF Trainer、多worker及跨设备保持未验证范围，不从当前夹具扩大保证，也不承诺硬件故障时跨库事务回滚或断电持久性。
2. 遗留模块：按消费者归并其余再导出与重复助手，分离 Study 组成和命令。保留来源字节，legacy 只减，CLI help 与单项测试可独立运行
3. 测试合同：明确 Python 3.10 的测试依赖策略，逐项复核 level/target/成本及严格清单枚举，不能跳过必需项换取通过
4. 日志可靠性：确定实际容量与保留要求，隔离诊断落盘故障和正式科学终态，保留首错及追加失败，不吞错或让日志代替结果
5. 其它入口与分析器：五方法配对、ESP 读出和 input-mask 图已归两柱，继续整理其余 Study 消费者。五方法 published 复核不是 native 重验；原 Run/ESP 来源锁不匹配仍拒绝，后续若需跨源码历史重验必须独立完成来源合同，不以新 helper 重签。其他读端遇归档附注也须按真实字段与 binding 定义，不推广窗口特例。默认完整矩阵与旧 profiling 不因入口迁移获得执行授权。

# 四、交付边界
每轮 20–30 分钟，以当前 HEAD 的快速单元与有界 CUDA 验证、静态检查和审查证据更新同一 PR。保留平台实际分支保护，本轮不再尝试合并，不启用 autoMerge、Actions，不修改 workflow、设置或规则。

临时材料仅放用户指定 DreamSoul/tmp 下的任务目录，不写 C 盘。不另开新算法，CUDA验证按实际设备与负载安排，不下载、升级依赖或修改main。固定 ESP 来源树、ORIGIN.json、SpecBench 与旧 Run 保持原身份，研究恢复另按唯一计划与当前授权执行。
