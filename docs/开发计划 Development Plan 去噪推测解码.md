**去噪推测解码 开发计划**
更新日期：2026年11月5日。职责依据 [架构设计](<架构设计 Architecture Design 去噪推测解码.md>) 与 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)，科学问题由 [研究计划](<研究计划 Research Plan 去噪推测解码.md>) 独立维护。

# 一、当前范围
继续 `codex/structure-cleanup-20261105` 并更新 PR 106。远端必需状态阻断合并时继续工程整理，不修改规则或伪造状态。本轮比较基线为上一轮 `36a74db`，算法、默认参数、历史输入和冻结观测合同保持。结构参考限定远端RPipe main固定提交 `414ccb7be5d4eb3739ccc09a4774cad7e8e1a9e6` 的已读文件，不以本机其它checkout作参考。

本轮约30分钟顺序为：4分钟核规范、远端来源与CUDA；10分钟完成epoch/早停的独立进程对照和必要修复；7分钟处理checkpoint提交前后写入故障；4分钟整理fair_branch旧入口；5分钟完成干净提交两计划、记录和同分支推送。验收分别核连续/恢复状态与停止位置、旧bundle保全及失败终态、显式输入与新输出根、当前HEAD逐项结果。此处是工程任务，不新增科学Study。

# 二、六类核对与落实
| **类别** | **本轮处理** | **仍需完成的具体差距** |
| --- | --- | --- |
| 目录结构 | 删除四个 API facade，构建直接调用既有工厂，真实 tracker 装配归 algorithm/tracker.py；删除九个纯诊断再导出 | 70 个顶层遗留文件、14 条 structure 命令记录尚未迁完；父包 structure/control 仍有间接装配链 |
| 配置与支持版本 | 文件 SHA256 统一使用 provenance.file_sha256 的有界流式读取，去除 Python 3.11 专属 file_digest 依赖 | Python 3.10 尚未实机运行，test_package 的 tomllib importorskip 会阻断该版本的严格计划，需确定测试兼容策略；不可据模拟缺 API 测试声称全版本支持 |
| 入口脚本 | 共用launch_once委托sharedCLI，重验证先预检再载validator；独立worker固定import根、source/plan检查，生成PS1保存资源分配与退出码，默认无盲重试 | fresh与office三类入口已入flow并要求显式输入/计划/新输出，四个folded例外删除；fair_branch/pilot等其余legacy命令仍待按协议迁移，盘符清单不涵盖所有拼接路径 |
| 日志与输出 | 新诊断 JSONL 与安全单行呈现，旧文本只读；status/logs/report 统一实际路径与时序 | report 已要求外部新目标且只读检查；output_space_extension 离线槽位分析改显式输入/输出并独占创建。新增显式外部 Study 根，白名单快照与原源双校验，全部 runs/shared/docs/index/锁留运行根；轮转/保留及日志故障隔离未完成 |
| 测试 | 新增行为和失败用例逐项登记，归细已迁测试的 target 与部分 unit 层级，静态直接反向导入清单归零 | 其余用例仍大量继承 integration，target 较宽；read_plan 尚未完整核枚举与成本/资源语义，需逐用例复核，不能只以字段存在认定合规 |
| 文档 | 研究计划补三项既有 Study 的问题/边界/恢复条件，以及新设计接入前的选择与判读；同步当前设计和日期记录 | 提供固定 Git 原件只读接口，槽位与followup分析已迁入两柱；其余完整分析器仍依赖旧工具/计划映射，原 helper 身份存在未知，不能替换路径冒充同一来源 |

表中当前源码状态与对应实际验证从 [开发记录](<开发记录 Development Record/2026-11-05 宿主结构收束.md>)、[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) 定位。没有把完整原生重放或整个规范的落地写成已完成。

# 三、下一轮优先级与验收
1. 恢复后续：native读取、纯准入、应用及Data v2已按CR0/CR4接入，CR5步数预算的微型CUDA独立进程对照见日期运行记录。后续补epoch预算、早停和checkpoint写入故障边界；实际研究恢复仍须复用来源冻结机制绑定代码、模型与数据内容。HF Trainer、多worker及跨设备保持未验证范围，不从当前夹具扩大保证，也不承诺硬件故障时跨库事务回滚。
2. 遗留模块：按消费者归并其余再导出与重复助手，分离 Study 组成和命令。保留来源字节，legacy 只减，CLI help 与单项测试可独立运行
3. 测试合同：明确 Python 3.10 的测试依赖策略，逐项复核 level/target/成本及严格清单枚举，不能跳过必需项换取通过
4. 日志可靠性：确定实际容量与保留要求，隔离诊断落盘故障和正式科学终态，保留首错及追加失败，不吞错或让日志代替结果
5. 其它入口与分析器：继续迁移fair_branch/pilot等专用入口，按消费者整理其余历史分析器；不启动旧矩阵。

# 四、交付边界
每轮 20–30 分钟，以当前 HEAD 的快速单元与有界 CUDA 验证、静态检查和审查证据更新同一 PR。保留平台实际分支保护，本轮不再尝试合并，不启用 autoMerge、Actions，不修改 workflow、设置或规则。

临时材料仅放用户指定 DreamSoul/tmp 下的任务目录，不写 C 盘。不另开新算法，CUDA验证按实际设备与负载安排，不下载、升级依赖或修改main。固定 ESP 来源树、ORIGIN.json、SpecBench 与旧 Run 保持原身份，研究恢复另按唯一计划与当前授权执行。
