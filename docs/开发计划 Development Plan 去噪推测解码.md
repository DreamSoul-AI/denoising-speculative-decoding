**去噪推测解码 开发计划**
更新日期：2026年11月5日。职责依据 [架构设计](<架构设计 Architecture Design 去噪推测解码.md>) 与 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)，科学问题由 [研究计划](<研究计划 Research Plan 去噪推测解码.md>) 独立维护。

# 一、当前范围
继续 `codex/structure-cleanup-20261105`，基线 `5ef5ec9`，更新Draft PR106到dev。本轮约30分钟：核对6分钟、归位10分钟、验证10分钟、交付4分钟。目标是测试按真实职责镜像并细化元数据，不改变生产模块、算法、科学阈值、来源身份或旧Run。

远端RPipe main本轮只读核为 `414ccb7be5d4eb3739ccc09a4774cad7e8e1a9e6`，已读其README和代码结构设计测试章节。采用实际实现职责镜像、importlib完整路径收集、工程/GPU/研究证据分开；不复制模板源码，不访问本机RPipe或dayan。

## 本轮目标、实施与验收
| **步骤与预估** | **具体改动** | **验收** |
| --- | --- | --- |
| 核对，6分钟 | 读取最新约定、代码/测试规范、现有测试和真实被测模块，核基线计划与参数 | 13个旧函数48参数、945总函数；区分包级打包合同、共享部件、CLI局部链路和结构检查 |
| 归位，10分钟 | gpu_admission三函数迁_shared/structure镜像并删除动态loader；mask_settings三函数合已有config镜像；package中的manifest/load_lock三函数合sources、write_lock三函数合CLI、initializer检查合parent_imports | 参数、异常、断言与路径原义保持；两个真实仓库打包用例留原文件；不建生产文件，不复制fixture |
| 合同与验证，10分钟 | 精确更新13个ID及其所有计划引用，显式target/level/type/cost/结果/优先级；快速CPU合同与顺序CUDA/native计划，记录设备 | 迁前后参数后缀一致，所有迁移参数有净HEAD结果；GPU集合不增，945声明与AST相等；20旧锁及生产字节保持，报告可只读重建 |
| 交付，4分钟 | 同步README必要状态、代码结构设计、日期记录，审查并普通推原分支、更新PR | 两checkout干净，PR仍Draft→dev；main/dev不改，必需检查不绕过，制品不入Git |

真实owner分别为 `studies._shared.structure.gpu_admission.classify/parse_xml`、`structure.config.spec_for`、`structure.sources.manifest/load_lock`、`flow.cli.write_lock_for` 和宿主包初始化结构。XML与分类、纯配置、load_lock为快速CPU单元；CLI写锁跨来源/布局的局部IO路径按integration登记，来源扫描和Git观察成本与轻量解析分开，不因字段名包含GPU就伪称CUDA执行。保留包级位置与打包检查的原根定位，不把它们硬塞入无关科学leaf。

原test_package对tomllib的模块级importorskip只随剩余打包用例保留；被移出的六个共享IO和初始化检查不再因无tomllib漏收。Python3.10实机未验，本轮不因此宣称解决全部版本兼容。

本轮只迁移既有合同，不新增测试函数或行为。验证应证明位置、元数据和行为保持，不为改名制造无关红绿用例。独立审查核原函数AST（允许canonical alias、ROOT等义定位）、参数后缀、保留断言与原计划选择；未知、未执行、跳过及漏项不能记通过。

```text
python -m ruff check src studies tests
python -m mypy
python tests/run.py --plan cpu-quick --output-root <新CPU根>
python tests/run.py --plan cuda-native-smoke --output-root <新native-CUDA根>
python tests/run.py --plan cuda-validation --output-root <新CUDA根>
```

CPU预算120秒、两CUDA各300秒不扩大；GPU计划顺序执行，运行前确认torch.cuda.is_available并记录实际设备和负载，忙时等待或缩小规模，不CPU回退。生产数值与模型验证仍用CUDA，纯保存标量/配置/观察stub/来源IO不冒称GPU。四项历史原字节缺失继续严格拒绝，无新候选只核已知375字节日志、manifest及两索引，不重扫refs/ACL/全盘，不重建或补签原件。来源拒绝在净HEAD执行一次，必须无输出且未进入张量加载。

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
本轮按约30分钟的计划连续实施到合同、验证与交付完成。以当前HEAD的快速单元与有界CUDA验证、静态检查和审查证据更新同一PR。保留平台实际分支保护，不尝试合并，不启用autoMerge、Actions，不修改workflow、设置或规则。

临时材料仅放用户指定 DreamSoul/tmp 下的任务目录，不写 C 盘。不另开新算法，CUDA验证按实际设备与负载安排，不下载、升级依赖或修改main。固定 ESP 来源树、ORIGIN.json、SpecBench 与旧 Run 保持原身份，研究恢复另按唯一计划与当前授权执行。
