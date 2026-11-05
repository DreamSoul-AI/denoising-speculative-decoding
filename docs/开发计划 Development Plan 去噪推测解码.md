**去噪推测解码 开发计划**
更新日期：2026年11月5日。职责依据 [架构设计](<架构设计 Architecture Design 去噪推测解码.md>) 与 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)，科学问题由 [研究计划](<研究计划 Research Plan 去噪推测解码.md>) 独立维护。

# 一、当前范围
基线 `ce37fee`，继续 `codex/structure-cleanup-20261105` 更新Draft PR106到dev。本轮约30分钟，先制定计划立即实施：核依赖6分钟、归位及CUDA适配10分钟、净提交验证10分钟、文档与交付4分钟。算法、生产协议、科学阈值及旧Run身份不改。

本轮只读核远端RPipe main=`414ccb7be5d4eb3739ccc09a4774cad7e8e1a9e6`，读取代码结构设计的实际职责镜像、轻量API和独立验证章节；不访问本机RPipe/dayan，不同步模板实现。

## 目标、步骤与验收
| **步骤与预估** | **文件及实际工作** | **验收** |
| --- | --- | --- |
| 核对，6分钟 | 固定基线逐函数读取typed request、audit、CLI和test_decode夹具消费者，核真实模型loader与设备能力 | 26旧函数101参数：CPU11函数38参数、CUDA15函数63参数；旧decode14函数35参数另列未执行范围 |
| 实施，10分钟 | typed3函数归structure/api/test_decode_api；audit12归artifact/test_observation，CLI11归flow/test_cli。原request和session/sink/run合已有tests/conftest中的纯request与惰性bundle工厂，清除test-to-test imports | 原参数、异常、断言与数值协议保留；没有新生产模块/算法或重复夹具。纯request路径不导入科学栈；保留decode仅复用原CPU语义，不静默接卡 |
| 合同与验证，10分钟 | 真实数值15函数移出CPU计划、入既有CUDA计划，纯合同11函数入CPU快速计划；实际session/model/embedding/forward记录CUDA并核状态恢复 | 945登记不增；预计CPU快速1355参数，CUDA226项执行，native11项。单卡、CPU120秒/两CUDA各300秒预算不扩；63新CUDA参数和38纯合同参数在净HEAD有独立结果 |
| 交付，4分钟 | 同步设计、研究执行边界及日期记录，独立审查AST/设备/元数据/原件，再普通推送 | 两checkout干净，PR保持Draft→dev，main/dev不改、保护不绕过、制品不入Git |

轻量typed request只经标准库API和纯decode_request构造，不为参数校验加载FakeSession、Torch或Transformers。共享数值fixture的torch、generate、mask等在惰性factory内装配，显式指定device；CUDA分支必须先确认可用，embedding、forward输入/输出均核实际设备并写回执，不以默认设备标签代替执行。诊断property只登记一次且在call结束前完成，不在teardown改写冻结事实。

audit的non-object envelope与模型路径逃逸，以及CLI的参数、比较、输出准入和构造失败属于纯IO/合同；其余audit10函数56参数、CLI5函数7参数实际创建/执行数值，作为CUDA工程验证。真实728参数CLI用例仍调用原本地loader，测试级委托装配将会话放CUDA并核forward；生产CLI未新增device选项，不把测试注入能力写成用户默认能力。只保存CUDA权重不能证明loader/forward在GPU。

本轮不扩大旧decode14函数35参数的执行范围，只迁走其公用helper以清除audit/CLI对测试文件的依赖；其原函数body/参数/显式CPU构造语义须由AST审查保持，既有元数据与后续CUDA迁移单独登记。不能据本轮CUDA通过宣称旧CPU数值集合已执行或跨设备bitwise等价。

验证入口保持Ruff、默认mypy及tests/run.py的cpu-quick、cuda-native-smoke、cuda-validation。两个GPU计划顺序运行，先记录torch.cuda.is_available、实际设备/dtype/backend和负载；忙时等待或缩小规模，不CPU回退。报告从落盘结果只读重建，核逐参数终态、SHA、链接、20旧锁与四项原字节缺失的严格拒绝。来源核对只沿已知375字节日志/manifest/两索引，不重扫refs/ACL/全盘，不重造或补签原件；最终拒绝driver在净HEAD只执行一次，无输出且未进入张量加载才合格。

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
