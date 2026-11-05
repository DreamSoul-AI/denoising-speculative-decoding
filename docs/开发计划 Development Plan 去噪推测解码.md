**去噪推测解码 开发计划**
更新日期：2026年11月5日。职责依据 [架构设计](<架构设计 Architecture Design 去噪推测解码.md>) 与 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)，科学问题由 [研究计划](<研究计划 Research Plan 去噪推测解码.md>) 独立维护。

# 一、当前范围
基线 `5bacb4d`，在 `codex/structure-cleanup-20261105` 更新Draft PR106到dev。本轮约30分钟，只将模型所有权与KV值保全两项旧CPU数值合同迁到既有CUDA镜像，不改算法、生产接口或科学输入。

| **步骤与预估** | **目标** | **验收** |
| --- | --- | --- |
| 核对，5分钟 | 读取AGENTS、相关规范及远端RPipe main代码结构设计 | 固定远端414ccb7及职责镜像边界，不访问本机RPipe/dayan |
| 实施，8分钟 | 两项原ID接已有cuda_decode，独立快照防别名，细化元数据 | 原配置、5条断言和raises保留；完整state_dict及对象/存储保全，KV原位置[0,1,2,4]；后两函数及conftest保持 |
| 文档，4分钟 | 同步唯一计划、README、代码结构、研究边界与日期记录 | 不增加平行计划、生产模块或算法主张 |
| 验证，10分钟 | 净HEAD快速CPU、顺序native/CUDA、报告重建/来源拒绝及独立审查 | 945登记；四计划819/445/115/11函数；CPU1359、CUDA284质量+1测量、native11，原120/300/300秒预算不扩；实际CUDA设备落盘 |
| 交付，3分钟 | 普通推当前分支并回读Draft PR106 | 两checkout干净、目标dev、main/dev不改、不绕过必需检查、制品不入Git |

借用模型保持原LlamaConfig的16/8/16/1配置及全部未显式默认值（含max_position_embeddings2048），不替换为通用tiny工厂。使用CUDA初始化与embedding计算，比较完整state_dict 12项；native forward预期0。CPU RNG字节只用于状态合同检查，不算CPU数值替代。KV原输入/期望均CUDA，克隆发生在update前。两项从cpu-pr精确排除并进入cuda-validation；cpu-quick整个声明不改。GPU调度估计195秒只作准入参考，不是性能测量。

验证先确认torch.cuda.is_available和实际GPU，忙时等待、不CPU回退。正式净HEAD仅一次严格历史export/cross-Phi拒绝；四原件仍未知、20旧锁保持，只核已知日志/manifest/两索引，不扫refs/ACL/全盘、不补签。未执行/缺结果不记通过，完整cpu-pr与远端CI不冒称本地快速计划。

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
