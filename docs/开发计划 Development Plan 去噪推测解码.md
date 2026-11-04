**去噪推测解码 开发计划**
更新日期：2026年11月5日。当前工作依据 [架构设计](<架构设计 Architecture Design 去噪推测解码.md>) 与 [代码结构设计](<代码结构设计 Code Structure Design 去噪推测解码.md>)。研究主张与实验条件单独维护在 [研究计划](<研究计划 Research Plan 去噪推测解码.md>)。

# 一、当前实施范围
从最新 dev 创建工作分支，按职责收拢冗余代码和文档。结构参考本地 RPipe 的两柱，保留六阶段和实际需要的模块。算法由 Enmao 重新设计，本轮保持数学、默认参数、历史输入和观测合同。

# 二、实施与验收
| **任务** | **涉及文件与执行顺序** | **验收** |
| --- | --- | --- |
| 宿主剩余顶层模块落位 | data.py → structure/data/prompts.py，model.py → structure/model/hf_causal.py，audit.py → structure/artifact/observation.py，analysis.py → structure/algorithm/counts.py。先更新设计，再移动并更新全部当前调用方与 monkeypatch | 数据、模型、审计和统计无旧路径垫片，无失效导入，原行为和异常反例通过 |
| 命令职责收束 | 原 flow/cli.py → flow/study_cli.py，科学 cli.py → flow/cli.py，更新模块入口、控制台元数据、Study 调度生成器、测试及静态清单。flow 初始化只保留包标记 | 科学 help 不导入 Torch/Transformers，audit/compare 不创建会话，Study CPU smoke 路径仍可用 |
| 测试镜像与门禁 | 同步受影响测试、manifest 的 test_id/target、依赖合同和缩减清单，不增加 legacy 例外 | 当前源码上的有界 cpu-pr，逐参数结果可定位，缺结果不能记通过 |
| 研究计划整理 | 独立整理当前问题、六方面条件、暂停和恢复条件，历史公式与结论保留来源身份 | 研究计划不以现有代码推导新算法或虚构实验，不与开发待办混写 |
| Study 遗留收拢 | 后续按消费者归并顶层脚本与重复助手，迁移 trajectory_observation 的命令与实际声明。先读对应协议和历史消费者 | 不新增同职责小文件，legacy 只减，保留原始数据与冻结来源 |
| 旧流程 API facade | 后续核全部调用方，消除 api 向实现层反向依赖，保留真实构建与 tracker 行为 | import_direction 清单归零，真实构建与收尾反例通过 |

前三项与计划整理组成当前轮。Study 旧脚本、API facade 和尚未迁移的测试是明确剩余工作，不以文档改名宣称整体重构完成。每轮 20–30 分钟形成可检查成果，再报告下一步。

# 三、交付与检查边界
工作分支通过 PR 合回 dev，按最新用户授权完成集成，不直接推 dev，不合 main、不强推、不启用 Actions、不改 workflow 或仓库设置与保护。跨责任或公共合同变化仍需受影响 Owner 的平台评审，不能以本地自检冒充 Approve。

运行当前 HEAD 的离线 CPU 计划与受影响静态检查，结果绑定实际提交、基线、工作区差异和环境。真实模型、GPU、下载、大文件和新算法实验不在本轮范围。临时材料与验证根使用用户指定 DreamSoul/tmp 下的独立任务目录，不写 C 盘，不安装或升级依赖。

# 四、来源与长期约束
流程是本仓自有实现，RPipe 只作结构参考，不恢复内嵌副本。固定 ESP 来源树及 ORIGIN.json、原始 SpecBench 与历史 Run 保持身份和字节。开发决定见 [开发记录](<开发记录 Development Record/INDEX.md>)，实际命令和结果见 [运行记录](<运行记录 Run Record/INDEX.md>)。研究执行须按唯一研究计划另获当前授权。
