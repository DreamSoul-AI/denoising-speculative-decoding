# denoising-speculative-decoding 仓库约定



去噪推测解码的研究与工程维护。研究路线和执行条件以唯一研究计划、开发计划及下文现有边界为准。



## 研发入口



- [项目首页](README.md) 维护用途、安装和使用方式。

- [研发文档导航](docs/README.md) 定位现行设计、计划与历史记录。



## 一、项目与唯一文档



- GitHub 地址：[DreamSoul-AI/denoising-speculative-decoding](https://github.com/DreamSoul-AI/denoising-speculative-decoding)。本地位置以当前关联源码目录为准，不写死盘符、用户名或下载目录

- 项目中文名为“去噪推测解码”，已由 Enmao 确认。正文使用中文，源码标识、接口、命令与原始来源标题保留必要原文

- [README](README.md) 维护项目现状及入口，[通用调研 General Research](<docs/调研 Search/通用调研 General Search/通用调研 General Research 去噪推测解码.md>) 维护机制、比较、来源和当前判断。历史测量在运行记录

- [架构设计](<docs/架构设计 Architecture Design 去噪推测解码.md>) 维护目标职责、合同、状态、运行部署和验证，[代码结构设计](<docs/代码结构设计 Code Structure Design 去噪推测解码.md>) 唯一维护目标目录、Python 文件责任、依赖与测试镜像

- [研究计划](<docs/研究计划 Research Plan 去噪推测解码.md>) 是唯一中文计划，Study 按研究问题分章。旧英文计划与冻结子计划只从固定 Git 历史取证，不恢复并行正本

- [文献调研](<docs/调研 Search/文献调研 Literature Search/文献调研 Literature Review 去噪推测解码.md>)、[开发计划](<docs/开发计划 Development Plan 去噪推测解码.md>)、[缺陷记录](<docs/缺陷记录 Bug Report 去噪推测解码.md>)、[开发记录](<docs/开发记录 Development Record/2026-10-31 工程重构与文档重整.md>) 和 [运行记录](<docs/运行记录 Run Record/2026-10-31 历史证据归位.md>) 分别维护学术证据、工程待办、活动缺陷、演进和实际结果

- 上述根入口和专项文档继续作为现有入口，文档角色与命名按 AGENTS.md 的文档分工与研发迭代章节维护。研究报告按职责归属：学术证据归文献调研，工程机制归通用调研，开发决定与演进写开发记录，实际执行从运行记录定位，提交与 PR 引用对应记录。Study、测试和工具不另建 README、PLAN、NUMBERS、审查、交接或逐事报告

- 从目标与质量场景推导职责、接口、状态和依赖，现有实现只作为行为和故障证据。按奥卡姆剃刀说明每个模块的必要性、不增加的后果和持续成本，不以旧文件树倒推目标设计

- 图示保留在正文 Mermaid 中，区分源码依赖、运行调用和部署。图语法检查不能替代合同或实现验证



## 四、固定依赖、数据与执行边界



- 仓库不再内嵌 `src/rpipe/`，也不设顶层 `tools/`。Study 流程在 `spec_mtp` 内自有实现，目录以 RPipe 为结构模板持续对齐。通用流程缺陷改本仓实现，并对照该模板；第三方 ESP 来源树及 `ORIGIN.json` 保持字节

- `src/spec_mtp/structure/algorithm/reference/esp_reproduction_artifacts/upstream/` 的 README、code、results 整体及旁侧 `ORIGIN.json` 保持字节，所有版权许可证保留。第三方原码不能称作者官方实现

- 原始 `datasets/spec_bench/` 保留，已渲染输入、日志、trace、图和运行源码快照按制品管理，不能据生成物清理授权扩展删除其它源码或正式数据

- 科学宿主 `spec_mtp` 与每个 `studies/<name>/` 按 `structure/`、`flow/` 两柱组织。新增机制、函数、类和脚本必须位于这两柱之内；顶层只有 `__init__.py`、`__main__.py` 和薄入口。要改这条规则，先改代码结构设计，再改代码。现有违反冻结在 CPU 门禁的 legacy 清单，清单只减不增。保留默认六阶段，不复制整套库或无职责空壳

- 新执行先确认研究范围并更新唯一计划，以新 version/Run 冻结实际源码、输入、配置和模型。旧 Run 不覆盖、不重签，来源不明保留未知

- T0 正确性门为完整提交链的验证前向 argmax 审计，独立 AR 一致率只诊断。接受数/τ、数值自检和速度收益分别报告。采样分布保证未完成验证，不能由 T0 推广

- ESP 旧长矩阵暂停，七格范数与11格独立配对不自动接卡。BF16 相对独立 AR 的算子数值机理排查永久停止。2026年10月20日确认的 K=[A,b] 仿射残差校正作为独立 Study，先 CPU 小规模、再冻结 K 跨数据集，无校正/仅 bias/完整 K 三臂；它不是已实现 Kalman 或已证泛化。代码重构只做有界离线 CPU 工程验证，历史命令、计划、登录态不构成新 GPU、下载或测速授权

- 未来获准执行时复用 ModelScope，不覆盖既有 CUDA torch；设备和环境路径现场定位。取消先阻止新派发，已执行副作用与未知结果保留，不盲目重试



## 五、协作、平台与完成检查



- 当前架构重构只推送本任务隔离工作分支，完成当前 HEAD 本地有界 CPU 回归，创建面向 `dev` 的代码 Draft PR 后等待评审，不合并、不强推、不启用 autoMerge、不发布。既往普通开发授权不扩大当前范围

- 写入前重读目标、保留他人修改。运行、验证和临时材料留任务临时目录或显式外部运行根，长期正文只保留一份必要摘要

- 飞书操作用对应技能和 `lark-cli --as user`，先核身份和用法，失败区分环境/登录/资源权限，不静默换机器人，不自行切浏览器

- 飞书引用沿用实际取得的完整分享链接和锚点。同步、上传、发消息和邀请按本次授权，不凭旧链接自动执行

- 完成检查编码、标题、链接、表格、必要 Mermaid 解析与 diff 范围。旧文档/证据消费者已知失效应明确报告，不能宣称全量测试或远端 CI 通过



- 本任务不修改 Actions、CI/CD、workflow 或仓库设置，不读取/创建凭据。验证产物不进 Git。



## 共享资源入口

- 本工作树通过 Knowledge 的 shared-repository-resources 工具接入，本机配置为未跟踪的 `.shared/shared-resources.config.json`。`.shared/data`、`.shared/models` 分别直连共享池 data、models；只将 `.shared/artifacts` 按仓库隔离。data/models 下不加仓库、Study 或研究 topic 层，按提供方、资源身份与版本组织。
- 已接入 `.shared` 时，ArtifactLayout 的 data/model 路径及 Study 的 ModelScope/HF 下载消费者使用这些共享入口（模型缓存为 `.shared/models/modelscope` 或 `hf`，数据缓存为 `.shared/data/hf_datasets`）。独立、未接入共享池的 Study 保留旧布局；损坏资源链接报错，不悄悄重建本地副本。历史 Run 和冻结脚本的旧绑定不改写。

- 使用共享资源时，从 [Knowledge](https://github.com/DreamSoul-AI/knowledge) 查找共享资源工具，按 Workspace 配套示例将数据、模型与项目产物接入 `.shared/`。本机路径从个人配置读取，实际链接与消费者按项目核对。

## Workspace 采用记录



- 来源 [Workspace 098bd2e](https://github.com/DreamSoul-AI/workspace/tree/098bd2eccf23eb1e6d77a3eec6db51e6bc387daf)。采用范围为Learnings 命名与分发、文档规范配置、长作业检查点与恢复、调研、研究计划与研究结论归档、发布前分支对齐、可选共享资源入口。项目专用流程见本文件。

- 单项更新：[Workspace 65657c5](https://github.com/DreamSoul-AI/workspace/tree/65657c50cf6d98ac1898ae0183cb9ca98498e0a9)。采用旧对话重读、新对话定位、文档主动补链接、浏览器收尾、飞书专节及 CLI 任务临时目录、共享资源主动检查；保留项目专有约定和原整版基线。共享 data/models 直接复用，仅 artifacts 按仓库隔离，接入示例见 [Workspace 教程](https://github.com/DreamSoul-AI/workspace/blob/main/templates/software-engineering/examples/shared-resources/README.md)。

- 单项更新：[Workspace f01fab8](https://github.com/DreamSoul-AI/workspace/tree/f01fab8598f570a45eca8320e546d636268b700a)。采用通用文档规范及飞书/GitHub 平台章节入口、首次接入默认值和已有选择保留、代码架构设计与代码结构设计并列命名；图表简洁要求随通用规范读取。保留本项目专有约定、既有文档路径及原整版基线。[规范分发副本](https://github.com/DreamSoul-AI/workspace/blob/f01fab8598f570a45eca8320e546d636268b700a/%E6%96%87%E6%A1%A3/%E8%A7%84%E8%8C%83/%E9%80%9A%E7%94%A8/%E9%80%9A%E7%94%A8%E6%96%87%E6%A1%A3%E8%A7%84%E8%8C%83.md)。
- 单项更新：[Workspace 823d496](https://github.com/DreamSoul-AI/workspace/tree/823d4965035899d66a2d57751f1d28fbf934f99a)。采用临时产物与共享入口分工、资料根定位、目录升级检查、双端文档先改本地及面向读者的汇报；源码、测试、项目 docs 与专有约定保留，原整版基线继续沿用。
- 单项更新：[Workspace ea16de5](https://github.com/DreamSoul-AI/workspace/tree/ea16de5eda676aadaeefc1897ce0b83c062de2b6)。采用常规问题直接处理、结合目标和进度主动建议下一步；Knowledge 能力更新沿 [Workspace 教程](https://github.com/DreamSoul-AI/workspace/blob/main/资料/教程和工具/使用教程.md#workspace-与-knowledge-更新) 默认接入，保留通用与个人来源、项目专有约定和原整版基线。
- 单项更新：[Workspace f020aa5](https://github.com/DreamSoul-AI/workspace/tree/f020aa5b6965766f7308fa31b0836e478c9da914)。采用加粗只包文字、所有标点放在外面的统一写法，移除未采用的本机配置默认目录及专门忽略项；配置位置和相应保护沿用项目约定，保留其他规则及原整版基线。
- 单项更新：[Workspace a9da11e](https://github.com/DreamSoul-AI/workspace/tree/a9da11e30412aa3ec52a88cf76150da129b5063b)。文档按用途和目标读者组织正文，补齐独立阅读所需的背景与依据；保留原整版采用基线及项目约定。
- 本轮适用更新：[Workspace 84f9696](https://github.com/DreamSoul-AI/workspace/tree/84f969679eb06b7d742ee7b384c13e371df9d425)。核对并合并当前适用的共用规则，补充新对话项目选择和按场景理解输入的约定，保留项目专有内容、个人配置、现有布局及原整版采用基线。
