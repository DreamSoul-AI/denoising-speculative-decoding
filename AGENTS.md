# 去噪推测解码 项目协作约定
适用于本仓库及其子目录。当前用户要求优先，开始前读取目标文件、相关规范和已有成果的最新内容。

# 一、项目与唯一文档
- GitHub 地址：[DreamSoul-AI/denoising-speculative-decoding](https://github.com/DreamSoul-AI/denoising-speculative-decoding)。本地位置以当前关联源码目录为准，不写死盘符、用户名或下载目录
- 中文建议名“去噪推测解码”待 Enmao 汇总确认。正文使用中文，源码标识、接口、命令与原始来源标题保留必要原文
- [README](README.md) 维护项目现状及入口，[通用调研](<docs/去噪推测解码 通用调研.md>) 维护机制、比较、来源和历史验证摘要
- [架构设计](<docs/去噪推测解码 架构设计.md>) 维护目标职责、合同、状态、运行部署和验证，[代码结构设计](<docs/去噪推测解码 代码结构设计.md>) 唯一维护目标目录、Python 文件责任、依赖与测试镜像
- [研究计划](<docs/去噪推测解码 研究计划.md>) 是唯一中文计划，Study 按研究问题分章。旧英文计划与冻结子计划只从固定 Git 历史取证，不恢复并行正本
- 第一方长期文档只保留上述根入口和四份文档。研究报告价值内容归入通用调研，日常进度写提交和 PR。Study、测试和工具不另建 README、PLAN、NUMBERS、审查、交接或逐事报告
- 从目标与质量场景推导职责、接口、状态和依赖，现有实现只作为行为和故障证据。按奥卡姆剃刀说明每个模块的必要性、不增加的后果和持续成本，不以旧文件树倒推目标设计
- 图示保留在正文 Mermaid 中，区分源码依赖、运行调用和部署。图语法检查不能替代合同或实现验证

# 二、如何定位 DreamSoul 资料
1. 从当前项目关联的文件夹、README 资料链接或工作区入口找到 DreamSoul 文档库，按实际可访问位置读取根约定和相关索引，不假定任何本机绝对路径
2. 按任务读取现有成果、对应规范正文与最新交接。通常从相对于实际资料库的 `文档/规范/研发/开发/`、`文档/规范/研发/研究/` 和 `文档/规范/飞书/` 定位
3. 架构任务再定位 `研发/研究/软件开发 Software Engineering/软件架构设计 Software Architecture Design/`，读取通用调研、大纲、端侧智能体案例与来源索引。采用其场景、黑盒合同、多视图和验证方法，按本项目裁剪
4. 本地已关联正文时直接使用，需确认云端最新内容时沿原分享链接读取。没有本地资料库时，使用下列飞书索引定位，不复制重复资料库
5. 路径或链接失效时先查当前位置。无法定位或没有权限就说明缺口和影响，继续不依赖它的工作，不按同名猜链接或伪称读过正文

# 三、规范索引与按任务读取
总入口为 [飞书规范总索引](https://dreamsoul.feishu.cn/docx/UmU1d2EwLoeET5xCJChcOKPknyb) 与 [工具与教程总索引](https://dreamsoul.feishu.cn/docx/VSQ5dlJl3o0hbIxrEhUc8qKEnBc)。索引用于定位，细则以实际正文的范围、修订和项目状态为准。

| **任务类型** | **开始前读取** | **落实重点** |
| --- | --- | --- |
| 代码开发与修复 | [代码规范](https://dreamsoul.feishu.cn/docx/FKrsdHO9bolq4YxV2KAcosghnme)及相关实现/调用方 | 保留真实输入与持久化校验，复用既有依赖，最小完整改动，避免隐式状态 |
| 多人协作、分支与 PR | [多人协作代码开发规范](https://dreamsoul.feishu.cn/docx/WihCdOtlqoeYjaxrpeXckFomnCh) | 工作分支 → PR → dev，再由 dev 发布 PR → main，跨责任/合同变化请受影响 Owner 评审 |
| 测试编写与执行 | [测试规范 代码](https://dreamsoul.feishu.cn/docx/KE4IdW6EWoKdvExSkdvcjTJFnsf) | 镜像源码，五维元数据，独立参数结果，按影响/成本选择，未执行与缺结果不记通过 |
| 日志与运行取证 | [日志规范 Logging Specification](https://dreamsoul.feishu.cn/docx/RBlFdcSSBo93q9x6p8kcdU6wnYb) | 记录实际运行身份/事件/结果，业务证据与诊断日志分开，未知不填假零，排除敏感载荷 |
| CI/CD、构建、交付与发布 | [CI CD 执行规范](https://dreamsoul.feishu.cn/docx/NrJ4d0WVGo9t4uxAVwScFTBVnRb) | 验证绑定提交/基线/环境，失败与漏收集阻断，本地检查和自动流水线分开，不擅改保护规则 |
| 技术、工具与架构调研 | [通用调研规范](https://dreamsoul.feishu.cn/docx/YcWIdE4INoLqoLxR5O8cbl7hnTf) | 问题驱动，解释机制和代价，直接来源就近引用，固定版本/阅读深度/未知 |
| 论文与学术研究 | [文献调研规范](https://dreamsoul.feishu.cn/docx/QKXhdAW2doZCvUxHSoFcpT0Mn5e) | 原论文与官方实现优先，按问题组织，分开论文主张、自实现、第三方、实际测量 |
| 研究计划与实验设计 | [研究计划规范](https://dreamsoul.feishu.cn/docx/POdQdHYeqo7ujpxmJH1ctOMPn1g) | 保留建议采用状态，连接文献、数据、模型、算法、指标和对照，写明暂停及恢复条件 |
| 中文文档与排版 | [飞书文档规范](https://dreamsoul.feishu.cn/docx/DdmxdrNvBo9AMXxkAYmc4EJ6nwe) | 中文层级、混排、加粗表头、就近引用及正式内容清理，实际上传时再验原生引用和文本绘图 |

入口齐全不代表全部规范已落地。当前任务只应用实际读取且适用的要求，不加载无关历史限制。

# 四、固定依赖、数据与执行边界
- `src/rpipe/` 全树（含其 docs 和许可）固定于 ORIGIN 的 v0.3.1 提交，既存 Windows 差异保留。下游不改副本，通用缺陷在 RPipe 仓独立修复后另行确认完整同步
- `src/spec_mtp/structure/algorithm/reference/esp_reproduction_artifacts/upstream/` 的 README、code、results 整体及旁侧 `ORIGIN.json` 保持字节，所有版权许可证保留。第三方原码不能称作者官方实现
- 原始 `datasets/spec_bench/` 保留，已渲染输入、日志、trace、图和运行源码快照按制品管理，不能据生成物清理授权扩展删除其它源码或正式数据
- 科学宿主的目标为标准 Python 包，只有 Study 使用 RPipe 声明和实际需要的 `structure/flow` 薄适配。保留默认六阶段，不复制整套库或无职责空壳
- 新执行先确认研究范围并更新唯一计划，以新 version/Run 冻结实际源码、输入、配置和模型。旧 Run 不覆盖、不重签，来源不明保留未知
- T0 正确性门为完整提交链的验证前向 argmax 审计，独立 AR 一致率只诊断。接受数/τ、数值自检和速度收益分别报告。采样分布保证未完成验证，不能由 T0 推广
- 当前实验已暂停，七格范数与11格独立配对不自动接卡。BF16 相对独立 AR 的算子数值机理排查永久停止。历史命令、计划、登录态不构成新 GPU、下载或测速授权
- 未来获准执行时复用 ModelScope，不覆盖既有 CUDA torch；设备和环境路径现场定位。取消先阻止新派发，已执行副作用与未知结果保留，不盲目重试

# 五、协作、平台与完成检查
- 当前任务只推送文档工作分支，创建面向 `dev` 的 Draft PR 后等待评审，不合并、不强推、不启用 autoMerge、不发布。既往普通开发授权不扩大当前范围
- 写入前重读目标、保留他人修改。运行、验证和临时材料留任务临时目录或显式外部运行根，长期正文只保留一份必要摘要
- 飞书操作用对应技能和 `lark-cli --as user`，先核身份和用法，失败区分环境/登录/资源权限，不静默换机器人，不自行切浏览器
- 飞书引用沿用实际取得的完整分享链接和锚点。同步、上传、发消息和邀请按本次授权，不凭旧链接自动执行
- 完成检查编码、标题、链接、表格、必要 Mermaid 解析与 diff 范围。旧文档/证据消费者已知失效应明确报告，不能宣称全量测试或远端 CI 通过
