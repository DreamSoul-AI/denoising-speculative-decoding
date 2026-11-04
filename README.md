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

算法由 Enmao 重新设计，当前进行结构整理与有界 CPU 验证。新算法实验、GPU 长跑、测速和模型下载暂停，旧协议仅作为历史证据。每轮从 dev 工作分支经 PR 集成，不直接修改 main。

第三方 ESP 来源树及 `ORIGIN.json` 保留，不能把它称为作者官方实现。原始 SpecBench 数据保留于 `datasets/spec_bench/question.jsonl`。

科学 CLI 的 `decode` 必须显式提供 `--model-sources <JSON>`：清单位于模型目录外，以相对路径映射该完整、自包含的本地模型目录内全部文件的 SHA256，不缺项、不含目录外成员。输出也放在模型根外；解析后逃逸模型根的符号链接会被拒绝，不直接承诺兼容引用外部 blobs 的 Hub 快照。载模前、运行头及结束记录写出前分块读取核验实际文件；漂移时保留部分观测，不写完成末行。运行头另记实际模型 config、backend、device、dtype 和能力，调用者原 `sources` 标签保持原义。审计带模型来源的新记录时使用 `audit <raw> --sources <标签JSON> --model-sources <原模型清单JSON>`；缺少独立模型清单不能记来源完整。旧无模型来源字段的记录仍只比较原标签，不代表重新验证实际模型。`audit/compare` 不加载模型或重新生成观测。

旧逐条运行证据、日志、图和报告保留在固定历史/外部运行根，未恢复进 Git。历史读端测试从固定 Git 提交 `b8b20b5ada7fd860894114cbf224ce749af430cf` 按需提取原始 blob 到测试临时目录，核对对象哈希与写入字节；缺少 Git 对象直接失败，不下载或伪造原件。当前分析器和校验门读取这些不可变输入，不导入或执行归档源码，不修改原哈希与断言；相关15个参数项已恢复到 CPU 计划，不再排除。此验证只复核保存的历史观测与读端合同，不重新运行历史模型，也不代表全部历史结果已独立复验。

离线报告使用 `python -m studies._shared.flow.cli report <Study> --out <外部新文件.md>`，输出路径必填，拒绝已有文件、输入目录和源码 checkout；报告检查不回写旧 process 或 gate。`output_space_extension` 的已保存槽位诊断迁入 `python -m studies.output_space_extension.flow.process.run --raw <保存张量目录> --execution <原运行矩阵.json> --repository <本地仓库> --commit <完整提交> --execution-git-sha256 <Git原件SHA256> --output-root <新外部目录>`。它只读原件并加载 CPU 张量，不载入模型；实际运行矩阵须匹配旧报告哈希，与固定 Git 原件分别记录字节身份并核对 JSON 内容。缺失或漂移直接失败，不自动修换行或重签旧来源。其余旧完整分析器仍待逐项迁移。

诊断日志新写 `assets/logs/events.jsonl`，状态、日志与报告读端兼容旧 `run.log`。应用启动与 Run 身份分开，普通日志不保存异常原文或模型输入。流式文件哈希兼容 Python 3.10 所需 API，但最低版本及全部依赖组合尚未实机验证。

# 三、开发与验证
工作分支经 PR 面向 `dev`，完成检查并按当次明确授权集成，不自动发布。当前 HEAD 的本地结果、历史红项和未覆盖范围记在 PR 正文，不用旧提交结果代替。

执行离线 CPU 计划：`python tests/run.py --plan cpu-pr --output-root <源码树外的新验证目录>`。静态清单先验证必需项和精确排除，逐参数 setup/call/teardown 写独立记录，缺收集、缺终态、失败、跳过和证据写失败均非零。整批预算420秒，在测试边界检查；子进程测试各有有限超时，单个本地计算不支持强制中断。运行产物不进 Git。

本轮不调整现有 Actions、CI/CD 或设置。远端结果仅作参考，本地 CPU 回归不覆盖 CUDA、真实权重/数据、吞吐或采样分布保证。已有 CUDA 环境保持原样，不下载模型或覆盖 torch。

## 研发维护与文档现状

Python 3.10+、PyTorch 与固定 Transformers 条件见 [pyproject.toml](pyproject.toml)，当前研究宿主为 `spec_mtp`。目录入口：[产品或库源码](src/spec_mtp/)、[研究声明与适配](studies/)、[测试](tests/)、[当前文档](docs/)。运行与测试按本 README、相关设计和实际配置选择，不把历史命令视为新的实验授权。

文档职责按 AGENTS.md 维护。通用调研不代替文献调研，研究计划不代替开发计划。历史测量在运行记录，结构调整在开发记录。本轮从已关联 DreamSoul 资料库读取研发根约定及相关代码、协作、测试、研究计划、文献、文档和架构规范。本地资料的读取不代表云端同步或最新远端核验。
