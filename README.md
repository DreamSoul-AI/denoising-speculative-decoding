# 去噪推测解码
研究免训练多 token 起草与验证：以 ESP 为参照，比较输入嵌入 EMA、速度外推和输出隐藏状态掩码。中文名称为“去噪推测解码”。仓库地址：[DreamSoul-AI/denoising-speculative-decoding](https://github.com/DreamSoul-AI/denoising-speculative-decoding)。

截至2026年10月21日的 8B 非官方 Table 1 独立复验已经完成，论文目标未复现。数字和原件在 [历史证据归位](<docs/运行记录 Run Record/2026-10-31 历史证据归位.md>)。提交 token 的验证前向 argmax 自检与独立 AR 一致率分别报告，τ 是提交数与解码调用数之比，不能据此认定实际速度收益。

已实现的 K 校正是协议 `postcommit-final-norm-v1` 的合成工程夹具，不是 2026年10月31日会议中的 trajectory 与 observation 联合更新，也不是 Kalman 滤波。160 对数据协议和真实模型收益都还没有执行。会议方法的默认公式和合成测量见 [研究计划](<docs/研究计划 Research Plan 去噪推测解码.md>) 与 [联合 mask 检查](<docs/运行记录 Run Record/2026-11-01 联合mask检查.md>)。

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
| [开发记录 Development Record](<docs/开发记录 Development Record/2026-11-04 首分叉记录.md>) | 首分叉记在 `office_traj.py`，见 [2026-11-04 首分叉记录](<docs/开发记录 Development Record/2026-11-04 首分叉记录.md>)。此前顶层再导出删除见 [2026-11-04 删除 masks 与 decode 顶层再导出](<docs/开发记录 Development Record/2026-11-04 删除 masks 与 decode 顶层再导出.md>)。解码循环见 [2026-11-04 解码循环迁入 loop](<docs/开发记录 Development Record/2026-11-04 解码循环迁入 loop.md>)。完整目录见 [INDEX](<docs/开发记录 Development Record/INDEX.md>)。 |
| [运行记录 Run Record](<docs/运行记录 Run Record/2026-11-04 Qwen3-8B 首分叉.md>) | 最近一次是 `office-ar-first-fork-v1`：summarization 128 条的共享分叉比例 0.546875，fp32 参照未加载。见 [2026-11-04 Qwen3-8B 首分叉](<docs/运行记录 Run Record/2026-11-04 Qwen3-8B 首分叉.md>)。此前失败子集在 [2026-11-04 Qwen3-8B trajectory失败子集](<docs/运行记录 Run Record/2026-11-04 Qwen3-8B trajectory失败子集.md>)。此前墙钟在 [2026-11-04 Qwen3-8B trajectory墙钟](<docs/运行记录 Run Record/2026-11-04 Qwen3-8B trajectory墙钟.md>)。完整目录见 [INDEX](<docs/运行记录 Run Record/INDEX.md>)。 |

# 二、当前实现与使用边界
当前安装包为 `denoise-spec-mtp`。`spec_mtp` 与每个 `studies/<name>/` 的目标布局都是 [RPipe](https://github.com/DreamSoul-AI/RPipe) 的 `structure/` 与 `flow/` 两柱：`structure` 定义一次 Run 由什么组成，`flow` 运行阶段链。顶层只保留 `__init__.py`、`__main__.py` 和薄入口。新增机制必须先落在这两柱里；要改这条规则，先改代码结构设计。解码合同在 `structure/api/decode_api.py`。树构造、节点编辑、树注意力和解码循环在 `structure/algorithm/decode/`。mask provider、K 拟合和注册表在 `structure/algorithm/mask/`。`contracts.py`、`tree.py`、`masks.py` 与 `decode.py` 已删除，调用方导入这些真实路径。其余解码内容仍在 `data.py`、`model.py`、`audit.py`、`analysis.py`、`cli.py`，Study 顶层还有一批历史脚本。这些仍是冻结遗留，迁移分步进行。流程骨架已经在 `spec_mtp.flow` 与 `spec_mtp.structure`，不再内嵌 `src/rpipe/` 副本，也不再 `import rpipe`。`python -m spec_mtp` / `spec-mtp` 提供 `decode/audit/compare`；旧 make/launch/process 研究命令明确拒绝并提示使用 `python -m studies._shared.flow.cli`。

第三方 ESP 来源树及 `ORIGIN.json` 保留，不能把它称为作者官方实现。原始 SpecBench 数据保留于 `datasets/spec_bench/question.jsonl`。

科学 CLI 的 `decode` 必须显式提供 `--model-sources <JSON>`：清单位于模型目录外，以相对路径映射该完整、自包含的本地模型目录内全部文件的 SHA256，不缺项、不含目录外成员。输出也放在模型根外；解析后逃逸模型根的符号链接会被拒绝，不直接承诺兼容引用外部 blobs 的 Hub 快照。载模前、运行头及结束记录写出前分块读取核验实际文件；漂移时保留部分观测，不写完成末行。运行头另记实际模型 config、backend、device、dtype 和能力，调用者原 `sources` 标签保持原义。审计带模型来源的新记录时使用 `audit <raw> --sources <标签JSON> --model-sources <原模型清单JSON>`；缺少独立模型清单不能记来源完整。旧无模型来源字段的记录仍只比较原标签，不代表重新验证实际模型。`audit/compare` 不加载模型或重新生成观测。

旧逐条运行证据、日志、图和报告保留在固定历史/外部运行根，未恢复进 Git。历史读端测试从固定 Git 提交 `b8b20b5ada7fd860894114cbf224ce749af430cf` 按需提取原始 blob 到测试临时目录，核对对象哈希与写入字节；缺少 Git 对象直接失败，不下载或伪造原件。当前分析器和校验门读取这些不可变输入，不导入或执行归档源码，不修改原哈希与断言；相关15个参数项已恢复到 CPU 计划，不再排除。此验证只复核保存的历史观测与读端合同，不重新运行历史模型，也不代表全部历史结果已独立复验。

# 三、开发与验证
工作分支经 PR 面向 `dev`，完成检查并按当次明确授权集成，不自动发布。当前 HEAD 的本地结果、历史红项和未覆盖范围记在 PR 正文，不用旧提交结果代替。

执行离线 CPU 计划：`python tests/run.py --plan cpu-pr --output-root <源码树外的新验证目录>`。静态清单先验证必需项和精确排除，逐参数 setup/call/teardown 写独立记录，缺收集、缺终态、失败、跳过和证据写失败均非零。整批预算420秒，在测试边界检查；子进程测试各有有限超时，单个本地计算不支持强制中断。运行产物不进 Git。

本轮不调整现有 Actions、CI/CD 或设置。远端结果仅作参考，本地 CPU 回归不覆盖 CUDA、真实权重/数据、吞吐或采样分布保证。已有 CUDA 环境保持原样，不下载模型或覆盖 torch。

## 研发维护与文档现状

Python 3.10+、PyTorch 与固定 Transformers 条件见 [pyproject.toml](pyproject.toml)，当前研究宿主为 `spec_mtp`。目录入口：[产品或库源码](src/spec_mtp/)、[研究声明与适配](studies/)、[测试](tests/)、[当前文档](docs/)。运行与测试按本 README、相关设计和实际配置选择，不把历史命令视为新的实验授权。

文档职责按 AGENTS.md 维护。通用调研不代替文献调研，研究计划不代替开发计划。历史测量在运行记录，结构调整在开发记录。飞书规范正文本次没有读到，入口索引不能代替那些正文。
