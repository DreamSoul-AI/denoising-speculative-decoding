**去噪推测解码 代码结构设计**
更新日期：2026年11月4日。状态：目标布局改为 `structure` / `flow` 两柱。行为合同由 [架构设计](<架构设计 Architecture Design 去噪推测解码.md>) 维护。按下面第四节分步搬迁，现有违反冻结在该节的 legacy 清单里。步骤 1 已把解码合同放到 `structure/api/decode_api.py`，`contracts.py` 只再导出原名字。步骤 2 已把树拆到 `structure/algorithm/decode/`，`tree.py` 只再导出原名字。步骤 3–8 未做。

# 一、目录原则与范围
`spec_mtp` 与每个 `studies/<name>/` 都按 RPipe 的两柱组织：`structure/` 定义一次 Run 由什么组成，`flow/` 运行 Study 的阶段链。除 `__init__.py`、`__main__.py` 与薄入口外，`src/spec_mtp/` 和 `studies/<name>/` 的顶层不得再有其他 `.py` 文件。`studies/__init__.py` 只做 Study 包标记。依赖下限保持 Python 3.10 与已固定的 Transformers，目录调整不授权升级依赖。

**规则（硬性）**：所有新增的机制、函数、类、脚本，必须位于 `structure/` 或 `flow/` 之内。顶层不得新增模块；不得以“临时脚本”“分析脚本”“pilot”为名在 structure/flow 之外，或在 `structure/` 里放独立 `main()` 脚本。唯一例外是薄入口：只做参数解析并转调 `flow` 的函数，不含算法、不含统计、不含数据读写，且不超过约 30 行。要放宽这条规则，先改本文，再改代码。

新增一段代码时依次问：

1. 它是“怎样做一次计算/怎样组成一次 Run”吗？是 → `structure`。
2. 它是“按阶段把很多次 Run 跑起来、收集、汇总、写出、做 Study 级处理”吗？是 → `flow`。
3. 两者都不是 → 不加。先回到本文补定义，再写代码。

历史 Study 声明保留在 `studies/`。第三方 ESP 树保持原字节。Study 流程在 `spec_mtp/flow` 与 `spec_mtp/structure`，以 https://github.com/DreamSoul-AI/RPipe 为结构模板，不再有内嵌 `src/rpipe/` 或顶层 `tools/`。

# 二、目标目录
下面是目标布局。`structure/` 与 `flow/` 里已有的 RPipe 阶段骨架保持不动。解码合同已经在 `structure/api/decode_api.py`。树已经在 `structure/algorithm/decode/`。其余解码模块仍在扁平路径，见第五节。

```text
denoising-speculative-decoding/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── src/
│   ├── spec_mtp/
│   │   ├── __init__.py                      # 包标识，导入不执行
│   │   ├── __main__.py                      # 薄入口，转调 flow
│   │   ├── structure/
│   │   │   ├── api/                         # 契约与协议，不含实现
│   │   │   ├── control/                     # Run 身份、配置、生命周期
│   │   │   ├── data/                        # 数据源、模板、渲染、划分
│   │   │   ├── model/                       # 模型加载与会话
│   │   │   ├── algorithm/                   # 计算：解码、mask、拟合、指标
│   │   │   │   └── reference/esp_reproduction_artifacts/
│   │   │   │       ├── ORIGIN.json
│   │   │   │       └── upstream/            # 第三方原树整体，保持现有成员及字节
│   │   │   ├── system/                      # 设备、运行时、缓存与临时目录
│   │   │   ├── artifact/                    # 产物写出、索引、读取、溯源
│   │   │   └── make/                        # Recipe、注册、构建
│   │   └── flow/
│   │       ├── prepare/ execute/ collect/ summarize/ write/ process/
│   │       └── cli.py                       # decode/audit/compare 的命令实现
│   │   # 搬迁完成后，顶层不再保留 cli/contracts/data/model/decode/masks/tree/audit/analysis.py
├── studies/
│   └── k_affine_residual/        # 合成夹具；协议 postcommit-final-norm-v1
│       ├── study.yaml
│       ├── experiment_config.yaml
│       ├── structure/
│       │   ├── __init__.py
│       │   ├── recipe.py
│       │   ├── correction.py
│       │   └── evaluation.py
│       └── flow/
│           ├── __init__.py
│           ├── collect/
│           │   ├── __init__.py
│           │   └── run.py
│           ├── summarize/
│           │   ├── __init__.py
│           │   └── run.py
│           └── process/
│               ├── __init__.py
│               └── run.py
├── datasets/spec_bench/question.jsonl     # 当前原始数据，保持字节
├── tests/
│   ├── run.py                            # 测试选择/结果适配，不是另一 runner
│   ├── manifest.json                     # 静态测试元数据与计划，不是运行证据
│   ├── spec_mtp/
│   │   ├── test_main.py
│   │   ├── test_cli.py
│   │   ├── test_contracts.py
│   │   ├── test_data.py
│   │   ├── test_model.py
│   │   ├── test_decode.py
│   │   ├── test_masks.py
│   │   ├── test_tree.py
│   │   ├── test_audit.py
│   │   ├── test_analysis.py
│   │   ├── test_dependencies.py
│   │   └── test_structure_flow_gates.py   # 两柱门禁；测试路径在步骤 8 前不随源码迁移
│   ├── studies/k_affine_residual/
│   │   ├── structure/
│   │   │   ├── test_recipe.py
│   │   │   ├── test_correction.py
│   │   │   └── test_evaluation.py
│   │   └── flow/
│   │       ├── collect/test_run.py
│   │       ├── summarize/test_run.py
│   │       └── process/test_run.py
│   └── tools/test_run_tests.py
└── docs/
    ├── 通用调研 General Research 去噪推测解码.md
    ├── 文献调研 Literature Review 去噪推测解码.md
    ├── 架构设计 Architecture Design 去噪推测解码.md
    ├── 代码结构设计 Code Structure Design 去噪推测解码.md
    ├── 研究计划 Research Plan 去噪推测解码.md
    ├── 开发计划 Development Plan 去噪推测解码.md
    ├── 缺陷记录 Bug Report 去噪推测解码.md
    ├── 开发记录 Development Record/
    └── 运行记录 Run Record/
```

目录树展开目标科学模块、K Study 和测试的落点。其它既有 Study 声明、诊断脚本和旧实现当前保留。纯命名空间 `__init__.py` 及固定第三方依赖不递归生成同名测试，包导入副作用由 `test_main.py` 验证。第三方路径按现有字节保留，不为标准布局移动来源。

不在 Study、tests 或 datasets 下规划 README、PLAN、NUMBERS 或报告。排除项与测试说明在本文和架构设计，研究安排只在唯一中文计划，运行证据放外部运行根。

每个 Study 的目标是具备 `study.yaml`、`experiment_config.yaml`、`structure/`（仅该 Study 专属的配置与适配）和 `flow/<phase>/run.py`。顶层 `.py` 只允许 `recipe.py` 与 `__init__.py`，以 `studies/_template` 为准。`trajectory_observation` 仍然没有 `study.yaml`。仍带 `main()` 的旧测量脚本继续留在 legacy 清单里。`fresh_replace.py` 的选句、汇总和命令都在 `structure/fresh_replace.py`。Qwen3-8B 这一轮的选句、拟合、比较和命令都在 `structure/office_qwen8b.py`。同预算树形探索的选句、形状、汇总和命令都在 `structure/office_tree.py`。Qwen3-8B 的同预算树形探索复用 `structure/office_tree.py`，选句常量仍在该文件，命令加 `--model 8b`，不另建脚本。trajectory-only 的尾段与全部跨类协议都复用 `structure/office_traj.py`，不另建脚本，不增加 legacy 条目；选句常量和已冻结哈希都在该文件。

| **protocol** | **slice** | **notes** |
| --- | --- | --- |
| 默认 | creative_writing 尾段 | 哈希已冻结 |
| `xcat` | open_qa | 哈希已冻结 |
| `xcat2` | general_qa | 哈希已冻结 |
| `xcat3` | classification | 哈希已冻结 |
| `xcat4` | closed_qa | 哈希已冻结 |
| `xcat5` | brainstorming | 哈希已冻结 |
| `xcat6` | information_extraction | 哈希已冻结 |
| `xcat7` | summarization | 哈希已冻结。接受数归档名 `sd-office-8b-traj-xcat7` / `20261104-045236`，与墙钟产物分开 |

full 胜出子集的离线描述在 `structure/office_traj_failset.py`，选句和 SHA256 仍调用 `office_traj.py` 里已经冻结的函数。这个文件没有 `main`，也不增加 legacy 条目。四个旧脚本的拆分仍未做。

# 三、structure 与 flow 各装什么
**structure（组成一次 Run）**

| **子层** | **本项目放什么** | **例子** |
| --- | --- | --- |
| `api` | 契约与协议，不含实现 | `MaskPolicy`、`HiddenMaskPolicy`、`ObservationSink`、`ExtraBranchPolicy` |
| `control` | Run 身份、配置、生命周期 | 既有阶段骨架 |
| `data` | 数据源、prompt 模板、渲染、划分 | SpecBench、Dolly、HEADS/TAILS 模板 |
| `model` | 模型加载与会话 | `HFCausalLM`；模型路径由配置给出 |
| `algorithm` | 一切计算，含解码算法 | `decode/`（循环、树、树编辑、树注意力）、`mask/`（ESP、EMA、Kalman、hybrid gain、bonus-aligned、output-hidden、校正）、`mask/fit.py`（K 拟合的纯数学）、`metric.py`（τ、符号检验、pooled τ 的纯公式） |
| `system` | 设备、运行时、环境、缓存与临时目录 | 现在写死的盘符路径与 `os.environ` 改动搬到这里，且只能由配置驱动 |
| `artifact` | 产物的写出、索引、读取、溯源 | 观测 JSONL、结果行读取、哈希/平台/commit 记录 |
| `make` | Recipe、注册、构建 | `build_mask_provider` 的注册表 |

**flow（把 Study 跑起来）**

| **阶段** | **本项目放什么** |
| --- | --- |
| `prepare` | 离线拟合并冻结 K（输入是 fit 集，输出是带模型/数据哈希的 K 文件）；准备 shared 资源 |
| `execute` | 按臂运行解码 |
| `collect` | 收集每块观测、接受计数、调用计数；pair 对齐 |
| `summarize` | 单 Run / 单臂汇总（接受长度、τ） |
| `write` | 写出证据与索引 |
| `process` | Study 级处理：配对检验、pooled τ、跨模型/跨数据比较、画图、审计 |
| `cli` | `decode` / `audit` / `compare` 的命令实现；薄入口转调这里 |

`api` 不导入其他 structure 子层。`structure` 不导入 `flow`。`flow` 可以导入 `structure`。`spec_mtp` 不导入 `rpipe` 或 `studies`。

# 四、历史遗留、门禁与分步迁移
`studies/_shared/structure/` 里的 `diagnose_*` / `analyze_*` / `audit_*` / `capture_*` / `reconcile_*` 一次性 CLI 已折进 `window_diagnostics`、`extension_diagnostics`、`q178_diagnostics`、`q241_diagnostics`、`sampling_diagnostics`、`geometry_4b`、`esp_diagnostics`。旧文件名只保留导入垫片，不再带 `main` / `argparse`，`structure_scripts` 清单随之缩小。各 Study 顶层的 `analyze_*` / `plot_*` 和其他顶层 `.py` 仍视为 legacy 冻结：不再新增，不再修改逻辑。修 bug 时一并迁到 `flow/process` 或宿主 `structure/algorithm/metric`。字节保护的 ESP upstream 树不进入这些清单，门禁也不扫描它的内容。

CPU 门禁在 `tests/spec_mtp/test_structure_flow_gates.py`。清单在 `tests/spec_mtp/legacy_allowlists/`。每份清单与当时扫到的违规逐项相同：代码里多一条违规，或清单里留着已经不再违规的条目，测试都失败。清单因此只能缩小。`origin/dev` 上已经有同一份清单之后，再往清单里加条目也会失败。2026-11-03 在本分支上冻结的条数：

| **门禁** | **清单** | **冻结条数** |
| --- | --- | --- |
| 文件落点：`src/spec_mtp` 顶层只有 `__init__.py` 与 `__main__.py`；`studies/<name>` 顶层只有 `recipe.py` 与 `__init__.py`；其余 `.py` 必须在 `structure/` 或 `flow/` | `file_inventory.json` | 81 |
| 导入方向：structure 不导入 flow；api 不导入其他 structure 子层；`spec_mtp` 不导入 `rpipe` / `studies` | `import_direction.json` | 4 |
| `structure/` 内不得有 `if __name__ == '__main__'`，也不得有 `argparse` | `structure_scripts.json` | 77 |
| `structure/` 与 `flow/` 的代码不得出现盘符字面量 | `drive_letters.json` | 10 |
| 宿主与 Study 的模块在 import 时不得写 `os.environ` | `import_environ.json` | 4 |

这 4 条导入违规都是现有 `structure/api/*_api.py` 再导出 `algorithm`、`data`、`model`、`system`。`structure` 导入 `flow` 的静态 import，以及 `spec_mtp` 导入 `rpipe` / `studies`，当前条数是 0。77 条脚本记录来自 40 个文件：40 个带 `main` 守卫，其中 37 个同时使用 `argparse`。`trajectory_observation` 的 `fresh_replace.py`、`office_qwen8b.py`、`office_tree.py`、`office_traj.py` 把原来的 `flow/execute` 命令折进已经拥有协议的 structure 模块。门禁对这四个路径闭合豁免，清单仍是 77 条，不能再加别的 structure 脚本。

测试镜像的目标是 `tests/spec_mtp/structure|flow` 与 `tests/studies/<name>/structure|flow`，`tests/manifest.json` 的 `test_id` 随路径一起改。步骤 8 之前不改现有测试路径。新增 provider 或算法要在 `structure/algorithm/config.py` 注册配置键；该文件还没有建立，本步不建。

搬迁分步进行。每步只搬位置，不改数学、不改默认参数。公开名在过渡期用 shim 保留。已有 Run 的源码路径以产生该结果的提交为准，不回写历史结果。ESP upstream 树不搬、不改字节。

| **步骤** | **内容** | **状态** |
| --- | --- | --- |
| 0 | 本文、架构设计、AGENTS、README，以及上表 CPU 门禁 | 已完成 |
| 1 | `contracts.py` → `structure/api/decode_api.py`；旧路径保留 shim | 已完成 |
| 2 | `tree.py` → `structure/algorithm/decode/` 下的树、树编辑、树注意力 | 已完成 |
| 3 | `masks.py` 拆到 `structure/algorithm/mask/`；注册表与配置 schema 分开 | 未做 |
| 4 | `decode.py` → `structure/algorithm/decode/loop.py` | 未做 |
| 5 | `data.py`、`model.py`、`audit.py`、`analysis.py` 进入对应子层，并消除与 `structure.data` / `structure.model` 的重名 | 未做 |
| 6 | `trajectory_observation` 补齐 Study 声明，并拆开四个旧测量脚本。这四个文件的命令已经折进对应 structure 模块，不代替这一步 | 未做 |
| 7 | `cli.py` → `flow/cli.py`；`python -m spec_mtp` 保留薄入口 | 未做 |
| 8 | 删除 shim，测试镜像改路径，`.importlinter` 按两柱重写 | 未做 |

# 五、宿主模块的目标位置
下表是目标路径。解码合同和树已经迁入。其余当前文件仍在“当前路径”一列。行为边界仍是现在这些模块已经承担的合同。

| **目标文件** | **当前路径** | **责任与边界** |
| --- | --- | --- |
| `src/spec_mtp/__init__.py` | 同左，保留 | 包标识及必要版本；导入不注册、不加载模型、不发起执行 |
| `src/spec_mtp/__main__.py` | 同左，保留 | 薄入口，转调 `flow`，不在 import 时执行 |
| `structure/api/decode_api.py` | 已迁入；`src/spec_mtp/contracts.py` 再导出原名字 | DecodeRequest/Result、AuditResult、Prompt、Observation、ModelSession、MaskPolicy、Sink，以及显式的 `ExtraBranchPolicy`。无其它项目模块 |
| `structure/data/` | `src/spec_mtp/data.py` | 读取本地输入、qid/顺序/大小校验、显式 tokenizer 渲染和编码。不创建模型，不决定 Study 因素 |
| `structure/model/hf_causal.py` | `src/spec_mtp/model.py` | 打开明确本地模型或借用已构造模型、能力与 tied-storage、forward/cache、close。不调用 decode |
| `structure/algorithm/decode/loop.py` | `src/spec_mtp/decode.py` | prefill、树块循环、验证/bonus、预算/EOS、mask 提交与 KV 保留。只调用注入的模型、策略和 sink |
| `structure/algorithm/mask/` | `src/spec_mtp/masks.py` | 各 mask provider 与 K 拟合的纯数学。块内预测不得污染 committed 状态。provider 的注册在 `make`，配置键在 `structure/algorithm/config.py` |
| `structure/algorithm/decode/tree.py`、`tree_edit.py`、`tree_attention.py` | 已迁入；`src/spec_mtp/tree.py` 再导出原名字 | 树构造、预算内的节点编辑、attention/position。不持有模型，不执行 forward |
| `structure/artifact/observation.py`；无模型审计在 `flow/process` | `src/spec_mtp/audit.py` | 版本化 JSONL、来源与计数核验。不 import 模型 |
| `structure/algorithm/metric.py`；跨 Run 汇总在 `flow/summarize` 或 `flow/process` | `src/spec_mtp/analysis.py` | 固定口径的 τ、配对和长度诊断。纯计算，不执行模型 |
| `flow/cli.py` | `src/spec_mtp/cli.py` | `decode` / `audit` / `compare` 的参数、装配与结果呈现。help 不加载模型 |

KV 的物理存储留在模型会话，保留哪条路径由解码循环决定，索引由树模块计算。三处共用同一长度事实。

共享合同只收多个调用方真实使用的数据与最小行为，不建通用 DTO 转换层、`utils.py`、Manager、插件容器或全局可变注册表。模型加载只能发生在明确打开会话时，普通 package/help/import 不取 GPU、文件锁或网络。

共享合同只收多个调用方真实使用的数据与最小行为，不建通用 DTO 转换层、`utils.py`、Manager、插件容器或全局可变注册表。模型加载只能发生在明确打开会话时，普通 package/help/import 不取 GPU、文件锁或网络。

# 六、K Study 的具体文件与依赖
| **目标文件** | **职责** | **依赖与限制** |
| --- | --- | --- |
| `studies/k_affine_residual/structure/recipe.py` | 从冻结因素组装 postcommit 三臂、文档划分、tiny 随机模型与载荷，向 RPipe 注册实际入口 | 只经 RPipe 公开合同、宿主 `data/model/audit/analysis/contracts` 及本 Study correction/evaluation，提取同一因果位置 latent，不修改/调用真实投机解码器 |
| `studies/k_affine_residual/structure/correction.py` | float64 中心化 ridge λ=.001 拟合 A/b、独立 b0，冻结参数核对及三臂应用 | `spec_mtp.contracts` 和必要既有数值库，不 import RPipe/模型/CLI；截距不正则，fit/apply 分开，b0 不取完整 b |
| `studies/k_affine_residual/structure/evaluation.py` | P/y/m 及同位置 z(m)/z(y) 对齐、文档级三臂 MSE、先冻结 K/b0 再提取 holdout/cross 的合同 | 本 Study correction、宿主 `contracts/audit/analysis`，不加载模型，不自判 Run 成功，拒绝未来 token/下一词标签与 fit/test 泄漏 |
| `studies/k_affine_residual/flow/collect/run.py` | 将实际原始结果转换为 evaluation 所需输入，返回本 Study 指标 | RPipe 公共阶段合同与本 Study evaluation，不重复计算/生成 raw，不改因素 |
| `studies/k_affine_residual/flow/summarize/run.py` | 把 evaluation 的完整性/科学质量结果映射到阶段结果 | 同上，缺失/失败明确拒绝，模型预测改善与研究假设支持分开 |
| `studies/k_affine_residual/flow/process/run.py` | 核对声明三臂、拟合/测试数据隔离、冻结 K、因素和 seed 的矩阵完整性 | 同上，成功子集不能作为完整矩阵，不执行补跑 |

所有列出的 `__init__.py` 仅命名空间或必要的 `run` 导出，不含业务逻辑。prepare、execute、write 优先复用默认阶段，只有实际框架合同无法表达时才重新评审增加适配，不能预造六阶段空壳。

tiny 随机模型可由 Study recipe 使用官方构造器按冻结配置建立，再交给宿主模型会话；该特有模型构造不反向进入科学包，也不加载预训练0.6B权重。配置、tied 实际存储与所有权均需核对。

输入资产与运行根由明确执行方案和 RPipe 接入处理。K 文件不是全局可变参数，绑定模型/表示空间、fit IDs/hash 和协议，先保存 full K 与独立 b0，评价前后核 SHA，不原地重拟合。当前起点是已提交位置的 postcommit replacement：m=mean(E[P]) 为连续 embedding，y 为已提交 token，同位置比较 soft(m)/E[y] 最终 normalized hidden；不是实际未来槽或真实命中率实验。完整数据/模型/指标定义见研究计划第五章，本轮没有执行或下载。

# 七、源码依赖
## （一）允许方向
目标方向：`flow` 可以导入 `structure`；`structure` 不导入 `flow`；`structure/api` 不导入其他 structure 子层；解码通过注入的模型会话进入具体接入，源码不导入 HF 实现的反向依赖。`spec_mtp` 不导入 `rpipe` 或 `studies`。Study 可以导入宿主公共模块。宿主没有 Study 路径，也不反向注册。

扁平模块在搬迁完成前仍然存在。这段时间里，`contracts/data/decode/masks/tree/audit/analysis` 继续不导入 `cli` 与具体 `model`。模型反向不导入解码。

依赖规则同时约束静态 import、类型引用和动态加载。静态工具不能证明全部运行关系。数据泄漏、owner 对齐、cache 或取消正确性由行为测试验证。
## （二）现在执行的检查与迁完后的 Import Linter
`.importlinter` 现在仍是扁平模块合同：整个 `spec_mtp` 禁止 `rpipe` 与 `studies`，核心扁平模块禁止 `cli` 与 `model`。两柱方向由第四节的 pytest 门禁执行，遗留的 api 再导出记在 `import_direction.json`。步骤 8 再把 `.importlinter` 改成下面的目标合同，并删掉已经不存在的扁平模块名。在那之前替换合同会把尚未搬走的 api 再导出判成新失败。

```ini
[importlinter]
root_package = spec_mtp
include_external_packages = True

[importlinter:contract:science_without_framework]
name = Scientific package has no Study or RPipe dependency
type = forbidden
source_modules =
    spec_mtp
forbidden_modules =
    rpipe
    studies

[importlinter:contract:structure_without_flow]
name = structure does not import flow
type = forbidden
source_modules =
    spec_mtp.structure
forbidden_modules =
    spec_mtp.flow

[importlinter:contract:api_without_siblings]
name = api does not import other structure sublayers
type = forbidden
source_modules =
    spec_mtp.structure.api
forbidden_modules =
    spec_mtp.structure.control
    spec_mtp.structure.data
    spec_mtp.structure.model
    spec_mtp.structure.algorithm
    spec_mtp.structure.system
    spec_mtp.structure.artifact
    spec_mtp.structure.make
```

这组目标合同只覆盖 `spec_mtp`。Study 侧的同类方向由同一 pytest 门禁扫描。不能因工具没有扫到动态加载就声称全部跨树约束通过。

# 八、具体测试镜像与单一信号
下表的测试文件是当前路径。源码迁到两柱之后，测试改到镜像的 `tests/spec_mtp/structure|flow` 与 `tests/studies/<name>/structure|flow`，`tests/manifest.json` 的 `test_id` 一起改。步骤 8 之前不改这些路径。

| **当前源文件** | **当前测试文件** | **主要合同与重点反例** |
| --- | --- | --- |
| `src/spec_mtp/__main__.py` | `tests/spec_mtp/test_main.py` | package 导入/help 零注册/载模/网络，入口转发和退出 |
| `src/spec_mtp/cli.py` | `tests/spec_mtp/test_cli.py` | decode 参数到假会话完整旅程、audit/compare 零执行、坏路径及中文空格 |
| `structure/api/decode_api.py`（`contracts.py` 再导出） | `tests/spec_mtp/test_contracts.py` | 请求/结果版本、有限数值、缺失语义、真实需要的能力与错误字段；旧路径与新路径是同一对象 |
| `src/spec_mtp/data.py` | `tests/spec_mtp/test_data.py` | qid/顺序/输入边界、template/EOS/context、坏文件和重复记录 |
| `src/spec_mtp/model.py` | `tests/spec_mtp/test_model.py` | 假 HF forward/cache、实际 tied-storage、能力拒绝与 close 再失败 |
| `src/spec_mtp/decode.py` | `tests/spec_mtp/test_decode.py` | 可枚举 token 链、prefill/bonus/EOS、取消派发、拒绝路径/KV 和 audit sink 故障 |
| `src/spec_mtp/masks.py` | `tests/spec_mtp/test_masks.py` | 初始化、冻结、β/γ、零范数/NaN、proxy 不改 committed，逐方法参数独立 |
| `structure/algorithm/decode/tree.py`、`tree_edit.py`、`tree_attention.py`（`tree.py` 再导出） | `tests/spec_mtp/test_tree.py` | static/dynamic 预算、父/深度/owner/position、错路径和 attention 可见性；旧路径与新路径是同一对象 |
| `src/spec_mtp/audit.py` | `tests/spec_mtp/test_audit.py` | token/argmax/owner/来源漂移、坏半行/截断/混 Run、原子写失败、AR 只诊断 |
| `src/spec_mtp/analysis.py` | `tests/spec_mtp/test_analysis.py` | pooled/均值与 prefill 口径、缺 prompt/长度差、胜平负/负例及不可比 |
| 稳定宿主依赖合同 | `tests/spec_mtp/test_dependencies.py` | 按单条禁止边参数化，隔离执行，核心不导入具体模型/框架或 Study |
| 两柱布局与 legacy 清单 | `tests/spec_mtp/test_structure_flow_gates.py` | 文件落点、导入方向、structure 内的 main/argparse、盘符字面量、import 时写环境变量；清单与现状逐项相同，只减不增 |
| K `structure/recipe.py` | `tests/studies/k_affine_residual/structure/test_recipe.py` | 三臂装配和冻结配置、一次公共载荷调用，拒绝未知臂/失配来源 |
| K `structure/correction.py` | `tests/studies/k_affine_residual/structure/test_correction.py` | 合成已知 A/b、零残差、三臂、错误维度/非有限、冻结后不拟合 |
| K `structure/evaluation.py` | `tests/studies/k_affine_residual/structure/test_evaluation.py` | m/y/owner 错位、未来标签泄漏、拟合/测试重叠、冻结 K 跨集身份和完整配对 |
| K `flow/collect/run.py` | `tests/studies/k_affine_residual/flow/collect/test_run.py` | 原始样本到指标的适配，不补算/补生成，缺数据保留 |
| K `flow/summarize/run.py` | `tests/studies/k_affine_residual/flow/summarize/test_run.py` | 数学/来源/冻结失败阻断，纯测量与假设结论不混同 |
| K `flow/process/run.py` | `tests/studies/k_affine_residual/flow/process/test_run.py` | 三臂/数据集/seed 缺失、重复、子集拒绝，不发起补跑 |
| `tests/run.py` | `tests/tools/test_run_tests.py` | 按清单先选后导入、逐参数/attempt 落盘、必需项缺失/超时/写失败非零 |

每行细分多个单信号用例，level/type/cost_class/result_type/priority 为标签与静态元数据。源码镜像不增加 `unit/integration/e2e` 一级目录。pytest 使用仓库根、`src` 和明确的 import 模式，镜像目录不加入产品模块搜索路径，避免同名测试目录遮蔽 `spec_mtp`。

`__init__.py` 纯命名空间不逐个测自身，第三方树不生成本仓逐文件逻辑测试，已有旧消费者在实际改造时按稳定合同迁移，不把旧字符串快照当科学规则。共享 fake/合成数据先放用例附近，至少多个真实调用方复用时才提取 `tests/_helpers`，不用大 fixture 目录预铺空壳。

# 九、验证入口与产物界限
`tests/run.py` 只适配现有 pytest 的选择、运行和结果回调。唯一 `tests/manifest.json` 保存静态元数据与计划，执行前列必需项、条件、资源、预算及排除理由，执行结果逐项写独立外部验证根。失败、收集缺失、取消与结果落盘失败不能把必需计划记为通过，详细字段与阻断条件见架构设计第七章。

| **位置** | **内容与规则** |
| --- | --- |
| `src/spec_mtp/` | 公共科学实现及完整保护的第三方来源子树，运行不回写 |
| `spec_mtp/flow`、`spec_mtp/structure` | 宿主与 Study 的目标两柱；流程由本仓实现，结构对齐 RPipe 模板。解码合同和树已经搬入，其余解码模块仍在扁平路径。不再内嵌副本 |
| `studies/` | 静态研究声明和有职责薄适配，不生成长期报告/图/证据快照 |
| `datasets/spec_bench/` | 当前原始静态数据，保留来源与字节，不当运行缓存删 |
| `tests/manifest.json` | 静态计划和用例元数据，可入 Git，不与同名运行 manifest 混用 |
| 显式外部运行/验证根 | 冻结输入、K、raw token/审计/trace、日志、逐项结果及可重建报告，不进源树 Git |
| `docs/` | 唯一当前入口；历史测量在运行记录，演进在开发记录 |

第三方 upstream 的八份来源成员（README、四份 code、三份 results）以及 `ORIGIN.json` 属来源完整性例外，仍保留其原位置和字节。

# 十、当前差距与实施检查
两柱搬迁完成了步骤 0、步骤 1 和步骤 2。解码合同在 `structure/api/decode_api.py`，调用方仍可从 `spec_mtp.contracts` 导入。树构造、深度 1 编辑和块注意力在 `structure/algorithm/decode/`，调用方仍可从 `spec_mtp.tree` 导入。第五节其余目标路径还没有对应的新文件。下面这些是另外的职责残留：

- 成功路径的 `prefill_time` / `decode_time` 仍留在解码统计里，因为已有读端用 0 表示“本次不计时”。时钟或同步失败时这两个键缺席，计数照常返回，汇总也不把缺席写成 0 时长。
- `studies/_shared/structure/evaluate.py` 仍负责预热、计时和 CSV。它调用宿主解码器，不再自己计算 τ。
- B2 已作废：没有内嵌 RPipe 副本。结构模板见开发计划第三节。
- 2026年10月31日会议的联合 mask 已有默认实现和合成测量。合成形式搜索已停止，见研究计划第七章。真实模型比较未批准。

旧 `spec-mtp make/launch/process` 会明确拒绝并指向 `python -m studies._shared.flow.cli`。删除历史 Study、第三方树或原始数据不在本次范围。

## 已经落位的部分
科学 CLI 不再代理研究调度。Study 共享接入在 `studies/_shared/`。K 的三个科学函数和 collect/summarize/process 仍在该 Study 的 `structure/` 与 `flow/`。`structure/algorithm/reference` 只保留不可改的 ESP 来源树。只读来源 manifest 覆盖科学包与被复用的 `_shared` 接入。步骤 3–8 还没有开始。

`.importlinter` 仍检查扁平模块。两柱方向由 `tests/spec_mtp/test_structure_flow_gates.py` 执行；工具未跑 Import Linter 时，不把那份目标合同写成已经生效。`tests/manifest.json` 声明函数级元数据并展开 pytest 参数 ID，`tests/run.py` 将每个参数/阶段/attempt 的证据写到外部新目录。历史读端测试通过 `tests/conftest.py` 的共享 `historical_file` fixture，从固定 `b8b20b5ada7fd860894114cbf224ce749af430cf` 按需读取不可变 Git blob 到测试临时目录，逐对象及写入字节校验；缺对象明确失败。原15个参数项恢复为必需覆盖，现清单没有历史排除；分析器与校验门仍执行当前源码，旧源码仅作来源哈希输入，不导入执行，不重跑历史模型。其余收集项无遗漏才执行。
