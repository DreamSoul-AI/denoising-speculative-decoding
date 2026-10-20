# Study 使用指南

本仓库按固定 [RPipe main v0.3.0 的 Study 指南](https://github.com/diaoenmao/RPipe/blob/804b3823ab6d2af46cbda1a65688ca8b76c60b22/studies/README.md)组织研究，实际执行使用 `src/rpipe/`，来源见 [ORIGIN](../src/rpipe/ORIGIN)。本指南保留上游的声明、两柱扩展、阶段、来源和报告合同，并适配 training-free 解码。入口为 `python -m spec_mtp`；Factory、调度、Flow、Tracker、artifact 和 readout 均由同一份 RPipe 提供。

## 1. Study / Experiment / Run

| **比较项** | **Study** | **Experiment** | **Run** |
| --- | --- | --- | --- |
| 职责 | 一轮研究的声明、共享资产和终验 | 一个研究因素组合，不含 seed | 一个因素组合 × 一个 seed 的实测 |
| 声明 | `study.yaml` | `axes` 的笛卡尔积；无轴时为一个组合 | `seeds`，配置中必须含 seed |
| 磁盘 | `studies/<name>/` | 不设独立目录 | `runs/<id>/` |
| 分析 | 当前 index 的完整性、来源和质量 | 跨 seed 的 mean / std / min / max / n | config、原始观测、指标、result、日志 |

配置展开为 `experiment_config.yaml` 基底 → `study.yaml.fixed` → 轴值和 seed，后者覆盖前者。基底不是一个 Experiment 实例。修改配方前核对最终 Run config，不能只看其中一份 YAML。seed 不表示并发；相同因素和 seed 的独立复跑要更换 `algorithm.config.version`。

## 2. 声明、扩展与目录

```text
studies/<name>/
  README.md                     问题、入口和执行边界
  study.yaml                    名称、recipe、fixed、axes、seeds、flow
  experiment_config.yaml        data/model/algorithm/system 基底
  recipe.py                     注册已有专项 source；不解码或下载
  flow/<phase>/                 有专项职责才添加
    __init__.py                 仅导出
    run.py                      阶段实现；helper 可单独放置
  structure/                    该 Study 专用建构；没有则不建空壳
  docs/
    STUDY_REPORT.md             人写的结果与结论
    NUMBERS.md                  report 的数字表；不替代结论
    figures/                    真实结果图和图注
```

从 [_template](_template/README.md) 复制，改目录和 `study` 名称，在项目计划中写独立章节。现有 `smoke`、`input_mask_baseline` 与模板；只有项目计划中明确授权并冻结的部分进入执行，其余研究草案不是可执行 Study，历史接受数见 [history-extrapolation.md](../docs/history-extrapolation.md)。一次性环境排错放 `.tmp/diagnostics/`，必要结论并入关联报告。

SpecMTP 声明使用 `recipe: recipe.py` 和 `source: spec_mtp`。Study recipe 的 import 调用 `register_sources()`，Run prepare 的 `register(ctx)` 检查构造合同；注册不读取 prompt、下载权重或执行大模型。库默认 source 保持原状。复用的 ESP/EMA/AR/alignment、mask、tree、KV cache 和数据/模型适配在 `src/spec_mtp/`，只有本轮独有实现才放 Study `structure/`。

`flow.study_phases: true` 启用阶段包。当前 `flow.prepare_shared: false` 是本项目已验证的 torchvision-only make 边界适配，并不取消 shared：Run prepare 将 SpecBench 写入 `shared/data/specbench/`，ModelScope/HF 权重写入 `shared/model/`。数据源在 `datasets/spec_bench/`。保留现有 ModelScope 配方和 [origin 约定](../docs/layout.md#不要写-origin-foreign)，不机械复制其他 Study 的准备策略。

## 3. 阶段职责与成功条件

| **阶段** | **本项目职责** | **当前扩展位置** |
| --- | --- | --- |
| prepare | 库检查来源、recipe 注册，构造 Data/Model/Algorithm，保存输入、权重与实际环境 | 默认阶段 + recipe / 专项 builder；无 Study 空壳 |
| execute | 保存真实输出 token、AR 对照、接受/回退和 cache 观测 | 默认阶段调用专项 Algorithm |
| collect | 从原始观测计算 CSV 与 Tracker 指标 | `flow/collect/run.py` |
| summarize | 核验全部提交token=验证forward argmax、指标及来源；AR exact仅诊断 | `flow/summarize/run.py` |
| write | 在库定稿 result 前再核验来源、配置和输入 | `flow/write/run.py` |
| process | Run scope 用默认处理；Study scope 只检查当前 index 并终验因素/seed/来源/证据 | `flow/process/run.py` |

write 先执行 Study hook 再由库定稿，其他阶段先库后 Study；prepare 需要构造前职责时才使用 `before(ctx)`。默认阶段不能复制成第二套 runner。Study scope 没有单 Run layout，不得在每个 Run 重跑矩阵。子集不完整、混合来源或专项门失败时，通用 `complete` 与专项质量不能报告完整通过。

指标口径见 [architecture.md](../docs/architecture.md)：当前T0硬门是全部提交token等于对应验证前向logits的argmax，独立AR一致率及逐prompt首分叉仅诊断，不阻塞summarize/write/process；历史冻结协议与Run不重签。accepted 为所有深度接受的 draft 数，沿用预算/EOS 截断前统计；`accept_d1` 等是深度诊断。new_tokens 包含 prefill 首 token，committed 是 decode pass 提交量，τ 为总 committed / 总 decode_calls。速度单列条件和单位，不代替正确性与研究结论；γ 不叫 slope。training-free 不意味着去掉 mask 更新的 autograd；不改变权重、mask/EMA、树/cache、位置/attention、接受、EOS 或预算语义来获得通过。

## 4. 来源冻结与复跑

- RPipe `provenance.json` 覆盖库 Python、两份声明、recipe、两柱 Python、`provenance.include`；其 plan 哈希是 index 与 Run config，不自动包含人写的 `PLAN.md`。额外固定参考输入按需显式声明，不能误称文档已自动冻结。
- `spec_mtp_sources.json` 补齐外部专项包，覆盖 Python 文件的新增、删除和修改，并锁住声明中已存在的本地输入。实际模型/tokenizer、prompt、配置、dtype、TF32、backend 和环境在构造及执行证据中记录，成功门与 Study 终验复查。
- 显式 `make` / `launch --remake` 建立来源锁；普通 run/launch 保留已有锁。首次无锁 run 可以建锁，`freeze: true` 时则必须先 make。正式研究建议启用 freeze；已有专项锁即使未启用 freeze 也拒绝漂移。
- 源码、输入或配方改变，先换 version 再 make/remake，不为旧成功产物重新签收。`--include-done` 不是新验收；直接 native RPipe/FlowRunner 存在同 ID 失败回写边界，见 [C-004](../docs/bugs.md)，正常操作使用薄 CLI 和新 Run ID。
- 日志、OOM、失败和重复反例保留。解码 KV/mask/RNG 的 checkpoint 恢复未验收，不把上游训练 `resume: latest` 写成已支持的解码恢复。

## 5. 高效率排班与时长

开跑前把排班写入唯一项目计划对应章节。先以 `make --round 1` 串行校准实际权重、KV、树、mask 与进程峰值；没有测量时不套用 RPipe 视觉训练的 auto 显存预算。测量后可用 `system.config.vram_bytes` 记录保守预算，小步增加独立 Run 并发。不同模型或明显不同耗时分组，每组等进程退出、显存释放后再进入下一组。

每个 Run 独占产物、KV cache、mask、RNG 和 logger；一次只启动一条 launch 调度链。速度对照串行运行相同任务量，记录预热、CUDA 同步、backend、生成长度和整体墙钟。seed 数量不决定并发。make 的估计是组内最慢估计求和，不能当作实测速度；计划填 est，报告填 actual，缺失就明确记未记录。

## 6. 实际命令与产物

在仓库根目录使用现有环境，Windows 约定见 [windows.md](../docs/windows.md)。不覆盖 GPU PyTorch。PowerShell 示例：

```powershell
$env:PYTHONUTF8 = '1'
python -m spec_mtp make studies/<name> --num-gpus 1 --init-gpu 0 --round 1
python -m spec_mtp launch studies/<name> --num-gpus 1 --init-gpu 0
python -m spec_mtp process studies/<name>
python -m spec_mtp status studies/<name>
python -m spec_mtp logs studies/<name>
python -m spec_mtp report studies/<name>
```

make 展开配置/index 和脚本；launch 使用同一调度器并在结束后执行 Study process。CPU Run 由 `system.config.device: cpu` 进入 CPU 队列，不绑定 GPU。`run --skip-launch` 可查看展开结果，但不是终验；不复制上游 `data` 的 torchvision 统计命令作为 SpecBench 准备入口。生成脚本和旧 evaluate/test 兼容入口不能取代正式来源与阶段门。

`index.json` 只列本轮；`runs/<id>/config.yaml` 是完整配置，`result.json` 是定稿状态；各 Run 原始观测、CSV、Tracker、来源证据和 `assets/logs/run.log` 用于追溯。根 `process.json` 聚合当前 index 的 Experiment，report 生成 `docs/NUMBERS.md`，不会写人类结论。`runs/`、`shared/`、`scripts/`、index/process/provenance/专项锁和 `.tmp/` 不提交；报告、计划、图及精选来源证据随 Git 提供。

## 7. 报告与验收

报告按 Experiment 写结论，给出 mean/std/min/max/n、指标单位、阈值与条件；单 seed 的 std=0 不能证明稳定性。Run 表保留 ID、seed、指标、est/actual 和证据引用，区分执行完成、质量通过、失败、跳过和未执行。原始日志本机可打开，但 clone 不附带；未入库路径必须明确标记为本机文本，不能提供失效的正式超链接。

RPipe 要求报告附图：解码使用真实接受数或轨迹图及来源图注，不编造训练 learning curve。`smoke` 的图由历史正式 JSON 绘制，图不能成为性能或论文结论。模板未执行时不填数字或引用不存在的图片。相同设备/输入/精度/随机性/预算的旧新 token 和轨迹对照先于完整矩阵；失败边界、安装和实际 CLI 验证范围分别报告。

| **Study** | **职责与状态** | **正式依据** |
| --- | --- | --- |
| [smoke](smoke/README.md) | 随机 tiny 三 variant 的可复用门；历史 CPU/CUDA 串行验收已完成 | [PLAN](<../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies-smoke-docs-plan>)、[报告](smoke/docs/STUDY_REPORT.md)、[NUMBERS](smoke/docs/NUMBERS.md) |
| [_template](_template/README.md) | 新研究起点，不能直接运行，不代表真实模型已通过 | [计划模板](<../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies--template-docs-plan>)、[报告模板](_template/docs/STUDY_REPORT.md) |
| [output_space_prerequisite](output_space_prerequisite/README.md) | tied输出hidden零/一阶两槽链式四格已通过；接受数17/10/10，单prompt负结果 | [PLAN](<../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies-output-space-prerequisite-docs-plan>) |
| [output_space_small_matrix](output_space_small_matrix/README.md) | q172–q174小矩阵/匹配前缀诊断通过；混合小样本观察 | [PLAN](<../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies-output-space-small-matrix-docs-plan>)、[报告](output_space_small_matrix/docs/STUDY_REPORT.md) |

新增 [input_mask_baseline](input_mask_baseline/README.md) 使用真实 Qwen3-0.6B / SpecBench q171，四个固定 case 是宏观方案 §5.11 / E1–E2 的前置门；计划与结果分别见其 docs。通用四阶段从 src/spec_mtp/flow 同名包导入，研究包内的默认 prepare/execute/make 不填空实现。

[output_space_cross_task](output_space_cross_task/README.md)是q241–244 summarization证伪草案，只有输入分词预检，未make/launch，不存在新测量结果。q178后续provider算子对照见[extension报告](output_space_extension/docs/STUDY_REPORT.md)。

[output_space_cross_task](output_space_cross_task/README.md)已按独立EXECUTION_PLAN完成q241–244真实四格和20固定移植，全部AR一致但正负混合；第一阳性q241另冻canonical跟进出现11/12→14/14反例（路径仍不同）。此前“草案未执行”是历史状态，以[当前报告](output_space_cross_task/docs/STUDY_REPORT.md)为准，未改旧PLAN或重签Run。

[4B同链复验](output_space_4b_recheck_v2/README.md)及[固定新4条](output_space_4b_holdout/README.md)真实四Run/Study门通过；[固定输入EMA](output_space_4b_input_control/README.md)两新Run复用其输出/AR。当前只比较接受/主calls/AR，不测速。output_space_4b_recheck是失败v1原位归档，勿继续其pending或重签锁。


2026-10-15 阶段顺序：历史Qwen3-4B **FP32**结果归档并按PR流程合dev → ESP Table1复现（Qwen3-8B/BF16/T0、BC10/30/60、动态树与剪枝、mean mask λ=.1）→ Qwen3-4B/BF16五方法加AR配对。8B仅用于该Table1复现，其他研究暂停；各阶段实际结果见下，不用历史FP32替代BF16证据。


2026-10-15 阶段结果：[8B全480 argmax审计](esp_table1_bf16_full_argmax/README.md)四格及原生process完成，318,830个输出token审计通过，AR仅诊断。BC30 τ1.563378、BC60单mask1.647301未复现目标1.63/1.74，三聚合及双mask对照全部保留；PR #10经CI合dev。[4B BF16五方法配对](qwen3_4b_bf16_five_methods/README.md)11/11及原生process通过，809,118个Spec提交token自检全过；全部20组逐prompt对照、AR差异、数字表及真实图见[报告](qwen3_4b_bf16_five_methods/docs/STUDY_REPORT.md)，[集成与CI](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/12)。冻结PLAN不回写。未新增FP32模型实验、算子机理或测速；未强推、合main、删除分支或数据，8B未用于其他研究。

## 项目计划归并（2026-10-16）

唯一入口为[中文](<../docs/研究计划 Denoising Speculative Decoding.md>)/[English](<../docs/Research Plan Denoising Speculative Decoding.md>)。全部60份旧子计划已归入各自章节，来源清单保存旧Git commit/blob/工作树换行SHA。旧Run不重签、不remake；复放旧合同使用固定历史Git版本。当前Study声明的provenance.project_plan记录项目路径、章节及历史来源；这份元数据随study.yaml冻结，RPipe不自动读取仓库外路径或实时中央文档。新执行需显式验证不可变Git计划blob，不能误称额外字段本身会读取/冻结中央文档。

当前单GPU串行，准入=已测进程树峰值主机RAM+2GiB余量及实际GPU空闲/工作集可容纳；不逐个核查CPU脚本，不杀外部作业。旧并发/显存阈值章节仅历史通用说明。BF16/T0、不测速、固定480 prompt，当前独立8B复验后接已预注册4B范数扫描。


## K affine residual CPU stage

[k_affine_residual](k_affine_residual/README.md) is a separately authorised postcommit latent calibration Study. Its fixed random model and fit/holdout/cross-data comparisons do not reproduce future-slot hit rate or establish data independence. Plan and registration precede native execution; no old queue resumes.
