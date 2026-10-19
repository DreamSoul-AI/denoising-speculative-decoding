# layout

`src/rpipe/` 保留上游 RPipe `main` 固定来源，提交 `7a15c46b75c5eb951cccd60ed7b9fc3463db90f5`（`src/rpipe/ORIGIN`，v0.3.1）；Windows 例外和 [R1 本地补丁](rpipe-migration.md#r1-local-backport-20261020) 逐文件登记，原 blob 存在 `docs/rpipe-source/blobs/`。`src/spec_mtp/` 只放 SpecMTP，经 Study recipe 注册。不要把 SpecMTP 写进 `rpipe` 的默认 source。

```text
src/
  rpipe/                   vendored RPipe；Windows 上 __init__.py 先 import torch
  spec_mtp/
    recipe.py              compatibility exports for structure/api/recipe.py
    flow/cli.py            薄入口；根 cli.py 保留 console script 兼容导出
    flow/{collect,summarize,write,process}/   专项阶段；checks.py 兼容导出
    structure/artifact/evidence.py   实际权重/输入/runtime 证据
    structure/             algorithm / data / model / system / control / artifact
datasets/spec_bench/       SpecBench 源文件；Run prepare 拷进 shared/data
studies/                   见 studies/README.md
configs/hf/                论文格子的旧 YAML；不是入口
tools/                     ci_local.py, _process_tree.py (private process ownership), check_rpipe_pin.py,
                           windows_selfcheck.py, profile_cpu.py,
                           decode_fingerprint.py, package_selfcheck.py
tests/spec_mtp/            跟着 src/spec_mtp
docs/                      architecture, layout, testing, windows, brainstorm,
                           history-extrapolation；research/；papers/
```

产物：`studies/<name>/runs/<id>/`。`runs/`、`shared/`、`scripts/`、`index.json`、`process.json`、`provenance.json`、`spec_mtp_sources.json` 不进 git。

## shared/

RPipe 的 make 共享下载只认识 torchvision 数据集。SpecMTP Study 写 `flow.prepare_shared: false`，但 **Run prepare 仍然把东西写进 `shared/`**：

- SpecBench jsonl → `studies/<name>/shared/data/specbench/`
- ModelScope / HF 权重 → `studies/<name>/shared/model/`

同一 Study 里不要让每个 Run 再下一遍。源文件在 `datasets/spec_bench/`。

Study `recipe.py` 在 import 时调用 `register_sources()`。

## 不要写 `origin: foreign`

这是本仓库相对 RPipe 的规则。缺省就是 foreign，并且不改环境变量。一旦写出 `origin`，prepare 会设置 `HF_ENDPOINT`。权重走 `model.config.hub: modelscope`。只有需要 RPipe 去改 HF 镜像时才写 `origin: domestic`。

## Study 合同（2026-10-13）

目录入口统一 README.md，项目唯一中英文研究计划保存所有Study计划章节，docs/STUDY_REPORT.md保存人类结论；NUMBERS 保留 report 数字。两份 YAML 仍按基底 → fixed → axes/seed 合并，不搬动已有研究配置。smoke 的原交付 version 和正式 JSON 保留，新实测采用隔离目录与新 version。阶段仅保留 collect/summarize/write/process 的专项扩展，默认 prepare/execute 通过 recipe、builder 与 Algorithm 执行。training-free 报告用真实解码结果图适配上游学习曲线要求，不增加训练或恢复承诺。完整合同见 [Study 指南](../studies/README.md)。

当前结构迁移详见 [rpipe-migration.md](rpipe-migration.md)：data 的业务从 __init__ 移至 datasets，跨层输入与注册由 structure/api 组装，纯证据核验在 structure/artifact/validation。make、prepare 与 execute 无专项编排时直接复用库；不补空壳。

`tools/diagnose_output_windows.py` follows the existing direct-decoder profiling utilities. It observes fixed q173 windows, binds tool/PLAN/native raw/source hashes and refuses mismatching replay. Full selected tensors stay in a new .tmp directory; scalar diagnosis belongs to the Study. Scientific src, original Run and source locks are retained.

`tools/diagnose_mask_scale.py` uses the saved window tensors to cross the two observed directions/norms at one predeclared position, with four isolated causal forwards. It reuses the existing model wrapper and scalar helpers; it does not alter decoder/provider/config or choose a new default.

`tools/diagnose_proxy_transfer.py` scopes one predeclared provider-return vector replacement to a diagnostic replay. It infers seed ownership from actual root progress and verifies the returned trace, keeping post-intervention paths rather than forcing baseline acceptance. Its PLAN/results and offline `analyze_transfers.py` live in output_space_small_matrix; no scientific source/default/Run changes.

`tools/diagnose_cross_prompt.py` runs the predeclared q171/q172/q174 selection with the unchanged observation/transfer helpers. It binds both original Studies and compares same-position committed anchor/velocity rather than assuming they are equal. Selection and failure/termination controls live in the [central cross-prompt plan](<研究计划 Denoising Speculative Decoding.md#legacy-studies-output-space-small-matrix-docs-cross-prompt-plan>); full local tensors remain ignored.

`tools/diagnose_extension.py` consumes output_space_extension's predeclared native matrix and frozen semantic-only window_selection_v1.json. It reuses existing observe/transfer/compare helpers; samples follow actual paths and noncommon roots remain explicit. First/second-slot offline inspection lives in that Study's analyze_results.py, with second-slot scalar expansion marked post hoc. Scientific src/defaults and earlier measurement tools remain unchanged.

`tools/diagnose_q178_numerics.py` uses process-only deterministic FP32 flags and an explicitly scoped provider-state synchronization to canonical AR-prefix hiddens. Actual root is read from the unchanged mutable KV cache and checked at forward; proxies/decoder/cache code remain intact. Its controls/teacher/repeat budget is frozen in the [central numerical plan](<研究计划 Denoising Speculative Decoding.md#legacy-studies-output-space-extension-docs-numerical-plan>).

`tools/diagnose_study_windows.py` is the manifest-bound four-prompt coordinator for output_space_followup. Existing choose_window/observe/transfer/compare/slots implementations are imported unchanged and individually hash-bound. It records both slot rankings and validity with actual/noncommon paths. New-study native phases stay thin imports; raw tensors/weights remain local.

`tools/diagnose_q178_operators.py`提供固定q178后续provider槽精度／norm归约后端对照。测量v1源码原文在`studies/output_space_extension/docs/operator_tool_v1.py.txt`，当前CLI只补运行示例；分析核对除模块docstring外AST一致，原报告tool哈希不重签。

`tools/diagnose_q178_norm_kernel.py`比较固定入口／后续CUDA norm实现图，区分返回值等价与有效干预；`tools/diagnose_q178_qproj.py`执行固定layer0行的FP32 packing／FP64-rounded对照及独立cache块重放。两者共享Study局部`kernel_common.py`来源加载门，不改科学模型或默认配方。

`tools/diagnose_q178_phi_rms.py`固定入口Phi完整高精度映射与layer0 post-attention RMSNorm row4对照；`tools/diagnose_q241_canonical.py`按首个cross-task阳性准则执行q241原值／移植canonical状态跟进。复用旧canonical/transfer helpers，raw mmap只读、真KV root门不变。

工具诊断：`tools/diagnose_q178_rounding_factors.py`只做已存输入的算术分解；`tools/diagnose_q241_state_factors.py`只在固定真实KV根选择性回填anchor/velocity，沿用原decoder与transfer。

`tools/diagnose_q241_local_factors.py`固定source719/current720 cache/tree并交换state/proxy；`tools/diagnose_state_extension.py`执行预声明q242/q243两seed状态因素矩阵，不改正式解码器。

`tools/diagnose_sampling_paths.py`做真实非T0 AR耦合路径及旧source719条件抽样；`tools/diagnose_q241_owner1.py`固定实际source717／owner1映射，只核对应rows5/6的seed。

`tools/diagnose_sampling_expansion.py`补固定四prompt三seed全采样路径；`tools/diagnose_rare_events.py`按旧owner1上下文做native抽样及稳定Bernoulli检测能力计算，零模型forward。

`tools/diagnose_sampling_continuation.py`固定六新prompt/seed诊断；`tools/diagnose_rare_stress.py`分开原温度零观测与T=2条件压力测试，不改变默认采样器/配置。Study局部analyze_continuation与analyze_sampled_depth复用旧计数/观察器，只保存统计和语义轨迹，完整draw/tensors仍在.tmp。

4B Study局部untimed_4b.py仅关闭decoder模块计时并以相同权重directCUDA materialize；注册范围是该原生Run进程，科学src/default不变。四phase导出沿用from .run import run；v1失败源保留，v2/native holdout/control为有效入口。两个analyze_results与write_report复核原生计数/AR/来源，selectedEMA复用已测输出，权重/全Run/shared仍不入Git。

`tools/launch_once.py`调用原生SpecMTP/RPipe CLI，仅关闭可选失败重试，避免AR诊断失败格重复执行：`python tools/launch_once.py launch studies/<name> --round 1 --console shared`。退出恢复调度器，不修改vendored RPipe。

`tools/revalidate_run.py` uses a separately loaded verifier-argmax gate with the unchanged execution package. It finalises existing raw evidence without prepare/execute, retains the previous failed result by SHA256, and supports native Study process after all declared cells are revalidated.

`tools/analyze_4b_geometry.py` reads existing full480 traces for position/depth/trim/norm audits without model forwards. `tools/capture_4b_geometry.py` adds read-only hooks to the unchanged 4B decoder after observational non-preemptive admission, binds targets to the actual accepted path, and keeps raw vectors local; see the Study's `docs/GEOMETRY_PLAN.md`.
- `tools/analyze_4b_velocity.py`：已保存向量的径向/切向分解与全量 seed 深度转移，零模型 forward。
- `tools/diagnose_4b_mask_windows.py`：4B 固定 cache/树的mask对照；`--early-template`取冻结早期窗口，`--reciprocal --input-origin`反向重建原输入轨迹，`--direction-controls`执行固定模板/三种子同范数随机/零速度/内容初始化。数据与prompt不改。准入只看GPU0实际驻留Python及空闲显存，不枚举或登记CPU进程。
- `tools/analyze_4b_window_results.py`：固定窗口 logits 的原生树重建、节点预算与正确父节点的候选覆盖反事实，不冒称实际接受收益。
- `src/spec_mtp/structure/algorithm/reference/esp_reproduction_artifacts/`：非作者官方HF第三方原码；ORIGIN绑定commit与逐文件SHA，upstream文件不格式化或改写。`tools/audit_esp_third_party.py`仅作外部BF16固定输入与数学门验收，协议见Table1 Study的`docs/THIRD_PARTY_PLAN.md`，不替换原生解码器。
- `studies/qwen3_4b_bf16_directions/`：固定480条BC30方向对照；Study作用域只控制输入速度方向，复用原生解码/验证/Untimed合同和旧零阶结果，单队列串行，不修改科学包或数据。
- `tools/analyze_4b_full_directions.py`：复用五方法native读出，新增方向相对template和旧五方法的逐promptτ/accepted/calls配对；pending仅记进度，不补结果或写平局。
- `studies/qwen3_4b_bf16_directions_recovery/`：原队列终态后仅补未完整因素，原方向control不变，准入拒绝在同Run prompt边界等待而不丢弃进度；新version/Run，保留旧资源失败及已成功格。

`tools/analyze_esp_table1_gap.py`仅用标准库读取既有8B/4B证据，复算Table1差距、尾部/prefill计数与分组/配对账本；不加载模型或GPU，不改变原Run。

`tools/reconcile_esp_table1_contract.py`只在CPU重渲染既有Table1输入、核对完整轨迹/EOS及调用权重分解；前100 token仅为已观测路径计数，不是新生成。需要本机原Run与已有tokenizer，不下载权重或加载模型。

`tools/analyze_esp_repeat_roots.py`只读已有单mask原480轨迹，核验重复root被剪枝时的回退数量；同调用数乐观账本不是真实关闭剪枝收益，不加载模型或GPU。

`tools/gpu_admission.py`提供只读NVIDIA XML显示/CUDA分类，供新Study准入复用；不杀或挂起进程。`tools/audit_4b_norm_queue.py`逐完成格核验native/raw/argmax并汇总三种τ，支持七格续跑来源映射；不启动GPU或重试execute。
