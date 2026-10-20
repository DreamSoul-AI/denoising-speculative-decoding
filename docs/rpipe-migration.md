# RPipe 职责迁移与验收

## 2026-10-20 第九批实际 Run 缓存元数据

接续 S1/R1 的既有持久化计划准入合同：生成的 Run 子进程按 `runs/<run_id>/config.yaml` 执行，缓存 job 的显式 mode/device 也须与该实际配置一致，不能仅核 job.config 指向的外来文件。原 job.config 文件存在时的校验保持；实际 Run 配置存在时额外核对显式字段，不给旧计划省略的可选字段新增必填要求。配置缺失的原虚拟计划兼容、已完成过滤、原异常处理与 make 重建信号保持，不新增状态或解析框架。

本次对应上游已提交补丁 `7bbfe8b08b9041fc7f99dbd8d81438507acb2354` 的局部合同；本仓独立公共入口红绿另行核验，不用上游通过代替下游验收，也不整体升级固定副本。

“显式”在本批指非空声明，与上游job.get语义相同；None／空字符串继续作为未声明，不能只按键存在而新增旧计划拒绝。独审后追加四个原读取合同反例，再用后续普通提交收窄两条件；旧f696d0c时点及原件保留，不改写提交历史。

独立真实两入口先 make 合法来源，再改变缓存路径指针，`launch --mode eval` 原先实际进入 train hook；设备反例只调用计划读取，不启动 CUDA。最小补丁只加实际配置检查。固定原102、当前103、十一项精确差异、十份原 blob及许可证保持；manifest只更新原scheduler本地摘要与理由，不改原身份或放宽 checker。完整反例、原件和适用范围见 [记录](record.md) 与 `.tmp/debt-batch9/`，非科学模型验收。

## 2026-10-20 第四、五批读取与拒绝残留合同

继续 S1/R1 固定副本迁移：仅补原有容错读取边界漏接的 `RecursionError`，包括四个结果读取器、持久化 jobs 计划、Tracker state/JSONL 及只读 activity；严格 decoder、report、异常继承和原有 I/O 范围不变。先用实际超过本机 JSON 深度边界的公开 CLI 反例确认，再局部加入同上游已有的异常处理。Python 3.13 的 2000 层夹具仍能解析，首轮保留为无效控制，不计源码反例；新夹具采用 10000 层并保留严格拒绝对照。结果、state、JSONL 和 activity 的只读原字节保全；JSONL 邻接有效观测继续保留坐标和数值。坏缓存计划沿原 make 信号重建。

另参考上游 `d8fe6d0dcf3701c3d4cdd35cdfc986501d42963d` 的一行拒绝传播：未选入本次 jobs 的旧 eval 需要纠错但 Run 已被占用时，也登记到既有 `refused_runs`，由 CLI/Bash 原拒绝门阻止 Study 汇总发布。不得改写被占用 eval、旧 token 或先前 Study process，不重试该拒绝；空闲纠错与本轮已成功任务保持。下游须有独立公开入口红绿，不以上游通过替代。

固定 v0.3.1、102 条原来源记录、许可证、科学源码与历史 Study 保持。新增 activity 一处精确本地差异及对应原 Git blob，原白名单不放宽为任意路径；其余原 blob 和指纹不改。原件分别保存在 `.tmp/debt-batch4/` 与 `.tmp/debt-batch5/`；只做离线 CPU 工程控制，不下载、训练或启动 GPU/计费资源。

<a id="r1-local-backport-20261020"></a>
## 2026-10-20 固定副本 R1 本地迁移合同

本次按用户明确要求迁入完整 Run 执行权保护。上游来源仍固定为 `7a15c46b75c5eb951cccd60ed7b9fc3463db90f5`（v0.3.1），ORIGIN 首行、版本号和 MIT LICENSE 不变；只参考 RPipe `e33bcc8818f43e1f7ba875f6cd8a447d3378d864` 的 R1 合同，不迁入 R2 来源复用、shared 资源业务或其余版本升级。此项授权仅覆盖本次可审查补丁。

`flow/runner.py` 在任何日志、阶段或失败结果写入前取得非重入 OS 锁，覆盖完整阶段链；持锁执行写入新的 `execution.json` attempt token。调度器的环境 token 只消费一次，直接 Runner 调用仍使用新身份。OS 在异常、中断或进程死亡后释放锁，残留锁文件无需删除。调度器所有结果失效写入也在锁内进行；子进程的零或非零退出均须核验它实际持有的 token，忙碌或失主执行不改他人的结果，也不默认重试。`run-one` 忙碌退出 75；`launch` 和生成的 Bash payload 遇拒绝退出非零，并在 Study process 写入前停止。不同 Run 各自持锁。

有调度 token 时，锁内先完成初始冻结检查，不依赖是否启用 Study 阶段；通过后才写新 token、失效旧成功结果及进入业务。初始拒绝不会变成本次子进程的已准入身份，调度器据此保留旧结果而不重试；prepare 原有的后续复检保持。解锁或关闭本次锁句柄的 OSError 保留同一业务/中断异常并附诊断；没有主错时照常失败，未知程序异常不吞。

准入时读旧结果仅把缺失、损坏 JSON、非对象 artifact 与递归损坏视为无可复用旧结果，继续持锁业务。PermissionError 等其他读取 I/O 首错原样传播，业务阶段不进入，旧结果保持；不以宽泛 OSError 捕获掩盖证据读取失败。

旧布局没有上游新的 shared/scheduler 分拆；本地只新增 `structure/artifact/execution.py`，将所需 OS 锁与 token 集中在其中。旧 `structure/make/schedule.py` 保持公开位置，同时补齐锁建目录前的 Run ID、重复路径、jobs/batches 一致性验证；不复制第二套 Runner。既有 `tools/launch_once.py` 的显式关闭重试调用保持兼容。

[manifest.json](rpipe-source/manifest.json) 记录完整固定副本每个文件的原 Git blob、原字节 SHA256 和规范换行 SHA256；仅原有 Windows `__init__.py`、本次 `flow/runner.py`、`flow/cli.py`、`structure/make/schedule.py` 列出本地差异及可离线恢复的四份原 blob，新增 execution.py 单列补丁身份。Git 原 blob 保留原换行字节；工作树核验只将 CRLF 归一为 LF，避免 Windows/Linux checkout 差异冒充代码变化。`python tools/check_rpipe_pin.py` 核对文件集合、固定来源、明确补丁及原 blob；CPU workflow 和 package 测试均调用此检查。原件属于开发迁移证据，无新增生产 package-data 配置。

102 条原来源记录按排序、紧凑 JSON 序列化的固定指纹为 `69bb1b44fc73cdfa253e5e1c2d182f9db76a3c03717f85e5776bc604f4c9362c`（包含文件名、Git blob、原 SHA 与规范换行 SHA）；原 SHA 映射独立核对指纹为 `fe8fe06f22737235b5e2d09267ba82b03fbc6cc1eb213feabddb451614ed2550`。检查器固定前者及本次 patch ID/reference，只允许上述五个本地差异、四条原 blob 精确安全路径；不能靠给任意文件新增 local_sha256 或改原来源清单放行。三个既有框架文件保留 CRLF 工作来源，原件通过 `.gitattributes` 禁止换行转换。

历史 Run、来源锁、raw、科学算法、配置、结果与队列不改签。本次验收使用新的离线 CPU Toy Study 和本仓两个公共入口，原件保留 `.tmp/debt-batch2/`；上游通过不替代下游验证，CPU 工程合同不代表科学训练或真实模型复验。

## 2026-10-20 第三批固定副本读取与计划准入残留

本批按用户最新要求继续处理 S1/R1 迁移残留。固定来源仍为 v0.3.1 的完整 102 条原记录；只补已有损坏 JSON 容错读取器漏接非对象 artifact 的边界，以及持久化 jobs/batches 在过滤已完成 Run 前的原始计划验证。参考上游 RPipe `e33bcc8818f43e1f7ba875f6cd8a447d3378d864`，不整体升级副本，不改变 decoder 校验、异常继承、既有 I/O 容错范围或 R2。

调用范围为 scheduler 的结果复用判断、readout status、Run/Study 两级 process 和从 jobs.json 恢复计划；两个公共模块入口均需验证。已有非法 JSON 被视为不可复用结果，合法 JSON 的 list/null/scalar 也须走同一分支，status/process 不改原 result；launch 仍由持锁 Runner 写本次新结果。计划先验证完整原始 jobs/batches，不能先丢弃错误项或已完成重复项后宣称空计划成功；坏持久化数字应拒绝复用，再由已有 make 路径重建，不允许裸解析异常绕过入口处理。

先在新的离线 CPU Toy 工程目录留下失败与对照，再做最小局部移植和回归。新增三个 reader 文件列入精确本地差异白名单，保存对应原 Git blob；当前共八项本地差异、七份原 blob，103 个活动文件。102 条原来源记录及固定指纹保持。独立调用链复核还检查了传入 loader 的间接读取；严格 loader 与 report 的既有严格拒绝不改。原件与无效控制器调用分别保留 `.tmp/debt-batch3/`，无 GPU、真实模型下载或历史 Study 重跑。

## 2026-10-20 第三批 make 写入权迁移合同

同批跨仓复核继续确认上游 R1 `2dc46fa35c64bc1905caf31b8e45449f1b6bdb44` 的 make 持锁边界也需在固定副本分别验收。真实反例使用两个展开 Run，持有后一个 Run 时，配置入口、高层展开与原生 make 均仍覆盖配置；正常入口作为对照。不改研究 version、历史 Run 或整体模板版本。

旧布局的高层 `expand_study` 位于 `structure/make/__init__.py`，因此最小适配必须同时登记该文件和 `structure/make/write.py`。从原函数抽出纯候选解析和内部配置 writer，两个公共入口先取得全部目标的既有非重入 Run 锁，再写配置；高层锁持续覆盖 recipe、index 与 provenance，避免调用再次取锁的公共配置函数。候选解析不写业务文件。锁准入可以创建 Run 目录和 `.rpipe.lock`，不承诺多 Run 发布事务，不涉及 R2 或新的来源业务。

新增两份原 blob 与精确本地差异后，固定原 102 条记录、103 个活动文件及许可证保持；本副本总计十项本地补丁、九份原 blob。验证后目标忙不先改前目标、正常发布、真实 Runner 在各 writer 阶段被拒绝，以及 writer 异常后全部锁释放。专项薄入口的 Study 来源锁仍按原有独立合同写入，本次模板锁覆盖 Run config/index/provenance，不改专项来源协议。

## 2026-10-14 接续范围

原交接执行基线为 RPipe main v0.3.0 `804b3823ab6d2af46cbda1a65688ca8b76c60b22`，研究包为 spec_mtp。继续 refactor/spec-rpipe，原 PR #4/#5 已进入 dev，新增工作不复用历史实测身份。src/rpipe 与原始证据长期保留；不覆盖 D:/GitHub/diaoenmao/RPipe 中其他任务的修改。

## 旧职责到当前职责

| **旧实现 / 接续前状态** | **目标落点与复用** | **迁移边界** |
| --- | --- | --- |
| b1f6191 的 spec_mtp/flow/runner、context、调度和 artifact 副本 | 固定 src/rpipe 的同一执行库 | 前次已迁移，不恢复第二套框架 |
| eval.py 的解码执行 | structure/algorithm/eval.py | 原 mask/EMA/tree/KV/接受/预算实现保持，execute 调用已有 Algorithm |
| eval.py 的 collect、Tracker 与 CSV 写出 | flow/collect/run.py | 移动原函数，不改变汇总与数值口径 |
| flow/checks.py 的纯证据核验 | structure/artifact/validation.py | artifact 位于两柱内；读取 Run 证据，不解码 |
| flow/checks.py 的 Run 成功门 / Study 聚合门 | flow/summarize/run.py、flow/write/run.py、flow/process/run.py | summarize 和 write 复用相同门；process 区分 scope，只验当前 index |
| 根 cli.py 的薄入口实现 | flow/cli.py；根 cli.py 为兼容导出 | 模块入口与 console script 均调用同一 RPipe，无第二套状态/恢复 |
| 根 recipe.py 的注册/构造检查 | structure/api/recipe.py；根 recipe.py 为兼容导出 | 注册不下载或执行计算，各层注册经 API 组装 |
| data/__init__.py 的业务函数 | data/datasets.py；__init__ 仅导出 | dataset、tokenization/EOS 和旧 get_prompts 接口保持 |
| Algorithm 直接调用 Data/Model 输入桥 | structure/api/data_api.py | 原 resolve_prompts/resolve_eos 移入 API；data/prompts 为兼容导出 |
| artifact 核验调用 Algorithm 设置 | structure/api/algorithm_api.py | 经 API 访问原 task/spec/config 合同，不复制解析 |

prepare 与 execute 使用 RPipe 默认阶段，不填空 hook。专项 make 复用 RPipe 的展开、脚本、调度和 provenance，未增加无职责的 make 包。Study 的阶段包只导入 src 中的实现，__init__ 不放业务。兼容 facade 保留已有实际调用，不恢复旧 runner。

## 阶段输入、输出与成功门

prepare 读 make 产生的完整 config、来源锁和输入，recipe 构造 Data/Model/Algorithm，并按原约定落 shared/data、shared/model。execute 写 decode_inputs 与 decode_observations；collect 原样计算 CSV/Tracker；summarize 检查 exact_match=1、token/轨迹及输入/权重/runtime；write 再验后由 RPipe 定稿。process 的 Run scope 用默认处理，Study scope 验当前因素×seed、来源与证据，失败同步 complete=false。配置不在 Flow 中改写。

旧根 CLI / recipe / flow.checks / data.prompts 仅保留导出以兼容已有入口；新 Study 使用阶段入口。新源码和文件移动进入专项 manifest，make/remake 与新 version 先于实测。历史 root smoke 不运行，不重新签收其锁或成功产物。

## 验证与研究边界

先导入与 AST/调用方核对，再做隔离 pytest 的真实 CLI/失败边界和固定 36 格 token/接受/cache 对照。配置、精度、TF32、backend、RNG 与原配方保持。再执行真实预训练模型的最小 baseline Study，Toy/Stub、随机 tiny 和真实模型的质量门分别报告。GPU 并发和大矩阵仍不执行。

宏观问题来自 research/research-plan-v6.md §5.11 / E1–E2：先确认已有输入空间基线和 AR 的可比较原始证据，再进入未实现的输出空间方法。真实最小 Study 不检验 H1–H13，不把输入空间 EMA 冒称 Kalman。计划、数字和报告只放对应 Study。

## v0.3.1 接入设计（2026-10-14）

固定新基线为 7a15c46b75c5eb951cccd60ed7b9fc3463db90f5。现有职责迁移保持，不再次搬动算法或阶段。升级前逐文件比较完整副本，旧副本备份在 .tmp/v031-20261014/vendor-v030；新副本保留完整源码、MIT LICENSE 与 Windows torch-first 唯一例外。

上游变化只有 __init__.py 的版本号和 structure/artifact/readout/report.py。report 先读当前 index，核对 Study 名、Run ID、seed、factors，再由 artifact 的 atomic_write_text 写 NUMBERS。它允许身份完整的部分聚合，不意味着研究质量通过。专项 CLI 的 report 先执行来源预检与上游只读身份校验，再做 Study 证据终验；缺失 raw、exact-match 失败或矩阵不完整先拒绝，不能因上游允许部分聚合而放宽。旧 process、新 index、损坏/缺失清单、重复/缺 Run、seed/factors 漂移和原子替换失败均需保全旧 NUMBERS；不重签旧 Run。

配置与阶段输入输出不变；新来源需新 version 和 Run ID。先验证 import、离线模板六阶段、专项 tiny 数值/来源/证据失败门，再验证 wheel/sdist 源码外安装、Windows 和原计划真实四格。CUDA torch 保留，真实四格先串行校准，不更改配方或扩大矩阵。C-004 另做新隔离复现，不在 vendor 中修改 Runner。

report 共存回归发现专项 CLI 用空 StudyProcessContext 复验，缺 raw 时虽然拒绝 report，旧 process.complete 仍为 true。修正为先用上游 render_numbers 只读核对当前计划身份，再加载已核对的 process 到上下文，最后执行专项 Study gate；失败写 complete=false，成功继续调用原 report 原子导出。旧身份拒绝发生在专项写回之前，保持旧 process/NUMBERS。解码与指标不变。

## B-008 有效因素终验设计（2026-10-14）

v3四个真实Run成功，但Study expected_factors_match=false：上游index的factors是合并后的有效Run取值，原专项把它与cartesian轴补丁比较。含嵌套algorithm.config轴时，version/eval与继承mask在index存在而补丁不含这些字段。不是数值或上游执行失败。

修复从冻结experiment_config.yaml与study声明独立推导期望：复用上游expand_patches、experiment_config_from_mapping/run_config_from_merge及experiment_entries分组；不读取当前index/result作为期望，不改配置、轴、阈值或科学Algorithm。Study hook传入实际基底；已有单参declared_factors保留仅轴展开兼容。子集、继承固定字段被篡改、seed或来源失败仍拒绝。先用嵌套轴CPU真实CLI做红/绿回归。v3 raw/index/process/失败保留，不重签成功Run；源码变化后以新version v4复验同四case，这是失败整轮的纠错验收，不重跑历史正式矩阵。

## Output-space next-stage contract (2026-10-14)

New [Study PLAN](<研究计划 Denoising Speculative Decoding.md#legacy-studies-output-space-prerequisite-docs-plan>) defines tied order0/order1, two-slot chain, fixed gains and normalized final-hidden mapping. Output provider owns state/owner masks; Model exposes final normalized hidden without all-layer storage; decoder owns observation timing and keeps existing verifier/tree/cache. Existing input providers and scientific presets remain unchanged. No untied/general-tree/Kalman claim.
