# testing

第八批 S2 预期结果报告：原始 phase outcome 保持 pytest 值，另保存 wasxfail 属性及原因（包括空字符串）。重建分别列 xfail／非严格 xpass，普通失败仍优先，阶段不全仍不完整，完整报告继续要求所有用例为普通 passed。pytest 原生退出政策保持；strict XPASS 原生 failed 无该属性，仍按 failed，不解析消息另造政策。旧事件缺标记时不补造历史预期结果。

真实 pytest 子进程的 call/setup/teardown、空原因、普通 skip/fail、strict XPASS 与正常控制首红 8F/4P，修复后含既有持久化检查共23P；report-only CLI 与 API 重建不改 manifest/events 原字节。数字和证据见 [记录](record.md#2026-10-20-s2-预期结果持久化)。不涉及固定副本或研究模型。

第七批 S3 总收尾合同：`ci_local.main` 在解锁、关闭协调锁、写 summary 或输出保留路径时发生 I/O 次错，仍须尝试其余收尾。已存在的 setup/execution 首错保持同一对象和所属树诊断，最终 I/O 诊断独立附加；没有首错时，首个收尾错误继续失败。发生任何收尾错误即保留本轮证据。范围不扩到初始锁获取或新的平台支持；Windows 上 POSIX flock 故障用 API 注入验证，不冒称 Linux 实跑。

首错仅指本次 main 调用，不能继承调用者正在处理的异常。真实 CPU setup 超时和最终 I/O/diagnostic、ambient 调用、无首错控制及相关故障共25项通过；分阶段反例、中间20P与最终验收见 [记录](record.md#2026-10-20-s3-本地-ci-最终-io-首错保全)，不重复累计。

未选 eval 拒绝回归从合法三 mode make 进入 `launch --mode inference`，覆盖两模块入口、父失败/更新、busy/idle。真实 inference 必须单次成功；busy eval 和原 Study process 完整保全，idle 允许原纠错。4F/4P 后仅补共享 scheduler 一行拒绝传播，8 个新入口和既有 Bash 拒绝门共 9P；证据与边界见 [record](record.md#2026-10-20-s1-未选-eval-忙碌拒绝传播)。

第四、五批的深 JSON 合同使用 10000 层实际输入，不修改解释器递归限制。两公共入口覆盖四个结果读取路径、坏 jobs 缓存重建、activity、Tracker state/JSONL；严格 decoder 必须继续拒绝，坏行前后有效曲线及原字节必须保留。有效首红 14F/1P 加 activity 2F，相关正常控制与修复回归 27P，pin/docs 10P；逐轮时点见 [record](record.md#2026-10-20-s1-深-json-读取残留)，不合并重叠数字。当前十一项精确本地差异、十份原 blob、102 条固定原记录，由原 checker 核验。

`test_make_ownership.py` 验证配置 API、高层展开、原生 make 的后目标 busy/idle，以及实际 FlowRunner 在配置/index/provenance发布期间被拒绝、writer 异常后所有锁释放。仅新建 CPU Toy 声明，不构造模型或执行训练。原 activity.json 进度与锁目录不属于业务字节回滚合同；原件、无效控制及实际相关范围见 [make 接续记录](record.md#2026-10-20-s1-make-发布持锁接续)。

## 2026-10-20 第三批固定副本残留

`tests/spec_mtp/test_run_ownership.py` 新增两个真实模块入口的 status/Study process/launch 和 `run-one --phases process` 非对象结果回归，以及直接 scheduler 的 list/null/scalar/非法 JSON 对照。只读入口保留原 result 字节；launch 仍经持锁 Runner。持久化计划测试覆盖过滤前的非对象、重复完成 Run、错批次、缺批次、坏数字和负 round，合法顺序/完成项过滤保留。

65 项相关入口/pin/docs 先通过；独立复核增加 individual 补丁后，21 项 process-only/pin/docs 通过，两组有重叠。逐轮数字和源码时点见 [record](record.md#2026-10-20-s1-第三批迁移残留)。可复现命令为 `python -m pytest tests/spec_mtp/test_run_ownership.py tests/spec_mtp/test_package.py tests/spec_mtp/test_docs.py`；本机各调用另有 20/30 秒 CLI 截止、180 秒整轮 Windows Job 守卫，只清理所属进程。`python tools/check_rpipe_pin.py` 同时核八项精确补丁、七份原 blob 和不变的 102 条原来源。完整原件在 `.tmp/debt-batch3/`，不随 clone 提供；本机 CPU 工程检查不替代真实模型、CUDA 或全部支持版本验收。

`tests/spec_mtp/` 测 SpecMTP 以及它和这份 `src/rpipe/` 的合同。RPipe 自己的行为在上游测。

```text
set PYTHONUTF8=1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]" --no-deps
ruff check src tests studies tools
mypy
python -m pytest tests/spec_mtp
```

工作流：`.github/workflows/cpu-checks.yml`。重放：`python tools/ci_local.py`。GPU Study 和真实权重不在 CI。CPU冒烟仍检查独立AR exact=1作为固定fixture回归；原生成功门是完整verifier argmax审计，AR差异仅诊断。

固定副本按 [R1 迁移合同](rpipe-migration.md#r1-local-backport-20261020) 验证：`python tools/check_rpipe_pin.py` 核对固定来源加显式本地补丁，`tests/spec_mtp/test_run_ownership.py` 从本仓 `spec_mtp run-one/launch` 竞争新的离线 CPU Toy Run，检查旧成功/失败结果、日志、checkpoint、token 的逐文件字节保全、拒绝不默认重试、不同 Run 并行、异常/中断/死亡恢复；另检验一次性 token、零/非零失主退出、有效失败重试、非法计划在建锁目录前拒绝，以及生成 Bash 的拒绝 payload。所有测试子进程有单次期限和所属收尾，Toy hook 等待最多20秒；Bash payload 在 Python 中执行，本机未执行 Bash 外壳。Windows 本机结果不能代替新版 Ubuntu CI。

同一选集还覆盖初始冻结检查先于新 token 和结果失效、真实 launch 父门通过后源发生变化的子进程拒绝、锁获取/标记/解锁/关闭 I/O 次错保全，以及未登记 local SHA 和伪改原来源清单的拒绝。S2 显式相对报告目录在 pytest 配置时转绝对路径，避免用例切换 cwd 使事件写入丢失；真实 pytest 子进程回归核验完整报告。

准入旧结果的读取边界另通过实际 `run-one` 与 Runner 的 token 路径回归：合法 JSON `[]` 和损坏 JSON 均能进入 owned 阶段；注入 PermissionError 时保留同一异常对象、不进入阶段且旧结果字节不变。

本地 `cli()` 测试子进程默认单次300秒，可按用例传入更短的 `timeout`；正常退出、超时与中断均收尾本次命令拥有的进程；Windows用挂起创建→归属私有Job→恢复避免后代逃出归属，Linux收尾原独立进程组，要求/proc及waitid(WNOWAIT)，其他平台在启动前明确拒绝（主动setsid的后代不在组内，Linux本机未实测）。捕获输出使用临时文件，避免后代持有管道阻塞入口；清理权限不足或未确认时保留错误，并在 `.tmp/cli-timeouts/` 保留命令、stdout和stderr。`ci_local.py` 从每个job的checkout开始消费workflow的 `timeout-minutes`（当前30分钟），各step使用剩余job时间与自身期限的较小值，setup也使用剩余时间；整轮期限触发后，所属树停止与直属进程回收各另限10秒，Git工作树清理另限30秒。超时或非零退出保留工作树、step日志及 `summary.json`，成功默认清理。该期限对应命令执行范围，不等同GitHub runner镜像、缓存、排队及PR merge commit的完整重现。

默认pytest入口将最终选中的nodeid清单、每次尝试的setup/call/teardown事件及逐阶段耗时写入 `.tmp/test-reports/<run>/`。manifest采用同目录临时文件及原子替换，更新中断保留上一份完整记录和未完成临时文件。重建仅容忍末尾未换行的损坏事件，保留原件并标记不完整；中间或已换行的损坏记录报错。失败或skip保留原因，teardown失败覆盖call通过；缺失阶段不能记为完整通过，所有尝试保留。`--test-report-dir <new-dir>`可指定独立目录；进程退出后用 `python -m pytest tests/spec_mtp --test-report-only <run-dir>`从manifest/events重建部分 `report.json`，其中missing/retry列出未完成与待复核项。CPU workflow在结束时尝试上传原件；本地CI重放把对应目录复制到自身run的artifacts后保留该run，成功工作树仍可清理。硬终止runner导致上传步骤未执行时，仍需取得原磁盘文件才能恢复。

现有 GPU 环境不执行上面的 CPU torch 安装。测试使用独立 --basetemp=.tmp/<new-id>。Ruff 的 vendored facade 例外见 [兼容记录](bugs.md)。tiny 接受数是回归基线；完整链还检查原始 token、来源漂移、证据缺失和 Study 完整性。

Transformers固定为已验证的`5.12.1`，与本轮4B完整结果的`decode_inputs.json.runtime.transformers`一致。2026-10-16，未设上限的安装解析到`5.20.0`后，CI出现75 failed/427 passed，原正数长度`cache.crop(keep_len)`被其新API拒绝（[失败记录](https://github.com/DreamSoul-AI/denoising-speculative-decoding/actions/runs/38786263761)）。本次只固定安装版本，不修改科学缓存源码或正在运行的GPU环境；更高版本兼容性尚未验收。

CPU 剖析：`python tools/profile_cpu.py`。输出身份：`python tools/decode_fingerprint.py`。Windows：`python tools/windows_selfcheck.py`。

分支门：`.github/workflows/branch-flow.yml` 的 `Branch flow` 只核对 PR metadata，不 checkout 或运行 PR 代码。工作分支先进入 dev；main 只接受同仓库 dev。默认分支为 main，两个长期分支均要求现有 cpu-checks 与 Branch flow，保留既有审批、防强推和防删除规则。

本次职责重构新增初始化器无业务定义、ModelScope local_files_only 参数和真实 Study 无权重展开回归。结构检查补充实际 tiny CLI、source 增删改/证据失败门，不能替代数值验收。真实模型前置门及其来源/图留 [input_mask_baseline](../studies/input_mask_baseline/README.md)，Toy/Stub 模板检查与随机 tiny 分开报告。

## v0.3.1 与当前 Windows 设备

`tests/spec_mtp/structure/artifact/test_readout_report.py` 覆盖旧 process/new index、Study/Run/seed/factors 身份、缺失/损坏清单、上游合法部分聚合、原子写失败及旧表保全；实际专项 make→launch→process→report 共存回归检查缺 raw 后 complete=false。

```text
python studies/smoke/validate_migration.py --out .tmp/<new-id>/tiny.json --compare studies/smoke/docs/tiny_decode_reference.json
python tools/windows_selfcheck.py --smoke --out .tmp/<new-id>/windows.json
python tools/package_selfcheck.py --workspace .tmp/package-<new-id>
```

package_selfcheck 构建 wheel/sdist，分别 target 安装，不建新 venv、不替换 GPU 依赖；从源码外目录证明导入来自独立 installed，核 ORIGIN/LICENSE 和两个模块/console 入口，再跑四 Run native Toy/Stub 与三 Run SpecMTP tiny 的 make/launch/process/status/logs/report、重复 launch 保全。它不验证从零安装全部依赖或真实模型质量。正式结果见 smoke 的当前设备报告；真实模型资源边界见 input_mask_baseline。

## 固定窗口状态／舍入因素（2026-10-15）

本轮测试清单：test_state_rounding_factors.py覆盖Phi全native/full64端点及输入不变、RMS阶段形状与native数学、真实cache根选择、单坐标同步与另一坐标/proxy保全、异常清理、等计数但不同路径、非加性交互。保留test_phi_rms、test_q178_numerics、test_cross_task_hypotheses的旧门，不用合成fixture替代真实GPU结论。

GPU实际检查另见[状态因素报告](../studies/output_space_cross_task/docs/STUDY_REPORT.md)与[算术因素报告](../studies/output_space_extension/docs/STUDY_REPORT.md)：六生成AR／native全张量／单坐标门和80直接算术格原值门通过。原生16输出与旧8格不重跑；本轮算术格不声称新的下游接受路径验收。

本轮收尾核验：37项针对性测试及Ruff通过，5个Study的科学source/RPipe冻结门通过，80格保存向量的标量差重新计算一致，18个精确字节路径的暂存blob与本机文件一致。完整原生矩阵、安装包、更多seed/层未重跑；科学源码/default/旧helper/成功Run无改动。

## source719／root720与跨prompt因素扩展（2026-10-15）

test_local_state_proxy.py覆盖固定state/proxy输入保全、fresh cache与因果attention阻止未来mask影响当前验证、mask变化确实改变后续输出、冻结六窗口、T0 seed不抽样、真实root720 source-owner与候选时序回归。实际24固定块三次bit重复、两canonical原值生成全张量/AR重现、四单位36生成AR/单坐标/teacher门见[最新报告](../studies/output_space_cross_task/docs/STUDY_REPORT.md)。不将greedy seed当独立随机样本，也不将direct block当完整生成。

收尾核验：42项针对性测试、Ruff与5个Study科学source/RPipe冻结门通过；新分析重跑保持原证据逐项一致，26个精确字节路径暂存blob与本机文件一致。旧helper/协议/科学src/default无改动，本轮3800生成新token、24额外固定块、4teacher均在预声明预算内。完整安装包/长矩阵/随机采样温度未重跑。

## 非T0耦合与owner1（2026-10-15）

test_sampling_causality.py检查原multinomial RNG被消耗且四因素共享draw、seed重复可复现而抽样token有变化、positive-temperature draft仍按top1构树、output_hidden在两温度／两seed／两order下等于fresh AR，以及owner1 rows5/6/PIDs719/720。真实权重另完成两条T=.8完整AR路径／self/graft门、原owner0和新owner1各128条件draw、12owner1固定块三重复，见[最新报告](../studies/output_space_cross_task/docs/STUDY_REPORT.md)。条件draw不是完整生成，也不以64次罕见候选零命中证明概率为零。

收尾核验：47项针对性测试、Ruff与5个Study科学source/RPipe冻结门通过；联合分析重跑与原formal证据一致，34个精确字节路径暂存blob与本机文件一致。1600生成new、12额外块、256条件draw在新协议预算内；旧source/default/配方/工具/Run未改。未做多prompt采样泛化、罕见事件统计检验或安装包／长矩阵重跑。

## 多prompt采样与罕见事件边界（2026-10-15）

test_rare_path_boundaries.py新增p=0/1端点、越界/N0拒绝、tiny p非零概率稳定计算、零观测单侧上界与最小95%命中检测数回归。真实10新单位70生成／7000new全部sampled AR／self／单槽scope门通过；两温度各4096native条件draw均零候选命中但不能排除10^-11～10^-9质量，见[报告](../studies/output_space_cross_task/docs/STUDY_REPORT.md)。不以分布质量证明有限精度sampler的极小概率可达性。

H7轨迹计数回归继续覆盖第一次收益／最终损失、净零抵消、方向保留、非语义metadata忽略、错误总计数与不共享前缀拒绝。真实新10单位中q242/123及125首变-1而净+5/+2，报告写回后实际执行下一假设，零额外model forward。

收尾核验：58项针对性测试、Ruff与5个Study科学source/RPipe冻结门通过；矩阵/rare/H7分析重跑与正式证据一致，45个精确字节路径暂存blob与本机文件一致。新7000生成new、8192条件draw、H7零新forward均在新合同范围。旧science/default/helper/Run不改，未扩展多任务/多温度/硬件，也未证明native rare尾可达性。

## 新seed方向反例与罕见条件压力测试（2026-10-15）

test_sampling_continuation.py新增12项：首命中使用真实seed、零/全命中Clopper-Pearson端点、空/错位/非法观察拒绝、压力条件命中不能提高原条件检测力、H8零/反向/保留/infeasible判读。test_sampled_depth.py新增7项：两侧相同正确第一候选且实际node1接受、合法node2翻转、第一候选不同/错误、未接受、EOS/缺少第二目标/无翻转控制。

真实六单位4200new/3806forward全AR；163840native条件draw含T=2命中2/7，原温度仍零且检测力不足。H10检查34共有原冻结窗口无合法depth2翻转；回归与合成fixture不替代上述GPU或有限捕获边界。

收尾核验：80项相关回归、Ruff与5个Study科学source/RPipe冻结门通过。新增19项统计/合法性回归，六单位42生成4200new3806forward、163840native条件draw、H10零新增forward均在冻结预算；正式分析再次执行逐项一致，来源/model/输入/旧helpers核对保持。源/default/旧合同/成功Run未改，未推送/PR/合并/删除。新sampled同KV四因素、原温度native极小尾可达性、安装包/速度/多设备/长矩阵未重验。

## 4B模型接续与禁用计时（2026-10-15）

`test_4b_pairing_contract.py`核对11格BF16配置、480条同输入、来源/输入不符在forward前拒绝、时钟与capture异常恢复、CUDA loader保持BF16；`test_output_hidden.py`覆盖BF16多深度/兄弟树与非首兄弟真实hidden/cache路径，`test_smoke_study.py`新增tiny HF BF16五方法+独立AR六Run原生CLI及owner/范数篡改拒绝。均为CPU tiny/Stub回归，不代表真实4B结果；实际运行与赢平输见[4B PLAN](<研究计划 Denoising Speculative Decoding.md#legacy-studies-qwen3-4b-bf16-five-methods-docs-plan>)。

`test_4b_bf16_readout.py`覆盖接受数/calls各自赢平输、调用数反向判赢、缺失/错位/重复拒绝、AR长度分叉、11格pending不补零，以及native验收后480条合成读出格式；相对AR调用数控制涵盖胜/平/负和qid错位拒绝，保留长度不同的反例。该测试的mock仅验证读出与保留反例，不冒充真实模型或独立native验收。

`test_4b_bf16_plot.py`覆盖胜负矩阵行/列方向、互换W/L时保留平局、调用数不重复反向、未完成/失败/缺格/缺对/重复/计数不一致拒绝，以及合成计数PNG渲染。图形8项与读出8项回归通过；合成布局已检查，不代表模型结果。真实图仅在11格和native process通过后生成，并记录绘图器SHA。

test_4b_untimed验证三method tiny token/全部语义/calls保持、全局时钟不改及异常恢复、CUDA loader dtype/kwargs/恢复。test_4b_phases验证四phase的函数导出；test_4b_pairs保留正负零及拒绝缺配对。test_4b_input_control验证历史EMA字段/同链/原26与4输入manifest复用。原native配置展开新增4B Study，仍不加载真实权重。实际模型硬门来自10个成功4B native Runs，而非tiny fixture：first26/new4四格+EMA两格，所有输出和两类AR逐token一致。

## 4B mask 几何诊断（2026-10-15）

`test_4b_geometry.py` 覆盖 trace/root 位置与总计数、未来目标槽位及尾端跳过、共同前缀和重复点拒绝、固定 state/proxy 首槽无直接速度项、原生 tiny BF16 三方法只读观测器的输出/forward/argmax/hook 恢复，以及准入容量、未知作业、已核验 CPU 脚本字节和父 PID 链。几何汇总回归保留 raw 槽范数比与映射后范数比的区别。tiny 回归不替代真实模型证据；真实 12 格及全 480 条 CPU 复核见[诊断报告](research/4b-mask-geometry-2026-10-15.md)。
## 4B 固定窗口与CPU状态分析（2026-10-15）

`test_4b_mask_windows.py`覆盖槽差径向/切向、同prompt相邻转移、原生shadow纯推演、同前缀输入控制、局部BF16控制续接及argmax/bonus目标对齐、GPU准入、条件选择拒绝、树预算/父节点/重复剪枝、并列排名区间和固定模板差分代数近似。真实局部干预和CPU分析见[阶段报告](research/4b-mask-window-2026-10-15.md)。候选覆盖界不当作GPU实际接受数，条件样本不估计总体τ。

模板对照新增纯速度替换保留更新mean、零速度等价updating ESP（区别frozen mean）、四格tiny BF16原生窗口的控制续接/argmax/bonus门；该文件共10项通过。扩展CPU读出时对旧24行及5组比较作真实数据回归，全部保持。真实4B新增两批24格、100次forward及75次局部选择自检见报告第八节，完整logits仅留本机。

最新准入回归只允许调用nvidia-smi，覆盖WDDM桌面与本PID放行、GPU驻留外部Python拒绝、显存不足拒绝/已加载后不重复扣容量、空计算进程列表放行；不再检查CPU进程或SHA。早期同前缀16格、44次forward及54次局部选择自检见报告第九节。

追加原输入一阶轨迹的反向capture回归，核验native mask逐元素相同、当前owner不变、原输入后续mask恢复及旧Run续接；随机偏置同范数、独立种子可重复与内容初始化保留完整mean均覆盖。窗口11项加文档3项共14项通过，全仓Ruff通过。真实双向24格72次forward/72次局部选择检查、方向32格76次forward/99次检查见报告第十/十一节，不能把重复控制当独立样本。

## 非官方ESP原码验收

全量方向新增`test_4b_full_directions.py`七项：五种声明控制逐token提交时完整mean与零阶一致、固定v保持、随机同范数/可重复、content后缀初始化、五格tiny BF16原生decoder的argmax全覆盖，以及τ配对按精确比值判平和category pooled聚合的差别。只验证控制与原生边界，不冒充真实480条结果。

`test_esp_third_party.py`核对8个原字节文件SHA、三个Python文件可编译、prefix父行/剩余输出上限、直接调用原第三方验证器的self-match假阳性反例，以及真实预检失败不许转为有效τ/全480的回归。四项CPU回归通过，不能替代真实8B测量。原码排除Ruff/mypy自动规范约束，保留原有语法警告；外部审计工具与新增测试仍执行检查，真实BF16预检见[三预算审计](research/esp-third-party-audit-2026-10-15.json)。

研究计划模板引入时同步固定Transformers为已验收的5.12.1，避免CI自动安装5.20.0导致正数cache.crop接口被拒绝（75失败/427通过）；来源与恢复验证见[PR22](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/22)的626930b。未改缓存科学源码，未替换正在运行的GPU环境。

## 范数续跑与五方法准备（2026-10-16）

`test_study_phase_exports.py`核验四phase经原生dispatcher可调用、七格为旧axes精确后缀，以及篡改公共合同/轴序/旧已执行格的拒绝。`test_prepared_methods_argmax.py`对原五方法×BC30/60做十项真实CPU tiny原生decoder测试，显式cast BF16并检查实际tied权重：全部提交token审计、AR不一致只诊断、token损坏拒绝；不替代真实4B结果。完整618项与随后新增10项分别执行通过，证据见五方法独立复验Study的docs/cpu_readiness_v1.json。
