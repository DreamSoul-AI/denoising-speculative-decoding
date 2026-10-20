# 异机交接

<a id="sd-wrap-up-20261018"></a>
## 2026-10-18 最终交接：已收口，未启动下一阶段

### 当前结论与证据入口

正式中文总结已随[PR #50](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/50)合入dev，集成提交`a073845ae17870c918f550b9e3bfd81fe04594ab`；见[主研究计划收尾章节](<研究计划 Denoising Speculative Decoding.md#sd-stage-closeout-20261016>)，无需重做报告。8B为明确标注的非作者官方复现，固定SpecBench480、BF16/T0、BC10/30/60单mask；正确性门为提交token等于对应验证前向logits的argmax，239,137/239,137通过，独立AR一致率仅诊断。pooled τ为1.399047031、1.563377504、1.647301264，BC30/60比论文1.63/1.74低0.066622496/0.092698736（约0.067/0.093），**论文目标未复现**。τ不是实测加速倍数。

可随clone取得的证据：

- [8B完整三格结果](../studies/esp_table1_nonofficial_revalidation/docs/repeat_evidence_v1.json)：三种τ汇总、逐prompt计数/AR首分叉、输出摘要、原始文件SHA；[对应报告](../studies/esp_table1_nonofficial_revalidation/docs/STUDY_REPORT.md)。
- [4B原五方法加AR](../studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json)：两预算、逐prompt接受/调用胜平负；[方向对照](../studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json)：e2−e1 τ1.507777653、随机均值1.494473978、零1.422420206、content1.364829262。随机同范数偏置也有效，不证明范数是唯一原因。
- [首个0.5×范数格恢复证据](../studies/qwen3_4b_bf16_bias_norm_continuation/docs/first_cell_recovery_v1.json)：τ1.471389646，80,940个token自检通过；不重跑execute，失败记录和来源SHA保留，不冒充完整范数扫描。
- [CPU准备验收](../studies/qwen3_4b_bf16_paired_revalidation/docs/cpu_readiness_v1.json)、[PR #50 CI](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/50)：历史本机618+10分别检查，远端整套628通过；不把旧测试结果说成本次重新全跑。

### 收口状态与新机器接手

BC60、父队列和监视器已自然退出。七格等待器64880、入账器50944已停止，临时GPU启动器已停用；剩余七格与后续11格仅准备、未执行，不再按旧章节自动接卡。已合主题分支`experiment/sd-nonofficial-repeat-norm`与`analysis/sd-pending-results`的本地/远端引用均已删除；累计36本地/10远端，见[PR合后清单](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/50#issuecomment-6217073928)。实验数据/结果/模型缓存未删；旧执行工作树在原`4bb28c7` detached保留。主仓旧绝对路径只是本机定位信息，不应在新机器照路径重建重复目录。

新机器应显式checkout `dev`或上述集成提交（仓库默认main不含全部研究交付），先读本节和主计划，再按[测试说明](testing.md)配置环境。仅复算公开4B表格可运行`python studies/qwen3_4b_bf16_five_methods/verify_published.py`；这是标准库计数核对，不下载/载模。现有`tools/analyze_esp_table1_gap.py`也仅复算已提交的旧480计数，不是重新执行或对作者结果做验证。

Git不包含完整Run、原始token/logits、权重、共享缓存和本机.tmp证据。只clone可阅读正式聚合/逐条摘要、配置和源码，**不能据此再次独立核验原始前向argmax或恢复本机Run**。需要raw级对账时，应先取得原Run归档，按正式JSON逐文件SHA核验，再使用冻结执行commit和Study合同；缺文件就记录缺失，不补造raw、不重签旧成功Run、不make/remake覆盖历史。权重优先ModelScope，缺失时仅使用`HF_ENDPOINT=https://hf-mirror.com`（含datasets），两者均无则报告；不回退huggingface.co。基线硬编码海外模型/数据地址需改国内源并保留patch清单，不能把修改版叫原字节未改。

### 下一步排查差距如何开题（仅交接要求，尚未执行）

新阶段须等Enmao确认范围。开题写入唯一中英主计划的独立章节，不新建子PLAN、不沿用暂停队列自动执行。用单一问题约束范围：**固定480输入和BF16设置下，哪个可核验的协议差异能够解释BC30/60的τ缺口？**

1. 先列证据缺口与可证伪判断。复用[输入/计数对账](research/esp-table1-cpu-contract-reconciliation-2026-10-16.md)及[协议缺口登记](research/esp-table1-protocol-gaps-2026-10-16.md)，把已核验本地事实与未知作者合同分开；不重复循环追入口，不把猜测写成原因。
2. 明确唯一比较变量、基线commit/来源、固定prompt token IDs/数据与模型SHA、单mask预算、停止/EOS、τ分子分母/prefill和任务权重。先用已有文件在CPU闭合计数；三种聚合不能事后挑最接近论文的一种。作者合同缺失时，结论只能是非官方实现的敏感性分析，不能宣称已经归因作者差距。
3. 如确认需要新执行，先登记最小配对范围、预计资源、停止/失败条件和验收指标，再用新version/新Run；保留全部正负结果。主指标仍pooled τ与论文差值，另列两聚合、逐prompt接受/调用/τ胜平负、输出长度、argmax完整覆盖、AR诊断。正确性失败停止验收，AR失配不阻塞数学门。固定输入不调prompt，不测速，不做算子数值链；bias/random/Kalman等其他问题不自动纳入。
4. 运行准入只看实际CUDA计算、空闲显存和实测host峰值+2GiB；显示/远程桌面不阻塞，不杀其他任务。任何GPU执行都需在新范围确认后再排单队列；本次wrap-up没有make、launch、下载或实验forward。

本次仅更新交接与修复主计划页首断链/旧状态；正式报告主体及所有结果文件保持不变。未解决的作者协议、树/剪枝差别、计数权重仍为候选解释，不作已证实因果。

当前主模型Qwen3-4B；0.6B rare/单位置阶段已归档暂停。first26同链复验、4条新prompt负结果及历史输入EMA控制均已实际完成，见本文末尾“4B主模型配对接续”及[主报告](../studies/output_space_4b_recheck_v2/docs/STUDY_REPORT.md)。下列0.6B记录保留其历史范围。

## q178算子对照与下一Study草案（2026-10-14）

root85之后provider槽计算S64、norm精度N64、norm后端N32各self/一阶←零阶，共8次AR一致；所有完整语义路径保持self8/27、移植9/26。profiler真实关联provider的NormTwoOps reduce_kernel；这只收紧后续provider运算的必要性，入口85和transformer内部舍入仍未独立隔离。正式[报告](../studies/output_space_extension/docs/STUDY_REPORT.md)保留完整边界、scalar和资源。新[跨任务Study](../studies/output_space_cross_task/README.md)固定summarization q241–244并已分词，仅草案、未make/launch；固定行q_proj kernel隔离协议未执行。[最小提交候选](<研究计划 Denoising Speculative Decoding.md#legacy-docs-research-publication-plan>)为依赖PR6树的437f201，33文件逐blob哈希与PR草稿本地保存，后续研究暂缓。源码/default/旧合同与工具不改，未推送/合并/删除，用户workspace保留。


## q178数值隔离与q179–q182（2026-10-14）

严格FP32 q178重复全selected张量/语义路径相同，microdelta首个共同根89来自pos88的块/row计算路径不同；规范前缀anchor/v同步两self为8、两移植为9，轨迹与strict移植相同，排除committed微差作为这次收益必要条件，不排除proxy内舍入或锁定具体CUDA kernel。新4Run/16outputs/20诊断AR一致，q179原8/9→双10、q182原4/3→双4，q180/181null；q182 root47合法second hit恢复且source45/root47 state相同。累计12prompt表和所有坏点见[后续报告](../studies/output_space_followup/docs/STUDY_REPORT.md)，原PLAN保持冻结，结果单独PLAN_REVIEW。科学src/default/旧helper/Run不改，不推送合并删除，fullweights/cache不入Git。

## q175–q178扩展与累计8prompt（2026-10-14）

先在新PLAN写H1排序边界/H2实际owner历史/H3终止或未使用owner，再固定4新prompt并完成原生4Run与20次诊断，全部AR一致。q175/176/177 null；q178原零/一accepted均8但深度分布不同，root85单槽双向各变9，存在损失/收益相抵，初次root86第二槽在相同anchor/v下翻转depth2。后期state有微小浮点差异，不当作完整q173机制复现。第二槽分析明确事后，raw及全部坏点/新旧root保存，8prompt表与详细事实见[扩展报告](../studies/output_space_extension/docs/STUDY_REPORT.md)。科学src670/default/旧helper/Run保持，不推送、合并或删数据。

## q171/q172/q174固定跨prompt复现（2026-10-14）

按CROSS_PROMPT_PLAN首个共同命中前根规则选择63/42/41，复用未修改的observe/transfer工具，完成6原轨迹观测+6双向移植+3原值回填。15次输出与独立AR相同、全部验证/提交/KV轨迹不变，六移植94同位窗口anchor/v与recipient原值逐元素相同。q171/q172 first seed分别每方向34/7根变化但margin未跨top1边界；q174 terminal proxy变化后EOS停止。未复现q173命中转移，不能概括代理传播必导致收益；也未证伪所有其他窗口。正式合同、完整scalar/轨迹与离线核验见[Study报告](../studies/output_space_small_matrix/docs/STUDY_REPORT.md)。src/default/原Run不改，本地保存，不推送/合并/删数据。

## q173单槽历史代理移植（2026-10-14）

按预声明TRANSFER_PLAN完成三次单槽重放，均33token与独立AR一致、前缀/原值回填/单槽范围/实际owner门通过。一阶root83第二槽换零阶后，root94/97均失去命中，五固定根committed anchor/velocity与原一阶逐元素不变；反向零阶在velocity为零时恢复root94而未恢复root97。离线张量逐元素确认83→84(owner0)→86(owner1)代理传播边。只证明单qid/seed/入口的局部历史影响，不能宣称全局EMA机制或双向完全转移。正式合同、scalar结果、可复现工具与raw SHA见[Study报告](../studies/output_space_small_matrix/docs/STUDY_REPORT.md)；科学src670c744e0535dcad/default/旧Run与NUMBERS保留，未推送、合并或删数据。

## q173固定窗口与尺度诊断（2026-10-14）

按WINDOW_PLAN/SCALE_PLAN完成只读q173 root83/94/97重放和4个尺度反事实前向。三case完整token/接受/KV不变、src670c744e0535dcad保持；观察器不新增decoder调用，1teacher+9prefix辅助后，窗口/模型/runtime一致性均通过。root94/97一阶seed target margin从零阶负值转正；root83的2×2方向/范数对照排除这两个尺度的范数差异单独解释错误。事后结构复算显示owner0 slot1为Phi(proxy0)，清零当时velocity不改slot1，收益不能表述为当前速度直接相加。科学配方/旧Run未改，未推送、合并或删数据；正式证据与复跑入口见[Study报告](../studies/output_space_small_matrix/docs/STUDY_REPORT.md)，本机selected张量SHA在报告保存。

## 输出空间三prompt小矩阵（2026-10-14）

依据最新单prompt报告与offline_diagnosis，先固定q172–q174和预算，再用同源码670c744e0535dcad、FP32/eager/ModelScope snapshot/链式配方完成4新Run×3prompt。全部来源/原始证据/逐token AR/Study完整性门通过；accepted（输入/零阶/一阶）为1/1/1、5/1/3、0/1/1。q173 root94/97定位一阶相对零阶的2次额外接受，root83输入链仍领先；q174改善集中在句末。不调参、不宣称稳定优势。正式[小矩阵报告](../studies/output_space_small_matrix/docs/STUDY_REPORT.md)与可复算的paired-prefix诊断已保存。源码/旧成功/失败Run保持不变，本轮不推分支、不合PR、不删数据。

## 输出空间最小阶段（2026-10-14）

用户要求自主继续，采用feature/output-space-prerequisite（父1219a1d/PR6；PR6当前CI全部成功但未合并）。输出零/一阶normalized final-hidden masks只支持实际tied模型与unpruned静态链，复用验证/KV/RPipe六阶段。先写[PLAN](<研究计划 Denoising Speculative Decoding.md#legacy-studies-output-space-prerequisite-docs-plan>)再实现；270全量、Ruff、56文件mypy与旧36格严格轨迹通过。四个新真实Qwen3-0.6B/q171/FP32/seed123/100token Run已成功，来源/原始证据/Study完整性门通过。输入均值链accepted17，输出order0/order1均10，三者均与独立AR的47token完全一致。保留负结果，不调参、不扩大矩阵。报告、数字和图见[新Study](../studies/output_space_prerequisite/README.md)。源码digest670c744e0535dcad，旧input baseline历史证据未改、未重签。wheel/sdist 41条CLI命令及两独立安装target的output-hidden解码通过；正式材料已保留，最终提交按本地分支核验。本轮无新公开推送、合并或发布。

## ModelScope四格正式接续（2026-10-14）

用户已明确批准本轮代码、脱敏研究报告与验收证据公开推送及dev PR，权重/缓存/完整Run与内部资料仍排除。refactor/spec-rpipe已推送，实际 [PR #6](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/6) 已创建，base=dev、head=refactor/spec-rpipe、状态OPEN且可合并，无合并或发布。ae86c3c的Branch flow与push/PR cpu-checks已启动；本段文档提交后以PR最新head的实际checks为准，不把旧提交CI当成最终结果。下方拒绝/待授权文字为推送前历史检查点。

用户明确要求ModelScope下载并继续工作后，已显式准备新Qwen3-0.6B snapshot，保留CUDA torch，Run保持离线。加载与解码资源按单GPU/round1串行核验。v3四Run成功但Study因B-008嵌套轴期望误比较失败；修复从冻结基底/声明复用原生合并分组，不放宽门，v3 raw/失败完整保留。v4新身份四格全部成功，Study/source/evidence/有效因素/完整性质量门通过。科学decoder/Data/Model、q171/FP32/eager/T0/seed123/预算100/EOS/原mask配方未改。

两ESP各accepted19、生成47，独立AR相同输出；alignment四步accepted1/rejected3，28层raw核对通过。v3/v4全部token/接受/KV/AR计数/对齐sums等值（不比较耗时）。本次n=1只是实际前置门，不支持输出空间/Kalman或完整矩阵结论。正式资料见 [四格报告](../studies/input_mask_baseline/docs/STUDY_REPORT.md) / [v4证据](../studies/input_mask_baseline/docs/v4_execution.json)；下载snapshot、资源、原始证据与图均保存，weights/raw不入Git。

新工程验证已实际完成36格tiny严格等值及最终源码wheel/sdist双入口全链。B-008新增嵌套轴/子集回归通过；全量曾253通过/1复制fixture漏正式JSON失败，复制范围已修正并单独3声明用例通过，最终全量254 passed（177.46秒，0 failed/errors/skipped），Ruff与55文件mypy通过；正式增量记录在smoke/v031_validation.json，不将旧251批次替代新源码验收。上游Issues36/37保持真实链接，不改上游源码或权限。旧拒绝推送状态留在下方历史段；当前继续按工作分支→PR/dev流程，不自动合并或发布。

推送前历史检查点：本轮实现和验收已提交为5cfdacc9b0386d490cfc4d741b72305c450c5983。当时推送refactor/spec-rpipe再次被自动审批拒绝，理由是需对这批代码、研究报告和证据向该公开仓库/分支的特定授权；已向用户明确请求。实查远端仍49dec9c，dev仍1b6d59c，没有新PR，当前提交没有远端CI。旧49dec9c的cpu-checks成功不代表本轮；具体PR正文与同步状态保存在本机.tmp/input-mask-ms-20261014。除这一外部同步步骤外，当前范围本机实现、工程与真实四格前置验收均完成。

## v0.3.1原工程检查点（2026-10-14，95c0887）

补查缓存环境覆盖及本机模型目录仍未取得Qwen3-0.6B，4B/8B未替代。C-007的pending eval误显示train/过滤漏项已用原生CLI最小复现，读数不改变调度或文件；已提交 [Issue #37](https://github.com/DreamSoul-AI/RPipe/issues/37)。继续以真实config/jobs核对执行mode。工程逐项证据核对通过，四格真实解码、资源校准、推送/PR/新CI仍缺必要条件，整体目标未完成。

实现与本机证据已本地提交 cb5c36d57e02d068d42ad6b1f0d5195cf5fd1186。推送/草稿 PR 的组合操作被自动审批拒绝，未执行；随后确认目标仓库为公开仓库且当前账号有 push 权限。仍待明确外传批准，不绕过拒绝。当前新提交没有远端 CI 结果，交接49dec9c的绿色检查只证明旧提交。PR正文已备妥于本机.tmp/v031-20261014/pr-body.md。

已从交接提交49dec9c接续 refactor/spec-rpipe，保留未跟踪 history_mtp.code-workspace。当前完整 vendor 为固定7a15c46 / v0.3.1，Windows torch-first 唯一例外与MIT保留；旧副本只备份，不删除。职责映射沿用既有迁移，科学 Algorithm、Data、Model 与数值配方未改。

- report 采用上游身份/原子门，专项先只读身份校验再做真实 process envelope 质量写回。B-007 已修：缺 raw 拒绝并写 complete=false，旧身份与写入失败保全旧表。
- 新设备安装不覆盖CUDA torch；本机251项全量、Ruff、55文件mypy、固定36格CPU token/AR/接受/KV轨迹、Windows双导入与新version smoke、新设备三variant CUDA tiny、wheel/sdist真实全链均通过。模板与随机tiny不替代真实模型质量；正式证据在 smoke/v031_validation.json 与报告。
- C-004 在当前main同一v0.3.1的两步Toy/Stub新Run复现并备份前后result，已提交 [Issue #36](https://github.com/DreamSoul-AI/RPipe/issues/36)。未改上游代码或权限。
- input_mask_baseline 新version为input-mask-baseline-v031-win5090-20261014-v2，四Run已make，succeeded=0/failed=0/pending=4。Study/default ModelScope离线核验均CacheNotFound E1022；等待Qwen3-0.6B snapshot路径。资源inventory不等于校准，真实四格尚未launch，研究质量未通过。

接手先读取当前工作树、远端/CI与 [四格PLAN](<研究计划 Denoising Speculative Decoding.md#legacy-studies-input-mask-baseline-docs-plan>) / [前检](../studies/input_mask_baseline/docs/v031_device_preflight.json)。权重可取得后只核对原配方并串行校准/执行四格，不能以其他规模模型或Tiny替代。PLAN已经进入锁；任何来源/输入调整先明确变更与新version，不对旧成功Run重新make。原始工程日志/raw在本机.tmp/v031-20261014，clone只有正式摘要；历史下面49dec9c段保留当时状态。

## 49dec9c 原交接（2026-10-14）

本次按用户要求先提交推送检查点，然后在另一设备接续。研究验收未完成，不能将本记录当作整体通过。接手先检查实际工作区、HEAD、远端和本机环境，保留已有修改，不切回默认分支旧设计。

| **项目** | **状态** |
| --- | --- |
| 仓库与工作分支 | DreamSoul-AI/denoising-speculative-decoding；refactor/spec-rpipe；本轮基于 dev 集成 1b6d59c，不改 branch spelling |
| 当前执行基线 | src/rpipe/ORIGIN 第一行 804b3823ab6d2af46cbda1a65688ca8b76c60b22，v0.3.0；101 个上游文件只有 Windows torch-first 例外，保留 MIT |
| 新上游固定参考 | v0.3.1 commit 7a15c46b75c5eb951cccd60ed7b9fc3463db90f5；annotated tag cd7a750de0d13889916eba84b53ed4a339d47a32；未更新 vendor |
| 已完成职责重构 | [rpipe-migration.md](rpipe-migration.md)；flow 阶段、API 桥、artifact 核验、数据 init 拆分与兼容 facade；没有第二套 runner |
| 新研究 Study | [input_mask_baseline PLAN](<研究计划 Denoising Speculative Decoding.md#legacy-studies-input-mask-baseline-docs-plan>)：Qwen3-0.6B，q171 translation 新 prompt，frozen_mean / gamma_last / AR / alignment，seed123；未执行 |
| 工程证据 | [smoke 报告](../studies/smoke/docs/STUDY_REPORT.md#八职责重构检查点2026-10-14)、[本轮 JSON](../studies/smoke/docs/refactor_validation.json)：239 回归、Ruff、55 文件 mypy、36 格完整轨迹对照、4 Run 原生离线模板六阶段通过 |
| 分支政策 | main 为 default，工作 PR → dev → 同仓库 dev PR → main；保留 CI，不直接推长期分支，不自动合并/发布 |

## 接手顺序

1. clone/fetch 实际工作分支。先读 AGENTS、README、layout、architecture、testing、windows、rpipe-migration、bugs、record、studies 入口及目标计划/报告、history-extrapolation、brainstorm、research/readme 与 extraction。研究草案不自动成为实施合同；不要重复前次已完成的框架迁移和正式实验格子。
2. 核对固定 v0.3.1 的 README、发布报告、concept/layout/code/structure/flow、研究/反馈、Study 和测试规范以及实际源码。相对 804b3823，src 只有 __init__.py 版本号及 structure/artifact/readout/report.py 变化；后者先核当前 index 的 Study/Run/seed/factors 身份，再原子写 NUMBERS。它不是新的调度器/解码实现，也不自动修复 C-004。完整文档差异仍须按实际调用审阅。
3. 先同步升级设计和兼容记录，再更新完整 vendored 副本与 ORIGIN 到固定新提交，保留 Windows torch-first 例外和许可证，记录全部本地差异。上游 report 身份/保全门与本项目 Study exact-match/source/evidence/completeness 门必须共存；接口变化不得覆盖科学实现。复验当前 index 与旧 process、损坏/缺失 index、部分合法聚合、原子写失败和专项拒绝边界。
4. 新设备先记录 Python/torch/transformers/ModelScope、CUDA、实际 dtype、TF32、backend、硬件与资源。已有 GPU 环境只做 no-deps 项目安装并按需补缺，不覆盖 CUDA torch，不加 sitecustomize/全局补丁。ModelScope 权重缓存不会随 clone 提供；新 Study local_files_only=true，先按项目规则显式准备缓存，否则如实停在缺资源，不偷偷下载/改变配方。
5. 完成安装与新基线 smoke/失败回归，使用独立 pytest --basetemp。安装包须在源码目录外检查 wheel/sdist 的真实导入、ORIGIN/LICENSE、两个 console 入口和实际 make→launch→process/readout。历史 934dac3 安装通过只作旧证据；本轮尚未做 current-refactor 分发包全链和 windows_selfcheck。
6. 先保留固定 tiny 原始输出，再比较新链：`python studies/smoke/validate_migration.py --out .tmp/<new-id>/tiny.json --compare studies/smoke/docs/tiny_decode_reference.json`。36 格的 token、AR token、接受和 KV 轨迹是严格门。源码新增/删除/修改、输入/模型/tokenizer/运行环境漂移、exact-match 失败定稿、证据缺失、index 子集与 Study scope 也要覆盖，不能只查退出码。
7. 为 input_mask_baseline 设新的 algorithm.config.version（新设备/新基线/日期），PLAN 若调整先明确原因；冻结后不随意改 PLAN 或重签旧成功 Run。原配方 float32/eager/[14]/1mask/BC30/T0/100budget/seed123/thinking off，β/λ/γ/数据与 EOS 不选优调整；只验证已计划四格。先串行测量显存/主存/线程/I/O，四格独占状态/产物，不做竞争资源速度排名；原生 make/launch 参数以实际 CLI 为准。不得跑完整大模型矩阵、扩大并发或声称 KV/mask/RNG resume 已支持。
8. 将执行、原始 token/接受/cache、来源、失败与终验写入对应 Study。accepted 各深度、生成 token、τ 与速度口径分开，exact_match=1 是硬门。AR/alignment 不伪造 ESP 指标，四步 hindsight 不称 Figure 2 完整复现，未实现的输出空间/Kalman/PLD/STAND/LADE 不补写成支持。
9. 上游通用问题按固定 feedback.md：先本地记录、最小合成复现、检索 open/closed Issues 和 PR、脱敏，再可直接创建/补充 DreamSoul-AI/RPipe Issue（已有明确授权），回写真实 URL。C-004 须确认 v0.3.1/当前 main 是否仍有行为；未复现写疑似。此授权不含上游代码/PR/合并/权限修改。最后完成本项目文档、检查、分阶段 commit/push，PR 面向 dev；不自行合并新 PR 或发布。

## 证据和资源迁移边界

正式报告/参考 JSON/模板 YAML 在 Git 中；runs/shared/.tmp、原始日志、权重、缓存、凭据不在 Git，也不随 push/clone 提供。本机历史原始证据已随 checkout 补齐到新目录，其内部原执行路径未改写。异机需要新 version/Run，不可把新的 make 锁签在旧成功目录上。若要复制原始制品，先核对脱敏、来源与大小，复制不等于当前验收。

本机 checkout 已迁至 denoising-speculative-decoding，editable 指向新路径。Windows 返回错误导致部分移动后已逐文件补齐并核对 Git；原目录仅临时目录残留，没有继续强删。目标设备直接按新名称 clone 即可，不继承本机绝对路径。上游本机工作区的其他任务改动始终未触碰。

本机初轮 239 回归和 36 格对照在目录移动之前运行，执行树指纹见工程 JSON；目录移动及交接文档整理后再次全量 239 passed / 0 skipped，Ruff、55 文件 mypy、editable 导入、模块/console CLI 和 Git 连通性检查通过。GPU CUDA tiny、555 格和 wheel/sdist 的已有数字见各自历史提交/日期，不冒充本轮或新设备验收。

## 当前未完成

v0.3.1 源码升级与完整回归；当前重构安装包/Windows selfcheck；新真实模型四格 Study 与串行资源校准；上游 C-004 新最小复现和实际 Issue。真实权重 GPU 容量、并发、解码恢复、长矩阵与新设备科学验收均未完成。用户要求本机先收尾交接，因此本轮不启动新的正式解码任务。

## q178 CUDA图／固定行与cross-task checker（2026-10-14）

[最新报告](../studies/output_space_extension/docs/STUDY_REPORT.md)保存norm8格／q_proj6格全部AR一致和完整路径8/27、9/26。norm实际换NormTwoOps→mul/sum/sqrt，但所有返回值bit相同，不能当新数值鲁棒性证据；q_proj第0层85块row4换GEMM→GEMV或FP64-rounded确有微差并传播，但接受不变。4克隆cache块重放logits/finalhidden精确相同。q241–244已有[可跑checker](../studies/output_space_cross_task/check_hypotheses.py)与[状态](../studies/output_space_cross_task/docs/STUDY_REPORT.md)，仅合成MRE/质量边界测试，未launch真实batch；H2未检。入口Phi／其他层行／q179182canonical仍开放，下一target须新固定合同。scientificsrc/default/旧PLAN/helpers未改，rawKV/tensor/profiler/cache不入Git，未远程／PR／删数据。

## 真实cross-task与入口Phi／RMS推进（2026-10-14）

新cross-task fourRuns/16outputs、20窗口诊断全部独立AR一致，input/zero/one85/58/45；四条及八移植方向都有语义/计数变化，混合正负而无一阶优势。q178 entryPhi64两向量1.49e-8有效差、RMS64 row4有效差，8格路径保持8/9；singleton RMSbit相同单列。首个阳性q241另冻canonical合同，native11/12→canonical14/14，自值也变且两canonical路径仍不同，是接受计数差不保持的反例，不是移植全失效／committed唯一因果。完整[实际报告](../studies/output_space_cross_task/docs/STUDY_REPORT.md)及[精度报告](../studies/output_space_extension/docs/STUDY_REPORT.md)给来源、图、窗口坏点、资源和哈希。旧草案/批次hypotheses状态保留，真实执行与H2跟进分别增补，不重签成功Run。源码/default/旧工具不改，weights/cache/raw/profiler不入Git，不远程/PR/删数据。

## q178算术因素／q241选择性state归因（2026-10-15）

最新[状态报告](../studies/output_space_cross_task/docs/STUDY_REPORT.md)与[算子报告](../studies/output_space_extension/docs/STUDY_REPORT.md)：六生成AR一致、两native全张量重现，单坐标回填scope/真实KV根门通过；80direct算术格零forward。单因素精度均无最终差，联合中间精度才恢复旧FP64端点，不能锁定单kernel。q24113／13新格收紧“+1保持”，自值同步非加性；双canonical14／14在root720 state相同但proxy不同，四增四减抵消。解释性固定单位，不是新增随机样本。待验证是首个分叉上游冻结后的即时state与历史proxy分离，未执行、不假装完成。raw/cache/weights/profiler仅本机，不远程／PR／删数据。

## source719／root720时序与非加性扩展（2026-10-15）

[最新报告](../studies/output_space_cross_task/docs/STUDY_REPORT.md)与新两个冻结协议：source719固定KV/tree/state，proxy-only把root720候选13变264并精确重现真实graft两seed logits；state-only有非零mask变化但第一判定不变。当前720映射在候选树之后，仅改变后续seed。24块三重复bit一致，两原值控制全AR/trace/tensor重现。q242交互0/0、q243+6/-3，两个T0 seed全语义和captured seed/mask bit一致，两个新增prompt反驳非加性普遍，不构成随机seed泛化。未做温度采样／其他owner／长路径直接效应；raw仅本地，不远程/PR/删数据。

## 非T0完整路径／owner1范围（2026-10-15）

[最新报告](../studies/output_space_cross_task/docs/STUDY_REPORT.md)：真正T=.8两seed独立AR输出不同，14生成全token一致，实际窗口697/715效果-1/+1；旧source719条件抽样恢复0→64/63不泛化为新路径稳定收益。source717实际owner1 rows5/6交换proxy精确复制真实下一719 seed，11→323但均不命中1008，64draw各零但概率非零。12块三重复/两原值控制全tensor验证，条件与full-path证据严格分开；下一缺口是新 sampled窗口等state decomposition、更多prompt/温度和罕见候选效应，不声称已解决。未远程/PR/合并/删除。

## 多prompt采样／罕见功效／H7反例（2026-10-15）

[最新报告](../studies/output_space_cross_task/docs/STUDY_REPORT.md)：4prompt×3seed12单位，旧2保留、新10全部sampled AR，6增6减；q244重复负限于当前输入/T=.8/k2/100预算，q242重复正。4096draw每温度仍零罕见候选命中，检测功效不足且GPU极小概率可达性未验。矩阵报告写回后另冻FIRST_DELTA_PLAN并执行H7，新q242123/125首变-1终+5/+2收紧局部预测。下一缺口是新sampled窗口等state decomposition/原生samplerrare数值尾，不声称已完成。raw本机、科学source/default/旧工具不改，未远程/PR/合并/删数据。

## 新prompt／seed反例与罕见压力测试（2026-10-15）

新增q242244 seed126及q245246 seeds126127共六单位，42生成／4200new／3806forward全AR；graft-self为-5,+1,+3,-6,-3,-4，q242此前三正和q244此前三负都在新seed反转。总18单位8增10减、六prompt不平衡样本，不作人口泛化。固定旧owner1/verifier的T=.8/1各16384 fresh native draw仍零，T=2另131072实际候选11/323命中2/7、首seed52396/21263；改变温度只证明对应条件可观测，不证明原温度尾可达性或全路径收益。报告写回后新冻H10并执行34共有固定窗口检查，无合法depth2反例；q244126和q245127首方向/最终方向反转也保留。新统计与合法性回归已落地；源/default/旧helper/协议/Run不改，不推送/PR/合并/删除。下一缺口是新sampled同KV四因素state/proxy实验，而不是继续小量零命中。

## 4B主模型配对接续（2026-10-15）

按用户新指示转Qwen3-4B，0.6B报告/commits/raw原位归档，不再扩大rare或单位置。检索只找到旧4B输入空间EMA结果，未定位完整旧output0/output1正结果；旧树[14]/[7,2]不与新同链计数混比。既存14文件ModelScope离线目录复用，实际SHA/model digest绑定，原下载origin/immutable revision仍未确认。

同FP32/eager/T0/nonthinking/EOS/max100/[1,1]/k2/BC9：first26输入/零/一accepted238/329/321、主calls1955/1866/1876，独立AR2191；全source/model/token/Run/Study门通过。写回报告后执行预声明新4条：51/39/47、189/201/194，负结果全留。随后固定历史输入EMAβ.9999/λ.1/γ1两新Run，first26为240/1952、new4为39/202；输出/独立AR成功格按哈希复用不重跑。结论限于first26总收益，新prompt不稳定，一阶未普遍超过零阶，不推速度结论。

eval.warmup=false，Study局部仅关闭decoder clock，兼容时长字段是0占位；同权重直接CUDA载入避免完整host副本，不改科学src/default。原v1基线execute26全AR后collect导出模块失败，失败Run/锁/raw保留；v2新身份修正四phase callable后成功，不重签旧成功。后续4B应先冻结新的跨类别/长度/预算全路径计数协议，不恢复0.6Brare；no push/PR/merge/delete。正式主/new4/EMA报告分别在output_space_4b_recheck_v2、output_space_4b_holdout、output_space_4b_input_control。


2026-10-15 最新执行顺序：已有 Qwen3-4B 结果全部为 **FP32**，本轮先按主题提交、工作分支 PR 经 CI 合入 dev。其他研究暂停；最高优先复现 ESP Table 1（Qwen3-8B/BF16/T0、BC10/30/60、动态树与剪枝、mean mask λ=0.1）。8B 仅用于该复现；随后 Qwen3-4B/BF16 五方法加 AR 配对。新增实验不用 FP32、不测速。不得强推、合 main、删分支或数据。


2026-10-15 FP32 收尾已核实：工具/测试 bdb3f6c、证据 8a0969c；[PR #7](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/7) 的 Branch flow 与两个 cpu-checks 均 SUCCESS，已于 2026-10-14T19:17:08Z 合入 dev，merge 8677376e1ac878a159a3095711eac43420d97bde。未合 main、未强推、未删除分支/数据。8B/BF16 仅按 [Table 1 计划](<研究计划 Denoising Speculative Decoding.md#legacy-docs-research-esp-table1-bf16-reproduction-plan>) 继续；其他研究暂停。


2026-10-15 ESP v2 G.2协议修订：Qwen3 BC60主配置改为单mask/29 draft，双mask/19 draft固定作对照；旧冻结PLAN哈希不变。新修订预检单maskτ=1.636982、Δ1.74=-0.103018、exact3/13；双mask复用τ=1.579408、Δ=-0.160592、exact5/13，均为失配诊断值，不称成功。详见[修订报告](../studies/esp_table1_bf16_qwen_protocol/docs/STUDY_REPORT.md)。全480四格已冻结并串行启动，BF16/T0/256/seed123、Eq5 λ=.1 updating mean保持；4B frozen mean是后续不同基线，其他研究暂停。完整Run/权重/缓存留本机；没有测速、强推、合main或删除数据。


2026-10-15 最新τ主线：[全480 argmax审计](../studies/esp_table1_bf16_full_argmax/README.md)替代无argmax记录的v1。BF16/T0/默认归约/λ=.1 updating mean/动态树剪枝完全保留，BC60单mask为主、双mask仅对照。AR一致率及逐条首分叉独立报告，不作为τ执行门；全部提交token必须等于对应验证forward的argmax。v1部分日志/旧冻结合同保留；v2四Run在模型加载前缓存接入失败，保留失败后补接同一ModelScope本地snapshot，以v3新四Run串行运行。当前第一格BC10 Run dfe1b9c8d5d12e92已开始真实BF16解码，未完成全480。停止compact attention、单行RMSNorm与逐层数值机理；4B五方法仍等待Table1结果。新增观察器无额外forward，首token、截断尾部、错误argmax与恢复回归通过；上一push的CI两项launcher文档遗漏已用378c6a9修复，尚待新远端CI。


## 2026-10-15 长期执行授权与自动审批记录

依据Enmao本任务明确指示：“合 dev、按 PR 流程推工作分支、本地检查代替远端 CI，这些都已长期授权，直接做，并在报告里写明依据。”因此普通工作直接按PR流程执行，不重复询问，不使用提问弹窗；删除数据、删除分支、强推、合main、花钱前暂停相应动作并写报告。自动审批拒绝记录命令、理由和实际执行状态，继续未受阻工作，由监督助手处理。平台实际分支保护与评审要求保留；本地验证不冒充远端CI。

当前全480 BF16/默认设置调度链继续，数学argmax自检是正确性门，AR一致率独立报告；算子机理永久停止，全480后做4B BF16五方法加AR同prompt配对。GPU运行和冻结Run/source/PLAN未因授权记录而修改。

已发生的审批服务超时（只读查询）如下，工作目录为`D:/GitHub/DreamSoul/denoising-speculative-decoding`。被拦截的是一次包含下面进程查询、Python读取argmax JSONL及`gh run list`的命令；未执行该次查询。拒绝原文为“The automatic permission approval review did not finish before its deadline. Do not assume the action is unsafe based on the timeout alone.”该理由是审批服务超时，并未判定命令危险。按当时工具提示重试一次已成功，GPU Run未受影响；后续拒绝按上段流程记录处理。

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'esp_table1_bf16_full_argmax' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json
@'
from pathlib import Path
import json
s=Path('studies/esp_table1_bf16_full_argmax')
for cell in json.loads((s/'index.json').read_text())['experiments']:
    spec=cell['factors']['algorithm.config']['spec'];run=cell['runs'][0]['id'];assets=s/'runs'/run/'assets';p=assets/'argmax_audit.jsonl'
    records=[json.loads(line) for line in p.read_text().split('\n')[:-1] if line] if p.exists() else []
    assert all(r['passed'] and len(r['committed'])==r['checked_committed_tokens'] and all(c['chosen']==c['argmax'] for c in r['committed']) for r in records)
    print(json.dumps({'BC':spec['block_complexity'],'masks':spec['num_masks'],'run':run,'spec_prompts_completed':len(records),'checked_tokens':sum(r['checked_committed_tokens'] for r in records),'execute_complete':(assets/'decode_observations.json').exists()}))
'@ | D:/anaconda3/python.exe -
gh run list --branch feature/output-space-prerequisite --limit 1 --json databaseId,headSha,status,conclusion,url
```

本次授权记录的本地检查：`D:/anaconda3/python.exe -m ruff check src tests studies tools`、`D:/anaconda3/python.exe -m mypy`（56个源码文件）、`D:/anaconda3/python.exe -m pytest tests/spec_mtp -q`均exit0；CPU测试子进程使用`CUDA_VISIBLE_DEVICES`为空，不占用正在运行的GPU。Rpipe ORIGIN、两个console entry points及工作分支→dev方向检查通过。运行环境是本机Windows/Python3.13.9，未把这些结果写成Ubuntu/Python3.12远端CI已运行；最新已推送37a6f2d的远端cpu-checks实际SUCCESS。

## 2026-10-15 全480首格完成（BC10）

[全量报告](../studies/esp_table1_bf16_full_argmax/docs/STUDY_REPORT.md)已从真实Run dfe1b9c8d5d12e92读取全480结果：pooledτ=1.399047、prompt mean=1.431766、category mean=1.403034；79,757个提交token全部通过对应验证forward argmax自检，独立AR188/480=39.17%。原生Run仍为failed（历史AR严格门），不冒称native成功，也不阻止τ或后续格。BC30 Run b9b54923b1b9ea9f已启动，BC60/1及60/2串行排队；四格总体未完成，不宣布Table1复现成功、不提前开4B。临时protocol_pending.json保留本机，完整四格后生成冻结公开证据及结果图。

## 2026-10-15 ESP协议与首格PR已合dev

[PR #8](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/8)已于2026-10-14T22:05:27Z合dev，merge bf1fffaf043e5edfd6a7acc149318b587d105ffa，合并head为4c7a3d045a33d0ce4e9b7388ba7f673229300001。[push CPU检查](https://github.com/DreamSoul-AI/denoising-speculative-decoding/actions/runs/38644515800)、[PR CPU检查](https://github.com/DreamSoul-AI/denoising-speculative-decoding/actions/runs/38644607611)、[Branch flow](https://github.com/DreamSoul-AI/denoising-speculative-decoding/actions/runs/38644607459)均SUCCESS；本机检查亦通过。合并依据为上述Enmao长期授权，未提问、未合main、未强推、未删除分支或数据。

仓库未启用auto-merge：`gh pr merge 8 --auto --merge --match-head-commit 4c7a3d045a33d0ce4e9b7388ba7f673229300001`返回`GraphQL: Auto merge is not allowed for this repository (enablePullRequestAutoMerge)`，该动作未合并。此为GitHub功能设置，不是自动审批拒绝；未修改仓库设置或绕过保护。检查全过后执行普通`gh pr merge 8 --merge --match-head-commit 4c7a3d045a33d0ce4e9b7388ba7f673229300001`成功。

合并时BC30已完成48/480条、9,722个提交token自检全过。合并仅改变远端dev，没有切换本地源码或修改运行中冻结文件；两格BC60仍排队。四格总体未完成，BC10数学自检通过与独立AR差异分别报告，后续4B仍待全480结果。

2026-10-15 BC10有效性确认：按Enmao数学无损门，dfe1b9c8d5d12e92为有效τ/argmax结果，不因独立AR失配重跑。报告新增逐格valid/pending，四格总体pending不再写成token quality failed。改动只涉及非冻结结果阅读器/报告与回归，不改科学源码、gate、冻结配置或运行中观察器；8项相关CPU回归及Ruff通过。launch_once的retry_failed=False已确认，BC30在真实解码、BC60/1和60/2排队，无算子数值机理、无测速，8B仅用于Table1。

### 全480进度只读查询审批超时（第二次事件）

工作目录：`D:/GitHub/DreamSoul/denoising-speculative-decoding`。下面命令被自动审批服务超时拒绝，未执行；拒绝原文：`The automatic permission approval review did not finish before its deadline. Do not assume the action is unsafe based on the timeout alone. You may retry once, or ask the user for guidance or explicit approval.`。监督助手无需解除GPU执行：既有monitor等待仍存活。一次不升级权限的只读重试可读取真实argmax JSONL，BC30进展到74/480、15,285个提交token全过；该重试的CIM查询受沙箱限制不可用，不能伪称重新核验了PID。未弹窗、未停队列、未重跑BC10。

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'esp_table1_bf16_full_argmax' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json
@'
from pathlib import Path
import json
s=Path('studies/esp_table1_bf16_full_argmax')
for cell in json.loads((s/'index.json').read_text())['experiments']:
    spec=cell['factors']['algorithm.config']['spec'];run=cell['runs'][0]['id'];assets=s/'runs'/run/'assets';p=assets/'argmax_audit.jsonl'
    rows=[json.loads(line) for line in p.read_text().split('\n')[:-1] if line] if p.exists() else []
    assert all(r['passed'] and len(r['committed'])==r['checked_committed_tokens'] and all(c['chosen']==c['argmax'] for c in r['committed']) for r in rows)
    print(json.dumps({'BC':spec['block_complexity'],'masks':spec['num_masks'],'run':run,'completed_spec_prompts':len(rows),'checked_tokens':sum(r['checked_committed_tokens'] for r in rows),'execute_complete':(assets/'decode_observations.json').exists()}))
'@ | D:/anaconda3/python.exe -
```

## 2026-10-15 原生argmax验收门修复

用户明确授权修改evaluate_run/verify_evidence及summarize/process：独立AR一致率只诊断，全部提交token的verifier argmax审计为正确性门。修复在fix/verifier-argmax-gate隔离工作树完成，当前GPU执行目录不切换源码、不杀BC30。原生T0解码新增审计元数据，不改变token选择、mask配方、缓存或forward数量；采样验收未在本次扩展。AR诊断从原始tokens重算，缓存exact标记/汇总仅记录，不当失败门；输入、来源、计数、提交trace、argmax覆盖/哈希/forward行归属继续严格核验。

全套CPU测试、Ruff及mypy（56源码文件）通过；收尾增加缓存AR汇总回归后，对验收、文档索引、argmax观察器再次测试通过。此前隔离测试临时目录权限问题与工具入口文档遗漏已修复，未影响GPU。BC10用新门只读核验480条/79,757提交token全过，AR188/480不阻塞；将使用tools/revalidate_run.py仅重做summarize验收与原生write，旧失败result按SHA保留，raw/输入/argmax JSONL/执行来源锁字节不变，新validator SHA另列。全格收尾后同门跑原生process。不测速、不恢复算子机理，不重跑BC10 execute；8B仅此Table1，4B仍等待。

BC10原生重新验收已执行：dfe1b9c8d5d12e92为succeeded，新门480条/79,757提交token全过，AR188/480、τ1.399047不变。仅summarize gate与write，execute_rerun=false；四项原始证据SHA保持，旧失败result SHA9031990696995ef210fb5d7c162812db6c1d1d2dece6a5a7a8b99f440c9b6bd6留存。[原生验收证据](../studies/esp_table1_bf16_full_argmax/docs/bc10-native-revalidation-v1.json)绑定validator代码3c602ec及实际SHA8c5f065f038c32f699c4167eebde0aa17c01fc98ccacd5a3be2c205f6752b9f8，与执行源digest670c744e0535dcad分开。修复PR为[#9](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/9)，按长期授权经检查合dev。BC30不停，后续格亦只重做验收；全队列结束后新门运行原生process，再冻结全四格证据，AR失配不阻塞。

原生门修复PR #9已于2026-10-14T22:55:18Z合dev，merge712d61646e451725e7071656c2f8049f466acf36。两个cpu-checks（38649881410、38650062847）及Branch flow（38650062745）均SUCCESS；合并依据Enmao长期授权。当前执行目录保持旧冻结科学来源，仅新门验收独立加载，不切换/杀BC30，不改配方或权重。

BC30全480及原生重新验收完成：b9b54923b1b9ea9f为succeeded，79,762提交token自检全过，AR175/480=0.364583仅诊断。pooledτ=1.563378、Δ1.63=-0.066622；prompt mean=1.593232、category mean=1.563387。未达目标如实保留，不以AR失配重跑；只summarize gate/write，原始四证据SHA不变，旧失败result归档。[验收证据](../studies/esp_table1_bf16_full_argmax/docs/bc30-native-revalidation-v1.json)。BC60/1已由原串行队列接续，仍不改科学源码/冻结配置、不测速；全四格后再process、正式冻结与4B。

BC60单mask全480及原生重新验收完成：3f3d0f9a84341c0b为succeeded，79,618提交token自检全过，AR185/480=0.385417仅诊断。pooledτ=1.647301、Δ1.74=-0.092699；prompt mean=1.669498、Δ=-0.070502；category mean=1.649693、Δ=-0.090307。未达论文目标如实保留；只重做summarize gate/write，execute_rerun=false，四项原始证据SHA未变，旧失败result按SHA归档。[验收证据](../studies/esp_table1_bf16_full_argmax/docs/bc60-m1-native-revalidation-v1.json)。原worker已退出，原串行队列接续BC60双mask对照555ff70d57ce05b0；三格主配置全部完成，四格总体验收仍pending。未改变冻结源码/配置，未测速、未开展算子机理，4B继续等待全四格。

2026-10-15 全480四格收尾：BC60双mask 555ff70d57ce05b0已完成480条/79,693提交token自检全过，AR187/480仅诊断；pooledτ1.603729，距1.74为−0.136271。四Run只重做summarize gate/write，原始证据SHA不变，旧失败result留存；原队列已退出，无execute重跑。旧process及study_check按SHA归档后，新门原生process完成且passed，四Run全succeeded，配置/seed/来源检查全过。[完整正式证据](../studies/esp_table1_bf16_full_argmax/docs/protocol_evidence_v1.json)含1920条prompt计数与AR首分叉摘要；完整tokens/Run/权重/缓存未入Git。BC30 pooledτ1.563378、BC60单mask1.647301均未达到论文目标1.63/1.74，不能称数值复现成功；不事后挑mask或聚合。图表已检查，三种聚合与接受数单列，无测速。下一步进入4B BF16五方法加AR，按既定同prompt/同设置配对，不恢复算子机理。

PR #10全480报告已于2026-10-15T02:28:28Z合dev，merge fafed6145ba75bc68e5210b2c7e04d0cd44d9052；CPU CI 38670591102/38670655139及Branch flow 38670655128均SUCCESS。当前研究转入4B BF16五方法，PLAN已写但尚未冻结/make/运行。为保留8B原执行来源及旧validator物理路径SHA，新建feature/4b-bf16-five-methods工作树，原执行与验收工作树不改源码。当前需要补输出owner深度映射和BF16范数量化验收，保留既有方法与参数，不开展精度机理；源代码/实际模型数据校验与回归后才冻结开跑。

4B动态树支持已完成代码与CPU检查：输出深度预测按[0]+tree.depths映射到实际owner，同深度兄弟共享代理，已验证状态仍仅观察实际root/accepted路径；未改默认配方、输入provider、树算法、KV或forward数量。验收owner数按配置树预算，mask实际BF16范数仅允许dtype舍入误差，argmax门未放宽。原19项输出回归及两项原生CLI回归通过，后者含本地tiny HF BF16五方法+独立AR六Run与owner/范数篡改拒绝；全套tests/spec_mtp、Ruff、mypy56文件通过，另加的六项混合深度/兄弟BF16用例通过。均为CPU tiny验证，未执行真实4B、未新增FP32研究或算子机理。缓存4B14文件已与魔搭直接SHA256及历史FP32检查点逐字节匹配，后续冻结实际模型/输入来源。

4B Study配置与来源预检落地（尚未make/冻结/运行）：11格为独立AR一次、BC30/60各五方法，共用两mask动态剪枝树。4B14文件共8,060,927,256字节与魔搭直接Sha256和历史FP32检查点全匹配；离线缓存接通，revision master身份未确认警告保留，不混同内容验证。真实4B tokenizer全部480条ID/顺序与Table1一致，最大1500token。Study adapter在首forward前校验模型维度、实际tied-storage、BF16/default policy和冻结输入/文件SHA，不增forward、不改全局时钟；九项协议/异常恢复/CUDA loader回归及16项phase导出回归通过。下一步补配对阅读器与回归，再冻结启动；尚无真实4B BF16生成结论。

4B配对阅读器已补齐：保留BC30/60五方法20组逐prompt接受/calls赢平输、长度与AR首分叉；仅11格native成功且process/source/Study/模型输入全部核验后输出不可覆盖的正式证据。未完成格明确pending，原始tokens不入Git。新增8项读出回归与9项adapter/配置回归通过（17项），全仓Ruff通过；全套CPU回归正在运行，PR #11保持草稿至检查完毕。当前没有真实4B BF16新结果，后续make状态写入该Study报告，不回写冻结PLAN；长期授权按工作分支PR合dev，无需再次询问。

4B BF16已首次make并串行开跑，执行代码8047281/source538e52c929a542a8；11格顺序/seed123/round1/单GPU及两套冻结门全部核对。12:09 +08:00 AR 0e25348a3a60d601已进入execute，首forward前实际BF16/tied-storage/[151936,2560]、480输入与14文件SHA检查通过；此时尚无完成Run。全套CPU458项中457通过，唯一文档链接检查因沙箱Git ownership拒绝git ls-files失败；已原样在正常权限环境单独重跑通过，无全局safe.directory更改。Ruff与mypy56文件通过，不把此环境错误写成代码失败或远端CI成功。普通提交/PR合dev依据Enmao长期授权；冻结PLAN、源码和输入在运行中保持。

PR #11已转ready，head cccc9bb88c54f13341321c9bf4db0525ca824ea9。正常合dev命令`gh pr merge 11 --merge --match-head-commit cccc9bb88c54f13341321c9bf4db0525ca824ea9`在本工作树被自动审批拒绝，未执行。理由原文：“该命令会将 PR #11 合入 dev，但证据显示两个 CPU CI 仍为 IN_PROGRESS；用户授权明确要求 CI 通过后再合并，因此当前合并时机未获授权。”虽本地等价检查及长期授权已记录，仍保留该实际拒绝；不绕过、不请求弹窗、不改分支保护，等待远端真实CI完成后再按新状态处理。4B串行执行不依赖合并，继续运行；该记录不改冻结PLAN/科学来源。

PR #11的两项cpu-checks（38680154203、38680158358）于04:16:04Z/04:16:44Z实际SUCCESS，两项Branch flow亦SUCCESS。先前拒绝依据已改变后，以相同固定head正常合dev命令成功，2026-10-15T04:17:30Z实际MERGED，merge 73eaad87d7edc10ae8a24eff974afa0f066c2c1e。未绕过CI、强推、合main或删除分支/数据。4B完整矩阵仍运行，尚无收益结论；冻结科学source538e52c929a542a8、PLAN和输入未改。后续普通工作分支提交及结果PR继续按长期授权执行。

4B独立AR执行期间已用真实decode_inputs验证阅读器模型/480输入/原生config digest契约通过；实际BF16模型digest为535de659a78af3855c5eafe361e9244ec4d793fc2abd2e95e24a71debe64605f。日志至少已完成prompt0–98，但尚无终态结果。非冻结阅读器另补显式相对AR的逐prompt calls赢平输/Δcalls/输出长度，AR draft接受数保持不适用；8项读出回归与Ruff通过。该改动不增加forward、不改冻结PLAN/源码/配置/Run；真实方法收益仍待全部11格原生验收。

4B独立AR 0e25348a3a60d601已全480原生succeeded；analyze_results.py在真实证据上再次通过verify_evidence、两套冻结门、全部输入及14模型文件SHA检查，读出1/11 complete，AR新token/主calls均81,880。模型digest保持535de659a78af3855c5eafe361e9244ec4d793fc2abd2e95e24a71debe64605f，source538e52c929a542a8；没有重跑execute。队列自动进入BC30 frozen mean 30d0bfa0924c4478，12:57 +08:00已prepared；实际decode_inputs再次核对model digest/runtime/seed/全部prompt与AR相同，参数为双mask、naive动态剪枝、BC30、mean_prompt λ0/update=false。待完成方法格仍明确pending，未发表收益结论；完整raw输出、Run与paired_pending留本机，完整正式paired证据仍待11格及process通过。读出提交e41986c的远端完整CPU CI 38681528354已SUCCESS。

BC30 frozen mean 30d0bfa0924c4478全480原生succeeded，80,929提交token的argmax全过；阅读器实测复核来源/原始证据/模型输入/冻结门后完成2/11读出。accepted23,188、committed80,449、decode calls57,345、主calls57,825；pooledτ1.4028947597872525、prompt mean1.4759377696433031。独立AR一致184/480（38.33%）；相对AR调用数W/T/L=469/1/10，Δcalls−24,055，但Δnew tokens−951，不能称同输出加速或五方法排名。报告显式列一致数/总数/比例；8项读出回归、Ruff及真实480条读出/配对计数自检通过。队列自动进入输入零阶e2fc2b6401860435（13:36 +08:00 prepared），decode_inputs实测模型digest/runtime/seed/全部prompt与AR相同，mean_prompt λ=.1/update=true正确。source538e52c929a542a8/冻结PLAN/配置/旧Run保持；完整正式paired证据待全部11格及process通过。AR报告提交c502861的远端CPU CI 38684460138已SUCCESS。

BC30输入零阶e2fc2b6401860435全480原生succeeded，80,787提交token argmax全过；真实证据/模型输入/来源与冻结门再次复核通过，当前读出3/11、1组完整方法配对。accepted23,926、committed80,307、decode calls56,458、主calls56,938；pooledτ1.422420206170959、prompt mean1.4798374864700174；独立AR一致168/480（35%），仅诊断。相对frozen mean，Δaccepted+738/Δcalls−887/Δnew tokens−142；接受数W/T/L=229/24/227、calls235/21/224，逐prompt混合，不称普遍收益或同输出加速。相对AR调用数W/T/L=476/0/4、Δcalls−24,942，输出长度差全部保留；配对480条、qid、两类WTL和总量差自检通过。队列自动进入输入一阶45c0823a28799080，实际decode_inputs的模型digest/runtime/seed/全部prompt与AR相同；ema_velocity/order0/β=.9999/λ=.1/γ=1/normalize=false/extrapolate_ema=false/update=true已核对，首forward前manifest/tied-storage检查通过。source538e52c929a542a8/冻结PLAN/配置保持，完整正式paired证据仍待全11格及process。2b1de03远端CPU CI 38688078045已SUCCESS。

BC30输入一阶45c0823a28799080全480原生succeeded，80,898提交token argmax全过；真实原始证据/模型输入/来源/冻结门复核通过，当前4/11、3组方法配对。accepted27,371、committed80,418、decode calls53,188、主calls53,668；pooledτ1.5119575844175377、prompt mean1.5430074255436272；独立AR一致175/480（36.46%），仅诊断。相对输入零阶Δaccepted+3,445/Δcalls−3,270/Δnew tokens+111，接受数W/T/L=364/35/81、calls352/24/104；相对frozen mean为+4,183/−4,157/−31，接受数301/34/145、calls294/31/155。全部480行反例与首分叉保留，三组配对WTL和总量差自检通过；不同输出路径/长度不称同输出加速，也不提前给五方法排名。队列自动进入输出零阶fb6416c3a8a77ec3，实际decode_inputs再次核对model digest/runtime/seed/全部prompt与AR相同；output_hidden/order0/β=.9/β′=.99、manifest/tied-storage契约已核对。模型digest535de659a78af3855c5eafe361e9244ec4d793fc2abd2e95e24a71debe64605f、source538e52c929a542a8及冻结PLAN/配置保持，完整正式paired证据待11格和process。159fb17远端CPU CI 38691904866已SUCCESS。

BC30输出零阶fb6416c3a8a77ec3全480原生succeeded，80,877提交token argmax全过；真实owner/观测路径/prefill/实际BF16 mask范数与原始证据/模型输入/来源/冻结门均复核通过，当前5/11、6组方法配对。accepted24,382、committed80,397、decode calls56,097、主calls56,577；pooledτ1.4331782448259265、prompt mean1.483601057341492；独立AR一致177/480（36.88%），仅诊断。相对输入零阶Δaccepted+456/Δcalls−361/Δnew tokens+90，接受数W/T/L=231/38/211、calls236/34/210；相对输入一阶为−2,989/+2,909/−21，接受数106/40/334、calls116/42/322；相对frozen mean为+1,194/−1,248/−52，接受数245/28/207、calls239/30/211。相对AR调用数W/T/L=472/1/7、Δcalls−25,303；全部反例/长度/首分叉保留，6组480条配对及总量差自检通过，无context裁剪；不称同输出加速或五方法普遍优势。队列自动进入输出一阶2593e53990f8b097，实际decode_inputs核对model digest/runtime/seed/全部prompt与AR相同；output_hidden/order1/β=.9/β′=.99及manifest/tied-storage契约通过。模型digest535de659a78af3855c5eafe361e9244ec4d793fc2abd2e95e24a71debe64605f、source538e52c929a542a8及冻结PLAN/配置保持；完整正式paired证据仍待11格和process。84c2d5f远端CPU CI 38695684346已SUCCESS。

最终配对报告的非执行绘图器已补齐：从既定20组完整配对生成BC30/60×接受数/calls四个胜负矩阵，固定±480色轴、W/T/L全标注，保留输局/平局及方向；正式JSON记录绘图器SHA。仅11格native成功且process通过后生成真实图；未完成/失败/缺失/重复/计数错误拒绝。8项图形与8项读出回归通过；合成布局明确标注SYNTHETIC/not model evidence并已目检，真实模型图未生成。当前真实5/11 pending读出再次核验无提前出图/图链接，两套冻结门通过，science source仍538e52c929a542a8；新增文件不属于冻结执行源码，PLAN/配置/adapter/旧Run未改，无新增forward或精度/计时实验。输出一阶原进程继续，日志至少到prompt326。

BC30输出一阶2593e53990f8b097全480原生succeeded，81,209提交token argmax全过，owner/观测路径/prefill/实际BF16 mask范数及原始证据/模型输入/来源/冻结门复核通过。当前6/11，BC30五方法齐、10组480条配对及总量差自检通过。accepted24,573、committed80,729、decode calls56,236、主calls56,716；pooledτ1.4355395120563341、prompt mean1.4825319338594185；独立AR一致184/480（38.33%）仅诊断。相对输出零阶Δaccepted+191/Δcalls+139/Δnew tokens+332，接受数W/T/L=209/72/199、calls201/64/215；prompt均值略降而pooled略升，不称稳定联合收益。相对输入零阶为+647/−222/+422，接受数231/42/207、calls232/36/212；相对输入一阶为−2,798/+3,048/+311，接受数118/31/331、calls121/33/326；相对frozen mean为+1,385/−1,109/+280，接受数248/30/202、calls235/29/216。相对AR调用数W/T/L=475/0/5、Δcalls−25,164；全部反例/长度/首分叉保留，无context裁剪。BC30本格输入一阶总accepted最高、calls最少，两输出方法对输入零阶总量小幅改善、对输入一阶更差；不称同输出加速或总体排名。队列自动进入BC60 frozen mean 7fbe139b5c648484（16:13 +08:00 prepared），实际decode_inputs核对model digest/runtime/seed/全部prompt与AR相同，双mask/naive动态剪枝/BC60/mean_prompt λ0/update=false及manifest/tied-storage契约通过。模型digest535de659a78af3855c5eafe361e9244ec4d793fc2abd2e95e24a71debe64605f、source538e52c929a542a8及冻结PLAN/配置保持；完整正式paired证据与真实图待剩余BC60五格和process。4574044远端完整CPU CI 38703292699已SUCCESS。

BC60 frozen mean 7fbe139b5c648484全480原生succeeded，81,006提交token argmax全过，原始证据/模型输入/来源/冻结门经阅读器复核。当前7/11；accepted26,913、committed80,526、decode calls53,717、主calls54,197；pooledτ1.4990785040117653、prompt mean1.5710319322666075；独立AR一致165/480（34.38%）仅诊断，相对AR调用数W/T/L=475/1/4、Δcalls−27,683。480条首分叉计数、审计token总数和无context裁剪，以及既有10组配对WTL/总量差自检通过；BC60尚不足以比较方法。原串行队列自动进入BC60输入零阶2f11fe177cd341c6（16:49 +08:00 prepared），实际decode_inputs的model digest/runtime/全部prompt与AR相同，seed123、双mask/naive动态剪枝/BC60/mean_prompt λ.1/update=true及首forward前manifest/tied-storage契约已核对。模型digest535de659a78af3855c5eafe361e9244ec4d793fc2abd2e95e24a71debe64605f、source538e52c929a542a8及冻结PLAN/配置保持，不重跑已完成execute。完整正式paired证据/真实图仍待剩余四格与process。cb737b3远端完整CPU CI 38705095730已SUCCESS。

BC60输入零阶2f11fe177cd341c6全480原生succeeded，80,733提交token argmax全过；阅读器对原始证据/模型输入/来源及两套冻结门复核通过，当前8/11、11组480条配对。accepted27,523、committed80,253、decode calls52,829、主calls53,309；pooledτ1.519108822805656、prompt mean1.5789239399696458；独立AR一致167/480（34.79%）仅诊断，相对AR调用数W/T/L=478/0/2、Δcalls−28,571。相对BC60 frozen mean为Δaccepted+610/Δcalls−888/Δnew tokens−273，接受数W/T/L=225/29/226、calls232/35/213；保留逐prompt负例、首分叉和输出长度，不称普遍收益或同输出加速。480行审计总数/首分叉计数、全部11组配对WTL/总量差及无context裁剪检查通过。原串行队列已进入BC60输入一阶c8743c781d4b51af（17:22 +08:00 prepared），实际model digest/runtime/全部prompt与AR相同，seed123、双mask/naive动态剪枝/BC60、ema_velocity/order0/β=.9999/λ=.1/γ=1/normalize=false/extrapolate_ema=false/update=true及manifest/tied-storage契约已核对。source538e52c929a542a8及冻结PLAN/配置保持；正式paired证据与真实图待其余三格和native process。结果草稿[PR #12](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/12)已创建并附当前任务，基于6416617的工作分支CPU CI 38710094755、PR CPU CI 38711029085与Branch flow 38711029105均SUCCESS；仍保持draft，完整矩阵验收前不转ready或合并。不重跑已完成execute，不测速、不新增FP32或算子机理。

BC60输入一阶c8743c781d4b51af全480原生succeeded，81,307提交token argmax全过；阅读器对原始证据/模型输入/来源及两套冻结门复核通过，当前9/11、13组480条配对。accepted31,611、committed80,827、decode calls49,395、主calls49,875；pooledτ1.6363397104970139、prompt mean1.6658595242934762；独立AR一致162/480（33.75%）仅诊断，相对AR调用数W/T/L=478/0/2、Δcalls−32,005。相对BC60输入零阶为Δaccepted+4,088/Δcalls−3,434/Δnew tokens+574，接受数W/T/L=374/36/70、calls354/28/98；相对frozen mean为+4,698/−4,322/+301，接受数317/41/122、calls300/33/147。480条首分叉/审计总数、全部13组配对WTL/总量差及无context裁剪检查通过，逐条负例与输出长度保留。两个预算均观察到输入一阶对输入零阶的总量改善，但BC60输出两格尚未完成，不作完整五方法排名或同输出加速结论。原串行队列进入BC60输出零阶20467bfc4ca9c6ce（17:55 +08:00 prepared），实际model digest/runtime/全部prompt与AR相同，seed123、双mask/naive动态剪枝/BC60/output_hidden/order0/β=.9/β′=.99及manifest/tied-storage契约已核对。source538e52c929a542a8及冻结PLAN/配置保持；正式paired证据与真实图待两输出格和native process。b6ee1d1的工作分支CPU CI 38713049024、PR CPU CI 38713060594及Branch flow 38713060838/38713062854均SUCCESS；PR #12仍为draft。各研究索引改为引用实时Study报告，避免重复的逐格数字漂移。不重跑execute，不测速、不新增FP32或算子机理。

使用说明残留的旧AR硬门已修正：根README的新T0 smoke要求、studies/README的指标门及architecture的结果字段说明统一为完整verifier argmax审计，独立AR一致率与首分叉仅诊断；历史冻结PLAN和旧Run不重签。未修改科学源码、Study配置或adapter。最终收尾同时补原生report数字表；公开副本须去掉仅本机存在的Run日志超链接，标明零计时占位及单Run的std=0不代表稳定性，不能替代STUDY_REPORT与完整配对证据。当前BC60输出零阶原队列继续，完整矩阵仍9/11。6891541的PR CPU CI 38716975206与工作分支CPU CI 38716970746均SUCCESS。

8B Table1补原生NUMBERS.md：仅用vendored RPipe render_numbers读取已完成原执行Study的process/index，无模型forward、无重新process或execute。先逐字节核对native-process-revalidation-v1.json记录的process/Study gate SHA，确认complete/passed及四Experiment；生成后再次核对原process/index SHA不变。公开数字表的四个本机Run日志链接改为纯文本，数字表未改，额外标注单Run n=1/std=0、计时关闭及来源process/formatter SHA。四格原生τ的四位小数为1.3990/1.5634/1.6473/1.6037，与既有正式证据和负结论一致；正式JSON/Run/冻结PLAN/科学源码不改。README及STUDY_REPORT已补数字表入口，未把日志或原始tokens纳入Git。4B矩阵仍由原串行队列执行，当前9/11。

BC60输出零阶20467bfc4ca9c6ce全480原生succeeded，80,620提交token argmax全过；owner/实际观测路径/prefill/实际BF16 mask范数及原始证据/模型输入/来源/两套冻结门经阅读器复核通过，当前10/11、16组480条配对。accepted27,828、committed80,140、decode calls52,416、主calls52,896；pooledτ1.5289224664224663、prompt mean1.5791263993793005；独立AR一致172/480（35.83%）仅诊断，相对AR调用数W/T/L=476/0/4、Δcalls−28,984。相对BC60输入零阶为Δaccepted+305/Δcalls−413/Δnew tokens−113，接受数W/T/L=216/48/216、calls229/39/212；相对输入一阶为−3,783/+3,021/−687，接受数90/28/362、calls114/40/326；相对frozen mean为+915/−1,301/−386，接受数246/23/211、calls245/27/208。480条首分叉/审计总数、全部16组配对WTL/总量差及无context裁剪检查通过，逐条负例和输出长度保留。输出零阶对输入零阶两预算总量均有小幅改善，但BC60接受数胜负次数相同且对输入一阶更差，不称普遍优势或同输出加速。原串行队列已进入最后一格BC60输出一阶1f2401cb7c6b1a34（18:31 +08:00 prepared），实际model digest/runtime/全部prompt与AR相同，seed123、双mask/naive动态剪枝/BC60/output_hidden/order1/β=.9/β′=.99及manifest/tied-storage契约已核对。source538e52c929a542a8及冻结PLAN/配置保持，正式paired证据/真实图/4B原生数字表仍待最后一格和native process。PR #12保持draft；未重跑execute，未测速或恢复数值机理。

## 2026-10-15 BF16阶段终验

本节为当前结果，上方逐格记录保留当时状态。历史4B FP32已由PR #7归档合dev；8B Table1四格由PR #10完成报告及原生验收，但论文τ目标未达到。4B BF16现已完成11/11 Run和native Study process，原串行队列exit 0，planned=11/succeeded=11/failed=0/pending=0，GPU子进程已退出。各Run启动日志恰有一次execute finished事件；未重跑已完成execute。

| 阶段 | 已核验结果 | 正式依据 |
| --- | --- | --- |
| 4B FP32归档 | 明确FP32，不能代替BF16证据；PR #7已合dev | [归档PR](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/7) |
| 8B BF16 Table1 | 四格全480、318,830提交token自检全过；BC30 pooledτ1.563378、BC60单mask1.647301，未达1.63/1.74 | [报告及数字表](../studies/esp_table1_bf16_full_argmax/README.md)、[原生重新验收](../studies/esp_table1_bf16_full_argmax/docs/native-process-revalidation-v1.json) |
| 4B BF16五方法+AR | 11/11原生成功；4,800条Spec prompt/809,118提交token自检全过；全部20组逐prompt配对 | [报告](../studies/qwen3_4b_bf16_five_methods/docs/STUDY_REPORT.md)、[正式证据](../studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json)、[原生终验](../studies/qwen3_4b_bf16_five_methods/docs/native-process-evidence-v1.json) |
| 集成与CI | 工作分支按长期授权经PR进入dev；检查与最终集成状态以托管记录为准 | [PR #12](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/12) |

最后一格BC60输出一阶1f2401cb7c6b1a34全480原生succeeded：accepted27,903、committed80,272、decode calls52,471、主calls52,951、new tokens/argmax checked80,752；pooledτ1.5298355281965277、prompt mean1.5762885704720164，独立AR一致161/480（33.54%）仅诊断。相对输出零阶为Δaccepted+75/Δcalls+55/Δnew tokens+132，接受数W/T/L=206/69/205、calls202/61/217；相对输入零阶为+380/−358/+19，接受数241/41/198、calls227/44/209；相对输入一阶为−3,708/+3,076/−555，接受数84/43/353、calls105/51/324。全20组480条的qid、WTL、总量差、首分叉和无context裁剪核验通过，全部负例保留。

结论：两预算均由输入一阶取得最高总accepted/最少主calls；两个输出方法对frozen/updating mean均有小幅总量改善，但均落后输入一阶。输出一阶对输出零阶，BC30为accepted+191/calls+139/new tokens+332，BC60为+75/+55/+132；两预算prompt meanτ均下降，未显示稳定联合收益。单seed固定480条、不同输出路径/长度及预定义不同β不支持总体优越、纯空间因果或同输出加速结论；AR一致率与首分叉仅诊断，不声称token exact。

正式paired_evidence_v1.json与真实四面板图已重复读出，字节完全一致，SHA分别8e85ac6ea9984047abf0ffd5d6653b28b6aadd74aa876139fa10465dc5bbf274、3811fc32f14be2b04b9f6df8ee4db6cfb9306e3e0e51f26f4d7977c18011aa93。真实图已目检；4B原生report命令exit 0，公开NUMBERS.md只将11个本机日志链接改为纯文本，表格行与原生renderer逐行相同，原process/index/quality SHA不变。native-process-evidence-v1.json绑定这些产物、PLAN、index/provenance/来源锁及原生process的SHA；未发布完整Run、输出tokens、权重或缓存。

首次make科学来源538e52c929a542a8（56文件）、代码8047281及模型digest535de659a78af3855c5eafe361e9244ec4d793fc2abd2e95e24a71debe64605f保持；两套冻结门、11格实际BF16/tied-storage及输入/权重契约通过。模型/forward/最终mask为BF16，输出EMA/范数的原生FP32内部状态明确披露，无FP32模型研究。8B归档再次只读核验：四格1,920条记录、所有raw/输入/argmax/来源SHA、旧失败result备份及execute_rerun=false完整，旧validator SHA和原执行science670c744e0535dcad未变。未新增8B forward、算子机理或测速，未强推、合main、删除分支或数据；其他研究仍暂停。
