# 开发记录

## 2026-10-20 S1 深 JSON 读取残留

两个公共模块入口在合法但过深的 JSON 上，结果读取、缓存计划和 Tracker 历史路径出现 **14 failed / 1 passed**，58.825 秒；activity 另 **2 failed**，8.233 秒。严格 decoder 对照确实抛 RecursionError。仅在原来已有容错处理的八处 catch 中补这一异常，不改变严格 decoder、异常继承或 I/O 范围。只读结果/state/JSONL/activity 原字节保全，坏 JSONL 邻接有效观测的坐标与数值保持；坏 jobs 缓存沿原 make 路径重建。

同一组及正常非对象/非法 JSON 对照 **27 passed**，147.649 秒（所属树守卫 148.8815 秒）；固定来源及文档门另 **10 passed**，3.684 秒。所有轮次输入 SHA 前后相同、所属进程清理确认。首次 2000 层夹具在本机 Python 3.13 仍可解析，首轮阈值断言无效，原件留在 `.tmp/debt-batch4/depth-red/`，不计偿还。有效反例改为 10000 层，没有通过降低递归上限制造产品故障。

原 v0.3.1 的 102 条来源记录、固定指纹和许可证保持，当前 103 文件；仅新增 activity 精确差异及原 Git blob，现为十一项差异、十份原 blob。独立调用链复核、Python 3.10 语法、Ruff 与 offline pin 通过；实际 Windows/Python 3.13.9。逐轮命令、JUnit、原始日志及哈希在 `.tmp/debt-batch5/depth-source-verification.json`；未运行 CUDA、下载、训练或收费资源。

## 2026-10-20 S1 make 发布持锁接续

三条真实入口（配置 API、高层展开、原生 CLI make）在后目标已持锁时仍写配置，独立有效红例 **3 failed / 3 passed**。按旧布局在 make/__init__.py 与 write.py 复用既有 Run 锁；配置候选和内部 writer 由原代码拆出，高层持锁覆盖 recipe/config/index/provenance，不再次调用会重入取锁的公共入口。原 ensure→recipe 顺序保持，不迁 R2，不引入多 Run 事务。

相关 make/package/recipe/docs 首轮 **73 passed / 1 failed**，唯一失败是测试把原有 activity.json 进度输出也误纳入业务字节保全；未改生产处理，收窄到原合同并保留原件，最终新 make 全部 **10 passed**，整轮守卫 **4.7289 秒**、所属清理确认。最初 base description 被 Study run_description 覆盖的无效控制也单列保留，不计源码反例。精确轮次、JUnit 与源码/来源 SHA 见 `.tmp/debt-batch3/make-source-verification.json`。

固定原 102 记录、当前 103 文件、许可证不变；白名单增加 make 两处，当前十项差异、九份原 blob。锁准入允许 Run 目录/锁文件与原进度输出，不承诺未展开历史 Run 或原子回滚。专项来源锁沿既有独立合同；无研究模型、训练、GPU、下载或计费。

## 2026-10-20 S1 第三批迁移残留

固定副本的 scheduler、status、Study aggregate 漏接非对象结果异常，两个真实模块入口的 status/process/launch 以及 scheduler 首红为 **10 failed / 1 passed，26.448 秒**。持久化计划过滤前不验完整 jobs/batches、损坏数字裸抛及负 round 另为 **7 failed / 1 passed，1.243 秒**。只补四个已有容错 reader 的 CorruptArtifactError 和原始计划前置验证；decoder、异常继承、既有 OSError 边界及科学源码不改。

首组修复后相关入口/pin/docs **65 passed，128.457 秒**，整轮守卫 **129.4786 秒**。独立复核再找到公开 `run-one --phases process` 的同因遗漏，两个入口先 **2 failed / 2 passed，16.290 秒**，补齐 individual reader 后其公共入口及 pin/docs **21 passed，20.234 秒**，守卫 **21.3929 秒**。这两组有重叠，65 项发生在 individual 补丁之前，不充作最终全量验收；全部所属进程清理确认。最初 custom pytest 参数发现失败发生在收集前，原件单列，不计源码反例。

固定 v0.3.1 身份、102 条原来源与许可证保持；当前精确八项本地补丁、七份原 blob 均由 offline checker 验证，伪造原清单或任意新增本地路径仍拒绝。Python 3.10 语法及 Ruff 通过，实际本机为 Windows/Python 3.13.9。完整命令、日志、JUnit、阶段 report 与逐文件来源核验保留 `.tmp/debt-batch3/`；不随 clone 提供。未启动 GPU、下载、真实模型研究、旧队列、付费资源或数据清理；远端新 HEAD 检查另行核对，不借用旧 CI。

## 2026-10-14

- 继续 refactor/spec-rpipe，保留用户及上游工作区修改。前次 PR #5 已进入 dev（1b6d59c），当前分支快进到同树集成提交后继续；不推送 dev/main，不发布版本。
- 先写 rpipe-migration 职责映射，再迁移专项阶段、API 桥和证据核验。Study 阶段仅导入 src；默认 prepare/execute 不复制空壳，旧 CLI/recipe/checks/prompts 只保留兼容导出。业务从 data/__init__ 移出；科学解码核心保持。ModelScope 的现有 local_files_only 传至 snapshot_download，并有边界回归。
- 从宏观方案提取 input_mask_baseline 的四格真实模型前置门，先写 PLAN，核对研究六方面与固定 knowledge skills。只完成配置/来源和离线缓存解析，未运行真实模型；研究范围见该 Study。工程验证、失败和 tiny 轨迹见 [smoke 本轮报告](../studies/smoke/docs/STUDY_REPORT.md#八职责重构检查点2026-10-14)，不把旧正式接受数改成新结果。
- 用户要求改本地目录：Windows Move-Item 返回错误并部分移动。逐文件核对目的路径与碰撞后补齐 28875 个文件，无覆盖；Git HEAD/索引及历史证据保留。实际 checkout 为 denoising-speculative-decoding，editable 无依赖重装并核对 import / direct_url。旧路径只剩临时目录残留，未强删或修改系统设置。历史证据中的执行路径原样保留。
- 改名与交接文档后再次全量 239 passed / 0 skipped，Ruff、55 文件 mypy、模块/console CLI 和 Git 连通性门通过；没有重复历史正式模型格子。
- 用户要求本机收尾、push 并异机交接。已查 RPipe v0.3.1 的 annotated tag cd7a750de0d13889916eba84b53ed4a339d47a32，实际 commit 为 7a15c46b75c5eb951cccd60ed7b9fc3463db90f5。两项 src 变化为版本号和 readout/report.py 的 B-030 身份与原子写入修复；实际 research/feedback/README/flow 文档已取得。当前执行副本仍是 v0.3.0，升级及异机验证按 [交接记录](handover.md) 接续，不提前报告通过。

## 2026-10-13

- 按用户要求继续 refactor/spec-rpipe；fetch 后工作树与 origin/dev 的科学内容一致，保留当前分支，未修改 RPipe 上游。依据固定 v0.3.0 Study 合同重写指南、模板和 smoke 计划，统一 README / PLAN / STUDY_REPORT 命名并更新导航。历史计划保留，正式 JSON、smoke 配方与 accepted 数字不变。
- 模板修正“注册即下载”的说明和未展开的 {mask} description，改用上游支持的 experiment/seed 占位符，扩展现有配置展开回归；数据、模型、算法、seed 和预算不变。报告模板区分 all-depth accepted 与深度诊断，并记录阶段/来源、排班、失败/恢复边界。
- Study 报告补充由旧交付 JSON 生成的真实 tiny 结果图与同 Study 手动绘图入口，不执行解码或重新签收旧 Run。首轮 NumPy-first 绘图在本机触发已知 C-003 DLL 导入故障；torch-first 重放通过，脚本沿用本仓库局部导入顺序，未修改环境/系统。图源、指标和研究边界见 [smoke 报告](../studies/smoke/docs/STUDY_REPORT.md)，临时故障脚本和验证日志留 .tmp/studies-rewrite-20261013。

- 本次 Study 重写验证：236 个现有回归全部通过（独立 --basetemp），覆盖模板/三 variant 配置展开、隔离 CPU tiny 实际 CLI 和失败门；Ruff 0.17.0 与 mypy 2.4.0（40 源文件）通过。另核对 Git 索引大小写敏感链接、完整 YAML 语义和三个历史 JSON 保留；图脚本实际执行并检查输出。未重复数值矩阵、GPU/安装包/恢复/并发验收，科学源码及包配置未改。

- 仓库改名为 DreamSoul-AI/denoising-speculative-decoding，默认分支从 dev 调整为 main；同步当前 checkout 的 origin。工作分支 → PR → dev → 同仓库 dev 的 PR → main，补充只读 metadata 的 Branch flow 门，不合并当前 PR 或发布版本。保留原审批、CPU CI、禁止直接推送/强推/删除规则。

- refactor/spec-rpipe 初始 HEAD c3263c0，与远端一致；保留并纳入既有 8 文件 prepare_shared 修改。远端 fetch 原指向已删除分支，显式获取实际 refs 完成同步，没有切分支。
- 固定上游 804b3823ab6d2af46cbda1a65688ca8b76c60b22；101 个文件经 Git 对象核对，仅 Windows torch-first __init__.py 不同，新增 ORIGIN 和匹配的 MIT 许可证。
- 复用现有 RPipe 执行库，专项 collect/summarize/write/process 经 Study 扩展接入；新增原始解码和输入证据、源码及输入漂移门。旧兼容 evaluate/test 调用保留，采样与 mask 更新不迁移到框架。
- 安装与 CI 工作流保留。兼容差异、首轮失败与处理见 [问题记录](bugs.md)；数值、Run ID、环境和复跑代码见 [smoke 报告](../studies/smoke/docs/STUDY_REPORT.md)。
- 首轮全测试 222 passed；补齐 Study 来源、AR/alignment 证据和 seed 轴合同后 224 passed。全流程最终来源验收以报告所列新 version 为准。

- 实现提交 7c89d7c 后，全测试 227 passed；新 CPU/CUDA tiny Run 和 wheel/sdist 各自完成实际阶段链，正式证据见 smoke 报告。CUDA 名称误报 B-003 已修复，final02 失败保留。未改 Actions 配置。
- 最终实现提交 934dac3：CLI 前置来源拒绝保全旧 result，Study 终验同步通用 envelope，report 先复验当前 index。最终 229 passed、Ruff/mypy 通过；该提交新 CPU/CUDA tiny 与 wheel/sdist 各自通过。C-004 的首次旧 Run 终态未备份无法恢复，未编造恢复结果。最终证据见 smoke delivery_validation.json。
- PR #4 面向 dev；首次 push CI 38367588147 发现本项目既有 tomllib.loads(bytes) 安装脚本错误，未进入 lint/tests。仅改 UTF-8 文本读取并执行同一依赖提取片段验证，不覆盖本机 GPU 环境、不关闭 Actions，后续远端门单独核对。
- 60fd569 的远端安装/lint 通过，mypy 38367956325 复现本项目 datasets/ namespace 遮蔽。设置 namespace_packages=false（所有 src 包已有 __init__.py），不屏蔽 attr-defined 或模块检查；在仓库根目录固定 mypy 复验全部 40 文件通过。科学源码与 Run 配方仍与 934dac3 一致。
- 4f104bb 的远端门已通过安装/lint/mypy/入口，运行测试 228 passed，仅正式 NUMBERS 的本机 raw log 链接失败（B-006）。保留数字、改为明确本机文本路径，正式链接只依赖入库报告/JSON，未改测试门限或科学实现。

## 2026-10-14 v0.3.1 当前设备接续

从49dec9c接续，交接提交cpu-checks已通过。先审阅固定新版本文档/接口并改设计，再备份/比较101文件、升级完整vendor到7a15c46；唯一源码差异Windows torch-first和MIT保留，不改上游仓库。发现并修复下游report拒绝raw缺失后旧complete仍true的问题，专项门与上游身份/原子门共存。固定36格严格相等，新设备251回归、Ruff/mypy、Windows与wheel/sdist全链通过；失败日志和边界见smoke报告/JSON。

C-004全新Toy/Stub最小复现、公开Issue/PR去重及脱敏完成，实际提交 [RPipe#36](https://github.com/DreamSoul-AI/RPipe/issues/36)。真实四格只make新version，缓存缺失有ModelScope E1022证据，未launch/未校准/未声明研究质量通过。阶段commit/push与面向dev草稿PR按当次授权进行，不自动合并、发布或改变远端规则。

本轮实现本地提交 cb5c36d；自动审批拒绝组合 push/草稿PR，理由为当前可信消息未明确确认外传目标及公开资料风险。已只读核实目标DreamSoul-AI/denoising-speculative-decoding为public且有push权限；未推送、未创建PR、未触发新提交CI。保留完整本地提交与待审PR文本，等待明确批准。

接续补查：缓存环境覆盖为空，本机D盘模型目录只有4B/8B。四个0.6B Run仍无result，未launch；不把缓存目录未知写成整个设备绝对不存在权重。核对当前源码digest、251项JUnit、41条包验收命令、CUDA raw源码身份及Data/Model/Algorithm与交接的空diff，工程证据保持成立。CUDA实测发生在49dec9c的未提交树，源digest8573dfa6b73d44fb已与当前实现核对相同。

C-007使用新Toy/Stub make及三条status命令复现，原生config/jobs为eval而pending读数train，eval过滤0条；readout保持文件哈希。已按授权去重脱敏反馈 [Issue #37](https://github.com/DreamSoul-AI/RPipe/issues/37)。源码未改；仅补充兼容事实、正式精选证据及本地交接，不重新运行已通过的工程批次。

## 2026-10-14 ModelScope四格与B-008纠错

用户要求ModelScope下载并继续工作后，在Study shared准备新0.6B缓存，记录实际文件哈希和加载/解码资源，保留CUDA torch及全部科学配方。v3四Run成功，但整轮因素门误拒绝；先设计、红例，再从声明/基底复用Rpipe有效合并分组修复B-008，不信任当前index作为期望，不放宽子集/seed/source/evidence门。新的v4四Run/Study全部通过，v3-v4原始token/AR/接受/KV和alignment数值相等；AR计时误比较仅修对照脚本，所有失败保留。正式事实在四格报告/JSON/图，不重复长矩阵。

新增来源JSON使复制fixture的*.json排除过宽，最终首轮253 passed/1 failed；已只排除生成artifact，声明3用例复验通过。36格tiny等值，最终源码wheel/sdist全链通过，Ruff/mypy通过；最终全量单独保存，未把Toy/Stub或旧批次算真实模型质量。

最终本机全量254 passed（177.46秒，0 failed/errors/skipped），Ruff/mypy、36格严格数值、wheel/sdist 41命令全部通过。正式证据追加smoke/v031_validation.json，不覆盖历史来源；真实v4 complete/passed=true，v3拒绝保留。

用户明确批准具体公开内容与仓库/分支后，推送ae86c3c成功并创建PR #6（refactor/spec-rpipe→dev）：https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/6 。Branch flow和push/PR cpu-checks已启动，本交接更新后核验PR最新head检查；不自动合并或发布。既有用户workspace文件保持未跟踪。

2026-10-14 用户要求自主继续，不反复询问。下一最小阶段在feature/output-space-prerequisite上按v6提取tied零/一阶链式Study，先写PLAN与职责映射再实现。已接入最后归一化hidden、按owner代理状态和精确观测时序，12初轮针对性通过，后补EOS/真实前缀hidden等值共14项通过；原生CLI专项门通过，owner证据篡改拒绝。尚未声明真实模型结果/整体完成，完整回归与旧轨迹对照继续。

输出空间v1四格真实Run已成功、Study complete/passed=true；源码670c744e0535dcad，三个投机输出均与47token独立AR相同。匹配输入均值链17、输出零阶10、一阶10；负结果保留，没有调整超参或扩大矩阵。串行资源与正式JSON/图/报告已记录，完整270测试/36格严格对照通过。

最终源码wheel/sdist 41CLI命令全部通过，两安装target的新输出方法真实解码均与AR一致；来源/PLAN/inputs最终只读核验，无重复实验或重新签来源。正式成果本地分支保存，PR6保持原提交与绿色CI，不自动合并。

2026-10-14 按监督请求接续，先读最新STUDY_REPORT/offline_diagnosis，选最低新增实现成本的3新prompt小矩阵。预声明q172–174、4Runs、固定配方/预算，源码670c744e0535dcad不改；串行make/launch/process/report及9投机输出对独立AR完全一致。accepted为输入6/零阶3/一阶5，逐prompt和匹配prefix差异全保留；q173一阶与零阶root94/97接受差异对应2次calls减少，机制原因仍缺hidden/logit观测。正式报告/复算工具和资源证据保存；没有调参、大矩阵、远端写入或数据清理。

2026-10-14 接续三prompt矩阵，先写WINDOW_PLAN并对q173 root83/94/97只读重放：三case token/语义轨迹复现，正确seed时序/teacher/isolatedprefix门通过。再固定2×2实测方向/范数4查询，root83输出方向换成输入范数仍错，输入方向扩大范数仍对，未据此调配方。事后离线源状态重构说明slot1无当前velocity直接项；各类测量与有限机制解释正式归档，raw selected张量仅本机。新增工具6项针对性通过，一次unused import的Ruff问题已去除并复验，科学src/原Run保持。

2026-10-14 接续77fe411，先固定q173 root83 owner0第二槽的零/一阶互换与一阶原值回填三case，无搜索/新增默认。实测99token全部与独立AR一致、单槽scope与实际接受owner门通过；一阶←零阶accepted3→1且两后期seed/logits等于零阶，反向零阶←一阶accepted1→2仅恢复root94。离线核验两段代理传播，前一方向五固定根committed anchor/v仍与一阶原值完全相同，后一方向v仍0，收紧历史/当前速度解释但保留单窗口边界。源码/旧Run/权重不改，资源与失败边界入库，新3项针对性通过；不推送、合并或删数据。

2026-10-14 接续6a953bc跨prompt验证，按预声明首个共同hit前根固定q171/172/174 root63/42/41。旧两tool完整保留，新runner复用函数，15次AR逐token与原轨迹/单槽/实际owner/state门通过，accepted仍10/1/1。q171/172代理和first seed/margin受影响但不跨排序边界；q174 EOS边界仅下一proxy改变。全部null与有限解释入STUDY_REPORT，资源/raw SHA/离线复算正式归档，新2项针对性检查不一致状态也保留。未改科学源码/默认，未重签旧Run，不推送合并或删数据。

2026-10-14 接续7a02501，先写q173位置/margin/owner历史假设，再固定q175–178与fallback。原生make/launch/process/report四Run和20诊断均AR逐token通过，q175/176/177 null，q178双向各8→9但有深度2损失/恢复、first层收益/损失和末尾depth2补偿。同位root86状态完全相同，后期浮点delta保留不声称纯代理全中介。初次tokenizer BatchEncoding JSON失败在make前，修独立清单步骤并核对IDs，不改源码；所有raw/失败说明保留。14项针对性/新Study展开/docs、Ruff/mypy56通过，累计8prompt矩阵与正式证据本地归档，不推送合并删数据。

2026-10-14 接续a42313d，先冻结NUMERICAL_PLAN和q179–182 PLAN。q178 strictFP32双向各两次重复selected全bit相同；规范AR前缀state同步两self8、两移植9且全语义trace同strict，定位root89anchor pos88不同块row来源，不锁定低层kernel。q179–182四Run与20对照AR一致，原0/1为8/9、3/3、2/2、4/3，双向为10/10、3/3、2/2、4/4，坏点/微差保留。q182source45/root47 state相同但secondmargin跨界，12prompt表与PLAN_REVIEW正式归档，原合同sha不变。离线import错误在数据/模型前修同目录import并保留说明；17针对性/展开/docs及Ruff/mypy56通过。科学source670/default/旧helper未改，不推送合并删数据，fullweights/cache/selected仅本机。

## q178代理算子对照与后续草案（2026-10-14）

固定旧canonical state，先写OPERATOR_PLAN，再执行native及S64/N64/N32的self/单槽移植共8格，288tokens/220forwards全部AR一致且完整路径保持8/27和9/26。实际CUDA profiler记录NormTwoOps归约kernel关联；变体mask有<=5.96e-8差异，不能当作无效干预，也不能由此锁定transformer舍入源。入口85未变，后续provider必要性受限；未测入口舍入、组合干预或反向。原工具和合同冻结不改，新离线分析/报告分开。新cross_task固定q241–244、分词692/647/595/901，复用原生配方与flow，只草案；q_proj固定行协议另列未执行。最小功能候选437f201依赖PR6，33提交文件哈希清单和PR正文仅本地，不新建PR、推送、合并或删除。验收以本轮实际记录为准，未重跑历史全量／完整矩阵。

本轮相关检查累计26项通过，Ruff/mypy56通过，四旧来源冻结／16原始门与候选33blob SHA通过。新draft YAML冒号与工具导航首轮失败保留并修复；测量源码／分析器原字节归档，当前CLI只补docstring且AST一致，旧测量不重签。

## CUDA norm图、q_proj固定行和证伪checker（2026-10-14）

先冻结新norm／q_proj执行合同，再串行8+6生成共504tokens/385生成forward+4块辅助forward，全部独立AR一致。norm实际kernel图不同但返回值完全相同，标为数值无效对照；q_proj row4 GEMM/GEMV/FP64替换有效并传入后续proxy/hidden，完整路径仍self8、graft9。4source blocks的独立cache重放全bit通过，来源/strict/model/snapshot绑定保持。q241–244写export/check/mre脚本及缺格/重复/非法depth2/未检H2规则，只跑合成逻辑与配置验证，不启动真实cross-task。报告收紧入口Phi和未测算子／prompt范围，不改源/default/旧合同，不推远程、开合PR或删数据。

本轮28项相关检查、Ruff/mypy56通过，四旧来源锁／两新raw tensor SHA保持；无真实cross-task index/runs。工具先满足运行示例／导航合同再测量，测量工具、common及合同原字节未改。

## 真实跨任务、入口映射与canonical反例（2026-10-14）

为固定q241–244写EXECUTION_PLAN并更换version，新make四Run。首格43秒Flow/资源/AR门通过后，只launch其余3格，全部成功；process/report/checker export后冻结Root716/658/600/905，20诊断1950tokens/1704forwards全AR一致。原生input/zero/one85/58/45、移植双向有得失。其后q178 Phi64+post-attention RMS8格288tokens/220forwards：Phi/RMS64有效微差、singleton等价，路径仍8/9。按原准则单独冻结第一阳性q241 canonical4格400tokens/349生成forward+792-position teacher，native11→12、canonical14→14且路径不同，限制H2数量差保持及自值干预解释；源/default/合同不改，不推/PR/删。详见正式报告。

收尾核验：30项针对性测试、Ruff、mypy56源码与5Study冻结门通过；10个测量工具/协议/快照Git blob逐字节一致，修复旧operator_analyzer快照CRLF存档差异，源码/default不变。

## q178舍入因素与q241状态因素（2026-10-15）

按c4cb011证据另冻两个协议，完成80直接算术格与六生成／600new／525forward。Phi/RMS所有单因素最终都等于native，多阶段FP64保留才非零；q241 anchor-only/velocity-only13／13，旧native11／12、双同步14／14，自值交互-1反驳局部可加性。canonical同位root720 state相同、proxy不同且第一margin翻转，四收益四损失相抵；自值首变756三收益。正式报告保留唯一归因边界和未执行局部直接效应；未改科学src/default/旧helpers/成功Run，未远程／PR／删除。

本轮收尾核验：37项针对性测试及Ruff通过，5个Study的科学source/RPipe冻结门通过，80格保存向量的标量差重新计算一致，18个精确字节路径的暂存blob与本机文件一致。完整原生矩阵、安装包、更多seed/层未重跑；科学源码/default/旧helper/成功Run无改动。

## source719即时因素与跨prompt非加性（2026-10-15）

新协议下q241两canonical原值完整重现，source719/current720四格各三重复。固定KV/tree/state，只换source719 proxy精确恢复真实graft两条seed logits，正确第一候选13→264；即时state微差改mask但未改判定，root720当前映射不改变当前验证，影响后续输出。q242两seed交互0/0、q243两seed+6/-3，36生成/3600new/3124forward+四teacher全AR一致，贪心seed完全重复，不当独立样本。原source/default/旧合同/工具不改，不远程/PR/删数据。

收尾核验：42项针对性测试、Ruff与5个Study科学source/RPipe冻结门通过；新分析重跑保持原证据逐项一致，26个精确字节路径暂存blob与本机文件一致。旧helper/协议/科学src/default无改动，本轮3800生成新token、24额外固定块、4teacher均在预声明预算内。完整安装包/长矩阵/随机采样温度未重跑。

## 非T0完整路径与owner1（2026-10-15）

另冻两协议，q241/T=.8 seeds123/124两独立AR路径不同，14生成全token审计一致，固定实际entry697/715移植10→9与9→10。原source719条件multinomial(T=.8/1，各64)代理恢复0→64/63，属于旧固定上下文。新增source717 owner1四格三重复精确重现下一719 seed logits，候选11→323却均不等于AR1008，随机验证各0/64但概率非零。总1600生成new、12额外块、256条件draw，零teacher；全负结果、概率精度与泛化边界正式保存。科学source/default/旧helpers/合同无变，不远程/PR/合并/删数据。

收尾核验：47项针对性测试、Ruff与5个Study科学source/RPipe冻结门通过；联合分析重跑与原formal证据一致，34个精确字节路径暂存blob与本机文件一致。1600生成new、12额外块、256条件draw在新协议预算内；旧source/default/配方/工具/Run未改。未做多prompt采样泛化、罕见事件统计检验或安装包／长矩阵重跑。

## 多prompt完整采样、rare功效与后续H7（2026-10-15）

新增10单位70生成／7000new／6328forward全AR，补12单位4prompt×3seed，6增6减，q242全正/q244全负/q241243混合。rare两温度各4096native抽样仍零候选命中，Bernoulli0/4096上界.000731远高于10^-11～10^-9，不能排除罕见效应或证明GPU尾可达性。先写回Study，再冻并执行H7：q242123/125第一-1最终+5/+2两个反例，计数通过共有/独有root严格重构；不强迫后期对齐、不冒充state因果分解。原source/default/helper/合同/Run不改，不远程/PR/合并/删数据。

收尾核验：58项针对性测试、Ruff与5个Study科学source/RPipe冻结门通过；矩阵/rare/H7分析重跑与正式证据一致，45个精确字节路径暂存blob与本机文件一致。新7000生成new、8192条件draw、H7零新forward均在新合同范围。旧science/default/helper/Run不改，未扩展多任务/多温度/硬件，也未证明native rare尾可达性。

## 新prompt／seed反例与罕见条件压力测试（2026-10-15）

985c7c1后另冻两协议，扩展六单位、42生成4200new3806forward全AR，H8的q242126正/q244126负均被-5/+1反驳；q245混合、q246两负均保留。原温度各16384native draw仍零；独立标记T=2条件131072draw出现候选11/323首命中seed52396/21263、计数2/7，不能外推原温度可达性或完整路径收益。写回报告后继续H10，原固定捕获34共有窗口无合法depth2翻转；新增两H7符号反转事例。19新增统计/合法性回归，不改科学源/default/旧工具/成功Run，不推/PR/合并/删。

收尾核验：80项相关回归、Ruff与5个Study科学source/RPipe冻结门通过。新增19项统计/合法性回归，六单位42生成4200new3806forward、163840native条件draw、H10零新增forward均在冻结预算；正式分析再次执行逐项一致，来源/model/输入/旧helpers核对保持。源/default/旧合同/成功Run未改，未推送/PR/合并/删除。新sampled同KV四因素、原温度native极小尾可达性、安装包/速度/多设备/长矩阵未重验。

## 4B主模型配对接续（2026-10-15）

按用户新指示转Qwen3-4B，0.6B报告/commits/raw原位归档，不再扩大rare或单位置。检索只找到旧4B输入空间EMA结果，未定位完整旧output0/output1正结果；旧树[14]/[7,2]不与新同链计数混比。既存14文件ModelScope离线目录复用，实际SHA/model digest绑定，原下载origin/immutable revision仍未确认。

同FP32/eager/T0/nonthinking/EOS/max100/[1,1]/k2/BC9：first26输入/零/一accepted238/329/321、主calls1955/1866/1876，独立AR2191；全source/model/token/Run/Study门通过。写回报告后执行预声明新4条：51/39/47、189/201/194，负结果全留。随后固定历史输入EMAβ.9999/λ.1/γ1两新Run，first26为240/1952、new4为39/202；输出/独立AR成功格按哈希复用不重跑。结论限于first26总收益，新prompt不稳定，一阶未普遍超过零阶，不推速度结论。

eval.warmup=false，Study局部仅关闭decoder clock，兼容时长字段是0占位；同权重直接CUDA载入避免完整host副本，不改科学src/default。原v1基线execute26全AR后collect导出模块失败，失败Run/锁/raw保留；v2新身份修正四phase callable后成功，不重签旧成功。后续4B应先冻结新的跨类别/长度/预算全路径计数协议，不恢复0.6Brare；no push/PR/merge/delete。正式主/new4/EMA报告分别在output_space_4b_recheck_v2、output_space_4b_holdout、output_space_4b_input_control。


2026-10-15 最新执行顺序：已有 Qwen3-4B 结果全部为 **FP32**，本轮先按主题提交、工作分支 PR 经 CI 合入 dev。其他研究暂停；最高优先复现 ESP Table 1（Qwen3-8B/BF16/T0、BC10/30/60、动态树与剪枝、mean mask λ=0.1）。8B 仅用于该复现；随后 Qwen3-4B/BF16 五方法加 AR 配对。新增实验不用 FP32、不测速。不得强推、合 main、删分支或数据。


2026-10-15 FP32 收尾已核实：工具/测试 bdb3f6c、证据 8a0969c；[PR #7](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/7) 的 Branch flow 与两个 cpu-checks 均 SUCCESS，已于 2026-10-14T19:17:08Z 合入 dev，merge 8677376e1ac878a159a3095711eac43420d97bde。未合 main、未强推、未删除分支/数据。8B/BF16 仅按 [Table 1 计划](<研究计划 Denoising Speculative Decoding.md#legacy-docs-research-esp-table1-bf16-reproduction-plan>) 继续；其他研究暂停。


2026-10-15 ESP v2 G.2协议修订：Qwen3 BC60主配置改为单mask/29 draft，双mask/19 draft固定作对照；旧冻结PLAN哈希不变。新修订预检单maskτ=1.636982、Δ1.74=-0.103018、exact3/13；双mask复用τ=1.579408、Δ=-0.160592、exact5/13，均为失配诊断值，不称成功。详见[修订报告](../studies/esp_table1_bf16_qwen_protocol/docs/STUDY_REPORT.md)。全480四格已冻结并串行启动，BF16/T0/256/seed123、Eq5 λ=.1 updating mean保持；4B frozen mean是后续不同基线，其他研究暂停。完整Run/权重/缓存留本机；没有测速、强推、合main或删除数据。


2026-10-15 最新τ主线：[全480 argmax审计](../studies/esp_table1_bf16_full_argmax/README.md)替代无argmax记录的v1。BF16/T0/默认归约/λ=.1 updating mean/动态树剪枝完全保留，BC60单mask为主、双mask仅对照。AR一致率及逐条首分叉独立报告，不作为τ执行门；全部提交token必须等于对应验证forward的argmax。v1部分日志/旧冻结合同保留；v2四Run在模型加载前缓存接入失败，保留失败后补接同一ModelScope本地snapshot，以v3新四Run串行运行。当前第一格BC10 Run dfe1b9c8d5d12e92已开始真实BF16解码，未完成全480。停止compact attention、单行RMSNorm与逐层数值机理；4B五方法仍等待Table1结果。新增观察器无额外forward，首token、截断尾部、错误argmax与恢复回归通过；上一push的CI两项launcher文档遗漏已用378c6a9修复，尚待新远端CI。

2026-10-15 BF16阶段结果：[8B Table1](../studies/esp_table1_bf16_full_argmax/docs/STUDY_REPORT.md)四格全480及原生process通过，318,830提交token自检全过；BC30 pooledτ1.563378、BC60单mask1.647301未达到1.63/1.74，三聚合/双mask对照/AR差异全部保留，PR #10经CI合dev。[4B BF16五方法+AR](../studies/qwen3_4b_bf16_five_methods/docs/STUDY_REPORT.md)11/11及原生process通过，809,118个Spec提交token自检全过，全部20组逐prompt对照与首分叉保留。输入一阶本次总accepted最高/calls最少，两输出对mean基线小幅总量改善，速度项对输出零阶无稳定联合收益；不外推总体优越或同输出加速。正式JSON与真实图重复读出字节一致、图已目检；两Study原生数字表去掉本机日志链接，原始Run/权重/cache未上传。[集成与CI记录](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/12)；冻结PLAN/科学来源保持，不新增FP32模型实验、8B其他用途、算子机理或测速，其他研究暂停。
