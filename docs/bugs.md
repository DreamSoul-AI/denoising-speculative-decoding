# 问题与兼容性记录

## 2026-10-20 S1 固定副本读取与计划准入残留

合法但非对象的 JSON 结果绕过旧副本四个既有容错 reader，导致 status、Study process、launch 与 Run process-only 入口崩溃；非法 JSON 对照已有未完成处理。已显式捕获 CorruptArtifactError，不改严格 decoder 或异常继承。缓存计划必须在过滤已完成项前验证原 jobs/batches，多重集不符、非对象 job、损坏数字和负 round 拒绝复用，保留旧默认及合法顺序。反例、分组回归和来源边界见 [开发记录](record.md#2026-10-20-s1-第三批迁移残留) 与 [迁移合同](rpipe-migration.md#2026-10-20-第三批固定副本读取与计划准入残留)；均为本分支工程验证，未合并集成。

## 2026-10-14 交接前记录

- v0.3.1 接入待验：发布固定为 7a15c46b75c5eb951cccd60ed7b9fc3463db90f5；B-030 修复旧 process 与新 index 身份不一致时仍 report 成功，以及 NUMBERS 写入保全。当前专项 report 已先执行 Study 终验，升级时须复用上游身份/原子写入门并回归专项拒绝路径；不能用上游发布结果代替本项目验收。执行副本尚未更新，C-004/C-005 等边界未据此关闭。
- 本轮数据/API/阶段职责迁移及 ModelScope 离线参数传递的针对性与全量回归通过。详细验证和迁移过程的失败留 [smoke 报告](../studies/smoke/docs/STUDY_REPORT.md#八职责重构检查点2026-10-14)。
- Windows 目录改名出现部分移动，已核对并补齐到新 checkout，editable 导入正常；旧路径残留仅临时目录，未强删。该环境故障不归因于 RPipe。
- 向上游的通用反馈尚未提交。C-004 的原始旧 Run 终态未备份不可恢复；接手须在新隔离 Toy/Stub Run 低成本复现、核对 v0.3.1/当前 main，检索 open/closed Issues 和 PR 去重，按已授权的 feedback 规则脱敏后反馈，回写真实 Issue URL。

## 2026-10-13

| **编号 / 来源** | **复现与影响** | **处理 / 验证** |
| --- | --- | --- |
| C-001 / 固定 RPipe v0.3.0 接口边界 | SpecMTP 注册 Data 后仍不能使用 make 的默认共享预构造：prepare_shared_data 先调用 endpoint_for，local_prompts/SpecBench 没有 torchvision host。Recipe 在 Run prepare 才运行，无法改写 make 下载职责 | 保留当前工作区已做的 prepare_shared:false；Run Data/Model builder 继续落 shared/data 与 shared/model。make→launch 回归验证，不修改 vendored 下载逻辑 |
| C-002 / 固定 RPipe 源码与下游 lint 策略 | 对固定副本运行 Ruff F401，train/__init__ 与三个 api facade 的公开转导出触发 8 项错误 | pyproject.toml 精确到四文件忽略 F401；保留上游源码与公开接口。Ruff 仍检查其余规则和全部专项代码 |
| C-003 / Windows 本机导入兼容 | 固定上游 numpy-first 与本机科学依赖 DLL 顺序不兼容，当前分支已有 torch-first 例外 | 保留唯一 src/rpipe/__init__.py 例外；windows_selfcheck 分别验证两种入口。非逐字节未改的副本 |
| B-001 / 下游证据门 | 只读 summary.csv 和 sources digest，可能接受缺失 raw token、执行后源码变更或手改摘要 | execute 保存 inputs/observations；summarize/write/process 校验真实 token、接受/cache 轨迹、配置、输入文件和来源；失败门回归 |
| B-002 / 下游历史 Run 验收 | 仓库 smoke 已有旧源码锁；新源码普通 run 的首轮测试被拒绝 | 保留失败日志与旧产物；测试改隔离目录，正式 smoke 加新 version。禁止对旧结果重新签收 |
| B-003 / 下游 CUDA 证据核对 | GPU tiny 原始 exact-match=1，但 legacy settings.device=cuda 与实际 model.device=cuda:0 被误判配置漂移，summarize 正确留下 failed 而未定稿成功 | 用已记录的实际 settings 表示重建有效 eval 配置；完整 config digest 仍拒绝真正变化，实际设备独立记录。新增别名回归，保留 final02 失败，用 final03 复验 |
| C-004 / 固定 RPipe 同 ID 失败写入 | 初次旧 smoke 的 Run 3b0724f5e0396351 在 prepare 来源拒绝后，FlowRunner 回写 failed result；运行前未备份原 result，原终态无法确认/恢复，raw/log 未删除 | 专项薄 CLI 在进入库之前预检专项锁及上游 freeze，run/launch/run-one/process 拒绝不触碰旧 result；新增字节保全回归。仍要求新 version，不修改 vendored Runner；直接 python -m rpipe/FlowRunner 仍遵循库原失败回写行为，旧 Run 不可直接复跑 |
| C-005 / 固定 RPipe 聚合先于 Study 终验 | RPipe 先写通用 process.json，再调用 Study process；专项来源/证据门失败时通用 complete 可能仍为 true | Study hook 将自身质量结论写回同一 process envelope，失败设置 complete=false；薄 report 先复验当前 index，不从失败矩阵生成新数字表。保留 Run 的成功状态与证据 |
| B-004 / 本项目 CI 安装脚本 | Actions run 38367588147 在 Install CPU dependencies 将 read_bytes 传给 tomllib.loads，TypeError；lint/type/tests 全未执行 | 改为 read_text(encoding=utf-8)，本机执行同一依赖提取片段通过；保持 CPU torch 排除解析与 Actions 启用，远端新提交复验 |
| B-005 / 本项目数据目录与类型解析 | 本机及 Actions 38367956325 的 mypy 把根 datasets/ 识别为 namespace package，遮蔽已安装 Hugging Face datasets，报 load_dataset 不存在 | 本项目 src 包均有 __init__.py；设置 mypy namespace_packages=false，保持所有现有严格规则，根目录直接复验 40 files passed。早期从目录外检查虽通过但未根除该命名冲突，现已明确记录 |
| B-006 / 本次正式 readout 文档 | Actions 38368656580 的运行测试 228 passed，仅 NUMBERS.md 三个本机 raw log 超链接在 clone 中失效；本机目录存在掩盖了文档依赖缺失 | 正式 Numbers 保留原数字，将 raw log 改成明确标注不随 clone 提供的文本路径；正式证据链接指向入库 JSON/报告，不放宽链接测试 |

Study 文档适配边界 C-006：固定上游 Study 指南的 learning curve、训练 resume 与 torchvision data 命令服务于训练案例；本项目为 training-free 解码。报告保留真实接受数/轨迹图和来源，不添加训练曲线、KV/mask/RNG 恢复承诺或额外下载入口。模板与 smoke 计划/报告按此合同核对，未修改上游。

上述 C 项描述固定上游与本项目的边界，不是对 RPipe 上游的修改请求。正式数字与验证范围见 [smoke 报告](../studies/smoke/docs/STUDY_REPORT.md)。

## v0.3.1 当前设备检查（2026-10-14）

完整副本已升级到固定 7a15c46，101 文件仅 Windows torch-first 本地例外，MIT 保留。首轮 pytest 因新设备缺 Hypothesis 停在收集；补齐后 250 passed / 1 failed，新增 report 共存检查发现专项入口在质量拒绝后未更新旧 process.complete（B-007，下游问题）。已先更新设计，再让上游只读身份校验先于专项终验，并传入真实 process envelope；针对性 46 项已通过；最终全量回归另见当前 smoke 验收。旧身份拒绝和原子写失败保全旧 NUMBERS，不更改 vendor Runner 或科学配方。

C-004 已在新隔离 Toy/Stub 单 Run、两步 CPU、v0.3.1（当前 main 也是同一 7a15c46）复现：先成功，再给 Study YAML 添加无语义注释以触发 freeze 拒绝，native run-one 退出 1，但相同 ID 的 result 从 succeeded 覆盖为 failed，前后 SHA256 不同。拒绝发生在计算前，原日志和 raw 未清理；本轮保存了 result.before/after。专项 CLI 的预检保全仍通过。属于通用证据保全改进，Runner 当前文档允许 failed 回写，不将其描述成未文档化的算法缺陷。Issue 去重完成（open/closed Issues 空；PR #29/#33 的恢复/报告修复未处理此入口），待反馈真实 URL。

C-004 已反馈：[RPipe Issue #36](https://github.com/DreamSoul-AI/RPipe/issues/36)，2026-10-14 实际创建。只提交合成 Toy/Stub 配方、公开 commit 与脱敏现象，未提交本地路径、私有资料、数据/权重、凭据，未修改上游代码或权限。

C-007 / readout 显示边界：四格实际 config 与 jobs 均为 eval，但 pending Run 的 status mode 显示 train。v0.3.1 的 _mode 仅从轴或已落 result.control 读取，未成功 Run 又没有 algorithm.mode 轴时回退 train；这不改变真实调度的 mode。当前设备尚缺真实权重，不能从该显示推断训练发生；专项未修补 vendor。

B-007 当前设备最终 251 项全量通过，新增 report 字节保全/质量写回检查通过。Windows 工具初次 freeze 缺锁拒绝与工具导航两项失败均保留，修正为显式 make 和登记用法后通过。当前未解除的真实模型限制是 Qwen3-0.6B cache 缺失；不是 CUDA 不可用。

C-007 最小复现补充：固定v0.3.1，两步Toy/Stub声明但不launch，仅CPU make；生成config和jobs均为eval，status列为train，--mode eval显示0条，--mode train显示1条。已确认读数误分类，未影响实际调度/未执行计算。复现脚本初次误用rows键导致KeyError，核对list_runs真实返回runs后修正；不归因于库。原始make与复现保留.tmp/v031-20261014/c007。

缓存范围补查：未发现MODELSCOPE_CACHE/HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE有效覆盖；D盘模型目录只有Qwen3-4B和8B，不替代计划中的0.6B。不能据此断言所有磁盘位置都不存在权重，仍需实际缓存路径。

C-007 已完成真实CLI过滤复验，三命令均exit0，未改index/config/jobs字节；已公开反馈 [RPipe Issue #37](https://github.com/DreamSoul-AI/RPipe/issues/37)。反馈仅合成Toy/Stub声明与公开commit，未提交内部路径、真实权重或资料；保留vendor，不擅改四格轴来绕过显示问题。

B-008 / 下游Study完整性：v3 Qwen3-0.6B四Run全部succeeded，ESP各exact_match=1、source/evidence门通过，Study却incomplete。原因是原declared_factors比较轴补丁，上游index保存合并后配置；algorithm.config嵌套轴的version/eval/继承字段使expected_factors_match=false。先改设计，待复验有效因素推导，不放宽门；v3 artifacts已保留。

B-008回归过程：嵌套轴CPU make/launch/process/report与子集拒绝用例通过；一条新增单元用例错误假定whole mapping轴会递归保留Study fixed里的version。实际expand_patches.set_dotted先用整个轴值覆盖该fixed子树，再与基底合并，因此期望应为base-v1。已核对接口并纠正用例，未修改上游或科学配方；首轮full在已采集该错误用例后中止并保存日志。当前真实四格version在基底，未受该覆盖边界影响。

B-008最终来源验证首轮full为253 passed/1 failed，唯一失败为test_recipe复制Study时笼统排除所有JSON，漏带新增provenance.include的正式snapshot/校准JSON。已改为只排除明确的生成artifact，保留正式来源材料，不取消provenance冻结或改路径匹配；fixture与源码门分别复验。

B-008实际v4终验通过：四Run succeeded、expected_factors_match=true、seed/source/evidence/complete/passed均真。嵌套轴CPU端到端/子集拒绝、真实q171新身份四格和v3-v4原始数值对照均通过。旧v3结果/失败不重签。ModelScope master revision未确认警告保留，实际模型与文件digest锁定；原缓存缺失条件由用户授权的显式下载解决。
