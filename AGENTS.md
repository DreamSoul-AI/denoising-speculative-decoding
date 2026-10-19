# Denoising Speculative Decoding Agent 工作约定

适用于本仓库。长期约定在这里。目录、解码口径、怎么跑 Study、测试入口在下表。数字在对应报告里。

## 1. 怎么工作

1. 先读 [README.md](README.md) 和相关入口。不必每次通读整个 `docs/`。
2. 文档和实现不一致时，先核对原因。已记下的接受数在 [history-extrapolation.md](docs/history-extrapolation.md)。
3. 改目录或分层看 [layout.md](docs/layout.md)。`src/rpipe/` 是上游拷贝（`src/rpipe/ORIGIN`，RPipe `main` v0.3.1）。`src/spec_mtp/` 只放 SpecMTP。不改拷过来的框架文件，Windows 先 import torch 除外；2026-10-20 用户明确授权的固定副本 R1 迁移范围、原件及差异按 [迁移合同](docs/rpipe-migration.md#r1-local-backport-20261020) 核对，不构成后续任意改动许可。不要把 SpecMTP 注册进 `rpipe`。
4. 按改动范围验证，并改被这次改动碰到的文档。

## 2. 文档入口

| **需要** | **入口** |
| --- | --- |
| 目录、vendored RPipe、`origin` 规则、shared | [layout.md](docs/layout.md) |
| 解码、mask、和论文不一致的地方 | [architecture.md](docs/architecture.md) |
| 怎么跑 Study | [studies/README.md](studies/README.md) |
| RPipe 职责映射与迁移边界 | [rpipe-migration.md](docs/rpipe-migration.md) |
| 测试与 CI | [testing.md](docs/testing.md) |
| Windows | [windows.md](docs/windows.md) |
| 还没拍板的下一步 | [brainstorm.md](docs/brainstorm.md) |
| 唯一项目研究计划 | [中文](<docs/研究计划 Denoising Speculative Decoding.md>) / [English](<docs/Research Plan Denoising Speculative Decoding.md>) |
| 研究证据、历史草稿与规范入口 | [research/readme.md](docs/research/readme.md) |
| 论文 PDF | [papers/index.md](docs/papers/index.md)。PDF 不进 git |
| 问题与兼容性 | [bugs.md](docs/bugs.md) |
| 开发记录 | [record.md](docs/record.md) |
| 异机接手与待验范围 | [handover.md](docs/handover.md) |
| 已经记下的接受数 | [history-extrapolation.md](docs/history-extrapolation.md) |

一般技术文档文件名用小写；Study 目录入口用 `README.md`，报告和数字用 `docs/STUDY_REPORT.md`、`docs/NUMBERS.md`。项目计划只保留 docs/ 下上述中英文两份（全称文件名按Enmao 12:53要求）；子Study计划写入项目计划独立章节，不新建PLAN.md。

## 3. 协作

- 保留工作区里已有改动。临时输出放 `.tmp/`。Study 产物（`runs/`、`shared/`、`scripts/`、`index.json`、`process.json`）不进 git。
- 默认分支为 `main`。工作分支通过 PR 先进入 `dev`，再由同仓库 `dev` 的 PR 进入 `main`；不直接推 `dev`/`main`，不强推或删除长期分支。
- Enmao已长期授权：本仓库范围内的正常提交、按PR流程推工作分支、合入`dev`及用本地检查代替远端CI直接执行，不重复询问、不使用提问弹窗。授权依据及审批记录见[handover.md](docs/handover.md)。
- PR按`cpu-checks`和`Branch flow`的检查范围完成验证；可提供远端结果或等价本地检查证据，写明实际命令、结果及未验证范围，不把未运行的远端CI写成通过。保留平台实际强制的分支保护和评审要求。
- 自动审批拒绝时，在报告写明准确命令、工作目录、拒绝理由和执行状态，继续不依赖该动作的工作，交监督助手处理，不弹提问窗。
- 删除实验数据、强推、合入`main`或花钱前暂停相应动作，在报告记录待处理事项，不弹提问窗。Enmao已授权已合dev的本地/远端topic及废弃worktree登记清理：远程合后即清，本地退出后清，每约5个合dev PR全面复查；核对tip未变、祖先关系、运行进程与数据。main/dev及GPU活动分支/worktree保留，未合过时项先列PR说明。废弃目录的数据原地保留，不以清worktree删除实验结果；清单见[清理记录](docs/research/remote-branch-cleanup-2026-10-15.md)。

## 4. 代码放在哪

- `src/rpipe/`：调度、Flow、Tracker、artifact、make。提交见 `ORIGIN`。
- `src/spec_mtp/`：Data / Model / Algorithm（`(eval, spec_mtp)`）、`structure/api/recipe.py`、`flow/<phase>/run.py`、专项 artifact/evidence/validation；根 recipe/cli 与 flow.checks 仅兼容导出。注册发生在 Study `recipe.py`（import 时 `register_sources()`，Run 时 `register(ctx)`）。
- 入口：`python -m spec_mtp` 是薄封装，其余转给这份 `rpipe`。不恢复 `src/main.py`。不要加 `sitecustomize.py`。

## 5. 实验

- 开跑前在唯一项目计划写该Study独立章节，并在Study声明记录不可变计划commit/blob与章节（问题、对照、指标、停止规则、[studies/README.md](studies/README.md) §5 排班）。
- `python -m spec_mtp make studies/<name>`，再 `python -m spec_mtp launch studies/<name>`。
- `flow.prepare_shared: false`（RPipe make 下载只认 torchvision）。Run prepare 仍把 SpecBench 和权重写进 `shared/`。**不要写 `origin: foreign`**（[layout.md](docs/layout.md)）。权重用 ModelScope。
- 实验验收遵循最新用户确认及冻结协议。当前BF16/T0研究的正确性门是全部提交token等于对应验证前向logits的argmax；独立AR一致率仅报告、不作失败门，见[当前研究计划](<docs/研究计划 Denoising Speculative Decoding.md#legacy-docs-research-esp-table1-bf16-reproduction-plan>)。历史执行证据与已有成功Run不重签。按本次用户授权，旧AR门失败且execute完整的Run可仅重新验收：保留失败result备份、raw/输入/来源哈希及新validator SHA，不重跑execute。永久停止BF16 ESP相对独立AR的算子数值机理排查。接受数、τ及其口径写入报告，不把γ叫成slope。
- 已有数字的格子不重跑。不复用一个 Run 目录。源码或输入变化用新的 algorithm.config.version；make/remake 不签收旧成功 Run。
- Study collect 从 execute 的原始观测计算指标，summarize 跑成功门，write 再核验，process 只终验当前 index。
- Windows：`PYTHONUTF8=1`，`D:\anaconda3\python.exe`，不另建 venv。
