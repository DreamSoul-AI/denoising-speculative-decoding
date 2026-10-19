# Denoising Speculative Decoding

ESP reproduction + EMA-velocity masks.

当前研究计划：[中文](<docs/研究计划 Denoising Speculative Decoding.md>) / [English](<docs/Research Plan Denoising Speculative Decoding.md>)。原作者可运行入口未核验到，按Enmao 2026-10-16最新指示停止追找；现有实现标明**非作者官方复现**。旧[全480证据复核](docs/research/esp-table1-nonofficial-full480-recheck-2026-10-16.md)数学门全过，BC30/BC60单mask pooled τ=1.563378/1.647301，未达论文1.63/1.74。[8B独立复验](studies/esp_table1_nonofficial_revalidation/docs/STUDY_REPORT.md)三格已全部完成，239,137个token数学自检通过；后续七格范数和11格独立配对暂停，未追加实验。[最终交接与后续开题要求](docs/handover.md#sd-wrap-up-20261018)。已有4B五方法加AR及六格方向对照完整结果继续复用，不改prompt、不测速。

Training-free multi-token prediction:

1. **ESP** (Embedding-Space Probing), [arXiv:2603.17942](https://arxiv.org/abs/2603.17942). PDF is local-only: [docs/papers/index.md](docs/papers/index.md).
2. **EMA-velocity**: the same pipeline, mask from EMA of input-embedding differences.

Agent notes: [AGENTS.md](AGENTS.md). Layout and the SpecMTP `origin` / shared rule: [docs/layout.md](docs/layout.md). Decode / paper mismatches: [docs/architecture.md](docs/architecture.md). Structure mapping: [docs/rpipe-migration.md](docs/rpipe-migration.md). How to run a Study: [studies/README.md](studies/README.md). Recorded accepts: [docs/history-extrapolation.md](docs/history-extrapolation.md).

主仓执行副本的正式证据与本机归档对账：[ESP 工作副本对账](docs/research/esp-working-copy-reconciliation-2026-10-15.md)。

`src/rpipe/` is a file copy of `diaoenmao/RPipe` `main` at `7a15c46b75c5eb951cccd60ed7b9fc3463db90f5` (v0.3.1). SpecMTP registers onto that copy through a Study recipe.

## Quick start

Existing GPU environments: preserve CUDA torch; install the project with `pip install -e ".[dev]" --no-deps` and supply missing non-torch dependencies separately. The following CPU setup is for a new CPU environment.

### New CPU environment

```text
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
set PYTHONUTF8=1
python -m pytest tests/spec_mtp
```

For a fresh [smoke](studies/smoke/README.md), copy `studies/smoke` to an isolated directory and set a new Study name and `algorithm.config.version` before `python -m spec_mtp make <copy>` and `launch <copy>`. New T0 speculative Runs require a complete verifier-argmax audit: every committed token must equal its verification forward's argmax. Independent AR exactness is diagnostic; all Runs still require complete raw and source evidence. Source/input changes require explicit make/remake; do not rerun historical Run IDs. See the [acceptance report](studies/smoke/docs/STUDY_REPORT.md).

GPU cells are new Studies under `studies/`. Copy `_template/`. SpecBench lives in `datasets/spec_bench/` and is copied into that Study's `shared/data` on the first Run. ModelScope / HF weights land in `shared/model`. Do not start from `src/main.py`.

## Development

Repository: [DreamSoul-AI/denoising-speculative-decoding](https://github.com/DreamSoul-AI/denoising-speculative-decoding). Default branch: `main`. Changes enter `dev` through a work-branch PR, then reach `main` through a PR from this repository's `dev`. Both paths require `cpu-checks` and `Branch flow`; existing approvals remain required.

Research-to-Study extraction: [protocol](docs/research/rpipe-study-extraction.md), with the real-model [input-mask baseline](studies/input_mask_baseline/README.md). This prerequisite does not implement or validate the output-space / Kalman hypotheses.

Current checkpoint, validation and remaining real-model prerequisite: [handover](docs/handover.md). The execution copy is fixed at v0.3.1; current-device acceptance is tracked in the handover and Study reports.

Current-device v0.3.1 checks: [smoke acceptance](studies/smoke/docs/STUDY_REPORT.md) and [validation](studies/smoke/docs/v031_validation.json). The four real-model input-mask prerequisite Runs completed with ModelScope Qwen3-0.6B, FP32/eager and strict source/evidence gates; see the [report](studies/input_mask_baseline/docs/STUDY_REPORT.md).

Next stage: [output-space chain prerequisite](studies/output_space_prerequisite/README.md), tied final-hidden order0/order1 with a matched frozen-input chain. Implementation and four-case real-model correctness gates passed. On the fixed development prompt the output masks accepted10 drafts each versus17 for the matched input chain; no superiority or Kalman claim.

Follow-up: [three-prompt small matrix](studies/output_space_small_matrix/README.md) and matched-prefix analysis. All outputs remain AR-identical; q172–q174 acceptance counts for input/order0/order1 are1/1/1,5/1/3,0/1/1. These mixed development observations do not establish stable superiority.

历史Qwen3-4B **FP32**：[同prompt first26复验](studies/output_space_4b_recheck_v2/docs/STUDY_REPORT.md)两阶总接受均高于输入，[固定4条新输入](studies/output_space_4b_holdout/docs/STUDY_REPORT.md)未保持收益，[历史输入EMA同链控制](studies/output_space_4b_input_control/docs/STUDY_REPORT.md)已执行。三批全AR一致，计时关闭；这些结果及此前0.6B机理/罕见事件记录保留归档，不代替当前BF16证据。


历史原作者代码核查（2026-10-15）：[官方来源核查与复现报告](docs/research/esp-official-code-reproduction-2026-10-15.md)。尚未核验到作者 ESP 可运行实现，OpenReview 附件访问不完整；当次未迁入源码或启动GPU。该记录保留原作者代码要求下的历史结论；当前非官方独立复验按上文最新授权执行。

2026-10-15 历史自实现结果：历史4B FP32归档经PR #7合dev；[ESP Table1全480 BF16复现](studies/esp_table1_bf16_full_argmax/docs/STUDY_REPORT.md)四格和原生process完成，BC30 pooled τ=1.563378、BC60单mask=1.647301，未达到1.63/1.74，负结果保留，PR #10经CI合dev。[4B BF16五方法加AR配对](studies/qwen3_4b_bf16_five_methods/docs/STUDY_REPORT.md)已完成11/11及原生Study process，同480条、BC30/60、双mask动态剪枝树，809,118个Spec提交token自检全过。输入一阶本次总accepted最高、主calls最少；两输出方法对mean基线有小幅总量改善，输出一阶相对输出零阶未显示稳定联合收益。全部逐prompt负例与AR差异保留，AR一致率不作失败门；[集成与CI记录](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/12)。其他研究暂停；8B仅限该Table1复现，未新增FP32模型实验、测速、强推、合main或删除分支和数据。
