# SD research plan: fixed-data reproduction and mask-direction controls

English translation of the [Chinese master](<研究计划 Denoising Speculative Decoding.md>), handover status updated2026-10-18. The formal closure report is already merged into dev in PR #50; its body below is retained without rerunning experiments or re-signing evidence.

> Current conclusion: all239,137 mathematical self-checks pass. BC30/60 tau1.563377504/1.647301264 remain about0.067/0.093 below the paper; the target values are not reproduced.
> All three8B cells finished. The remaining seven norm cells and11 independent paired cells were not executed; older queues remain paused.
> See [handover](handover.md#sd-wrap-up-20261018) for clone-ready evidence boundaries and requirements for a future gap investigation. This update documents preparation only and does not authorize a new experiment.

<!-- SD_CLOSEOUT_20261016_START -->
<a id="sd-stage-closeout-20261016"></a>
## Reproduction phase summary

### What the numbers mean

The aim is to check ESP Table1 acceptance efficiency. Text is divided into tokens, which can be characters, words or word fragments. A draft forward explores several candidate tokens and a verification forward decides what can be committed. **BC10/30/60 are per-round draft/verification token-computation budgets.** Larger budgets permit more candidate exploration. Qwen3 uses a single mask in each main cell, allowing4/14/29 draft nodes, with a dynamic tree and pruning.

**Tau is the average number of new tokens committed per verification round.** The main value divides total committed tokens across480 prompts by total verification calls, excluding the initial prefill token from both totals. Tau1.563 means about1.563 committed tokens per round. No speed measurement was performed; tau is not measured speedup. Prompt-equal and category-equal aggregates are retained separately.

This is explicitly a **non-author-official reproduction**. No verified executable author experiment entrypoint was available, and HF third-party code was never labeled official. Settings remain SpecBench480, Qwen3-8B, BF16, T0, maximum256 new tokens/EOS stopping, lambda0.1 updating mean, dynamic tree and pruning.8B is restricted to Table1. No prompts are tuned, no timings measured and no operator-level numerical study continued.

### 8B Table1 results

| Cell | This repeat pooled tau | Paper target | Repeat minus paper | Status |
| --- | ---: | ---: | ---: | --- |
| BC10/one mask | 1.399047031 | No target fixed for this cell | — | full480 succeeded |
| BC30/one mask | 1.563377504 | 1.63 | -0.066622496 | full480 succeeded; target not reproduced |
| BC60/one mask | 1.647301264 | 1.74 | -0.092698736 | full480 succeeded; target not reproduced |

Sources: [BC10](../studies/esp_table1_nonofficial_revalidation/docs/bc10_checkpoint_v1.json), Run80ff7a41958d8134, and [BC30](../studies/esp_table1_nonofficial_revalidation/docs/bc30_checkpoint_v1.json), Run88d13ec4ecf0023c. The completed BC60 is a8e6bba1dd1d80e2. [Formal repeat evidence](../studies/esp_table1_nonofficial_revalidation/docs/repeat_evidence_v1.json) binds all three cells, per-prompt first AR divergences and hashes. BC30 prompt-equal/category-equal tau values are1.593232237/1.563387272, both below1.63. Changing aggregation does not establish reproduction. BC60 prompt/category tau values are1.669498033/1.649693144, both below1.74. Each of the three cells ties its old implementation on accepted/calls/tau for all480 prompts, with480 equal output and AR hashes.

### 4B: the benefit is not unique to e2−e1

The completed fixed480 Qwen3-4B BF16/T0 BC30 direction controls give:

| Control | Pooled tau |
| --- | ---: |
| Fixed e2−e1 | 1.507777653 |
| Mean of three norm-matched random directions | 1.494473978 |
| Zero bias/input zero order | 1.422420206 |
| Content initialization | 1.364829262 |

Bias changes the draft placeholder representation while the input text stays fixed. Norm means vector magnitude; direction means where it points. The original direction is the difference between the first two input-token embeddings. Random controls replace direction at matched initial norm. Sources are [formal direction evidence](../studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json) and the [per-Run and paired report](research/4b-full-direction-controls-2026-10-16.md). All485,860 committed-token audits across six cells pass. Random seeds0/1/2 give1.495188134,1.464349086,1.523884714: all exceed zero bias, and one exceeds the original direction.

The evidence supports that a norm-matched random bias can also help; e2−e1 is not uniquely responsible. It does not prove that direction is irrelevant or that magnitude is the sole cause. Random seeds still vary, and content changes initialization and subsequent velocity updating, with lower tau. The magnitude sweep is incomplete: only template0.5x finished480, tau1.471389646, with all80,940 token audits passing. Its collection export failure was recovered without execute rerun or raw-SHA changes; see [recovery evidence](../studies/qwen3_4b_bf16_bias_norm_continuation/docs/first_cell_recovery_v1.json). The seven remaining cells are stopped under the20:55 instruction, with all existing material retained.

The original completed five-method plus AR comparison has BC30 input-first1.511957584, output-zero1.433178245 and output-first1.435539512. All methods, BC60 and per-prompt wins/ties/losses are in the [original paired evidence](../studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json). Fixed-direction controls and the original updating-velocity recipe are different configurations and must not be pooled into one ranking. The later11-cell independent repeat was only made, never executed, and will not start.

### Verification, gaps and unresolved questions

The gate requires **every committed token to equal the argmax of its corresponding validation-forward logits**, including first-token coverage, ownership and hash/commit chains. BC10 passes79,757 checks, BC30 79,762 and BC60 79,618: all239,137 pass, as do all three native Runs and Study process. AR is the ordinary token-by-token generation baseline. Agreement with independent AR is diagnostic:188/480,175/480 and185/480 for BC10/30/60 respectively. BF16 tree and single-sequence AR forwards have different shapes and can diverge. This is disclosed and does not replace the mathematical gate. Output lengths can differ across methods, so paired counts are not equal-length speed comparisons.

Fixed inputs, model snapshots, native source/plan freezing, Run configurations, raw audits and aggregation have been checked. All618 CPU regression tests pass, plus ten explicitBF16/actually-tied tiny-model checks for all five methods at both budgets, complete commit coverage and corrupted-token rejection. These are software checks, not substitutes for real4B/8B evidence; see [CPU readiness evidence](../studies/qwen3_4b_bf16_paired_revalidation/docs/cpu_readiness_v1.json).

Passing correctness does not reproduce the paper's target tau. BC30 remains0.066622496 below target and BC60 0.092698736 below; none of the three reported aggregate conventions reaches its corresponding target. Possible, unproven explanations are author-side input/template/stopping contracts that have not been fully verified in the available sources, differences in nonofficial tree/pruning details, and unknown paper counting/weighting details. CPU counting reconciliation and three aggregates have not removed the gap; see [contract reconciliation](research/esp-table1-cpu-contract-reconciliation-2026-10-16.md) and [gap register](research/esp-table1-protocol-gaps-2026-10-16.md).

Unresolved questions are the exact contract responsible for the paper gap; separate magnitude/direction contributions; why output-zero/first underperform input-first and the output velocity adds little; and effects of differing output lengths/trajectories on comparisons. **No next-phase proposal is made.**

### Closure status

Only the existing BC60 was allowed to continue and has now naturally completed. Queue64880 and ledger50944 have exited; two temporary launchers are preserved by SHA and disabled, with no source/results/weights/cache deletion. Final source/mathematical acceptance is complete. The merged experiment/sd-nonofficial-repeat-norm local/remote branch was deleted;17 unused legacy preparation files were SHA-archived and159 generated bytecode files cleaned. All experimental results and model caches remain; the old data checkout remains detached at its unchanged commit. Deliver one milestone PR to dev; the current unmerged branch stays until merge. Enmao review determines any later phase; older experiment queues in this plan are no longer execution instructions.
<!-- SD_CLOSEOUT_20261016_END -->

## I. Current scope, status and execution priority

Older scheduling below is historical. The20:55 closure summary governs; all later experiment queues are paused.

Enmao's final instructions of2026-10-16 12:44/12:53 govern: stop searching for an official ESP entrypoint and label the existing stack **non-author-official reproduction**. Study the fixed480 SpecBench Table1τ gap and whether input-bias magnitude explains acceptance gains more strongly than the particular direction. No new Kalman/method work, prompt changes or timing; BF16/T0 only.

Status: the four original8B nonofficial Table1 cells are complete with passing mathematical gates. BC30/60τ=1.563378/1.647301, below1.63/1.74. The4B five-method-plus-AR and six BC30 direction cells each cover480. The three norm-matched random directions averageτ=1.494473978, versus1.507777653 for e2−e1 and1.422420206 for zero bias. Enmao's working mechanism hypothesis is that magnitude is the primary factor and direction matters little. Existing results reject e2−e1 uniqueness, but random values1.464349086–1.523884714 vary; arbitrary-direction equivalence and the magnitude-response curve remain unestablished. Retain original evidence and limits rather than rewriting historical results.

Execution order: A, follow the newly authorized request to run an independent full480 nonofficial Table1 revalidation on8B, or the largest existing model admitted by host RAM, using single-mask BC10/30/60,λ0.1 updating mean,dynamic/pruning,256/EOS andseed123. Reuse old complete results as comparison, but use a new version/Run and never overwrite or re-sign. B,4B BF16 BC30 magnitude scan at0/0.5/1/2×, retaining e2−e1 and random seeds0/1/2. Reuse complete zero and1× results; run only the eight new combinations of four directions×0.5/2. C, assess curves, paired accepted/calls/τ win/tie/loss, actual bias norms and output lengths. No seed selection or prompt changes. Do not expand magnitude scans toBC60 yet.

GPU policy now favors useful work without idle hardware: **one GPU task at a time; admission is measured process-tree peak host RSS plus2GB available host RAM** (conservatively2GiB), with actual GPU vacancy and sufficient VRAM for the model/workset. The former peak-plus2GiB VRAM rule is superseded. Do not inventory/hash CPU scripts as a gate, kill other tasks or run full≈32GiB quantization training. Historical scheduling snapshot: at≈13:02 on2026-10-16 a quantization Python task occupied the GPU; host availability was≈19.99GB. Wait for that task to finish naturally, then use one serial queue; do not claim execution yet. Historical8B sampled process-tree peak RSS was17305047040 bytes; plus2GiB requires19452530688 bytes. Sampling is not a guarantee for new trajectories; record resources and retain failures. The old661MiB VRAM shortfall is not a host-RAM blocker under the new policy.

This file and the [Chinese counterpart](<研究计划 Denoising Speculative Decoding.md>) are the only project plans. Sixty Study plans, specialized protocols, reviews and v6 are consolidated as independent ChapterXI sections; old standalone files are removed by the explicit12:53 instruction. The consolidation manifest preserves source bytes/SHA, Git blob identity, EOL and target anchors. Old independent-PLAN, AR-exact and official-source requirements are historical, superseded by current scope. Never re-sign Runs or treat today's plan as the original frozen bytes. New Runs cite immutable project-plan commit/blob and section. Renaming must not interrupt other running work.

Literature-reading depth and limitations follow. Night-time PR22–40 results/source audits/reconciliation are incorporated in ChaptersIX/X. The independent8B repeat is running; the norm experiment remains pending. Progress is not a validated result.

## II. Literature review

This is a targeted evidence update, not a systematic review or a rereading of all60 historical references. Information cutoff and this round's checks are dated2026-10-16. Sources are arXiv originals/abstracts, the official Spec-Bench repository, prior source audits and result files. Foundational references were followed from v6; the actual search “SpecBench official github hemingxia Spec-Bench” identified the current official entry `hemingkx/Spec-Bench`. Future checks may use training-free multi-token prediction, embedding-space probing, speculative verification, dynamic draft tree and mask initialization; this search scope does not authorize new experiments.

Include relevant author papers, official data/code sources and the user-specified third-party implementation. Different models, precision, trees, training-based drafters or datasets provide context, not a fair local ranking. Exclude unsupported aggregate numbers, marketing summaries and unrelated method expansion. Failure to verify official code does not prove it was never released. Discovery, reading depth, author claims, code audit and execution evidence are recorded separately.

| Source and location | Question and actual reading depth | Data/model | Algorithm | Metrics/conditions | Evidence, limits and use |
| --- | --- | --- | --- | --- | --- |
| [ESP v2](https://arxiv.org/html/2603.17942v2), §3, Table1, Appendix G.2 | Abstract, Eq4/5, Table1 and appendix locations checked this round; earlier relevant-text analysis in the [reproduction plan](<研究计划 Denoising Speculative Decoding.md#legacy-docs-research-esp-table1-bf16-reproduction-plan>) | SpecBench; local8B reproduction cells | Embedding masks, dynamic tree, pruning | Targets in SectionVIII; paper timing is not a deliverable here | Closest baseline; distinguish paper, author implementation, self-implementation and third party. Local failure is not proof the original method fails |
| [Leviathan et al.,2211.17192v2](https://arxiv.org/abs/2211.17192v2) | Abstract checked; sampling proof not re-proved here | Original Transformer conditions, not our4B matrix | Speculative decoding and target verification | Original sampling differs from this T0 study | Separates drafting/verification responsibilities; our same-forward argmax audit does not establish every sampling distribution |
| [Chen et al.,2302.01318](https://arxiv.org/abs/2302.01318) | Abstract checked | Author models/sampling, not pooled with local results | Speculative sampling | Not a local T0 comparison | Correctness/sampling context; no new sampling experiment |
| [EAGLE-2,2406.16858v2](https://arxiv.org/abs/2406.16858v2) | Abstract checked for dynamic-tree motivation | Author trained-drafter setting | Dynamic draft trees | Different from training-free masks | Context dependence is an alternative explanation; no local baseline run is claimed |
| [Medusa,2401.10774](https://arxiv.org/abs/2401.10774) | Abstract checked for the multihead route | Author model plus prediction heads | Multi-token heads | No matched local measurement | Records a trained alternative; training-free does not imply immunity to distribution bias |
| [Official Spec-Bench](https://github.com/hemingkx/Spec-Bench), data/spec_bench and Readme | Official repository description checked; actual category counts computed from frozen manifest | Original480 inputs,13 local categories | Evaluation platform/data source |480 inputs are not independent direction draws | Repository shows Apache-2.0; this is not a separate audit of every constituent dataset's terms |
| [Specified HF source and audit](research/esp-third-party-reproduction-2026-10-15.md) | Existing source/API and real mathematical audit reviewed through its report | Originalq121,8B BF16, three precheck budgets | Unchanged third-party API, not author official; CLI conditions differ | Six forwards, first-token gate failure, no validτ | Attempt stopped; neither official nor full480 reproduction, and no algorithm repair/new method |

The selected route remains the existing frozen-model mask comparison, with narrower claims about output-space superiority and a unique template direction. Future Lens, LatentMAS, smoothing and Kalman references in v6 remain historical leads: all underlying texts, theorem assumptions and novelty claims were not rechecked here. Literature claims about losslessness or throughput do not replace the project's actual gate or its no-timing boundary.

## III. Research questions and hypotheses

| Question/hypothesis | Conditions and falsifiable observation | Current assessment and next evidence |
| --- | --- | --- |
| R1 (highest priority): do8B paperτ values reproduce? | Compare the principal Table1 targets under Qwen3 single-mask settings; a material gap is not a successful numerical reproduction | Self-implementation gap, unverified author entry point, failed third-party precheck; not evidence the original ESP method is ineffective |
| H1 (completed/archived): output orders0/1 outperform input order1 here | Same480,4B BF16, budget/tree/stopping; both outputs weaker overall fails to support H1 | Full main matrix does not support H1; the output velocity term lacks stable joint benefit; see SectionIX |
| H2 (archived/deferred): gains require the specific e2−e1 direction | Random directions with clear gains contradict the word “only”; failure of all three would merely be compatible with that explanation | All three predetermined random cells beat zero; random2 also beats template. This does not show arbitrary directions work |
| R2 (archived/deferred): how does content initialization behave? | Preserve original inputs and full-prompt mean, changing only the native velocity-initialization suffix | Full480 content is complete: pooled tau1.364829262 versus zero1.422420206. Its norm is not separately matched, so this is not a pure direction-effect comparison |

### Table1 gap attribution and candidate controls (historical, before12:44)

> The PR40 candidates and VRAM gate below are historical. G-M/G-L/G-P and official-source searches are no longer the next priority. The magnitude scan and host-RAM admission in current scope govern.

Latest authorization permits existing-method attribution controls when actual GPU admission passes, even while author code is missing. No repeated source searches, new methods or prompt changes. Numbers come from the [CPU input/accounting report](research/esp-table1-cpu-contract-reconciliation-2026-10-16.md), [gap ledger](research/esp-table1-gap-analysis-2026-10-16.json) and [current attribution/admission JSON](research/esp-table1-attribution-hypotheses-2026-10-16.json).

| Verified observation | Conditional explanation and remaining limit |
| --- | --- |
| BC30/60 pooled1.563378/1.647301: gaps−0.066622/−0.092699 versus1.63/1.74 | Local results are valid but below target; this does not invalidate author ESP |
| Prompt weighting adds0.029855/0.022197, still leaving0.036768/0.070502; between-category call terms−0.029186/−0.022591 | Accounting could explain this component if authors used prompt means. Their actual weights are unknown; do not replace the primary metric or claim established paper-gap causality |
| All480 inputs rerender exactly in four cells without filtering;320/323 BC30/60 early outputs end in EOS | Excludes local cross-cell input drift, missing examples and context termination, but not differences from author qids/template/thinking/tokenizer/stopping |
| Existing-prefix100 pooled1.564682/1.647816; tail accounting effects0.002248/0.003039 | Checked prefix/clipping changes do not close the gaps. This is not new100-token generation, and including prefill lowersτ |
| Repeat-root fallbacks260/256 over50712/48041 calls, affecting111/115 prompts, all nonterminal | Optimistically adding one token per event at unchanged calls adds0.005127/0.005329, reaching only1.568504/1.652630. This is neither actual pruning-off gain nor a global trajectory bound |

Replay original Runs with`python tools/analyze_esp_repeat_roots.py`; all counts must match the current JSON. The [read-only analyzer](../tools/analyze_esp_repeat_roots.py) and terminal/contract-counterexample regressions are retained.

Repeat-root counts use original traces where the first verification-committed token equals the current root and accepted_nodes is empty; local pruning excludes this parent-repeat candidate. The observed local exclusion cost cannot by itself account for the gap at fixed calls. Changes to tree/numerics/later state could alter full paths. Mean updating, candidate coverage and missing author contracts remain unisolated; attribution is not unique.

Use only existing implementation/method switches. All executable controls are **non-author-official**, with new Study/version/Run, the original480,8B BF16/eager,T0,seed123,one mask and originalλ0.1/tree/EOS/argmax contract except for the declared factor. Reuse completed BC30/60 updating-mean baselines. Report per-prompt accepted/calls/τ win/tie/loss alongside output lengths; independent AR remains diagnostic.

| Hypothesis/priority | Exact contrast and prediction | Falsifier / limit | Readiness |
| --- | --- | --- | --- |
| G-M: mean updating harms8B candidate coverage (first) | Only set`spec.mask.update=false`, retaining mean_prompt initialization; two BC30/60 cells in order30→60 against existing updating means. Predict higher pooledτ at both budgets with supportive accepted/call pairs, not only longer outputs | Failure to improve both rejects the cross-budget version. Reaching1.63/1.74 is still a frozen control, not author Eq5λ0.1 reproduction. Do not change defaults or select only favorable cells | Native switch; no scientific-code changes; separate Study freeze and capacity admission required |
| G-L: actual100/256 stopping explains the gap (secondary) | Only set`eval.max_new_tokens=100`, retaining EOS; BC30/60. Expect values near observed-prefix1.564682/1.647816; compare actual counts/paths | Remaining below target rejects closure by this local stopping control. Cross-Run nonexactness stays diagnostic; no operator study. CPU evidence predicts low information gain | Native parameter; after G-M only for a protocol need, not idle-GPU occupancy |
| G-P: repetition pruning substantially harms acceptance (low) | Only set`spec.tree.pruning=false`; same budgets/node counts, BC30/60 versus original pruning=true | Observed direct opportunity is only≈0.005. Larger full-path effects require reporting propagation rather than claiming direct repeat-token recovery or author-gap causality | Native switch; CPU does not support it alone closing the gap; not queued |
| G-A: author/local implementation contracts differ (priority when available) | Obtain original author repo/commit/entry, align qids/token digests/budget/mean/stopping/counts, then run Table1 | Missing entry or differing contracts prevents claiming same-protocol author reproduction; never rewrite missing source | Still missing; no loop over unchanged searches |

G-M is pending admission, not running. Do not make/launch before admission, avoiding implicit calibration or reused Run IDs. Once admitted, first freeze a separate Study PLAN with both cells and stop rules, then reuse the original untimed/argmax adapters and native make/launch in one serial queue. Mathematical/source-gate failure retains evidence and stops expansion; AR mismatch does not trigger reruns. Retain resource failures without preemption. G-L/G-P are not automatically expanded into matrices; bias/random/Kalman and operator mechanisms remain deferred. Native configuration validation passed on all six declared factor/budget combinations without constructing inputs/models or creating Runs.

Current admission JSON:24455 MiB total,21518 MiB free,no other GPU Python. Historical whole-device sampled peak22093 MiB includes background;73 parent-only samples span1962–2413 MiB. Subtracting the lowest background gives a20131 MiB incremental planning estimate; plus conservative2 GiB(2048 MiB) requires22179 MiB, a661 MiB shortfall. Capacity admission failed; GPU execution is zero. Do not compare whole-device peak directly with free memory and double-count background. This is not an isolated process/allocator measurement or guaranteed capacity. Recheck actual GPU use and obtain an isolated peak if needed; CPU processes do not block admission. Do not shorten prompts or reduce data to pass the gate.

The contribution sought is traceable reproduction boundaries and comparisons of existing methods/specified controls, not a new algorithm or universal optimum. Literature motivates the route; data/model bound inference; algorithm choices define treatments; metrics and paired experiments test the questions. β, state definitions, output paths and norms remain confounders: input/output differences are not attributable solely to the choice of space.

Current CPU findings: single-mask BC30/60 remain0.066622/0.092699 below the paper. Prompt-average or six-source equal weighting still misses the targets; end-clipping accounting contributes only0.002248/0.003039. Author task counts and a complete protocol are missing, so the gap cannot be uniquely attributed. See the [gap report](research/esp-table1-gap-analysis-2026-10-16.md).

## IV. Data

Retain existing `datasets/spec_bench/question.jsonl` from the official Spec-Bench project, without replacing it with a fresh upstream download. SHA-256 is `7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21`. Source, input IDs, EOS and order contracts are in the [4B prompts manifest](../studies/qwen3_4b_bf16_five_methods/docs/prompts_manifest.json). The historical immutable upstream commit is not independently confirmed; current GitHub HEAD does not substitute for it. Constituent-license and pretraining-contamination audits are incomplete.

The unit is one original first-turn prompt,480 total in original order, first/last question_id121/90. math_reasoning, qa, rag, summarization and translation each contain80; coding, extraction, humanities, math, reasoning, roleplay, stem and writing each contain10:13 categories. Preserve the original non-thinking chat template, all questions and stopping rules; no rewriting, truncation or template tuning. The original4B precheck records a maximum input length1500. Pair all480 and retain category results.

There is no training/fitting split. This frozen evaluation set has been repeatedly inspected; it is not a fresh independent holdout for selecting the best direction seed. Full duplicate/contamination identification has not been done. Every Run checks identities, order, input IDs, model content and stopping settings; tokenizer IDs are not exchanged across models without verification. Raw tokens/traces, full Runs, weights and caches remain local. Git holds manifests, summaries and evidence links.

## V. Models

Use existing ModelScope `Qwen/Qwen3-4B`: qwen3,36 layers, hidden size2560, vocabulary151936. Each Run checks actual tied input/output storage rather than trusting the configuration. Inputs are frozen IDs and masks; outputs are target logits, final hidden states and committed tokens. Weights, forwards and final masks use BF16/eager/T0. Native output-EMA/norm internal FP32 states are retained: this is not a claim that every internal calculation is BF16, or an additional FP32-model experiment.

See [modelscope_snapshot](../studies/qwen3_4b_bf16_five_methods/docs/modelscope_snapshot.json) and the [paired evidence](../studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json) for actual dtype, tied storage, model_digest and runtime. Fourteen model-file contents have been checked; historical remote revision identity remains incomplete. Weight licensing was not reverified this round and is Unverified, not inherited from this repository's license. Recorded environment is torch2.11.0+cu130, Transformers5.12.1 and RTX5090 D v2; finer details are Run-owned. Installation pins verified Transformers5.12.1; it does not adapt5.20's new cache API or replace the running environment.

Qwen3-8B belongs only to the ESP Table1 reproduction. 0.6B and4B FP32 remain separate historical archives. Weights, tree and stopping rules are fixed within a budget. Only the five existing methods or requested controls vary; no additional trained drafter is introduced.

## VI. Algorithms

Reuse [SpecMTP](../src/spec_mtp/structure/algorithm) and the native RPipe Study lifecycle, without another experiment framework. These are the actual local definitions: the user-defined frozen-mean ESP baseline is distinct from the paper's updating mean.

| Existing method | Mask/state definition | Fixed parameters and boundary |
| --- | --- | --- |
| ESP frozen mean | Original prompt embedding mean, frozen throughout generation | `update=false`; user-specified baseline |
| Input order0 | Full prompt mean, then `mean ← mean + λ(e−mean)` on commits | λ=0.1; updating mean; the direction-control zero cell reuses it |
| Input order1 | `mask_j = mean + γ j v`; native EMA of embedding differences updates v | β=0.9999,γ=1,λ=0.1,order0 level; normalize/extrapolate_ema disabled |
| Output order0 | Native final-normalized hidden/proxy state predicts u; mask is `r u/‖u‖`, with r the mean vocabulary-embedding norm | order0; existing owner/history semantics; no inverse embedding map |
| Output order1 | Add native EMA velocity to the same hidden prediction, then apply the same norm mapping | order1,β=0.9,proxyβ′=0.99; other state definitions are not identical to input order1 |
| Independent AR | Tokenwise greedy generation with the same model, inputs, seed and stops | One shared480 reference, not repeated inside every Spec Run |

Tied models use no inverse mapping. Native proxy updates follow root/candidate owners and accepted history, without teacher correction or extra verification forwards. Tree/pruning, KV cache, bonus and EOS/length clipping remain native. Accepted counts are before clipping; committed counts reflect actual clipping. No training loss, gradients or hyperparameter optimization is introduced. Full actual settings are recorded in each evidence cell.

Requested controls change only input velocity construction: template fixes `v=e2−e1`; random0/1/2 use independent CPU Generator seeds0/1/2, normalize a random vector in FP32 to the same prompt-template-difference norm, cast to BF16 and keep v fixed during generation; zero sets v=0. Content initializes the original EMA on the suffix starting at the first user-content token, preserves the full-prompt mean and continues native updates. That suffix includes the original trailing template; inputs are unchanged and its norm is not separately matched. Fixed template is not native input_first, which keeps updating velocity.

Current ESP full480 and five-method results are local self-implementations/extensions, not author-code reproductions. The specified HF dataset `Harshvardhan-Mestha/esp-reproduction-artifacts` is pinned to `ad76f8da29f3bfc7faa20488193c6aa83a9f708c`; eight original files are retained in [reference/ORIGIN](../src/spec_mtp/structure/algorithm/reference/esp_reproduction_artifacts/ORIGIN.json). Its missing LICENSE remains Unverified. The external audit called the unchanged native API, not the CLI; deviations and failure are in the [third-party report](research/esp-third-party-reproduction-2026-10-15.md). It is not repaired into a new baseline.

Configuration/source changes require a new Study version/Run; successful historical Runs are not re-signed. Direction protocol `fe59d35` and recovery protocol `0e7c46f` are in [PR22](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/22). Recovery changes only real-GPU resource refusal into same-Run waiting at prompt boundaries. Original failures and105 complete prompt-executions of wasted work are retained, never stitched into a valid480.

## VII. Metrics

| Role | Definition, unit and direction | Aggregation | Implementation/criterion |
| --- | --- | --- | --- |
| Primary pooledτ | Σdecode committed / Σdecode calls, tokens/decode call, higher is better | All480; excludes prefill's first token | Native raw plus shared CPU readout; report each predetermined budget without choosing the favorable aggregation |
| Two supplementary aggregates | Equal prompt-meanτ; equal mean of13 category-pooledτ values | Both separate from pooledτ | Recomputed from the same result; not asserted to be the paper's task weighting |
| Accepted/main calls | Higher accepted is better; fewer prefill+decode calls is better | Separate per-qid W/T/L and total deltas | No forced joint winner; different output paths/lengths are not equal-output speedups |
| Paired promptτ | Integer cross-products compare two ratios exactly | Directions versus template and existing five methods | Matching480 identities/order required; retain all losses |
| Hard correctness gate | Every committed token equals the correct-owner argmax of that verification forward, including prefill | Full token coverage plus owner/hash chains | All must pass; source, nonfinite, count or owner failures cannot be replaced by AR diagnostics |
| AR diagnostics | Whole-sequence exact rate, each first divergence, output length | One shared independent AR reference | Reporting only; no native summarize/process blocker or operator-level numerical study |
| Resource work | Model calls, committed tokens, work consumed by failed cells | Successful and failed work separate | Scheduling/BC60 decisions, not latency/throughput; zero timing fields are placeholders |

Every number identifies a file, cell and Run. Author values, local measurements and unmeasured items are distinct. No retrospective significance threshold or numerical “acceptable gap” is invented. The T0 same-forward audit does not establish arbitrary-temperature distributional equivalence or token equality across independent forward shapes.

## VIII. Experimental design and order

| Experiment/question | Controls/treatments | Fixed conditions/repeats | Budget, state and interpretation |
| --- | --- | --- | --- |
| E1: ESP Table1 source and numbers | Author-source check, self-implementation reference and specified third-party precheck separated |8B BF16/T0, original480; primaryBC10/1,30/1,60/1; supplementary60/2; λ0.1 updating mean | Four self-implementation cells completed but miss targets; author entry unverified; third-party stopped after precheck failure; the new nonofficial8B repeat is queued/running |
| E2: five methods plus AR | Six definitions share inputs; compare five masks withinBC30/60 |4B BF16/T0; two masks, naive dynamic/pruned tree, draft budgets9/19; seed123+index; max_new_tokens256/EOS; default reduction | One shared AR plus10 Spec Runs:5280 prompt-executions but only480 evaluation inputs;11/11 accepted |
| E3: requested full480 directions | Template,random0/1/2,zero,content; versus template and the five methods | E2 BC30 conditions; mean/tree/stops/inputs fixed; direction seeds predetermined, decoder seed unchanged | Five new full paths plus reused zero:2400 valid new prompt-executions; all six cells complete; recovery native process passed |
| E4: evidence/document integration | CPU recomputation, provenance/bounds, bilingual plan, PR/dev integration | Same raw data and existing tools; preserve successful/failed cells | No additional GPU matrix; formal E3 readout/plot/report complete; no directionBC60 this round, with reasons in the final report |

### A. Baseline reproduction and follow-up

See the [author-source check](research/esp-official-code-reproduction-2026-10-15.md), [8B execution evidence](../studies/esp_table1_bf16_full_argmax/docs/protocol_evidence_v1.json) and third-party report. No verified executable author entry is available; an arbitrary similarly named repository is not a substitute. If an author version is obtained later, align its data/model/native entry within Enmao's scope and disclose adaptations. Template adoption does not automatically launch8B again.

| Condition | Planned target/source | Local measurement/source | Gap/assessment |
| --- | --- | --- | --- |
|8B BC10/1 | Additional user-requested budget; no invented Table1 target |τ1.399047, protocol_evidence `cells` | Mathematical gate passes; no corresponding Table1 target delta |
|8B BC30/1 | Table1τ1.63 |τ1.563378, same file |−0.066622; self-implementation misses the main target |
|8B BC60/1 | Table1τ1.74; Qwen3 single mask follows AppendixG.2 |τ1.647301, same file |−0.092699; target not reproduced |
|8B BC60/2 | Predetermined supplementary control, not the single-mask primary setting |τ1.603729, same file | Cannot select this cell or another aggregation to claim the primary reproduction |
| HF third-party budgets | Mathematical precheck of the original API under BF16/T0/originalq121 | Six forwards; first-token argmax margins40.75/33.5/33.25 in the [audit JSON](research/esp-third-party-audit-2026-10-15.json) | No validτ or full480 run; nonofficial; negative result retained |

The reproduction gap limits conclusions about original ESP. Local4B pairs support comparisons only within the common self-implementation. Historical checkpoint revision, template/length/aggregation choices are incompletely aligned: they are protocol gaps, not proven causes. BF16-versus-independent-AR operator investigations remain stopped.

### B. Coverage and components

E3 is a requested explanatory control, not a new-method main table. Template/random norms match before BF16 casting, with actual norm ratios recorded afterward. Content changes initialization/subsequent state and is not norm matched. Fixed-v controls are separate from the continuing native input_first EMA. Four local windows remain historical analysis, never a substitute for480; final reporting retains categories and every paired loss.

### C. Execution and experimental selection

The original GPU queue ended naturally with exit0 after content completed. CPU source/argmax/native-process checks and the formal readout passed. Plan PR23 and result PR22 are merged into dev; their remote topics are deleted, with frozen E3 PLANs unchanged. Normal commits/pushes, dev merges and deletion of their merged remote work branches are authorized. No main merge, force push or local data/Run/branch deletion. Old PRs/issues are closed only after verifying status and relationships, not merely age.

GPU admission uses actual nvidia-smi occupancy and sufficient free memory, not unidentified CPU processes. Group-boundary handoffs happen after full groups finish; one real quantization interleave has completed, without interrupting SD groups. Brief resource conflicts wait at prompt boundaries in the same Run. Source/argmax/OOM failures remain real failures, never fixed by prompt changes. Successful template and old zero are not rerun.

This is a predetermined discrete comparison, not adaptive parameter search or prompt tuning. Three direction seeds are three fixed biases, not independent training repeats. Identical first template tokens make each seed's vector identical across480 prompts. Paired descriptions are useful, but do not establish confidence across models, training randomness or arbitrary directions. No new significance claim is made. Decide directionBC60 after fullBC30 using actual token/forward work and the remaining question; never discard losing cells or describe unrun work as complete.

<a id="current-execution-20261016"></a>

### New execution contract,2026-10-16: nonofficial Table1 repeat and bias-magnitude scan

This section preregisters new execution; consolidating old chapters does not reregister historical results. Admit GPU work only when actually vacant and measured peak host RSS plus2GiB is available. Never overlap quantization or kill another task. The13:02 quantization occupancy means pending resources, not started. Use independent Studies/versions/Runs and native RPipe/SpecMTP phases, preserving raw assets, failures and negative results. Cite this project-plan section and immutable Git commit/blob; do not create separate Study PLAN files or modify vendored RPipe.

A: new`esp_table1_nonofficial_revalidation`, existing Qwen3-8B BF16/eager,T0,original SpecBench480/non-thinking template/input IDs,seed123+index,256/EOS,mean_prompt with per-tokenλ0.1 updates,naive dynamic/pruned tree. Predeclare BC10/1,30/1,60/1 with draft budgets4/14/29, in that serial order. If8B measured host peak+2GiB cannot fit, report the limit and use the largest existing checkpoint with validated resource measurements; never label4B as8B. Reuse old complete8B as comparison; Enmao's12:44 instruction explicitly authorizes this independent new GPU repeat, not overwriting old cells or another CPU-only revalidation. Report reproduction of localτ, differences from paper1.63/1.74, three aggregations, per-qid counts/AR diagnostics/output lengths. No favorable filtering, metric selection or dtype changes.

B: new`qwen3_4b_bf16_bias_norm_scan`, original ModelScope4B content, BF16/eager/T0, original480/template/IDs,seed123+index,256/EOS,BC30/two masks/draft9,dynamic/pruned tree,meanλ0.1, originalβ0.9999,normalize=false,extrapolate_ema=false. Only existing direction control and native step_scale change: m_i=updating_mean+i*s*v. v is fixed e2−e1 or CPU Gaussian direction seed0/1/2 normalized to the template-vector norm, retained across commits; s∈{0,0.5,1,2}. Do not substitute native updating input_first for fixed template, introduce content directions or Kalman.

| Direction | s=0 | s=0.5 | s=1 | s=2 |
| --- | --- | --- | --- | --- |
| e2−e1 | Reuse old input_zero | New480 | Reuse old template | New480 |
| random0 | Same shared zero, not independent samples | New480 | Reuse old random0 | New480 |
| random1 | Same shared zero | New480 | Reuse old random1 | New480 |
| random2 | Same shared zero | New480 | Reuse old random2 | New480 |

Eight new cells, fixed order: template0.5→template2→random0/0.5→random0/2→random1/0.5→random1/2→random2/0.5→random2/2. Each requires480; never drop seeds or select winners midway. Bind reused0/1× to existing formal evidence SHA, with one zero baseline. Do not re-sign or count reused work as new GPU compute. Three seeds are fixed directions, not independent training repeats. Retaining e2−e1 distinguishes magnitude response from direction interactions.

Primary readout: each direction/scale pooledτ, plus prompt means and equal-category pooled means. Preserve both the arithmetic random-three mean curve and each seed; do not hide variation or recommend the best scale as a new method. Per-qid accepted/calls/τ win/tie/loss versus same-direction1× and shared zero, output lengths, per-depth attempts/accepts and first verification-block acceptance are reported. Observe initial v norm, effective scaled-bias norm and mean norm to verify the actual scaling. Separate first-block common-input acceptance/coverage from later divergent-path totals; do not call different-prefix effects same-position causality. Consistent responses across directions support magnitude dominance; interactions, seed spread or inconsistent curves remain counterevidence, not suppressed to claim direction independence.

Hard gate: every committed token, including prefill and actual EOS/cap-trimmed output, equals its correct owner's same-verification-forward argmax, with native source/input/count/owner checks. AR exactness/first differences remain diagnostic; norm cells reuse old AR rather than generate more. No teacher,warmup or timing; decoder timing fields are zero sentinels and resource timestamps are not TPS/SR. Retain source/math/nonfinite/OOM failures, stop the affected cell without automatic retries. Wait for foreign GPU work only at group handoffs/prompt boundaries, without preemption. Exit/release between models; one serial queue only. Pending is not zero or a tie.

### CPU preparation addendum,2026-10-16: norm endpoints and acceptance boundary

Follow the17:52 instruction: retain the originalBC60 worker and measured-host-peak-plus2GiB gate; use the wait for CPU work. The eight new4B cells remain after8B. The read-only entry verified30 Spec source files for five old endpoints,405083 stored argmax/forward-owner records and81880 shared AR tokens. Raw paths, formal source hashes and values are in the [endpoint checkpoint](../studies/qwen3_4b_bf16_bias_norm_scan/docs/reused_readiness_v1.json) and [preparation report](../studies/qwen3_4b_bf16_bias_norm_scan/docs/STUDY_REPORT.md). No make, model forward or new-cell acceptance occurred;0.5/2x remain unobserved.

For all480 original-input-plus-prefill prefixes with a first block, template/random0/random1/random2 versus zero have first-block accepted W/T/L of26/341/113,56/361/63,55/367/58,30/361/89. Original full-path tau remains1.507777653/1.495188134/1.464349086/1.523884714 versus zero1.422420206. These constrain a universal initial-block-gain explanation. Prompt W/T/L is not a mean accepted-count difference; subsequent divergent paths, call weights and output lengths require separate reporting. No new method or operator study follows. Bind magnitude scans to E3 fixed-v endpoints; retain E2 native updating input-first1.51195758 separately.

The reader now explicitly requires all eight preregistered index settings in order, one Run per cell and exact full-spec agreement. Missing/duplicate/reordered cells, multiplier/beta drift and multiple Runs are rejected. --check-reused only inspects old endpoints, always reports complete=false/new_cells_inspected=false and draws no new curve; normal mode retains native source/input/argmax/process gates. Fifty CPU tests passed in total, including18 norm tests and the related readout/pairing/documentation checks. New cells remain unmade; immutable plan commit/blob, scientific code/defaults and data/prompts are unchanged.

### Saved-evidence CPU audit addendum (2026-10-16; no new experiment)

The existing `tools/analyze_4b_geometry.py --audit-raw` rechecked all66 native files across the11 E2 five-method-plus-AR cells. All809118 saved Spec tokens passed the mathematical audit; the81880 AR tokens are reference data only. All ten prior depth/position/trimming summaries and six tau decompositions match exactly: this is replication of existing accounting, not a new mechanism finding. [Audit evidence](research/4b-paired-raw-audit-2026-10-16.json); [interpretation and command](research/4b-mask-geometry-2026-10-15.md#八2026-10-16-cpu完整原始审计与首块对账). Each of20 first-block comparisons covers480 shared prefixes. Output order0/order1 tie on all480 first blocks at both budgets. Relative to input-first, either output method wins/ties/loses114/339/27 atBC30 and131/306/43 atBC60 (output minus input). These local wins coexist with lower full-path tau; do not extrapolate them to whole-path gains or infer zero prompt velocity from the output-order ties. Shared prefill proposals and later velocity propagation have different timing.

The original BC60 worker66500 continues waiting on the unchanged host-RAM gate; retain processes and data. This CPU audit does not accept new Runs, load a model, change prompts/defaults, measure speed or investigate operators. Switch the frozen execution tree for the eight4B magnitude cells only after the original8B queue exits naturally and its final acceptance completes. Missing cells remain pending; no new methods or fabricated curve.


### 19:15 RAM and idle-GPU scheduling addendum

The latest authorization retains the original8B measured-peak-plus2GiB gate and permits the registered eight4B magnitude cells to use an idle GPU if safe working-set reclamation remains insufficient. The earlier requirement to wait for the8B queue to exit before make is superseded for this fallback. First calibrate4B capacity on the unchanged longest fixed input, then admit formal cells using the measured4B peak plus2GiB. Only reversibly suspend this session’s verified RAM-wait BC60 worker, resume in finally, and give8B priority between complete4B cells when its original admission becomes available. Scientific settings,480 inputs and original preregistration remain unchanged; capacity probes provide no formal tau result. [Protocol and observations](research/sd-memory-fallback-2026-10-16.md).


The4B capacity probe completed: unchanged longest inputq295/1500 tokens, all256 capacity-token audits passed. Sampled process-tree peak8808693760 bytes gives a10956177408-byte formal4B gate. BC60 was resumed in finally; this is not a formal magnitude-scan tau result.

<a id="paired-followup-20261016"></a>
### 19:46 preparation for an independent4B five-method-plus-AR repeat

The original E2 matrix already completed11/11 cells. Do not invent missing cells or re-sign historical Runs. The latest19:46 instruction authorizes preparation of `qwen3_4b_bf16_paired_revalidation`, with new version/Run identities, after the current eight magnitude cells. Retain the original480 input/token-ID records, ModelScope Qwen3-4B snapshot, BF16/eager/T0/256/EOS and seed123+i. Preserve E2 exactly: ESP frozen mean (lambda0,update=false), input-zero updating mean (lambda.1), input-first mean+gamma (beta.9999), output-zero final-hidden norm mapping, and output-first native hidden EMA velocity (beta.9,beta-prime.99). BothBC30/60 use two masks and dynamic pruning; five methods per budget plus one shared independent AR give11 cells. This frozen-mean ESP differs from8B Table1's lambda.1 updating mean.

CPU queue: reuse the untimed adapter, native token validator and load_cell/summarize/compare_rows. Preflight merged configurations, all11 ordered cases and immutable input/weight/method hashes. Prepare20 method pairs with accepted/calls/tau WTL, lengths and AR first divergence. Every committed Spec token must match its own verifier-forward argmax; independent T0 AR is diagnostic and differences do not fail Spec acceptance. Missing, failed or incomplete raw evidence stays pending, without formal tau. Run CPU tests and preparation/make only; do not start another GPU queue early.

GPU queue: admit the originalBC60 first whenever its host-RAM gate becomes available; otherwise finish each current magnitude cell serially. Only after all eight cells succeed natively and pass per-cell accounting may the11-cell repeat take the GPU. Reuse measured4B BC30 maximum-input capacity plus2GiB host reserve and actual GPU nonpreemption. BC60 shapes have no separate peak guarantee: sample every Run, retain the per-prompt2GiB reserve, wait or preserve failures rather than lowering the gate or terminating work. The11 new cells are explicitly an independent repeat, not repairs to old results; prompts/methods remain fixed. Keep local thematic commits and open a milestone PR into dev. A read-only per-cell magnitude ledger checks native/input/source/token evidence and three tau aggregates; no complete curve before all eight.


CPU preparation update: native declaration expansion and the unmade readout preflight passed (0/11,complete=false). All73 combined regressions and Ruff passed; old11-cell results do not fill new cells. Read-only watcher75696 uses tools/audit_4b_norm_queue.py; it records no tau before a new native cell completes, then retains three aggregates and file hashes using the strict readout. The19:53 display-whitelist/NVIDIA-type module is used only by the new Study; active8B/norm sources remain frozen. The local probe excludes ten display contexts and defers for the current known pure-C Python context. Branch review found no further deletable topics; active/unmerged topics remain.

## IX. Execution records and conclusions

| Record | Actual evidence | Outcome/boundary |
| --- | --- | --- |
| E1 self-implementation |8B protocol_evidence and [stage report](research/sd-stage-report-2026-10-15.md) | Four mathematical audits pass,318830 tokens;BC30/60 single-mask targets missed; not author-code reproduction |
| E2 main matrix |[paired_evidence_v1.json](../studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json), `cells`/`pairs` |11/11 native acceptance;4800 Spec prompt-executions and809118 audited tokens;τ below |
| E3 complete evidence |[direction_evidence_v1.json](../studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json), [final report](research/4b-full-direction-controls-2026-10-16.md), [PR22](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/22) |`complete=true`; all six cells complete480. Original resource-failed Study remains passed=false; recovery Study passed=true; no failed result is resigned |
| Source/third-party negative evidence | Author-source report, third-party ORIGIN and audit | Unverified source discovery is not proof of no release; no fabricatedτ or8B rerun after failure |

| Accepted main method |4B BC30 pooledτ |4B BC60 pooledτ |
| --- | --- | --- |
| ESP frozen mean |1.402895 |1.499079 |
| Input order0 |1.422420 |1.519109 |
| Input order1 |1.511958 |1.636340 |
| Output order0 |1.433178 |1.528922 |
| Output order1 |1.435540 |1.529836 |

Each value comes from E2 `cells[budget/method].totals.pooled_tau`. Shared AR is diagnostic and is not assigned a fabricated draft-acceptanceτ. All per-prompt accepted/calls pairs, AR divergence and lengths remain in the source file; rounding here does not replace them. Input order1 is strongest overall in this main batch. The output-order1 increment is small; one favorable count alone does not establish joint gains or equal-output speedup.

E3 valid pooledτ: template1.507777653,random0=1.495188134,random1=1.464349086,random2=1.523884714,zero1.422420206,content1.364829262, all in the linked formal six-cell evidence. Random2 versus template has pairedτ267 wins/47 ties/166 losses. All three random seeds exceed zero, contradicting the claim that gains here require only e2−e1. Per-prompt counterexamples, different output lengths and unmatched content norms remain. This is neither a recommendation to select the best seed as a new method nor proof every direction works. Zero reuse and failed resource work stay explicit; recovery success cannot turn the original failed Study into a passed one.

No directionBC60 matrix is added this round: the five valid new cells already consumed2400 prompt-executions,405073 tokens and276523 model calls, and all three random directions falsify direction uniqueness in this batch. BC60 would test cross-budget generalization rather than resolve the content-norm confound. Failed complete partial work separately records20977 tokens/13854 calls and is never stitched into valid cells. CPU regression and PR22/dev closeout are complete. Current execution follows ChapterVIII: independent nonofficial8B repeat, then4B magnitude scan. Preserve the existing gap ledger; no renewed source search, prompt/budget/aggregation tuning,0.6B/operator/Kalman or new-method expansion.

Historical source-audit chronology (superseded by12:44; not current work):

Latest CPU source refresh: the [official-path recheck](research/esp-official-recheck-2026-10-16.md) confirms unchanged published Table1 targets1.63/1.74. Ten Figure8 labels and eight sampling-table labels do not establish the Table1 inclusion/weighting contract. Keep all480/13 local categories. No runnable author entry was verified within the stated access limits. This completed source audit is no longer an execution prerequisite.

The [author repository/commit/attachment follow-up](research/esp-author-attachment-followup-2026-10-16.md) completed this bounded check: unchanged homepage commit, zero hits in three exact code queries, and six forum-ID hits yielding no verified author implementation. Three conventional OpenReview attachment candidates returned403 without code. No verified author code was obtained. Source searching has stopped and the current repeat is explicitly nonofficial.

The [Table1 protocol gap register](research/esp-table1-protocol-gaps-2026-10-16.md) separates local facts, missing author artifacts, CPU reconciliation and official launch triggers. This round read a public author post without obtaining a verified code entry; its paper short-link did not resolve, and OpenReview was not refreshed. PR36 rechecked old nonofficial full480 evidence. The newly authorized repeat uses separate version/Run identities without overwriting old results.

The [conference/HF channel follow-up](research/esp-table1-protocol-gaps-2026-10-16.md) adds the author-linked HF account scope: each public models/datasets/spaces API returned zero repositories, and the visible conference-page text supplied no code entry. Private/organization assets and recording content were not reviewed. Retain this historical coverage record without periodic refresh.

The [P1 input/P2 accounting CPU reconciliation](research/esp-table1-cpu-contract-reconciliation-2026-10-16.md) is complete: all fixed480 first turns rerender token-exactly in all four cells, without filtering (21–1500 input tokens;80 source records have multiple turns, but only the first is used). The320/323 BC30/60 outputs below256 end in EOS. Pooled-minus-prompt-mean differences of−0.029855/−0.022197 mainly reflect between-category call weights. Existing-trace first100-token pooled τ is1.564682/1.647816, below1.63/1.74; this is observed-prefix accounting, not a new100-token generation. Preserve the full-path primary results. Author qids/template/thinking/revision, actual100/256 stopping, per-prompt counts and task weights remain missing, so attribution is not unique. Attribution remains nonunique. Next execute the frozen independent repeat and magnitude scan with unchanged inputs/stopping/aggregation. Reconcile author artifacts only if new evidence becomes available; no active repeated search.

### Overnight and current merged-PR ledger (UTC; all merged into dev)

Source: retrieved GitHub PR metadata. Author-search PRs are historical only; searching stopped under the12:44 instruction. PR36 rechecked existing full480 evidence rather than running a new GPU matrix. The independent repeat is listed in the snapshot above.

| PR | Merged UTC | Merge commit | Scope/evidence |
| --- | --- | --- | --- |
| [#22](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/22) | 2026-10-15T20:02:48Z | `d792ca578` | Six direction cells/full480, three random seeds and recovery limits |
| [#23](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/23) | 2026-10-15T19:38:57Z | `b00ecc0a1` | Standards and bilingual project plan |
| [#24](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/24) | 2026-10-15T20:17:26Z | `617278abb` | Status, GPU handoff and remote-topic inventory |
| [#25](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/25) | 2026-10-15T20:37:42Z | `0dc6568df` | Existing Table1 gap and CPU evidence priorities |
| [#26](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/26) | 2026-10-15T23:52:56Z | `c467c21b9` | Published Table1/source audit |
| [#27](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/27) | 2026-10-16T00:02:06Z | `3eac53c9e` | Author-path readiness checklist |
| [#28](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/28) | 2026-10-16T00:19:34Z | `8e2132b33` | Author commit/attachment access scope |
| [#29](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/29) | 2026-10-16T00:49:08Z | `635a9a589` | Coauthor/history refs audit |
| [#30](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/30) | 2026-10-16T01:08:43Z | `1e734ff61` | Organization README candidates |
| [#31](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/31) | 2026-10-16T01:31:50Z | `eed894989` | Release/arXiv source audit |
| [#32](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/32) | 2026-10-16T01:51:52Z | `a9e6a6653` | Source versions/public assets |
| [#33](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/33) | 2026-10-16T02:21:33Z | `d5f75ef02` | Publication channels/OpenReview access limits |
| [#34](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/34) | 2026-10-16T02:36:24Z | `a3acf6795` | Identified-author public branch/release refs |
| [#35](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/35) | 2026-10-16T02:53:18Z | `c4f589a44` | Organization code-search leads |
| [#36](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/36) | 2026-10-16T03:26:47Z | `6ce71665e` | Nonofficial four-cell full480 raw recheck; no new GPU matrix |
| [#37](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/37) | 2026-10-16T03:57:26Z | `776a2b4a2` | Protocol gap register |
| [#38](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/38) | 2026-10-16T04:18:15Z | `8d0f49cc4` | Public author HF/conference channel scope |
| [#39](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/39) | 2026-10-16T04:37:41Z | `437954bbe` | P1 input/P2 count and aggregation CPU reconciliation |
| [#40](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/40) | 2026-10-16T04:57:05Z | `aed4eee23` | Repeated-root pruning opportunity and falsifiable hypotheses |

Reading depth is in ChapterII; all60 legacy references, theorem conditions and unverified citations remain in the v6 ChapterXI archive. Consolidation is not a new full-text review. Remaining gaps include author input/stopping/tau weights, nonofficial-to-paper implementation differences, finite random seeds and content-magnitude confounding. Next: the independent repeat, then the preregistered eight norm cells; no renewed author search/operator chain.

New evidence: the [BC10 full480 repeat](../studies/esp_table1_nonofficial_revalidation/docs/STUDY_REPORT.md) reproduces the old local result prompt by prompt; the other budgets are unfinished. PR44 byte preservation, PR45 norm preparation and PR46 readout/checkpoint support current execution, without author-official or new-method claims. No rerun for AR disagreement, prompt/operator changes or re-signing of the immutable085722e plan reference. Only these two bilingual project plans remain.

## X. GitHub use and plan changes

This is the current project-level plan, with a chapter-matched English translation. Historical standalone Study plans are consolidated in ChapterXI and their pre-make contracts remain verifiable at fixed Git commits; results/figures live in their STUDY_REPORT/evidence. Raw tokens/traces, full Runs, weights and caches stay out of Git. Native records retain code/source/input/model identity, actual configuration/environment, seed, command, RunID, failures and artifacts. Do not recreate a historical Run ID or re-sign successful evidence.

| Date | Change/reason | Affected scope |
| --- | --- | --- |
|2026-10-16 | Rewrite under Enmao's new ten-chapter template, add English, import pinned RPipe standards/templates and integrate targeted literature/current evidence | New default plan entry; v6 retained as history; not retrospective preregistration or a frozen-experiment change |
|2026-10-16 | Record author/self/third-party distinctions, argmax gate versus AR diagnostics, three directions and recovery | Narrow conclusions without expanding GPU budget or changing prompts |
|2026-10-16 | Pin installed Transformers5.12.1 after5.20 cache-API drift broke CPU CI | Installation reproducibility only; no scientific cache change or replacement of the live environment |

Plan PR23 and result[PR22](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/22) are merged into dev and their remote topics deleted; see the [cleanup record](research/remote-branch-cleanup-2026-10-15.md). [Standards README](research/standards/README.md) records originals/translations, source and SHA values. The historical60-reference bibliography is not claimed as fully reviewed this round. Author originals, official data entry and the local reports linked in the literature table form this round's references; factual claims are located beside their supporting sources.

Closeout: PR41 merged into dev at5ef98ba after both CI checks passed. The bilingual consolidation is complete. Cleanup removed27 local/one remote merged topics and retired one unused worktree registration while retaining all files;859 experiment-file hashes matched. The activeGPU tree remains. Open PR/issue queries were both zero. The [cleanup record](research/remote-branch-cleanup-2026-10-15.md) lists exact names and the roughly-every-five-merged-PR policy. List unmerged stale candidates before action; never delete experiment data.

Norm-scan preparation: scripts and9 CPU checks passed; not made/launched. Auto-review rejected an early make while8B was active because GPU/model preparation could overlap; the command did not execute. The exact command/reason is in the [approval record](research/remote-branch-cleanup-2026-10-15.md). Wait for natural8B queue exit, then prepare serially without a workaround.

### Consolidation validation and replay boundary

The full-name git mv is commit7c609f9. All60 standalone plans are consolidated before explicitly authorized git rm; original text/formulas and fixed Git identity remain. Navigation, report generators and live tool reads are relocated. RPipe provenance.include forbids paths outside a Study, so removed plan entries become project-path/section/historical-Git-blob metadata rather than invalid external includes. Metadata is frozen with study.yaml; native RPipe does not automatically read the central plan. New execution such as the running8B repeat explicitly verifies its pinned plan Git blob.

No historical Run is re-signed. Migrated declarations/tools are not the original execution source; complete old replay requires its original Git version. Historical plan hashes verify immutable Git bytes plus recorded line endings. The active8B job runs in an isolated worktree. No scientific src/spec_mtp, vendored src/rpipe, data, original result JSON or weights changed.

Validation:10 targeted navigation/60-section/historical-byte/third-party tests passed; ruff passed; mypy passed56 files. Full CPU-suite and actual remote CI status are recorded in the PR, without claiming unrun checks. Four byte-pinned standards retain their three known external DEM-reference paths, documented in their README and specifically exempted from clone-local link checking.

## XI. Consolidated subtopic and historical plans

All60 former standalone plans now have independent chapters here and in the Chinese master. Historical requirements, untested proposals, FP32/operator/Kalman work and author-search priorities are archival, superseded by Chapter I and the current execution contract. English originals are retained; Chinese originals are translated with their settings, gates, failure boundaries and formulas. These chapters are not new execution authorization. Immutable original bytes, SHA256, Git objects and line endings remain in the [consolidation manifest](research/plan-consolidation-manifest.json). Never re-sign historical Runs using the current plan.

- [docs/research/esp-table1-bf16-reproduction-plan.md](#legacy-docs-research-esp-table1-bf16-reproduction-plan)
- [docs/research/publication-plan.md](#legacy-docs-research-publication-plan)
- [docs/research/research-plan-v6.md](#legacy-docs-research-research-plan-v6)
- [studies/_template/docs/PLAN.md](#legacy-studies--template-docs-plan)
- [studies/esp_table1_bf16_full/docs/PLAN.md](#legacy-studies-esp-table1-bf16-full-docs-plan)
- [studies/esp_table1_bf16_full_argmax/docs/PLAN.md](#legacy-studies-esp-table1-bf16-full-argmax-docs-plan)
- [studies/esp_table1_bf16_full_argmax/docs/THIRD_PARTY_PLAN.md](#legacy-studies-esp-table1-bf16-full-argmax-docs-third-party-plan)
- [studies/esp_table1_bf16_numerics/docs/PLAN.md](#legacy-studies-esp-table1-bf16-numerics-docs-plan)
- [studies/esp_table1_bf16_preflight/docs/PLAN.md](#legacy-studies-esp-table1-bf16-preflight-docs-plan)
- [studies/esp_table1_bf16_qwen_protocol/docs/PLAN.md](#legacy-studies-esp-table1-bf16-qwen-protocol-docs-plan)
- [studies/esp_table1_bf16_reduction_check/docs/PLAN.md](#legacy-studies-esp-table1-bf16-reduction-check-docs-plan)
- [studies/esp_table1_bf16_shape_check/docs/PLAN.md](#legacy-studies-esp-table1-bf16-shape-check-docs-plan)
- [studies/input_mask_baseline/docs/PLAN.md](#legacy-studies-input-mask-baseline-docs-plan)
- [studies/output_space_4b_holdout/docs/PLAN.md](#legacy-studies-output-space-4b-holdout-docs-plan)
- [studies/output_space_4b_input_control/docs/PLAN.md](#legacy-studies-output-space-4b-input-control-docs-plan)
- [studies/output_space_4b_recheck/docs/PLAN.md](#legacy-studies-output-space-4b-recheck-docs-plan)
- [studies/output_space_4b_recheck_v2/docs/PLAN.md](#legacy-studies-output-space-4b-recheck-v2-docs-plan)
- [studies/output_space_cross_task/docs/CANONICAL_PLAN.md](#legacy-studies-output-space-cross-task-docs-canonical-plan)
- [studies/output_space_cross_task/docs/EXECUTION_PLAN.md](#legacy-studies-output-space-cross-task-docs-execution-plan)
- [studies/output_space_cross_task/docs/FIRST_DELTA_PLAN.md](#legacy-studies-output-space-cross-task-docs-first-delta-plan)
- [studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md](#legacy-studies-output-space-cross-task-docs-kernel-execution-plan)
- [studies/output_space_cross_task/docs/KERNEL_PLAN.md](#legacy-studies-output-space-cross-task-docs-kernel-plan)
- [studies/output_space_cross_task/docs/LOCAL_FACTOR_PLAN.md](#legacy-studies-output-space-cross-task-docs-local-factor-plan)
- [studies/output_space_cross_task/docs/OWNER_PLAN.md](#legacy-studies-output-space-cross-task-docs-owner-plan)
- [studies/output_space_cross_task/docs/PLAN.md](#legacy-studies-output-space-cross-task-docs-plan)
- [studies/output_space_cross_task/docs/RARE_EVENT_PLAN.md](#legacy-studies-output-space-cross-task-docs-rare-event-plan)
- [studies/output_space_cross_task/docs/SAMPLED_DEPTH_PLAN.md](#legacy-studies-output-space-cross-task-docs-sampled-depth-plan)
- [studies/output_space_cross_task/docs/SAMPLING_CONTINUATION_PLAN.md](#legacy-studies-output-space-cross-task-docs-sampling-continuation-plan)
- [studies/output_space_cross_task/docs/SAMPLING_EXPANSION_PLAN.md](#legacy-studies-output-space-cross-task-docs-sampling-expansion-plan)
- [studies/output_space_cross_task/docs/SAMPLING_PLAN.md](#legacy-studies-output-space-cross-task-docs-sampling-plan)
- [studies/output_space_cross_task/docs/STATE_EXTENSION_PLAN.md](#legacy-studies-output-space-cross-task-docs-state-extension-plan)
- [studies/output_space_cross_task/docs/STATE_FACTOR_PLAN.md](#legacy-studies-output-space-cross-task-docs-state-factor-plan)
- [studies/output_space_extension/docs/NORM_KERNEL_PLAN.md](#legacy-studies-output-space-extension-docs-norm-kernel-plan)
- [studies/output_space_extension/docs/NUMERICAL_PLAN.md](#legacy-studies-output-space-extension-docs-numerical-plan)
- [studies/output_space_extension/docs/NUMERICAL_PLAN_REVIEW.md](#legacy-studies-output-space-extension-docs-numerical-plan-review)
- [studies/output_space_extension/docs/OPERATOR_PLAN.md](#legacy-studies-output-space-extension-docs-operator-plan)
- [studies/output_space_extension/docs/OPERATOR_PLAN_REVIEW.md](#legacy-studies-output-space-extension-docs-operator-plan-review)
- [studies/output_space_extension/docs/PHI_RMS_PLAN.md](#legacy-studies-output-space-extension-docs-phi-rms-plan)
- [studies/output_space_extension/docs/PLAN.md](#legacy-studies-output-space-extension-docs-plan)
- [studies/output_space_extension/docs/ROUNDING_FACTOR_PLAN.md](#legacy-studies-output-space-extension-docs-rounding-factor-plan)
- [studies/output_space_followup/docs/PLAN.md](#legacy-studies-output-space-followup-docs-plan)
- [studies/output_space_followup/docs/PLAN_REVIEW.md](#legacy-studies-output-space-followup-docs-plan-review)
- [studies/output_space_prerequisite/docs/PLAN.md](#legacy-studies-output-space-prerequisite-docs-plan)
- [studies/output_space_small_matrix/docs/CROSS_PROMPT_PLAN.md](#legacy-studies-output-space-small-matrix-docs-cross-prompt-plan)
- [studies/output_space_small_matrix/docs/PLAN.md](#legacy-studies-output-space-small-matrix-docs-plan)
- [studies/output_space_small_matrix/docs/SCALE_PLAN.md](#legacy-studies-output-space-small-matrix-docs-scale-plan)
- [studies/output_space_small_matrix/docs/TRANSFER_PLAN.md](#legacy-studies-output-space-small-matrix-docs-transfer-plan)
- [studies/output_space_small_matrix/docs/WINDOW_PLAN.md](#legacy-studies-output-space-small-matrix-docs-window-plan)
- [studies/qwen3_4b_bf16_directions/docs/PLAN.md](#legacy-studies-qwen3-4b-bf16-directions-docs-plan)
- [studies/qwen3_4b_bf16_directions/docs/RECOVERY_PLAN.md](#legacy-studies-qwen3-4b-bf16-directions-docs-recovery-plan)
- [studies/qwen3_4b_bf16_directions_recovery/docs/PLAN.md](#legacy-studies-qwen3-4b-bf16-directions-recovery-docs-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/GEOMETRY_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-geometry-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_ACTIVE_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-active-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_DIRECTION_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-direction-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_EARLY_TEMPLATE_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-early-template-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_RECIPROCAL_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-reciprocal-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_TEMPLATE_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-template-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_WINDOW_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-window-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-plan)
- [studies/smoke/docs/PLAN.md](#legacy-studies-smoke-docs-plan)

<a id="legacy-docs-research-esp-table1-bf16-reproduction-plan"></a>

### docs/research/esp-table1-bf16-reproduction-plan.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [docs/research/esp-table1-bf16-reproduction-plan.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/docs/research/esp-table1-bf16-reproduction-plan.md); original SHA256 `3da5be5ca14baeb826ac4a85f581187363aa30d9a5190f94fae69b3b14296833`.

#### ESP Table1 reproduction — historical consolidated protocol
The 2026-10-15 author-code-only requirement and its unresolved OpenReview attachments are historical. The 2026-10-16 12:44 instruction supersedes that restriction: stop searching and use the existing stack explicitly as non-author-official. PR7 merged after CI at 8677376e1ac878a159a3095711eac43420d97bde; prior4B evidence was FP32 and 0.6B mechanism work paused.

##### Sources and unresolved contract
ESP arXiv v2, sections3.1/3.4/3.5/4, Tables1/3 and AppendicesD/G.2 were read, including the local PDF table on page7. Qwen/Qwen3-8B weights use ModelScope and8B is restricted to Table1 reproduction. Targets are BC30 tau1.63 and BC60 tau1.74; BC10 has no Table1 target. Table1 says256-token output/H100, whereas section4 says100/A100: use256 with EOS and record lengths. Do not relabel the old100-token static setup.
Precision, immutable model revision, thinking mode, truncation and task aggregation weights are unspecified. BF16 is user-required; non-thinking chat, first raw turn, no truncation and all480 are declared local choices. Record RTX5090Dv2/24GB and software differences; no S/R timing.
Official Spec-Bench commit fd2c1cd7d2201ef71db4c5f4e455008f017967bf, data/spec_bench/question.jsonl, has identical480 objects/order to the local copy. Remote SHA256=4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04; local=7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21. Preserve both; never drop difficult prompts.
ModelScope content verification (2026-10-15) checked16 files including five weight shards against frozen inventory and BC10 actual inputs. This proves then-current master content, not the paper's unspecified revision or original download channel; no old lock was rewritten.

##### Frozen scientific contract
AppendixG.2 says Qwen3 uses one mask for BC60, unlike LLaMA3's two. Table3 single-mask width[29] maps to60=2*(1+29); its numeric rows concern LLaMA and G.2 occurs near Figure8, so this is a model-specific protocol interpretation, not author confirmation of Table1. Main10/1,30/1,60/1; auxiliary60/2 remains separately reported, never selected after results. Old60/2 preflight is auxiliary; retain its bytes and reuse its measured cells by hash.
BF16/T0 only; initialize prompt embedding mean, then update m by lambda=.1 per generated token. Dynamic Top1 tree, cumulative path probability, repeated parent/child token replaced by next candidate, pruning=true, native naive attention/cache. Seed123, max256/EOS, no warmup, all480 first turns, thinking=false, max_prompt_len40960 with no exclusions. Draft budgets4/14/29, auxiliary19; record actual widths/BC. Algorithm1 B versus Eq7 is ambiguous; state the current Eq7/Top1 mapping without tuning budgets.
Correctness is complete verifier-argmax coverage of every committed token including prefill. Independent AR agreement/first divergence are diagnostics only. Default BF16 operators/reduction, no operator investigations; decoding clocks disabled and zero fields are placeholders.
Report pooled committed/decode_calls excluding prefill, prompt-mean tau and equal-category pooled mean; also calls including prefill, generated tokens, accepted drafts and tail trim. Compare raw differences and two-decimal values, without claiming the author's unknown aggregation.

##### Execution and stopping
Use native prepare/execute/collect/summarize/write/process with new Study/version/Run, per-file model hashes and actual BF16 checks. Old successful/failed Runs are not re-signed. Preflight selects the first prompt per category and all four cells; it is not full480. Full cells run serially with no concurrent model loads. Preserve OOM/protocol/source failures; wait or create a new Run after correction, no FP32/teacher/sample filtering or killing others. Report all missing cells as pending.
Historical four-cell preflight bound was13*4*2*256=26624 output tokens (one new60/1 cell6656 after reusing three old cells); full four-cell generation plus AR bound480*4*2*256=983040. Record capacity and other processes without performance comparisons. Current admission is the project-level measured-host-peak+2GB policy, superseding old capacity-only thresholds.

##### Subsequent4B paired contract
After8B, compare frozen ESP mean, input zero updating mean(.1), input first mean+gamma, output zero norm-mapped final hidden, output first final hidden+EMA velocity, plus one shared AR, on identical prompts/tree/budget/length/seeds. Record accepted/calls, per-prompt WTL and AR diagnostic. Frozen ESP here differs from Table1 updating mean. Tied models use hidden scaling directly, no inverse mapping. Freeze beta/gamma/internal precision and actual tied-storage checks; branch/depth/non-first-sibling support has CPU regression coverage. Historical statement “no4B BF16 results yet” is superseded by the completed paired evidence in the main plan.

##### Completed historical execution
Four480 cells in esp_table1_bf16_full_argmax succeeded after native gate revalidation: pooled tau1.399047/1.563378/1.647301/1.603729. BC30 gap=-.066622; BC60/1=-.092699; auxiliary60/2=-.136271. All aggregations remain reported; rounded paper targets were not reproduced. AR exact counts188/175/185/187 out of480 are diagnostic.
The local observer binds selected rows to their actual verifier forward/logits, including prefill; uncommitted trimmed tails are excluded. It adds no forward, arithmetic or tree change; failure stops execute, and AR has no observer.
Preserve old AR-gate failures by SHA. Only summarize/write/process were revalidated with a new validator SHA; execute_rerun=false, raw/source locks unchanged. Earlier v1 partial logs remain; v2 four jobs failed before decoding due to cache connection, then v3 used the same verified snapshot. No rerun merely for AR mismatch.
Reduction/shape reports and tests are archived; compact attention, linear/RMSNorm, reduction toggles and layer tracking are permanently stopped. Weights/cache/full Runs stay local.

##### Authorization
2026-10-15 standing authorization covers normal topic push/PR/dev merge and accurately described local checks replacing remote CI, subject to platform protection; no popups. Auto-review rejection records exact command/reason and continues unaffected work. No main merge, force push, spending or data deletion. Remote merged-topic deletion follows the later explicit branch-cleanup authorization. The native summarize/process gate follows verifier argmax, never AR exactness.

</details>

<a id="legacy-docs-research-publication-plan"></a>

### docs/research/publication-plan.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [docs/research/publication-plan.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/docs/research/publication-plan.md); original SHA256 `602c2cba074f8c78b0e2d54d731ca80745e6f018fedf3f96fdca747ef45719ad`.

#### Historical minimal publication candidate
2026-10-14: preparation only, no push/new PR/merge. The feature/output-space-prerequisite branch included subsequent research, so pushing it was not a minimal feature PR.
Local origin/dev was1b6d59c7f6507b75e3ff4e91b0a3142bcad5b8cd and origin/refactor/spec-rpipe1219a1d88f10c2faee345b8d9bf6a42fd1e0110e; these were local refs, not a new remote verification. Existing PR6 carried native flow, RPipe0.3.1 and source gates; CI/merge was not requeried.
A future clean branch must start from dev containing PR6, rather than attaching437f201 to obsolete dev or merging dependencies to bypass authorization. No branch/cherry-pick/rebase/history modification occurred here.
Candidate sequence: dependency PR6 through1219a1d; sole required feature437f20140ffc72a788814d16d3084d2677e66d60 (tied hidden zero/first static chain, rejection boundaries, four-cell acceptance). Deferred research:9ba5c3a q171;4a4ae8d q172–174;77fe411 q173 windows/direction/norm;6a953bc single-slot history replacement;7a02501 cross-prompt nulls;a42313d q175–178;f3b95ea canonical/strict replay q178 and q179–182; then provider isolation/drafts/publication preparation at actual HEAD. No new Study was run for the last item.
437f201 has33 files, five scientific modules, acceptance/tests/docs, parent1219a1d. Research tools depend on earlier helpers/Studies; review a complete dependency chain rather than selecting its last commit. A separate research PR should begin after the feature commit, retain minimal tools/tests/scalars and preserve recoverable original commits/local data; that reorganization was not performed.
publication-manifest.json pins each candidate Git blob/SHA/size and dependency, grouped source/tests/Study-contract/docs. Hashes refer to commit objects, not later worktree bytes. Exclude subsequent research, unexecuted cross-task Studies, full Run/shared/scripts/.tmp, weights/cache/selected tensors/profiler traces/user workspace/internal information/environment logs. Do not broaden33files with git add . or delete data for appearance.
Before actual push verify licenses, redaction, dev dependencies, cpu-checks/Branch flow and ownership. Historical270checks/36cells/install validation are437f201 evidence, not new remote CI.
pr-draft.md describes only the feature's final behavior and prerequisite negatives. Twelve-prompt exploration, kernel names and unexecuted protocols stay in research reports, without speed promises. Target dev, never main; preparation alone was not publication authorization.

</details>

<a id="legacy-docs-research-research-plan-v6"></a>

### docs/research/research-plan-v6.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [docs/research/research-plan-v6.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/docs/research/research-plan-v6.md); original SHA256 `cde92995ae885650648cb0efaa9f19c83090545c03dcb9b78a9b15a8ad9291fb`.

---
title: "denoising speculative decoding: Training-Free Kalman-Filtered Output-Space Masks for Parallel Drafting"
subtitle: "denoising speculative decoding Research Plan (paper-style v6 draft)"
date: "2026-10-13"
---

> Historical exploratory draft (2026-10-13), containing unverified theory, then-pending ideas and results under other conditions. It is not current execution or established contribution. Current [Chinese](<研究计划 Denoising Speculative Decoding.md>) / [English](<Research Plan Denoising Speculative Decoding.md>) plans supersede it; retaining this text does not reactivate Kalman or higher-order work.

#### denoising speculative decoding: Training-Free Kalman-Filtered Output-Space Masks for Parallel Drafting

> **Historical official project name**: **denoising speculative decoding** (lowercase); `denoise_spec_mtp` was still the repository directory, used only as a path.
>
> **Type**: paper-style research plan, not a paper.
>
> **Status labels**:
> - [MEASURED]: only Appendix C input-space/tree experiments;
> - [UNTESTED]: main method and Kalman variants have **no measurements**;
> - [CONJECTURE]: unproved statement;
> - [AUTHOR DECISION PENDING]: requires the author to decide.
>
> **Changes from v5**:
> 1. Unify naming and explain denoising (§1.5);
> 2. Add adaptive Kalman (§5.9): state-space/dimension factorization, prediction/correction, discrete algebraic Riccati steady-state gain, innovation-based online Q/R, proxy noise $R'>R$ and slot uncertainty;
> 3. Add Algorithm 3, complexity and numerical stability;
> 4. Add AR/difference/Kalman hierarchy (§5.10);
> 5. Add Propositions 8–10, H12–H13, E16–E18 and references [55]–[60].

---

##### Abstract

Most speculative drafters require training and depend on their training distribution; EAGLE-3 acceptance declines outside it [44][45]. ESP [12] is training-free and verifies the previous draft while drafting at mask positions in one forward, but uses input means without explicit predictive information.

The draft proposes **denoising speculative decoding**: regard adjacent final hidden states as a **noisy trajectory of next-token predictions**, filter order-$n$ state $(h,\nabla h,\dots,\nabla^nh)$ and extrapolate future masks. Three layers:

- **Fixed gains**: per-order EMA / α-β-γ filtering;
- **Main adaptive variant**: Kalman model $s_{t+1}=Fs_t+w$, $y_t=Hs_t+v$, covariance-recursive gains and per-sequence innovation-based online $Q,R$ [56][57][58][59], without training;
- **Proxy handling**: draft-prefix hidden observations use larger noise $R'>R$.

Accumulate state along each node prefix, predict only within the block and process all masks in one forward. Tied models use norm-scaled hidden states directly; untied models use LatentMAS ridge alignment [34].

Proposed theoretical results:

- Losslessness under arbitrary soft masks;
- v4 fixed EMA equals steady-state Kalman for a local linear trend without observation noise, with closed-form $\beta$ (Proposition 8);
- Steady-state gain is mean-square optimal among constant gains, yielding optimal $\beta$ (Proposition 9);
- Prediction covariance gives slot uncertainty (Proposition 10).

All experiments remain pending, including grid-searched fixed-$\beta$ controls, transfer without dataset retuning, cross-domain and trained-baseline comparisons.

---

##### 1. Introduction

###### 1.1 Background
Speculative decoding [1][2] reduces target forwards through cheap drafts and parallel verification while preserving the output distribution. Drafter classes:

- **Trained**: Medusa [3], Hydra [7], EAGLE-1/2/3 [4][5][6], MTP [9][10], learned masks/lookahead tokens [18][27][39][40][41];
- **Training-free retrieval/n-gram/iteration**: [8][17][19][20][21][29];
- **Training-free embedding probing**: ESP [12].

###### 1.2 Historical positioning claim: training-free means no overfitting or transfer problem
Trained drafters fit target behavior on a distribution; domain/length/template shifts can lower acceptance. Reported EAGLE-3 degradation across domains [44], long contexts [45] and long reasoning [46] motivates online adaptation [47] and drafter selection [48].

The proposed method adds **no trainable parameters**. Frozen-target masks draft using its hidden states and $W_{\text{in}},W_{\text{out}}$. The historical draft therefore claimed:

- No drafter overfitting to training data;
- No training-to-deployment distribution-transfer problem;
- No retraining across models, only closed-form precomputation when untied.

This does not claim better absolute acceptance than in-domain trained methods; cross-domain stability and training-free comparison are H1/H8.

###### 1.3 Problem
ESP prompt mean is a bag-of-words statistic without explicit next-token prediction. One shared vector serves all slots/nodes, leaving attention to infer continuations.

###### 1.4 Proposed contributions

1. Prefix-conditioned higher-order output masks: track order-$n$ hidden states, predict/correct along each draft prefix and extrapolate within a parallel block.
2. Adaptive, dimension-factorized Kalman with innovation-based $Q,R$, proxy $R'>R$ and covariance $P$ slot uncertainty; per-sequence and training-free.
3. Hierarchy: online AR⊃fixed-coefficient AR/Newton differences; constant-gain EMA/α-β-γ⊂steady-state Kalman⊂adaptive Kalman (§5.10).
4. Tied direct hidden/norm mapping; untied LatentMAS [34].
5. Theory:
   - Losslessness (Proposition 1), speedup bound (2);
   - Unification (5), bias–variance (6,7);
   - Fixed-EMA/steady-state-Kalman correspondence and closed-form $\beta^\star$ (8);
   - Steady-state gain optimality (9);
   - Slot uncertainty (10).
6. Experimental hypotheses H1–H13, all untested.

###### 1.5 Meaning of denoising
Model the final hidden sequence as
$$y_t=Hs_t+v_t,$$
a predictable smooth trend $Hs_t$ plus a token-dependent, approximately unpredictable component $v_t$.

Estimating $s_t$ **denoises** the trajectory:

- EMA is fixed-gain denoising;
- Kalman is optimal linear denoising for given $Q,R$ (Proposition 9);
- When $\kappa_0<1$, the anchor itself is smoothed; v4 uses $\kappa_0=1$ and does not denoise the anchor.

Denoising means **output-hidden state estimation/filtering**, not diffusion denoising.

Whether the original name `denoise_spec_mtp` intended this meaning is [AUTHOR DECISION PENDING].

##### 2. Motivation

**(1) Predictive output hidden versus input mean.** The LM head maps final $h_i$ to next-token $p_\theta(\cdot\mid x_{\le i})$, encoding the full prefix. ESP $\bar e=\frac1t\sum e(x_i)$ averages inputs. Interpretability work decodes multiple later tokens from one hidden state [36].

Using an already predictive state as a mask is motivation for better continuations, **not evidence** [UNTESTED].

**(2) First-order direction.** An anchor predicts the immediate token; farther slots $j\ge2$ extrapolate along EMA hidden differences. The author included velocity as meaningful; E4 tests actual distant-slot acceptance.

**(3) Prefix conditioning.** Node $i$ masks predict after $\tilde x_{1:i}$. Root-shared masks at deeper nodes (repository state §5.11) are misaligned; per-prefix conditioning addresses it.

**(4) Availability.** Required hidden states are already computed in the previous block; no additional target forward.

**Untested soft-input distribution-shift risk.** Final hidden and input embeddings differ statistically. LatentMAS [34] reports alignment benefits for reasoning; here this is an E3 risk check, not evidence.

---


**(5) Higher-order information.** Adjacent hidden differences describe changing predictions: first order gives direction, second order curvature. Sufficient short-window smoothness could improve farther slots $j\ge2$.

Tokenwise jumps contain approximately unpredictable components amplified by differencing (Proposition 6). Optimal order is empirical (H10), expected to be low [CONJECTURE].

##### 3. Related Work

Reference numbers [1]–[43] retain v3; [44]–[48] were added with titles checked.

- Speculation/ESP: foundations/surveys [1][2][14], trained [3]–[7][9][10][16], training-free [8][17]–[23][29][30], tree verification [11][15], diffusion drafting [24]–[26], ESP baseline [12].
- Trained feature/hidden drafting: EAGLE [4]–[6] learns feature autoregression, Medusa/Hydra [3][7] train hidden heads, Hidden Transfer [35] learns intermediate-to-future transforms.
  - Difference: target hidden is used without trained modules; target masks still draft.
- Trained masks/lookahead: PaSS [18], BiTA [39], ParallelSpec [40], Samragh et al. [27], SPACE [41] learn masks; this draft constructs them analytically at inference.
- Drafter distribution shift:
  - Osprey [44]: math-trained Qwen3-8B EAGLE-3 acceptance5.06 in-domain versus1.86 on code.
  - OWL [45]: EAGLE-3 acceptance≈1.28 on LongSpecBench, below training-free PLD, speed0.81×.
  - P-EAGLE [46]: short-trained drafters lose up to25% acceptance on long reasoning.
  - Online Speculative Decoding [47] adapts drafters to shifted queries.
  - HedgeSpec [48] chooses domain-expert drafters online.
  - These numbers are original-paper claims, not repository measurements. Matched cross-domain stability here is H8 [UNTESTED].
- Soft/continuous input: Coconut [31] (trained), Soft Thinking [32], Mixture of Inputs [33], LatentMAS [34] (training-free), CoCoMix [42], prompt tuning [38]. They change generation content; this draft uses only uncommitted soft masks and seeks unchanged output tokens.
- Hidden decoding/extrapolation: Future Lens [36], Tuned Lens [37], LCM [43], FlowCast [13].
- Novelty: no prefix-conditioned output-extrapolation masks with same-forward ESP verification had been identified; systematic confirmation remains pending.

---

- **Classical extrapolation and filtering**:
  - Kalman filtering [49], constant-gain α-β/α-β-γ tracking [50], including tracking-index steady-state analysis;
  - Exponential smoothing/trends: Holt [52], Gardner [53], state-space exponential smoothing [54];
  - Linear prediction [51].

Newton forward/backward interpolation and extrapolation use standard numerical-analysis identities; textbook citation remains pending. Applying them to positional LLM final-hidden sequences for speculative masks was a novelty lead requiring systematic search.

- **Kalman filtering and adaptation**:
  - Discrete algebraic Riccati equations determine steady-state gain [49];
  - Kalata tracking indices yield optimal α-β/α-β-γ gains [50]; Gray & Murray [55] give the α-β-γ analytic form;
  - Unknown-noise estimation: Mehra innovation-based variance identification [56][57], Sage–Husa sequential prior statistics [58], Mohamed–Schwarz innovation-covariance matching [59].

The draft applies these methods to LLM hidden trajectories to construct speculative masks without training.

##### 4. Problem Formulation

###### 4.1 Notation

| Symbol | Meaning |
|---|---|
| $W_{\text{in}}=E,\ W_{\text{out}}=W\in\mathbb R^{V\times d}$ | Input embedding / LM head; tied means $W=E$ |
| $f_\theta$ | Frozen model producing final-normalized positional hidden $h$ and logits $Wh$ |
| $g(\cdot)$ | Greedy $\arg\max p_\theta$ |
| $k$ | Chain length / masks per node |
| $r$ | Root: last committed token at position $p$ |
| $\tilde x_{1:k}$ | Current chain draft |
| $\Phi$ | Output-to-input mapping |
| $\bar\nu=\frac1V\sum_w\lVert E_w\rVert$ | Mean input-embedding row norm |

###### 4.2 Parallel block, attention mask and positions

**Block sequence**, with $N_B=(k+1)(k+1)$ positions:
$$B=\big[\underbrace{n_0,n_1,\dots,n_k}_{\text{nodes}}\ \big|\ \underbrace{M^{(0)}_{1..k}}_{\text{node 0 masks}},\dots,\underbrace{M^{(k)}_{1..k}}_{\text{node }k\text{ masks}}\big],\qquad n_0=e(r),\ n_i=e(\tilde x_i).$$

**Indices:** node $n_i$ has block index $i$; $M^{(i)}_j$ has index $\iota(i,j)=(k+1)+ik+(j-1)$.

**Attention** $\mathcal A\in\{0,-\infty\}^{N_B\times(p+N_B)}$: all $p$ cached prefix columns are visible; within-block rules are
$$\mathcal A[n_i,n_{i'}]=0\iff i'\le i;$$
$$\mathcal A[M^{(i)}_j,n_{i'}]=0\iff i'\le i;$$
$$\mathcal A[M^{(i)}_j,M^{(i')}_{j'}]=0\iff i'=i\wedge j'\le j;$$
$$\mathcal A[n_i,M^{(\cdot)}_{\cdot}]=-\infty\ \text{(nodes cannot see any mask)}.$$

**Positions:**
$$\mathrm{pos}(n_i)=p+i,\qquad \mathrm{pos}(M^{(i)}_j)=p+i+j.$$
$M^{(i)}_j$ occupies the $j$th token after prefix $\tilde x_{1:i}$, as in ESP. Different-node masks may share position IDs, e.g. $M^{(0)}_2,M^{(1)}_1$, without conflict because they do not attend each other and have different prefixes.

**Parallelism:** all mask contents depend only on quantities known before this forward (§5.1); process $N_B$ positions together. After accepting $A$ drafts, use outputs at $M^{(A)}$ for next-round drafts/state.

**KV:** retain only prefix,$n_0,n_{1..A}$; discard masks and rejected-node KV.

###### 4.3 Acceptance and objective
Greedy acceptance length:
$$A=\max\{a\le k:\tilde x_i=g(\text{prefix},r,\tilde x_{<i})\ \forall i\le a\},\qquad\tau=A+1.$$
Maximize $\mathbb E[\tau]$ subject to AR-consistent outputs.

---

##### 5. Method

###### 5.1 Previous-round quantities and proxy observations
If the previous round accepted through $A'$, its mask $M^{(A')}_\ell$ produced current draft $\tilde x_\ell$ and final hidden $\tilde h_\ell$, $\ell=1..k$.

Since that mask position equals current node $n_{\ell-1}$, $\tilde h_\ell$ is a **proxy observation** of its hidden state predicting $\tilde x_\ell$.

Committed-position hidden states are **exact observations**, computed at accepted nodes (Algorithm2,line10).

###### 5.2 Order-$n$ state and transition
**State:** $s=(s^{[0]},\dots,s^{[n]})\in\mathbb R^{(n+1)\times d}$ estimates forward differences $\Delta^rh$ at the current position.

**Transition:** $\Delta^rh_{t+1}=\Delta^rh_t+\Delta^{r+1}h_t$, with highest $\Delta^n$ constant (constant-$n$th-difference / acceleration models). Therefore
$$s_{j}=F\,s_{j-1},\qquad F=\begin{pmatrix}1&\gamma_1&&\\&\rho_1&\gamma_2&\\&&\ddots&\ddots\\&&&\rho_n\end{pmatrix}\otimes I_d.$$

- Standard $\gamma_r=\rho_r=1$: $F=I+N$ with upper-shift $N$;
- $\gamma_r$ is step size; $\rho_r\le1$ damps trends as in exponential smoothing [53].

**Newton extrapolation (Proposition5(a)).** For $\gamma=\rho=1$, binomial expansion and $N^{n+1}=0$ give
$$F^j=\sum_{r=0}^n\binom{j}{r}N^r,\qquad \hat h_{t+j}=[F^js]^{[0]}=\sum_{r=0}^n\binom jr\,\Delta^rh_t.$$
the Newton forward-difference formula.

**Backward differences.** With $\nabla h_t=h_t-h_{t-1}$, an order-$n$ polynomial trajectory satisfies
$$\hat h_{t+j}=\sum_{r=0}^n\binom{j+r-1}{r}\nabla^r h_t,$$
the Newton–Gregory backward formula. Coordinates convert by $\Delta^r=\sum_{m=0}^{n-r}\binom{r+m-1}{m}\nabla^{r+m}$, truncated at $n$, from $\Delta=\nabla(1-\nabla)^{-1}$.

###### 5.3 Observation correction: per-order EMA / α-β-γ
**Predict/correct:** for either exact or proxy observation $y$, predict then correct:
$$s^-=Fs,\qquad s\leftarrow s^-+K\otimes(y-s^{-[0]}),\qquad K=(\kappa_0,\kappa_1,\dots,\kappa_n)^\top.$$

- $n=1$ gives α-β and $n=2$ α-β-γ filtering [50];
- Fixed $K$ corresponds to steady-state constant-gain Kalman [49];
- Each gain component is a scalar shared across all $d$ dimensions, without learning.

**Independent per-order β, the primary parameterization.** Set $\kappa_0=1$ to take the observation directly as anchor, as v4; EMA each $r\ge1$:
$$s^{[r]}\leftarrow\beta_r\,s^{-[r]}+(1-\beta_r)\big(s^{[r-1]}_{\text{new}}-s^{[r-1]}_{\text{old}}\big).$$
Here $s^{[r-1]}_{\text{new}}-s^{[r-1]}_{\text{old}}$ is the actual lower-order state change, yielding a separate EMA for each difference. Relation to $K$:

- $n=1$: exactly $\kappa_1=1-\beta_1$ (Proposition5(b));
- $n\ge2$: both are constant-gain linear filters, but the gain mapping is more complex. This is a parameterization choice; implement per-order EMA and compare numerically with $K$ inE15.

**Separate gains for observation types:**

- Exact committed observations use $K$ or $\beta_r$;
- Noisier draft-prefix proxies use more conservative $K'$ or $\beta'_r\ge\beta_r$.

###### 5.4 Prefix accumulation and within-block extrapolation
**Prefix accumulation:** committed state $s^c$ was last corrected by $h_{\text{last}}=H[n_{A'}]$. Node states:
$$s^{(0)}=\mathrm{Corr}_{K'}\big(Fs^c,\ \tilde h_1\big),\qquad s^{(i)}=\mathrm{Corr}_{K'}\big(Fs^{(i-1)},\ \tilde h_{i+1}\big)\ (1\le i\le k-1),\qquad s^{(k)}=Fs^{(k-1)}.$$
End node $k$ has no proxy and only predicts.

**Within-block extrapolation** at node $i$:
$$M^{(i)}_j=\Phi\big([F_{\text{blk}}^{\,j-1}s^{(i)}]^{[0]}\big),\qquad j=1..k.$$
$F_{\text{blk}}$ may differ from $F$, e.g. separate damping $\rho$, to ablate recurrence/damping. Without new observations, recursive updates are $s_j=F_{\text{blk}}s_{j-1}$.

**Special cases (Proposition5):**

- $n=0$: anchor only, $M^{(i)}_j=\Phi(\tilde h_{i+1})$;
- $n=1$, $K=(1,1-\beta)$, $F_{\text{blk}}=\left(\begin{smallmatrix}1&\gamma\\0&\rho\end{smallmatrix}\right)$: v4 recurrence with $\rho=\beta'+(1-\beta')\gamma$;
- $\rho=1$: frozen velocity.

###### 5.5 General linear-predictor control
**Form:**
$$\hat h_{t+1}=\sum_{l=0}^{L}a_l\,h_{t-l},\qquad a\in\mathbb R^{L+1},\ \text{scalar coefficients shared across dimensions}.$$
Iterated multi-step predictions equal powers of companion matrix $C(a)$.

**(i) Order-determined fixed coefficients.** Take $L=n$ and $\nabla^{n+1}\hat h_{t+1}=0$:
$$a_l=(-1)^l\binom{n+1}{l+1},\qquad l=0..n.$$
For example $n=1$: $2h_t-h_{t-1}$; $n=2$: $3h_t-3h_{t-1}+h_{t-2}$. Coefficients satisfy affine $\sum_la_l=1$, preserving constants. This is exactly unsmoothed difference extrapolation in§5.2 (Proposition5(c)).

**(ii) Online least squares, training-free per-sequence test-time fitting.** Fit the last $W$ committed positions:
$$a^\star=\arg\min_a\sum_{\tau=t-W+1}^{t}\omega^{t-\tau}\Big\lVert h_\tau-\sum_l a_lh_{\tau-1-l}\Big\rVert^2+\mu\lVert a-a^{\text{Newton}}\rVert^2.$$

- Solve an $(L+1)\times(L+1)$ normal equation, optionally with $\sum a_l=1$;
- Per-step cost $O(WL^2d)$, or recursive least squares $O(L^2d)$;
- Makhoul linear prediction [51] estimates coefficients online without training data.

**Relation to differences:** both are linear time-invariant or slowly varying predictors:

- $C(a^{\text{Newton}})$ is similar to $F=I+N$, using delay rather than difference coordinates;
- EMA-corrected differences at steady state equal an infinite-order exponentially decaying linear predictor. Holt/ARIMA equivalence is classical; precise citation remains to verify.

###### 5.6 Mapping $\Phi$
As in v4:

- Tied: $\Phi(h)=\bar\nu h/\lVert h\rVert$;
- Untied: $\Phi(h)=\bar\nu hW_a/\lVert hW_a\rVert$, $W_a=(W_{\text{out}}^\top W_{\text{out}}+\lambda I)^{-1}W_{\text{out}}^\top W_{\text{in}}$ [34].

Linearity permits applying $W_a$ to state components before extrapolation; untied models need only $(k+1)(n+1)$ $d\times d$ products per round.

###### 5.7 Cost

| Item | Storage | Per-round computation |
|---|---|---|
| Committed state $s^c$ | $(n+1)d$ | One $O(nd)$ correction per committed token |
| Node states $s^{(0..k)}$ | $(k+1)(n+1)d$ | $O(knd)$ |
| Within-block extrapolation | — | $(k+1)k$ masks at $O(nd)$ each; total $O(k^2nd)$ |
| Untied mapping | $d^2$ for $W_a$ | $O((k+1)(n+1)d^2)$ |
| Optional linear-prediction LS | $Wd$ recent hidden states | $O(WL^2d)$ |
| Reference block forward | — | Approximately $2N(k+1)^2$ |

Example: $k=4,n=3,d=4096$ gives untied mapping $20\cdot2d^2\approx6.7\times10^8$ FLOPs, versus an8B block $2\cdot8\times10^9\cdot25=4\times10^{11}$, approximately0.2% (estimate, not measurement).

Only recent $n+1$ committed hidden states or equivalent state, plus previous $M^{(A')}$'s $k$ hidden states, need extra caching.

###### 5.8 Algorithms
**Algorithm 1: order-$n$ prefix-conditioned mask construction**
```text
Input : committed state s^c ∈ ℝ^{(n+1)×d}; proxies h̃_1..h̃_k; F, F_blk; gains K′ (or β′_1..β′_n); Φ
Output: M⁽⁰⁾..M⁽ᵏ⁾ ∈ ℝ^{k×d}  (fixed before the forward)
 1  s ← s^c
 2  for i ← 0 to k do                                  ▷ prefix accumulation along the draft
 3      s⁻ ← F·s
 4      if i < k then s ← Corr(s⁻, h̃_{i+1}; K′)         ▷ s⁻ + K′⊗(h̃_{i+1} − s⁻[0]) or per-order EMA
 5      else s ← s⁻                                    ▷ chain end: predict only
 6      s⁽ⁱ⁾ ← s
 7  for i ← 0 to k do                                  ▷ independent → parallel
 8      u ← s⁽ⁱ⁾
 9      for j ← 1 to k do
10          M⁽ⁱ⁾_j ← Φ(u[0]);  u ← F_blk·u                ▷ s_j = F s_{j−1}
11  return M⁽⁰⁾..M⁽ᵏ⁾
```

**Algorithm 2: greedy chain decoding**
```text
Input : prompt x_{1:t}; frozen f_θ; k; n; Φ; N_max
Output: Y (identical to greedy AR, Prop. 1)
 1  if untied: W_a ← solve(WᵀW + λI, WᵀE) in fp32
 2  (H, L) ← f_θ([e(x_{1:t}) | ē×k])                     ▷ prefill
 3  s^c ← init from H[t−n..t−1] (backward differences → Δ via §5.2)
 4  r ← argmax L[t−1]; x̃ ← argmax L[masks]; h̃_{1..k} ← H[masks]; Y ← [r]
 5  while |Y| < N_max and r ≠ EOS do
 6      M ← Alg1(s^c, h̃, F, F_blk, K′, Φ)
 7      (H, L) ← f_θ([e(r), e(x̃_{1..k}) | M⁽⁰⁾..M⁽ᵏ⁾]; 𝒜, pos as §4.2)
 8      A ← max{a : argmax L[n_{i−1}] = x̃_i ∀ i ≤ a}
 9      C ← [x̃_{1..A}, argmax L[n_A]]; truncate; Y ← Y ∥ C
10      for i ← 0 to A: s^c ← Corr(F·s^c, H[n_i]; K)      ▷ exact observations
11      h̃_{1..k} ← H[M⁽ᴬ⁾]; x̃ ← argmax L[M⁽ᴬ⁾]; r ← last(C)
12      keep KV of prefix + n_0..n_A
13  return Y
```
Line10 corrects committed state once per accepted node using exact hidden. The bonus hidden becomes available at next-round root, one step later.

###### 5.9 Adaptive Kalman, the main adaptive variant
**State-space model:**
$$s_{t+1}=F s_t+w_t,\quad w_t\sim\mathcal N(0,\mathbf Q);\qquad y_t=Hs_t+v_t,\quad v_t\sim\mathcal N(0,\mathbf R);\qquad H=[1,0,\dots,0].$$
$s_t\in\mathbb R^{(n+1)\times d}$ and $y_t$ is observed final-layer hidden.

**Dimension factorization:** $\mathbf Q=Q\otimes I_d$, $\mathbf R=RI_d$, with $Q\in\mathbb R^{(n+1)\times(n+1)}$, scalar $R>0$. Full covariance is $P\otimes I_d$; $d$ identical-parameter $(n+1)$-dimensional filters share $P,K$.

- **Rationale:**
  - $P$ has only $(n+1)^2$ elements and a scalar gain vector, consistent with v5 dimension-shared gains;
  - Statistically, cross-dimensional correlation is ignored and noise scales are assumed equal.
- **Known violations:**
  - RMSNorm is followed by learned per-dimension weights, producing unequal scales and some large dimensions;
  - Hidden states are anisotropic.
- **Optional still-factorized extensions:**
  - Per dimension: $\mathbf R=\mathrm{diag}(R_1..R_d)$, $\mathbf Q=\mathrm{blockdiag}(Q_m)$, with $d$ small filters and $O(dn^2)$ cost;
  - Group parameters over $G$ groups;
  - Prewhiten by RMSNorm weights before a shared filter.
- E17 compares these three choices.

**Predict/correct**, identical per dimension, in $(n+1)$ form:
$$\text{Predict:}\ s^-=Fs,\qquad P^-=FPF^\top+Q;$$
$$\text{Correct:}\ e=y-Hs^-,\quad S=HP^-H^\top+R,\quad K=P^-H^\top S^{-1},\quad s=s^-+K\otimes e,\quad P=(I-KH)P^-(I-KH)^\top+KRK^\top.$$
Use Joseph covariance updates to preserve symmetry/positive definiteness.

**Steady-state gain:** if $(F,H)$ is observable and $(F,Q^{1/2})$ stabilizable, $P^-$ converges to a discrete algebraic Riccati solution:
$$P^-_\infty=FP^-_\infty F^\top-FP^-_\infty H^\top(HP^-_\infty H^\top+R)^{-1}HP^-_\infty F^\top+Q,\qquad K^\star=P^-_\infty H^\top(HP^-_\infty H^\top+R)^{-1}.$$
$F=I+N,H=[1,0..]$ is observable. Exact convergence assumptions and textbook citation remain to verify.

**α-β correspondence (Kalata):** for discrete white-noise acceleration ($n=1,Q=\sigma_w^2GG^\top,G=(T^2/2,T)^\top,R=\sigma_v^2$), steady gain $K^\star=(\alpha,\beta/T)$ depends on tracking index $\lambda=\sigma_wT^2/\sigma_v$. Commonly quoted form:
$$r=\frac{4+\lambda-\sqrt{8\lambda+\lambda^2}}{4},\qquad\alpha=1-r^2,\qquad\beta=2(2-\alpha)-4\sqrt{1-\alpha}.$$
For α-β-γ, relations include $\gamma=\beta^2/(2\alpha)$ [55].

**Verification limit:** these expressions were checked only through Wikipedia's Alpha beta filter article and ControlSystems.jl, citing Kalata1984 [50]/Gray & Murray1993 [55]. Original papers were not read; verification remains pending. Gain conventions differ by a factor2 ($\gamma$ versus $2\gamma$), with originals authoritative.

**Innovation-adaptive online $Q,R$**, training-free per sequence: commonly quoted formulas below remain to verify against originals.

- **Innovation covariance matching**, Mohamed–Schwarz [59]: scalarize the second moment over window $W$:
  $$\hat C_t=\frac1{Wd}\sum_{\tau=t-W+1}^t\lVert e_\tau\rVert^2.$$
  Estimate from it:
  $$\hat R=\max\big(\hat C_t-HP^-H^\top,\ r_{\min}\big),\qquad \hat Q=K\hat C_tK^\top\ (+\epsilon I).$$
- **Sage–Husa recursion** [58], forgetting $b\in(0,1)$ and $\delta_t=(1-b)/(1-b^{t+1})$:
  $$\hat R_t=(1-\delta_t)\hat R_{t-1}+\delta_t\big(\tfrac1d\lVert e_t\rVert^2-HP^-H^\top\big),$$
  $$\hat Q_t=(1-\delta_t)\hat Q_{t-1}+\delta_t\big(K\tfrac1d\lVert e_t\rVert^2K^\top+P-FP_{t-1}F^\top\big).$$
  Project back onto the positive-definite cone with eigenvalue clipping and lower bounds each step.
- **Mehra** [56][57]: identify/test $Q,R$ using innovation autocorrelation. White innovations diagnose consistency and are an E17 metric.
- **Initialization:** warm up with committed prompt hidden states, correcting without drafting. Prompt $\lVert\nabla^rh\rVert$ can provide moment estimates using Proposition6: independent noise gives $\mathbb E\lVert\nabla^{n+1}y\rVert^2=\lVert\nabla^{n+1}p\rVert^2+\binom{2n+2}{n+1}\sigma^2d$.

**Two observation types:**

- Exact committed hidden: noise $R$;
- Draft proxy $\tilde h$: $R'=R+R_\delta$, $R_\delta\ge0$, implementing larger noise.

Estimate $R_\delta$ online without training: when a node is accepted, compare exact $H[n_i]$ with previous proxy $\tilde h_{i+1}$ at the same position:
$$\hat R_\delta\leftarrow\text{EMA}\Big(\tfrac1d\lVert H[n_i]-\tilde h_{i+1}\rVert^2\Big).$$
Correct prefix accumulation with $R'$ (§5.4); within the block only predict.

**Slot-uncertainty extension:** prediction covariance at node $i$, within-block step $j$:
$$P^{(i)}_j=F^{\,j-1}P^{(i)}F^{(j-1)\top}+\sum_{m<j-1}F^mQF^{m\top},\qquad \sigma^{2}_{i,j}=HP^{(i)}_jH^\top$$
per dimension. Possible uses:

- (a) Adaptive draft length $k_{\text{eff}}=\max\{j:\sigma^2_{0,j}\le\theta\}$, requiring variable/truncated attention blocks;
- (b) Adaptive mask temperature $T_j\propto\sigma_{i,j}$ in M2 expected-embedding mapping;
- (c) Acceptance-probability feature.

These are **untested extensions**, excluded from main results. Covariance versus actual acceptance is conjecture C13.

**Algorithm 3: Kalman mask provider**
```text
State : s^c ∈ ℝ^{(n+1)×d}, P^c ∈ ℝ^{(n+1)×(n+1)}, Q, R, R_δ, innovation window 𝓦
Init(prompt hiddens y_1..y_t):
 1  s^c ← backward-difference init from y_{t−n..t}; P^c ← p_0·I; Q,R ← moment init (Prop. 6) or defaults
 2  for τ in warm-up positions: (s^c, P^c, e) ← KF(s^c, P^c, y_τ, R); push e to 𝓦
ObserveCommitted(y):                                   ▷ exact hidden of an accepted node
 3  (s^c, P^c, e) ← KF(s^c, P^c, y, R); push e to 𝓦
 4  (Q, R) ← AdaptQR(𝓦, P^c, K)                         ▷ covariance matching / Sage–Husa; clamp to PSD, R ≥ r_min
ObserveProxyError(y_exact, h̃_proxy):
 5  R_δ ← EMA(‖y_exact − h̃_proxy‖²/d)
Masks(proxies h̃_1..h̃_k):                              ▷ called before the block forward
 6  s ← s^c; P ← P^c
 7  for i ← 0 to k do
 8      s ← F·s; P ← F P Fᵀ + Q
 9      if i < k then (s, P, _) ← KF_update(s, P, h̃_{i+1}, R + R_δ)
10      (s⁽ⁱ⁾, P⁽ⁱ⁾) ← (s, P)
11  for i ← 0 to k do                                  ▷ parallel over nodes
12      u ← s⁽ⁱ⁾; Σ ← P⁽ⁱ⁾
13      for j ← 1 to k do
14          M⁽ⁱ⁾_j ← Φ(u[0]); σ²_{i,j} ← Σ[0,0]              ▷ σ² optional (extensions)
15          u ← F_blk·u; Σ ← F_blk Σ F_blkᵀ + Q
16  return M, σ²
KF_update(s, P, y, R):  e ← y − s[0]; S ← P[0,0] + R; K ← P[:,0]/S
                        s ← s + K⊗e; P ← (I−K hᵀ)P(I−K hᵀ)ᵀ + K R Kᵀ  (h = e_0, Joseph); return (s, P, e)
```
KF means prediction plus KF_update.

**Complexity:**

- Covariance recursion is independent of $d$, $O(n^3)$ per step, negligible for $n\le3$;
- State updates are $O(nd)$ per step;
- Per round:
  - Prefix accumulation $O(k(nd+n^3))$;
  - Within-block prediction $O(k^2(nd+n^3))$;
  - Adaptive estimation $O(d+n^3)$;
- Per-dimension extension scales covariance work to $O(dn^3)$;
- Compared with fixed-gain v5, extra cost is only small order-$n$ matrices.

**Numerical stability:**

- FP32 states/covariance;
- Joseph covariance update and per-step symmetrization $P\leftarrow(P+P^\top)/2$;
- Lower bounds and positive-definite projection for $R,Q$;
- Higher-order $F$ powers can grow for $n\ge2$; within-block damping $\rho<1$ may help;
- $S=P_{00}+R$ is scalar, requiring no matrix inverse.

###### 5.10 Predictor hierarchy

| Level | Form | Coefficient/gain source | Assumption | Training-free |
|---|---|---|---|---|
| Online small AR | $\hat h_{t+1}=\sum_{l\le L}a_lh_{t-l}$, coefficients from per-sequence weighted LS | Test-time sequence fit | Locally stationary linear dynamics | Yes |
| Newton / fixed AR | $a_l=(-1)^l\binom{n+1}{l+1}$, equivalent to unsmoothed $F=I+N$ | Order $n$ | Local noiseless order-$n$ polynomial | Yes |
| Constant-gain EMA/α-β-γ | $s\leftarrow Fs+K\otimes(y-[Fs]_0)$ | Manual or grid-searched $\beta$ | Polynomial trend plus noise, fixed gain | Yes, parameter selection needed |
| Steady Kalman | Same, $K^\star(Q,R)$ from DARE | Given $Q,R$ | Linear Gaussian known-noise state space | Yes |
| Adaptive Kalman | Time-varying $K_t$, innovation-estimated $Q,R$ | Per sequence online | Unknown, slowly varying noise | Yes |

```text
 online AR (fit a_l per sequence) ──special case: fixed a_l = Newton──▶ Newton difference (fixed AR, no smoothing)
          ▲ steady-state Kalman ≡ infinite-order AR with geometrically decaying a_l (classical equivalence, ref. TBD)
 constant-gain filter (EMA / α-β-γ) ──K chosen optimally for (Q,R)──▶ steady-state Kalman ──Q,R estimated online──▶ adaptive Kalman
 Newton difference = difference-state filter without smoothing (β_r = 0, Prop. 5c); fixed EMA (n=1) = steady-state Kalman of local-linear-trend, R→0 (Prop. 8)
```

The three predictors in brief:

- Linear prediction fits a small AR online at inference;
- Differences give fixed AR coefficients under local polynomial assumptions;
- Kalman is the state-space form with optimal/adaptive gains; fixed EMA is its steady special case (Proposition8).

###### 5.11 Read-only repository implementation audit,2026-10-13
**Then-current state:**

1. Every node already owns $k$ mask slots: prefix-conditioned parallel structure exists.
   - `tree_attention.block_layout` is `[root | node_1..node_N | masks(root) | masks(node_1) | … ]`, $k$ masks per owner;
   - Mask $j$ sees owner attention (prefix+ancestors+owner), its own masks1..j−1 and itself;
   - Position `rel[owner]+j+1`;
   - Nodes never see masks.

   This matches§4.2 attention/positions; a chain is `branches:[1]*k`. Next-round seeds use deepest accepted owner's masks via `mask_indices_of(tree,k,n_star)`.

2. Mask **contents** are identical across owners.
   - `provider.masks()` returns $[k,d]$; `masks.unsqueeze(0).expand(1+num_nodes,k,-1)` copies it to every owner;
   - Only committed input embeddings update state via `on_token_committed(new_embed,prev_embed)`; no draft-prefix or hidden observations.

3. `HFCausalLM.forward` accepts `output_hidden_states`, but SpecDecoder does not request it; final normalization remained to verify.

**Proposed changes only, no code edited:**

| Change | Location | Detail |
|---|---|---|
| Provider returns $[1+N,k,d]$ from draft prefixes/proxy hidden | mask_providers.py | Add masks_per_owner(draft_tokens,draft_hiddens,...), retain ESP interface |
| Replace shared expand with owner tensor | decoding.py block assembly | Reshape in block_layout owner order |
| Request final hidden, retain $H[n_A],H[M^{(A)}]$ | decoding.py,hf_causal.py | Final layer only, not all layers |
| Update from exact committed hidden | decoding.py commit loop | Replace/coexist with input-embedding updates |
| Untied precompute $W_a,\bar\nu$ | New module | FP32, once |
| Attention/positions | tree_attention.py | **Unchanged**: structure depends only on tree; efficient cache remains applicable |

General-tree prefixes follow ancestor paths; fold state along the ancestors.

**Correction to v3§11:** shared masks correctly described numeric vectors, but attention/position were already owner-specific and parallel. Prefix conditioning needs content/provider/assembly changes, not attention/position redesign.

---


**Higher-order implementation note, v5, not implemented:**

- Cache committed $s^c\in\mathbb R^{(n+1)\times d}$ or an $n+1$ hidden ring, plus previous $k$ hidden states at $H[M^{(A)}]$;
- Add `observe_committed(hidden)` and `masks_per_owner(proxy_hiddens)->[1+N,k,d]`;
- Configure order,$F/F_\text{blk}$ ($\gamma_r,\rho_r$),$K,K'$ or $\beta_r,\beta'_r$, predictor∈{difference,linear_fixed,linear_ls};
- Change mask_providers.py,decoding.py,hf_causal.py final-hidden request; tree_attention.py unchanged;
- Prefer FP32 state; higher-order BF16 differences have cancellation risk checked inE12.

##### 6. Theoretical Analysis

###### Proposition 1: losslessness with soft masks
**Conditions:** discrete node embeddings, attention $\mathcal A$ with no node-to-mask edges, $\mathrm{pos}(n_i)=p+i$, exact arithmetic.

**Claim:** for arbitrary, even node-specific soft masks:

- (a) Greedy output equals tokenwise AR;
- (b) Draft distribution $q$, acceptance $\min(1,p/q)$ and residual resampling [1][2] yield the AR sampling distribution.

**Argument:** node logits ignore masks and equal AR conditionals on $(\cdot,\tilde x_{1:i})$. Draft $q$ was fixed before this round, so the standard speculative-sampling lemma applies positionwise. KV retains discrete-input nodes only. $\square$

Floating-point implementations use exact_match checks. The repository's existing $T>0$ path is not residual sampling; (b) needs a new implementation.

###### Proposition 2: chain speedup bound
$$S\le\frac{\mathbb E[\tau]\,c_1}{c_B+c_\Phi}\le\mathbb E[\tau]\le k+1.$$
$c_1$ is single-token forward cost, $c_B$ block cost and $c_\Phi$ mapping cost.

###### Proposition 3: hidden/logit extrapolation equivalence
Extrapolation is a linear combination and $W$ linear, so $W\hat a_j$ equals the same logit extrapolation. Probability-space extrapolation differs. $\square$

###### Proposition 4: tied ridge alignment is approximately identity
For $W_\text{out}=W_\text{in}=E$, $W_a=(E^\top E+\lambda I)^{-1}E^\top E$, and
$$\lVert W_a-I\rVert_2=\frac{\lambda}{\sigma_{\min}^2(E)+\lambda}.$$
When $\lambda\ll\sigma_\min^2$, $W_a\approx I$. Thus tied models can use direct norm-scaled hidden; actual $\sigma_\min$ remains to measure inE8.


###### Proposition 5: zero/first order and linear prediction as special cases
(a) **Newton:** for $\gamma_r=\rho_r=1$, $[F^js]^{[0]}=\sum_{r\le n}\binom jr s^{[r]}$.

*Argument:* $F=I+N$, $N^{n+1}=0$ and commutation permit binomial expansion. $\square$

(b) **v4 is $n=1$.** Choose $\kappa_0=1,\kappa_1=1-\beta,F=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$. Corrected anchor is the observation, velocity $v\leftarrow\beta v+(1-\beta)(y-y_\text{prev})$. Within-block $F_\text{blk}=\left(\begin{smallmatrix}1&\gamma\\0&\rho\end{smallmatrix}\right)$ yields $\hat a_j=a+\gamma c_j(\rho)v$. $n=0$ is anchor only.

*Argument:*

- Predict $s^{-[0]}=y_\text{prev}+v,s^{-[1]}=v$;
- Anchor correction $s^{[0]}=s^{-[0]}+(y-s^{-[0]})=y$;
- Velocity correction $s^{[1]}=v+(1-\beta)(y-y_\text{prev}-v)=\beta v+(1-\beta)(y-y_\text{prev})$;
- The $(0,1)$ entry of $F_\text{blk}^{\,j}$ is $\gamma\sum_{s<j}\rho^s$. $\square$

(c) **Fixed linear prediction equals unsmoothed differences.** Direct historical backward differences ($\beta_r=0$) give one-step coefficients $a_l=(-1)^l\binom{n+1}{l+1}$.

*Argument:* solve $\nabla^{n+1}\hat h_{t+1}=\sum_{m=0}^{n+1}(-1)^m\binom{n+1}m h_{t+1-m}=0$. $\square$

###### Proposition 6: one-step bias and noise amplification
**Setup:** observations $h_\tau=p_\tau+\varepsilon_\tau$, deterministic $p$, independent zero-mean noise with covariance $\sigma^2I_d$, and unsmoothed order-$n$ extrapolation $\hat h_{t+1}=\sum_{l=0}^na_lh_{t-l}$ from5(c).

**Claim:**
$$\mathbb E\lVert\hat h_{t+1}-p_{t+1}\rVert^2=\underbrace{\lVert\nabla^{n+1}p_{t+1}\rVert^2}_{\text{bias}^2}+\underbrace{\sigma^2d[\tbinom{2n+2}{n+1}-1]}_{\text{noise amplification}}.$$

*Argument:*

1. By5(c), $p_{t+1}-\sum_la_lp_{t-l}=\nabla^{n+1}p_{t+1}$.
2. Therefore error is $-\nabla^{n+1}p_{t+1}+\sum_la_l\varepsilon_{t-l}$.
3. Cross expectation is zero; noise variance is $\sigma^2d\sum_la_l^2$.
4. Vandermonde gives $\sum_{l=0}^n\binom{n+1}{l+1}^2=\binom{2n+2}{n+1}-1$. $\square$

**Values:** noise factors for $n=0,1,2,3$ are1,5,19,69, growing approximately $4^{n+1}$.

**Order criterion:** increasing $n$ to $n+1$ improves one-step MSE iff
$$\lVert\nabla^{n+1}p\rVert^2-\lVert\nabla^{n+2}p\rVert^2>\sigma^2d\Big[\tbinom{2n+4}{n+2}-\tbinom{2n+2}{n+1}\Big].$$
higher-order trajectory differences decay sufficiently quickly and noise is small.

###### Proposition 7: multi-step variance
Under the same setup, $j$-step unsmoothed extrapolation is $\hat h_{t+j}=\sum_{l=0}^n\ell_l(j)h_{t-l}$, Lagrange bases on nodes $\{0,-1,\dots,-n\}$, with noise variance $\sigma^2d\sum_l\ell_l(j)^2$.

Each $\ell_l$ is degree$n$, so variance grows $O(j^{2n})$.

*Argument:* polynomial extrapolation is Lagrange interpolation evaluated at$j$; the rest follows Proposition6. $\square$

**Implication:** distant high-order slots have faster variance growth, motivating damping $\rho<1$ or lower within-block order.

###### Conjecture C11: smoothing
EMA $\beta_r>0$ reduces noise at the cost of lag. Some $(n,\beta_{1..n})$ maximizes $\mathbb E[A]$, with low optimal order≤2. Propositions6/7 cover only unsmoothed cases; steady smoothed-error analysis using tracking indices [50] is unfinished.

###### Conjecture C12: hidden noise
Predictable trend is limited and independent noise approximate. E13 can estimate bias/noise from committed-hidden $\lVert\nabla^rh\rVert$ versus$r$. Growth near $\sqrt{\binom{2r}{r}}$ suggests noise dominance.


###### Proposition 8: fixed EMA as steady Kalman, closed-form $\beta^\star$
**Setup:** $n=1,F=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right),H=[1,0]$, local-linear-trend $Q=\mathrm{diag}(q_0,q_1)$ with positive level/slope noise, and $R\to0$.

**Claim:** steady gain $K^\star=(1,\kappa_1^\star)$, where
$$\kappa_1^\star=\frac{\pi}{1+\pi},\qquad \pi=\frac{\phi+\sqrt{\phi^2+4\phi}}{2},\qquad\phi=\frac{q_1}{q_0}.$$
Thus v4 fixed EMA ($\kappa_0=1,\kappa_1=1-\beta$) is this model's steady Kalman, with
$$\beta^\star=1-\kappa_1^\star=\frac1{1+\pi(\phi)}.$$
$\beta^\star$ decreases with signal-to-noise$\phi$: as$\phi\to0$ (almost constant slope), $\beta^\star\to1$ (long memory); as$\phi\to\infty$, it tends0.

**Argument:**

1. $R=0$ gives zero corrected position variance, so steady $P=\mathrm{diag}(0,p)$.
2. $P^-=FPF^\top+Q=\left(\begin{smallmatrix}p+q_0&p\\p&p+q_1\end{smallmatrix}\right)$.
3. $K=P^-H^\top/P^-_{00}=(1,p/(p+q_0))$.
4. Corrected slope variance $p+q_1-p^2/(p+q_0)$ equals$p$ at steady state, yielding $p^2-q_1p-q_1q_0=0$ and positive root $p=q_0\pi$.
5. Substitute to obtain $\kappa_1^\star=\pi/(1+\pi)$.
6. After correction, $s^{[0]}=y$ and $s^{[1]}=v+\kappa_1(y-y_{\text{prev}}-v)=(1-\kappa_1)v+\kappa_1\nabla y$: an EMA with $\beta=1-\kappa_1$. $\square$

**Implications:**

- Searching $\beta$ is equivalent to searching the signal-to-noise ratio $\phi$ in the local-linear-trend family with noiseless observations.
- For $R>0$, optimal $\kappa_0<1$ also denoises the anchor. This additional freedom exceeds the fixed EMA family of v4/v5 and is the proposed theoretical source of a Kalman gain (H12).
- Whether every order-wise EMA with $n\ge2$ is a steady-state Kalman filter for some diagonal $Q$ remains unproved (C14).

**Relation to Kalata:** Kalata [50] uses discrete white-noise acceleration (rank-one $Q$), obtaining a closed form with $\alpha<1$ for $R>0$ (§5.9, verification pending). This proposition instead uses diagonal $Q$ and $R\to0$; both are steady-state Kalman special cases of the alpha-beta family.

###### Proposition 9 (mean-square optimal steady-state gain; existence of optimal beta)
For a linear Gaussian model with given $(Q,R)$:

- Kalman filtering minimizes prediction mean-square error among all linear (indeed all measurable) estimators [49].
- Among constant-gain filters $K$ with stable error dynamics, $K^\star$ (the DARE solution) minimizes steady-state prediction error covariance (standard result; textbook citation pending).

**Corollary:** For the model family in Proposition 8, a unique optimal $\beta^\star(\phi)$ exists. If the true hidden trajectory approximately follows some $(Q,R)$, the grid-optimal beta should approach $\beta^\star(\hat\phi)$; E16 tests this.

**Interpretation:** Optimal beta is determined by the data's noise structure. Sequence-wise adaptive estimation of $(Q,R)$ corresponds to selecting beta per sequence, potentially avoiding dataset-specific tuning (H12).

###### Proposition 10 (slot prediction variance recursion)
Within-block prediction satisfies $P^{(i)}_{j+1}=F_{\text{blk}}P^{(i)}_jF_{\text{blk}}^\top+Q$, hence
$$\sigma^2_{i,j+1}=[F_{\text{blk}}P^{(i)}_jF_{\text{blk}}^\top]_{00}+Q_{00}.$$
For $F_{\text{blk}}=I+N$, $n\ge1$, $[FPF^\top]_{00}=P_{00}+2P_{01}+P_{11}$. If $P_{01}\ge-P_{11}/2$ (for example, Riccati covariances with nonnegative position/slope error correlation), $\sigma^2_{i,j}$ strictly increases with $j$ when $Q_{00}>0$.

**Proof:** Row zero of $F=I+N$ is $(1,1,0,\dots)$; substitute in the quadratic form. $\square$

Monotonicity without this condition is unproved (C15).

###### Kalman conjectures

- **C13:** $\sigma^2_{i,j}$ is negatively correlated with slot acceptance and can inform adaptive $k$ or temperature.
- **C14:** Order-wise EMA for $n\ge2$ can represent a steady-state Kalman filter with diagonal $Q$ and $R\to0$.
- **C15:** Proposition 10 remains monotone without the covariance sign condition.
- **C16:** Estimated $(\hat Q,\hat R)$ predict optimal order: compare higher-order components of $\hat Q$ with $\hat R$ using Proposition 6's corollary.

###### Other conjectures

- **C6 (predictiveness):** Expected accepted tokens satisfy output-space order zero > ESP mean.
- **C9 (first order):** First-order extrapolation improves acceptance at farther slots ($j\ge2$).
- **C10 (prefix conditioning):** Prefix conditioning outperforms shared state, with larger gains at greater node depth $i$.
- **C7 (distribution shift):** Soft inputs may produce out-of-distribution activations; whether norm scaling helps is untested.

---

##### 7. Hypotheses
Let $\Delta_x(\pi,\pi')$ denote the per-prompt accepted-token difference. Use paired bootstrap 95% confidence intervals and Wilcoxon tests on the confirmation set, with Holm correction across hypotheses.

- **H1 (main effect):** $\Delta(\text{Ours},\text{ESP})>0$ on at least two model scales and two data slices.
- **H2 (order-zero predictiveness):** $\Delta(\text{zero-only},\text{ESP})>0$.
- **H3 (first order):** $\Delta(\text{zero+first},\text{zero-only})>0$, concentrated at $j\ge2$.
- **H4 (prefix conditioning):** Prefix-conditioned > shared $(a^{(0)},v^{(0)})$, with descriptively nondecreasing gains by node depth.
- **H5 (within-block recursion):** Recursive > frozen for $\rho\ne1$; negative controls: $\gamma=1$ or $k=1$.
- **H6 (mapping):**
  - Tied: hidden plus norm scaling >= raw hidden.
  - Untied: $W_a$ mapping >= direct hidden.
  - Negative control: on tied models, $W_a$ mapping equals identity.
- **H7 (offset discrimination):** Significant improvement over fixed offset $\pi_{\text{off}}$.
- **H8 (cross-domain stability):** Define retention $\mathrm R=\mathbb E[\tau]_{\text{OOD}}/\mathbb E[\tau]_{\text{ID}}$ using the trained baseline's training distribution. Hypothesize $\mathrm R_{\text{Ours}}>\mathrm R_{\text{EAGLE-3}}$ and a smaller Ours/EAGLE-3 tau gap on OOD than ID. For the training-free method, retention reflects differences in task difficulty.

- **H9 (higher orders):** Some $n\ge2$ improves over order one, primarily at $j\ge2$.
- **H10 (optimal order):** Expected acceptance is unimodal over $n\in\{0,1,2,3\}$; empirical optimum agrees descriptively with Proposition 6's estimate using $\lVert\nabla^r h\rVert$.
- **H11 (predictor form):** EMA differences outperform unsmoothed extrapolation of the same order; online least-squares versus difference filtering uses a two-sided test.
- **H12 (adaptive vs optimal fixed):** On each confirmation set, adaptive Kalman expected acceptance is noninferior to beta optimized on that same dataset (margin $-\delta$, author confirmation pending), and superior to beta transferred from another dataset, avoiding retuning.
- **H13 (estimated Q/R predicts order):** Order predicted from $(\hat Q,\hat R)$ and Proposition 6 agrees with E13's empirical optimum; compare per dataset descriptively and report Spearman correlation.

---

##### 8. Experimental Plan

###### 8.1 Data

- **Main data:** Spec-Bench [14], 480 prompts: first 26 for selection, next 52, last 52, and full confirmation set.
- **Cross-domain (H8):** Define ID/OOD for each trained baseline using its training data:
  - EAGLE-family training is ShareGPT-like conversation, subject to the chosen checkpoint documentation [verification pending]; conversation subsets are ID.
  - OOD includes math, code, retrieval augmentation and summarization in Spec-Bench, plus long-context data such as LongSpecBench [45] [availability pending].
  - Optional: train an EAGLE-3 drafter on math only to replicate [44] [compute and author confirmation required].
- **Template:** Chat template on/off is a proposed factor in this historical draft.

###### 8.2 Models

| Model | Tied | $d$ | Mapping |
|---|---|---:|---|
| Qwen3-0.6B | Yes | 1024 | Hidden + scaling |
| Qwen3-1.7B | Yes | 2048 | Hidden + scaling |
| Qwen3-4B | Yes | 2560 | Hidden + scaling |
| Qwen3-8B | No | 4096 | $W_a$ adaptation |
| Qwen3-14B | No | 5120 | $W_a$ adaptation |

Tied status and dimensions come from HF configs. Public EAGLE-3 drafter coverage remains [unverified] and determines the H8 model selection.

###### 8.3 Baselines

- Autoregressive.
- ESP mean.
- Input-space extrapolation (Appendix B).
- Fixed offset $\pi_{\text{off}}$.
- Training-free: PLD, Lookahead [8], STAND [21].
- Trained: public EAGLE-3 [6], optionally Medusa [3]/Hydra [7], for H8 and reference only.

###### 8.4 Metrics

- Primary: total accepted tokens and expected tau.
- Acceptance by slot and node depth.
- Cross-domain retention $\mathrm R$.
- Wall-clock $S$ and $c_\Phi/c_B$ (historical proposal).
- exact_match = 1 gate (historical; superseded by the current verifier-argmax gate).
- Mechanism: mask input norms and activation statistics (C7).

###### 8.5 Experiment matrix ($k\in\{1,\dots,4\}$; all untested in this draft)

| ID | Type | Setting | Hypothesis |
|---|---|---|---|
| E1 | Baseline reproduction | Original ESP tree setting (appendix) | — |
| E2 | Main comparison | Ours vs ESP vs input extrapolation | H1 |
| E3 | Mapping | Scaling on/off, raw hidden, expected embedding, $W_a$ with lambda scan | H6, C7 |
| E4 | Order | ESP -> zero-only -> frozen first -> recursive first, by slot $j$ | H2, H3, H5 |
| E5 | Prefix conditioning | Conditional vs shared $(a^{(0)},v^{(0)})$, by depth $i$ | H4 |
| E6 | Space | Hidden vs logit (should match) vs distribution | Proposition 3 |
| E7 | Offset discrimination | vs $\pi_{\text{off}}$ | H7 |
| E8 | Scale/tied | 0.6B–14B; $W_a$ vs identity on tied models | H6, Proposition 4 |
| E9 | Sampling | Residual speculative sampling, $T\in\{0.7,1\}$ | Proposition 1b |
| E10 | Wall clock | $S$, $c_\Phi/c_B$ | Proposition 2 |
| E11 | Cross-domain | ID/OOD x {Ours, ESP, PLD, EAGLE-3}; tau, retention, speed | H8 |
| E12 | Failure analysis | Cold start, long context, OOD activation, BF16 | Boundaries |

**Cross-domain table template (E11)**

| Method | Trained | ID tau | OOD tau | Retention | OOD speed |
|---|---|---:|---:|---:|---:|
| EAGLE-3 | Yes | Untested | Untested | Untested | Untested |
| PLD | No | Untested | Untested | Untested | Untested |
| ESP | No | Untested | Untested | Untested | Untested |
| Ours | No | Untested | Untested | Untested | Untested |


**Higher-order ablations (all untested)**

| ID | Type | Setting | Hypothesis |
|---|---|---|---|
| E13 | Order | $n\in\{0,1,2,3\}$; beta per order in {0,0.5,0.9,0.99}; grid or sequential selection; slot breakdown; empirical difference norms | H9, H10, Proposition 6, C12 |
| E14 | Predictor | Difference filter vs fixed linear ($L=n$) vs online LS ($W\in\{16,64\}$, $L\in\{1,2,3\}$) | H11 |
| E15 | Gain/damping | Order-wise EMA vs gain form; proxy $K'$ vs $K$; block damping $\rho\in\{1,0.9,0.7\}$ | C11, Proposition 7 |

**Kalman/adaptation experiments (all untested)**

| ID | Type | Setting | Hypothesis |
|---|---|---|---|
| E16 | Adaptive vs fixed | Fixed beta grid (same-dataset optimum/cross-dataset transfer) vs steady-state Kalman with prompt moment estimates vs adaptive Kalman (covariance matching/Sage–Husa, windows16/64, forgetting0.95/0.99); compare optimal beta with $\beta^\star(\hat\phi)$ | H12, Propositions 8/9 |
| E17 | Model assumptions | Shared/per-dimension/grouped/prewhitened; fixed proxy-noise multiplier vs online $\hat R_\delta$; Mehra innovation-whiteness tests | §5.9, C12 |
| E18 | Order/uncertainty | Predict order from Q/R and compare E13; slot-variance/acceptance correlation; extensions adaptive $k$/temperature | H13, C13, C16 |

---

##### 9. Preliminary Results
The v4 main method has no measurements in this draft. Existing repository numbers concern input space, trees and $k=1$ (Appendix C), not evidence for this proposed method.

##### 10. Limitations

1. No measurements yet in this historical draft.
2. Cold start can only use the ESP mean.
3. Proxy hidden comes from a previous iteration under a different prefix, introducing error when draft and true prefixes differ.
4. Chain-end node $k$ has no proxy and requires extrapolation.
5. Distribution shift from soft inputs remains untested.
6. Absolute acceptance may trail trained in-domain methods; positioning depends on H8.
7. Residual sampling needs a new implementation.
8. Novelty requires systematic verification.

9. Propositions 6/7 assume independent noise and no smoothing; real hidden trajectories may violate them. Smoothed-error analysis is unfinished.
10. Higher differences are sensitive to BF16 precision.
11. Kalman variants assume dimension decomposition and linear Gaussian dynamics; adaptive Q/R estimates may be unstable on short sequences.

##### 11. Timeline & Venue (proposed)

| Phase | Work | Duration |
|---|---|---|
| P1 | Implement §5.11 subject to authorization; check hidden normalization; 4B first26 | 2 weeks |
| P2 | E2–E7 | 4 weeks |
| P3 | E8–E11 including EAGLE-3 cross-domain | 4 weeks |
| P4 | E12, tree appendix, writing | 3 weeks |

Suggested venues: ICML/NeurIPS/ICLR or ACL/EMNLP.

##### 12. Open Questions for Author

1. Is proxy-hidden alignment accepted? At the chain end, extrapolate or reuse the previous anchor?
2. Should draft-prefix velocity use the committed-history beta?
3. Confirm H8 ID/OOD and whether to train a drafter for [44].
4. Implement residual speculative sampling?
5. Model scope, dtype and budget.
6. Confirm denoising means filtering output-hidden trajectories (§1.5) and the paper positioning.

---

7. Choose independent beta per order (recommended) or gain form; separate proxy gain?
8. Is order<=3 sufficient? Put online LS in the main text or appendix?
9. Is Kalman the principal adaptive variant or main method? Set H12 noninferiority margin.
10. Shared/per-dimension/prewhitened decomposition?


##### References

1. Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. <https://arxiv.org/abs/2211.17192>
2. C. Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv 2023. <https://arxiv.org/abs/2302.01318>
3. T. Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. <https://arxiv.org/abs/2401.10774>
4. Y. Li et al. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML 2024. <https://arxiv.org/abs/2401.15077>
5. Y. Li et al. *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees.* EMNLP 2024. <https://arxiv.org/abs/2406.16858>
6. Y. Li et al. *EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test.* NeurIPS 2025. <https://arxiv.org/abs/2503.01840>
7. Z. Ankner et al. *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding.* COLM 2024. <https://arxiv.org/abs/2402.05109>
8. Y. Fu et al. *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.* ICML 2024. <https://arxiv.org/abs/2402.02057>
9. F. Gloeckle et al. *Better & Faster Large Language Models via Multi-token Prediction.* ICML 2024. <https://arxiv.org/abs/2404.19737>
10. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv 2024. <https://arxiv.org/abs/2412.19437>
11. X. Miao et al. *SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification.* ASPLOS 2024. <https://arxiv.org/abs/2305.09781>
12. R. Goel et al. *Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing.* ICML 2026. <https://arxiv.org/abs/2603.17942>
13. D. J. Bajpai et al. *FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching.* ICLR 2026. <https://arxiv.org/abs/2602.01329>
14. H. Xia et al. *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding.* ACL 2024 Findings. <https://arxiv.org/abs/2401.07851>
15. Z. Chen et al. *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding.* NeurIPS 2024. <https://arxiv.org/abs/2402.12374>
16. M. Stern, N. Shazeer, J. Uszkoreit. *Blockwise Parallel Decoding for Deep Autoregressive Models.* NeurIPS 2018. <https://arxiv.org/abs/1811.03115>
17. A. Santilli et al. *Accelerating Transformer Inference for Translation via Parallel Decoding.* ACL 2023. <https://arxiv.org/abs/2305.10427>
18. G. Monea et al. *PaSS: Parallel Speculative Sampling.* ENLSP Workshop @ NeurIPS 2023. <https://arxiv.org/abs/2311.13581>
19. N. Yang et al. *Inference with Reference: Lossless Acceleration of Large Language Models.* arXiv 2023. <https://arxiv.org/abs/2304.04487>
20. Z. He et al. *REST: Retrieval-Based Speculative Decoding.* NAACL 2024. <https://arxiv.org/abs/2311.08252>
21. W. Song et al. *Accelerated Test-Time Scaling with Model-Free Speculative Sampling.* EMNLP 2025. <https://arxiv.org/abs/2506.04708>
22. J. Zhang et al. *Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding.* ACL 2024. <https://arxiv.org/abs/2309.08168>
23. F. Liu et al. *Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting.* NeurIPS 2024. <https://arxiv.org/abs/2404.18911>
24. J. K. Christopher et al. *Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion.* NAACL 2025. <https://arxiv.org/abs/2408.05636>
25. G. Li et al. *DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding.* arXiv 2025. <https://arxiv.org/abs/2510.02358>
26. S. Agrawal et al. *Structuring The Future: Diffusion LLM Speculative Decoding via Calibrated Draft Graphs.* arXiv 2025. <https://arxiv.org/abs/2509.18085>
27. M. Samragh et al. *Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential.* arXiv 2025. <https://arxiv.org/abs/2507.11851>
28. S. Kou et al. *CLLMs: Consistency Large Language Models.* ICML 2024. <https://arxiv.org/abs/2403.00835>
29. G. Oliaro et al. *SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications.* NeurIPS 2025. <https://arxiv.org/abs/2411.04975>
30. H. Xia et al. *SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration.* ICLR 2025. <https://arxiv.org/abs/2410.06916>
31. S. Hao et al. *Training Large Language Models to Reason in a Continuous Latent Space* (Coconut). COLM 2025. <https://arxiv.org/abs/2412.06769>
32. Z. Zhang et al. *Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space.* arXiv 2025. <https://arxiv.org/abs/2505.15778>
33. Y. Zhuang et al. *Text Generation Beyond Discrete Token Sampling* (Mixture of Inputs). arXiv 2025. <https://arxiv.org/abs/2505.14827>
34. J. Zou et al. *Latent Collaboration in Multi-Agent Systems* (LatentMAS). ICML 2026 Spotlight. <https://arxiv.org/abs/2511.20639>; code <https://github.com/Gen-Verse/LatentMAS>
35. P. Wu et al. *Parallel Decoding via Hidden Transfer for Lossless Large Language Model Acceleration.* arXiv 2024. <https://arxiv.org/abs/2404.12022>
36. K. Pal et al. *Future Lens: Anticipating Subsequent Tokens from a Single Hidden State.* CoNLL 2023. <https://arxiv.org/abs/2311.04897>
37. N. Belrose et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv 2023. <https://arxiv.org/abs/2303.08112>
38. B. Lester, R. Al-Rfou, N. Constant. *The Power of Scale for Parameter-Efficient Prompt Tuning.* EMNLP 2021. <https://arxiv.org/abs/2104.08691>
39. F. Lin et al. *BiTA: Bi-Directional Tuning for Lossless Acceleration in Large Language Models.* arXiv 2024. <https://arxiv.org/abs/2401.12522>
40. Z. Xiao et al. *ParallelSpec: Parallel Drafter for Efficient Speculative Decoding.* arXiv 2024. <https://arxiv.org/abs/2410.05589>
41. H. Yi et al. *Generation Meets Verification: Accelerating LLM Inference with Smart Parallel Auto-Correct Decoding* (SPACE). ACL 2024 Findings. <https://arxiv.org/abs/2402.11809>
42. J. Tack et al. *LLM Pretraining with Continuous Concepts* (CoCoMix). arXiv 2025. <https://arxiv.org/abs/2502.08524>
43. LCM team. *Large Concept Models: Language Modeling in a Sentence Representation Space.* arXiv 2024. <https://arxiv.org/abs/2412.08821>
44. F. Bie et al. *Osprey: Target-agnostic Pre-training Makes Stronger Drafters in Speculative Decoding.* EMNLP 2026 (arXiv comment: accepted at EMNLP 2026). <https://arxiv.org/abs/2609.09338>
45. J. Lee et al. *OWL: Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs.* arXiv 2025. <https://arxiv.org/abs/2510.07535>
46. M. Hui et al. *P-EAGLE: Parallel-Drafting EAGLE with Scalable Training.* arXiv 2026. <https://arxiv.org/abs/2602.01469>
47. X. Liu et al. *Online Speculative Decoding.* arXiv 2023 (venue verification pending). <https://arxiv.org/abs/2310.07177>
48. H. Liu et al. *Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs* (HedgeSpec). ICLR 2026 (arXiv comment). <https://arxiv.org/abs/2510.20064>

Titles and first authors of [44]–[48] were checked via arXiv API on 2026-10-13. Numbers quoted from [44]–[46] come from their HTML full texts, not repository measurements.


49. R. E. Kalman. *A New Approach to Linear Filtering and Prediction Problems.* Journal of Basic Engineering, 1960. <https://doi.org/10.1115/1.3662552>
50. P. R. Kalata. *The tracking index: A generalized parameter for α-β and α-β-γ target trackers.* 22nd IEEE Conference on Decision and Control, 1983. <https://doi.org/10.1109/CDC.1983.269580>
51. J. Makhoul. *Linear prediction: A tutorial review.* Proceedings of the IEEE, 1975. <https://doi.org/10.1109/PROC.1975.9792>
52. C. C. Holt. *Forecasting seasonals and trends by exponentially weighted moving averages.* International Journal of Forecasting, 2004 (reprint of the 1957 report). <https://doi.org/10.1016/j.ijforecast.2003.09.015>
53. E. S. Gardner Jr. *Exponential smoothing: The state of the art.* Journal of Forecasting, 1985. <https://doi.org/10.1002/for.3980040103>
54. R. J. Hyndman, A. B. Koehler, R. D. Snyder, S. Grose. *A state space framework for automatic forecasting using exponential smoothing methods.* International Journal of Forecasting, 2002. <https://doi.org/10.1016/S0169-2070(01)00110-8>

Titles, venues and years of [49]–[54] were checked through Crossref on 2026-10-13. A textbook source for Newton differences remains missing. The original damped-trend attribution (usually Gardner & McKenzie) is unverified; only review [53] is cited.

55. J. E. Gray, W. J. Murray. *A derivation of an analytic expression for the tracking index for the alpha-beta-gamma filter.* IEEE Transactions on Aerospace and Electronic Systems, 1993. <https://doi.org/10.1109/7.220956>
56. R. K. Mehra. *On the identification of variances and adaptive Kalman filtering.* IEEE Transactions on Automatic Control, 1970. <https://doi.org/10.1109/TAC.1970.1099422>
57. R. K. Mehra. *Approaches to adaptive filtering.* IEEE Transactions on Automatic Control, 1972. <https://doi.org/10.1109/TAC.1972.1100100>
58. A. P. Sage, G. W. Husa. *Algorithms for sequential adaptive estimation of prior statistics.* 1969 IEEE Symposium on Adaptive Processes (8th) Decision and Control, 1969. <https://doi.org/10.1109/SAP.1969.269927>
59. A. H. Mohamed, K. P. Schwarz. *Adaptive Kalman Filtering for INS/GPS.* Journal of Geodesy, 1999. <https://doi.org/10.1007/s001900050236>
60. P. R. Kalata. *The Tracking Index: A Generalized Parameter for α-β and α-β-γ Target Trackers.* IEEE Transactions on Aerospace and Electronic Systems, 1984. <https://doi.org/10.1109/TAES.1984.310438> ([50] is the same-title 1983 CDC version; use the journal version for the tracking-index formula, original full-text verification pending).

Titles, venues and years of [55]–[60] were checked through Crossref on 2026-10-13. Qualifications:

- Kalata's alpha-beta closed form in §5.9 was checked only through secondary sources.
- Sage–Husa and covariance-matching updates follow common citations; original details remain unverified.
- Textbook sources for Joseph form and DARE convergence remain missing.
---

##### Appendix A: six-part research-standard coverage

| Aspect | Location | Status |
|---|---|---|
| Literature | §3, References (60 entries) | Metadata checked; novelty unverified |
| Data | §8.1 | Cross-domain split unverified |
| Models | §4.1, §8.2–8.3 | Tied checked; all untested here |
| Algorithms | §4.2, §5.1–5.10, §6, §5.11 | Formalized; code unchanged |
| Metrics | §4.3, §8.4, §7 | Implementation pending |
| Experiment design | §7 H1–H13, §8.5 E1–E18 | All untested in this draft |

##### Appendix B: input-space extrapolation control
Maintain states in input-embedding space:
$$\hat v^e\leftarrow\beta\hat v^e+(1-\beta)(e_{\text{new}}-e_{\text{prev}}),\qquad m_j=\hat e+\gamma c_j(\rho)\hat v^e.$$

- With long memory, $\hat v^e\approx d_1$, reducing to a fixed offset (C1).
- Input-space prefix conditioning is exact: draft token embeddings are known before forward, allowing the input anchor and velocity to fold along the draft. This controls the prefix-conditioning effect.
- Repository `extrapolate_ema: true` partly implements recursion with $\beta'=\beta$.

**Tree extension:** Node prefixes follow ancestors; Proposition 1 generalizes to ancestor-only attention.

##### Appendix C: existing repository measurements (input space, tree [14], k=1)
Source: `docs/history-extrapolation.md`.

Conditions: Qwen3, FP32, greedy, one seed; exact_match=1 in all cells. Unrelated to the v3 output-space method.

| Model | Slice | Frozen mean | Updating mean | Selected input first order |
|---|---|---:|---:|---:|
| 0.6B | first 26 / next 52 / last 52 | 726 / 1550 / 1598 | 701 / 1473 / 1468 | 723 / 1553 / 1575 |
| 1.7B | first 26 / next 52 / last 52 | 778 / 1747 / 1723 | 757 / 1639 / 1626 | 780 / 1709 / 1719 |
| 4B | first 26 / next 52 / last 52 | 680 / 1462 / 1476 | 704 / 1479 / 1525 | 777 / 1686 / 1723 |
| 4B | Full480 | 12809 | — | 15009 |

See v2 Appendix C for beta/gamma scans and tree [7,2].

</details>

<a id="legacy-studies--template-docs-plan"></a>

### studies/_template/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/_template/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/_template/docs/PLAN.md); original SHA256 `abc881fbcdb43e8ccbd622541a551e76e79c10a4af8a037428dba27e600060d1`.

#### Historical Study-plan template

A blank template is neither experimental evidence nor authorization. After copying, specify the actual question, prior conclusions, primary metric and necessary AR/ESP/EMA/alignment comparisons; do not present unimplemented external baselines as supported or differing paper protocols as complete reproduction. Define a new Study, axes (or one combination), each combination×seeds, total Runs and version. Base→fixed→axis/seed precedence must match expanded configs/index. Seeds are not concurrency. List complete existing cells that should not rerun by default.

Record repository commit, RPipe ORIGIN, recipe, specialized lock and extra inputs; model/tokenizer identity, hub/revision/files/digest and frozen-weight semantics; data/source/slice/template/tokenization/input digest; BC/mask initialization/tree/pruning/β/λ/γ/acceptance/fallback/KV contract; temperature/seed/new-token/EOS/context/prompt limits; actual dtype/TF32/mixed precision/backend/dependencies/device/random flags; metric/denominator/depth diagnostics and distinct correctness versus speed. The original template required exact_match_rate=1, a historical rule superseded for currentBF16 studies. ExampleFP32/T0 YAML is not the scientific target. Never select a backend lacking required4D tree masks. Source/recipe changes require new versions and explicit make/remake, not re-signing success.

Historical scheduling: measure actual weight/KV/tree/mask/process capacity serially with make--round1 before recording vram_bytes or increasing independent-Run concurrency; visual-training estimates are not decoding calibration. Each Run owns state/artifacts/logger; wait for complete group exits and avoid duplicate launch chains. Any speed comparison would require equal work, serial execution, warmup/sync/backend/generated length/wall-time records; current no-timing policy supersedes that optional template advice. Retain failure IDs,exit codes/logs; fresh recomputation uses a new version. KV/mask/RNG checkpoint resume is unvalidated. Record estimates after make, actuals afterward, using each group's slowest estimate and summed groups; never fabricate missing times.

Recipe/specialized Algorithm construct inputs/model and execute raw observations; collect computes metrics; summarize checks the historical exact/evidence gate; write verifies sources before library finalisation; Study process verifies only the current index. Completeness requires every declared factor/seed success, original token/AR contract, complete source/config/input/model/runtime/acceptance/cache evidence and same-device old/new numerical prechecks; no passing subsets, mixed sources, selected repeats or relaxed thresholds. Human reports give Experiment mean/std/min/max/n,units/conditions/conclusions and real figures; NUMBERS is not a conclusion. Separate static,CPU tiny,realGPU,installation,recovery/parallel validation, and disclose local-only logs. This template is now consolidated here; new work uses a project-plan section rather than another PLAN file.

</details>

<a id="legacy-studies-esp-table1-bf16-full-docs-plan"></a>

### studies/esp_table1_bf16_full/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/esp_table1_bf16_full/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_full/docs/PLAN.md); original SHA256 `0d8ac40e958037c9aef53bf212dc3157a23ee98277134eb28de016e2f3e10c0a`.

#### Historical full480 ESP protocol before the argmax-gate revision

The complete common contract is consolidated in the full_argmax chapter: ESP v2 sections/tables/G.2,ModelScope8B BF16/T0, original Spec-Bench commit and both formatting hashes, 256/EOS versus the paper's100/256 and A100/H100 ambiguity, local non-thinking first-turn choices, unknown author revision/precision/weights, native naive dynamic Top1/pruning, per-token prompt-mean updateλ.1,seed123,no timing, all480 and three aggregations. This version already had PR7 merged at8677376e1ac878a159a3095711eac43420d97bde and paused0.6B/4B FP32 work. BC10/1,30/1,60/1 are primary (draft4/14/29);60/2 is predeclared control (draft19). G.2's Qwen-specific statement motivates single-mask60, with the Table3/Figure8-versus-Table1 limitation explicitly retained. Reuse old three-cell preflight by frozen hashes and add only60/1, never select masks or erase mismatch.

Unlike the later argmax revision, this historical version retained native exact_match_rate=1 even while running all480 despite preflight mismatch. A failed cell'sτ was diagnostic and could not be called successful; preserve output/AR hashes, first divergences, native statuses and complete negative results. Repair real source/model/protocol/capacity problems with a new Run, notFP32,teacher correction, sample removal or gate relaxation. The full order was10/1→30/1→60/1→60/2, with bothBC60 variants always included. Default numerical policy matched the original preflight rather than mixing reduction/shape interventions.

Run only after preflight exit and approximately19GiB free, one round1 native chain, releasing models between cells and disabling retries. Four preflight cells have26624 generated-token upper bound (only new60/1 adds6656); four full480 cells with freshAR have983040, no warmup/teacher/timing. Commands used native make and launch_once against esp_table1_bf16_full. Retain every failure/success/source lock without re-signing, keep raw Runs/tokens/weights/cache local, and report all threeτ aggregates/target differences rather than selecting the closest. Four-BF16 five-method+AR work follows only afterward under the common contract, including tied scaling without inverse and validated owner/proxy extension before using dynamic/pruned trees.

Historical recorded SHA256 values (source ordering; identities remain in the immutable original):

- `4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04`
- `7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21`

</details>

<a id="legacy-studies-esp-table1-bf16-full-argmax-docs-plan"></a>

### studies/esp_table1_bf16_full_argmax/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/esp_table1_bf16_full_argmax/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_full_argmax/docs/PLAN.md); original SHA256 `88f524f6f33eda74c43792826a8a80f91f550261d3ce3943b7fc595e2d005e4c`.

#### ESP Table1 Qwen3-8B/BF16: full480 and argmax contract

Historical2026-10-15 priority:4B FP32 tools/tests and evidence were committed separately; PR7 passed Branch flow/cpu-checks and mergeddev at8677376e1ac878a159a3095711eac43420d97bde. No main. Suspend0.6B rare-event/mechanism and4B FP32 extensions. Source: ESP arXiv2603.17942v2§3.1/3.4/3.5,§4,Tables1/3,AppendixD/G.2; body and localPDFp7 were read. Prefer ModelScope Qwen/Qwen3-8B, restricted at that time to ESP reproduction. Table1 targets:BC30τ1.63,BC60τ1.74; BC10 has noTable1target. Use the caption's256-token/H100 setup rather than silently substituting§4's100-token/A100:256 is a cap with EOS and actual lengths recorded. LocalRTX5090Dv2/24GB and library versions differ; noS/R timing. Author precision,immutable revision,thinking,template/truncation and task-weighting contract remain unspecified; BF16/non-thinking first turns/no filtering are explicit local choices, not author confirmation.

Spec-Bench sourcehemingkx/Spec-Bench commitfd2c1cd7d2201ef71db4c5f4e455008f017967bf,data/spec_bench/question.jsonl. All480 parsed objects/order match the pinned source; remoteSHA4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04 versus local7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21 differs only in formatting. Retain both identities and never remove difficult samples.

G.2/PDFp18 says Qwen3BC60 uses one mask and LLaMA3 two. This model-specific statement supersedes generic§4 defaults; Table3 single-mask width[29] givesBC=2(1+29)=60, though its row isLLaMA andG.2 discussesFigure8,not an explicitTable1mask declaration. Thus60/1 is a documented protocol judgment, not author confirmation;60/2 remains a predeclared control, never selected by results. Older60/2 preflights/numerical controls stay controls with original bytes/Run IDs; revise with newPlan/Run for60/1 and frozen-hash reuse of completed old cells. Preflight/full reports list bothBC60variants,target differences,AR diagnostics and pending rather than zeros.

FixedBF16/T0,mean_prompt initialization and per-generated-token m←m+0.1(e−m),native naive dynamic attention/cache,Top1 expansion/cumulative-path ranking,parent-repeat replacement,pruning=true,seed123,max256/EOS,warmup=false,all480first turns,enable_thinking=false,max_prompt_len40960 with no lost examples. Main masks10/1,30/1,60/1 give4/14/29draft nodes;60/2gives19. BC=(k+1)(1+draft_nodes); check actual block budget/tree widths, disclose Algorithm1 B versusEq7 ambiguity rather than changing budgets to chase target. Study-local decoding clocks return zero sentinels; do not change global clocks or report measured speed.

Report pooledΣcommitted/Σdecode_calls,promptmean and equal-category pooledmean without selecting the closest. Committed includes bonus,excludesprefill and respectsEOS/cap trimming. Separately report trueprefill+decode model calls,new tokens,preclipaccepted and target raw differences/two-decimal agreement. No claim that an aggregation is the author's exact denominator.

Use native prepare→execute→collect→summarize→write→process with newStudy/version/Run; verify weightSHA/ModelScope filenames/sizes/actualBF16 and disclose unknown cache revision. First13prompts(one/category)×fourcells establish loading,dynamic-tree,256/dtype/native-gate behavior,not fullTable1. Freeze and serially run four full480cells afterward. Under the updated full-study contract,AR mismatch is visible and separate,not aτ execution gate; complete same-forward argmax is mandatory. Preserve native oldAR-gate failure statuses. Source/model/protocol failure orOOM requires actual repair/resources and a newRun,notFP32,teacher correction,omitted counterexamples or weakened gates. Retain negative results. Wait for resources; never stop other tasks or load models concurrently.

Four tasks shared24GB. Record initialfree/device capacity samples(including background,not exact allocator peaks),not disturbed wall-time or timing placeholders. Preflight upperbound13×4×2×256=26624tokens,with only the new60/1cell adding at most6656; fullfourcells withfreshAR upperbound480×4×2×256=983040. No teacher/warmup. Initial full-study launch required preflight exit and≈19GiBfree,one round1 chain,eachmodel exit before next cell,no automatic retries. Historical commands referenced esp_table1_bf16_full; preserve failed/successful versions without re-signing. Public reports contain settings/counts/dtype/AR exact/outputSHA/first-divergence summaries; completeRuns/tokens/cache/weights stay ignored locally.

After four full480cells and correctargmax,plan the4B BF16 five-method+AR pair under identical prompt/tree/budget/length/seed:ESP frozenmean versusinputupdatingmeanλ.1,inputmean+gamma,outputfinal-normalizedhidden norm mapping,andoutputhidden+EMAvelocity. Frozenmean is user-defined,not paperEq5. Tiedmodels use directscale mapping,notinverse. The then-currentoutputprovider supported only an unprunedstaticchain: extendingdynamic/prunedowner/proxy mapping must be minimal and validated before faircomparison,with beta/gamma/internalprecision/actualtiedstorage frozen separately. Do not start4B early or substituteFP32.

2026-10-15argmax amendment stoppedcompactattention,single-rowRMSNorm,layerwise/reduction/shape mechanism work; existingnegative reports/FP32labels were committed. v2keepsv1weights/input/defaultreduction/tree/mask/sampling and adds only scoped observation: bindselect_token to latestforward logits storage,forwardnumber/row/argmax/chosen; coverprefill and everyEOS/cap-trimmedactualcommit. Check trimmedtail selections too,but do not count them ascommittedcoverage orτ. No extraforward,weight/operator/dtype/candidate/samplingchange;ARhasnoobserver. FlushperpromptJSONL andembedfullauditstats. v1completedlogs lackedargmax,so retainRun/frozencontract/logs and use newversionforcompletefourcells; stop only ownoldlaunch/run processes,nototherprojects. HistoricalnativeARstrictgates may stillfail summarize; launch_once disablesretries and proceeds,whileτreads completeexecute/collect. Separatelifecycle,AR firstdifferences andargmaxcoverage;nevercallnativefailure success or claimindependentARexactness.

</details>

<a id="legacy-studies-esp-table1-bf16-full-argmax-docs-third-party-plan"></a>

### studies/esp_table1_bf16_full_argmax/docs/THIRD_PARTY_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/esp_table1_bf16_full_argmax/docs/THIRD_PARTY_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_full_argmax/docs/THIRD_PARTY_PLAN.md); original SHA256 `d6ceece7973d504235a29e8f89c177d543f1b0e9bcfba38222f72f024ee29821`.

#### Byte-preserved nonofficial third-party ESP acceptance (2026-10-15)

Authorized sourceHarshvardhan-Mestha/esp-reproduction-artifacts commitad76f8da29f3bfc7faa20488193c6aa83a9f708c is not author code. Preserve all fourcode/entry files in src byte-for-byte,plusREADME/resultsJSON,withORIGINper-fileSHA and omittednonexecutingposter/logbook manifest. Keep fullclone local; do not alterentryorrepairalgorithm. Originalrun.shinstallsdependencies andrunnerhardcodesFP16,tenexampleprompts,BC60/twomasks andtiming,so do notexecute those actions. External acceptance calls originalESPDecoder.generate_esp using existingModelScope8B BF16/eager,originalTable1frozen480manifest,identicalinputIDs,T0greedy,max256,BC10/30/60singlemask,seed123. Keep its originalflatcandidates,draftsoftmaxT1,noKVreuse: this is external compatibility/correctness acceptance,not fullpaperdynamic/pruned equivalence. Replace only module-localtimingwithconstant,not globalclock; noAR/TPS. Bindweights/model,data,prompt,source/tool/protocolSHA.

Pilotonlyfixedfirstq121,threebudgetsserially,max256unchanged,atmost24modelforwardspercell. Mathgate precedesfull480; failedoriginalalgorithmisnotrewritten or fixedbycandidate/datachanges. Inspect sameverificationforwardlogits for theactualprefix abouttobeappended:firsttokenuseslastpromptrow,lateronesuseprecedingcommitteddraftrow,notdraft-selformask-with-uncommitted-candidatesrows. Everyplannedcommittokenmustmatchparent-rowargmax; rejectbefore firstbadappend,retainposition,margin,forwardcounts/privateIDs. No independentARoroperatorstudy. Originaldecoderappendsall drafts throughlastself-matchindex,so checkthewholeplannedprefix ratherthanonlyreportedn_acc positions. Countproposalforwardsseparately; failedvalid_tau=null,notupstreamdeclaredτ,andnopaperdifference. Compatibility/resourcefailurealsoretainedwithoutautoretry/sourcefix. Otherbudgets canindependentlypilotthesameq121onthesamemodel,totalcap72forwards. Explicitlymarkfull480unexecuted; upstreamresultsareexternal,notlocalmeasurements.

Historicaladmission:noforeignGPU Python,≥20GiBfreebeforeload,recheckeachbudget; noCPUinspection/SHA,backgroundwaiter/overlappingqueue/preemption/killing.8BonlyforthisTable1attempt,existingModelScopecache,noCUDAreplacement/venv. Reportsourcepaths,numbers,incomparablesettings,nonofficial/failed/unexecutedboundaries. Commit/push/PR→devanddeletemergedremote;retainlocalbranches/cache/weights/tokens/failures;nomains/force/spending/popups.

</details>

<a id="legacy-studies-esp-table1-bf16-numerics-docs-plan"></a>

### studies/esp_table1_bf16_numerics/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/esp_table1_bf16_numerics/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_numerics/docs/PLAN.md); original SHA256 `c1d91a12ec50c6ee1d9fd26151e8754e23067478cd5e4ef28150d43944878538`.

#### Retired BF16 numerical-gate isolation

Historical8B ESP-only work, now archived and not authorized to resume. InitialBC10 preflight had6/13 AR-exact; q81 first differed at new-token index9 (40233 versus20397). Fix q81/checkpoint/prompt,BC10,dynamic/pruning,λ.1,T0 and cap32. Compare a2×2 of allow_bf16_reduced_precision_reduction true/false and deterministic false/true. Weights/activations remainBF16; native HF RMSNorm/softmax internal high-precision reduction stays unchanged, without aFP32 model, decoding clocks or warmup.

Set CUBLAS_WORKSPACE_CONFIG=:4096:8 identically before first CUDA computation. The first repeated-original-condition cell checks whether this environment/length change alters the early divergence, not re-signing the old Run. Separate main effects/interactions and never label a combined result as one kernel's cause. Record actual flags and restore prior values after success/failure. Persistent divergence rejects sufficiency; disappearance on one short question still requires13 questions/three budgets before full480. Preserve strict historical exact_match_rate=1 and every failed output/Run; total generation bound4×1×2×32=256, with checkpoint hashing separate. One shared24GB queue starts only after all original preflights exit/release; noFP32, speed,teacher forcing or sample selection. Bind prior rawSHA/token manifest and16 model files. v1 was made but never ranGPU; injected CPU failure showed patch.object could not delete/restore a CUDA backend attribute. v2 uses explicit try/finally assignments and a new version/Run, retaining the unexecuted v1 directory.

</details>

<a id="legacy-studies-esp-table1-bf16-preflight-docs-plan"></a>

### studies/esp_table1_bf16_preflight/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/esp_table1_bf16_preflight/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_preflight/docs/PLAN.md); original SHA256 `d19405874a3b628648f52c5bb817fbe0a66d2173fc3669595de90ca5b1931ba8`.

#### Historical ESP8B BF16 three-cell preflight

Uses the common full480 ESP contract in the full_argmax archive for papersections/targets,caption256/H100versus§4 100/A100ambiguity,unspecifiedauthorprecision/revision/thinking/truncation/taskweights,Spec-BenchpinnedcommitandtwoformattingSHA,ModelScope8B/RTX5090Dv2/24GB,BF16/T0,mean_prompt/per-tokenλ.1,dynamicTop1/cumulative-ranking,parent-repeatpruning,nativenaiveattention/cache,seed123,max256/EOS,warmupfalse,first-turnnonthinking,maxprompt40960,threeaggregations,prefill/bonus/clippingandnontimingreporting. This earlier2026-10-15plan had4BFP32commits pushedandPR7awaitingCPUCIbeforedev;0.6B/4BFP32workpaused. BC10hasnotarget;30/60target1.63/1.74.

Version-specific differences: BC10/1,30/1,60/2withdraftbudgets4/14/19;thisisbeforeQwen3BC60single-maskcorrection. Onlyonepromptpercategory,13total,atthreebudgets;notTable1full480. Initialfree21408MiBwith2628MiBbackground,nonperformancemeasurement. Preflighttokenupperbound13×3×2×256=19968;laterfullthreecellswithAR480×3×2×256=737280. Verifyactualblockbudget/treewidths,weightsSHA/ModelScopefilenames/sizes/BF16andunknownrevision;allnegativeRunsevidencekept.

The originalstrictgate required independentARtokenexactnessandnativeexact_match_rate=1beforefull480. Onmismatchretainfirstdivergence/rawtokensandmakeasame-precisionminimalreproducer,neverFP32/teacher/sampleselectionorweakenedgate. ThisstrictAR requirement is historicalandwaslaterexplicitlysuperseded; do notreactivateoperatorwork. OOM/foreignGPUoccupancyretainsfailureandwaitsorusesnewRun,neverkillothers;onequeue,noparallelmodels. Onlyafterpreflightexactnesswouldanewfull480Study/versionrunseriallywithallthreeτaggregations/targetdifferencesandfailurebounds,notforceasuccessclaim.

Later4Bfive-method+ARphasefollowedthecommonarchivedfull_argmaxdescription: identicalprompt/tree/budget/length/seed,ESPfrozenmeanversusupdatingmean/inputfirst/output0/output1,realBF16,actualtied-storage,normmappingwithoutinverse,andvalidateddynamic/prunedownerextensionbeforefaircomparison.4Bwasqueuedonly,withitsownthen-requiredfrozenplan/beta/gamma/internalprecisioncontract.

Historical recorded SHA256 values (source ordering; identities remain in the immutable original):

- `4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04`
- `7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21`

</details>

<a id="legacy-studies-esp-table1-bf16-qwen-protocol-docs-plan"></a>

### studies/esp_table1_bf16_qwen_protocol/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/esp_table1_bf16_qwen_protocol/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_qwen_protocol/docs/PLAN.md); original SHA256 `782d0022f811ca80ca716de8b8cf89007df16aa758bfaec0dcc5fb3f59442418`.

#### Qwen3-specific BC60 preflight correction

Inherit the full common ESP contract consolidated in full_argmax: paper/model/data sources and byte identities, original480 first-turn non-thinking settings, BF16/eager/T0,256/EOS,seed123,mean_prompt/per-tokenλ.1,dynamic Top1/pruning,native naive attention/cache, target30=1.63 and60=1.74, three aggregations and explicit bonus/prefill/clipping definitions. Precision/revision/thinking/truncation/author weighting and H100/A100 versus100/256 remain unconfirmed; no performance claim. PR7 had passed and mergeddev at8677376e1ac878a159a3095711eac43420d97bde; old0.6B/FP32 studies stayed paused.

This is the revised13-question preflight, not full480. Preserve the original preflight plan/Run bytes. Add only the previously unrun60/1 cell with29 nodes under a new version/Run; reused_preflight.json pins old10/1,30/1,60/2. Reused results are not new successes, and all four table rows remain fixed, including both60 variants' pooled/prompt/categoryτ differences from1.74 and AR quality. G.2 says one mask for Qwen3 and two for LLaMA3; Table3 width29 givesBC60, but its LLaMA row and G.2's Figure8 context do not prove the Table1 setting. Keep this protocol judgment and avoid result-dependent mask selection.

This historical preflight still used fresh independentAR and native exact_match_rate=1; mismatchedτ remained diagnostic. Do not alter native reduction/determinism or mix reduction/shape controls into the same-setting table. A later separate full480 Study freezes all four cells and preserves preflight mismatch as a known risk, rather than using the small sample as full evidence. Only real source/OOM problems first require repair. Single native launch chain, one new Run, approximately21GiB free, no stopping others, no automatic retries or vendored RPipe changes. New-cell upper bound13×2×256=6656, no warmup/teacher/clocks. Native make/launch_once used esp_table1_bf16_qwen_protocol. The later4B plan shares the common five-method/tied/owner-extension requirements and was not executed early. All source,negative and native lifecycle records stay preserved.

Historical recorded SHA256 values (source ordering; identities remain in the immutable original):

- `4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04`
- `7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21`

</details>

<a id="legacy-studies-esp-table1-bf16-reduction-check-docs-plan"></a>

### studies/esp_table1_bf16_reduction_check/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/esp_table1_bf16_reduction_check/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_reduction_check/docs/PLAN.md); original SHA256 `28d3aa91fd0c0ddf901332814825e4b12df5b50bf48e45e2fcb60d277abc2fa2`.

#### Retired reduction-control expansion

Historical8B ESP-only control, now archived. Original13-question/three-BC AR-exact counts were6/6/5; q81 first differed at index9. In the q81 2×2, reduction=false/deterministic=false made32 tokens exact with unchanged AR. Fix original13,BC10/30/60,max256/EOS,T0/eager,dynamic/pruning,prompt-meanλ.1 updating,checkpoint/input/source. Only allow_bf16_reduced_precision_reduction=false changes; deterministic remainsfalse. CUBLAS_WORKSPACE_CONFIG=:4096:8 is fixed, with the original-condition q81 control confirming no32-token effect from it.

Model and ESP states stay nativeBF16; native RMSNorm/softmax internalFP32 reduction remains. Record actual flags and preserve historical freshAR token-exact gate. Serial three cells, no automatic retries; all must pass before a separate new-version full480 triple. Any mismatch retains counterexamples, without filtering,target chasing,teacher correction,FP32 or4B advancement. Even13 passing questions are not Table1 success or author-confirmed backend. Upper bound13×3×2×256=19968, no warmup/teacher/timing. Start only after all four numerical cells exit, using one24GB queue; capacity samples include background, not speed. Preserve raw/model/data/token hashes, failed Runs and unexecuted v1; do not re-sign.

</details>

<a id="legacy-studies-esp-table1-bf16-shape-check-docs-plan"></a>

### studies/esp_table1_bf16_shape_check/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/esp_table1_bf16_shape_check/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_shape_check/docs/PLAN.md); original SHA256 `0e16b11ffb543a5f634b6de244dd26b7700865c84b2636ad364aa2856f7e423f`.

#### Retired q81/BC10/96-token shape control

Historical8B ESP token-exact investigation only; no new mask,4B or Kalman work, and not authorized to resume now. Reduced-precision reduction=false/deterministic=false made q81's first32 exact, but only4/13 originalBC10 questions at256 were exact; q81 diverged again at index74. The historical hypothesis was that differing batch-row counts choose different BF16 linear GEMM reductions.

Two fixed checkpoint/q81/input/seed123 cells useBF16,T0/eager,BC10 one mask,dynamic/pruning,λ.1 updating mean,cap96/EOS and the same disabled reduction/false determinism. Only the second cell runs each nn.Linear's original forward row-by-row and concatenates its original shape, for bothESP andAR. Keep weights/activation dtype,RMSNorm/attention/RoPE,tree/cache/mask/acceptance intact. The first native cell checks the256→96 prefix. Parent LLM-forward counts stay native, but internal GEMM calls increase; no speed or compute-cost claim. This is an explicit numerical control, not author implementation or a general fix. Report if AR itself changes and compare only each freshAR, never force agreement with the old sequence.

Persistent failure means linear batch shape is not sufficient; the old proposed attention/norm follow-up remains historical and superseded by the later prohibition. Even success would require13 questions/three budgets then full480. Run serially only after all reduction-check cells exit; no retries. Bound2×1×2×96=384 generated tokens, no warmup/teacher/timing. Retain failures/sources and new version/Run; ExitStack restores patches on normal/error exits. CPU regressions coverBF16 row splitting/permutation,dtype/unchanged weights and exception restoration.

</details>

<a id="legacy-studies-input-mask-baseline-docs-plan"></a>

### studies/input_mask_baseline/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/input_mask_baseline/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/input_mask_baseline/docs/PLAN.md); original SHA256 `1e56b2024b6dedcbc8867471bc55db2308a303c356fee879c5fda65d6c6700dc`.

#### Historical input_mask_baseline prerequisite

Question from research extraction/v6§5.11/E1–E2: do pretrained frozen_mean/gamma_last match AR token-for-token with reliable acceptance/cache/source/phase evidence? Any token mismatch,weight/runtime drift or missing evidence rejects this prerequisite; increased acceptance is not a gate and H1–H13/novelty are not tested. Four complete algorithm.config patches: ESP frozen_mean,ESP gamma_last,AR,alignment frozen_mean; seeds[123],four Runs. Preserve common eval/version through recursive expansion; final version input-mask-baseline-v031-ms-20261014-v4.

Data: local SpecBench translation skip10/take1,q171,first turn,Qwen non-thinking template,max_prompt_len2048. One prompt,not selected for tuning or a rerun of historical first-ten/category results. ModelQwen3-0.6B explicitly downloaded from ModelScope under the2026-10-14 instruction into shared/model/modelscope; Run local_files_only=true,no execution-time replacement/download. Freeze actual model/tokenizer/config hashes; mutable master is not immutable revision. Original scientific recipe isFP32/eager/CUDA,recording TF32/matmul/random flags,T0,max100,EOS,seed123,AR/per-prompt observations. Do not globally cast other models. One mask,static[14],pruning,BC30; gamma_lastβ.9/λ.1/γ1 and frozen mean/no update retain original semantics,including training-free mask-autograd boundaries. Alignment uses four hindsight steps,frozen mean,separate layer cosines/sums/counts; it is not the100-Dolly Figure2 experiment.

Pin RPipev0.3.1/7a15c46 with ORIGIN/license and only the Windows torch-first exception. Declarations,recipe,phase code and the then-separate plan enter provenance; SpecMTP manifest adds package sources,while raw captures data/model/tokenizer/input/config/runtime. Reject source adds/deletes/edits and finalisation drift. Changes require new version/make,never re-sign success. Native prepare/execute/collect/summarize/write/process verify current four cases×seed. All Runs and ESP exact_match_rate=1 plus complete/source/specialized quality must pass; AR needs real output/summary and alignment real sums/counts,not fake speculative exactness. Report pretrim all-depth accepted,depth acceptance,new tokens,committed,calls,τ separately. n=1 proves neither stability nor mechanism; no speed ranking.

Before these four real Runs, pass source imports,offline Toy/Stub template gates,isolated tiny/failure regressions and36 original-trajectory controls. One GPU round1 queue,exclusive state/artifacts,record VRAM/host/thread/I/O/peak conditions; no uncalibrated concurrency or decoding resume. Preserve make estimates and actual observations without substituting one for the other. The historical cache preparation copied .cache/modelscope/models/Qwen--Qwen3-0.6B into Study shared/model/modelscope/models while retaining snapshots/master/files; weights stay out of Git. Use native make/launch/process/status/logs/report,UTF8,PYTHONPATH=src,OMP_NUM_THREADS=1. OOM/numerical/source failures stop expansion and retain evidence. Scope is four cases/one prompt/seed123/100 tokens,not tuning,new output implementation or a full research matrix. Separate execution completion,flow quality and scientific support; retain raw/local-only assets and selected publicJSON/figures/reports without rewriting old failures.

New-device preparation: upstream0.3.1 and downstream report-quality writeback changed sources,so v1/v2/v3/v4 use new IDs; v1 had no real successful cell. Keep q171/0.6B/FP32/eager/CUDA/seed123/100/EOS/cases. DeviceRTX5090Dv2,torch2.11.0+cu130,CUDA13,Transformers5.12.1. Resource inventory is not real model calibration. Initially no required snapshot existed in project,default ModelScope or usualHF caches; do not substitute another model,launch,or fabricate tokens/acceptance/peaks. Preserve v2's missing-cache status and four unexecuted Runs.

User's 'download from ModelScope' response authorized explicit snapshot_download before make/Run,under v3,with provider/cache/file hashes and mutable-master warning. Resolve offline afterward and use one snapshot for all four cases,not old-device-cache attribution. Load-only calibration:596049920 parameters,FP32/eager/cuda:0,allocator peak2384724992 bytes,sampledRSS5139087360 bytes,intra-op threads1. Freeze calibrationJSON/snapshot hashes before running. Load-only cuDNN benchmarkfalse had no cuDNN computation; formal deterministic=false System sets benchmarktrue and records it,without overwriting science from calibration defaults. Run the first frozen_mean once via native run-one while observing resources; native launch then skips its success and runs remaining three. External0.5s RSS/thread/I/O/device-memory samples include background and can miss short peaks; separate sampled decode capacity from allocator load peak,not performance.

B-008: v3's four Runs succeeded (bothESP exact,realAR,complete alignment),but Study finalisation failed because nested-axis expected factors were compared wrongly. Retain v3 failure and successes. The fix independently derives expected factors from frozen base/declaration using native merge/grouping,without changing decoder/Data/Model,axes,input,precision,budget or numerical gate. After nested-axis/subset CPU and engineering checks,v4 reruns the same four cases with new IDs; never re-sign v3. Continue validated serial loading/resource observations and compare all recorded values except wall-time.

</details>

<a id="legacy-studies-output-space-4b-holdout-docs-plan"></a>

### studies/output_space_4b_holdout/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_4b_holdout/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_4b_holdout/docs/PLAN.md); original SHA256 `1ef5c46f3c3a824e45ffff9f354c4def5f3c7fc281cd1ad19c44df030603ff29`.

#### Qwen3-4B matched output-space recheck

Frozen before new model forwards,2026-10-15 after1274bb2. User directs primary model4B, suspend0.6B rare/single-position analysis, no speed tests/push/PR/merge. Located old4B input-space first26 single-slot680/704/777, two-depth[7,2]621/704 and full48012809/15009; actual prior output0/output1 raw/result/config not found in stated search. History inventory distinguishes evidence from user recollection; this validates a new matched chain, not exact reproduction of unlocated positive output-space numbers.

##### Fixed input and comparisons

SpecBench category file order skip2/take2, categories=['summarization', 'translation'], 4 prompts, manifest IDs/rendered tokens frozen before make. Seed123, per-prompt generator123+index, nonthinking first-turn template, no shortening/drop/replacement. Qwen3-4B existing14-file local snapshot whose paths/sizes match official ModelScope listing; reuse via shared/model/modelscope junction, local_files_only=true. Actual SHA/config/index/tied-storage/model digest/dtype/backend/source/runtime gates; remote immutable revision and original download origin remain unconfirmed. Source670c744e0535dcad unchanged.

Four native Runs: matched input frozen-mean ESP, output0, output1 and fresh independent AR. All methods same static unpruned[1,1], k2/BC9, FP32/eager/CUDA, max100/EOS/T0. Outputbeta.9/beta_prime.99 and mapping/state semantics unchanged; no tuning. The input method is current research ESP frozen mean, not historical selected EMA or single-slot[14]; those numbers are background only. Actual tied weight storage mandatory. Declare H1 zero-input accepted total>0 and one-input accepted total>0; any nonpositive contrast contradicts that bounded gain prediction, not a correctness failure. Separate one-zero contrast and per-prompt/category signs. No superiority/population assertion from these fixed prompts.

Count accepted drafts before budget/EOS clipping; count main generation model calls as prefill+decode for spec and ar_calls for AR. Internal AR audit calls separate, not added to main-path call comparison. Pair by exact qid/input/output and seed; every spec output must equal own AR AND separate AR Run token-for-token. Retain complete depth/accepted/cache/call traces and all negative/tie rows. Source/input/model/context/OOM/nonfinite/AR failure stops expansion, preserve failed Runs; never relax gate or change precision for a good result.

##### No benchmarking and resource schedule

eval.warmup=false. Study-only UntimedEval replaces decoder module clock with constant zero during execute, without touching global time or scientific src; native-required timing fields are explicit disabled zero sentinels, no latency/throughput/speedup measurement or conclusion. Native collect/report compatibility fields may be zero, and formal report excludes them. Wrapper tests prove clock restoration/token/state/call identity. Build same HF model directly onCUDA via device_map to avoid a full16GB FP32 host copy; same wrapper/config/actual model digest, no offload/quantization. Study-local adapters/hash frozen and restore scoped construction hooks.

SingleGPU/round1 serial. First Run is capacity calibration (same planned prompts, no repeated successful cells), then remaining three; .5s RSS/available-RAM/globalGPU sampling for OOM risk only. Est first26~15min/all4 (planning, not throughput); stop resource failure rather than kill another task. Budget4x4x100 plus3x4x100 internal AR audits = 2800new tokens, zero teacher/warmup, no single-window interventions. Realized EOS lower bound reported. Model loading/hashing/native finalization may be substantial but no timing ranking.

##### Follow-up and archival

After first26 all quality gates pass, execute separately versioned predeclared holdout: summarization/translation skip2/take2 (qid243244163164), four new prompts, same four Runs/configs; regardless positive/negative primary signs, no selection by outcomes. Its purpose is limited out-of-development paired confirmation, not full matrix or mechanistic analysis. Old0.6B reports/raw/weights/cache/commits remain historical, no new rare/window work. Archive status and unresolved questions with links, not deletion/moving raw. Formal tables/paired count plots/provenance/report/local commit; raw Runs/model/cache local only. No remote, new PR or merge.

</details>

<a id="legacy-studies-output-space-4b-input-control-docs-plan"></a>

### studies/output_space_4b_input_control/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_4b_input_control/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_4b_input_control/docs/PLAN.md); original SHA256 `256422682fbfe1e2a86f334653c3be279c10263dc77c0c6d7aa190df58e3d5b3`.

#### Fixed historical input-EMA matched-chain control

Frozen2026-10-15 after first26 and fixed4-new-prompt reports, before any new EMA forwards. Primary input/zero/one accepted238/329/321 and calls1955/1866/1876; new4 accepted51/39/47 and calls189/201/194, all independentAR. Historical selected input EMA beta=.9999/lam=.1/gamma1 improved wider[14]/[7,2] input baselines, but its strength on this chain is unknown. This control addresses that comparison gap without retuning outputs or searching prompts/parameters.

Two native EMA-only Runs: reuse exactly original first26 and fixed4 manifests/seed ordering/model snapshot/runtime. Same FP32/eager/T0/EOS/max100/static unpruned[1,1]/k2/BC9; input EMA levelorder0, beta.9999, lam.1, step_scale1, normalizefalse, extrapolate_emafalse, update=true from current named mean_plus_gamma implementation. No other beta/gamma/lam tested. Frozen mean/output0/output1/fourthAR successful Runs are reused by hashes and never rerun. eval.num_eval_prompts26 is upper limit; second dataset has exactly4 rows, no filtering/truncation. Require exact original input IDs/count/order and full output equal own freshAR AND preceding separateAR, same actualmodel/source/runtime/flags; keep all signs and failures.

H2 bounded strong-baseline persistence: both outputs accepted totals exceed the fixed selected input EMA in each group. Report signs per prompt and group, not just pooled30; a nonpositive contrast rejects that bounded group's persistence prediction. A weaker EMA than frozen mean remains a result, not proof it is a stronger method here. Prior EMA selection used first26, so these are comparative development outcomes, not held-out population significance.

Budget2*(26+4)*100=6000 new generatedtokens including ownAR audits, zero teacher/warmup; no new output or independentAR generation. Same tested Study-only directCUDA placement/disabled decoderclock/native phases, 0 time sentinels not measurements; count main prefill+decode separately from audit ar_calls. Serial singleGPU/round1/.5s capacity sampling, no timing rank. Stop/preserve source/model/input/context/OOM/nonfinite/AR/Run gate failure, no replacement or same-success re-signing. Weights/cache/fullRuns local; formal paired counts/complete semantic evidence/report/local commit. No scientificsrc/default change, 0.6B rare/window work, push, PR or merge.

</details>

<a id="legacy-studies-output-space-4b-recheck-docs-plan"></a>

### studies/output_space_4b_recheck/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_4b_recheck/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_4b_recheck/docs/PLAN.md); original SHA256 `e7e34debe59388340c71cee2e922f9a1778b2e629689eac39a2c86e64fa28a63`.

#### Qwen3-4B matched output-space recheck

Frozen before new model forwards,2026-10-15 after1274bb2. User directs primary model4B, suspend0.6B rare/single-position analysis, no speed tests/push/PR/merge. Located old4B input-space first26 single-slot680/704/777, two-depth[7,2]621/704 and full48012809/15009; actual prior output0/output1 raw/result/config not found in stated search. History inventory distinguishes evidence from user recollection; this validates a new matched chain, not exact reproduction of unlocated positive output-space numbers.

##### Fixed input and comparisons

SpecBench category file order skip0/take2, categories=all13, 26 prompts, manifest IDs/rendered tokens frozen before make. Seed123, per-prompt generator123+index, nonthinking first-turn template, no shortening/drop/replacement. Qwen3-4B existing14-file local snapshot whose paths/sizes match official ModelScope listing; reuse via shared/model/modelscope junction, local_files_only=true. Actual SHA/config/index/tied-storage/model digest/dtype/backend/source/runtime gates; remote immutable revision and original download origin remain unconfirmed. Source670c744e0535dcad unchanged.

Four native Runs: matched input frozen-mean ESP, output0, output1 and fresh independent AR. All methods same static unpruned[1,1], k2/BC9, FP32/eager/CUDA, max100/EOS/T0. Outputbeta.9/beta_prime.99 and mapping/state semantics unchanged; no tuning. The input method is current research ESP frozen mean, not historical selected EMA or single-slot[14]; those numbers are background only. Actual tied weight storage mandatory. Declare H1 zero-input accepted total>0 and one-input accepted total>0; any nonpositive contrast contradicts that bounded gain prediction, not a correctness failure. Separate one-zero contrast and per-prompt/category signs. No superiority/population assertion from these fixed prompts.

Count accepted drafts before budget/EOS clipping; count main generation model calls as prefill+decode for spec and ar_calls for AR. Internal AR audit calls separate, not added to main-path call comparison. Pair by exact qid/input/output and seed; every spec output must equal own AR AND separate AR Run token-for-token. Retain complete depth/accepted/cache/call traces and all negative/tie rows. Source/input/model/context/OOM/nonfinite/AR failure stops expansion, preserve failed Runs; never relax gate or change precision for a good result.

##### No benchmarking and resource schedule

eval.warmup=false. Study-only UntimedEval replaces decoder module clock with constant zero during execute, without touching global time or scientific src; native-required timing fields are explicit disabled zero sentinels, no latency/throughput/speedup measurement or conclusion. Native collect/report compatibility fields may be zero, and formal report excludes them. Wrapper tests prove clock restoration/token/state/call identity. Build same HF model directly onCUDA via device_map to avoid a full16GB FP32 host copy; same wrapper/config/actual model digest, no offload/quantization. Study-local adapters/hash frozen and restore scoped construction hooks.

SingleGPU/round1 serial. First Run is capacity calibration (same planned prompts, no repeated successful cells), then remaining three; .5s RSS/available-RAM/globalGPU sampling for OOM risk only. Est first26~15min/all4 (planning, not throughput); stop resource failure rather than kill another task. Budget4x26x100 plus3x26x100 internal AR audits = 18200new tokens, zero teacher/warmup, no single-window interventions. Realized EOS lower bound reported. Model loading/hashing/native finalization may be substantial but no timing ranking.

##### Follow-up and archival

After first26 all quality gates pass, execute separately versioned predeclared holdout: summarization/translation skip2/take2 (qid243244163164), four new prompts, same four Runs/configs; regardless positive/negative primary signs, no selection by outcomes. Its purpose is limited out-of-development paired confirmation, not full matrix or mechanistic analysis. Old0.6B reports/raw/weights/cache/commits remain historical, no new rare/window work. Archive status and unresolved questions with links, not deletion/moving raw. Formal tables/paired count plots/provenance/report/local commit; raw Runs/model/cache local only. No remote, new PR or merge.

</details>

<a id="legacy-studies-output-space-4b-recheck-v2-docs-plan"></a>

### studies/output_space_4b_recheck_v2/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_4b_recheck_v2/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_4b_recheck_v2/docs/PLAN.md); original SHA256 `4a92e2c3fcddd4760a6823c66da536120fed0811d09741b174a8e18d3f06eeac`.

#### Qwen3-4B matched output-space recheck

Frozen before new model forwards,2026-10-15 after1274bb2. User directs primary model4B, suspend0.6B rare/single-position analysis, no speed tests/push/PR/merge. Located old4B input-space first26 single-slot680/704/777, two-depth[7,2]621/704 and full48012809/15009; actual prior output0/output1 raw/result/config not found in stated search. History inventory distinguishes evidence from user recollection; this validates a new matched chain, not exact reproduction of unlocated positive output-space numbers.

##### Fixed input and comparisons

SpecBench category file order skip0/take2, categories=all13, 26 prompts, manifest IDs/rendered tokens frozen before make. Seed123, per-prompt generator123+index, nonthinking first-turn template, no shortening/drop/replacement. Qwen3-4B existing14-file local snapshot whose paths/sizes match official ModelScope listing; reuse via shared/model/modelscope junction, local_files_only=true. Actual SHA/config/index/tied-storage/model digest/dtype/backend/source/runtime gates; remote immutable revision and original download origin remain unconfirmed. Source670c744e0535dcad unchanged.

Four native Runs: matched input frozen-mean ESP, output0, output1 and fresh independent AR. All methods same static unpruned[1,1], k2/BC9, FP32/eager/CUDA, max100/EOS/T0. Outputbeta.9/beta_prime.99 and mapping/state semantics unchanged; no tuning. The input method is current research ESP frozen mean, not historical selected EMA or single-slot[14]; those numbers are background only. Actual tied weight storage mandatory. Declare H1 zero-input accepted total>0 and one-input accepted total>0; any nonpositive contrast contradicts that bounded gain prediction, not a correctness failure. Separate one-zero contrast and per-prompt/category signs. No superiority/population assertion from these fixed prompts.

Count accepted drafts before budget/EOS clipping; count main generation model calls as prefill+decode for spec and ar_calls for AR. Internal AR audit calls separate, not added to main-path call comparison. Pair by exact qid/input/output and seed; every spec output must equal own AR AND separate AR Run token-for-token. Retain complete depth/accepted/cache/call traces and all negative/tie rows. Source/input/model/context/OOM/nonfinite/AR failure stops expansion, preserve failed Runs; never relax gate or change precision for a good result.

##### No benchmarking and resource schedule

eval.warmup=false. Study-only UntimedEval replaces decoder module clock with constant zero during execute, without touching global time or scientific src; native-required timing fields are explicit disabled zero sentinels, no latency/throughput/speedup measurement or conclusion. Native collect/report compatibility fields may be zero, and formal report excludes them. Wrapper tests prove clock restoration/token/state/call identity. Build same HF model directly onCUDA via device_map to avoid a full16GB FP32 host copy; same wrapper/config/actual model digest, no offload/quantization. Study-local adapters/hash frozen and restore scoped construction hooks.

SingleGPU/round1 serial. First Run is capacity calibration (same planned prompts, no repeated successful cells), then remaining three; .5s RSS/available-RAM/globalGPU sampling for OOM risk only. Est first26~15min/all4 (planning, not throughput); stop resource failure rather than kill another task. Budget4x26x100 plus3x26x100 internal AR audits = 18200new tokens, zero teacher/warmup, no single-window interventions. Realized EOS lower bound reported. Model loading/hashing/native finalization may be substantial but no timing ranking.

##### Follow-up and archival

After first26 all quality gates pass, execute separately versioned predeclared holdout: summarization/translation skip2/take2 (qid243244163164), four new prompts, same four Runs/configs; regardless positive/negative primary signs, no selection by outcomes. Its purpose is limited out-of-development paired confirmation, not full matrix or mechanistic analysis. Old0.6B reports/raw/weights/cache/commits remain historical, no new rare/window work. Archive status and unresolved questions with links, not deletion/moving raw. Formal tables/paired count plots/provenance/report/local commit; raw Runs/model/cache local only. No remote, new PR or merge.

##### Activation v2 after the preserved collect failure

First26 v1 input Run completed execute with26/26 exact AR and zero timing sentinels, accepted238/decode_calls1929, but Study collect/__init__ exported a module rather than callable run; final status failed. No scientific acceptance of that Run. Preserve v1 declarations/locks/raw/results unchanged. V2 new Study and algorithm.version repairs only four phase exports to existing from .run import run pattern; model/config/inputs/output algorithms and source670c744e0535dcad unchanged. Add phase-entry callable regression before new make. New first input Run is a fresh calibration under v2, not re-signing v1. The not-yet-made holdout receives the same callable source repair before its first freeze/forward. All successful cells are still never rerun.

</details>

<a id="legacy-studies-output-space-cross-task-docs-canonical-plan"></a>

### studies/output_space_cross_task/docs/CANONICAL_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/CANONICAL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/CANONICAL_PLAN.md); original SHA256 `0536c43a5e6809e3ac6c451cd401018f7371b8b4d79d0f5dc430303e78ac7d39`.

#### First cross-task H2 canonical follow-up

Predeclared 2026-10-14 after native and fixed-window cross-task outcomes, before any canonical forward. Separately versioned diagnostic, not an adaptive cell appended to the native batch; all existing PLAN/EXECUTION_PLAN/checker/raw/hypothesis bytes remain unchanged.

Selection exactly follows the prior follow-up criterion: first file-order feasible unit with accepted-changing transfer isq241, entry716, order1<-order0 changes11→12. No search for best gain or alternative root. Test whether that contrast persists under canonical committed anchor/v at every actual root>716; incoming proxies retained, true KV root verified. Native controls use original runtime/defaults/seed123, ModelScope/snapshot/source and original masks from the completed diagnostic.

Four generations: native order1 self, native one<-zero, canonical self, canonical one<-zero; budget400 tokens. Two native controls must reproduce previous complete AR/semantic trace and all captured tensors. One teacher-forced full independent AR prefix792 positions (692 prompt+100 new) with final hidden only; exact-root cosine>=.99999 against baseline sampled roots. Reuse native canonical_states() and synchronized_transfer(); no new EMA/decoder implementation. Teacher packing differs and is not exact real arithmetic. Canonical self may change; compare its graft against it, not only native baseline. Full output AR equality and original through716 semantics mandatory.

After716 replace only anchor/v with per-prefix CPU-FP32 chronological native observe_committed states; preserve incoming proxies at each patch, but later forward-generated proxies may change. Do not force previous roots/owners. Four generations plus1 teacher, no warmup/other auxiliary model forward. Source/snapshot/oldraw tensor SHA/newtool/oldhelpers/currentplan/model/runtime before/after; load large1.9GB raw via read-only mmap, not a full eager copy. SerialGPU/OMP-MKL1/.5s sampling; priornative~6.7GBRSS/7.3GiBGPU, teacher792-position logits adds memory, calibrate actual/stop on pressure/OOM/hash/nonfinite/output/cos/scope/root failure.

If canonical contrast remains+1 with stable self, this supports bounded persistence and excludes committed numeric microdifferences as necessary for this local contrast under synchronization. Vanishing contrast is a bounded counterexample; changed self constrains attribution. Do not infer pure proxy mediation, default gain or other prompt generality. Original hypotheses_v1.json H2=not_tested is retained as batch history; follow-up result is separate. Full weights/cache/tensors/Run local only; no source/default/remote/PR/deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-execution-plan"></a>

### studies/output_space_cross_task/docs/EXECUTION_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/EXECUTION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/EXECUTION_PLAN.md); original SHA256 `7051210ee105a6b449a8bf0562e2efd7a75c10e1357d31c3f8bfbaa3ad271040`.

#### q241–q244 real execution addendum

2026-10-14 after4c9ad8e, before first real make. Current user authorizes the bounded real cross-task batch. Original PLAN/CHECKER_PROTOCOL draft bytes remain unchanged; their draft-only status is historical. This addendum activates exactly their fixed native/window/H1/H3 protocol, no new hypothesis or prompt/window search.

Version output-space-summarization-241-244-20261014-v1, same rendered IDs692/647/595/901 and11-file local ModelScope snapshot/source670c744e0535dcad. Four native cells input/output0/output1/AR, T0/EOS/max100, beta.9/beta_prime.99/static[1,1]/k2, original native runtime defaults. Only this new Study version/provenance changes, no scientific/default recipe changes.

Before first make verify snapshot/dataset/rendered IDs, reuse shared model via junction, singleGPU/round1/OMP-MKL1. Run the first frozen_mean_chain cell serially for capacity calibration, record .5s sampled RSS/GPU/available RAM; proceed with one launch chain for remaining cells only after first gate/resource check. No completed cell reruns or reuse of older Run directories. Native token budget1600+1200 audit+72 unchanged warmup; diagnostics20x100=2000, no auxiliary model forwards. Stop/preserve OOM/nonfinite/hash/output/root/scope/context failures; do not shorten prompts or relax gates.

Native source/factor/raw/AR/Study/process gates→report→checker export→frozen window selection→unchanged diagnose_study_windows→checker verdict. Preserve actual positive/null/infeasible and legitimate depth2 witnesses; H2 remains untested until another frozen canonical protocol. Execution evidence/full scalar traces formal; raw tensors/cache/weights/completeRun local only. Parallel independent work is planning/coding only; GPU diagnostics for q178 wait until native/window GPU processes exit.

No push, new PR, merge or deletion. Existing workspace preserved.

</details>

<a id="legacy-studies-output-space-cross-task-docs-first-delta-plan"></a>

### studies/output_space_cross_task/docs/FIRST_DELTA_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/FIRST_DELTA_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/FIRST_DELTA_PLAN.md); original SHA256 `bda13e540bb5bfa0b2853fc3dac85cdd8d999dbace65fff959d27830b40f5350`.

#### First local acceptance change versus final full-path direction

Frozen after the completed expansion and its STUDY_REPORT update, before inspecting/scoring first differences,2026-10-15. This is an explanatory test on already observed final outcomes, not a new held-out stochastic sample. No new model forwards/draws/inputs/window selection.

H7: the sign of the first chronological semantic self/graft difference's accepted-node-count delta predicts the sign of the final accepted-count delta. Use all10 new completed sampled units plus2 old development rows separately; do not choose a helpful root or omit cancellation/nulls. Compare TRACE_KEYS only, because mask norm/hidden diagnostic metadata can differ before any accepted-path effect. Shared prefixes imply the first divergent trace record has the same actualroot; require this, do not force later alignment.

Verify per-cell accepted equals sum of trace accepted_nodes before final budget/EOS clipping. Preserve actual common roots and each side's unique roots. Reconstruct final delta exactly as common-root node-count difference + graft-only accepts - self-only accepts; this is a trace accounting identity, not tensor causal mediation or a forced baseline path. Report firstgain→finalloss, firstloss→finalgain and firsteffect→netzero counterexamples. Identical traces must imply zero delta; missing/truncated/duplicate-root records fail, never default to zero.

Complete script with tests for sign reversal, cancellation, preserved-direction controls, first-record prefix mismatch and damaged counts. Bind original sampled reports, expansion analysis, tool/plan/source before/after; formal structured comparisons only. Then update report with actual H7 verdict and new/old scope. No source/default/old evidence edit, remote/PR/merge/delete.

</details>

<a id="legacy-studies-output-space-cross-task-docs-kernel-execution-plan"></a>

### studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md); original SHA256 `65519e3e0dbe7aaf8ae391afb6b1a02637eb02016e2214e33fd65cd63705c578`.

#### Fixed q_proj protocol execution addendum

2026-10-14 after a50f9e0, before measurements. KERNEL_PLAN.md remains the original draft byte-identical. Current authorization permits the reversible local experiment described there, independently of provider norm controls.

Execute exactly the six planned canonical q178 generations: native self/graft read-only captures of source blocks85/86, plus source85 owner0 slot1 row4 q_proj singleton-FP32 or CPU-FP64-rounded output substitutions, each self/graft. Reuse previous canonical states/donors; no teacher forward. Four cloned-cache exact block replay forwards only. All selected logits/final hiddens in those replays must reproduce captured values bit-for-bit; restore a fresh cloned cache, never original live cache. Pre-forward cache snapshots hold independent storage and their KV tensors remain unchanged after replay.

Profile layer0 q_proj packed[1,9,d] and selected singleton[1,1,d] for each source85 path; correlate aten::mm/addmm CPU IDs to actual CUDA kernels, retain shapes and names even if equal. Direct numerical comparisons additionally cover captured block86. High-precision CPU linear uses exactly represented FP32 activations/weights/bias, casts output only back toFP32. It changes one operation, not model precision. No operator/root/layer search.

Native capture cells must match prior canonical full semantic traces/selected samples and independent AR. At patch assert actual source85, row4, untouched other rows, finite replacement, one application and removable hooks. Complete variant AR/scope/state gates remain mandatory; native paths need not hold after substitutions. Budget600 new tokens plus4 exact block auxiliary forwards; source/plan/model/11-file snapshot/tool/helpers/oldraw/native config/inputs bound before/after, strict deterministic process settings, serialGPU/.5s resources. Preserve failures and no source/default/remote/data deletion.

Separately q241–244 receives a runnable evidence-export/verdict script and deterministic synthetic minimal reproductions. This turn does not launch its real-model batch. H1/H3 real results remain unmeasured; H2 needs a later frozen canonical follow-up. This explicitly limits the next-step claims.

</details>

<a id="legacy-studies-output-space-cross-task-docs-kernel-plan"></a>

### studies/output_space_cross_task/docs/KERNEL_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/KERNEL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/KERNEL_PLAN.md); original SHA256 `662bb5e126e48dbfdf5e8a5ad69d8039f7fa842c8e7796eea468f5c1a41c54b7`.

#### Transformer proxy-rounding / CUDA packing protocol draft

Unexecuted protocol; separate from the completed provider NormTwoOps controls. Those controls replace provider math on identical inputs, and do not observe all operations that create transformer proxy hiddens. Do not attribute pos88/root89 committed delta to the profiled provider kernel.

##### Question and fixed unit

Q178, original root85 owner0 second-slot graft, strict FP32 canonical anchor/v controls, same ModelScope snapshot/source/seed126/AR112. Capture source blocks85 and86 for self and one<-zero, including both owner0 mask rows and exact/draft rows. Self/graft source inputs differ semantically, so their hidden delta is not an estimate of rounding error. Numerical counterfactuals instead compare each source block with itself using exactly the same inputs, cache and model weights.

Predeclare operator target: layer0 q_proj linear output, selected owner0 mask slot1 row of block85. This fixed choice tests a narrow GEMM/packing hypothesis and avoids searching for a kernel with a favorable outcome. A null at this target leaves later layers, RMSNorm, attention and hidden store rounding unresolved.

##### Capture, identical-input replay and attribution

1. Read-only hooks retain inputs_embeds, positions, complete4D attention and every pre-forward KV layer tensor; clone before mutable cache pruning, verify storage independence. Persist shapes/dtypes/strides, hashes and exact layer0 q_proj input row/weight/bias/output. No hooks edit science src. Retain source row/true owner from the decoder and remove hooks in finally. Two complete strict canonical generations reproduce old self8/27 and graft9/26, independent AR and old selected windows. If capture alters them, stop.
2. Restore a fresh cloned pre-forward cache for each auxiliary replay. Native exact block replay must reproduce all selected logits/hiddens bit-for-bit. Profile the original q_proj on its actual packed input, and the identical selected row via singleton FP32 linear; associate CPU operator external IDs, dimensions and concrete CUDA launch names. A name can remain equal across shapes: record it, do not invent a different kernel variant.
3. Replay only that same row with CPU FP64 linear from exactly represented FP32 input/weight/bias, round output to FP32. Compare native packed FP32, singleton GPU FP32 and CPU FP64-rounded output. The GPU packing comparison fixes dtype and arithmetic operator; the FP64 comparison changes arithmetic rounding. CPU FP64 is a local reference, not exact full-model truth. Other rows/layers are unchanged.
4. Two interventions during complete canonical decoding: replace only the predeclared row of the predeclared source-block q_proj output with singleton FP32 output, or CPU FP64-rounded output. Run each with self and graft; retain a native original-value self/graft pair from capture. At every patch assert all other output rows bit-equal, same inputs/weights/state, actual block/row, and no candidate/owner forcing. Immediate proxy hidden/mask/logit differences and later traces are outcomes, not fixed to old values.
5. If captured native packed versus singleton differs and the scoped replacement changes the self/graft contrast, this demonstrates sensitivity to the q_proj operation's packing/backend realization at that input. It does not isolate an internal CUDA instruction or explain every path-dependent hidden delta. If FP64 replacement alone changes the contrast, it shows local arithmetic-rounding sensitivity. If neither changes it, reject necessity of that layer0/source-row operation for this fixed contrast; preserve all other open operations.

##### Budget and gates

Six generations (capture native self/graft+four interventions)x100=600 tokens, at most4 auxiliary exact block model forwards (2 sources per path); direct linear replays2 sources x3 modes with reused weights, no new full model or FP64 model. One GPU process, OMP/MKL1, strict runtime recorded, .5s resources and fresh output directory. Extra capture cache increases memory: calibrate first self capture serially; stop/preserve on pressure or cloning failure. Hash source/model/11 snapshot files/input/native/previous raw/tool/this protocol before/after. AR equality, native/capture/replay gates, legal depth2 flags and patch cleanup required. CUPTI unavailability preserves the kernel-name gap, no installation or guessed attribution.

No code for this protocol has been run or added as a scientific method. No new default, push, merge or deletion. Draft choices are intentionally narrow and falsifiable; a later operator change requires a new version before measurement.

</details>

<a id="legacy-studies-output-space-cross-task-docs-local-factor-plan"></a>

### studies/output_space_cross_task/docs/LOCAL_FACTOR_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/LOCAL_FACTOR_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/LOCAL_FACTOR_PLAN.md); original SHA256 `3fb58d846c188137ac70bc6d39f270922bfe79ca009a2de1acb07593a74122c2`.

#### q241 fixed-block state/proxy timing experiment

Frozen after4a02551 before new forwards, 2026-10-15. Explain the observed first divergence atroot720; selection is post-observation, not a new independent sample. Decoder constructs the candidate tree from the preceding mask logits before current provider mapping. Therefore test source719 and current720 separately, not a mislabeled current-state intervention on already constructed candidates.

Two canonical self/graft full generation controls, same q241 seed123/entry716/budget100/default runtime, must reproduce prior full AR/trace/stats/all captured tensors. Capture fresh-cache frames from canonical self at719 and720; record both paths at actual716→717→719→720. Clone actual input/positions/attention/KV before each selected block, and capture logits/final hidden. Do not fabricate unvisited718. No teacher, no new Run or old re-signing.

At each fixed frame, cross immediate committed state (canonical self versus native self captured at that same root) with incoming proxy (canonical self versus canonical graft captured at that same root). Recompute only6 mask embeddings using unchanged native OutputHiddenMaskProvider, holding FP32 native scale/order1/beta.9/beta_prime.99. Hold root/node embeddings, positions, attention and every KV entry. Execute all4 cells3 times with fresh cloned cache:24 block forwards total. Native-equivalent cell must exactly reproduce full saved logits/hidden; repeated outputs must be bit-identical. All root/node logits must be bit-identical across mask-only changes because tree attention prevents future masks leaking into verification. Saved cache, states/proxies and first3 input rows stay unchanged. Report if any factor is itself ineffective rather than replacing it with arbitrary noise.

Source719 owner0 rows3/4 seedroot720. Report first/second AR margins, generated seed IDs, mask-input change and final proxy hidden; root720 candidate acceptance comes from the previously fixed tree, not its current masks. Root720 interventions test immediate verification equivalence and subsequent seed differences, not full downstream generations. Check source owner/positions against original actual trace and record all four steps' candidates, decisions and state/proxy differences. This separates mapping effects on a fixed cache/tree context; it does not claim effects generalize to the graft context or uniquely partition full path effects. Keep nulls and backend numerical differences explicit.

Budget200 newtokens plus24 block forwards, zero teacher/warmup. SingleGPU/OMP-MKL1/.5s memory calibration; stop/preserve OOM, nonfinite, cache/root/scope/output/byte/runtime drift. Bind raw reports/tensors/plan/tool/old helpers/model/11-file ModelScope snapshot before/after; full frames/cache/tensors local only. No science/default edits, push, PR or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-owner-plan"></a>

### studies/output_space_cross_task/docs/OWNER_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/OWNER_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/OWNER_PLAN.md); original SHA256 `da5faf4eaccd41f97f098dfc5f3630e0780c87b5bd59c05e2a46eaff954b36f8`.

#### q241 actual owner1 fixed-frame state/proxy control

Frozen after6cca8fc before new forwards/draws, 2026-10-15. Next different actual owner issource717 owner1, selected from the already measured716→717→719 chain, not by favorable margins. Owner1 masks rows5/6 atpositions719/720 seedroot719; do not score owner0 rows3/4 as the next-path seeds.

Two q241 seed123/T0 canonical self/graft full generation controls must match prior AR/trace/stats/all captured tensors; capture canonical-self source717 input/positions/attention/cache/logits/hidden, fixed root/tree/KV. Reuse unchanged synchronized_transfer and prior teacher states, zero teacher. Four S/P cells (canonical versus native self committed state, canonical self versus graft proxies), three fresh-cache repeats per cell,12 block forwards. Only six mask embeddings change; preserve root/node verification outputs and saved cache, position, attention, state/proxy; native-equivalent cell exact, repeats exact. Compare actual owner1 seed logits with previous graft next-root719 seed, even if they fail to recover a correct greedy candidate. No arbitrary noise or alternative owner search.

Conditional acceptance at this other owner uses the old actual verification root719 logits from the saved fixed-frame experiment, native sampler T=.8/1, seeds200..263, same64 verifier draws under each S/P cell, exact candidate softmax masses. Distinguish changed candidate/seed from a restored hit. Root719/root720 contexts and mass differences differ, not paired evidence that an owner index itself caused the effect.

Budget200 full-generation tokens +12 direct blocks +128 conditional verifier draws, zero teacher/warmup. The sampled-path study remains separate. Bind old native/canonical reports+tensors/new plan/tool/old local helpers/scientific sources/11 snapshot/model/runtime before/after; .5s singleGPU monitoring. Raw cache/frame tensors local only; no source/default edits, old re-sign, remote/PR/merge/delete.

</details>

<a id="legacy-studies-output-space-cross-task-docs-plan"></a>

### studies/output_space_cross_task/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/PLAN.md); original SHA256 `07abd854854816d51c4770939e14e095e6f9931dbe62698528f6e58d19475df9`.

#### Cross-task falsification Study draft

Draft 2026-10-14, scientific source670c744e0535dcad after f3b95ea. This round prepares the Study and tokenizer input manifest only; no make, launch, native Run, source lock or result exists. Do not interpret expanded configs as measured results. Inputs and gates must be verified again before the first launch, then retain this contract byte-identical or use a new version if the contract changes.

##### Evidence motivating the questions

Twelve translation development units q171–q182: pooled input/output0/output1 accepted60/45/47. Natural order1 gains occur atq173/q179, loss atq182; the remaining nine have equal pooled accepted, sometimes differing depth paths. Fixed bidirectional transfer changes accepted on4 units and not on8, including terminal/unused-owner controls. Q178 strict canonical states preserve8→9; q182 has a later valid second-slot restoration under equal source/root anchor/v but lacks a full canonical-state control. Seed changes alone do not imply accepted changes. This is development evidence, not random sampling or significance.

The executed provider arithmetic/norm controls in the extension preserve q178 full self/graft paths across all three operator substitutions. Transformer-created proxy hidden rounding remains open, covered by KERNEL_PLAN.md; this draft does not claim it is ruled out.

##### Predeclared falsifiable hypotheses

H1 cross-task occurrence: at least one of four fixed summarization units has a natural output-order semantic difference or a fixed one-slot transfer semantic difference. If all four complete with none, reject this bounded occurrence prediction; do not replace prompts/windows after seeing nulls. Equal total accepted with different depth/owner/KV paths counts as semantic difference but must be distinguished from net gain.

H2 committed-state necessity: if a new unit has an accepted-count transfer change, it persists under a separately predeclared canonical anchor/v self-versus-graft comparison at the same fixed entry. A disappearing change with stable canonical self falsifies this local necessity exclusion; changed self constrains the comparison. Native and canonical results must be separate. The current cross-task budget does not include adaptive canonical Runs: retain the first eligible result and draft a new frozen protocol/version before such a run, rather than expanding this batch opportunistically. Q178 supports one instance, q179/q182 remain unresolved.

H3 first-slot-only explanation: any legal second-depth change while its first slot stays the same correct AR token falsifies it. Already contradicted onq178/q182; this batch tests whether that counterexample appears in another task. Never score second-depth margins when the first token is wrong or EOS removes the next target. Retain first-only gains/losses and no-change units.

H4 numeric-artifact explanation: q178 effect requires provider EMA/slot or norm rounding. Existing S64/N64/N32 controls contradict necessity of the post-entry provider stages in those fixed cells; initial root85 mapping was left native. The narrower untested transformer hypothesis is defined in KERNEL_PLAN.md; a kernel name alone is insufficient to accept it.

##### Fixed inputs and conditions

Summarization file order skip0/take4, q241/242/243/244. DatasetSHA7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21; rendered input lengths692/647/595/901 with exact ID/template hashes in prompts_manifest.json. Same local ModelScope11-file Qwen3-0.6B snapshot, tied actual weights, FP32/eager/CUDA, nonthinking, T0/EOS/max100, static unpruned[1,1]/k2/BC9, beta.9/beta_prime.99, original native deterministicfalse. Seeds123+i; algorithm version output-space-summarization-241-244-draft-v1. No prompt shortening; any input mismatch, unsupported context or source drift stops the batch and is reported.

Four new native Runs: input frozen mean, output0, output1, independent AR. Native make→launch→process→report, all source/model/input/factor/raw/AR/Study gates, full output token equality required. No successful old Run is rerun or re-signed.

After native success, freeze a selection manifest using existing choose_window(), identical to the q175–q182 procedure: earliest shared acceptance with preceding common rejection; otherwise first continuing shared rejection; otherwise continuing unused-owner control; no feasible continuing root is retained explicitly. Capture full original-root union from entry; use actual visited roots and owners after transfer. Two baseline observations, both donor directions and order1 original-value placebo per feasible unit, owner0 second slot[0,1] once only; independent AR and through-entry semantic equality hard gates. Original baseline/placebo must reproduce full native trace. Reuse measured helpers without editing their bytes. The manifest must supply rendered_prompts/question_id for diagnose_study_windows.py; native rendered IDs must equal the preflight IDs, not just qids.

##### Metrics, budget and decisions

For each unit report native accepted by depth, calls, all actual semantic trajectories, legal two-slot target margins, matched-state anchor/v equality, source owner and proxy/seed differences, gains and losses, infeasible/noncommon roots. Separate H1 occurrence, accepted net gain and H3 depth-only effects. No power/significance claim from n=4, no win-rate estimate from the prior12; no default retuning or speed conclusion.

Serial singleGPU/round1/OMP-MKL1. Native budget1600 new tokens+1200 internal AR audit+72 existing warmup; diagnostics at most20x100=2000 new tokens, no auxiliary transformer forward. Initial inputs are longer than the translation calibration, so first native Run calibrates actual host/KV/GPU usage before continuing. Prior6.7GB RSS/~6.3GiB globalGPU is context, not this batch's measurement. Sample.5s; retain est/actual and explicit background-memory limitation. Stop/preserve OOM, nonfinite, output/scope/source/hash/root failure, context shortening or resource pressure. No replacement prompt or relaxed gate.

Full weights/cache/Run/profiler traces/tensors local only; formal summaries and selected scalar provenance can be reviewed separately. No push/merge/delete. Next decision after this bounded batch: accept/reject the occurrence prediction, retain every positive/null, and choose a frozen canonical or transformer-operator follow-up based on the declared criterion. No full matrix is implied.

</details>

<a id="legacy-studies-output-space-cross-task-docs-rare-event-plan"></a>

### studies/output_space_cross_task/docs/RARE_EVENT_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/RARE_EVENT_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/RARE_EVENT_PLAN.md); original SHA256 `414eacd84fbca42458ed8258c6692033075c1b5e776d2515b1a2c533ddb5f11b`.

#### Actual owner1 rare-event detectability boundary

Frozen aftera5ede78 before new draws,2026-10-15. Use old actual root719 verification logits, source717 owner1 candidates11/323, no new model forward or altered temperature. T=.8 and1, fresh seeds300..4395,4096 independent-generator single-token native CUDA select_token calls per temperature (8192 verifier draws total). Reuse one realized draw bank across four S/P factor cells. Keep all sampled token IDs and RNG-consumption gates; do not turn zero observations into probability zero.

Bind actual CUDA-FP32 softmax masses and CPU-FP64 reference separately, same-backend reconstruction mandatory. Compute explicitly idealized Bernoulli detectability from mass p: P(at least one inN)= -expm1(N*log1p(-p)), trials for95% one-event detection=ceil(log(.05)/log1p(-p)), and one-sided95% upper bound afterzero/N= -expm1(log(.05)/N). Validate0<=p<=1/N>0 endpoints and stable tiny-p math. This is a theoretical probability-model power calculation, not proof that native finite-precision CUDA sampling implements extremely small probabilities exactly.

Budget8192 native verifier draws, zero model forward/teacher/generated continuation tokens; no billions-draw brute force, boosted logits or rare event replacement. Source/helper/report/raw/plan/tool hashes before/after, OMP-MKL1 and .5s resource monitoring. Negative repeat may be stable at this budget while detectability remains negligible; leave native sampler rare-event numerical reachability and complete-path accumulation unverified. Formal scalars/draws only; no source/default/old evidence edits, remote, PR, merge or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-sampled-depth-plan"></a>

### studies/output_space_cross_task/docs/SAMPLED_DEPTH_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/SAMPLED_DEPTH_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/SAMPLED_DEPTH_PLAN.md); original SHA256 `58c7651aec38eee35962c5c81078f0bd5663d30fa73846be34be7f4e5bb7784d`.

#### Sampled legal second-depth counterexample check

Frozen after the six-unit H8 report update, before inspecting new selected tensors or depth events,2026-10-15. Existing accepted totals are known; explanatory analysis on a fixed measured batch, not held-out population inference. No new model forwards/draws/window search.

H10 bounded first-slot-only prediction: among all originally captured shared self/graft roots of the six new sampled units, no legal second-slot acceptance flip exists with the same correct first candidate and bit-identical current committed anchor/velocity. A legal event requires both first draft candidates equal the corresponding fresh sampled AR token at root+1, node1 accepted by both actual paths, a non-EOS first target, second target present, and node2 membership differing. Preserve source-root/owner and second candidates, first/second targets, current state/proxy equality/max deltas and exact root-hidden equality. Same first candidate alone without both accepted or remaining second target is not a legal second-depth event. Root/node causal context may differ after earlier path changes; equal current state is not a full canonical or equal-KV factor control.

Score all captured actual root intersections from original six-target rules; retain unvisited/exclusive roots and invalid first-slot/EOS/budget controls, no substitute windows. Evaluate root+1/root+2 in full AR output IDs (includes input prefix), not generated-only offsets. Bit-equal current anchor/v legal events falsify H10; legal events with nonzero state microdifference constrain but do not falsify its equal-state version. If none captured, report not falsified in this finite capture, not a proof of first-slot-only full-path mechanism. Current H8 outcomes must not be used to drop negative units.

Bind diagnosis/tensor SHA, selection/protocol/tool/current source before/after. Pure predicate tests cover node1 prerequisites, second-node flip, EOS/missing target and no-flip control. Scalar/full path observations formal, raw/tensors local; no science/default/old helper/Run edit, remote/PR/merge/delete.

</details>

<a id="legacy-studies-output-space-cross-task-docs-sampling-continuation-plan"></a>

### studies/output_space_cross_task/docs/SAMPLING_CONTINUATION_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/SAMPLING_CONTINUATION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/SAMPLING_CONTINUATION_PLAN.md); original SHA256 `648c34a0e71e2a00501d2d0c99a254bc30a3c7da78917a7d8985494b7fc90d26`.

#### Additional prompts and seeds, native rare-event detection stress

Frozen after985c7c1,2026-10-15 before new draws or generation. Prior rare candidates were0/4096, not a first rare hit. The earlier source719 owner0 first-slot restoration and the H7 sign-reversal counterexamples are separate evidence.

##### Full sampled paths

Fixed six new units: q242/126, q244/126, q245/126, q245/127, q246/126, q246/127. New prompts are the next two summarization entries in file order (skip4/take2), never selected by model outcomes. Render with the same native tokenizer/chat template (enable_thinking=false); preserve all IDs, no shortening or replacing oversized prompts. Old q242/244 rendered IDs must equal native AR inputs. Bind dataset, tokenizer/model snapshot, input manifest, helpers and old source before/after.

H8 bounded sign persistence: q242/126 graft-self>0 and q244/126 graft-self<0, extending the prior three seeds. Either zero or opposite effect falsifies that bounded prediction. New q245/246 are exploratory replication, all positive/null/negative/infeasible results retained. No population significance or tuning.

Same FP32/eager/CUDA/native runtime, ModelScope weights, T=.8/max100/EOS/static[1,1]/beta.9/beta_prime.99. Seven generations per feasible unit: fresh independent sampled AR, output0/output1, two baseline observations, order1 self/graft. Reuse unchanged choose_window/frozen_targets/observe/transfer. Lock selection from native traces before hidden/logits capture, six roots maximum, single[0,1] once; complete actual paths retained. Require AR equality, native observer/full self stats and semantic trace replay, scope, before-entry identity and cache/source/model/runtime gates. Infeasible stays explicit, never scored as zero. Budget42 generations/4200new, zero teacher/warmup; existing12 units reused separately, not rerun. Test first-delta accounting again without claiming new tensor causal decomposition.

##### Conditional rare-event control and stress

Keep actual old owner1 source717 candidate11/323 and verifier root719 fixed. No transformer forwards, changed logits or artificial candidate embedding. Original T=.8 and1 each16384 fresh native select_token draws with seeds10000..26383, same draw bank for four S/P cells. Add explicitly changed-condition T=2 with131072 seeds10000..141071, not a default or full-path temperature change. Total163840 draws; no adaptive sample increase or optional stopping. Repeated seeds across temperatures are paired, not independent groups. Retain every draw locally, full hash, first-hit seeds, counts, CUDA-FP32 mass and CPU-FP64 reference.

Zero-forward preflight before choosing this stress budget measured T=2 mass11=3.130066761514172e-5 and323=3.7709174648625776e-5; ideal probability of at least one at131072 exceeds.98 for each. H9 bounded stochastic prediction: observe at least one hit for each candidate in this changed distribution. Zero contradicts that prediction at its stated ideal miss risk, not a universal sampler impossibility proof. Require preflight mass consistency and >=.98 ideal detection before sampling. Compute95% exact Clopper-Pearson marginal intervals; compare native masses descriptively, no independent-cell test because banks are shared. Original-condition zero bounds remain insufficient for1e-11..1e-9 mass. T=2 hits cannot establish original T=.8/1 tail reachability or useful complete-path acceptance.

##### Schedule and gates

Serial singleGPU/OMP-MKL1, .5s process RSS/available RAM and globalGPU monitor, no concurrent model/sampler tasks. Previous6new-unit estimate~110seconds and rare~120seconds is planning only. First new prompt calibrates no shortening; stop/preserve OOM, nonfinite, context/source/input/runtime/AR/scope mismatch. Raw weights/cache/tensors/draws stay local.tmp; formal scalar/provenance/semantic comparisons only. New tools and protocol exact-byte Git attributes before execution; no old tool/plan/Run/scientific src/default edits, push, PR, merge or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-sampling-expansion-plan"></a>

### studies/output_space_cross_task/docs/SAMPLING_EXPANSION_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/SAMPLING_EXPANSION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/SAMPLING_EXPANSION_PLAN.md); original SHA256 `9486949089770147504b702e601855f07a2a80fea7b421dc42b5713761b61556`.

#### Four-prompt three-seed sampled-path expansion

Frozen aftera5ede78 before new forwards,2026-10-15. Keep old q241/T=.8 seeds123,124 as development evidence, never rerun/re-sign them. Add onlyq241 seed125 andq242/q243/q244 seeds123,124,125:10 new units,12 total plotted with old rows distinguished. T=.8/max100/EOS/FP32/eager/static[1,1]/beta/beta_prime/default runtime and exact rendered IDs unchanged. Do not replace difficult/null units or force source719.

Per new unit seven generations: independent sampled AR, native output0/output1, two observations and order1 self/graft. AR output equality, no shortened context, observer/native full stats+semantic equality, original-value placebo and pre-entry/single-vector transfer gates mandatory. Reuse existing choose_window and frozen_targets (entry+first3+last2), selection based only on actual native traces and written before hidden/logit capture. Infeasible units remain explicit, not null effects. Same reference sampler/generator per seed, true sampled prefixes/owners retained; after transfer do not force original paths.

Budget70 generations/7000 generated new tokens including10 AR audits, zero teacher/warmup. Existing q241 rows are separately bound source reports/raw hashes; no pooling old units into a new held-out claim. Report every per-unit accepted/depth/calls/trace delta and sign, same-seed outputs, across-seed AR differences, and whether first local changes cancel/reverse over the complete path. No random-prompt population inference, speed comparison or stable-effect claim from3seeds. AtT>0 first-slot draft remains top1 while verification samples; different sampled windows lack a canonical equal-state decomposition.

ModelScope cached11 files/hash/model/source/baseline files/tool/plan/old helpers before/after; serialGPU/OMP-MKL1/.5s monitor, previous~5.5GBRSS/~7.1GiB globalGPU is context, not this measurement. First fresh unit calibrates memory; stop/preserve source/input/output/numerical/context/OOM failure, never relax AR gate or rerun completed old units. Formal scalar/full semantic traces only; raw/tensors/weights/cache local. No source/default/old contract edits, push, PR, merge or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-sampling-plan"></a>

### studies/output_space_cross_task/docs/SAMPLING_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/SAMPLING_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/SAMPLING_PLAN.md); original SHA256 `4b371bab1a53cabac17f749e0b1cdf83b07faff703dfe1e85378cd389e8999ca`.

#### q241 nonzero-temperature and conditional-sampling protocol

Frozen after6cca8fc before new generation/draws, 2026-10-15. No scientific/default/config/old helper/Run edits. Two complementary experiments must stay distinct: conditional stochastic acceptance at the old fixed source719 context, and genuinely sampled complete paths with different prefixes/windows.

Full-path q241/T=.8 seeds123,124, max100/EOS same recipe/default runtime. Each unit independently generates AR with the native shared-generator select_token sampler, then native output0/output1 with the same fresh seed; require exact output equality against that AR. Select entry only using unchanged choose_window on the two realized traces, freeze selection JSON before observing hidden/logits. Keep infeasible entries; no searching a gain or forcing719. Capture entry plus first3/last2 selected roots (at most6); re-observe two baselines and run order1 original-value self and order1<-order0 once each, preserving AR and full through-entry semantics. Drafts remain deterministic top1 even atT>0; verification multinomial is stochastic. Budget14 total generations/1400 newtokens (including two AR audits), zero teacher/warmup. All gate failures retained, no relaxed exactness or replacement seed. The two fresh AR paths must be reported different/equal, never asserted beforehand.

Conditional sampling: saved actual verification root720 logits held fixed; old source719 four S/P factor cells supply draft top1 candidates. Temperatures .8 and1, seeds200..263,64 coupled one-token verifier draws per temperature. Use CUDA native select_token(logits,T,fresh Generator) once per seed, same draw under all S/P conditions. Compare candidate==sampled-verifier-token and exact softmax candidate mass, not greedy AR margin or an invented sampled continuation. The same seed/RNG-state bank is reused at owner1; verifier tokens are drawn separately for its different context, then held across its factor cells. No model forward/new generation tokens; include RNG-active/state and draft-vs-verifier distinction in regression tests. It is a conditional Monte Carlo test on an old context, not64 sampled prompts or source719 reached under a new sampled trajectory.

The full-path windows may differ from719 and lack canonical equal-state controls. A nonzero graft effect there only establishes sampled-path occurrence, not pure proxy mediation or a general mechanism. Current prefix/seed/RNG/temperature/output/call counts and all nulls must remain visible. Bind plan/tool/old sources/helpers/native inputs/snapshot/local raw/model before/after, ModelScope cached only. SingleGPU/OMP-MKL1/.5s resources; stop/preserve OOM/nonfinite/hash/input/output/scope/context failures. Formal scalar/traces only, raw/cache/weights local. No push/PR/merge/delete.

</details>

<a id="legacy-studies-output-space-cross-task-docs-state-extension-plan"></a>

### studies/output_space_cross_task/docs/STATE_EXTENSION_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/STATE_EXTENSION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/STATE_EXTENSION_PLAN.md); original SHA256 `65247ceb024d9199e56d66823976c05f30adba703577fde4f0c1bb4e1a64ef97`.

#### Fixed q242/q243 two-seed state interaction extension

Frozen after4a02551 before new forwards, 2026-10-15. Test whether q241 self-interaction-1 appears universally in a small fixed extension. Select next two original file-order prompts q242 seeds124,127 andq243 seeds125,128, keeping same100-new-token T0/default recipe and rendered inputs; no outcome-based prompt replacement. Seeds are administrative replay probes: T0 select_token uses argmax, output_hidden init/mapping has no RNG draw, so these are not independent stochastic samples. Retain equality or differences rather than count four independent samples.

Per unit observe output0/output1 against original native full AR/trace/stats. Original fixed roots658/600 are retained. Capture at most6 targets chosen before new logits: entry, first3 original selection roots afterentry, last2 original selection roots afterentry; deduplicate and retain missing/new roots explicitly. Baseline controls prove new seeds reproduce the existing greedy paths or stop with a failed quality gate. One teacher on the full original AR prefix, final hidden only, cosine>=.99999 for captured exact roots; canonical states use unchanged chronological CPUFP32 helper.

Per unit reuse observed order1 baseline as native-self, generate native one<-zero plus anchor-only self/graft, velocity-only self/graft and both-synchronized self/graft:7 transfers plus2 baseline generations,8 logical factor cells. Actual KV root synchronization starts only after the fixed entry; other state/proxies at each single-coordinate patch unchanged. No force of native future root/owner path. All outputs must equal original independent AR; original-value self and native-beforeentry semantics mandatory; every captured synchronized coordinate checked with actual tensors and all actual-root scope records checked.

H5: count interaction Y11-Y10-Y01+Y00 per self/graft, retain zero/nonzero/sign and controls. Universal q241-like nonzero interaction across these fixed units is falsified by any complete zero cell; zero is not a global linearity proof. Changes across administrative seeds with otherwise identical greedy inputs require a numerical reproducibility investigation, not a seed benefit claim. H6 count/path equivalence and H3 legal second-depth witness remain descriptive checks, excluding wrong first, EOS, missing next AR target and unaligned source/root.

Budget3600 generated tokens across36 generations plus4 teacher forwards (two747 and two695positions), zero warmup. .5s monitoring, singleGPU/OMP-MKL1, cached ModelScope only. Sources/model/inputs/snapshot/oldhelper/selection/tool/plan before/after and native runtime gates. Stop on resource/OOM/nonfinite/scope/cosine/output/hash drift; preserve every null/counterexample, no relaxing gate or new prompts. Formal scalar/full semantic traces only; raw/cache/weights/complete Run local. No scientific source/default/old contract edits, push, PR or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-state-factor-plan"></a>

### studies/output_space_cross_task/docs/STATE_FACTOR_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_cross_task/docs/STATE_FACTOR_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/STATE_FACTOR_PLAN.md); original SHA256 `4b9a74e2fd5c7fe46e2941b53265f2843abee9bce4d9946d89a5d4e972abf4b6`.

#### q241 state-factor counterexample protocol

Frozen after c4cb011, before new generations. Reuse the already observed q241/seed123/entry716, 100-token AR and canonical teacher states. The selection is explanatory, not held-out confirmation. No new root/prompt/seed/default or native Run; old plans/tools/outputs remain unchanged.

Six generations: native self/graft audit controls, anchor-only self/graft, velocity-only self/graft. Reuse saved native and both-synchronized cells for the full 2x2 anchor/velocity by 2 donor table. At every actual KV root>716 overwrite only the named committed coordinate; assert the other coordinate and incoming proxies remain bit-identical at that patch. Later proxies/state/path may change. Capture the previously frozen targets only; actual new roots are reported as unobserved. Reuse transfer/decoder, native runtime, cached ModelScope/hash binding and teacher; no teacher/auxiliary forward/warmup; budget600 new tokens. Native controls must reproduce all old captured tensors and full semantic traces; AR/full through716 trace equality and cache-root gates mandatory.

H5: count effects of anchor and velocity synchronization are additive separately for self/graft. Compute interaction Y11-Y10-Y01+Y00; nonzero falsifies this local additivity hypothesis. Also retain all traces and same-root margins: equal counts never imply equivalent paths. H6: equal total acceptance implies path equivalence is already contradicted by the two old canonical cells; quantify the gains/losses and first divergence instead of treating another run as held-out validation.

Structured comparison uses chronological prefix divergence, common-root decisions, lost/new roots, true source owner, first/second AR margins and captured state/proxy/seed tensor deltas. Same-position canonical state equality can exclude differing committed anchor/v at that position as necessary for a local self/graft difference, but upstream proxy histories can retain earlier influences. State-factor effects are interventions, not a unique percentage decomposition of native committed microdifferences. Stop on any input/hash/source/output/nonfinite/root/cleanup failure, preserve evidence. Serial GPU, OMP/MKL1 and .5s resource sampling. No push, PR, source/default edits, deletion or weights/raw cache in Git.

</details>

<a id="legacy-studies-output-space-extension-docs-norm-kernel-plan"></a>

### studies/output_space_extension/docs/NORM_KERNEL_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_extension/docs/NORM_KERNEL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/NORM_KERNEL_PLAN.md); original SHA256 `441c33bf69bba47f46aecbde39027d5ed149418437a2bf783e952f7bdc9489bc`.

#### q178 CUDA norm implementation contrast

Predeclared 2026-10-14 after a50f9e0. This extends OPERATOR_PLAN without editing it. Same source/model/strict runtime/seed126/canonical states/root85 and independent AR112. The current user explicitly requests local kernel controls; no remote action/default change.

Fixed FP32 GPU alternative: sqrt(sum(slots * slots, dim=-1, keepdim=True)), versus native vector_norm. Identical slot values, dtype and scale/multiply/divide; the alternative changes the implementation graph, including intermediate square rounding, so it is not a pure instruction-only kernel experiment. Save both actual profiler launch names and CPU operator external-ID correlation. Reference each norm to CPU FP64 norm rounded toFP32 on held inputs. No guessed kernel names or speed claim.

Eight complete canonical generations: native self/graft controls, entry-only alternative self/graft, post-entry-only alternative self/graft, entry+post alternative self/graft. At entry85 replace only the existing donor/self vector[0,1], computed from that donor's original frozen provider state; other masks remain original. Post-entry variants replace only norm mapping for all subsequent actual provider calls with real incoming proxies and canonical anchor/v. Self changes constrain interpretation. Controls must reproduce prior canonical full paths and selected tensors. All cells require complete independent AR, pre/through85 semantic equality, real KV-root checks and patch cleanup. No forced owner/candidate/path.

Frozen-input operator comparisons use both baseline donor states at85 and both canonical states at86, four inputs; native recomputation must match saved masks. Profile native and alternative on canonical-self root86, once each; save traces locally. Log slots/native norm/alternative norm/reference errors and returned mask deltas. Actual matching inputs can yield unchanged results; do not drop ineffective cells.

Budget8x100=800 new tokens, zero teacher/auxiliary model forwards/warmup, bounded four-state mapping comparisons and two profiles. SerialGPU/OMP-MKL1/.5s sampling; prior~5.2GB RSS/global6.3GiB, stop on OOM/nonfinite/hash/output/scope/root/unsupported-op failure. Bind this plan/newtool/oldhelpers/oldraw/model snapshot/nativeconfig/inputs before/after. Fresh .tmp output; raw tensors/traces/cache/weights never Git.

Decision: stable self/graft paths exclude necessity of these tested norm graph differences at entry85 and/or subsequent calls for this fixed contrast. Changed contrast shows local sensitivity under that stage substitution, not unique attribution to one CUDA instruction. This cannot isolate transformer-generated proxy hidden; fixed q_proj protocol runs separately. Retain every local loss/gain and legal second-slot flag.

</details>

<a id="legacy-studies-output-space-extension-docs-numerical-plan"></a>

### studies/output_space_extension/docs/NUMERICAL_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_extension/docs/NUMERICAL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/NUMERICAL_PLAN.md); original SHA256 `cb41fb5759365d5f82c45d87c822524777d7d5bf3cbbc308a0f4bde3c3d74c49`.

#### q178 fixed precision and canonical committed-state controls

Predeclared 2026-10-14 before new controls, version q178-numerical-history-20261014-v1. Original q178 seed126 FP32/eager/T0/static[1,1]/k2/BC9/100tokensEOS, fixedroot85 owner0 second-slot graft, original independentAR112 total positions/36new tokens. No scientific src/default/nativeRun changes. Previous single-slot tool/observer/compare helpers remain byte-identical.

##### Questions and fixed cells

Q1: do strict FP32 deterministic repeats give the same masks/logits/state/acceptance, and do the tiny same-position differences from altered block/owner paths persist? Q2: if later committed anchor/velocity are synchronized to one canonical AR-prefix reference while actual proxy/seed/owner history is retained, does the 8→9 effect persist? Candidate rank is the immediate T0 acceptance rule; history is a possible upstream cause, not an alternative rule. Do not force an owner, candidate or old trajectory.

Strict process-only settings before CUDA initialization: CUBLAS_WORKSPACE_CONFIG=:4096:8, FP32/eager, TF32 matmul/cudnn off, float32 matmul precisionhighest, deterministic algorithmsenabled with warn_onlyfalse, cudnn deterministictrue/benchmarkfalse. Record requested/effective settings, do not edit nativeconfig/defaults. First two strict original output0/1 observations must reproduce old full AR and semantic trace; capture union of original roots>=85, by visited subset. Use their actual root85 donor tensors (not assuming old floats equal across runtime modes).

Then fixed strict cells: order1<-order0 twice, order0<-order1 twice, order1 original-value placebo once. Each must match independent AR and pre/through85 semantic prefix; repeats compare actual trace and all captured tensor values, not only pooledaccepted. Compare saved old-path state deltas separately; no expectation that determinism makes differently packed floating computations bit-equal.

One additional strict full112-position teacher-forced AR forward, final normalized hidden only, no generation. Verify teacher exact-root hidden versus observed baselines at all captured positions (cosine>=.99999). Reuse native OutputHiddenMaskProvider.observe_committed over those chronological exact hiddens to form per-order canonical anchor/velocity for each prefix lengthp (hidden throughp-1). Teacher packing differs from native blocks: this is an explicit numerical synchronization intervention, not a claim of exact math or a production recipe.

Four canonical cells: output0 self-value/no-graft control, output1 self-value/no-graft control, order1<-order0 and order0<-order1. Keeproot85 original state/masks and graft definitions; for every actual provider call root>85 overwrite only committed anchor/velocity with the corresponding canonical same-AR-prefix state, leave incoming proxies unchanged. Read actual currentroot from the real mutable KV cache length after native pruning, check again at modelprehook; never infer post-control roots from a baseline trace. Clone tensors and remove patch/hooks even on failure. Canonical self-value cells may have different later semantics; retain changes rather than require a false placebo equality. Compare graft versus its canonical same-order self control, not just original baseline, at true matchedprefixes. If a new root lacks an original capturedstate, canonical state still exists from the complete112-position teacher; no invented old-state evidence.

Budget11 decoder generations x100=1100new tokens (2 strict baselines+4 repeats+1 strictplacebo+4canonical), one112-position teacherforward, no other auxiliary query or warmup. Source670c744e0535dcad, model/snapshot11-file hashes, old native/raw/transfer tensors and this PLAN/new tool/helpers bind before/after. New .tmp/q178-numerical-20261014-v1, never overwrite prior evidence. SerialGPU/OMP-MKL1; prior same-model RSS~6.6GB/globalGPU~6.3GiB capacity, sample.5s/recordrealcalls. Stop/preserve if unsupported deterministic kernel, hash/output/teacher/scope/actualroot/nonfinite/OOM failure; no precision/contract change to rescue a result. Complete AR equality and actual source-owner/trace evidence required. No pushes/merges/deletion/weightcacheexport.

##### Interpretation boundaries

Identical repeats plus persistent path-dependent microdeltas supports floating differences from changed packing/commit paths rather than random noise; it does not identify every low-level CUDA kernel. Canonical synchronization preserving gains would rule out those committed-state microdeltas as necessary for the effect in these fixed cells, while still not isolating numeric differences carried inside proxy hiddens. Canonical controls changing acceptance must be reported and constrain causal claims. A vanished gain bounds robustness, not the entire method. Ranking versus history are separated by immediate seed/margin/accepted depth and later actual-owner propagation, with no global EMA claim.

All cells/results and limitations go into STUDY_REPORT; PLAN's predeclaration remains immutable and results are saved separately. New prompt batch executes separately with original runtime defaults; no concurrent GPU work.

</details>

<a id="legacy-studies-output-space-extension-docs-numerical-plan-review"></a>

### studies/output_space_extension/docs/NUMERICAL_PLAN_REVIEW.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_extension/docs/NUMERICAL_PLAN_REVIEW.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/NUMERICAL_PLAN_REVIEW.md); original SHA256 `7c84e30a2af76019581fe1f3657bd74c2fc52767d805e1ffed6ace678b3e0060`.

#### q178 numerical PLAN review

Results review2026-10-14; original NUMERICAL_PLAN.md and native PLAN.md are retained byte-identical, not re-signed. Evidence: numerical_diagnosis_v1.json and numerical_analysis_v1.json.

Q1 locally supported: strict FP32 repeats produce bit-identical selected logits/hiddens/masks/state and full semantic paths, while altered-versus-baseline path microdeltas persist. First observed common divergence is root89: anchor for exact position88 comes from accepted row2 of block86 in the original path, versus row0 of block88 in the altered path (the latter verified by tensor). This locates a deterministic block/KV/owner packing difference; no individual CUDA kernel is identified.

Q2 locally supported: canonical self/no-graft controls remain8/27; canonical grafts remain9/26 and match strict graft full semantic paths. At same actual AR-prefix root>85, canonical anchor/v are bit-equal across same-order control and graft; incoming proxies are left intact. Thus those committed-state microdeltas are not necessary for this effect under the declared canonical intervention. Teacher/canonical packing itself differs; self controls constrain interpretation, and numeric differences carried in proxy hiddens remain unisolated.

Immediate root86 second-slot rank crossing and subsequent owner1/2 history are different stages, not mutually exclusive causes. No forced-owner or fixed-candidate counterfactual was run. A future causal refinement could hold a specified proxy/owner-output vector under canonical states at one declared root, with self-value and AR gates; this is a proposal, not an executed result or changed default. Current newprompt generality evidence belongs to output_space_followup. No need to change precision/default/source from these diagnostic outcomes.

</details>

<a id="legacy-studies-output-space-extension-docs-operator-plan"></a>

### studies/output_space_extension/docs/OPERATOR_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_extension/docs/OPERATOR_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/OPERATOR_PLAN.md); original SHA256 `add72cf85dfc1a6cd4b9369fb3aa9b8258b549ad7a01f16ace80008a8a77e60f`.

#### q178 provider rounding and norm-backend isolation

Predeclared 2026-10-14 after f3b95ea, before this diagnostic. Existing NUMERICAL_PLAN and successful native contracts stay byte-identical. This is a diagnostic extension of q178, not a new default or a new scientific implementation.

##### Fixed interventions

Restrict to order1 at the existing root85, seed126, unchanged independent AR112 positions/36 new tokens. Reuse canonical committed states and original root85 self/donor vectors from the hash-bound previous strict evidence. Do not regenerate a teacher, change model precision, force candidates/owners or search roots. At root85 and earlier use the native provider byte-for-byte; only subsequent actual provider calls can change. Every call still receives its actual incoming proxy tensors. The canonical wrapper reads root from the real KV cache and replaces anchor/v at every actual root>85.

Eight generations: native self and one<-zero, then each of three operator variants with self and one<-zero. S64 computes simulated EMA/slots from the exact same FP32 inputs in CPU FP64, casts completed slots to FP32, and retains native GPU norm/scale/multiply/divide. N64 retains native GPU FP32 slots and replaces only the norm scalar with CPU FP64 vector_norm rounded to FP32. N32 retains those slots and substitutes CPU FP32 vector_norm, leaving scale/multiply/divide on GPU. Thus S64 bounds provider arithmetic rounding; N32 holds input dtype fixed while switching norm backend; N64 changes reduction precision. They do not isolate hidden-rounding inside transformer proxies or distinguish every instruction inside a kernel. No combined S64+N64 cell; interaction remains unchecked.

The repeated native self/graft must match previous canonical full AR/semantic trace and selected masks/state/hidden/logits at frozen roots. All variant cells must equal independent AR, preserve semantics through85 and preserve provider state/incoming proxies during the mapping itself. Later proxies/paths may change and must be retained. Self controls may change and must constrain attribution. At roots86/89/108 of the previously saved canonical self/graft, replay all mappings on identical saved provider inputs and report norm/output error versus native. Exact native-formula replication is a hard gate.

Profile exactly one native mapping of canonical-one-self root86 (2x1024 norm per owner) using installed torch.profiler CPU+CUDA. Save Chrome trace locally and extract concrete kernel names correlated with norm CPU operator via external IDs. If CUPTI/kernel events are unavailable, preserve that fact and do not infer names from CPU timing. Profiling is location evidence; N32/N64 are operator interventions. A norm kernel name plus changed margin is not proof that its internal reduction tree uniquely caused the old transformer hidden microdelta.

Budget8 generations x100=800 tokens, no teacher/auxiliary transformer forwards/warmup; six saved-state mapping inputs x4 modes plus one profiled native mapping. Single CUDA process/OMP-MKL1, strict settings identical to prior numerical tool, snapshot/source/model/old tensor/old report/native config/inputs/PLAN/tool/helpers bound before/after. Prior RSS~4.9GB/global GPU~5.7GiB plus244.8MB CPU evidence; sample.5s. Stop and preserve on mismatch/nonfinite/hash/root/OOM/unsupported deterministic op; profiler unavailability alone records a bounded gap. New output directory only. Full traces/scalars formal, raw tensor and profiler trace remain .tmp. No scientific src/default edits, push, merge, deletion or cache/weight export.

##### Decision rules

Compare graft versus its same-mode self, not pooled against a single native baseline. If S64 alone removes/restores the graft difference while self stays fixed, provider slot rounding is necessary for that fixed contrast under the intervention. If N32 alone changes it, norm backend implementation can matter despite identical input dtype; if N64 alone changes it, sensitivity to norm rounding is demonstrated. Stable paths across all three constrain necessity of these stages for this fixed q178 effect, not all numeric sensitivity or all prompts. Full trace/margin differences still matter when accepted counts stay equal. Next hidden-rounding/backend experiment must capture identical Q/K/V or norm inputs inside the model and intervene on one operation, with self/AR gates; current provider experiment cannot make that claim.

</details>

<a id="legacy-studies-output-space-extension-docs-operator-plan-review"></a>

### studies/output_space_extension/docs/OPERATOR_PLAN_REVIEW.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_extension/docs/OPERATOR_PLAN_REVIEW.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/OPERATOR_PLAN_REVIEW.md); original SHA256 `7b6762b4812736d140f1c56591cf8fcb2a7f77d59511ff0ce617a93d41764e29`.

#### q178 provider operator result review

2026-10-14. OPERATOR_PLAN.md and the archived measured tool source remain byte-identical. CLI usage-only docstring repair is checked by execution-AST equality; the original report hash is not re-signed. Eight outputs match independent AR; native self/graft repeat previous canonical selected tensors and full semantic traces exactly. Each S64, N64 and N32 self remains8/27; each graft remains9/26, with full paths equal to its corresponding native case. Provider differences up to5.960464477539063e-8 and small downstream seed/logit changes exist, so these are effective interventions, not identical implementations.

Profiler records84 CUDA launches and3 aten::linalg_vector_norm CPU calls, correlated via external IDs to the actual NormTwoOps reduce_kernel symbol recorded in operator_diagnosis_v1.json. That identifies the provider norm launch. The CPU-FP32 backend substitution and CPU-FP64-rounded norm substitution bound its necessity for this fixed contrast; they do not identify the internal reduction instruction or isolate the transformer kernels generating proxy hidden.

The post-root85 provider-rounding necessity explanation is narrowed by stable paths; root85 entry mapping was left native and remains an untested stage, while transformer hidden rounding and operator interactions remain open. Only order1 self/one<-zero was tested; reverse/order0 and combined stage replacements were not added. KERNEL_PLAN in the new cross-task draft gives a fixed-row linear packing/high-precision protocol, explicitly unexecuted. New Study is draft only. No default or scientific source change follows.

</details>

<a id="legacy-studies-output-space-extension-docs-phi-rms-plan"></a>

### studies/output_space_extension/docs/PHI_RMS_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_extension/docs/PHI_RMS_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/PHI_RMS_PLAN.md); original SHA256 `b5231de753f37521e59590d3279ece1182a785757f8252ca28d6b9a096dc1879`.

#### q178 entry Phi and fixed post-attention RMSNorm controls

Predeclared 2026-10-14 after4c9ad8e, while cross-task native GPU queue runs. Execute only after that queue and its transfer diagnostics exit. Same strict q178 root85/seed126/ModelScope/source/AR112/canonical states. No root/layer/row search or new default.

Eight generations: native self/graft controls, Phi64 self/graft, RMSNorm singleton-FP32 self/graft, RMSNorm reference64 self/graft. Controls reproduce old full canonical semantic trace/selected tensors. All variants preserve independent AR and through85 semantics; actual future owners/paths are not forced. Budget800 tokens, no teacher/auxiliary model forwards/warmup, four frozen RMSNorm direct comparisons plus paired native/singleton CUDA profiles.

Phi64: compute the donor's native FP32 simulated owner0 slot1 exactly (EMA/slot construction not changed), hold the native FP32 embedding-norm scale, compute full norm+scale multiplication+division in CPU FP64, cast completed vector once toFP32. Replace only root85[0,1], other mask vectors and subsequent provider mapping remain native; canonical anchor/v after85 retained. Report vector bit-equality/max/L2 first; an unchanged vector is numerically ineffective and cannot establish rounding robustness. Do not recompute scale or add arbitrary noise.

RMSNorm: fixed model.layers[0].post_attention_layernorm, source85 row4 only. This is a different operator stage from q_proj. Native capture retains same complete input/output plus shape/stride. Singleton mode directly calls module.forward on that row to avoid hook recursion; reference64 uses exact FP32 input/weight and FP32-represented epsilon, CPU FP64 square/mean/rsqrt/product, then casts once back. All other output rows remain bit-identical. Record nonzero/equal substitution, final masked hidden/proxy/seed changes and full actual paths. Direct comparisons use each native self/graft frozen input, not their semantically different inputs subtracted as rounding error. Bind installed RMSNorm source file in addition to runtime version, tool/common/plan/oldraw/source/model/11file snapshot/native config/inputs.

SingleGPU/OMP-MKL1, .5s resources. Capacity from q178 prior~5.4GB RSS/~5.8GiB globalGPU is context, not new measurement. Stop/preserve OOM/nonfinite/hash/scope/root/output/unsupported-kernel failure; no gain tuning. Raw tensors/Chrome traces/weights/cache/fullRuns local only; scalar and semantic evidence formal. No source/default edits, remote push, PR or deletion.

Interpretation: stable trajectories under effective Phi mapping or RMSNorm substitutions bound necessity of these numerical realizations for this fixed contrast, not necessity of the operator itself or all layers/contexts. Ineffective singleton or Phi cells remain explicit equivalence controls; do not pool them with effective numerical interventions. Cross-task H1/H3 results are separate, H2 still requires a later frozen canonical protocol.

</details>

<a id="legacy-studies-output-space-extension-docs-plan"></a>

### studies/output_space_extension/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_extension/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/PLAN.md); original SHA256 `61743ea13904291ceb770172a23a7a51af96c88133e68a94584e184ce06a0bab`.

#### Four-prompt extension and fixed single-slot intervention plan

Predeclared 2026-10-14 on parent7a02501. Version output-space-q175-178-20261014-v1. Same scientific source670c744e0535dcad, model/snapshot/default gains/decoder/sampler/cache. Existing native Study machinery and observe()/transfer()/compare_window() are reused. No code in scientific src is changed.

##### Existing q173 evidence and hypotheses

Evidence sources: output_space_small_matrix/docs/trajectory_diagnosis.json, window_diagnosis_v1.json, transfer_diagnosis_v1.json, cross_prompt_diagnosis_v1.json and cross_prompt_path_v1.json. These are development observations, not independent confirmation. q173's 29 common output0/1 roots differ in acceptance only at94/97; q171's36, q172's12 and q174's11 have no output-order acceptance difference. Full q173 mask/logit measurements exist only at the prior five fixed transfer roots, not every earlier root.

| qid | intervention root | root minus input length | first common hit | output0/1 accepted | transplant vector L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 171 | 63 | 0 | 64 | 10/10 | .09785780 |
| 172 | 42 | 3 | 43 | 1/1 | .05879461 |
| 173 | 83 | 14 | 84 | 1/3 | .05366028 |
| 174 | 41 | 9 | 42 (terminal) | 1/1 | .06347270 |

Q173 did not have the largest injected vector difference, and late window position alone is insufficient (q174 terminates; q171 has more remaining roots but no hit transfer). At q17394/97, baseline order0/1 target margins have opposite signs, deltas about+4.00275/+5.11003. The order1<-order0 transplant removes both hits with same committed anchor/v at five measured roots; reverse order0<-order1 restores94 but not97. Q171/172 transfers change margins by at most.42729/.59764 without sign crossing, and all semantic trajectories remain equal. Q174 has no continuing owner1 propagation after EOS.

H1 (rank-boundary hypothesis): historical proxy perturbation can alter candidate directions without acceptance change; q173's local gain requires specific candidate rankings to cross the correct-target top1 boundary. Mere nonzero velocity, displacement amplitude or propagated seed difference is insufficient. In new prompts report natural baseline margin sign opposition at all common visited roots, and actual matched acceptance/sign changes after fixed transfers. Natural acceptance differences with no transfer would bound this single-entry sufficiency, not invalidate the T0 ranking identity.

H2 (owner-path history hypothesis): the first common accepted-owner transition allows the preceding second slot to influence later first-seed proxies. A donor-like first seed may persist along owner0-only blocks, while later accepted-owner transitions can mix it back into recipient-specific history/velocity; this could explain q173 reverse transfer restoring94 but not97. Measure actual source owners, donor-first-seed equality/L2 at common roots, and alignment of equality loss with subsequent accepted roots. This is observational mediation after one intervention, not an independent test of every intermediate slot; retain exceptions.

H3 (terminal/unused-owner boundary): if no continuing accepted-owner path uses the injected slot, a null acceptance result is expected and is not evidence against all historical effects. Explicitly classify terminal/no-shared-hit/unused-owner controls instead of combining them with continuing propagation cases.

These hypotheses were written before observing new q175–q178 outputs. Do not retune or pick another window after seeing new margins.

##### Fixed new inputs and native matrix

Select contiguous next translation records, file order skip14/take4, q175/176/177/178. Freeze text, datasetSHA and independent tokenizer-rendered input IDs in prompts_manifest.json before make. Same locally prepared ModelScope Qwen3-0.6B 11-file snapshot, tied weights, FP32/eager/CUDA, nonthinking, T0/EOS, max_new_tokens100, unpruned static[1,1]/k2/BC9, beta.9/beta_prime.99. Seed123 with per-prompt generators123/124/125/126; these are four prompts, not stochastic seed replications.

Four native Experiments/Runs (input frozen mean, output0, output1, independent AR), each containing four prompts, new version/IDs. Run make/launch/process/report, round1 single GPU. Gate exact_match1, no context shortening, input/model/source/factor/provenance/raw/Study quality all pass. No warmup changes. At most1600 new tokens in native runs plus1200 internal AR-audit tokens and at most72 unchanged warmup tokens. Do not rerun already successful Run IDs or replace a null prompt.

##### Predeclared transfer selection and diagnostics

For each prompt using native output0/1 semantic traces only: choose the earliest common acceptance root whose immediately preceding common visited root is rejected by both; intervene at that preceding root. If none, choose the earliest common root rejected by both that has a later root. If still none, choose the earliest common nonterminal root and label unused-owner control. If no continuing common root exists, report no feasible slot-use window rather than replace the prompt. This is an explicitly planned fallback, not hidden selection by logits.

Freeze the resulting root/selection reason and complete union of all baseline roots at/after it in window_selection_v1.json before any hidden/margin observation. Each original-order observation captures only targets it actually visits; changed paths in transfers retain unvisited/new roots. No forced baseline owner or extra window search. Intervene only once at owner0 second slot[0,1]. Cross-order first-slot equality is measured, not a universal requirement; recipient first seed and all unselected tensor coordinates must remain unchanged at the intervention.

Two unchanged baseline observations plus order1<-order0, order0<-order1 and order1 original-value placebo for each feasible prompt, using frozen donor tensors. Budget at most20 diagnostic generations x100=2000 new tokens, no teacher/isolated-prefix/warmup forwards. Existing observe/transfer restore hooks/provider on failure and require unchanged baseline/placebo semantic paths, correct actual seed owner, only selected coordinate replaced, unchanged pre/through-intervention semantic trace and complete independent-AR tokens. Post-intervention semantic changes are retained, not rejected merely because a hit appears/disappears. Record per-root margin/state and all losses/gains, including changed/noncommon paths. Compare anchor/v with recipient at all matched roots, report bit/L2 differences rather than assume equality; zero-order velocity remains0.

Bind source/model/snapshot/runtime, native inputs/config/raw, original execution JSON, this PLAN, frozen selection, runner and unchanged helpers before/after. Stop on source drift, hash/token/scope/ownership/nonfinite/OOM/gate failure and preserve it, without swapping prompts, re-signing old outputs or tuning defaults. New .tmp/extension-native-20261014 and .tmp/extension-transfer-20261014-v1 hold local logs/tensors. Formal scalar/full-trace JSON and reports live here; do not export weights/raw tensors.

##### Capacity and completion

Same-model prior sampled RSS~6.1GB/device-global~5.8GiB guides serial capacity. OMP/MKL1, single GPU, no competing launch chain. Sample process-tree RSS/threads/IO, device-global memory and available RAM at.5s. Record actual wall times; heuristic make est is separate and observer synchronization precludes performance ranking. No full matrix or concurrency experiment.

Complete when four native Runs and available fixed transfers pass correctness/scope/provenance, all four prompt outcomes including unfeasible/negative cases and hypothesis boundaries are in STUDY_REPORT with a small combined matrix, and appropriate tests/static/docs checks/local commit are done. No pushes/merges/deletions. These eight accumulated development prompts (q171–q178) remain a small nonrandom sample, not significance or universal EMA attribution.

</details>

<a id="legacy-studies-output-space-extension-docs-rounding-factor-plan"></a>

### studies/output_space_extension/docs/ROUNDING_FACTOR_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_extension/docs/ROUNDING_FACTOR_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/ROUNDING_FACTOR_PLAN.md); original SHA256 `9124baecac7c7b3dfa1f044b9d4263335ecd47c730b4dceb596a95c1b96d12aa`.

#### q178 operator arithmetic factor protocol

Frozen after c4cb011 before new direct arithmetic comparisons. No model generation, prompt/layer/row search or default change. Reuse q178 entry85 FP32 donor slots, native scale and native self/graft layer0 post_attention_layernorm row4 captures. Bind old tensor/report, installed normalization implementation, tools/plan, source and cached weight bytes; read only the needed gamma from safetensors, no model loading.

Phi: 2x2x2 factors (norm32 native GPU versus CPU64 exact-denominator, scale*slot32 versus64, division32 versus64), cast final vector once to32. Native000 must reproduce stored entry exactly, all high111 is compared against the stored CPU64 mapping, including any CPU/GPU backend disagreement. Hold FP32 simulated slots and FP32 scale; no reinterpretation as full EMA precision. Retain all eight effects, non-additive interactions and native/all64 endpoint matching, not an automatic sole-cause label.

RMSNorm: all32 native square, mean-plus-epsilon, rsqrt, input*inverse and gamma product must reproduce saved packed output row4 exactly. Enumerate the 32 mixed precision factor cells on the same native self/graft input, holding the FP32-represented epsilon and FP32 gamma; low stages round input/output to32, high stages to64. Compare isolated factors and full64 against the stored CPU64 intervention; retain all changed-coordinate counts/max/L2, intermediate scalar values and backend disagreement. Packed versus singleton all32 differences are an explicit control. Arithmetic factors change precision and implementation; profiler documents executed kernels but cannot identify a hardware instruction as the unique cause. No downstream generation effect claimed beyond the prior eight-cell evidence.

No model/teacher/auxiliary forward or new tokens. CPU/GPU direct arithmetic with OMP/MKL1, saved profiles local only. Reject nonfinite/input mutation/shape/native mismatch/hash/source failure; save a new version and never overwrite old evidence. Formal scalar findings plus small tests; tensors/model/cache/Chrome traces remain local. No remote/PR/deletion.

</details>

<a id="legacy-studies-output-space-followup-docs-plan"></a>

### studies/output_space_followup/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_followup/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_followup/docs/PLAN.md); original SHA256 `e407142572ae457e725c294a9b52c102e1e36549ead24c7bb17dceb84d422ded`.

#### q179–q182 small follow-up plan

Predeclared 2026-10-14 on a42313d before new outputs. Version output-space-q179-182-20261014-v1. Hypotheses and existing facts from output_space_extension/docs/PLAN.md and STUDY_REPORT.md are retained: q178 has both single-slot transfers accepted8→9 with mixed losses/gains; root86 valid second slot flips under bit-identical committed state, later state changes by anchor~3.05e-5/velocity~2.38e-6. Ranking is the immediate acceptance rule; owner history can change future seeds, so they are different causal stages rather than exclusive explanations. Separate q178 numerical controls test sensitivity without altering default recipes.

Select next four translation records, file order skip18/take4, q179/180/181/182. Freeze datasetSHA, text and independent renderedIDs before make. Same ModelScope Qwen3-0.6B local 11-file snapshot, actual tied weights, FP32/eager/CUDA, nonthinking, T0/EOS/max100, static unpruned[1,1]/k2/BC9, beta.9/beta_prime.99. Source670c744e0535dcad; seed123 and per-prompt generators123/124/125/126. Original runtime deterministicfalse and defaults stay unchanged for native Runs.

Four new native Runs: matched input mean/output0/output1 and independent AR, each four prompts. New version/IDs, native make/launch/process/report and complete source/input/model/raw/factor/Study/AR gates. No context shortening, no rerunning old successful IDs or replacement prompt after a null. At most1600 native new tokens+1200 internal AR audit tokens+72 unchanged warmup tokens. Single GPU round1/OMP-MKL1, no concurrent launch chains. Prior sampled RSS~6.6GB/GPU-global~6.3GiB guides capacity; sample resources at.5s, not allocator/speed rankings.

Transfer selection exactly follows prior choose_window(): earliest eligible common acceptance with a preceding shared rejection; otherwise earliest continuing common rejection; otherwise continuing common unused-owner control; no continuing common root means explicit infeasible, prompt retained. Freeze window_selection_v1.json and runner/PLAN/execution hashes before any hidden/logits observation. Complete baseline root union from that position; each baseline captures its visited subset. Reuse existing observe(), transfer(), compare_window() and choose_window() without editing measured helpers. No forced old owner, root search or gain tuning.

Two original baseline observations and both one-slot directions plus original-value order1 placebo per feasible prompt, max20 generations x100=2000 diagnostic tokens, no auxiliary model forwards. Only owner0 second slot[0,1] once; unchanged remaining coordinates and pre/through-intervention semantics, true seed ownership and independent full AR output required. Original baselines/placebo full semantic equality required. Post-transfer path/state differences and adverse hits preserved. Measure valid first/second slot margins with second AR path flag, same-position recipient anchor/v bit equality and L2, donor-seed comparisons and actual/noncommon roots. Unvisited/new roots are explicit; no fabricated hidden for new roots outside frozen targets.

Hypothesis scope: test whether any new prompt has natural output-order differences or transplant effects beyond q173/q178; preserve all4 and local losses, not only pooledaccepted. Ranking threshold crossing without shared first-token ancestor is invalid for depth2. Nulls cannot exclude other windows; positive effects cannot establish a global method advantage or pure proxy causation when state changes. Four more prompts make12 accumulated development units, not random-seed repetition, significance or cross-task generalization.

Stop/preserve on source/hash/output/ownership/scope/nonfinite/OOM or native-gate failure; do not relax a gate or silently re-sign. New .tmp/followup-native-20261014 and .tmp/followup-transfer-20261014-v1 hold raw/local artifacts; formal scalar/full traces in this Study. Bind source670/model/runtime/snapshot11 files/nativeconfig/inputs/raw/PLAN/selection/tool/helpers before/after. No scientific source/default edits, pushes/merges/deletions; full weights/cache/tensors not in Git. Completion: native and feasible diagnostics checked, all results/limits in PLAN's retained contract plus STUDY_REPORT, appropriate tests/static/docs, local commit.

</details>

<a id="legacy-studies-output-space-followup-docs-plan-review"></a>

### studies/output_space_followup/docs/PLAN_REVIEW.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_followup/docs/PLAN_REVIEW.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_followup/docs/PLAN_REVIEW.md); original SHA256 `40f4492c630bf74aee3ac8b38d40384ebeb9ee6105edd1837fac8931990d2edc`.

#### q179–q182 PLAN result review

Results review2026-10-14. PLAN.md remains the predeclared frozen execution contract; this file records resulting decisions without re-signing successful Runs.

Generality observation expanded: q179's natural output1 accepted9 versusoutput0=8, whileq182 has3 versus4. Both directions onq179 give10; q182 output1<-output0 restores a second-depth hit3->4, reverse remains4. Q180/181 have changedseed but no accepted change. Preserve all gains and losses, no stable order1 advantage or new default inferred.

Ranking hypothesis requires legal second-slot paths, not only first-slot margins: q182root47 original second margin+.481377/-.039644, one<-zero+.334106 under same anchor/v at that root. This is a later owner-history example unlike q178's immediate root86 directsecond-slot graft. Q179 has multiple first-slot oppositions and mixed transfer hits/losses, with post-path microstate differences; no clean all-history mediation proof.

The separate q178 canonical/strict controls show committed-state microdeltas need not produce or preserve its9-count outcome, but no equivalent synchronization control was run onq179/q182. Scope12 existing translation prompts, one fixed window perunit/T0/same model; no statistical or cross-task claim. Proposals for later study: predeclare another-task batch, or one proxy-specific canonical-state intervention. They are not executed or a change to default recipe. Current boundedbatch is complete; no furtherroot/scale search was appended.

</details>

<a id="legacy-studies-output-space-prerequisite-docs-plan"></a>

### studies/output_space_prerequisite/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_prerequisite/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_prerequisite/docs/PLAN.md); original SHA256 `a37509bc64abf3fe5cbc952b0db16f40e06f2000fcfd9e1586ac598e857b55fe`.

#### Output-space prerequisite Study Plan

##### Question and scope

Extracted from research-plan-v6 sections 5.1–5.8/5.11 and E2/E3: can normalized final-hidden masks, conditioned on each draft prefix, use the existing simultaneous verifier without changing the greedy AR tokens? Compare output order0/order1 against frozen_mean and independent AR under the same two-slot chain. This is the first output-space numerical prerequisite, not a Kalman/full-paper or superiority test. Parameters are predeclared, never chosen by acceptance.

##### Fixed protocol

Qwen/Qwen3-0.6B from the already downloaded ModelScope snapshot (same file hashes as input_mask_baseline), local_files_only=true; tied weights required and verified, no untied fallback. SpecBench q171 translation skip10/take1, first turn, nonthinking template; FP32/eager/CUDA, seed123, T0, budget100, stop_on_eos=true. Four cases: frozen_mean chain, output_order0 chain, output_order1 chain, independent AR; each n=1. New version output-space-q171-20261014-v1. Tree static [1,1], k=2, pruning=false, actual BC=9; this differs from the earlier BC30 single-slot tree, so those older accepted counts are not a matched comparison. Reuse q171 as the predeclared development prompt, not a new held-out accuracy sample.

Output state FP32: order0 anchor; order1 anchor plus velocity. Exact observation y: anchor=y, velocity=beta*v+(1-beta)*(y-old_anchor), beta=.9. Node proxy observation uses beta_prime=.99 (conservative fixed choice, not estimated), chain-end has no proxy and only predicts h+v. Each owner starts from a private simulation state; slot j uses anchor+(j-1)*v, gamma=rho=1. Prompt warms exact state in position order. Prefill still uses mean-input masks, produces root token and k normalized final-mask hiddens; no extra target forward. The exact state ends at last prompt hidden, not the generated root: root hidden is unavailable until next block. After verification, observe root and accepted nodes in order; bonus hidden is observed next round. Proxy source is the preceding accepted owner's mask rows. All owner masks are fixed before the forward. Mapping Phi(h)=mean row norm of input embedding table * h/||h||; zero/nonfinite norms fail. Model weights never update.

##### Implementation mapping and evidence

Existing decoder/provider loop remains for old methods. Output provider in structure/algorithm handles state and per-owner content only; HFCausalLM exposes final-normalized hidden from a temporary final-norm hook with try/finally removal (no all-layer storage). Decoder requests it only for the new method; attention/position/tree/cache algorithms remain unchanged. Config API supports explicit spec, no new task/runner. Collect/summarize/write/process reuse current raw correctness/source/evidence gates. Tests verify hook equality/logits, cleanup, state math, owner separation, rejection/acceptance/KV, T0 and shared-seed sampled equality, unsupported untied/nonchain rejection. Existing36-cell strict baseline and full regression must remain equal.

Native make/launch/process/report, round1 single GPU; no concurrent experiment. Prior load/serial resource calibration establishes capacity for same model; sample the new decoder RSS/thread/I/O and device memory, do not claim exact allocator peak from sampling or speed ranking. Model/input/PLAN/source manifests freeze before make. New source never signs an old succeeded Run; historical input baseline remains immutable evidence and is not rerun. Any token mismatch, source/input drift, missing observation, OOM or gate failure stops expansion; preserve failures, change version for a corrected execution.

##### Success and interpretation

Execution: all four new Run statuses succeeded. Quality: exact_match=1 for all three speculative cases and independent full AR IDs equal, Run+Study source/evidence/complete gates pass, final hidden hook test proven, state/owner/observed position trace valid. Record accepted/depth/calls/new/committed separately, no speed pass threshold. n=1 cannot show mechanism advantage or support H1–H13. No full matrix, untied mapping, general trees, order>=2, adaptive Kalman, tuning, concurrent/recovery validation, merge or release.

##### Sources and reading

Local research v6 method/implementation sections read in full for this stage; fixed RPipe7a15c46 research.md read. Knowledge09078015 four research/PyTorch skill references and guides were read in prior stage; no installation/dependency added. Qwen downloaded config declares tie_word_embeddings=true; actual tensor storage equality is required. Local Transformers5.12.1 Qwen3Model/LlamaModel.forward norm->last_hidden_state inspected; validate against output_hidden_states[-1] on tiny. Official references: https://huggingface.co/Qwen/Qwen3-0.6B and https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook . No novelty claim or new literature review asserted.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-cross-prompt-plan"></a>

### studies/output_space_small_matrix/docs/CROSS_PROMPT_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_small_matrix/docs/CROSS_PROMPT_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/CROSS_PROMPT_PLAN.md); original SHA256 `24aaaa5aa2157bfe0c256af1c7a176946c9c246364976dbaec93a124c54d3b7d`.

#### Fixed cross-prompt single-slot replication plan

Predeclared 2026-10-14 before new hidden/logit observation or transfers. Version output-space-cross-q171-172-174-20261014-v1. Reuse observe() from diagnose_output_windows.py and transfer() from diagnose_proxy_transfer.py without editing them, scientific src, defaults or original Runs. New runner only selects saved native inputs and calls those functions; no decoder/sampler/cache/scheduler implementation.

##### Selection and falsification

Use all three other available translation prompts: q171/seed123 from output_space_prerequisite; q172/seed123 and q174/seed125 from output_space_small_matrix. The original output0/1 trajectories match in each prompt, although input-chain acceptance differs. These are prior failures of an order1 acceptance advantage, not new held-out population samples or stochastic repeats (T0).

Selection rule, using only retained semantic traces: find the first root where both output orders accept a draft; choose its immediately preceding common visited root, where both reject all drafts. The fixed intervention roots are q171:63 (first common hit64), q172:42 (hit43), q174:41 (hit42). Replace owner0 second-slot vector [0,1] once with the saved opposite-order vector from a new unchanged baseline observation. Keep first slot and every other coordinate/state untouched. Do not select after looking at direction/logits or add another root if results are null. q174's hit is terminal with bonus EOS, so it cannot demonstrate later continuing proxy propagation; retain this boundary/negative control, not an implied general falsification of all history effects.

For each prompt: two unchanged output0/1 baseline observations, followed by order1<-order0, order0<-order1 and order1<-order1 original-value placebo. Baselines must reproduce original complete tokens/semantic trace. Observe all declared baseline roots from the intervention to termination (union across the two orders); retain visited/unvisited roots when actual post-intervention paths change. Report all baseline failures and gains/losses, not just acceptance count. Targets are selected before hidden observation. No forced traversal along baseline acceptance.

##### Bindings and limits

Bind this PLAN, both existing helper hashes, new runner, two original execution JSONs, snapshot, native config/decode_inputs/decode_observations and source670c744e0535dcad before/after. Same prepared ModelScope Qwen3-0.6B 11-file snapshot; weights are local-only and actual model digest/runtime must agree. FP32/eager/CUDA, T0, tied model, EOS, budget100, static unpruned[1,1]/k2/BC9, native gains unchanged. Per-prompt generator seeds as above; input IDs and independent AR must match the exact native evidence index (q171 index0, q172 index0, q174 index2).

Budget15 generations x100=1500 new tokens: six unchanged observations, six real transfers and three placebos, no auxiliary model forward/teacher/warmup. Expected native lengths47/14/13; each replay must equal independent AR completely, without shortening/context-limit. Model call count must equal1+actual decode_calls. Baseline trace and placebo must reproduce originals exactly; pre/through-intervention trace must match in transfer cases. At actual forward assert selected call is the fixed root, donor equals declared saved tensor, and nonselected coordinates are bit-identical. Existing functions restore hooks/provider on failure. Record forward counts/resources and all full actual traces.

Compare captured committed anchor and velocity with recipient baseline at every common declared root: bit equality and max absolute/L2 deltas; inequality is a measured result, not automatically a gate failure or license to silently realign. Order0 velocity must remain zero. Incoming proxy and seed changes are measured separately, with first-seed target margins and baseline accepted-node lengths at matched roots. The first seed at intervention must be unchanged; actual source root/owner must match actual returned trace. Skipped/new roots remain explicit and do not become matched-state evidence.

Use one new .tmp/output-cross-prompt-20261014-v1 directory with baseline/transfer tensors and failure report; refuse existing output. Scientific full-source validation is retained, not rerun as new evidence. Three prompt versions/seeds are not a full matrix. Serial single GPU, OMP/MKL1. Prior same-model observed RSS~6.1GB/GPU-global~5.7GiB guides capacity; sample RSS/threads/device-global memory/available RAM/I/O at.5s, approximate and includes background. Stop on hash/output/scope/ownership/nonfinite/OOM failure and preserve it, no replacement case or retuning. No push, merge, data deletion, new default or original Run re-sign.

##### Conclusions allowed

Test whether q173-like hit loss/restoration appears under this fixed intervention outside q173. Changes in proxy/margin without acceptance change count as null for hit transfer, not a success. A null does not disprove every other root intervention; a changed hit does not prove global history mediation or stable method advantage. Explicitly report q174 termination boundary and existing matched-order baselines. Save scalar/full-trace evidence in this Study and local tensors ignored; results and scope go back to STUDY_REPORT with a local commit.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-plan"></a>

### studies/output_space_small_matrix/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_small_matrix/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/PLAN.md); original SHA256 `cfa06142c9380a3b6f4ecaa7367263919bb235a5152807faa8ae288e6fa0fa24`.

#### Output-space small-matrix Study Plan

##### Question and selection

Read the latest output_space_prerequisite report and offline_diagnosis before selecting this stage. q171 showed input-chain accepted17 versus output-order0/order1=10/10, despite different owner-mask statistics and identical output-order verification trajectories. Does this descriptive pattern persist on three further predeclared prompts, and at which matched committed-prefix positions do the greedy trajectories differ? This reuses the working implementation; not a mechanism/novelty/Kalman test.

Selected by contiguous dataset order, without observing new outputs: translation skip11/take3, question_id172,173,174; first turn German->English, nonthinking Qwen template. [Prompt manifest](../studies/output_space_small_matrix/docs/prompts_manifest.json) freezes exact text/reference and dataset hash. q171 is excluded and its raw/source/PLAN remain untouched. n=3 prompt units, not three independent seed replicates or a held-out population sample.

##### Fixed model/method/budget

Same ModelScope Qwen3-0.6B snapshot and 11 file hashes, actually tied weights, FP32/eager/CUDA, T0, EOS stop, max_new_tokens100 per prompt, max_prompt_len2048. Source670c744e0535dcad on local9ba5c3a, upstream7a15c46 v0.3.1, existing torch2.11.0+cu130/Transformers5.12.1 maintained. No algorithm/source changes. Native Run seed123; per-prompt generators use seed+index123/124/125 (T0 gives no sampling selection).

Four Experiments, one Run each with all three prompts: frozen_mean input chain, output_order0, output_order1, independent AR. All spec methods: unpruned static[1,1], k2, BC9, efficient, same prefill input mean. Output order1 beta=.9/beta_prime=.99/gamma=rho=1 and mean embedding-row-norm mapping unchanged. No tuning/sweep, adaptive Kalman, untied/general tree or full matrix. Measured output budget at most4*3*100=1200 new tokens, plus at most900 tokens for three per-prompt AR audits; existing warmup8 per case and internal audit remain, up to56 additional warmup tokens. No replacement of a low-acceptance prompt.

##### Design, expected information and diagnosis

Report all per-prompt accepted/depth1/depth2/calls/new/committed and pooled sums. Quality passes only if each speculative output and its internal AR equal the independent AR under identical prompt IDs and full budget, with no context-limit shortening. Acceptance improvement is not a pass threshold. Preserve negative/zero/mixed observations, do not prefer the best case.

For each same-prompt pair (input/order0/order1), align trace rows by absolute root_position and verify cache_before and committed prefix are equal. Compare accepted-node lengths and committed segments at these common positions; list noncommon positions, which arise because different acceptance lengths skip different future roots. Do not compare step number across different generated-prefix positions. Compare output0/order1 owner-mask difference statistics at matched roots, but do not call this full vector/logit agreement. Missing draft probabilities, full mask vectors and exact/proxy-hidden pairs prevent causal attribution: any mechanism explanation remains a hypothesis. This stage narrows whether q171 was the only observed pattern before instrumenting heavier diagnostics.

##### Execution, provenance and stop

New version output-space-q172-174-20261014-v1 and new Run IDs, separate Study. Freeze PLAN/prompt/model manifest/declarations/recipe/phase files and SpecMTP source before make; actual input/model/runtime evidence recorded by existing stages. Source/freeze/gate check before execution; same architecture/science code as the just-passed270tests/36 strict cells and wheel/sdist checks, so do not repeat those hardware experiments or alter code to create new evidence.

Native make --num-gpus1 --init-gpu0 --round1; launch same queue, then process/status/logs/report. Existing verified serial capacity of same model6GB device-global sampled peak guides round1; sample new process RSS/thread/I/O/device memory and available host RAM, distinguish heuristic est from measured Flow/launch wall. OMP/MKL1, no competing experiment. Any OOM/token mismatch/source drift/missing raw/gate failure stops expansion and preserves failed Run. Successful old IDs never re-signed; corrected execution must newversion.

##### Completion and limits

Execution complete if four new Runs succeed. Quality if all12 prompt outputs, full AR equality, current source/input/Study gates and independent prompt identity mapping pass. Save per-prompt paired trajectory diagnosis with actual raw hashes, NUMBERS/report/chart and redraw tool. Formal selected evidence local; full raw/log/modelcache ignored. No push, PR merge, data deletion or publication.

Three contiguous prompts are a small feasibility/development matrix; not a generalization, significance or method-superiority result. Negative evidence can motivate a predeclared proxy-vs-exact hidden diagnostic, but does not license immediate large-scale runs, retuning or declaring all output-space methods ineffective.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-scale-plan"></a>

### studies/output_space_small_matrix/docs/SCALE_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_small_matrix/docs/SCALE_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/SCALE_PLAN.md); original SHA256 `54eaf6cb1e6950ab713935f8fa31c8d291c2c000d5ddc6bd957ffa930b50090c`.

#### q173 root83 direction/norm counterfactual plan

Predeclared2026-10-14 after the window diagnosis: at root83, frozen-input seed mask norm=.1912895292, output-order0/1 seed mask norm=.9250697494 (4.84x); output0/order1 first vectors are exactly equal. Their correct-target margins are+.63981 versus-5.51714. The first-slot target is7602, same true prefixAR[:83]. Direction and norm differ simultaneously; original observations do not separate them.

Fixed 2x2: input/output recorded seed-vector direction crossed with the two observed norms, not a searched grid. Use q173 root83 only, seed input from selected_tensors.pt bound by the window diagnosis SHA. No new generated tokens, decoder replay, prompts, gain/norm optimization or scientific-source change. Four teacher-forced prefix+mask forwards maximum (two observed combinations reproduce the window isolated checks, two are counterfactuals), FP32/eager/CUDA/current runtime, same verified ModelScope snapshot. At most84 positions each, serial/threads1; version output-space-q173-scale-20261014-v1. Copy neither weights nor originalRun, delete nothing.

Require source/model/snapshot/runtime/input hash consistency, raw first vectors as saved, nonzero finite norms, observed combinations identical top1 and target-margin agreement<=1e-3 with recorded isolated logits. Preserve all four outcomes, including failures. Report target probability, top1/top2 and margin, distinguishing these isolated mask-position predictions from full-decoder accepted counts: counterfactual queries do not establish a stable method improvement and are not retroactive selection of a new default scale. No push/merge/full matrix.

The follow-up may separate a local direction effect from a local norm effect, including interaction. It cannot show generalization, select global beta/mapping, or explain every trajectory. The original3prompt matrix, q171 negative result and locked science recipe remain unchanged.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-transfer-plan"></a>

### studies/output_space_small_matrix/docs/TRANSFER_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_small_matrix/docs/TRANSFER_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/TRANSFER_PLAN.md); original SHA256 `f90669213dbdde85a625a087386764376daa9be8bb2a005cf24f98a0bf780b8b`.

#### q173 single-slot historical-proxy transfer plan

Predeclared 2026-10-14, before the three intervention replays. Version: output-space-q173-transfer-20261014-v1. Builds on the retained matrix, WINDOW_PLAN, SCALE_PLAN and post hoc structural diagnosis; it does not replace an original Run or add a formal acceptance cell.

##### Question and fixed intervention

At saved root83, output0/1 owner0 slot0 vectors are identical. Their slot1 vectors have the same observed norm (about .92507) but cosine .99831766. Both original paths accept one draft at root84 and next visit root86. Only exchange the returned owner0 slot1 vector at root83 (zero-based indices [0,1]), using the saved opposite-order current_masks tensor. Leave all other vectors, committed state, logits, decoder, cache, gains and defaults intact. This tests whether a single earlier mask can propagate through proxy history to later candidates without a direct slot0 velocity term. It cannot isolate all earlier velocity history or establish a general method advantage.

Three serial cases, no additional selections or reruns for an unfavorable result:

| recipient | root83 slot1 donor | role |
| --- | --- | --- |
| output_order1 | saved output_order0 | one-to-zero transfer |
| output_order0 | saved output_order1 | zero-to-one transfer |
| output_order1 | saved output_order1 | original-value placebo |

Replay the unchanged SpecDecoder directly with a scoped provider patch, following existing tools/profile_cpu.py and window tools. Do not use the unchanged-window observer's baseline acceptance path for post-intervention seed ownership. Infer the preceding selected owner from actual root progress, then check against actual returned trace. Observe only predeclared roots83/84/86/94/97 if visited; a root skipped by a changed acceptance path is recorded as unvisited, not forced or silently substituted. Capture first-seed margin, actual seed source owner/position, incoming proxies/anchor/velocity, current masks and mask hidden rows. Report all three cases, including null and adverse effects. No search over seed, root, scale or donor.

##### Bindings, budget and execution gates

Bind scientific source670c744e0535dcad, unchanged original execution_v1.json, original Run config/decode_inputs/decode_observations, WINDOW_PLAN and observer JSON, this PLAN, tool/helper hashes and selected_tensors SHA47166a3b41e0c00ed39110593fe7f83b20323c2e3714fe0355f2a43aaa9dc14c. Reuse the same locally prepared ModelScope Qwen3-0.6B 11-file snapshot, verify all hashes before/after, actual model digest and runtime equality. Tied model, FP32/eager/CUDA, unpruned static[1,1], k2/BC9, beta .9/beta_prime .99, T0/EOS, seed124, max_new_tokens100. No download, src or default recipe edit.

Budget at most three decoder generations x100 new tokens, no auxiliary teacher/prefix forwards and no warmup. Each generation must reproduce all saved independent-AR tokens (33 for this q173) without context shortening. Instrumentation forwards must equal1+actual decode_calls. Select the intervention call from the original root83 index; at actual forward assert it is root83 and exactly one provider call was modified. Before/through root83 verify full semantic verification/commit/KV trace and original root83 masks. Nonselected tensor coordinates must remain bit-identical, the replacement must equal its predeclared donor, and the patch/hooks must be restored even on failure. Placebo must reproduce original full tokens/semantic trace. Post-intervention semantic changes are permitted in the two transfer cases and must be retained.

Use a new .tmp/output-transfer-20261014-v1 output directory; refuse existing outputs. Preserve failed output and all earlier data, no push/merge/delete. Formal scalar JSON/report stays in this Study; local tensor payload stays ignored. Sample process-tree RSS/threads, device-global GPU memory, available RAM and I/O at .5s; these are approximate peaks with background occupancy, not allocator or speed rankings. Prior observed capacity is about6.1GB RSS and5.7GiB device-global GPU memory; serial execution only. Stop on binding/replay/ownership/scope gate failure, nonfinite tensor, OOM or resource failure; preserve failure and do not change contract to pass.

##### Interpretation boundary

If an exchanged second slot changes the next proxy and subsequent first-seed direction/margin while the first seed at83 is unchanged, this supports a local historical propagation route. If only immediate proxy changes or the later94/97 margins/acceptances persist, record that this single early intervention does not explain those later gains. Neither outcome identifies all hidden mediation or the contribution of current velocity in later slots. No stable benefit, global EMA causal claim, new optimized default or performance ranking follows from this one q173 window.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-window-plan"></a>

### studies/output_space_small_matrix/docs/WINDOW_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/output_space_small_matrix/docs/WINDOW_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/WINDOW_PLAN.md); original SHA256 `f6e966c71706fe13c07295e2a73037b300295e9048506b4e470c45f9d5689a7b`.

#### q173 matched-window diagnostic plan

##### Purpose and predeclaration (2026-10-14)

The fixed three-prompt matrix is retained. Diagnose q173 at root83 (input-chain advantage) and root94/97 (order1 versus order0 advantage), selected from the saved paired-prefix analysis before examining hidden/logit values. No other windows, gain search or replacement prompt. Version output-space-q173-window-20261014-v1 identifies an instrumented diagnostic replay, linked to the original four native Run IDs; it is not counted as new acceptance evidence or used to re-sign them.

Existing tools/profile_cpu.py measures the unchanged SpecDecoder directly; follow the same tool pattern, with read-only hooks around the existing forward/provider. No new decoder, sampler, cache routine, Run scheduler or scientific src edit. Tool source hash, this PLAN hash, original execution JSON/raw hashes, source670c744e0535dcad and ModelScope11-file hashes bind the measurement. Temporary tensors/logs go to a new .tmp/output-windows-20261014 directory; formal scalar evidence/report stays in this Study. No push, merge or data cleanup.

##### Fixed replay and budget

Same Qwen3-0.6B ModelScope snapshot, FP32/eager/CUDA, unpruned[1,1]/k2/BC9, order0/order1 gains, T0, EOS, budget100. Replay q173 for input/order0/order1 serially once per case, using the saved rendered input and original per-prompt generator seed124. No warmup or timing comparison; hooks may synchronize and distort wall time. Existing runtime flags must match. Full output and semantic verification/commit/KV trace must reproduce the corresponding saved matrix case exactly; otherwise preserve failure and do not interpret the measurements.

At most300 decoded new tokens across three replays (the saved outputs have33 each). One teacher-forced causal forward over the saved AR sequence supplies exact normalized hidden references; it is not a new AR generation. For each of3cases*3windows, at most1 additional causal prefix+actual first seed-mask forward checks that recorded seed logits have the same top1 as an isolated correct-prefix replay. Max10 auxiliary forwards, max103 positions per forward, singleGPU/threads1, no concurrent experiment. Capacity uses the prior same-model serial calibration and fresh device/RAM check, recording new sampled resources separately.

##### Observation timing and quantities

Current-block masks seed the NEXT tree. Acceptance at root p uses mask logits from the preceding block's deepest accepted owner's masks (first mask input at position p), not the masks being constructed at root p. Record seed source block root/owner, absolute mask positions and input vectors, normalized hidden/logits, as well as current root exact hidden and output-provider committed anchor/velocity/proxies. The original root verifier and cache keep logic remain untouched.

At each selected window, compute first-slot target token versus best alternative logit margin, top1/top2 IDs/probabilities and target probability; the target is the independently saved AR token at p+1. Compare seed-mask norm, proxy-versus-exact root hidden cosine and relative L2 error. Record second-slot candidate too, but label its AR comparison valid only if the first draft token matches the AR next token; an invalid draft prefix is not an exact hidden observation. Save full selected tensors locally to make scalar calculations reproducible, without publishing weights or full vocabulary logits.

The auxiliary prefix+mask input is AR[:p] followed by the actual stored seed mask at position p. It has the same true prefix as the preceding accepted owner, which is independently checked from the original block layout. Record differences in original versus isolated target margins; require equal top1 IDs and normalized-hidden cosine>=.99999. Root exact hidden versus teacher-forced AR hidden also requires cosine>=.99999. These are diagnostic consistency gates, not acceptance-improvement thresholds.

##### Interpretation and stop

All three replay/token/trace/source/runtime/input and window-completeness gates must pass. Verify hook removal and provider method restoration, forward counts (1+decode_calls per replay; no observer-induced target calls), and model digest unchanged. Stop on any mismatch, OOM, missing capture, source/file drift or consistency gate failure; no silent correction or retry on an old diagnostic output directory.

A seed margin crossing0 at the same verified prefix explains a greedy candidate/acceptance difference under these fixed observations. It does not by itself establish that EMA velocity is optimal, that proxy-error magnitude causes the gain, or that output-space masks outperform input masks. Keep both advantage and disadvantage windows and all failures. Do not retune, expand the matrix, change src or declare Kalman supported.

</details>

<a id="legacy-studies-qwen3-4b-bf16-directions-docs-plan"></a>

### studies/qwen3_4b_bf16_directions/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_directions/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_directions/docs/PLAN.md); original SHA256 `0ef4df6b1856c89e10b71d87a5cea16abfcbecff6391adbc086455cc0d5b4cf7`.

#### Fixed480 BC30 direction controls (2026-10-16)

Four windows cannot establish whether a direction is special. Complete BC30 on the accepted five-method Study's480 SpecBench first turns before deciding BC60 from actual work, without new methods, prompt/data changes or timing. Preserve order, non-thinking template, input IDs, EOS, seed123+index and max_new_tokens256. Use ModelScope Qwen3-4B BF16/eager,T0,two masks,dynamic/pruned tree,BC30/draft budget9,updating input meanλ0.1,γ1,normalize=false,extrapolate_ema=false; verify model/source/input before forward.8B and operator studies are excluded.

Six groups: template freezes v=e2−e1 and updates only mean, m_i=mean+i*v; random0/1/2 independently draw CPU Gaussian vectors with seeds0/1/2, normalize inFP32 to each prompt's template-vector norm, convert toBF16 and keep v fixed after every commit; zero reuses complete30/input_zero, which is updating rather than frozen mean; content initializes native input-firstβ0.9999 from the first user-content-token suffix, restores the full-prompt mean and subsequently uses native mean/velocity EMA. The suffix retains ending template tokens and has no extra norm matching. Everything else, including anchors/tree/cache/verification/stopping, stays native. Do not substitute native updating input_first for fixed template, or infer full480 equality from four-window ties. Compare template with old input_first explicitly. Reuse all old five-method/AR results with original source digests; differing package digests after reference-code import require byte checks on scientific decoder/provider/model, never re-signing old results.

Primaryτ=total decode committed/total decode calls; also report prompt means and equal category-pooled means. Accepted counts precede EOS/budget clipping; main calls include prefill. Pair every group versus template and every original method by question_id for accepted/calls/τ win/tie/loss; include output lengths and AR first differences, without equal-output speed claims. All committed tokens must equal their correct owner's same-forward argmax, with native hash/coverage/owner verification. Reused AR is diagnostic only, with no newAR GPU work. Each new group requires480 and native Run/Study process success; missing entries are not zeros/ties. Retain source/math/OOM failures without automatic retries or prompt repair.

Use one round1 native queue for five new Runs, not a second queue. Historical admission queriedGPU0's actual foreign GPU Python and≥14GiB free, rechecking each prompt, without CPU-script inspection or preemption. Record raw tokens/trace/argmax and initialization scalars locally. Freeze declarations, code and source manifest before running; CPU tests cover fixed v, unchanged mean/zero equivalence, random norms/seeds, content initialization and tinyBF16 gates. No decoding clocks; count forwards/tokens for the laterBC60 decision. Predeclare all fiveBC30 cells, never select winning directions for extension. Deliver full status/sources/limits through commit/push/PR→dev and remove the merged remote topic; retain local branches/data, never merge main, force-push or use popups.

</details>

<a id="legacy-studies-qwen3-4b-bf16-directions-docs-recovery-plan"></a>

### studies/qwen3_4b_bf16_directions/docs/RECOVERY_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_directions/docs/RECOVERY_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_directions/docs/RECOVERY_PLAN.md); original SHA256 `ff430b2b472938ba430ea1411d691535eebf25de3f96c6883415177fbf8e00e9`.

#### Recovery after foreign-GPU admission interruption (2026-10-16)

Original random0 Runbe910fe1fa042b28 completed46 prompts before nvidia-smi showed foreign PythonPID65348. It stopped before the next prompt under its frozen resource policy, not from mathematics orOOM; preserve46 raw paths and9028 passing audited tokens in resource_interruption_v1.json. No valid full480τ exists, and no partial/four-window stitching is allowed.

The original queue had automatically entered random1: do not interrupt it, launch a second queue or kill the other process. After natural termination inspect every factor; never rerun complete template or old zero. Recovery changes only version/Run IDs and resource handling, with all480 rerun for each incomplete factor. Preserve failures, inputs,prompt/model/dtype/control/seed/stopping/math contract, original control() and scientific sources; no KV checkpoint-resume claim. Partial work is retained but excluded from final valid counts.

Terminal update: random1/2 also encountered real foreign GPU Python; content failed admission before loading. The original queue exited1 naturally. Repeated assert-based interruptions wasted completed prompts. The new recovery Study keeps control() verbatim and56 scientific files, replacing only the two resource rejection sites with5-second waits before loading/at prompt boundaries in the same Run, without forwards or restarting during waits. Disclose changed helper/driver provenance; it is not byte-identical as a whole or a new scientific method. Still require actualGPU vacancy and≥14GiB free; CPU processes do not block, and failed groups are not automatically retried.

Keep the original Study's failed finalisation. Select one complete native successful Run per direction by the predeclared rule: original success first, otherwise its recovery factor, never byτ. Recovery native process and combined480/seed/parameters/model/runtime/argmax/source/control identity checks must all pass; any missing direction means incomplete. Reuse zero without re-signing. Report failed/repeated token/forward work separately from valid results and never as timing. Only the complete six-group result closes the target and enables PR→dev/remote cleanup; do not shrink the matrix to successful groups.

</details>

<a id="legacy-studies-qwen3-4b-bf16-directions-recovery-docs-plan"></a>

### studies/qwen3_4b_bf16_directions_recovery/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_directions_recovery/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_directions_recovery/docs/PLAN.md); original SHA256 `45c24df0cdca96ba2e8f31753ac8c281fddd88c04937f07780e2a9387187b394`.

#### BC30 full480 resource recovery (2026-10-16)

The original queue ended naturally: template480 passed; random0/1/2 stopped after46/50/9 prompts on real foreign GPU Python; content was blocked before loading. No interrupted cell has valid full480τ. Preserve every original Run/result/partial audit. Four incomplete factors restart all480 in a new Study/version/Run; never stitch partials or rerun complete template/zero. Keep original model/data/prompt/T0/BF16/eager/BC30,two masks,dynamic tree,seed123+index,max256/EOS,mean/direction/EMA. Random seeds0/1/2 are fixed norm-matched directions; content uses native suffix initialization/EMA without extra norm matching.

The only execution change is resources: before load and each prompt query actual nvidia-smi, requiring no foreign GPU Python and initial≥14GiB free. Otherwise sleep5 seconds and retry while retaining the Run/model/completed prompts; no forwards/restart,CPU enumeration,preemption or killing. Write waiting state to local.tmp/full-direction-resource-wait.json; decoder timing remains zero. Wait only at prompt boundaries, with no KV/checkpoint-resume claim. OOM,source or mathematical failures still fail without automatic retries. One round1 queue starts only after confirmed original exit1. Documentation/template alignment must not interrupt running experiments or alter their frozen plan bytes.

Keep56 scientific files and old input/model/source_refs; preserve control() AST/bytes exactly. Freeze the disclosed helper/two-call-site difference. CPU tests verify waiting/recovery, correctGPU-only detection, identical control and the existing tinyBF16 gate. All four480 cells and native process must pass. Combined results use original template, predeclared recovery random0/1/2/content and old input_zero: one complete valid Run each, noτ selection. Verify model/runtime/parameters/seed,all480 argmax/native sources. Report threeτ aggregations, pairs versus template and five methods, AR/output-length diagnostics and wasted resource work. DecideBC60 only after actualBC30 work; no8B/HF/new method/prompt/timing/operator work. CompletePR22→ready/dev/remote cleanup under authorization, without main,force-push,data deletion or popups.

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-geometry-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/GEOMETRY_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_five_methods/docs/GEOMETRY_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/GEOMETRY_PLAN.md); original SHA256 `86b5b92e283c059e0abae35db3b86aa06f54e630ea6ebd195a9cc6ab231efe00`.

#### Historical4B geometry and acceptance-depth protocol
2026-10-15: investigate why output zero/first trail input first using CPU evidence, then minimal4B observation. No8B, author-code reproduction, changed recipes/source/old make contract or re-signing successful Runs.

##### CPU, no model forwards
Read all480 prompts at both BCs/five methods from decode_observations.json, bound to paired_evidence_v1.json SHA. Check complete traces, attempt/accept per depth and totals. Use real attempts for conditional rates. Root bins[0,32),[32,64),[64,128),[128,192),[192,256) report the fraction of blocks accepting >=1/2, not a depth-conditional rate.
Decompose tau=1+accept_d1/calls+accept_d2/calls-trim_loss/calls with trim_loss=accepted+calls-committed: a counting identity, not cross-path causal attribution.
Check mask min/max norms against r, affected slots under fixed state/proxy, EMA updates and prediction contamination. Old logs lack vectors; do not infer cosine distributions from token IDs/norms.

##### Minimal admitted GPU observation
Choose the first coding/math_reasoning/summarization/translation item independently of outcomes. BC30, input first/output zero/output first; BF16/T0/eager/default, max64/EOS:12 diagnostics, <=768 generated tokens and768 total forwards. Reuse verified ModelScope4B, one serial process, no download/new AR/timing.
Historical initial admission checked twice for no foreign/unknown model jobs and >=18GiB freeVRAM, then each diagnostic; desktop and verified CPU CI were listed separately, without claiming an atomic global reservation. Current project admission supersedes this.
First two attempts deferred before loading. download_layers.py only downloaded/hashed; build_official_container_v2.py only built an extension/container without GPU/device parameters. Third loaded but stopped at a new CPU watcher before generation, fourth deferred. Windows venv child launchers were classified via verified CPU CI/release/watchers, explicit CPU-only RPipe tests and parent PID chains; unrelated unknown jobs still rejected. Preserve all deferrals, never count load as generation. Exact historical script SHAs remain in the source archive/manifest.
Use unmodified SpecDecoder/HFCausalLM/providers with read-only hooks: actual mask inputs, final-normalized hidden, raw state/proxy. Only root/accepted-path owners have real continuations. Target slot j is the owner's actual token j+1 embedding; count skipped tail targets. Compare true-owner hidden, raw proxy hidden, norm-scaled actual BF16 mask and true embedding separately.
Require full native verifier-argmax coverage, output matching a prefix of the old full Run and no hook-added forward. Shared AR prefix agreement is diagnostic. Vectors/raw tokens stay local; publish pointwise scalars, quantiles/counts/SHA. Geometry runs on CPU; no operator/reduction changes or BF16 divergence investigation.

##### Falsifiable hypotheses and limits
H1 wrong radius: test oldtrace ranges/newmask norms; correct radius does not validate direction/distribution.
H2 hidden/embedding incompatibility: compare raw/actual/true-owner/input mask/target; tied weights do not imply inverse mapping and cosine is not acceptance causality.
H3 velocity affects few slots/second-depth opportunities: combine slot identities and conditional acceptance; fixed first slot does not imply identical full paths.
H4 proxy error hides velocity effects: evaluate owners with truth; separate shared prefixes/different paths, no rejected-owner truth.
H5 beta/init/order differ across spaces: report velocity/anchor norms and interslot angles, not a space-only causal claim.
Observation only, no tuning/default changes/large matrix/teacher/operator study. If admission fails, deliver CPU results and explicit geometry gaps. Preserve negative results and report/PR dev.

Historical recorded SHA256 values (source ordering; identities remain in the immutable original):

- `10689bac808d0e5fa1958d306508f459d3c4637dc92a51bf5b6c1bb915545026`
- `b22cbe9814bcfcf85e9e6caade2a60a2258d2bb83f5e1d856124753bf8976e44`

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-active-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_ACTIVE_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_five_methods/docs/MASK_ACTIVE_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_ACTIVE_PLAN.md); original SHA256 `229e04dd5f4a3b8eb5b46d388b8e60f578da7dc87512a21a86ee680c11c0494d`.

#### Historical conditional second-depth windows
After the first24 cells (code ce7c329), all four unconditional next trees had true d1 outsideTop1, so they could not test velocity where depth2 is available. Retain all negatives without rerunning.
Select q121/q401 from original output-zero traces: first consecutive block pair with later accepted_nodes=[1]; intervene on predecessor roots3/19, next roots4/20. This is outcome-conditioned selection, not a full480 rate or random extension; report separately.
Reuse MASK_WINDOW_PLAN's same-history shadow, native input-first control and four local output formulas, no tuning. Two windows*six cells, two local forwards each, <=64 total including reconstruction.4B BF16/T0/eager/default ModelScope only; no8B, teacher, AR, timing or operators.
Nonpreemptive admission per context; preserve deferrals/unknown jobs. New .tmp/4b-mask-active-v1; keep files/branches/data. New tool gets its own SHA, ce7c329 Git bytes remain the original source; never re-sign old data using the new tool.
Require old control prefix, current-owner argmax and next control trajectory plus actual depth2 opportunity, first candidate hit and exactly one accepted. Preserve ranks/widths/acceptance and all WTL; AR remains old diagnostic without forward.
CPU reads saved actual GPU trees; do not require CPU softmax reconstruction equality. Pruned strict/tied-rank intervals distinguish guaranteed/possible coverage from actual acceptance, without investigating cross-device numerical ordering.
Failure stops with evidence, no replacement prompt/window to obtain success. Separate stage report and PR dev; no main/force/deletion.

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-direction-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_DIRECTION_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_five_methods/docs/MASK_DIRECTION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_DIRECTION_PLAN.md); original SHA256 `c824223f7a2c173c8cebf5f5971caca760ae34177cee42fbe82a54d266aa38c6`.

#### Historical fixed-data local direction controls
2026-10-15: four user-specified direction groups, no new main method or changed five-method definitions/data/prompt/template/token. Original11/11 full paired cells were complete. This protocol is local analysis only.
Freeze original output-zero q121/R1,q401/R0,q241/R1,q161/R2, same prefixes/cache/tree/positions/attention. Next block retains original output-zero rule. Eight cells/window=32, <=96 total forwards, no further CPU screening/fullmatrix.
Cells: zero=output-zero (not input-zero); input_first=native same-history; input_template=fixed original e2-e1 of first two prompt embeddings; input_random_0/1/2=fixed CPU Gaussian directions, each seed shared across four prompts, normalized to original difference norm; input_mean=zero velocity with updating mean; input_content=initialize velocity at first two user-content tokens, process remaining prompt with original beta, then committed tokens. Mean always starts from the full prompt and updates lambda=.1.
CPU Gaussian sampling is not FP32 model inference. Cast actual BF16, record norm ratio/cosine; error<1%, no seed search. Locate content using original first turn/tokenizer offsets; rerender only to verify frozen input_ids exact, always feed original manifest. All four starts are index3; mismatch stops rather than changing prompts.
Content retains native magnitude without template norm matching, so magnitude/direction are confounded; it reruns original velocity EMA from content, not just a fixed readout direction. Other beta=.9999/lambda=.1/gamma1, BC30/two masks/dynamic pruning fixed.
Report both target ranks, widths, actual next-block acceptance and WTL against input zero/template/native separately. All three random seeds separate, no best seed. Random/content gains falsify only the local “original direction only” hypothesis, not full480 generalization.
Current-owner argmax/path unchanged, output-zero replays old Run; all local committed choices equal verifier argmax. Compare template/native/zero measured controls with prior batch, no old re-signing/new AR. Bind data/manifest/content start/seeds/actual norms/cosines/tool/protocol/model/source/private logits SHA.
Single --template-control --early-template --direction-controls command; historical actual GPU Python/freeVRAM>=14GiB admission, no CPU process/source hashing. Foreign GPU/capacity defers, no preemption/kill/overlapping waiters. BF16/T0/eager/default4B only, no timing/operators. Control/math/source/nonfinite/OOM/budget failure stops without automatic rerun/precision change. Full weights/cache/tokens/logits local; scalars/report commit/push/PR dev.

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-early-template-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_EARLY_TEMPLATE_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_five_methods/docs/MASK_EARLY_TEMPLATE_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_EARLY_TEMPLATE_PLAN.md); original SHA256 `62f1e118711822047af8c82030ef16a0144b40848c9bfb98384c013197a21ce4`.

#### Historical early same-prefix template controls
After PR19's24 late/conditional controls, move to CPU-selected early windows without changing scientific source/defaults/old evidence. Admission now observes actual GPU processes/free memory, never CPU Python source hashes.
Original BC30 output-zero prefixes/cache/tree/history; four cells output-zero/native input velocity/fixed first-template difference/zero input velocity. Three input cells share updating mean, beta=.9999/lambda=.1/gamma1; input-zero is not frozen.
Frozen windows (current/next/first original input acceptance difference): q1211/2/2,q4010/1/1,q2411/3/3,q1612/4/3. First three are predecessors reaching the candidate root. Output-zero skips q161root3, so root2->4 is a nearby reverse candidate, not exact original inputroot3 reproduction. Conditional existing-result selection, different original/control paths, no outcome-based window replacement.
Read two mask-logit rows, natively construct next tree and verify one block. Report ranks,widths,acceptance and full seed-logit equality with native input. Current-owner argmax unchanged, outputzero replays old Run, all local commits verifierargmax; AR remains diagnostic without forward. Two intervention forwards/cell plus reconstruction, <=64 total; no path continuation.
Query GPU0 actual compute apps and free/used memory. Exclude ownPID; foreign GPU Python defers. WDDM desktop apps count toward memory but not Python experiments. Historical initialfree>=14GiB (about8GB weights plus transientKV/tree margin); after load check foreign GPU Python. No CIM/CPU source hashing, no atomic-global-lock claim.
One serial command, no background/overlapping waiting queue. Busy yields zero-forward deferred. Reuse verified ModelScope cache immediately when admitted. New output directory once, preserve failure; source/control/math/budget/OOM failures stop with no autoretry/precision change/operator tracing.4B BF16/T0/eager/default only. Bind protocol/tool/model/source/raw/private logits SHA; only scalars/report public, local weights/cache/tokens/logits preserved. PR dev, no main/force/deletion.

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-reciprocal-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_RECIPROCAL_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_five_methods/docs/MASK_RECIPROCAL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_RECIPROCAL_PLAN.md); original SHA256 `f8d78aecb23a1a8dca7b9c379c6ec28d3c00f0723dacb4e6f30fb89441e0fa1f`.

#### Historical early bidirectional mask replacement
After the16 early template cells, add output-first and reverse replacement from the original input-first path; no more CPU screening, changed science/defaults or old re-signing.
Serial origins: original BC30 outputzero and inputfirst. Freeze q121/R1,q401/R0,q241/R1,q161/R2, require roots exist; no replacement by results. Each origin/prompt: same-history outputzero,inputfirst,outputfirst, replacing only current-block owner masks, retaining prefix/cache/tree/positions/attention. Input beta=.9999/lambda=.1/gamma1; output beta=.9/proxybeta=.99 and original norm mapping.
Output origin reuses zero capture with read-only first-order shadow. Input origin rebuilds original path and uses native last_hidden_only observation of verified prompt/committed hidden to update both shadows; proxy is the previous real block's deepest-owner mask hidden. No writeback/teacher/newAR forward.
Current-owner argmax/accept path must equal native control; construct next tree from each cell's masklogits, verify one block, future masks follow origin (inputfirst or outputzero). Each committed choice verifierargmax. Native control masks exactly reconstruct actual masks, replay old Run, source/budget hardgated.
Twelve cells/origin, <=64 total forwards each. Record ranks,next widths/acceptance, currentmath/control/oldtrace diagnostics. CPU default comparison is outputzero, but report native control per origin (zero vs inputfirst), retaining all positive/null/negative results without mixing differing histories.
Output next roots2/1/3/4; input2/1/3/3; q161 cross-origin positions differ. Selection bias remains; acceptance flips show only fixed-state local causality, not a full480 causal share. No prefill replacement; decode root0 is not prefill.
Commands --template-control --early-template --reciprocal, plus --input-origin for input, separate new directories, one serial command, no overlap. Historical actualGPU admission free>=14GiB/no foreignGPU Python, no CPU checks/hashes; busy/capacity defers without killing. ModelScope4B BF16/T0/eager/default only; no8B/FP32model/timing/operators. Source/control/math/nonfinite/OOM/budget failure stops, no autoretry. Bind all evidence/tool/protocol/model/source/private logits SHA; local complete assets, scalarreport PRdev; no main/force/deletion.

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-template-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_TEMPLATE_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_five_methods/docs/MASK_TEMPLATE_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_TEMPLATE_PLAN.md); original SHA256 `75ab6681098d72304b40f1935f2d4e622aad7723a7f8f01922e6b9910ea92196`.

#### Historical input-velocity template-difference control
After PR18,138 saved input velocities differed from initial template difference by at most~.095% relative error, but algebraic proximity does not prove identical candidates/acceptance. Test this without changing defaults/science/oldRun/contract.
Same outputzero prefix/cache/currenttree, BC30/two masks/nine nodes,4B BF16/T0/eager/default. All input cells update fullprompt and same committed history with lambda=.1: zero=original outputzero; input_first=native beta=.9999/gamma1; input_template=updatedmean plus fixed first-two-token embedding difference; input_mean=updatedmean and zero velocity, not frozenESP.
Readout replacement only, no altered historical state; nextblock retains original outputzero maskrule. Same tool/tree, native outputinternalEMA unchanged, no precision/reduction study.
Unconditional q121/q401/q241/q161 firstroot>=8 =>9/8/8/8,16cells<=96forwards. Conditional q121/q401 predecessor of first nextblock accepted_nodes=[1] =>3/19,8cells<=48forwards; report separately, never an overall random rate.
For each, recompute same-block masklogits, natively construct/verify nextblock; report target ranks,d1/d2width,actualnextacceptance and full two-row seed-logit equality to nativeinput. Private full logits, public scalar/SHA. Any template/native acceptance difference falsifies exact substitutability locally; differing ranks/logits with equal acceptance still count as differences. Agreement cannot generalize480. Template-vs-zero tests localfixedbias without assuming benefit or worthless content; nativecontrol mismatch to prior batch is reported and stops additions, no operator chase.
Historical admission protocol/tool/test committed before GPU; preserve zero-forward deferrals. It originally allowed CPU watchers by scriptpath+SHA and parent chain, unknown jobs rejected. watch-version-final.py only called verified CPU CIquery; watch-version-dev.py likewise. watch-publication.py PIDs53808/64588 only remotePR/CI. Initial queue-stop refused with a livechild; later brief overlap was blocked by existing-directory protection, duplicate canceled and original exited, allzero-forward, preserved. Thereafter single serial admission only.
Attempt16 PIDs34924/63696 were quantization tokenizer/dataset/CPUtensor preparation (watch_calibration.py and prepare_official_calibration.py), no model/CUDA. Attempt17 PIDs44856/55352 were RPipe watch-publication-main.py, the same read-only CIquery; this did not authorize external main merges. Original script hashes are preserved below. These CPU registrations are historical, superseded by actualGPU-only admission.
Originally at most20resource retries30s apart, onlyzero-forwarddeferred; execution failures never automatically retried or directories reused. Manual confirmation could create a new directory after zero-forward deferral.
Require allcommits verifierargmax, unchangedcurrentowner and outputzero replay. OldARdiagnostic only. Math/source/nonfinite/budget/control failure stops. No8B/timing/defaultchanges/operatorwork/data deletion. Topicpush/PRdev, no main; later explicit cleanup authorization supersedes historical branchretention.

Historical recorded SHA256 values (source ordering; identities remain in the immutable original):

- `f33abb9086f09bbf917118b20a5fc10cd3c13cca8ec0690b6d0178ef4a399763`
- `b69fd1fe0c67a405afee52be9599cd79cb06459b5abc23b17860ea9014d2ce92`
- `4f9b8c486aff8b93cc18116100e9013290e469fe8ed362639b4ebf790bf2dece`
- `dc3f6db96ca5698f19feba0a7e6a23c2f5bd7602bbf29b38cb78c03c8b29e845`
- `669a03a0e26231c9e7fbb0bd1d261c892dbe2f57da1164abd06123f94c324baf`
- `336458e7229ec727348604c9142ec1a63283d2a77f00c81f235f34b24cdb3350`

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-window-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_WINDOW_PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_five_methods/docs/MASK_WINDOW_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_WINDOW_PLAN.md); original SHA256 `25a06da4c932f42184fbe092e03938f6b123f45f58193a56266a7c11d9f25bf7`.

#### Historical4B fixed-window output-mask controls
After PR17 full-depth/position decomposition and12 read-only geometrycells, add CPUvector decomposition and limited intervention; no scientific source/default/oldcontract/Run changes,8B,FP32model,timing or operatorchain.
CPU: bind geometry.json,vectors.pt and oldRunSHA. Decompose sameowner rawslotdifference into radial/tangent components, measure movement retained by normmapping to test “normalization removes most velocity”; vectorgeometry only. Across480 atBC30/60/groupthree methods by previousaccepteddepth, report nextd1rate/nexttree seedownerdepth: conditional, not crosspathcausal.
Slotinput is owner+1/+2, masklogits predict owner+2/+3. Nexttree root is deepestacceptedowner's bonus. Currentmasks cannot affect currentowner via fixed attention; do not count a current-mask intervention as currentacceptance gain.
GPU: q121/q401/q241/q161 from original outputzero path, firstroot>=8, reconstruct<=32newtokens, outcomeindependent. Fourcontexts*sixcells=24,<=128totalforwards, no fulltrajectory. Maintain sameprefix/cache/currenttree/verifiedhidden/proxy and read-only firstshadow(beta.9/proxybeta.99), no predictionsinto realhistory.
Sixcells for j=0,1: zero=originalorderzero anchor/originalr-normalization; input_first=samehistory input beta.9999/lambda.1/gamma1 with no outputscaling; first=a+jv/originalperslotradius; shifted=a+(j+1)v/originalradius; radial=a+jv dividedby sameowneroriginalanchornorm; both=a+(j+1)v dividedby sameanchornorm. Displacementcoefficient1 fixed, nosearch.
Native shadow.masks_per_owner pureprojection givesa/v for sameproxy: depth0/1 proxyanchor, depth2 a+=v. Only localslotformula/mappingchanges, originalBF16/T0/eager/default/BC30twomasks/ninenodes.
Firstforward sameblock/cache: currentroot/acceptedownerargmax matchescontrol; controlmatchesoldRun. From deepestowner two maskrows, nativebuild_tree; report oldoutputzero target's rawrank,postpruningbudgetcoverage,d1/d2width. Target is diagnostic, never teacherforce.
Nextforward uses thattree/sameprefixcache andsameorigin futuremaskrule; read nativeownerargmaxpath. Report acceptance,mathgate,control/oldtrace agreement. Two localforwards/cell, no continuation/newAR.
HistoricalCPUallowlist included quantizationbuild_v3(noDockerGPU/deviceargs) andRPipewatch-first/secondCIqueries/children; scriptSHAchecked,unknownjobsrejected. Never ran/modifiedexternaltools. SupersededbycurrentactualGPU-onlyadmission.
Historicaladmission: verifiedModelScope4B,twopreloadchecks/contenthash,eachcontextcheck; unknownGPU/modeljoborfree<18GiB deferred, no preemption/kill/globalatomiclockclaim.
Model/source/input/control/currentargmax/nonfinite/OOM or>128forwards stops withpreservedevidence,noprecisionchange/operatortrace. Ifnotadmitted,CPUresultsandgeometrygapsonly,noinventedmeasurements.
Publicscalarreport docs/research, completeweights/cache/logits/tokenslocal. Preserveallnegative/null. Fixedwindowscannotestablishfull480tau. Bindtool/protocol/model/source/rawSHA;topicPRdev,no main/force/deletion.

Historical recorded SHA256 values (source ordering; identities remain in the immutable original):

- `bc206af76763f1364fe2565be1be8cf2294342aaa414f3b263cbf115dafcceec`

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/qwen3_4b_bf16_five_methods/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/PLAN.md); original SHA256 `e0cf97834d9e095585fc941ac7dd76a6ca9b7712bde1cc0c6440b750cb617816`.

#### Qwen3-4B BF16 paired five-method comparison

Historical pre-make status: owner mapping/BF16 norm validation/CPU regressions and the paired reader were ready; execution status belongs in STUDY_REPORT rather than rewriting the frozen plan. ESP Table1 four full480 cells, argmax and native process had finished; PR10(fafed614) merged todev, targets missed and negative results retained. No8B or operator work in this Study. Test output order0/order1 against all five existing methods and a single independentAR on the same480 first turns; neither old4B FP32 nor26-chain results substitute.

PredeclareBC30 andBC60, both two-mask naive dynamic/pruned trees with node budgets9/19, preserving explicit second-mask velocity extrapolation. This is the existing4B definition, not8B's single-mask Table1 protocol. All methods share tree/budget/length/seed; output methods cannot receive a narrower chain. Eleven Runs: one shared480AR, then five methods×two budgets. Disable repeated internalAR in Spec Runs; retain all output IDs and first differences in the paired reader. No teacher,warmup or speed benchmark.

Methods: ESP frozen prompt mean(update=false), explicitly distinct from paper updating mean; input zero uses prompt-mean embeddingEMAλ=.1; input first uses mean_plus_gamma,level=order0,λ=.1,β=.9999,γ=1,normalize=false,extrapolate_ema=false; output zero uses final-normalized hidden(order0) scaled to r=mean_v norm(W_v), retaining proxy-state semantics; output first uses the same mapping(order1),accepted-historyβ=.9 and proxy-historyβ′=.99 without post-hoc tuning. IndependentAR shares model/input/seed/stops and runs once for all comparisons.

Reuse ModelScope Qwen/Qwen3-4B and freeze fileSHA,verifiable source,model digest and actual tied-storage pointers. Never substitute8B or unverified weights. Weights/forward/final masks areBF16,eager,default reduction,deterministic=false. Native outputEMA/norm mapping retainFP32 states; inputEMA retains native dtype. Do not claim every internal operator isBF16 or introduce aFP32 model/control. Use original SpecBench480/order,skip0/takeall,first-turn non-thinking template,max_prompt_len40960,without filtering/truncation. Generate4B tokenizer manifest and verify qid/category/source/order instead of blindly copying8B IDs. All Runs useT0,seed123+index,max_new_tokens256,EOS,warmup=false.

Before make: preserve originalk+1 output depth-proxy states and map[0]+tree.depths to actual root/candidate owners; siblings at equal depth share the existing prediction. Only real root/accepted-path hidden commits into verified history after forward; bonus is deferred and rejected branches never persist. No extra forwards/tree changes. Native owner-count checks use actual tree size; BF16 norms use dtype quantization tolerance, with nonfinite/zero/tamper rejection and unchanged argmax gate. Regress multi-sibling/multidepth,dynamic pruning,accepted nonfirst siblings,EOS/budget clipping,proxy contamination,old chain behavior and realBF16 CLI; inspect callers/evidence/docs. Freeze sources/model/data/untimed adapters before make; changed inputs/source require a new version/Run, never re-sign success.

Pair byqid/input/model/seed. Report preclip accepted,committed/new_tokens,prefill+decode calls,pooled/promptτ,AR exact and first difference. Every pair among five methods has separate accepted-higher and calls-lower win/tie/loss, including frozen/updating baselines and output1−output0; do not collapse metrics or conceal EOS/output-length differences. Retain both budgets'positive/negative/tied results; one seed proves neither universal advantage nor speed. The hard gate covers every actual committed token, including prefill,against its correct same-forward argmax. AR exactness is diagnostic and cannot trigger operator analysis. Source/config/count/cache/owner/norm/argmax failure,OOM or nonfinite values retain the failed cell; noFP32 switch,teacher correction,deleted counterexamples or partial-as-full claims.

Use one24GB GPU and round1/num-gpus1 chain:sharedAR,BC30 five methods,BC60 five methods,confirm exit between groups. Do not kill others. No warmup/decoding clocks/latency/throughput; timing fields are zero sentinels. Reports retain Run status,actual dtype/settings/source/inputSHA,pairwise results,AR first differences,argmax coverage/failure bounds. Raw Runs/tokens/weights/cache stay local; normal commit/push/PR→dev authorized,not main/force/data-or-branch deletion/spending.

Preflight:14 model files total8060927256 bytes matched ModelScope SHA and prior4B FP32 content; immutable download revision remained unconfirmed despite successful offline snapshot resolution and retained warning. All480 tokenized qids/categories/order/IDs matched the Table1 inputs,max1500,no truncation. Configqwen3/hidden2560/layers36/vocab151936/tied=true was recorded, but actual GPU storage sharing still required each Run's adapter. Completed tinyBF16 six-Run nativeCPU checks,wholeCPU compatibility,mypy56files,nine new protocol/recovery/loader/phase regressions and native expansion of11configs; this was not a real4B result claim. Reader reuses verify_evidence/source/Study/input gates and checks tied/dtype/model/runtime on each Run; retain all20 budget/pair groups. Emit immutable paired_evidence_v1.json only after11/11 and native process success; never fill missing/pending with zero. Tests cover wins/losses/ties,call direction,missing/misaligned identities,AR-length divergence,pending completeness and accepted evidence format.

</details>

<a id="legacy-studies-smoke-docs-plan"></a>

### studies/smoke/docs/PLAN.md

<details>
<summary>Historical plan (consolidated)</summary>

Original: [studies/smoke/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/smoke/docs/PLAN.md); original SHA256 `9b61ffa71abcea1e1d923d93639652a35bb9e20b6a38c62bbd5c01c1e7666560`.

#### Historical random-tiny smoke and migration validation

Pin RPipev0.3.0/804b3823ab6d2af46cbda1a65688ca8b76c60b22,retaining ESP/EMA/AR/alignment training-free semantics and actual six-phase evidence gates. Existing STUDY_REPORT is history,not a new signature or permission for a real-model matrix. Base experiment_config→fixed→one algorithm.config.variant axis[esp_static,esp_dynamic,ema_velocity],three Experiments×seed123,three Runs,no Experiment folders; current index defines finalisation. Historical version rpipe-v030-20261013-delivery is not reused: new tests copy into isolated.tmp,rename Study/version and verify expanded config/index before make/launch,never append to old Runs.

Random tinyLLaMA:vocab256,hidden64,intermediate128,layers2,heads4,KVheads2,positions1024. Four synthetic32-token prompts,data seed0,model/Run seed123. CPU/eager/FP32 with actual TF32/matmul/dependencies recorded. Retain original three variant mask/tree/EMA definitions,max32,T0,noEOS stop,AR and per-prompt evidence. Historical hard gate exact_match_rate=1 plus full tokens/acceptance/fallback/KV/input/model/runtime provenance. Old accept_d1 counts27/31/8 differ from all-depth accepted28/31/8. They are specific to recorded device/model/input/dependencies: compare full tokens/paths on the same device,establish new references after environment changes,and do not tune temperature,cast real models toFP32 or select repeats to erase counterexamples. τ/speed are not smoke gates or rankings.

Prepare uses recipe/builders for source/input/model; execute records observations; collect metrics;summarize rejects exact/evidence failures;write rechecks before RPipe finalisation;Run process stays default whileStudy process verifies current factors/seeds/source/quality. Test source add/delete/edit,input/config drift,missing raw,real token mismatch,incomplete subsets/seeds,mixed sources and Study scope. Installation validation covers actualwheel/sdist/console scripts/CLI,not merely import/exit. Use specialized locks plus native provenance. Three CPU Runs are serial,round1,existing dependencies,OMP_NUM_THREADS=1,without overlapping timing/resource competition. Native make/launch/process/status/logs/report use new IDs; retain failures. Fill estimates after make and actuals after execution; no unified historical wall-time exists to reconstruct. Separately authorized CUDA is only random-tiny three-variant serial validation,with actual dtype/TF32/attention/input; no uncalibrated concurrency or KV resume claim.

Success requires every factor/seed Run success,exact1,complete/source/quality agreement. Report three Experiments,n=1 limits,real historical figures and separate static/CPU/GPU/install coverage. Public selectedJSON/figures/reports differ from local rawRuns/shared/tmp. Preserve the2026-10-13 migration record: compare the555-case c3263c0 pre-edit fingerprint on unchanged device/dependencies; export oldb1f6191 src and use validate_migration.py for three variants×T{0,.7,1}×four prompts. Passive Python line tracing observes full tokens/acceptance/fallback/cache without modifying old code; T0 accept_d1 reference27/31/8. New rpipe-v030-20261013 versions (later final03/delivery) run the real phase/AR/alignment/failure CLI. Install wheel/sdist outside source into separate.tmp using existing scientific dependencies,and check import/entrypoints/ORIGIN/LICENSE/tiny chain,including Windows double-import entry. No downloads,real-model/restore validation or speed ranking; failures/counterexamples stay unrelaxed and local logs are not supplied by clone.

</details>

<a id="norm-continuation-20261016"></a>
### 2026-10-16 norm first-cell recovery and seven-cell continuation

The full480 execute of template/0.5, Run `6ee3f749ee20c450`, completed. Collection failed because the Study package did not export run. The same native collector and existing revalidate_run tool recovered acceptance: all80,940 committed tokens passed verifier argmax, all raw SHAs stayed unchanged, and execute was not repeated. The failed result and recovery evidence remain under `.tmp/norm-collect-recovery-20261016/`; no AR gate was relaxed.

Keep the frozen v2-capacity entrypoints unchanged. A separate continuation Study/new version contains only the remaining seven cells in their original order. Fixed SpecBench480/prompts/token IDs,4B BF16/T0, BC30/two masks, seed123, scales and science are identical, reusing NormEval. Only phase exports and the19:53 display-process admission classification change. CPU aggregation binds the recovered first cell and each new cell to their own native sources/gates; it never marks the old seven unexecuted Runs successful. Zero/unit-scale references are reused. Four phase-package callable regression checks precede execution.

8B BC60 `a8e6bba1dd1d80e2` has naturally passed its original RAM gate and is running with priority; do not pause or duplicate it. Continue seven cells serially after it exits, retaining measured host peak+2GiB and actual free VRAM gates. Display/remote-desktop contexts do not block and no processes are killed. Report three tau aggregations, paired wins/ties/losses, source paths and the operational recovery at the complete milestone PR.


Continuation readiness: native make created seven Runs, first d288934359a7ede1. One waiting queue60480 will not load before BC60 exits; read-only ledger78720 accepted the recovered first cell and retains the previous failed watcher history separately. Native dispatcher and positive/negative mapping checks, Ruff and mypy pass; the full suite found two tool-documentation failures; after fixing them,89 targeted checks pass. First-cell three tau aggregations and paired counts: [continuation report](../studies/qwen3_4b_bf16_bias_norm_continuation/docs/STUDY_REPORT.md). Branch review found only main/dev plus the merged branch still carrying active BC60 and the unmerged current branch; all retained, no additional deletions.

V4 operational update: documentation-only helper SHA change; superseded unexecuted seven IDs retained with metadata backup. Current first Run `6719b3ae2c089133`, serial waiting queue `64880`, read-only ledger `50944`. Earlier60480/78720 CPU processes exited before replacement; BC60 was never interrupted. Native make/source bindings renewed only for the new version, not old completed evidence.


<a id="k-affine-residual-20261020"></a>
## K affine residual correction: postcommit CPU stage (2026-10-20)

### 1. Question and status

The user authorised the independent `studies/k_affine_residual` method Study, not another pending decision. Test `z'=z+A(e(y)-e(m))+b`, where y is the last committed token. Model dependence and data independence remain hypotheses. This is new research, not old debt repayment; shared decoding algorithms and old Studies remain unchanged.

### 2. Prior evidence

Reuse the final-normalized hidden definition in the existing latent/output-space plan and `output_masks.py`/`mask_providers.py`. The local task `batch9-collaborator-note.json` distinguishes supplied formula, primary reading and inference: Kalman innovation is only a structural analogy, not an equivalence; no particular spectrum paper or flow-matching reproduction is presumed. No new retrieval/download is needed for this stage.

### 3. Hypotheses and six connected aspects

Can K fitted on documents from one dataset reduce same-position latent MSE on separate documents and another dataset? Prior hidden/residual definitions motivate the mechanism; cached real texts provide inputs; a small random tied model supplies genuine latent pairs; fixed ridge is the algorithm; none and independently fitted bias-only are controls; paired document MSE with frozen cross-dataset application is the experiment. A falsifiable prediction is lower full-K MSE than both controls on holdout and cross_dataset. Negative results are retained without tuning or sample replacement. One random model/seed cannot establish model dependence or general data independence.

### 4. Data and leakage

Read only the existing WikiText2 test Arrow cache under `D:/GitHub/DreamSoul/dayan/.tmp/language-inputs-20261015-23/data/WikiText2/test/` and repository `datasets/spec_bench/question.jsonl`. Split WikiText by top-level article, not neighbouring rows: first 12 unique articles with at least32tokens, first8 fit and next4 holdout. First8 SpecBench first turns form cross_dataset. No special tokens/chat template; lengths4/6/8/10/12/14/16/20 yield64/32/64pairs. Original cache name “test” does not make our fit partition an independent benchmark. Its local metadata license is empty; preserve it and do not redistribute corpus text. Record entire-document hashes, selection indices, token IDs and tokenizer/asset hashes in `docs/data_manifest.json` before model outputs. Cross-split whole-document and exact-prefix intersections must be zero. Never select inputs or tune using evaluated outputs.

### 5. Model

Current official Transformers tiny LLaMA builder, seed123, vocab151936, hidden32, intermediate64, 2layers, 4heads/2KVheads, positions128, tied embeddings, CPU/float32/eager. Tokenizer uses the local Qwen3-0.6B cache with `local_files_only=True`; its weights are neither read nor loaded. Run measures unique/stored parameters, actual tied pointers, state hashes and providers. This random model is not the pretrained0.6B model and supports no quality/performance claim.

### 6. Algorithm and causal definition

Postcommit replacement uses teacher-forced committed prefix P,y of length t; P has t−1 tokens and y is already committed input, never a future label. Set m=mean(E[P]), e(m)=m. At the same position t−1 obtain z(m) from P+soft(m) and target z(y) from P+E[y], using the official base model's final normalized hidden. Do not read tokens after t. This is not a future speculative-slot or real hit-rate reproduction.

Set x=E[y]−m and r=z(y)−z(m). Fit centered float64 ridge only on fit rows with lambda=.001, `A=solve(Xc.T Xc+lambda I,Xc.T Rc).T`, unpenalized intercept `b=mean(r)−A mean(x)`. Independently fit bias-only `b0=mean(r_fit)`; it is not the full-K intercept. None adds zero. Save fullK/bias and fit IDs/hash, freeze their bytes before extracting/evaluating holdout/cross; never refit. Model weights stay unchanged; only this small linear calibration is solved. Implementation stays Study-local, without shared algorithm changes or adapters.

### 7. Metrics and validity

Primary: per-row hidden MSE over32coordinates, average8positions within each document, then weight documents equally within each split. Report paired document full−none and full−bias differences. Auxiliary hidden cosine and all-vocab LM-head target-argmax agreement/KL(target||arm) are mechanism diagnostics, not label accuracy, tau, accepted tokens or quality. Validity requires160complete pairs, disjoint splits, correct positions, finite values, one frozen K, unchanged selected inputs/source/model. Improvement is not a success gate. Save all x/zmask/ztrue/prediction arrays, mappings,K/bias and per-row diagnostics; complete vocabulary logits are computed but not all persisted. No population significance or clone-reconstruction claim for uncommitted large local assets.

### 8. Execution, ownership and budget

Commit this plan and input/config registration first; then unit/contract tests and one native Study Experiment/seed through make→launch→process/status/report. All three arms share the Run and frozen fit. Prepare verifies immutable plan blob/assets; execute extracts actual pairs and freezes fit; collect recomputes raw metrics; summarize checks validity; write rechecks; process validates only current index. Unit tests do not replace the real Run.

Codex project display name is `denoising speculative decoding`, but the actual coordination chat is DreamSoul “自动化研究 AutoResearch”, ID `01a1561f-6351-7d70-aef6-f1fab17f2259`. Execute from SD isolation `D:/360安全浏览器下载/Agent/DreamSoul/tmp/技术债偿还 20261020/sd`; original `D:/GitHub/DreamSoul/denoising-speculative-decoding` is read-only. Do not claim same-named SD chat ownership or technical permission isolation. `/root/rpipe_debt_batch3` owns this new Study and related plan/index/tests; root owns Git/review. Adopt the six-step continuity/comparison/registration/bounded-execution/verification/decision and stop handling of the unpublished Automated Research draft, not its publication status.

Estimated single Run120s, owned-tree outer deadline180s; total CPU execution budget600s including tests/debugging. OMP/MKL1, CUDA_VISIBLE_DEVICES=-1, HF/TF offline. Reuse existing Windows Job/ProcessTree, fresh outputs, selected input before/afterSHA and owned0 cleanup; only own processes are cleaned. Estimated RSS2GiB is not called enforced unless a real sampler is connected. Stop at budget/failure and retain partial evidence; no auto enlargement/restart, download,GPU,cloud cost,large training or old queue. Further work requires a new scoped registration.

### 9. Evidence and limits

Pre-execution registration; actual Run/cost/results will be reported in [STUDY_REPORT](../studies/k_affine_residual/docs/STUDY_REPORT.md). Completion requires one valid native Run/raw/frozen provenance and phase finalisation, not winning MSE. Debt repaid0. Future slots, pretrained models, multiple models/seeds and benchmark quality/performance remain untested. Data independence is not established.

### 10. Change record and references

2026-10-20 user authorisation advances K into an independent Study; this explicitly defined CPU stage does not rewrite historical plans/gates. Study declaration and registration/data manifest bind sources/inputs. Recipe will verify and record the actual committed plan/blob during native prepare; do not invent uncreated Git identities. Protocol changes require a new version/Run. Adopt relevant parts of DreamSoul Research Plan Standard, template2026-10-20 and unpublished Automated Research Standard draft.
