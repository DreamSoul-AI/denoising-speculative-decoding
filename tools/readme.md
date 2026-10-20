# tools

Local utilities. Model-window diagnostics use explicitly prepared local weights; no automatic downloads.

| **script** | **purpose** |
| --- | --- |
| `diagnose_q178_operators.py` | Fixed post-entry provider slot precision and norm backend controls with canonical states; CUDA profiler names are localization evidence, full AR/path gates remain required. |
| `ci_local.py` | Replay `.github/workflows/cpu-checks.yml` in a fresh worktree and Python 3.12 venv. |
| `check_rpipe_pin.py` | Verify the fixed RPipe source and explicit local R1 patches against recorded hashes and preserved original blobs. |
| `decode_fingerprint.py` | Capture tokens and decode stats for every shipped preset; `--compare` proves a change is output-identical. |
| `profile_cpu.py` | cProfile of `esp_mtp.evaluate` on a tiny model. tau / speed are not paper numbers. |
| `diagnose_output_windows.py` | Read-only q173 matched-window replay around the unchanged decoder; requires identical tokens/traces, records previous seed masks/logits and exact hidden consistency. |
| `diagnose_q178_numerics.py` | Process-only strict FP32 repeats and AR-prefix canonical committed-state controls, with actual KV-root checks and two self-value controls. |
| `diagnose_study_windows.py` | Manifest-bound four-prompt runner reusing frozen window/observer/transfer/slot helpers; intended for the q179–q182 follow-up. |
| `diagnose_extension.py` | Fixed q175–q178 selection and transfer replications; reuse unchanged observer/transfer/compare helpers, allow measured path/state differences and preserve planned fallback controls. |
| `diagnose_cross_prompt.py` | Fixed q171/q172/q174 replication: reuse unchanged observer/transfer helpers, both directions and original-value placebo; retain nulls, full traces and matched committed-state differences. |
| `diagnose_proxy_transfer.py` | Three fixed q173 single-slot transfers around the unchanged decoder, with original-value placebo, actual post-intervention owner selection and lossless-output gates. |
| `diagnose_mask_scale.py` | Four fixed root83 prefix forwards cross two actual seed-mask directions with two observed norms; no generated tokens or tuned default. |
| `package_selfcheck.py` | Build wheel/sdist, install into separate targets, validate both module/console entries and native + SpecMTP CLI chains; no dependency replacement. |
| `windows_selfcheck.py` | Windows native import order (`python -m rpipe` vs `python -m spec_mtp`). |

```text
python tools/<name>.py --help
```

Package check: `python tools/package_selfcheck.py --workspace .tmp/package-<new-id>`. The directory must be new. It reuses installed scientific dependencies, so it checks distribution isolation rather than a fresh dependency environment. Windows check: `python tools/windows_selfcheck.py --smoke --out .tmp/<new-id>/windows.json`; each smoke uses a fresh version and preserves its raw logs.

`diagnose_q178_norm_kernel.py --out .tmp/<new-id>`：固定q178 CUDA norm图对照，保存实际kernel关联、entry/post和numerical-effect flags。
`diagnose_q178_qproj.py --out .tmp/<new-id>`：固定85块layer0 row4 packing／FP64对照，4克隆cache块精确重放。

`diagnose_q178_phi_rms.py --out .tmp/<new-id>`：固定entry Phi和post-attention RMSNorm精度对照，区分有效／bit等价。
`diagnose_q241_canonical.py --out .tmp/<new-id>`：首个跨任务阳性的独立canonical合同，保留自值变化及等计数不同路径。

`diagnose_q178_rounding_factors.py`：`python tools/diagnose_q178_rounding_factors.py --out .tmp/<new-id>`，原输入Phi三因素/RMS五因素算术矩阵，零模型forward，绑定缓存gamma和原值。
`diagnose_q241_state_factors.py`：`python tools/diagnose_q241_state_factors.py --out .tmp/<new-id>`，q241两native审计+四单坐标同步生成；固定协议、AR及原值审计失败即停止。

`diagnose_q241_local_factors.py`：`python tools/diagnose_q241_local_factors.py --out .tmp/<new-id>`，两原值完整生成+24固定块，定位候选构建与当前映射的时序。
`diagnose_state_extension.py`：`python tools/diagnose_state_extension.py --out .tmp/<new-id>`，两prompt两greedy seed、36生成／四teacher，复用原helpers，保留零交互和非加性反例。

`diagnose_sampling_paths.py`：`python tools/diagnose_sampling_paths.py --out .tmp/<new-id>`，两T=.8 seed各独立AR+四格/观测，冻结实际窗口，旧source719条件multinomial接受检查。
`diagnose_q241_owner1.py`：`python tools/diagnose_q241_owner1.py --out .tmp/<new-id>`，两canonical原值控制、owner1四格三重复、独立上下文条件draw；保留不恢复命中的负结果。

`diagnose_sampling_expansion.py`：`python tools/diagnose_sampling_expansion.py --out .tmp/<new-id>`，10新单位补完整4prompt×3seed，复用旧q241两单位不重跑。
`diagnose_rare_events.py`：`python tools/diagnose_rare_events.py --out .tmp/<new-id>`，两温度各4096native verifier draw，保留tiny质量与检测上界，不把零命中当无效概率。

`diagnose_sampling_continuation.py`：`python tools/diagnose_sampling_continuation.py --out .tmp/<new-id>`，固定六单位含q245/246新输入，原配置/源/AR/单槽scope硬门。
`diagnose_rare_stress.py`：`python tools/diagnose_rare_stress.py --out .tmp/<new-id>`，原温度各16384draw与单独T=2的131072draw，首命中seed／检测力／区间记录，零模型forward；不得用改变温度命中外推原条件。
复核：`python studies/output_space_cross_task/analyze_continuation.py --expansion .tmp/sampling-continuation-20261015-v1 --rare .tmp/rare-stress-20261015-v1`；固定二阶合法性：`python studies/output_space_cross_task/analyze_sampled_depth.py`。重跑正式复核只验证相同文件，不新增生成/draw或覆盖旧formal证据。
