# 2026-11-04 Qwen3-8B trajectory 墙钟

协议 `office-8b-traj-wall-v1`，标明探索。句子协议仍是 `office-8b-traj-xcat7-v1`。测量时 HEAD `beaaeab3b5ca558109c5315122ada91c141ace64`，dirty 为 false。逐臂 `wall_seconds` 在 `f38d10e`（PR #83）。本提交只把 xcat7 的产物前缀和戳改成 `sd-office-8b-traj-wall-v1` / `20261104-0758`，避免续跑把已归档、没有墙钟的接受数 jsonl 当成已完成，并覆盖 `sd-office-8b-traj-xcat7-summary-20261104-045236.json`。已归档的 xcat7 接受数文件仍在原名，摘要里没有 `wall_protocol`。本分支从 `origin/dev` 的 `f38d10e` 起。

冒烟命令是 `python -u -m studies.trajectory_observation.structure.office_traj --dest <已有的 grokbot 目录> --only smoke --protocol xcat7`，exit 0。正式命令把 `--only` 换成 `formal`，716.4 秒，exit 0。日志、header、冒烟、两支 jsonl 和 `sd-office-8b-traj-wall-v1-summary-20261104-0758.json` 都在 `D:\360安全浏览器下载\Agent\DreamSoul\tmp\grokbot`，文件名以 `sd-office-8b-traj-wall-v1-` 开头、以 `-20261104-0758` 结尾。这些文件不进 Git。

权重 `D:\models\Qwen3-8B`，BF16，CUDA，eager，`local_files_only`。没有下载模型或数据集。Dolly 用本机已有的 `datasets` 缓存。torch 2.11.0+cu130，Python 3.13.9，设备 NVIDIA GeForce RTX 5090 D v2，显存 24455 MiB。正式加载前空闲显存 21267 MiB，利用率 0%。compute-apps 没有报出数字形式的显存占用，加载前的空闲检查因此通过。没有另起一项 GPU 计算。

冒烟是正式 128 条的前 2 条（qid 0、1）、每条 8 token。两臂 argmax 都通过，两臂 `wall_seconds` 都是有限非负数。冒烟不进入符号检验。正式开始前用第 1 条 prompt、8 token、两臂各做一次不计时预热，预热后同步。预热没有写入正式 jsonl。

评测 128 条。类别 `summarization`，qid 0–127，没有跳过。SHA256 `f35b56bf668124a609c47dc51f97d40cabd75fdac5e6a3eb0fd133a25665020a`，与预注册一致。每条 64 token，T=0，`stop_on_eos` false。两臂都是 `method` esp、`num_masks` 1、`impl` efficient、`tree.mode` static、`branches` [4]、剪枝开。full 的 `observation.gain` 是 0.5，traj 是 0，`condition` 都是 none，`form` 都是 mean。协议名都是 `trajectory-observation-blend-v1`。每一步草稿节点数都是 4。块复杂度都是 10。argmax 全部通过。128 条两臂的 `wall_seconds` 都完整，墙钟汇总不是不完整。

| **比较** | **口径** | **胜** | **负** | **平** | **p** | **差均值** | **95% 区间** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| traj 对 full 的接受数 | `accepted_traj - accepted_full` | 97 | 17 | 14 | 8.636e-15 | +2.6640625 | [2.125, 3.203125] |
| traj 对 full 的墙钟 | `wall_seconds_traj - wall_seconds_full` | 99 | 29 | 0 | 3.842e-10 | −0.100136 秒 | [−0.130013, −0.069424] |

接受数的胜是 traj 的 `accepted` 更高。墙钟的胜是 traj 的 `wall_seconds` 更短。p 都是逐 prompt 的双侧二项符号检验。区间是 bootstrap 百分位，种子 0，10000 次。接受数的 `direction` 为 positive，区间不含 0，与已归档 xcat7 的 97:17:14、差均值 +2.6640625 相同。`wall_direction` 为 shorter：差均值小于 0，且更短的条数多于更长的条数。墙钟区间也不含 0。τ 均值 1.340886→1.418055，τ 合计 1.331792→1.410530。τ 差均值 +0.077169，区间 [0.062001, 0.092411]，不含 0。独立 AR 不一致：full 57，traj 52，只作诊断。

伴随描述：`prefill_time` 均值 full 0.046831 秒、traj 0.046489 秒；`decode_time` 均值 full 1.670983 秒、traj 1.571224 秒。这两段在 `generate` 里面读取，两端没有再同步。同向判断只用 `wall_seconds`。`wall_seconds` 均值 full 1.718967 秒、traj 1.618832 秒。

接受数的 `direction` 为 positive，并且 `wall_direction` 为 shorter。两个区间都不含 0。这 128 条上接受数与墙钟同向，并且一起离开了 0。这是探索，不记成确认。不改 K，不改 0.5，不改 64，不重选这 128 条，也不把树改成 [2, 2] 或 [3, 1]。没有做 `office-8b-traj-failset-v1`。
