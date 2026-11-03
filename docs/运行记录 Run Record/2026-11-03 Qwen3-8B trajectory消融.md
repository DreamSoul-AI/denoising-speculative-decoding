# 2026-11-03 Qwen3-8B trajectory 消融

协议 `office-8b-traj-ablate-v1`，标明探索。测量时 HEAD `b41ffa6c7dc83e1d9487cc8b964d27834d7d906a`，dirty 为 false。预注册在 `d82df53`，比较代码在 `b41ffa6`。`origin/dev` 的合并提交是 `d589f64`。命令是 `python -u -m studies.trajectory_observation.flow.execute.office_traj --dest <已有的 grokbot 目录> --only all`。整段命令约 345 秒，exit 0。日志、header、冒烟、两支 jsonl 和 `sd-office-8b-traj-ablate-summary-20261103-0905.json` 都在该目录，文件名以 `sd-office-8b-traj-ablate-` 开头、以 `-20261103-0905` 结尾。这些文件不进 Git。

权重 `D:\models\Qwen3-8B`，BF16，CUDA，eager，`local_files_only`。没有下载模型或数据集。Dolly 用本机已有的 `datasets` 5.0.0 缓存。torch 2.11.0+cu130，Python 3.13.9，设备 NVIDIA GeForce RTX 5090 D v2，显存 24455 MiB。开跑前空闲显存 21147 MiB，利用率 9%。compute-apps 没有报出数字形式的显存占用，加载前的空闲检查因此通过。没有另起一项 GPU 计算。

冒烟是正式 60 条的前 2 条（qid 647、648）、每条 8 token。两臂 argmax 都通过，不进入符号检验。

评测 60 条。qid 647–657、659–677、679–708。跳过 658（与 408 相同）、678（与 198 相同）。非空创作流在 qid 708 结束，从 647 起凑不满 128 条未用文本，没有回用 0–646。SHA256 `d23ad590c45307fa7c1981b272adaec243bfaca5a2e390eda394bf2fe42a4cca`，与预注册一致。每条 64 token，T=0，`stop_on_eos` false。两臂都是 `branches` [4]、`num_masks` 1。full 的 `observation.gain` 是 0.5，traj 是 0，`condition` 都是 none，`form` 都是 mean。每一步草稿节点数都是 4。块复杂度都是 10。argmax 全部通过。

| **比较** | **τ 均值** | **τ 合计** | **胜** | **负** | **平** | **p** | **接受数差** | **95% 区间** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traj 对 full | 1.267772→1.361124 | 1.258322→1.352899 | 47 | 5 | 8 | 1.285e-09 | +3.616667 | [2.733333, 4.550000] |
| full 相对 traj | 1.361124→1.267772 | 1.352899→1.258322 | 5 | 47 | 8 | 1.285e-09 | −3.616667 | [−4.550000, −2.733333] |

p 是逐 prompt 接受数的双侧二项符号检验。第一行的胜是 traj 的 `accepted` 高于 full，差值是 traj 减去 full。第二行是同一配对的相反数，用来看完整版相对 trajectory-only 的接受增益。区间是 bootstrap 百分位，种子 0，10000 次。`direction` 作用在第一行上，返回 positive。两行的区间都不含 0。τ 差（traj 减去 full）均值 +0.093352，区间 [0.069798, 0.117705]，不含 0。独立 AR 不一致：full 34，traj 33。

这一轮没有测速，也没有把 T0 的 argmax 通过写成采样分布保证。gain 0.5 的联合 mask 在这 60 条上接受更少。这是探索，不是确认。
