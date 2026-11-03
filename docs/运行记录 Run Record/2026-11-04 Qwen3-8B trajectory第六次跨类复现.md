# 2026-11-04 Qwen3-8B trajectory 第六次跨类复现

协议 `office-8b-traj-xcat6-v1`，标明探索。测量时 HEAD `ace2ac6a03672f34ce2af669f14afa825411b1f4`，dirty 为 false。预注册在 `d6a2ed9`，比较代码在 `ace2ac6`。本分支基于 `origin/dev` 的 `fab3a95`。命令是 `python -u -m studies.trajectory_observation.flow.execute.office_traj --dest <已有的 grokbot 目录> --only all --protocol xcat6`。整段命令约 703 秒，exit 0。日志、header、冒烟、两支 jsonl 和 `sd-office-8b-traj-xcat6-summary-20261104-042546.json` 都在该目录，文件名以 `sd-office-8b-traj-xcat6-` 开头、以 `-20261104-042546` 结尾。这些文件不进 Git。

权重 `D:\models\Qwen3-8B`，BF16，CUDA，eager，`local_files_only`。没有下载模型或数据集。Dolly 用本机已有的 `datasets` 5.0.0 缓存。torch 2.11.0+cu130，Python 3.13.9，设备 NVIDIA GeForce RTX 5090 D v2，显存 24455 MiB。开跑前空闲显存 20762 MiB，利用率 0%。compute-apps 没有报出数字形式的显存占用，加载前的空闲检查因此通过。没有另起一项 GPU 计算。

冒烟是正式 128 条的前 2 条（qid 0、1）、每条 8 token。两臂 argmax 都通过，不进入符号检验。

评测 128 条。类别 `information_extraction`，qid 0–127。非空条数 1506，是 `creative_writing`、`open_qa`、`general_qa`、`classification`、`closed_qa` 与 `brainstorming` 以外、且没有作为办公室切片用过的最大类别。其余非空条数是 summarization 1188。已用文本按 creative_writing 全部 709 条、第八轮 open_qa qid 0–127、第九轮 general_qa 选入的 128 条、第十轮 classification qid 0–127、closed_qa 选入的 qid 0–127、brainstorming 选入的 qid 0–127、本仓 SpecBench 各轮原文和 288 句校准模板。扫描 0–127 没有相同文本，所以没有跳过。information_extraction 全流与上述已用文本没有相同条目。更后面另有 3 条类内重复（416 与 283 相同，840 与 768 相同，1067 与 283 相同），第一次出现都在 127 之后，没有进入这 128 条。SHA256 `b7094d9ecf62a73e299d742016a3fdfcf4608ef5a7a32f7adc7703b86706d7c3`，与预注册一致。每条 64 token，T=0，`stop_on_eos` false。两臂都是 `branches` [4]、`num_masks` 1。full 的 `observation.gain` 是 0.5，traj 是 0，`condition` 都是 none，`form` 都是 mean。每一步草稿节点数都是 4。块复杂度都是 10。argmax 全部通过。

| **比较** | **τ 均值** | **τ 合计** | **胜** | **负** | **平** | **p** | **接受数差** | **95% 区间** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traj 对 full | 1.326882→1.415774 | 1.319804→1.410037 | 106 | 10 | 12 | 2.164e-21 | +3.046875 | [2.5234375, 3.6015625] |
| full 相对 traj | 1.415774→1.326882 | 1.410037→1.319804 | 10 | 106 | 12 | 2.164e-21 | −3.046875 | [−3.6015625, −2.5234375] |

p 是逐 prompt 接受数的双侧二项符号检验。第一行的胜是 traj 的 `accepted` 高于 full，差值是 traj 减去 full。第二行是同一配对的相反数，用来看完整版相对 trajectory-only 的接受增益。区间是 bootstrap 百分位，种子 0，10000 次。`direction` 作用在第一行上，返回 positive。两行的区间都不含 0。τ 差（traj 减去 full）均值 +0.088892，区间 [0.074741, 0.103290]，不含 0。独立 AR 不一致：full 59，traj 55。

这一轮没有测速，也没有把 T0 的 argmax 通过写成采样分布保证。gain 0.5 的联合 mask 在这 128 条 information_extraction 上接受更少，接受数差均值是 +3.046875 个 token。这是探索复现，不是确认。
