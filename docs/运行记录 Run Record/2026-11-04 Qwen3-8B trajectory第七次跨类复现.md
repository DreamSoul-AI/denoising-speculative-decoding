# 2026-11-04 Qwen3-8B trajectory 第七次跨类复现

协议 `office-8b-traj-xcat7-v1`，标明探索。测量时 HEAD `9867390f42e80d297ee06e28fff393c5b558d701`，dirty 为 false。预注册在 `3ee02ed`，比较代码在 `9867390`。本分支基于 `origin/dev` 的 `f665185`。命令是 `python -u -m studies.trajectory_observation.flow.execute.office_traj --dest <已有的 grokbot 目录> --only all --protocol xcat7`。整段命令约 723 秒，exit 0。日志、header、冒烟、两支 jsonl 和 `sd-office-8b-traj-xcat7-summary-20261104-045236.json` 都在该目录，文件名以 `sd-office-8b-traj-xcat7-` 开头、以 `-20261104-045236` 结尾。这些文件不进 Git。

权重 `D:\models\Qwen3-8B`，BF16，CUDA，eager，`local_files_only`。没有下载模型或数据集。Dolly 用本机已有的 `datasets` 5.0.0 缓存。torch 2.11.0+cu130，Python 3.13.9，设备 NVIDIA GeForce RTX 5090 D v2，显存 24455 MiB。开跑前空闲显存 20748 MiB，利用率 0%。compute-apps 没有报出数字形式的显存占用，加载前的空闲检查因此通过。没有另起一项 GPU 计算。

冒烟是正式 128 条的前 2 条（qid 0、1）、每条 8 token。两臂 argmax 都通过，不进入符号检验。

评测 128 条。类别 `summarization`，qid 0–127。非空条数 1188，是 `creative_writing`、`open_qa`、`general_qa`、`classification`、`closed_qa`、`brainstorming` 与 `information_extraction` 以外剩下的最后一个 Dolly 类别。已用文本按 creative_writing 全部 709 条、第八轮 open_qa qid 0–127、第九轮 general_qa 选入的 128 条、第十轮 classification qid 0–127、closed_qa 选入的 qid 0–127、brainstorming 选入的 qid 0–127、information_extraction 选入的 qid 0–127、本仓 SpecBench 各轮原文和 288 句校准模板。扫描 0–127 没有相同文本，所以没有跳过。summarization 全流与上述已用文本没有相同条目。更后面另有 1 条类内重复（263 与 85 相同）。85 已在这 128 条里，263 在 127 之后，没有再进入。SHA256 `f35b56bf668124a609c47dc51f97d40cabd75fdac5e6a3eb0fd133a25665020a`，与预注册一致。每条 64 token，T=0，`stop_on_eos` false。两臂都是 `branches` [4]、`num_masks` 1。full 的 `observation.gain` 是 0.5，traj 是 0，`condition` 都是 none，`form` 都是 mean。每一步草稿节点数都是 4。块复杂度都是 10。argmax 全部通过。

| **比较** | **τ 均值** | **τ 合计** | **胜** | **负** | **平** | **p** | **接受数差** | **95% 区间** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traj 对 full | 1.340886→1.418055 | 1.331792→1.410530 | 97 | 17 | 14 | 8.636e-15 | +2.6640625 | [2.125, 3.203125] |
| full 相对 traj | 1.418055→1.340886 | 1.410530→1.331792 | 17 | 97 | 14 | 8.636e-15 | −2.6640625 | [−3.203125, −2.125] |

p 是逐 prompt 接受数的双侧二项符号检验。第一行的胜是 traj 的 `accepted` 高于 full，差值是 traj 减去 full。第二行是同一配对的相反数，用来看完整版相对 trajectory-only 的接受增益。区间是 bootstrap 百分位，种子 0，10000 次。`direction` 作用在第一行上，返回 positive。两行的区间都不含 0。τ 差（traj 减去 full）均值 +0.077169，区间 [0.062001, 0.092411]，不含 0。独立 AR 不一致：full 57，traj 52。

这一轮没有测速，也没有把 T0 的 argmax 通过写成采样分布保证。gain 0.5 的联合 mask 在这 128 条 summarization 上接受更少，接受数差均值是 +2.6640625 个 token。这是探索复现，不是确认。Dolly 八个类别的办公室切片到此用完，不再另起新类。
