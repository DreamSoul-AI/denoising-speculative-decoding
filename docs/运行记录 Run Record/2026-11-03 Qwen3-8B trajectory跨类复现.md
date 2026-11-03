# 2026-11-03 Qwen3-8B trajectory 跨类复现

协议 `office-8b-traj-xcat-v1`，标明探索。测量时 HEAD `35462357ebc6473aca33b766ed8ab8f218ce0987`，dirty 为 false。预注册在 `f2b2b97`，比较代码在 `3546235`。`origin/dev` 的合并提交是 `ed2bd79`。命令是 `python -u -m studies.trajectory_observation.flow.execute.office_traj --dest <已有的 grokbot 目录> --only all --protocol xcat`。整段命令约 722 秒，exit 0。日志、header、冒烟、两支 jsonl 和 `sd-office-8b-traj-xcat-summary-20261103-0929.json` 都在该目录，文件名以 `sd-office-8b-traj-xcat-` 开头、以 `-20261103-0929` 结尾。这些文件不进 Git。

权重 `D:\models\Qwen3-8B`，BF16，CUDA，eager，`local_files_only`。没有下载模型或数据集。Dolly 用本机已有的 `datasets` 5.0.0 缓存。torch 2.11.0+cu130，Python 3.13.9，设备 NVIDIA GeForce RTX 5090 D v2，显存 24455 MiB。开跑前空闲显存 21013 MiB，利用率 0%。compute-apps 没有报出数字形式的显存占用，加载前的空闲检查因此通过。没有另起一项 GPU 计算。

冒烟是正式 128 条的前 2 条（qid 0、1）、每条 8 token。两臂 argmax 都通过，不进入符号检验。

评测 128 条。类别 `open_qa`，qid 0–127。非空条数 3742，是 `creative_writing` 以外、且没有作为办公室切片用过的最大类别。其余非空条数是 general_qa 2191、classification 2136、closed_qa 1773、brainstorming 1766、information_extraction 1506、summarization 1188。已用文本按 creative_writing 全部 709 条，加上本仓 SpecBench 各轮原文和 288 句校准模板。扫描 0–127 没有相同文本，所以没有跳过。open_qa 更后面另有 5 条与 creative_writing 相同、36 条类内重复，没有进入这 128 条。SHA256 `f6e6b1c71a20f1170692e22a2c2bd79fb31eaeb810b861553f7c76a41df25231`，与预注册一致。每条 64 token，T=0，`stop_on_eos` false。两臂都是 `branches` [4]、`num_masks` 1。full 的 `observation.gain` 是 0.5，traj 是 0，`condition` 都是 none，`form` 都是 mean。每一步草稿节点数都是 4。块复杂度都是 10。argmax 全部通过。

| **比较** | **τ 均值** | **τ 合计** | **胜** | **负** | **平** | **p** | **接受数差** | **95% 区间** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traj 对 full | 1.253489→1.338024 | 1.247525→1.331132 | 103 | 12 | 13 | 3.351e-19 | +3.2421875 | [2.71875, 3.765625] |
| full 相对 traj | 1.338024→1.253489 | 1.331132→1.247525 | 12 | 103 | 13 | 3.351e-19 | −3.2421875 | [−3.765625, −2.71875] |

p 是逐 prompt 接受数的双侧二项符号检验。第一行的胜是 traj 的 `accepted` 高于 full，差值是 traj 减去 full。第二行是同一配对的相反数，用来看完整版相对 trajectory-only 的接受增益。区间是 bootstrap 百分位，种子 0，10000 次。`direction` 作用在第一行上，返回 positive。两行的区间都不含 0。τ 差（traj 减去 full）均值 +0.084534，区间 [0.071050, 0.098127]，不含 0。独立 AR 不一致：full 56，traj 54。

这一轮没有测速，也没有把 T0 的 argmax 通过写成采样分布保证。gain 0.5 的联合 mask 在这 128 条 open_qa 上接受更少，接受数差均值是 +3.2421875 个 token。这是探索复现，不是确认。
