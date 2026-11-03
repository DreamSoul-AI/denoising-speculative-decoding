# 2026-11-04 Qwen3-8B trajectory 第六次跨类复现

协议 `office-8b-traj-xcat6-v1`，标明探索。预注册在 `d6a2ed9`。选句、两臂和汇总仍在 `studies/trajectory_observation/structure/office_traj.py`，命令仍在 `flow/execute/office_traj.py`，本轮加 `--protocol xcat6`，提交 `ace2ac6`。测量时 HEAD 就是 `ace2ac6a03672f34ce2af669f14afa825411b1f4`，dirty 为 false。没有新的 legacy 条目。本分支从 `origin/dev` 的 `fab3a95` 起，测量前没有再合并。

比较保持草稿节点数 4，`branches` [4]，`num_masks` 1。两臂都是现有 ESP，mask 为 `init` mean_prompt、`lam` 0.1、`update` true，再包上已有的 `spec.observation`。full 的 `gain` 是 0.5，`condition` none，`form` mean。traj 的 `gain` 是 0，其余相同。`gain` 0 是关闭 observation 一侧的现有开关，与被包装的 ESP 逐 token 相同。K 没有拟合，也没有乘进 mask。正式句子是 Dolly `information_extraction` 非空流里跳过已用文本后的前 128 条。该类非空文本 1506 条，是 creative_writing、open_qa、general_qa、classification、closed_qa 与 brainstorming 以外、且没有作为办公室切片用过的最大类别。qid 0–127。扫描这 128 条时没有与已用文本相同的条目，所以没有跳过。SHA256 `b7094d9ecf62a73e299d742016a3fdfcf4608ef5a7a32f7adc7703b86706d7c3`。每条 64 token。冒烟是这 128 条的前 2 条（qid 0、1）、每条 8 token，argmax 通过，不进入下表。

主比较的差值是 traj 的 `accepted` 减去 full。胜是 traj 更高。完整版相对 traj 的接受增益是 full 减去 traj。`direction` 作用在 traj−full 上，这次返回 positive。

| **比较** | **τ 均值** | **胜** | **负** | **平** | **p** | **接受数差** | **95% 区间** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| traj 对 full | 1.326882→1.415774 | 106 | 10 | 12 | 2.164e-21 | +3.046875 | [2.5234375, 3.6015625] |
| full 相对 traj 的增益 | 1.415774→1.326882 | 10 | 106 | 12 | 2.164e-21 | −3.046875 | [−3.6015625, −2.5234375] |

p 不卡 0.05。区间都不含 0。gain 0.5 的联合 mask 在这 128 条 information_extraction 上接受更少。接受数差均值是 +3.046875 个 token。这是探索复现，不是确认。不改 K，不改 0.5，不改 64，不重选这 128 条。测量经过在同日运行记录。
