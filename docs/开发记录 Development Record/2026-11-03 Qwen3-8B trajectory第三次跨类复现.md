# 2026-11-03 Qwen3-8B trajectory 第三次跨类复现

协议 `office-8b-traj-xcat3-v1`，标明探索。预注册在 `fc130b6`。选句、两臂和汇总仍在 `studies/trajectory_observation/structure/office_traj.py`，命令仍在 `flow/execute/office_traj.py`，本轮加 `--protocol xcat3`，提交 `a28b83f`。测量时 HEAD 就是 `a28b83f4bfd3981d7589c09ce6cc38103f1b17d3`，dirty 为 false。没有新的 legacy 条目。`origin/dev` 在预注册前以普通合并进入本分支，合并提交 `cead073`。冲突只在 README 入口和测试清单，保留本分支上已经写完的 general_qa 跨类记录。

比较保持草稿节点数 4，`branches` [4]，`num_masks` 1。两臂都是现有 ESP，mask 为 `init` mean_prompt、`lam` 0.1、`update` true，再包上已有的 `spec.observation`。full 的 `gain` 是 0.5，`condition` none，`form` mean。traj 的 `gain` 是 0，其余相同。`gain` 0 是关闭 observation 一侧的现有开关，与被包装的 ESP 逐 token 相同。K 没有拟合，也没有乘进 mask。正式句子是 Dolly `classification` 非空流里跳过已用文本后的前 128 条。该类非空文本 2136 条，是 creative_writing、open_qa 与 general_qa 以外、且没有作为办公室切片用过的最大类别。qid 0–127。扫描这 128 条时没有与已用文本相同的条目，所以没有跳过。SHA256 `01c21dbf4a774b2395bed3a8606d3fe20d92a0cb7d2562ff8e8d965b9f2bd35e`。每条 64 token。冒烟是这 128 条的前 2 条（qid 0、1）、每条 8 token，argmax 通过，不进入下表。

主比较的差值是 traj 的 `accepted` 减去 full。胜是 traj 更高。完整版相对 traj 的接受增益是 full 减去 traj。`direction` 作用在 traj−full 上，这次返回 positive。

| **比较** | **τ 均值** | **胜** | **负** | **平** | **p** | **接受数差** | **95% 区间** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| traj 对 full | 1.243178→1.352919 | 93 | 23 | 12 | 3.597e-11 | +4.0546875 | [3.09375, 5.015625] |
| full 相对 traj 的增益 | 1.352919→1.243178 | 23 | 93 | 12 | 3.597e-11 | −4.0546875 | [−5.015625, −3.09375] |

p 不卡 0.05。区间都不含 0。gain 0.5 的联合 mask 在这 128 条 classification 上接受更少。接受数差均值是 +4.0546875 个 token。这是探索复现，不是确认。不改 K，不改 0.5，不改 64，不重选这 128 条。测量经过在同日运行记录。
