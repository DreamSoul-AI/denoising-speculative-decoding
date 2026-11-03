# 2026-11-03 Qwen3-8B trajectory 第二次跨类复现

协议 `office-8b-traj-xcat2-v1`，标明探索。预注册在 `5dbdcb2`。选句、两臂和汇总仍在 `studies/trajectory_observation/structure/office_traj.py`，命令仍在 `flow/execute/office_traj.py`，本轮加 `--protocol xcat2`，提交 `2836d55`。测量时 HEAD 就是 `2836d55d915c2e7ec0d9abcbf39ce9d5e8a3d59e`，dirty 为 false。没有新的 legacy 条目。`origin/dev` 在预注册前以普通合并进入本分支，合并提交 `baa3a2e`。冲突只在 README 入口和测试清单，保留本分支上已经写完的 open_qa 跨类记录。

比较保持草稿节点数 4，`branches` [4]，`num_masks` 1。两臂都是现有 ESP，mask 为 `init` mean_prompt、`lam` 0.1、`update` true，再包上已有的 `spec.observation`。full 的 `gain` 是 0.5，`condition` none，`form` mean。traj 的 `gain` 是 0，其余相同。`gain` 0 是关闭 observation 一侧的现有开关，与被包装的 ESP 逐 token 相同。K 没有拟合，也没有乘进 mask。正式句子是 Dolly `general_qa` 非空流里跳过已用文本后的前 128 条。该类非空文本 2191 条，是 creative_writing 与 open_qa 以外、且没有作为办公室切片用过的最大类别。qid 0–90、92–128。跳过 91，因为它与 open_qa qid 93 的文本相同。SHA256 `a6a306ea868767da6cf8b6006bb4d91c5360387ae8b16ba1a20142cdd35ec677`。每条 64 token。冒烟是这 128 条的前 2 条（qid 0、1）、每条 8 token，argmax 通过，不进入下表。

主比较的差值是 traj 的 `accepted` 减去 full。胜是 traj 更高。完整版相对 traj 的接受增益是 full 减去 traj。`direction` 作用在 traj−full 上，这次返回 positive。

| **比较** | **τ 均值** | **胜** | **负** | **平** | **p** | **接受数差** | **95% 区间** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| traj 对 full | 1.265720→1.350626 | 101 | 17 | 10 | 1.006e-15 | +3.1484375 | [2.6171875, 3.6796875] |
| full 相对 traj 的增益 | 1.350626→1.265720 | 17 | 101 | 10 | 1.006e-15 | −3.1484375 | [−3.6796875, −2.6171875] |

p 不卡 0.05。区间都不含 0。gain 0.5 的联合 mask 在这 128 条 general_qa 上接受更少。接受数差均值是 +3.1484375 个 token。这是探索复现，不是确认。不改 K，不改 0.5，不改 64，不重选这 128 条。测量经过在同日运行记录。
