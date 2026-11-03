# 2026-11-03 Qwen3-8B trajectory 消融

协议 `office-8b-traj-ablate-v1`，标明探索。预注册在 `d82df53`。选句、两臂和汇总在 `studies/trajectory_observation/structure/office_traj.py`，命令在 `flow/execute/office_traj.py`，提交 `b41ffa6`。测量时 HEAD 就是 `b41ffa6c7dc83e1d9487cc8b964d27834d7d906a`，dirty 为 false。没有新的 legacy 条目。`origin/dev` 在预注册前以普通合并进入本分支，合并提交 `d589f64`。冲突只在 README 入口和测试清单，保留本分支上已经写完的 8B 树形记录。

比较保持草稿节点数 4，`branches` [4]，`num_masks` 1。两臂都是现有 ESP，mask 为 `init` mean_prompt、`lam` 0.1、`update` true，再包上已有的 `spec.observation`。full 的 `gain` 是 0.5，`condition` none，`form` mean。traj 的 `gain` 是 0，其余相同。`gain` 0 是关闭 observation 一侧的现有开关，与被包装的 ESP 逐 token 相同。K 没有拟合，也没有乘进 mask。正式句子是 Dolly `creative_writing` 从 qid 647 起、直到流结束的未用文本。非空创作流共 709 条，最后下标 708。跳过 658（与 408 相同）、678（与 198 相同）。去掉重复后是 60 条，凑不满 128，没有回用 qid 0–646。qid 是 647–657、659–677、679–708。SHA256 `d23ad590c45307fa7c1981b272adaec243bfaca5a2e390eda394bf2fe42a4cca`。每条 64 token。冒烟是这 60 条的前 2 条（qid 647、648）、每条 8 token，argmax 通过，不进入下表。

主比较的差值是 traj 的 `accepted` 减去 full。胜是 traj 更高。完整版相对 traj 的接受增益是 full 减去 traj。`direction` 作用在 traj−full 上，这次返回 positive。

| **比较** | **τ 均值** | **胜** | **负** | **平** | **p** | **接受数差** | **95% 区间** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| traj 对 full | 1.267772→1.361124 | 47 | 5 | 8 | 1.285e-09 | +3.616667 | [2.733333, 4.550000] |
| full 相对 traj 的增益 | 1.361124→1.267772 | 5 | 47 | 8 | 1.285e-09 | −3.616667 | [−4.550000, −2.733333] |

p 不卡 0.05。区间都不含 0。gain 0.5 的联合 mask 在这 60 条上接受更少。这是探索，不是确认。不改 K，不改 0.5，不改 64，不重选这 60 条。测量经过在同日运行记录。
