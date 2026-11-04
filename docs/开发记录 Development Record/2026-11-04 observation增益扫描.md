# 2026-11-04 observation 增益扫描

协议 `office-8b-traj-g-scan-v1` 的正式 GPU 已经跑完。四臂、冒烟、正式和汇总仍在 `studies/trajectory_observation/structure/office_traj.py`，命令是 `--protocol gscan`。没有新增 `office_traj_*.py`。产物前缀 `sd-office-8b-traj-g-scan-v1`，文件戳 `20261104-1813`。测量时 HEAD `68637799445f4fc5ee07c2b253f8bb91ce61a314`，dirty 为 false。命令、环境和原始汇总在 [2026-11-04 Qwen3-8B observation增益扫描](<../运行记录 Run Record/2026-11-04 Qwen3-8B observation增益扫描.md>)。

句子仍是 Dolly summarization，qid 0–127，SHA256 `f35b56bf668124a609c47dc51f97d40cabd75fdac5e6a3eb0fd133a25665020a`。长度 256，`stop_on_eos` false。四臂只改 `spec.observation.gain`：0、0.25、0.5、0.75。g=0 是 trajectory-only。冒烟（前 2 条、8 token）四臂 argmax 通过且节点数为 4，随后才跑正式。正式组有效：`argmax_all_passed` 为 true，`nodes_all_four` 为 true。

主指标是接受数。每个 g>0 与 g=0 在同一 128 条上配对。胜是该臂接受数更高。p 只作探索。τ 区间和接受数区间都是 bootstrap 百分位，种子 0，10000 次。墙钟只作描述。

| **臂** | **接受数合计** | **τ 均值** | **相对 g=0** | **接受数差均值** | **95% 区间** | **p** | **墙钟差均值（秒）** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g0 | 10235 | 1.460560 | 基线 |  |  |  |  |
| g025 | 9467 | 1.412226 | 21:100:7 | −6 | [−7.4296875, −4.6484375] | 1.606e-13 | 0.189740 |
| g05 | 8548 | 1.358059 | 6:121:1 | −13.1796875 | [−14.6953125, −11.7109375] | 6.388e-29 | 0.473827 |
| g075 | 8088 | 1.333320 | 5:118:5 | −16.7734375 | [−18.5314453125, −15.0623046875] | 4.240e-29 | 0.541773 |

三对 `direction` 都是 negative。0.25、0.5、0.75 的接受数都低于 g=0，没有一个混合增益胜过 trajectory-only。这是探索。不改 K，不改树，不改 256，不加 g=1，不把其中任何一个 g 写成默认增益。
