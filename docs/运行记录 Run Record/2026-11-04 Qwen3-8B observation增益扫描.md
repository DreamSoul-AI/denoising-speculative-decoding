# 2026-11-04 Qwen3-8B observation 增益扫描

协议 `office-8b-traj-g-scan-v1`，标明探索。测量时 HEAD `68637799445f4fc5ee07c2b253f8bb91ce61a314`，dirty 为 false。命令是 `python -u -m studies.trajectory_observation.structure.office_traj --protocol gscan --dest D:\360安全浏览器下载\Agent\DreamSoul\tmp\grokbot --only all`，`PYTHONPATH` 指到本仓 `src`。`HF_HOME`、`HF_DATASETS_CACHE`、`HUGGINGFACE_HUB_CACHE`、`TORCH_HOME` 指向该目录下已经存在的缓存，`TEMP` 与 `TMP` 指向 `D:\360安全浏览器下载\Agent\DreamSoul\tmp`。整段进程 4554.7 秒，exit 0。冒烟、header、四支 jsonl 和 `sd-office-8b-traj-g-scan-v1-summary-20261104-1813.json` 都在该目录，文件名以 `sd-office-8b-traj-g-scan-v1-` 开头、以 `-20261104-1813` 结尾。这些文件不进 Git。

权重 `D:\models\Qwen3-8B`，BF16，CUDA，eager，`local_files_only`。没有下载模型或数据集。Dolly 用本机已有缓存。torch 2.11.0+cu130，Python 3.13.9，设备 NVIDIA GeForce RTX 5090 D v2，显存 24455 MiB。正式加载前占用 3070 MiB，compute-apps 没有报出数字形式的显存占用。没有加载 fp32，没有另起一项 GPU 计算。

冒烟是正式 128 条的前 2 条（qid 0、1），每条 8 token，四个增益都跑。四臂 argmax 都通过，草稿节点数都是 4，块复杂度都是 10。冒烟不进入符号检验。正式开始前用第 1 条 prompt、8 token、四臂各做一次不计时预热。

正式句子是 summarization，qid 0–127，没有跳过。SHA256 `f35b56bf668124a609c47dc51f97d40cabd75fdac5e6a3eb0fd133a25665020a`，与预注册一致。每条 256 token，T=0，`stop_on_eos` false。四臂都是 `method` esp、`num_masks` 1、`impl` efficient、`tree.mode` static、`branches` [4]、剪枝开。mask 为 `init` mean_prompt、`lam` 0.1、`update` true。`condition` none，`form` mean，协议名 `trajectory-observation-blend-v1`。增益按 g0、g025、g05、g075 为 0、0.25、0.5、0.75。每一步草稿节点数都是 4。块复杂度都是 10。`argmax_all_passed` 为 true，`nodes_all_four` 为 true，组有效。

主指标是接受数。差是该臂减去 g=0。胜是该臂更高。p 是不含平局的双侧符号检验，只作探索。区间是 bootstrap 百分位，种子 0，10000 次。τ 均值按臂报告，τ 差是该臂减 g=0。墙钟差同样是该臂减 g=0，只作描述。128 条四臂的 `wall_seconds` 都完整。

| **臂** | **gain** | **接受数合计** | **接受数均值** | **τ 均值** | **相对 g=0** |
| --- | --- | --- | --- | --- | --- |
| g0 | 0 | 10235 | 79.9609375 | 1.460560 | 基线 |
| g025 | 0.25 | 9467 | 73.9609375 | 1.412226 | 21:100:7 |
| g05 | 0.5 | 8548 | 66.78125 | 1.358059 | 6:121:1 |
| g075 | 0.75 | 8088 | 63.1875 | 1.333320 | 5:118:5 |

| **臂** | **接受数差均值** | **95% 区间** | **p** | **τ 差均值** | **τ 95% 区间** | **墙钟差均值（秒）** | **墙钟均值（秒）** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g025 | −6 | [−7.4296875, −4.6484375] | 1.606e-13 | −0.048334 | [−0.060042, −0.037266] | 0.189740 | 6.692807 |
| g05 | −13.1796875 | [−14.6953125, −11.7109375] | 6.388e-29 | −0.102501 | [−0.114648, −0.090712] | 0.473827 | 6.976894 |
| g075 | −16.7734375 | [−18.5314453125, −15.0623046875] | 4.240e-29 | −0.127240 | [−0.140913, −0.114357] | 0.541773 | 7.044840 |

g0 的墙钟均值是 6.503067 秒。三对接受数 `direction` 都是 negative，区间都不含 0。三个混合增益的接受数都低于 trajectory-only，增益越高，接受数越少，墙钟越长。g=0.5 对 g=0 的 6:121:1 与墙钟按长度里 256 token 的 traj 对 full（121:6:1）是同一对比的反向计数。独立 AR 不一致条数是 g0 102、g025 102、g05 106、g075 108，只作诊断。

这是探索，不记成确认，也不选出默认增益。不改 K，不改树，不改 256，不加 g=1，不重选这 128 条。
