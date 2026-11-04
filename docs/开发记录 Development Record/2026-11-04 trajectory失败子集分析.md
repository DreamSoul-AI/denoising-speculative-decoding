# 2026-11-04 trajectory 失败子集分析

`office-8b-traj-failset-v1` 的描述函数在 `studies/trajectory_observation/structure/office_traj.py`。选句、SHA256 和 jsonl 读取仍用该文件里已经冻结的函数，以及 `office_tree.read_rows`。`office-8b-traj-xcat7-v1` 的已归档接受数文件名保持 `sd-office-8b-traj-xcat7` / `20261104-045236`，不读墙钟那一对。这些描述函数没有单独的 `main`，也没有 argparse，legacy 清单不增加。描述结果在 [2026-11-04 Qwen3-8B trajectory失败子集](<../运行记录 Run Record/2026-11-04 Qwen3-8B trajectory失败子集.md>)。
