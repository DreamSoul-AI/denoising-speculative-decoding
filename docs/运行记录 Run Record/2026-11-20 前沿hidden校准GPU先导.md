# 2026-11-20 前沿 hidden 校准 GPU 先导

## 范围与版本

用户本次明确允许使用 GPU 推进研究。执行前已登记[研究计划第六章](<../研究计划 Research Plan 去噪推测解码.md#六2026-11-20-gpu-有界先导已获本次执行授权>)，本轮只执行来源校准线的小规模先导。没有执行 Dolly 四格、输入／输出干预、Holt 或轨迹卡尔曼；回归 K 不能称为已实现卡尔曼滤波。

运行采用 [Jason 固定提交 534a79f9c00854895d514a90a727575c0580dc10](https://github.com/jasonchengqs/speculative_mtp/commit/534a79f9c00854895d514a90a727575c0580dc10) 的完整 ZIP 快照。来源树未改动；外部 pilot.py 调用其解码、states、前沿修正与拟合函数，添加提交链审计和有界实验编排。不是本仓当前 spec_mtp 实现的 GPU 验收。当前工作区基线为 243e6d7，已有其他修改保留。

环境为 Windows、D:\anaconda3\python.exe、torch 2.11.0+cu130、Transformers 5.12.1、RTX 5090 D v2，复用 D:\models\Qwen3-0.6B，BF16、eager、BC30、branches=[14]、单 mask、贪心、thinking=false。启动前显存剩余 19,983 MiB，GPU 利用率 47%，存在其他负载。没有下载模型、安装依赖或替换 CUDA torch。

## 数据、拟合与检查

现有 SpecBench 原始文件按类别打乱后轮流取 28 条，种子 20261120，fit 16／dev 4／eval 8，prompt ID 和原始文本互不重复。每条最多 32 个新 token，输入上限 256 token；实际输入、类别、行号及文本摘要留在 inputs.json。该开发数据已被历史研究使用，当前留出仅指本次拟合不重叠，不是最终确认集。

fit 取得 295 个配对，dev 81 个配对；逐对检查 bonus 等于下一轮 root，且下一轮位置等于本轮位置加 accepted+1，共通过 376 对。由来源 frontier_pairs 提取同位置 probe／下一轮 root hidden。完整 K 在 fit 岭回归，以 dev cov@14 从 0.1、1、10 选择正则，选择 1；没有合并 dev 重拟合。独立 bias 为 fit hidden 残差均值。完整 K 与 bias 都冻结后评测，在线更新关闭。

开发集 cov@14：无修正 0.4691，完整 K 0.6420。开发集 hidden MSE：无修正 15.7997，bias 14.7615，完整 K 13.4397。选参集指标存在选择偏差，仅用于机制与工程判断。

## 实际结果

| **评测臂** | **prompt** | **实际输出 token** | **decode 调用** | **实际保留 accepted** | **τ：decode 提交数／decode 调用** |
| --- | --- | --- | --- | --- | --- |
| 无修正 | 8 | 256 | 160 | 89 | 1.5500 |
| 独立 bias | 8 | 256 | 163 | 86 | 1.5215 |
| 冻结完整 K | 8 | 256 | 153 | 98 | 1.6209 |

τ 的分子排除每条 prefill 提交的第一个 token，共 248。accepted 按实际保留 token 计数，末轮截断可能保留 accepted 而不保留 bonus，所以不能简单以 accepted+调用数重建提交数；来源原始计数单独保存在逐 prompt stats。

完整 K 相对无修正的调用次数在 4 条减少、4 条持平；相对 bias 在 6 条减少、2 条持平。按 prompt 做 10,000 次配对重采样，τ 差的探索性 95% 百分位区间分别为 [0.0209, 0.1285] 和 [0.0533, 0.1411]。只有 8 条，不能据此宣称跨类别泛化或可靠加速。

fit／dev／eval 三臂全部实际提交链共 1,401 token，均核对实际选中的 token 等于该次验证前向的 argmax，并核对截断后输出与对应决策链一致。该审计不证明树注意力与独立 AR 数值完全一致，也不覆盖采样分布保证。评测输出逐 token 比较：完整 K 与 bias 为 8/8 相同，与无修正为 7/8 相同；该差异保留为诊断，不据此推断原因。本次没有运行独立 AR，也没有重启已停止的 BF16 算子机理排查。

模型导入后至实验结束计时 64.11 秒，进程返回码 0；本进程 CUDA 峰值已分配显存约 1.76 GiB。硬超时 600 秒未触发。每次生成有同步计时，但顺序固定、没有重复计时且存在共享负载，因此不报告速度比。

## 证据与下一步

原始 Run：`D:\360安全浏览器下载\Agent\DreamSoul\tmp\GPU前沿校准验证 20261120`。含来源 ZIP 和源码快照、pilot.py、analyze.py、研究计划冻结.md、identity_sha256.json、source_sha256.json、config.json、inputs.json、states.pt、full.pt、bias.pt、prompts.jsonl、fit.json、summary.json、analysis.json、progress.txt。模型权重／配置、执行脚本和计划均有 SHA256；输入数据和来源 Python 文件摘要分别记录。制品没有进入 Git。

结论为端到端采集、拟合、加载和解码链已经连通；本次完整 K 有初步正向接受信号，独立 bias 没有同样表现。下一阶段先在可用的 4B 缓存上扩大同一三臂开发验证，继续记录完整提交链与输出差异，再设计 Dolly 类别／thinking 因子与独立跨数据集确认。更大模型、重复计时和因子矩阵尚未执行；研究计划负责预算及判据，本记录只保存本次事实。
