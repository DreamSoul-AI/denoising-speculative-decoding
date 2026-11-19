**去噪推测解码 研究计划**
更新日期：2026年11月20日。这是唯一中文研究计划。本文对照当前代码写算法，并把已跑实验收成一张结论表。逐次预注册、旧英文计划和运行日志从 Git 历史与 [运行记录](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/8f25a61515cf331e3a3cf3ba0dfbc66f217e315c/docs/%E8%BF%90%E8%A1%8C%E8%AE%B0%E5%BD%95%20Run%20Record/INDEX.md) 取证，不在这里重写。工程待办见 [开发计划](<开发计划 Development Plan 去噪推测解码.md>)。

本轮重点与当前判断见第八章：已在外部固定来源快照上完成4B前沿hidden校准、模式与类别、校准重复、机制及实际计时。60分钟协议见第七章，实际证据与失败见[本轮运行记录](<运行记录 Run Record/2026-11-20 六十分钟前沿校准研究.md>)；第一、二章的本仓机制说明和旧协议结果保留原义。

当前评价口径已按用户要求调整：共享电脑持续被其他任务使用，后续方法比较以实际保留的 draft 接受数和每次验证调用的提交量 τ 为主，不以墙钟耗时、token/s 或速度比判断优劣。既有计时只保留为历史观察，不参与当前研究结论与方案选择。

# 一、当前算法与数学
本次沿用远端 dev 的已核实基线 [8f25a61](https://github.com/DreamSoul-AI/denoising-speculative-decoding/tree/8f25a61515cf331e3a3cf3ba0dfbc66f217e315c)。本章路径与行号对应该版本源码，当前关联工作树仍在 243e6d7。正式办公室比较用的是 ESP 轨迹加 observation 混合；卡尔曼类是同一解码循环上的其它 mask。

## 解码循环
`SpeculativeDecoder.generate`（`src/spec_mtp/structure/algorithm/decode/loop.py` 第 196–465 行）按模块文档（第 5–22 行）做两段：

- **Prefill。** prompt 嵌入与 k 个 mask 拼成一次因果前向（第 231–243 行）。最后一个 prompt 位置选出第一枚提交 token，作为下一棵树的根；k 个 mask 位置的 logits 留下，用来建第一棵草稿树。mask 的 KV 随后裁掉（第 264 行）。
- **静态树。** 每轮用上一轮 mask logits 的 softmax 建树（第 291–292 行）。正式配置走 `build_static_tree`（`src/spec_mtp/structure/algorithm/decode/tree.py` 第 75–107 行）：`branches` 的长度不超过 k；每一层只让该层概率最高的节点继续生孩子；剪枝打开时，子 token 不取父 token。`impl=efficient` 要求 `tree.mode=static`（`loop.py` 第 159–164 行）。
- **树注意力块。** 一块的顺序是根、草稿节点、每个 owner 的 k 个 mask（`loop.py` 第 320–336 行）。块内注意力在 `tree_attention.py` 的 `block_layout`（第 8–34 行）：根只看自己；草稿节点看祖先链和自己；某个 owner 的 mask 看该 owner 的注意力集合，以及同一 owner 上排在自己前面的 mask。`efficient` 把这块结构缓存下来（第 77–83 行）。下一轮的 mask logits 取自最深接受节点；没有接受草稿时取根（`loop.py` 第 450–453 行，`mask_indices_of` 第 37–41 行）。
- **贪心验证。** 温度 ≤ 0 时 `select_token` 取 argmax（`loop.py` 第 96–100 行）。验证从根往下走，预测 token 与某个子节点相同就接受该节点，否则停下（第 356–377 行）。停下的那一枚是 bonus。`stats['accepted']` 在 EOS 或长度截断之前累加匹配到的草稿节点（第 380–381 行）。
- **τ。** `prompt_tau` 是提交数除以 `decode_calls`；调用数为 0 时没有 τ（`src/spec_mtp/analysis.py` 第 10–16 行）。解码循环里的 `stats['committed']` 只在解码轮写入（`loop.py` 第 394–400 行），不含 prefill 交出的第一枚 token。模块文档写的是：τ = 每个 decode call 的提交 token（接受的草稿加 bonus）（第 22 行）。办公室行用的就是 `stats['committed'] / decode_calls`（`studies/trajectory_observation/structure/office_tree.py` 第 182 行）。

温度 ≤ 0 时，每个提交 token 记下当次前向的 argmax，整段生成结束后做审计（`loop.py` 第 246–250、473–484 行）。这是 T0 的 argmax 审计。它不提供采样分布保证。独立 AR 是否逐 token 相同，另计，只作诊断。

## ESP 基线 mask
`ESPMaskProvider`（`src/spec_mtp/structure/algorithm/mask/providers.py` 第 83–137 行）。默认 `init='mean_prompt'`、`lam=0.1`。

- 初值：\(m_0=\frac{1}{t}\sum_j e_j\)，k 个槽共用这一条均值（第 104–107 行）。
- 每提交一枚 token 更新一次，含 prefill 的第一枚和随后每一枚解码提交（`loop.py` 第 272–274、394–397 行）：

\[
m \leftarrow (1-\lambda)m + \lambda e_{\mathrm{new}},\qquad \lambda=0.1
\]

代码写成 \(m \leftarrow m+\lambda(e_{\mathrm{new}}-m)\)（`providers.py` 第 133–137 行）。\(e_{\mathrm{new}}\) 是该 token 的输入嵌入。`update=false` 时不改 m。

## Observation 与增益混合
`ObservationBlendMaskProvider`（`providers.py` 第 285–398 行）包住上面的轨迹。正式臂是 `condition='none'`、`form='mean'`。

\[
z=\sum_v p(v)\,e_v,\qquad p=\mathrm{softmax}(\ell)
\]

实现是 `softmax(logits.float())` 再乘嵌入表（第 343–346 行）。`form='top1'` 时 z 改成该行 argmax 的嵌入。z 只活一轮，`on_token_committed` 只转给基线，不把 z 写进轨迹（第 397–398 行）。

混合是

\[
m=(1-g)\hat m+g z=\hat m+g(z-\hat m),\qquad g\in[0,1]
\]

`g=0` 时直接返回轨迹（第 356–357 行）。**k=1 时只有根上的那个 mask 被混合。** `masks_by_depth` 的循环是 `for depth in range(k)`，`slots=k-depth`（第 359–361 行）。k=1 时 depth 只有 0，只改根的槽 0；深度 1 的草稿 owner 保持轨迹。正式运行的 `num_masks` 就是 1，所以增益只加在根 mask 上。

### 代码与 docstring 不一致
类文档（第 290–292 行）写：深度 d 的槽 j 在 \(d+j\le k\) 时改用 \(z_{d+j}\)，否则保持 \(\hat m_j\)。代码用 0 起计的 depth、槽下标，只改写 `j < k-depth`，也就是 \(d+j<k\)。张量有 k+1 行 owner（第 352–355 行），深度 k 的那一行不进循环，不会被混合。

拒绝条件化的文档（第 296–301 行）写槽 1 使用 \(z_{d+1}\)。`masks_for_tree` 取的是 `_probs[depth]` 和 `_observation[depth]`，depth 为该 owner 的树深度，根记 0（第 373–391 行）。用的是 \(z_d\)，不是 \(z_{d+1}\)。正式八类和 g-scan 的 `condition` 都是 `none`，这条分支没有进那些比较。

## 卡尔曼变体（按现实现）
都在 `providers.py`。除特别写出的槽以外，只改槽 0，其余槽仍是基线 mask。

| **类** | **公式** | **位置** |
| --- | --- | --- |
| `KalmanMaskProvider` | 标量随机游走。\(P^-=P+q\)，\(K=P^-/(P^-+R)\)，\(m\leftarrow m+K(e-m)\)。online 用 Sage–Husa 更新标量 \(q,r\)（遗忘因子默认 0.95）；frozen 锁住 prompt 结束时的 K。没有本轮 verify 的 z | 第 531–556 行 |
| `VerifyCorrectionMaskProvider` | \(m=\hat m+K(z-\hat m)\)，z 是上一枚已提交嵌入。标量、对角、仿射 \( \hat m+(z-\hat m)A^\top+b \)、低秩 \( \hat m+(z-\hat m)V U^\top+b \) | 第 688–720 行 |
| `OnlineVerifyKalman` | 逐维 \(K=(P+Q)/(P+Q+R)\)。full：槽 0 为 \(\hat m+K(z-\hat m)\)。trajectory 臂原样返回 ESP。observation 臂把先验换成随机游走状态 | 第 785–823 行 |
| `AlignedResidualKalman` | 状态是标量相关系数 F。槽 0 为 \(\hat m+F\cdot\nu_{\mathrm{prev}}\)。F 朝相邻残差的余弦更新，并夹到 \([-1,1]\)。R 用余弦残差的平方更新，不减 P | 第 884–918 行 |
| `HybridGainKalman` | \(m=\hat m+K\odot(z-\hat m)\)。创新 \((y-\hat m)-K\odot(z-\hat m)\)，步长 \(P^-\odot r/(R+P^-\odot r^{\odot 2})\)，\(P^-=P+Q\)。`observe='last'` 改到本轮 bonus 上学习 | 第 1022–1102 行 |
| `OutputHiddenHybridKalman` | 在 output-hidden 轨迹上，\(h\leftarrow h+(z-h)\odot K\)，再按嵌入表平均范数缩放。trajectory 模式不乘 K | 第 1229–1244 行；缩放在第 74–78 行 |

这些类都不是 `studies/k_affine_residual` 里、解码循环之外的 `postcommit-final-norm-v1`。

## 正式运行配置
八类 trajectory 消融、墙钟和 g-scan 共用（`studies/trajectory_observation/structure/office_traj.py` 第 110–133、898–910 行；`office_tree.py` 第 40、341 行）：

- 模型：Qwen3-8B，权重 `D:\models\Qwen3-8B`，BF16，CUDA，eager，`local_files_only`
- mask：`method=esp`，`num_masks=1`，`init=mean_prompt`，`lam=0.1`，`update=true`
- 树：static，`branches` [4]，剪枝开，`impl=efficient`。块复杂度 \((1+1)\times(1+4)=10\)
- 解码：温度 0，`stop_on_eos` false
- 数据：Dolly `databricks/databricks-dolly-15k` train。八类与墙钟按长度的 64 token 档是每条 64 token；g-scan 与墙钟的 256/512 档见下表
- 混合：`condition=none`，`form=mean`，协议名 `trajectory-observation-blend-v1`。traj / g0 的增益是 0；八类里的 full 是 0.5

# 二、结论总表
差值方向在表内写明。胜负平是逐条符号检验的胜:负:平。p 是双侧符号检验。区间是 bootstrap 百分位，种子 0，10000 次。这些办公室比较标明探索，不另设确认线。原始 jsonl 在运行记录所指目录，不进 Git。

八类的问题相同：增益 0.5 的联合 mask，是否比关掉 observation（增益 0）接受更多。设置就是上一节的正式配置，每类一条 Dolly 切片，64 token。效应是 traj 减 full 的接受数均值；τ 箭头是 full→traj。

| **名称** | **问题** | **设置** | **效应量** | **PR** | **结论** |
| --- | --- | --- | --- | --- | --- |
| `office-8b-traj-ablate-v1` | 联合是否多于 trajectory-only | creative_writing 剩余 60 条 | 47:5:8，+3.616667，τ 1.267772→1.361124 | [#69](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/69) | 增益 0.5 接受更少 |
| `office-8b-traj-xcat-v1` | 同上，换类 | open_qa 128 条 | 103:12:13，+3.2421875，τ 1.253489→1.338024 | [#70](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/70) | 同上 |
| `office-8b-traj-xcat2-v1` | 同上 | general_qa 128 条，跳过 qid 91 | 101:17:10，+3.1484375，τ 1.265720→1.350626 | [#71](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/71) | 同上 |
| `office-8b-traj-xcat3-v1` | 同上 | classification 128 条 | 93:23:12，+4.0546875，τ 1.243178→1.352919 | [#72](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/72) | 同上 |
| `office-8b-traj-xcat4-v1` | 同上 | closed_qa 128 条 | 95:22:11，+2.4921875，τ 1.357949→1.431103 | [#73](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/73) | 同上 |
| `office-8b-traj-xcat5-v1` | 同上 | brainstorming 128 条 | 95:17:16，+2.6171875，τ 1.290219→1.358632 | [#74](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/74) | 同上 |
| `office-8b-traj-xcat6-v1` | 同上 | information_extraction 128 条 | 106:10:12，+3.046875，τ 1.326882→1.415774 | [#75](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/75) | 同上 |
| `office-8b-traj-xcat7-v1` | 同上 | summarization 128 条 | 97:17:14，+2.6640625，τ 1.340886→1.418055 | [#76](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/76) | 同上。八个 Dolly 类到此用完 |
| `office-8b-traj-wall-v1` | 接受数更高时墙钟是否更短 | 同一 summarization，64 token | 接受数仍 97:17:14；墙钟 traj 更短 99:29:0，差 −0.100136 秒 | [#84](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/84) | 接受数与墙钟同向，都离开 0 |
| `office-8b-traj-failset-v1` | full 胜出的句子有没有共同长度 | 上面八组已归档 jsonl | 子集条数等于各组的负。字符均值四组更高、四组更低 | [#85](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/85) | 看不出按长度分开的失败模式 |
| `office-ar-inconsistency-v1` | 独立 AR 不一致是口径还是正确性 | 八组已归档行，不加载模型 | 八组都重叠，argmax 全过 | [#89](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/89) | 可隔离的口径问题 |
| `office-ar-first-fork-v1` | 不一致从哪一个 token 开始 | summarization 64 token；64 条至少一臂哈希不等 | 两臂首分叉下标与两侧 token 都相同的 35/64（0.546875）。fp32 对照未加载 | [#95](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/95) | 不贴共享参照，也不贴分叉分开 |
| `office-ar-bf16-determinism-v1` | BF16 上 AR 自己会不会翻 | 上面 64 条，只跑 AR | 两次整段 AR 一致率 1；整段对逐 token prompt 为 0.3125；重放对归档参照为 0.5 | [#96](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/96) | 贴「AR 路径不稳定」 |
| `office-8b-traj-wall-by-length-v1` | 墙钟差是否随长度变大 | summarization，64/256/512 | 墙钟差 −0.096467 / −0.474111 / −1.148640 秒，绝对值递增。512 的差主要在 verify | [#97](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/97) | length-dependent，分解在 verify |
| `office-8b-traj-g-scan-v1` | 哪个 g 优于 g=0 | summarization，256 token；g=0, 0.25, 0.5, 0.75 | 接受数合计 10235 / 9467 / 8548 / 8088。τ 均值 1.460560 / 1.412226 / 1.358059 / 1.333320。相对 g0 的胜负平 21:100:7、6:121:1、5:118:5，方向都是 negative。墙钟差 +0.189740 / +0.473827 / +0.541773 秒 | [#102](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/102) | 三个 g>0 的接受数都低于 trajectory-only，g 越大越少、墙钟越长 |
| 同预算树形 | 把 4 个节点拆成两层是否更多接受 | [2,2] 与 [3,1] 对 [4]，`num_masks=2`，64 token | 4B 与 8B 上 [2,2] 约少 1.8–1.9 个接受 token，区间离开 0。[3,1] 也是负方向，8B 区间含 0 | [#78](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/78) | 两层树更差。之后的 trajectory 协议保持 [4] |
| `kalman-mask-scalar-v1` | 无 verify 观测的标量滤波是否胜过 ESP | 合成确认种子 21/22/23 | 101:83:8，p=0.210 | [联合 mask 开发记录](<开发记录 Development Record/2026-11-01 trajectory与observation联合mask.md>) | 确认未通过。这不是带 z 的合并 |
| 离线 K（对角、仿射、岭、低秩） | 冻结的 \(K(z-\hat m)\) 是否胜过 ESP | 合成，以及 Qwen3-0.6B / 4B 短句 | 对角 91:80:21（p=0.445）。仿射探索 101:85:6（p=0.271）。0.6B 岭回归 128 条 50:40:38（p=0.343）。4B 秩 4 探索 15:10:7（p=0.424） | [联合 mask 开发记录](<开发记录 Development Record/2026-11-01 trajectory与observation联合mask.md>) | 小模型上弱或未通过。合成岭回归曾通过，没有带到 Qwen。0.6B 的 oracle 上界只说明被拟合的句子里函数类还有余量 |
| 在线卡尔曼、对齐残差、混合增益 | 在线更新 K 是否胜过 ESP | Qwen3-4B / 0.6B，CUDA，BF16，树 [4] | 在线 4B 为 2:24:6，0.6B 为 1:28:3。对齐残差 4B 新 32 条 5:9:18（p=0.424）。混合增益 128 条 4B 为 33:28:67（p=0.609） | [联合 mask 开发记录](<开发记录 Development Record/2026-11-01 trajectory与observation联合mask.md>) | 负或弱。额外分支在多一个节点时接受数上升，同节点预算的替换没有在 4B 与 0.6B 上同时通过 |
| `output-hidden-hybrid-kalman-v1` | hidden 空间的混合 K 是否胜过 ESP 和 hidden 原版 | SpecBench 209–336，64 token | 4B 对 ESP 60:51:17（p=0.448）。0.6B 对 ESP 1:125 | [#61](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/61) | 负结果，已关闭 |

八类、墙钟和 g-scan 的 argmax 审计都通过。独立 AR 不一致只诊断，不改写接受数。

# 三、未验证/待定
- **k>1 与深树。** 正式 observation 比较都是 `num_masks=1`、单层 [4]。同预算的 [2,2]、[3,1] 只比较了树形，接受数更低，没有把 observation 或卡尔曼放到 k>1 上再测。
- **observation 的注入方式没有拆开。** \(z=\sum p(v)e_v\) 是嵌入的凸组合，一般不落在 token 上，范数也会缩。正式运行固定 `form=mean`，只扫了 g。`form=top1` 的合成确认已经更差（ESP 联合·top1·拒绝对 ESP 轨迹 74:103，p=0.035），8B 上没有把「离开流形」和「范数变小」分成两臂。
- **温度只有 0。** 办公室配置 `temperature=0`，验证是 argmax。T>0 的采样匹配没有正式结果。T0 通过不能推广成采样分布保证。
- **模型只有 Qwen。** 8B 是正式轨迹和 g-scan；0.6B 与 4B 只出现在卡尔曼和小树形比较里。没有第二种模型族。

采样分布保证仍未完成。BF16 相对独立 AR 的算子数值机理排查保持停止。第四章记录本轮新增的研究问题和拟议验证。

# 四、本轮研究问题与拟议验证

2026年11月20日，由 EMA／Holt／卡尔曼的学习讨论，以及 Jason 对拟合数据和 thinking 设置的反馈形成这组问题。研究方向是观察位置、趋势与预测残差是否能帮助免训练起草，并检验收益对校准数据分布的依赖。下述方案按研究问题组织，实际模型、数据规模、计算预算与运行安排在执行前登记。来源校准线已启动并取得第六、七章记录的实测，状态估计线尚未新增执行。前两章的旧协议和历史结果保留原义。

## 从三份算法说明接入研究计划

三份说明提供本轮研究的机制和数学入口，Jason 的前沿修正提供一条具体的来源实现。研究保留两条互补主线，一条比较从零阶平滑到一阶状态估计，另一条复现和分析前沿残差校准。两条线先各自建立对照，再在有依据的配置上组合。

| **来源** | **进入本计划的内容** | **研究中的作用** |
| --- | --- | --- |
| [ESP 算法说明](https://dreamsoul.feishu.cn/docx/NL8gdTBZioubwfxj0BKchWXVnib) | soft mask、候选树、BC、position ID／RoPE、目标解码与验收 | 固定数据流、位置和预算，给所有方法提供同一验证合同 |
| [EMA 算法说明](https://dreamsoul.feishu.cn/docx/WNmAdTaxXoN5y5x1xydcBOH6nif) | 输入／输出观测，零阶、原始差分 EMA、Holt，锚点与速度 | 定义状态估计的候选与退化关系，检查同一残差更新了什么 |
| [卡尔曼算法说明](https://dreamsoul.feishu.cn/docx/KpWVdcrh7o0R1xx7PBocsoxTnLb) | 先预测再校正，误差协方差，历史 calibration 与在线更新 | 明确状态、观测、噪声模型和增益的来源，分别检验估计状态与拟合校准系数 |
| [Jason 534a79f](https://github.com/jasonchengqs/speculative_mtp/commit/534a79f9c00854895d514a90a727575c0580dc10) | bonus 输入差值到同位置 final hidden 残差的回归，数据入口修复 | 来源复现基线，与自有输入／输出修正按位置、信息和成本对齐 |

### 状态估计主线

令 $y_t$ 表示当前正式路径上可取得的表示，可取输入 embedding 或该位置真实 hidden。$\beta$ 一律表示旧速度的保留权重，$\alpha$ 表示位置的新观测权重。$t$ 按观测位置计数，一次 forward 的多个提交按路径顺序吸收。

| **候选** | **估计与更新** | **决定性的对照** |
| --- | --- | --- |
| 零阶 EMA／ESP 参照 | $M_t=(1-\alpha)M_{t-1}+\alpha y_t$，向未来槽重复同一状态。与 ESP 对齐还要统一 prompt 均值初始化与逐 token 调度 | 已有输入零阶作为基线，输出路线另定义 hidden 到输入 mask 的映射 |
| 原始差分 EMA | $V_t=\beta V_{t-1}+(1-\beta)(y_t-y_{t-1})$，锚点分别登记最近观测或零阶均值 | 先核对旧实现的锚点和速度，再分辨换锚点与换更新公式的效应 |
| 一阶 Holt | $\hat M_t=M_{t-1}+V_{t-1}$，$\varepsilon_t=y_t-\hat M_t$，$M_t=\hat M_t+\alpha\varepsilon_t$，$V_t=V_{t-1}+\alpha(1-\beta)\varepsilon_t$ | 对齐初值时，$\alpha=1$ 与最近观测锚点的差分 EMA 速度相同。先验证这个边界，再看保留位置估计的影响 |
| 一阶轨迹卡尔曼 | 状态为 $(M,V)$，以位置观测的同一残差修正二者，增益从误差协方差递推得到 | 与固定增益 Holt 共用表示、预测模型、读出层和验收，检验增益递推的增量 |

轨迹卡尔曼需写明 $F,H,P_0,Q,R$ 的含义和校准来源。首个候选可使用各表示维度共享的 $2\times2$ 协方差，作为待核验的简化假设。已知输入 embedding 的查表值确定，观测噪声在这里描述它对潜在轨迹的偏离。输出 hidden 的观测延迟及尺度映射单列，避免把前一位置预测出的 token 当成已有同位置 hidden。

状态估计产生初步外推向量，校准层决定实际 soft mask 或草稿 logits。轨迹滤波增益与离线校准 K 各自记录。比较状态估计时先固定或关闭校准层，比较校准层时先固定状态估计。拟合 K 冻结／在线更新、临时 draft 条件化、残差预测和加速度状态保留为后续选项，先取得对应一阶对照的结果再扩展。

### 观测来源、修正空间与验收

输入／输出有两个要分别记录的含义。一个是状态估计观察输入 embedding 还是输出 hidden，另一个是在模型输入处修正 soft mask，还是在前向后修正 final hidden／草稿 logits。Jason 的输出修正可以接在 ESP 输入均值探测之后。更换观测表示与更换修正位置分别是一个干预因素。

目标解码仍是可研究项。首轮方法比较固定贪心验收。后续如研究采样，另登记提案概率、接受和拒绝后的修正规则并验证分布目标；如研究 beam，登记路径评分、beam 宽度与保留规则。T0 的逐提交 argmax 审计继续只对应贪心合同。

## 灵感、证据与假设

宏观目标是提高同模型推测解码的实际生成效率，让方法的收益和适用条件可以被复核。当前阶段先回答修正空间、校准分布与生成模式三个问题。输出 hidden 已包含上下文，因而可能比输入 embedding 更容易获得有用修正，这是待检验的机制假设。Holt 和轨迹卡尔曼是后续状态估计候选。

Jason 提交的 534a79f 以 d91ed75 为父提交，已核对完整 diff、Dolly loader、统一分词入口、前沿修正、拟合脚本和 Qwen3-8B BC30 配置。它有两个与本轮直接相关的修改。

- Dolly 新增 all／none 类别选择及 offset，默认仍为 creative_writing。全类别记录的 category 字段目前统一写成配置值 all，按数据顺序取前 N 条。研究使用时须恢复原始类别和全库行号，记录真实类别分布。
- Dolly 与 SpecBench 共用 encode_prompt，都会传入 chat_template_kwargs。Qwen3 的 enable_thinking=false 因此可在拟合采集端生效。入口没有强制关闭 thinking，实际采集配置仍须核实。
- 同次提交还改变了 context 顺序、长输入处理和 EOS 收集。它没有改变 K 的拟合目标，fit_frontier_gain.py 只改了打印文字。尚未取得修正后实际采集命令、states 清单和冻结 K，因此不能据提交本身认定已完成修复后复验。

来源为 [Jason 提交](https://github.com/jasonchengqs/speculative_mtp/commit/534a79f9c00854895d514a90a727575c0580dc10)、[Dolly loader](https://github.com/jasonchengqs/speculative_mtp/blob/534a79f9c00854895d514a90a727575c0580dc10/denoise_mtp/src/data/dolly.py) 和 [统一分词](https://github.com/jasonchengqs/speculative_mtp/blob/534a79f9c00854895d514a90a727575c0580dc10/denoise_mtp/src/data/tokenization.py)。

| **研究问题** | **待检验假设** | **会改变判断的结果** |
| --- | --- | --- |
| 修正空间 | 输出 hidden 的上下文化信息能带来更好的草稿 | 输出只改善拟合误差，或计入额外成本后没有接受和速度收益，就收窄“输出更好”的判断 |
| 拟合分布 | 多类别校准在同等拟合预算下，比单类别更能迁移到未见 prompt 或类别 | 多类别在独立评测中不提高配对收益，或收益由样本量、输入长度差解释 |
| thinking 设置 | 拟合与推理的模式不匹配削弱 K 的收益 | 对齐输入后，模式对照的收益差区间仍不足以确定方向，保留未决状态 |
| 状态更新 | Holt 的位置残差反馈优于原始差分 EMA，协方差增益可在此基础上进一步改善 | 对齐观测和读出后，收益消失或成本抵消收益，就不推进更高阶状态 |

## 输入与输出的算法、信息和时点

用 b 表示本轮确定的 bonus token，m_b 是这个位置在本次前向使用的 soft mask，c 是 b 后面的目标 token。Jason 用输入差值 d_b=E[b]-m_b 加截距，拟合最终归一化 hidden 的残差。

$$u_b=[d_b;1],\qquad h_b^{\mathrm{corr}}=h_b^{\mathrm{probe}}+u_bK_{\mathrm{out}}.$$

修正发生在本轮验证得出 b 之后。下一轮同位置 root 的真实 hidden 提供标签，修正后的 hidden 经过 LM head 产生 c 的草稿分布。这里 K 是离线岭回归系数。公式具备残差校正形式，是否采用协方差递推属于另一项模型选择。[前沿修正实现](https://github.com/jasonchengqs/speculative_mtp/blob/534a79f9c00854895d514a90a727575c0580dc10/denoise_mtp/src/spec/frontier_gain.py#L105)

输入方案要同时给出预测目标和信号可用时间。b 已知时 E[b] 可以直接查表。若把 E[b] 当作同位置输入拟合目标并使用 d_b 作特征，映射存在恒等解 m_b+d_b=E[b]。这种配对只能检查数值和位置，不能作为未知未来 token 的预测证据。

| **候选** | **信号与操作** | **比较所回答的问题** |
| --- | --- | --- |
| 输出后修正 | 本轮前向结束后用已知 b 修正 h_b 的草稿分布，沿用 Jason 前沿位置 | 复现其同位置 hidden 残差修正的收益 |
| 输入前修正 | 在探测前，用当时已提交历史和已获得残差修正未来槽 embedding，随后正常前向 | 可部署的输入修正策略是否有效。须逐项记录它比输出后修正少了哪些当轮信息 |
| 同信息输入重算 | 得到 b 后修正同位置输入，并重新前向，取预测 c 的分布 | 对齐当轮信息后比较输入与输出干预。额外前向计入耗时，先作为机制对照 |
| 无修正与独立 bias | 对应位置、树和模式下的基线；bias 在相同拟合集独立拟合 | 估计完整 K 相对基线和仅截距的增量 |

输入前修正与输出后修正首先是系统方案比较。只有信号、目标 token 和可用时点一致时，才将差异归因于修正空间。输出在 b 位置的 logits 预测 c，输入 b 槽也须取得预测 c 的 logits，不能将输入 c 槽的输出与它错位比较。临时 draft 条件化可以作为后续输入特征，正式历史仍按提交路径更新。

## 最小验证顺序与数据设计

### 1. 先核对数据和位置合同

核对前后版本的类别计数、prompt ID、文本拼接、模板参数、截断和 EOS。在固定原始文本上检查实际 prompt token，确认 fit 和 inference 的模式。用可控的小规模输入核对 probe、bonus、下一轮 root hidden 的同位置配对，额外前向记真实调用数。

拟合采集、选正则与最终评测按 prompt 隔离，保留数据集全局 ID、原始类别、文本摘要和 token 摘要。Dolly 的旧类内 qid 与新 all 模式的 qid 不同，用原始 ID 和文本核对。既往八类切片、SpecBench 480 条及被反复使用的句式已经参与开发判断，明确作为开发证据，最终确认另留独立输入。

### 2. 先复现 Jason 输出修正

建议先选共同已有的 Qwen3-8B，BC30、k=1、branches=[14] 作为来源对齐候选，以实际本地缓存和核实后的配置为准。已有办公室结果的 [4]、64 token 属于另一设置，单独比较。固定权重、模板、EOS、生成长度、dtype、后端和贪心验证，在同批评测 prompt 上比较无校正、独立 bias 和完整冻结 K。

先报告拟合标签的误差与开发集 cov@14，再报告完整解码的提交数、接受数、调用数和 τ。Jason 拟合脚本会在选正则及内部 report 后，用全部采集数据重拟合最终 K。最终 K 的解码评测使用独立 prompt，内部 report 分数按对应的旧拟合子集解释。[拟合脚本](https://github.com/jasonchengqs/speculative_mtp/blob/534a79f9c00854895d514a90a727575c0580dc10/denoise_mtp/scripts/fit_frontier_gain.py#L79)

### 3. 分开检验类别覆盖与 thinking

在来源方法建立后，采用下列拟合条件，统一在关闭 thinking 的同批输入上评测。开／关均显式写出，先验证实际模板产物。其余分词处理沿用同一版本。

| **拟合数据** | **拟合 thinking** | **推理 thinking** |
| --- | --- | --- |
| creative_writing 单类别 | 关 | 关 |
| creative_writing 单类别 | 开 | 关 |
| 按原始类别分层的多类别 | 关 | 关 |
| 按原始类别分层的多类别 | 开 | 关 |

四个条件分别采集 states、选择正则并拟合 K。建议将输入行数预算固定，每条生成上限相同，另报告真正进入拟合的配对数、长度和 token 数。若配对预算差异足以改变结论，补同等配对数的敏感性对照。原始顺序前 N 条的 all 模式可用于来源复现，多类别效应对照使用预先固定的分层抽样。

此四格回答拟合类别覆盖和拟合模式在固定非 thinking 推理下的作用。若需区分“模式不匹配”与“thinking 拟合本身的影响”，在选定数据条件上再补推理 thinking 开的匹配对照，并重算该模式自己的无修正基线。每次改变 context 顺序、截断或 EOS 时单列处理因素。

### 4. 比较修正空间与泛化

先在相同 fit/dev/eval 划分和非 thinking 条件下比较上节三种输入／输出候选。纯空间机制比较采用同信息输入重算，效率比较采用可部署的输入前修正与输出后修正，并计入各自成本。

对选定方法区分同类别未见 prompt、拟合中未见类别、不同数据集。多类别拟合覆盖了某一类别时，该类别上的新 prompt 是类内留出。要主张未见类别迁移，另做留一类别拟合。先固定已校准 K 再读评测集，不在跨数据集评测端重选。

## 判读、预算与后续状态估计

主比较报告逐 prompt 的完整 K 相对独立 bias、相对无修正的配对收益，至少包括接受数差、合计 τ 及其配对差。报告按 prompt 重采样的 95% 区间和逐类别结果，区间覆盖零时保留未决。确认前登记主要比较、重复的校准抽样、样本量与判断阈值，功效与预算依据小规模方差和可用资源确定，避免把四格中的最优开发结果当成独立确认。

T0 审计覆盖完整实际提交链，失败时先排查实现。独立 AR 一致率按既定规则单列诊断。只有实际量过的耗时才报告速度，计入额外前向、hook、投影和同步成本，校准成本单列。hidden MSE 和 cov@14 是机制指标，不能替代接受或速度收益。若配对方向不稳、拟合收益不能迁移，先复核数据、位置和信息条件，保留负结果。

本轮采用两条研究主线。状态估计线按零阶参照、原始差分 EMA、固定增益 Holt、协方差增益卡尔曼建立递进对照。来源校准线按 Jason 核对与复现、拟合分布与模式、输入／输出干预比较推进。各线先固定表示、校准与验收，再在有证据的配置上组合。更高阶状态在一阶趋势获得可复核证据后讨论。已登记的模型、数据与 GPU 预算见第六、七章；其余正式确认与状态估计预算仍在执行前登记，未获授权的同步和发布不自动进行。

# 五、和开发计划的关系
要检验的主张和已经改变判断的结果在第二章。代码、目录和工程验收在 [开发计划](<开发计划 Development Plan 去噪推测解码.md>)。已归档 Run 的计数不因后来的解码器改动而回签。

# 六、2026-11-20 GPU 有界先导（已获本次执行授权）

本轮从来源校准线启动，先核验端到端可执行性。复用本地 Qwen3-0.6B；8B 原缓存不存在，不下载权重。固定 Jason 534a79f9c00854895d514a90a727575c0580dc10 的完整源码快照，使用现有 CUDA／torch，BF16、eager、BC30、单 mask、branches=[14]、贪心、thinking=false。此规模与 8B 来源结果分开解释。

使用已有 SpecBench 原始文件，固定种子 20261120，每类打乱后轮流选取 28 条，按顺序分为 fit 16／dev 4／eval 8，各组 prompt ID 和文本不得重叠。每条最多 32 个新 token，输入最多 256 token；保存原始行号、类别、模板参数及实际 token ID。该公开开发数据只用于工程先导，不宣称独立最终确认。

先采集无修正 states，在 fit 拟合完整仿射 K，以 dev cov@14 从 {0.1,1,10} 选择正则；仅 fit 拟合独立 bias。先导不将 dev 合入重拟合，eval 不参与选择。三臂为无修正／独立 bias／冻结完整 K，在同一 eval prompt 上运行，在线更新关闭。记录同位置配对、hidden MSE、开发集 coverage、完整提交链 argmax 审计、实际输出数、decode calls 和实际保留的 accepted 数。分别报告来源未截断计数，避免末轮预算截断污染接受数。单次仪器化时间仅作开销记录，不给出稳定速度结论。

本次单进程、GPU 0，硬上限 10 分钟，不启动矩阵或并行 GPU 作业。开始前核对可用显存；OOM、非有限 K、完整提交链审计失败或时间超限即停止该阶段并保留失败证据，不自动扩模型、重试或下载。快照、配置、数据清单、K、逐 prompt 回执和源码摘要保存在外部 Run 目录，不进入 Git。通过后再根据配对方差与实际成本登记较大模型、Dolly 类别／thinking 因子与输入／输出实验预算；此先导不代表这些实验已执行。

## 先导结果与下一阶段

本轮先导已执行，详见[运行记录](<运行记录 Run Record/2026-11-20 前沿hidden校准GPU先导.md>)。295 对拟合、81 对开发，所选正则为 1。8 条评测中无修正／bias／完整 K 的 decode calls 为 160／163／153，τ 为 1.5500／1.5215／1.6209。完整实际提交链共 1,401 token 的 argmax 审计通过。完整 K 与无修正输出有 1/8 条不完全一致，保留诊断；没有执行独立 AR 或速度确认。这是来源快照和 0.6B 的工程先导，不能转写为本仓实现、8B 复现或 Kalman 收益。

4B 扩大阶段采用已有 Qwen3-4B、相同三臂和信息时点，seed=20261121，预先保存分层的 fit48/dev8/eval16 清单，每条最多64个新token、输入上限512，保持组间ID和文本隔离；历史开发数据继续标为开发验证。单进程、GPU0、阶段上限20分钟；遇OOM、T0失败、非有限参数或超时即停止并保留事实。该阶段已在第七章的新授权下完成，原始Run另行冻结，没有覆盖先导；结果见[本轮运行记录](<运行记录 Run Record/2026-11-20 六十分钟前沿校准研究.md>)。

在 4B 开发验证后，依次推进三项研究：① Jason 提出的单／多类别与 thinking 模式匹配；② 对齐信息时点的输入／输出修正；③ 固定表示和校准后的差分 EMA／Holt／轨迹 Kalman。正式确认的独立数据、重复校准抽样和计时预算在开发方差与成本得到后登记；保留验收规则作为单独研究维度，当前贪心合同固定。

# 七、60 分钟自主研究推进（2026-11-20）

本次用户明确授权目标模式、60 分钟研究预算及 GPU 执行，常规方案自行决定。起点为 UTC 19:10（北京时间 03:10），最迟 UTC 20:10 停止派发并完成收尾。沿用 research-plan 和 python-environment 技能、当前唯一计划、三份算法说明及既有来源固定版本。工作不涉及量化项目，不改共享依赖、CI 或凭据，不下载模型，不将研究脚本和制品加入 Git。

| **时间预算** | **研究问题与执行** | **判据与产物** |
| --- | --- | --- |
| 0–8 分钟 | 核对资源与缓存，冻结源码、权重、数据和计划；把 0.6B 先导扩为可保存检查点的外部编排 | 先验证输入划分、模板、位置配对和完整提交链审计；失败留回执 |
| 8–28 分钟 | 4B：fit 48／dev 8／eval 16，输入最多 512、新 token 最多 64，三臂无修正／独立 bias／完整 K，BC30、非 thinking、贪心固定 | 以逐 prompt 配对 τ、实际接受、调用数与 T0 决定是否扩大；开发 cov@14 选 λ∈{0.1,1,10}；eval 不选参 |
| 28–45 分钟 | 来源校准分布与模式：优先使用离线可用 Dolly，固定单／多类别 × thinking 开／关，推理统一关；Dolly 不可用则采用 SpecBench writing／分层多类作为代理并明确边界 | 同等 prompt 预算、组间文本与 ID 隔离，保留实际类别及每格有效配对数；默认用 0.6B 完成有界探索，避免挤占 4B 主比较 |
| 45–53 分钟 | 以冻结 K 做独立输入／类别迁移或第二校准划分；离线补 bias、线性配对打乱等机制对照，核对输入恒等映射与输出残差的区别 | 避免将校准 MSE 改善直接解释为解码收益；缺少独立来源时不主张跨数据集泛化 |
| 53–60 分钟 | 汇总配对区间、类别差异、输出一致性、实际开销和失败；更新研究计划与日期运行记录 | 给出继续／暂停判断，明确未执行项；保留一份当前计划与一份本轮日期记录 |

4B 主阶段用 seed=20261121，SpecBench 中排除前一轮 28 条 prompt，fit／dev／eval 内按原始类别轮流抽取，原始文本不重叠。阶段硬上限 20 分钟；单 GPU 进程，不同时派发多个模型任务；显存不足则记录并转向 CPU 分析或 0.6B 有界研究，不杀其他任务。OOM、非有限 K 或 T0 失败终止对应阶段，不掩盖负结果，不为凑时间反复重跑。所有进程由总截止时刻和阶段硬超时共同约束，阶段间采用新 Run 保留源码与实际配置摘要。

优先级为完整且可复核的比较，其次覆盖研究维度。最终计时若存在其他 GPU 负载、固定执行顺序或仪器化干扰，就只报告真实耗时与资源，不宣称可靠速度提升。采样、beam、Holt 和轨迹 Kalman 的实现不在这 60 分钟里仓促加入；其接口与后续验证条件保留在计划中。

## 执行前修订：Dolly 四格与跨数据集迁移

在 4B 主阶段运行期间取得 Dolly 官方历史 JSONL：databrickslabs/dolly 的 5f9bfba300181fb35854d64c8bf1180ea75105f7，15,014 行、约 13.2 MB。原始数据进入本轮外部 Run，不改仓内正式数据；与 Jason 所用 HF 快照的逐字一致性未核实。加载库的失败由先导入 torch 的启动顺序规避，未安装或替换包。官方文件移除提交与首次失败下载均留存，不将失效地址记成数据来源。

根据 4B 当前实际进度，第二阶段改在同一 4B 上完成四格，避免模型不同混入模式比较：seed=20261122，creative_writing 单类 fit32/dev8、多类别分层 fit32/dev8；各自同一 prompt 分别以 thinking 开／关采集，共 4×40 条。eval16 按原始八类轮流抽取，拟合和开发均排除，推理 thinking 固定关。输入最多512、新 token最多64，BC30、贪心、dtype、后端与前沿位置合同固定。每格独立选择正则和拟合 bias；9臂为无修正及每格的 bias／完整 K。再加载第一阶段 SpecBench 冻结 K／bias，对同批 Dolly eval 作跨数据集迁移，不在 Dolly 端重选 SpecBench K。执行臂顺序按 prompt 轮换。

第二阶段最晚运行20分钟，受60分钟总截止约束。记录实际输入模板、原始类别与行号、有效配对数、thinking 输出中生成到的格式及截断情况。四格同等 prompt／生成上限不保证同等配对数，先报告其差异，再用离线同配对数拟合检查机制指标的敏感性。评测集不用于挑选四格赢家，结果为探索性方向比较。

## 预登记：独立校准重复与机制检查

在四格主比较结束、T0 通过且剩余预算足够时，第三阶段以 seed=20261123 从 SpecBench 中排除此前先导和 4B 主实验全部输入，抽取新的 fit48/dev8/eval64。新 K 和新 bias 仅由新的 fit/dev 形成；在同一 eval64 上同时评测无修正、新 bias／K、第一阶段冻结 bias／K，共5臂。输入和生成上限512／64、BC30及非 thinking 贪心合同不变。阶段最多16分钟并服从总截止；用于检查第一次校准的正向信号能否在另一批输入和另一份校准中保留，仍标作开发重复验证。

机制检查不使用评测结果选新正则。以各格已经选定的 λ，补 fit 特征行固定随机打乱的 K 作为配对信号负对照；在原始 fit 中取相同配对数重新拟合四格 K，比较固定评测 probe 的 cov@14 与 MSE。该离线结果不等于真实解码收益。输入同位置恒等关系 m+(E[b]-m)=E[b] 作数值合同核对，明确已知 bonus 的查表值与未知后继 token 预测不同。输出 oracle 使用下一轮真实 root hidden，仅是延迟标签上界，不能冒充当轮可用算法。

机制补充对照预登记：除特征行打乱外，增加仅逐维斜率加截距的对角仿射 K，使用第一阶段已经选出的 λ，不另选参。其参数量较完整跨维矩阵少，用来区分跨维映射与各维独立缩放；它是额外机制基线，不称为 Kalman 对角增益。若时间足够，在第一阶段相同留出输入上重跑无修正／原完整 K／bias／打乱 K／对角 K，明确属于已看过的开发输入上的解释性对照。全矩阵、打乱、对角和 bias 的拟合标签与位置合同一致。

## 预登记：有仪器化的实际耗时对照

若前述阶段完成且剩余预算不少于8分钟，再用重复验证预先冻结的 eval64 中前8条输入，固定生成64 token、stop_on_eos=false，比较 AR、无修正 ESP、原始 bias、原始完整 K。固定模型、dtype、eager、模板、输入、BC30，每臂先预热1条，再在每条输入上轮换4臂并重复3轮，解码器对象跨问题复用。K 固定来自第一阶段，不据计时结果选择模型。

所有生成保留完整实际提交链 argmax 审计；计时关闭离线 states 导出，保留必要校正 hook 与审计，其开销计入。前后 CUDA 同步，包含 prefill 和 decode，排除模型／解码器初始化。记录时间戳、实际 GPU 利用率和显存；共享 GPU 负载不受本任务控制，结果只称本机仪器化耗时观察，不能冒充纯生产 benchmark。AR 输出一致率只作诊断，不启动已停止的 BF16 算子机理排查。吞吐与端到端时延分别写明单位和比较口径，接受数／τ 不替代时间。

## 预登记：短校准向长生成迁移

若上述核心对照完成且总预算还剩至少10分钟，补一个不重新拟合的长生成检查：seed=20261126，排除本轮全部校准、开发和评测输入，选择8条 SpecBench writing 与8条 Dolly creative_writing，使用第一阶段 SpecBench 的冻结完整 K／bias 和无修正三臂。生成上限从64提高至256，输入仍最多512、stop_on_eos=true，其他合同固定，按 prompt 轮换臂顺序。它回答64-token校准在更长自然生成中是否仍有接受收益，不能作为全部任务的长序列结论。

每次调用按开始时已生成长度分为0–63／64–127／128–191／192–255区间，保存调用数及该调用实际提交数；跨边界的调用整体归开始区间。报告实际达到每区间的 prompt 数，避免忽略提前结束及幸存者选择。上限10分钟，并服从总截止，不覆盖既有 Run。

## 数据身份核对后的优先级修订

固定 HF revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a 后核对：HF文件15,011行，与官方GitHub历史文件15,014行不相同，历史四格实际使用的96条输入有83条在HF中逐字匹配。历史四格仍是有效的历史数据版本实验，不能回签为HF数据复现。

因此，在独立SpecBench重复后，优先补HF固定版本的同规模四格重复：seed=20261127，单／多类别各fit32/dev8、相同prompt分别thinking开／关、eval16、同样11臂。HF抽样排除历史四格全部原始文本，保存原HF全局行号。该比较换了数据版本与校准抽样，不能把两轮差异全部归因于版本改动；它用于检查核心方向在另一份真实HF输入上是否保持。使用相同4B、512／64长度、正则网格、BC30和T0合同，阶段最多13分钟，服从总截止。机制和实际计时保持核心优先；长生成顺延，只有预算仍足够才执行。

## 接入后续状态估计的观测合同

本轮校准线额外提供了延迟可观测残差：在bonus所在位置p，本轮有u_p=[E[b_p]-m_p;1]及probe hidden，下一轮到达同位置真实root hidden后，才能取得r_p=h_p^true-h_p^probe-u_p K_out。跟踪该残差的尺度或相关性，是后续研究候选；本轮没有将其实现为Holt或Kalman，也没有据静态回归收益证明动态增益有用。

ESP的输入mask按每枚正式提交token递推；前沿校准标签则是按前向获得、位置可能跳跃的观测流。若趋势速度以每token定义，必须保存真实位置间隔Δp并相应调整预测步长，不能把每一对前沿标签当作相邻token。已有prefix与accepted节点的真实hidden是否补齐规则观测、bonus hidden的一轮延迟、临时draft状态的丢弃时点，均先形成数据合同再实现。状态估计线先关闭校准层建立EMA／差分EMA／Holt的可比基线，再比较具有明确F/H/Q/R/P来源的Kalman；输入／输出干预仍以信息时点和额外前向成本对齐。

机制阶段再补一个有界离线诊断，服务后续状态估计：在每个prompt内部计算静态K校正后的残差r_p，检查相邻前沿标签的相关性及真实位置间隔。仅用第一阶段fit估计上一残差预测下一残差的标量系数g，夹到[0,1]；另固定alpha=0.1的残差EMA作为候选。评测时第i对只用已在当前前向到达的第i-1对及更早残差，首对预测为零。比较静态K、上一残差预测、残差EMA的固定路径MSE与cov@14，不用eval调g或alpha。此项不运行新的动态解码，不将离线改善写成接受或Kalman收益。

长生成检查的数据身份在执行前统一到已核对的HF版本：Dolly选样来自bdd27f4固定文件，并按文本排除历史与HF两轮全部输入；SpecBench继续排除先导与两轮4B全部输入。仅在计时结束后总预算仍剩至少10分钟时启动，阶段上限下调为8分钟，为最终验证与记录保留余量。

长生成v1在选样时失败，尚未执行生成：SpecBench共有13类，其中writing只有10条，已在前序划分用尽。当前剩余输入属于translation、summarization、rag、qa、math_reasoning五类。保留v1失败回执，不覆盖输入或Run。修订后的v2选择8条尚未用过的SpecBench summarization，以及8条尚未用过的Dolly HF creative_writing；它是摘要／写作两种场景的长生成探索，不能回签为原writing方案。其它冻结K、长度256、T0和阶段8分钟上限不变，仍受总截止约束。

# 八、本轮判断与下一步研究

实际结果统一见[六十分钟前沿校准研究](<运行记录 Run Record/2026-11-20 六十分钟前沿校准研究.md>)。本轮验证来自固定Jason源码快照，不是本仓科学宿主的GPU验收；没有新增Holt或Kalman实现。以下结论按研究问题更新，逐次运行事实不在本章复制。

| **研究问题** | **本轮证据与当前判断** | **下一步** |
| --- | --- | --- |
| 同位置final hidden残差校准 | 4B两份校准及新留出输入中，完整K的验收收益优于无修正和独立bias；打乱配对基本失去增量 | 保留静态完整K为来源基线，接入自有宿主前核对位置、权重身份、特征协议与完整提交链 |
| 校准类别与模式 | 历史Dolly和固定HF的新抽样都显示非thinking推理更受益于非thinking校准；单类和多类区间重叠 | 校准与推理模式显式匹配；类别覆盖仍是待检验因素，不将all或多类别直接视为更优 |
| 分布迁移 | 冻结SpecBench K在Dolly两份输入中仍有正向验收信号；拟合预算与Dolly不同 | 新一轮先对齐校准prompt数／有效配对数，并做至少3份校准抽样；不在迁移端重选K |
| 输入／输出修正 | 当前只验证输入差值到输出hidden的映射；同位置输入还原是恒等关系 | 分开做输入前修正、已知bonus后的输入重算、输出后修正；对齐目标位置和信息，计入额外前向 |
| 残差动态反馈 | 简单上一残差预测和残差EMA未显示候选覆盖收益，即使MSE略降 | 先复核LM读出相关指标和观测时序，不以MSE单独推进更高阶；保留具体负结果 |
| 验收效率 | 完整 K 在固定树形和验证合同下提高实际接受量及 τ；共享电脑的墙钟计时不作为收益证据 | 比较配对接受数、每次验证的接受量和提交量；后续不安排墙钟测速来选方法 |

## 当前主指标与比较合同

本节是后续实验和汇报的现行口径，覆盖前文涉及实际速度判据或扩大计时的未来建议；第七章既有计时协议及日期运行记录保留历史原义。

- **实际接受数 A**：最终保留在输出中的 draft token 数；排除 EOS 或长度截断后丢弃的候选，不计 bonus 和 prefill 首 token。
- **每次验证的接受量 A/C**：C 为 decode 验证调用数，用于比较每次验证接收多少 draft。
- **每次验证的提交量 τ=S/C**：S 为 decode 阶段实际提交 token 数，包含保留的 draft 和 bonus，不含 prefill 首 token。它表示每次调用推进多少 token，不是 token/s，也不称为实际加速比。

主比较固定同一批 prompt、模型、生成 token 预算、树形／BC、mask 数量、模板与目标解码规则，报告逐 prompt 配对差、区间和实际生成长度。输出预算相同时，接受更多、用更少验证调用推进同样多 token，是当前的收益判据。自然 EOS 导致输出长度不同时，同时报告 A、C、S、A/C 和 τ，不能仅按总接受数或总调用数排名。τ 采用合计 S／合计 C，逐 prompt 统计另列。

完整提交链 argmax 审计仍是正确性门。MSE、cov@14 用作机制诊断，时间仅用于作业预算、超时和运行事实记录；现有共享负载下不再据计时作方法优劣或加速结论。

## 数据和评价先行

SpecBench原始文件有13类，其中8个类别各10条，其余5类各80条。按原始轮流抽取再切分会耗尽小类别，使后续eval组成变化。本轮真实清单和结果保留；以后先为每类预留eval，再抽fit和dev，先检查配额和文本隔离，最后载入模型。比较类别政策时保留原始类别、行号、版本和长度分布。HF与GitHub历史数据不是同一文件，不能用同名或近似行数代替内容核对。

当前SpecBench和两份Dolly输入均属于开发研究，已经参与方案判断，最终确认应再锁定未参与选择的评价输入。后续候选规模为分层fit32/dev8/eval32、至少3个校准种子；它是下一轮设计起点，新的GPU时长和正式判断阈值在执行前另行登记。

## 状态估计与工程承接

状态估计线先以输入embedding建立零阶EMA、旧差分EMA与Holt的最小对照：共同初值、观测时点、树、读出和贪心合同固定，先关闭校准层。检查alpha=1等退化关系和实际mask时序后，再让Holt改变位置与趋势；随后才比较有明确F/H/P/Q/R及校准来源的Kalman。若观测来自前沿标签，按真实位置间隔预测，保留一轮hidden延迟。任何新观测只能改变以后要生成的候选，不能回改已算好的旧树。

组合状态估计和静态K时，先检验mask改变是否导致特征分布变化；需要重新拟合时，以相同校准预算独立登记。不要把表示、修正空间、状态阶数、拟合数据和验收规则同时改变。采样及beam规则另设合同，本轮仍只覆盖贪心。

来源基线接入本仓的实现职责、接口和验收进入[开发计划](<开发计划 Development Plan 去噪推测解码.md>)及对应设计后再开发。当前没有修改科学源码，也没有运行本仓全量CPU测试或远端CI；本轮的数值与提交链检查只支持所冻结的来源运行。

## 60分钟执行收尾

本轮GPU已停止派发：三臂基线、两份校准的独立输入重复、Dolly历史／HF两轮四格、冻结K迁移、配对与对角机制、同配对数诊断、残差预测诊断、三轮同步计时和256-token自然生成检查均已取得对应结果。长生成writing配额失败保留，另开摘要／写作v2完成。合计7个成功阶段、1个生成前失败Run；科学运行事实及完整限制见日期记录，不回签为本仓实现或独立AR无损保证。

长生成中，原64-token校准的冻结K仍有正向验收信号，但不同臂输出及EOS位置有差异。简单残差EMA只改善部分MSE而未改善固定路径候选覆盖，暂不据此推进更高阶。下一轮依本章的问题与数据合同继续，正式预算在新执行前登记。
