**去噪推测解码 研究计划**
更新日期：2026年11月4日。这是唯一中文研究计划。本文对照当前代码写算法，并把已跑实验收成一张结论表。逐次预注册、旧英文计划和运行日志从 Git 历史与 [运行记录](<运行记录 Run Record/INDEX.md>) 取证，不在这里重写。工程待办见 [开发计划](<开发计划 Development Plan 去噪推测解码.md>)。

# 一、当前算法与数学
行号对应当前 `origin/dev` 上的源码。正式办公室比较用的是 ESP 轨迹加 observation 混合；卡尔曼类是同一解码循环上的其它 mask。

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

采样分布保证仍未完成。BF16 相对独立 AR 的算子数值机理排查保持停止。本文不另排下一次实验。

# 四、和开发计划的关系
要检验的主张和已经改变判断的结果在第二章。代码、目录和工程验收在 [开发计划](<开发计划 Development Plan 去噪推测解码.md>)。已归档 Run 的计数不因后来的解码器改动而回签。
