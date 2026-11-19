**去噪推测解码 研究计划**
更新日期：2026年11月20日。这是唯一中文研究计划。本文对照当前代码写算法，并把已跑实验收成一张结论表。逐次预注册、旧英文计划和运行日志从 Git 历史与 [运行记录](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/8f25a61515cf331e3a3cf3ba0dfbc66f217e315c/docs/%E8%BF%90%E8%A1%8C%E8%AE%B0%E5%BD%95%20Run%20Record/INDEX.md) 取证，不在这里重写。工程待办见 [开发计划](<开发计划 Development Plan 去噪推测解码.md>)。

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

2026年11月20日，由 EMA／Holt／卡尔曼的学习讨论，以及 Jason 对拟合数据和 thinking 设置的反馈形成这组问题。研究方向是观察位置、趋势与预测残差是否能帮助免训练起草，并检验收益对校准数据分布的依赖。下述方案为拟议验证，实际模型、数据规模、计算预算与运行安排在执行前登记。前两章的旧协议和历史结果保留原义。

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

本轮采用两条研究主线。状态估计线按零阶参照、原始差分 EMA、固定增益 Holt、协方差增益卡尔曼建立递进对照。来源校准线按 Jason 核对与复现、拟合分布与模式、输入／输出干预比较推进。各线先固定表示、校准与验收，再在有证据的配置上组合。更高阶状态在一阶趋势获得可复核证据后讨论。最终模型与数据规模、GPU 预算、运行日期、同步和发布状态均待实际执行时确认。

# 五、和开发计划的关系
要检验的主张和已经改变判断的结果在第二章。代码、目录和工程验收在 [开发计划](<开发计划 Development Plan 去噪推测解码.md>)。已归档 Run 的计数不因后来的解码器改动而回签。
