# SD研究计划：固定数据复现与mask方向对照

中文正本，交接状态更新于2026-10-18。[English translation](<Research Plan Denoising Speculative Decoding.md>)。正式收尾报告已随PR #50合入dev，以下保留原报告正文，不重做实验或重签证据。

> 当前结论：数学自检239,137/239,137通过；BC30/60 τ=1.563377504/1.647301264，比论文低约0.067/0.093，未复现论文目标。
> 8B三格已完成；其余七格范数扫描和11格独立配对未追加执行，所有旧排班保持暂停。
> 后续如何开题及clone后的证据边界见[异机交接](handover.md#sd-wrap-up-20261018)；本轮仅补交接，不授权启动下一阶段。

<!-- SD_CLOSEOUT_20261016_START -->
<a id="sd-stage-closeout-20261016"></a>
## 复现阶段总结（给 Enmao）

### 这次在验证什么

目标是核对 ESP 论文 Table 1 的接受效率。模型把文本分成小片段，称为token（可以是字、词或词的一部分）；先同时提出若干候选片段，再验证并提交能接受的片段。**BC10、BC30、BC60 是每轮草稿与验证的 token 计算预算**；预算越大，容许探索的候选越多。本轮 Qwen3 三个主配置均为单 mask，分别最多4、14、29个草稿节点，采用动态树和剪枝。

**τ 表示一次验证平均提交多少个新 token。**本报告主值为所有480条的提交token总数除以验证调用总数，首个prefill token单列，不混入分子分母。例如 τ=1.563，意思是平均一轮验证提交约1.563个token。本轮只统计接受和调用次数，没有做速度测试；τ不是实测加速倍数。另保留逐prompt等权、逐类别汇总后等权两种口径。

本轮明确标为**非作者官方复现**：没有可核验的作者原生实验入口，使用本仓库已标注实现；HF第三方代码也没有当作作者官方。固定SpecBench全480、Qwen3-8B、BF16计算精度、T0（不随机采样，每步选得分最高项）、256最大新token/EOS停止、λ=0.1 updating mean、动态树加剪枝；8B仅用于Table1复现。数据和prompt不调整，不测速，不继续算子数值机理排查。

### 8B Table 1：已经得到什么

| 配置 | 本轮 pooled τ | 论文对照 | 本轮减论文 | 状态 |
| --- | ---: | ---: | ---: | --- |
| BC10 / 单mask | 1.399047031 | 本轮未指定该格目标 | — | 全480成功 |
| BC30 / 单mask | 1.563377504 | 1.63 | -0.066622496 | 全480成功，低于目标 |
| BC60 / 单mask | 1.647301264 | 1.74 | -0.092698736 | 全480成功，低于目标 |

BC10/BC30原数值来自[BC10结果](../studies/esp_table1_nonofficial_revalidation/docs/bc10_checkpoint_v1.json)与[BC30结果](../studies/esp_table1_nonofficial_revalidation/docs/bc30_checkpoint_v1.json)，Run分别为`80ff7a41958d8134`、`88d13ec4ecf0023c`。本轮BC60为`a8e6bba1dd1d80e2`，已完整验收。三格正式结果及逐条来源统一见[复验正式证据](../studies/esp_table1_nonofficial_revalidation/docs/repeat_evidence_v1.json)，`cells[].rows`和`token_audits`保留逐条AR首差异位置与输出摘要。BC30另两种汇总为逐prompt 1.593232237、逐类别 1.563387272，均低于1.63；汇总口径差别不能据此宣称目标已复现。BC60另两种汇总为逐prompt 1.669498033、逐类别 1.649693144，也均低于1.74。三个BC与旧实现逐prompt的接受数、调用数和τ各480平，输出/AR摘要各480一致。

### 4B：收益不依赖 e2−e1 这个特定方向

在固定480条、4B BF16/T0、BC30下，既有完整方向对照如下；这里的偏置作用于草稿预测用的占位向量，输入文字不变。范数表示向量的大小，方向表示它指向哪里；e2−e1是输入头两个token向量表示之差。随机组保留相同初始大小，只替换方向。

| 对照 | pooled τ |
| --- | ---: |
| 原固定 e2−e1 | 1.507777653 |
| 同范数随机方向三种子平均 | 1.494473978 |
| 零偏置 / 输入零阶 | 1.422420206 |
| 内容初始化 | 1.364829262 |

来源：[完整方向证据](../studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json)中`cells[*].totals`；[逐格来源、480条配对及解释](research/4b-full-direction-controls-2026-10-16.md)。六格共485,860个提交token的数学自检全过。三个随机方向分别为1.495188134、1.464349086、1.523884714；全部高于零偏置，其中一个高于原方向。

**现有证据支持：同范数随机偏置也能起作用，提升不是 e2−e1 独有。**但不能把它说成“方向完全无关”或“范数已经被证明是唯一原因”：随机种子之间仍有差异，内容初始化的结果更低，而且content同时改变初始化和后续速度更新行为。未完成的范数倍率扫描没有完整曲线；仅template0.5×首格全480，τ=1.471389646，80,940个token数学门全过。该格的collect入口问题已恢复、原始SHA未变、没有重跑execute，见[恢复证据](../studies/qwen3_4b_bf16_bias_norm_continuation/docs/first_cell_recovery_v1.json)。其余七格按20:55指示停止排队，全部已有资料保留。

原五方法加AR的完整4B配对已有结果：BC30输入一阶 1.511957584，输出零阶 1.433178245，输出一阶 1.435539512；其余方法、BC60及逐prompt胜平负见[原五方法结果](../studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json)。固定方向对照和原生更新速度的配方不同，不能直接混成同一个实验排名。后来准备的11格独立复验只make、未execute，现不启动。

### 已核验什么，仍有哪些差距

正确性门是**每个提交token等于本次验证前向对应logits的argmax（得分最高的候选）**，覆盖首token、owner和哈希/完整提交链。BC10核验79,757个、BC30核验79,762个、BC60核验79,618个，合计239,137个全过；三格原生Run与Study process均成功。独立AR是普通的逐个token生成对照；与它逐token一致的比例只作诊断：BC10为188/480，BC30为175/480，BC60为185/480。BF16树验证与单条AR前向形状不同，轨迹可能分叉；这里没有隐藏该差异，也不把AR失配当成数学门失败。不同方法输出长度可能不同，配对计数不能当等长速度比较。

固定输入、模型快照、原生来源/计划冻结、Run配置、raw审计与汇总已经核对。CPU完整回归618项全过，另加十项显式BF16且权重实际tied的微型模型检查全过，覆盖五方法×两预算、全部提交token审计与token损坏拒绝；这些是软件门检查，不替代真实4B/8B测量。记录见[CPU验收证据](../studies/qwen3_4b_bf16_paired_revalidation/docs/cpu_readiness_v1.json)。

**未复现的部分是论文目标数值，不能因为自检通过就说论文复现成功。**BC30仍低0.066622496，BC60低0.092698736；三种已报告汇总口径均未达到对应论文目标。已知可能原因包括尚未核验到完整的作者输入/模板与停止合同、非官方实现的树构建/剪枝细节可能不同，以及论文τ计数/权重合同仍有未知处。这些是解释候选，没有被本轮证明为因果。已完成的CPU计数对账和三种汇总没有消除差距，详见[输入与聚合对账](research/esp-table1-cpu-contract-reconciliation-2026-10-16.md)和[缺口登记](research/esp-table1-protocol-gaps-2026-10-16.md)。

尚未解决：论文数值差距具体来自哪项合同；偏置范数和方向各贡献多少；为何输出零阶/一阶仍落后于输入一阶、输出速度项收益很小；不同输出长度和轨迹对跨方法比较的影响。这些问题保留为未解决事项，**本总结不提出下一阶段方案**。

### 收尾执行状态

按Enmao20:55指示，只有原有BC60被允许继续，现已自然完成；七格等待队列64880与入账器50944已退出，两个临时启动器保留原文/SHA并改为`.disabled`，不再自动接卡。源码、结果、权重和缓存未删除。最终来源与数学门复核完成。已删除已合dev的`experiment/sd-nonofficial-repeat-norm`本地/远端分支；17份未执行旧准备源码逐SHA归档，159个生成字节码缓存清理，所有实验结果和权重缓存保留，旧数据工作树保留在原提交的detached状态。交付流程为一个里程碑PR合dev，当前未合工作分支在合入前保留；下一阶段等待Enmao评阅，不执行旧章节中的后续实验安排。
<!-- SD_CLOSEOUT_20261016_END -->

## 一、当前范围、状态与执行优先级

以下旧排期为历史登记；20:55之后以本页收尾总结为准，后续待执行实验全部暂停。

本计划按Enmao 2026-10-16 12:44/12:53最终指示更新：停止追找官方ESP入口，现有实现统一标为**非作者官方复现**。当前研究问题为固定SpecBench480上Table1 τ缺口，以及输入偏置的范数是否比特定方向更能解释接受收益；不新增Kalman/新方法、不改prompt、不测速，仅BF16/T0。

现状：既有旧8B非官方Table1四格完整且数学门全过，BC30/60为1.563378/1.647301，未达1.63/1.74。4B五方法加AR、六格BC30方向各480均已验收。三个同范数随机方向均值τ=1.494473978，对比e2−e1的1.507777653、零偏置1.422420206。Enmao据此提出“方向基本不重要，同范数偏置是主因”的当前机制假设；现有证据不支持e2−e1唯一性，但三个随机值1.464349086–1.523884714仍有波动，尚不能证明任意方向等效或确定范数曲线。完整数值和界限见第九章及原正式JSON，不用新的说法改写旧结果。

执行顺序：A，按最新“开跑复原”指示进行一轮新的8B（若主机RAM准入不足则可运行的最大既有规模）全480非官方Table1独立复验，BC10/30/60单mask，λ0.1 updating mean、dynamic/pruning、256/EOS、原seed123；旧完整结果复用作基线，显式新version/Run，不覆盖或重签。B，4B BF16 BC30做偏置范数扫描，倍率0/0.5/1/2；原e2−e1与随机种子0/1/2均预先保留，0与1倍复用已有完整结果，仅新跑四方向×0.5/2共8格。C，从τ曲线、逐prompt接受/calls/τ胜平负、实际偏置范数和输出长度检验机制；不择优种子或改主prompt。BC60范数泛化暂不扩组。

GPU政策已改为：原则上不空闲，有用实验可执行；**同一时间仅一个GPU任务，准入为所选任务已测进程树峰值主机RSS加2GB主机RAM**（执行保守取2GiB），同时实际GPU空闲且显存能容纳模型/工作集。旧“显存峰值加2GiB”门废止，不把CPU进程逐一登记/SHA作为门，不杀其他任务，不跑完整约32GiB量化训练。历史排班快照：2026-10-16约13:02实际GPU由量化Python占用，主机available约19.99GB；本项目等待该GPU任务自然结束后只启动一条串行链，不伪称已跑。8B旧实测峰值进程树RSS为17305047040字节，加2GiB需19452530688字节；这是采样峰值，不保证新路径不增长，运行中记录资源且保留失败。不得把旧VRAM差661MiB当新政策的主机RAM阻塞。

唯一有效项目计划为本文件和[英文对应版](<Research Plan Denoising Speculative Decoding.md>)。所有60份子计划/专项协议/评审/历史v6整合到第十一章独立章节，旧单独文件按12:53明确授权git rm；来源字节、Git blob、换行及新锚点见归并清单。历史“必须独立PLAN”“先AR exact”“追作者入口”等原文只作历史记录，不能覆盖本章。迁移不重签原Run，不把当前计划内容当原冻结字节；新Run记录不可变的项目计划commit/blob和章节。正在运行的其他任务不受改名影响。

文献与结果阅读深度、既有局限见后续章节。夜间PR22–40的完整结果/来源/对账均纳入第九/十章；8B复验已启动、机制范数扫描尚待执行；进度不是成功结果。

## 二、文献调研

本次为围绕上述问题的定向证据更新，不声称完成系统综述或重读旧稿全部60篇。信息截止及本次核对日期为2026-10-16。实际使用arXiv原文/摘要、Spec-Bench官方仓库、已有来源核查及结果文件；从旧v6参考文献追踪基础工作，并检索了“SpecBench official github hemingxia Spec-Bench”，核对当前官方入口为`hemingkx/Spec-Bench`。后续若补查，关键词围绕training-free multi-token prediction、embedding-space probing、speculative verification、dynamic draft tree和mask initialization，不扩大为新实验授权。

纳入与研究问题直接有关的作者论文、官方数据/实现来源，以及用户指定的第三方实现；跨模型、精度、树预算、训练式起草器或不同数据条件的数字只作背景，不放入本仓公平排名。排除营销综述、无法追溯的汇总数字及与当前范围无关的方法扩张。未查到可运行官方代码不等于证明作者未开源。阅读深度逐项列明，来源发现、论文主张、代码核查和实测分开记录。

| 文献与定位 | 问题与实际阅读范围 | 数据与模型 | 算法 | 指标及条件 | 证据、局限与用途 |
| --- | --- | --- | --- | --- | --- |
| [ESP v2](https://arxiv.org/html/2603.17942v2)，§3、Table1、Appendix G.2 | 本次核对摘要、Eq4/5、Table1及附录定位；既有相关正文分析见[复现计划](<研究计划 Denoising Speculative Decoding.md#legacy-docs-research-esp-table1-bf16-reproduction-plan>) | SpecBench；本仓对应Qwen3-8B复现格 | embedding mask、动态树与剪枝 | 目标τ见第八章；不采用论文测速为本次产物 | 最接近基线；必须区分论文、作者实现、自实现与第三方，不能用自实现负结果判定原方法无效 |
| [Leviathan等，2211.17192v2](https://arxiv.org/abs/2211.17192v2) | 本次摘要核对；原论文采样论证没有在本次重新证明 | 原文Transformer设置，不是本仓4B配对 | speculative decoding及目标分布验证 | 原文条件与本批T0不同 | 支持草稿/验证职责区分；不能由本仓同次forward argmax审计声称覆盖所有采样分布 |
| [Chen等，2302.01318](https://arxiv.org/abs/2302.01318) | 本次摘要核对 | 原作者模型与采样实验，不并表 | speculative sampling | 非本次T0数值对照 | 作为正确性与随机采样背景；本次不新增采样实验 |
| [EAGLE-2，2406.16858v2](https://arxiv.org/abs/2406.16858v2) | 本次摘要核对动态树动机 | 作者训练式起草模型设置 | 动态draft tree | 与本仓免训练mask条件不同 | 提醒接受行为与上下文相关；不是已执行的本仓baseline |
| [Medusa，2401.10774](https://arxiv.org/abs/2401.10774) | 本次摘要核对多头路线 | 作者模型及额外预测头 | 多token head | 没有本仓同条件实测 | 保留训练式替代路线，不据此声称免训练自动免于分布偏差 |
| [Spec-Bench官方仓库](https://github.com/hemingkx/Spec-Bench)，data/spec_bench与Readme | 本次官方入口/仓库说明核对，实际样本分布从冻结manifest计算 | 原固定480条；本地13个category | 评价平台与原始问题来源 | 480条是输入单位，不是独立方向重复数 | 数据来源；repo标Apache-2.0，不据此替原始组成数据另行确定所有使用条件 |
| [指定HF第三方源码及审计](research/esp-third-party-reproduction-2026-10-15.md) | 既有源码/API与真实数学门审计，本次复查交付报告 | 原固定q121、8B BF16，三预算预检 | 原字节API，非作者官方；CLI协议另有差异 | 六次forward后首token门失败；无有效τ | 终止该尝试；不能当官方或全480复现，也不修成新方法继续跑 |

文献地图的选择结果是继续固定模型的免训练mask比较，同时收紧“输出空间更优”和“固定模板方向唯一有效”的解释。旧v6中Future Lens、LatentMAS、指数平滑和Kalman等文献仍作为历史线索保留，未在本次重新核对全部正文、定理条件或新颖性，不把它们升级为本轮方法贡献。文献中关于无损与吞吐的主张，也不替代本仓实际正确性门和未测速边界。

## 三、研究问题与假设

| 问题/假设 | 条件与可反驳预测 | 当前判断与下一证据 |
| --- | --- | --- |
| R1（当前最高优先）：8B论文主要τ能否复现 | 按Table1目标和Qwen3单mask主协议比较；若主要值仍有明显缺口，不宣布数值复现 | 自实现有缺口，现有实现为非作者官方，第三方预检失败；停止追入口，以新独立复验检验可复现性，不据此判定ESP无效 |
| H1（已完成/归档）：输出零/一阶在本批优于输入一阶 | 同480、4B BF16、同BC/树/停止条件；若两输出整体弱于输入一阶则不支持 | 原五方法全量不支持；速度项也未显示稳定联合收益，见第九章 |
| H2（现有方向证据，范数扫描接续）：输入收益只来自e2−e1特定方向 | 三个预先固定的同范数随机方向均无收益才与该解释相容；出现明确随机收益即反驳其“唯一”措辞 | 三个完整随机格均高于zero，random2也高于template，不支持唯一方向解释；不推出任意方向均有效 |
| R2（归档/后置）：内容初始化在同条件下如何表现 | 保持原prompt与完整mean，只变原生速度初始化后缀；与template/zero配对 | content全480完成，τ1.364829262低于zero1.422420206；其范数未另匹配，不能解释为纯方向效应 |

最新CPU结果见[Table1差距分析](research/esp-table1-gap-analysis-2026-10-16.md)：BC30/60单mask距论文为−0.066622/−0.092699；逐prompt或六来源等权仍未达到目标，尾部计数差仅0.002248/0.003039。作者侧逐任务计数及完整合同缺失，不能唯一归因。

### Table1差距归因与候选控制（12:44前历史状态）

> 以下候选和显存门为PR40时点记录；最新不再优先排G-M/G-L/G-P，也不再追官方入口。当前执行以本章新增范数扫描及第一章主机RAM准入为准。

本轮按最新指示转向差距归因，不再循环检索未变化的公开渠道。以下数字来自[CPU输入/计数对账](research/esp-table1-cpu-contract-reconciliation-2026-10-16.md)、[原差距账本](research/esp-table1-gap-analysis-2026-10-16.json)及[本轮归因/准入证据](research/esp-table1-attribution-hypotheses-2026-10-16.json)，均保留固定480；事实、条件解释和待验证假设分开。

| 已核验事项 | 对τ偏低能说明什么 | 尚不能说明什么 |
| --- | --- | --- |
| BC30/60 pooled 1.563378/1.647301，距1.63/1.74为−0.066622/−0.092699 | 本地确有未复现差距；数学门通过的有效结果 | 不据此判断作者ESP方法无效 |
| prompt等权可提高0.029855/0.022197，仍差0.036768/0.070502；类间调用权重项为−0.029186/−0.022591 | 若作者用prompt等权，这部分差距可由口径解释；本地pooled/prompt差主要来自类间权重 | 作者实际权重未知，不能写成已证明的论文误差原因；不能换主指标报达标 |
| 四格480输入重渲染精确、无过滤；BC30/60的320/323条早停均EOS | 本地格间输入漂移、漏样和上下文上限不是该批差异来源 | 不能证明作者使用同一模板/thinking/tokenizer/qid或停止合同 |
| 既有轨迹前100 pooled 1.564682/1.647816；尾部纯计数差仅0.002248/0.003039 | 所检查的前缀/裁剪账面变化不能补齐缺口 | 非100-token新生成，不排除作者配置不同；prefill记入则更低 |
| BC30/60重复root回退260/256次，占decode calls的0.005127/0.005329 | 原路径逐次额外记1 token的乐观同调用数账本也仅到1.568504/1.652630，仍低于目标 | 不是关闭pruning后的真实收益或全路径上界；构树、数值和后续状态都可能改变 |

重复root计数可用`python tools/analyze_esp_repeat_roots.py`从原Run重放，与本轮JSON逐项一致才通过；[只读分析器](../tools/analyze_esp_repeat_roots.py)及终止/合同反例回归一并保留。

重复root计数直接读取原始trace：验证前向提交的首token等于该步root、accepted_nodes为空；本地pruning规则恰排除parent-repeat候选。BC30/60涉及111/115条prompt，calls为50712/48041；均为非终止块。此观察支持“局部候选排除有成本”，不支持把原轨迹上的这项直接损失当全部缺口。完整路径的更新mean、候选覆盖及作者合同仍未隔离，不能唯一归因。

以下只检验已有实现/既有方法开关，不发明方法、不改默认主配方或prompt。所有可执行控制均为**非作者官方**、新Study/新version/新Run；同一原480、8B BF16/eager、T0、seed123、单mask、原λ0.1/树/EOS与argmax门。原BC30/60更新mean完整格复用，不重跑。接受数/calls/τ逐prompt胜平负与输出长度共同报告，独立AR仍只诊断。

| 假设/优先级 | 精确对照与预测 | 能证伪的结果 / 边界 | 当前可执行性 |
| --- | --- | --- | --- |
| G-M：更新mean使8B候选覆盖受损（优先） | 只把`spec.mask.update`从true改false，保留mean_prompt初始化；新BC30/60两格，固定顺序30→60；比较已有更新mean。若该解释成立，冻结后的pooled τ应在两预算提高，接受/calls配对应支持而非仅输出变长 | 未同时提高则不支持跨预算版本；即使达到1.63/1.74也只是frozen控制，不是Eq5 λ0.1作者复现，不改默认或只选好格 | 原生开关存在，无科学源码改动；待完整Study冻结与容量准入 |
| G-L：100/256实际停止合同导致差距（次要） | 只改`eval.max_new_tokens=100`；BC30/60，EOS保留；预期接近已观测前缀τ1.564682/1.647816，比较实际计数/轨迹 | 若仍低于目标，则本地真实停止控制也不能补缺口；跨Run非exact只诊断，不进入算子链；已有CPU证据预示信息增益低 | 原生参数存在；G-M之后且有协议需求再做，不为占卡重复前缀结论 |
| G-P：重复剪枝显著降低接受（低优先） | 只把`spec.tree.pruning=false`，预算/节点数不变；BC30/60，对比原pruning=true | 原路径直接机会仅约0.005；若全路径更大变化须报告轨迹传播，不能叫直接重复token收益或作者差距原因 | 原生开关存在；CPU不支持其单独解释缺口，暂不排队 |
| G-A：作者实现合同与本地不同（证据到位优先） | 取得原字节作者repo/commit/入口后，对齐qid/token摘要、预算、mean更新、停止/计数，再按Table1运行 | 未取得入口或合同不同就不能称同协议官方复现；不自己补写源码 | 仍缺作者入口；不循环刷新旧来源 |

三因素/两预算共六份配置已通过原生validate，未构造输入/模型或创建Run。G-M为明确的待准入控制，不是已启动任务。未通过准入不make/launch，避免make隐式校准或产生同名Run；通过后先写独立Study的PLAN并冻结两格及停止规则，再复用原untimed/argmax适配和native make/launch，单队列串行。任一数学门/来源门失败保留失败并停止扩展，不因AR失配重跑；资源失败保留，不抢占其他作业。G-L/G-P不自动扩成全矩阵，bias/random/Kalman及算子机理继续后置。

本次容量观测见上述JSON：GPU总量24455 MiB，空闲21518 MiB，无其他GPU Python。历史全480整卡采样峰值22093 MiB包含背景；无native worker的73个样本背景为1962–2413 MiB。以最低背景扣除得到增量规划估计20131 MiB，加保守2 GiB（2048 MiB）为22179 MiB，本次缺661 MiB，容量门未过，GPU执行0。整卡峰值不直接与空闲比较以免重复计算背景；该差值也不是独立进程/allocator实测峰值，不能当保证容量。下次必须重测实际占用、必要时补隔离峰值；CPU进程不作阻塞门，不缩prompt/减数据来绕过门。

本次贡献目标是可追溯的复现边界与现有方法/指定对照比较，不声称新算法或普遍最优。文献界定基线和验证责任，数据与模型限定推断范围，算法限定变化因素，指标和配对实验检验上述问题。参数β、状态空间、输出路径及范数差异仍是混杂，不能把输入/输出比较全部差异归于一个空间选择。

## 四、数据

使用既有`datasets/spec_bench/question.jsonl`，来源入口为上述Spec-Bench官方项目。本轮不重新拉取最新数据替换；固定字节SHA-256为`7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21`。数据来源、输入IDs、EOS及顺序的合同见[4B prompts_manifest](../studies/qwen3_4b_bf16_five_methods/docs/prompts_manifest.json)。历史下载的上游immutable commit未另确认，不用当前GitHub HEAD代替历史版本；组成数据许可/预训练污染的逐项审计未完成。

样本单位为原顺序480条首轮prompt，question_id首尾为121/90。分布：math_reasoning、qa、rag、summarization、translation各80条；coding、extraction、humanities、math、reasoning、roleplay、stem、writing各10条，共13个category。输入采用原non-thinking chat template，不换模板、删题、截断或改写来调结果；最长1500输入token已由原4B预检记录。指标按所有480条配对，并保留各category结果。

本研究没有训练或拟合划分，使用冻结评价集完成既定比较；固定480已被反复分析，不能称为全新独立holdout，也不能据此挑最佳随机种子后报总体泛化。未做完整重复/污染识别，无法排除预训练重叠。每Run核qid、顺序、input_ids、模型内容及停止条件；不会把不同模型tokenizer生成的IDs未经核对直接互换。完整Run、输出token、trace、权重和cache只留本机；Git保留manifest、汇总和可复查证据入口。

## 五、模型

目标模型为现有ModelScope `Qwen/Qwen3-4B`：qwen3、36层、hidden2560、vocab151936，tied embedding/output storage由每Run实际指针核对，而非只看配置。输入为固定token IDs与指定mask，输出为目标logits、末层hidden及已提交token。权重、forward和最终mask使用BF16/eager/T0；原生输出EMA/范数内部FP32状态保留，不称全部内部计算均为BF16，也不新开FP32模型对照。

来源与文件内容见[modelscope_snapshot](../studies/qwen3_4b_bf16_five_methods/docs/modelscope_snapshot.json)，实际dtype/tied-storage/model_digest/runtime在[paired evidence](../studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json)。14个模型文件内容已核验；历史远端revision身份仍未完全确定，权重许可在本次未重新核验，记Unverified，不把本仓许可套到模型。运行环境为torch2.11.0+cu130、Transformers5.12.1、RTX5090 D v2；不同环境的细项以Run合同为准。安装依赖固定到已验证Transformers5.12.1，不适配5.20新缓存API，不替换正在执行的环境。

Qwen3-8B仅属于ESP Table1复现，现阶段不再为方向研究使用它。0.6B及4B FP32属于历史归档，与本批BF16结果分开。模型权重、解码树和停止规则在同预算比较中固定；新增因素仅为既有五方法或用户指定的方向控制，不新增训练式draft模型。

## 六、算法

复用现有[SpecMTP实现](../src/spec_mtp/structure/algorithm)及原生RPipe Study生命周期，不另建实验框架。以下是本仓实际比较定义，不把用户定义的ESP frozen mean写成论文的updating mean。

| 现有方法 | mask及状态定义 | 固定参数与边界 |
| --- | --- | --- |
| ESP frozen mean | 原prompt embedding均值，生成中冻结 | `update=false`，用户指定基线 |
| 输入零阶 | 完整prompt mean初始化，提交embedding后按`mean ← mean + λ(e−mean)`更新 | λ=0.1；即updating mean，方向控制zero直接复用此格 |
| 输入一阶 | `mask_j = mean + γ j v`；原生embedding差分EMA更新v | β=0.9999、γ=1、λ=0.1、order0 level；normalize与extrapolate_ema关闭 |
| 输出零阶 | 原生末层normalized hidden及代理状态生成u，再以`r u/‖u‖`作为mask；r为词表embedding范数均值 | order0；保留原owner/history语义，非输入embedding的逆映射 |
| 输出一阶 | 在同一末层hidden预测中加入EMA速度，再做同一范数映射 | order1、β=0.9、代理β′=0.99；不是仅改空间而其他状态全部相同 |
| 独立AR | 同一模型、原prompt、seed及停止规则的逐token贪心生成 | 共享一次480条参考，不在每个Spec Run重复执行 |

tied模型不做逆映射。输出代理只按原生root/候选owner和已接受历史更新，不额外teacher纠正或增加验证forward。树构造、剪枝、KV cache、bonus和EOS/长度裁剪继续使用原实现；接受数是裁剪前候选接受计数，提交数按实际裁剪计算。模型无训练目标或梯度更新；本次也不优化超参数。详细实际配置以正式evidence的`actual_settings`为准。

指定方向控制只改上述输入速度构造：template固定`v=e2−e1`；random0/1/2分别用CPU Generator种子0/1/2取固定随机方向，FP32归一到同prompt模板差分范数后转BF16，生成中v固定；zero为v=0；content从原首个user内容token起的后缀初始化原生EMA速度，保留完整prompt mean并继续原生更新。content后缀包括原末尾模板，不改输入，范数不额外匹配。template不等于会持续EMA更新的原生input_first。

代码分类：当前ESP全480及五方法是本仓自实现/扩展；作者官方入口尚未核验到，不能称官方代码复现。指定HF来源为dataset `Harshvardhan-Mestha/esp-reproduction-artifacts`，commit `ad76f8da29f3bfc7faa20488193c6aa83a9f708c`，八文件按原字节保存于[reference及ORIGIN](../src/spec_mtp/structure/algorithm/reference/esp_reproduction_artifacts/ORIGIN.json)，未附LICENSE记Unverified。既有外部验收调用其原生API而非原CLI，适配与失败见[第三方报告](research/esp-third-party-reproduction-2026-10-15.md)。不把它修成新baseline继续跑。

正式配置变化必须新Study版本/Run，已有成功Run不重签。方向原协议`fe59d35`及资源补齐协议`0e7c46f`属于[PR22](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/22)；固定480的科学步骤不变，补齐版仅将实际GPU资源拒绝改为prompt边界等待。原失败Run与105次完整prompt执行的资源损耗单列，不拼接成有效480。

## 七、指标

| 角色与名称 | 定义、单位、方向 | 数据与聚合 | 实现及判据 |
| --- | --- | --- | --- |
| 主指标pooled τ | Σdecode committed / Σdecode calls，token/解码调用，越高越好 | 全480；不含prefill首token | 原生raw与共同CPU读出；按预定预算逐格报告，不择聚合宣布复现 |
| 辅助两聚合 | 逐prompt τ等权均值；先逐category pooled再13类等权均值 | 两者单列，不冒称论文任务权重 | 同一结果文件重算；不隐藏与pooled的差异 |
| 接受数/主调用数 | accepted越高越好；prefill+decode calls越低越好 | 按qid配对分别报胜/平/负、总差值 | 不强行合成单一赢家；输出路径/长度不同，不能解释为等输出加速 |
| 逐prompt τ配对 | 用整数交叉相乘比较两个比值，保留平局 | 方向对template及既有五方法 | 480条顺序/身份必须一致，报告全部输局 |
| 正确性硬门 | 每个提交token等于该次验证前向正确owner logits的argmax，含prefill | 全输出token覆盖、owner/hash链核验 | 必须全过；来源、非有限、计数及owner审计失败不能用AR诊断代替 |
| AR诊断 | 全序列一致率、逐条首分叉、输出长度 | 同一共享AR参考 | 只报告，不阻塞native summarize/process，不转向算子数值研究 |
| 资源工作量 | 模型调用数、提交token、失败格已消耗工作量 | 成功与失败分开 | 支持排班和BC60决定；不测时延/吞吐，零计时字段是占位 |

全部数字定位到结果文件、cell和Run；作者值、本地实测和未测项分开。没有另设“差距可接受”的数值阈值或事后显著性阈值，未达到既定论文目标就如实报差距。数学自检覆盖T0实际提交，不据此宣称任意温度分布或跨forward形状逐token必然等同。

## 八、实验设计与执行顺序

| 实验与问题 | 对照及变化因素 | 固定条件与重复 | 预算、状态与判读 |
| --- | --- | --- | --- |
| E1：ESP Table1来源与数值复现 | 作者入口核查、自实现参考、指定第三方预检分别记录 | 8B BF16/T0、原480；主BC10/1、30/1、60/1，60/2附加；λ0.1 updating mean | 自实现四格完成但未达到目标；官方入口未确认；第三方预检失败后停止；新独立8B非官方复验已排队执行 |
| E2：五方法加AR | 六方法共享输入；BC30/60分别比较五种mask | 4B BF16/T0，双mask、naive动态树剪枝、draft budget9/19，seed123+index，max_new_tokens256/EOS，默认归约 | 1共享AR+10个Spec Run，共5280次prompt执行但只有480个评价输入；已11/11原生验收 |
| E3：指定方向全480 | template、random0/1/2、zero、content；每个方向相对template及旧五方法配对 | 同E2 BC30；mean/树/停止/输入全固定；方向种子提前指定，解码seed不变 | 五个新增完整路径+复用zero，新增有效预算2400次prompt执行；六格全完整，补齐Study process通过 |
| E4：范围与证据整合 | CPU重算、出处/边界、规范/中英文计划、PR/dev集成 | 复用同一raw及既有工具，保留所有有效/失败格 | 不增加GPU矩阵；E3终验/图/报告完成；BC60方向本轮不追加，理由见最终报告 |

### （一）基线复现与后续安排

实现来源与实际范围以[作者来源核查](research/esp-official-code-reproduction-2026-10-15.md)、[8B执行证据](../studies/esp_table1_bf16_full_argmax/docs/protocol_evidence_v1.json)及第三方报告为准。当前没有可运行作者入口的核验证据，不能用任意GitHub同名项目替代；后续若取得作者版本，再按Enmao范围对齐数据、模型和原生入口，记录必要适配。本次新8B独立复验依据12:44明确授权及第八章预注册执行，不依赖官方入口。

| 目标与条件 | 计划目标及出处 | 本地实测及出处 | 差距与判断 |
| --- | --- | --- | --- |
| 8B BC10/1 | 用户要求的附加预算；Table1不填臆测目标 | τ1.399047，protocol_evidence的`cells` | 完整数学门通过；无对应Table1目标差值 |
| 8B BC30/1 | 论文Table1 τ1.63 | τ1.563378，同上 | −0.066622；自实现未达到主要目标 |
| 8B BC60/1 | 论文Table1 τ1.74；Qwen3单mask依据Appendix G.2 | τ1.647301，同上 | −0.092699；未复现目标 |
| 8B BC60/2 | 预先登记附加对照，不替代单mask主配置 | τ1.603729，同上 | 不据对照格或另选聚合宣称主目标复现 |
| HF第三方三预算 | 原API在共同BF16/T0/原q121的数学门预检 | 六次forward；首token argmax margin40.75/33.5/33.25，见[审计JSON](research/esp-third-party-audit-2026-10-15.json) | 无有效τ、未执行全480；非官方，负结果保留 |

以上复现缺口影响对“原ESP效果”的结论；本仓4B配对只能支持共同自实现条件下的比较。论文checkpoint历史revision、模板/长度/聚合等未完全固定，作为协议缺口列出，不写成已经证明的原因。永久停止BF16相对独立AR的算子级机理链。

### （二）消融覆盖与组成

E3是对既有输入速度解释的指定控制，不是新增方法主表。随机方向和template匹配原范数，但BF16量化后使用实际范数比记录；content没有范数匹配，初始化及后续状态也不同。固定v与原生input_first持续更新的差异另列；不能把它们称作同一方法。四窗口局部干预仅是历史分析，不能代替全480；最终报告保留各category及全部逐prompt输局。

### （三）执行安排与优化方法

原GPU单队列已自然exit0，content完成；CPU正式读出、来源/argmax/原生process核对通过。计划PR23及结果PR22均已合dev，对应远程工作分支已删，不更改E3冻结PLAN。正常提交、push、PR合dev及合后远程工作分支清理按用户授权执行；不合main、不force push、不删本地分支、Run或数据。归档旧PR/issues只在状态和关联已核实后做，不因旧日期直接关闭。

GPU准入只看nvidia-smi实际占用与足够空闲显存，不把未识别CPU进程当GPU作业。每组完整结束后再让卡，共享GPU的量化插跑窗口已完成一次实际交接，不能中止当前组。资源短暂冲突在prompt边界等待同一Run，来源/argmax/OOM等失败仍如实终止该格并保留，不用改prompt修结果。既有成功template和zero不重跑。

这是预定离散对照而非参数搜索；无BO择优或根据测试结果换prompt。三种随机方向种子是三个固定偏置，不是三次训练种子重复；因为模板前两个token相同，每种子在480条中也给出相同方向。480条共享输入的配对有实际描述价值，不构成跨模型、独立训练或任意方向总体的置信证明。本轮不新增显著性宣称。方向BC60是否继续，在BC30全部结果后按实际token/forward成本和要回答的问题决定，不提前按优胜种子删减格子，也不把“尚未做”写成“已完成”。

<a id="current-execution-20261016"></a>

### 2026-10-16 新执行合同：非官方Table1复验与偏置范数扫描

此节为新执行预注册合同；整合历史章节不是重新注册旧结果。所有GPU工作以实际空卡和峰值主机RAM+2GiB准入，不与量化在同一时间跑，不杀任务。以13:02观测的量化占用为待资源状态，不虚报实验启动。每个新Study/version/Run独立，普通Run/Study全生命周期按native RPipe/SpecMTP执行；原始资产、失败和负结果保留。当前项目计划章节及不可变Git commit/blob作为来源引用，不另建Study PLAN文件，不修改vendored RPipe。

A：新`esp_table1_nonofficial_revalidation`，Qwen3-8B BF16/eager、T0、原SpecBench480、原non-thinking模板与输入IDs、seed123+index、256/EOS、mean_prompt逐tokenλ0.1更新、naive dynamic/pruning。固定三格BC10/1、30/1、60/1（draft4/14/29），按该顺序串行。若8B实测峰值主机RAM加2GiB不足，先报告并选择有本机权重及已验收峰值的最大可运行模型；不静默把4B叫8B结果。原8B完整数据仍为旧基线；本轮独立复验由12:44“开跑”指示明确授权，新Run不覆盖旧格。真实重做而不是再次CPU验收；报告是否复现本地旧τ、距论文1.63/1.74的差值、三聚合、逐qid计数/AR差异及输出长度。不为对齐目标筛样本/换口径/改dtype。

B：新`qwen3_4b_bf16_bias_norm_scan`，ModelScope原Qwen3-4B内容，BF16/eager/T0、原480/模板/input IDs、seed123+index、256/EOS，BC30双mask/draft9、dynamic/pruning，输入mean λ0.1，原β0.9999、normalize=false、extrapolate_ema=false。只用既有方向control与原生step_scale，m_i=updating_mean+i*s*v。v为原e2−e1或CPU随机种子0/1/2归一至该模板向量范数，跨提交固定；s∈{0,0.5,1,2}。不使用原生更新input_first冒充固定template，不另加content方向或Kalman。

| 方向 | s=0 | s=0.5 | s=1 | s=2 |
| --- | --- | --- | --- | --- |
| e2−e1 | 复用旧input_zero | 新480 | 复用旧template | 新480 |
| random0 | 同一旧zero，不重复计独立样本 | 新480 | 复用旧random0 | 新480 |
| random1 | 同一旧zero | 新480 | 复用旧random1 | 新480 |
| random2 | 同一旧zero | 新480 | 复用旧random2 | 新480 |

新八格顺序固定为template0.5→template2→random0/0.5→random0/2→random1/0.5→random1/2→random2/0.5→random2/2；每格必须完整480，不中途选优或按结果裁减种子。所有0/1倍来源SHA绑定既有正式JSON，只有一个零基线；不重签、不把复用计作新GPU计算。随机seed是三种固定方向，不是独立训练重复；同时保留e2−e1以区分范数响应与方向交互。

主读出为各方向/倍率pooled τ，另报prompt均值、category pooled等权；随机三方向算术均值曲线与逐种子曲线同时保留，不能只报均值遮住波动或择最佳倍率当新方法。逐qid相对同方向1倍和共享zero报告accepted/calls/τ胜平负，附输出长度、逐深度attempt/accept和第一验证块接受率；测初始v范数、倍率后的有效偏置范数及mean范数，检验实际实现确按倍率变化。前缀不同后的全路径差异不冒称同位因果；首块共同输入的接受/候选覆盖与全路径统计分开。“范数主因”需由跨方向一致的范数响应支持，若交互/种子波动大或曲线不一致则保留反例，不能事后把“方向无关”写成已证明。

正确性门：全部提交token逐一等于同次验证前向正确owner的argmax，含prefill首token、EOS/预算裁剪后的全输出，native来源/输入/数量/owner门必须全过。独立AR逐token一致率及首分叉只诊断；norm组复用原AR，不另加AR GPU。无teacher/warmup/时间测量，decoder timing保持零哨兵，不将资源观察时间当TPS/SR。任何来源/数学/非有限/OOM失败保留并停止该格，不自动重复；外部GPU工作只在组间交接或prompt边界等待，不抢占。跨模型先退出/释放后再加载；只允许一条串行调度链。当前状态pending，不用空值当0或平局。

### 2026-10-16 CPU准备补充：范数端点与验收边界

按17:52指示保留BC60原进程及实测峰值加2GiB门，等待期间推进CPU；八个4B新格仍排在8B之后。只读入口已核5个旧端点的30份Spec文件SHA、405083个已存argmax/前向owner记录和共享AR的81880个token，完整原始路径/正式源SHA及数据见[端点检查点](../studies/qwen3_4b_bf16_bias_norm_scan/docs/reused_readiness_v1.json)和[准备报告](../studies/qwen3_4b_bf16_bias_norm_scan/docs/STUDY_REPORT.md)。本次没有make、模型forward或新格验收；新倍率0.5/2保持未观测。

全480原input IDs+实际prefill token均相同且首块存在时，template/random0/random1/random2相对零偏置的首块接受数胜平负分别26/341/113、56/361/63、55/367/58、30/361/89。原完整路径τ仍为1.507777653/1.495188134/1.464349086/1.523884714，对比zero1.422420206。首块普遍获益解释受限，逐prompt胜负不等于均值接受差；后续分叉路径、调用权重和长度影响仍须分别报告，不由此新增方法或算子研究。范数扫描绑定E3固定v端点；E2原生更新输入一阶1.51195758单独保留。

验收脚本现显式要求索引与八格预注册及顺序完整相等，每格一个Run、cfg完整spec一致；缺格/重复/乱序/倍率或beta漂移/多Run均拒绝。--check-reused只读旧端点且始终complete=false/new_cells_inspected=false，不绘新曲线；正常模式仍保留native source/input/argmax/process门。范数18项及相关读出/配对/文档共50项CPU回归通过，当前新格尚未make，冻结计划commit/blob及科学源码、默认配方、数据/prompt未改。

### CPU原始审计补充（2026-10-16，非新实验）

复用`tools/analyze_4b_geometry.py --audit-raw`对E2五方法加AR的11格、66份原始文件逐SHA重核，10个Spec格809118保存token数学门全部通过，AR81880 token仅作参考。旧10格深度/位置/裁剪计数和6组τ分解逐项不变，属于复核而非新机制发现。[审计证据](research/4b-paired-raw-audit-2026-10-16.json)、[解释与命令](research/4b-mask-geometry-2026-10-15.md#八2026-10-16-cpu完整原始审计与首块对账)。20组首块配对各480条共享前缀；输出零阶/一阶首块均480平，对输入一阶BC30为114胜339平27负、BC60为131胜306平43负（后者输出减输入）。这与完整路径τ落后并存，不把首块改善外推为全程收益，不把输出两order首块平局解释成prompt速度为零；共享prefill候选的时序与后续速度传播应分开描述。

原BC60 worker66500等待RAM门的次序不变，保留全部进程与数据；CPU只读审计不签收新Run、不载模、不改prompt或默认配置、不测速、不追算子。原8B队列自然退出且终验完成后才可切换执行树准备4B范数八格；缺任一格继续标pending，不新增方法或填补曲线。


### 19:15 RAM与空档排班补充

按最新授权，8B原门保持实测峰值加2GiB；安全工作集回收仍不足时，允许4B已登记范数八格在空卡串行执行，不再强制等8B原队列自然退出才make。先做固定最长输入的4B容量标定，之后以真实4B峰值加2GiB准入；只可逆挂起已核实本会话RAM-wait的BC60，finally恢复，格间若8B具备准入先让其完成。科学设置/480输入/原冻结计划不变，容量探针不是正式τ结果。[协议与实测记录](research/sd-memory-fallback-2026-10-16.md)。


4B容量标定已实际完成：固定最长输入q295/1500 token、256容量token数学门全过；进程树峰值8808693760 bytes，正式4B门10956177408 bytes。原BC60已finally恢复；这不是正式范数τ结果。

<a id="paired-followup-20261016"></a>
### 19:46后续4B五方法加AR独立复验准备

E2原11/11格已经完成，不追加“缺失格”或重签历史Run。本次按19:46最新指示准备独立复验Study`qwen3_4b_bf16_paired_revalidation`，与当前范数八格区分版本与Run。固定原480条输入/token IDs、ModelScope Qwen3-4B快照、BF16/eager/T0/256/EOS/seed123+i。五方法严格沿用E2：ESP frozen mean（λ0、update=false）、输入零阶（λ0.1 updating mean）、输入一阶（mean+gamma，原β=.9999）、输出零阶（末层hidden按原均值范数映射）、输出一阶（原hidden EMA速度β=.9/β′=.99）；BC30/60均双mask，动态树/剪枝，每预算五格加一次共同独立AR，共11格。这里ESP frozen mean与8B Table1的λ0.1 updating mean不同，不混称。

CPU队列：复用旧untimed adapter、native逐token验证器、load_cell/summarize/compare_rows；预检原生合并配置、11格完整顺序、冻结输入/权重/方法SHA，输出20组方法配对accepted/calls/τ胜平负、输出长度及AR首分叉。数学门为每个Spec提交token等于本次验证forward logits argmax；AR为独立T0参考，其逐token差异只诊断。缺原始证据、失败或不全格只记pending，不填正式τ；不因AR失配拒绝有效Spec格。不载模的测试先通过，Study仅CPU准备/make，不提前开新GPU队列。

GPU队列：原BC60有RAM准入时优先；否则现有范数八格按完整格串行。八格全部原生成功并逐格入账后，才能接这11格独立复验。沿用4B实测BC30最大输入容量+2GiB host门与实际GPU非抢占；BC60的新形状尚无独立峰值保证，逐Run采样且每prompt保留2GiB，容量不足则等待/保留失败、不降低门、不杀进程。旧结果不重跑来凑格；新11格是本次明确排队的独立验证，不改变主prompt或方法。收尾同主题本地提交、完整结果里程碑再PR合dev。范数每个成功格另有只读入账检查点，核验native status、输入/来源、token门及τ三聚合；全八格前不生成完整曲线。


CPU实际准备更新：新Study已通过11格原生声明展开与未make只读预检（0/11、complete=false）；联合73项回归及Ruff通过，旧11格未填入本次新结果。tools/audit_4b_norm_queue.py只读监视器75696已启动，0/8新格原生完成时不记τ；每格成功后复用严格读出并保存三聚合/文件SHA。19:53新增显示白名单+NVIDIA类型模块仅供新Study，当前8B与范数源冻结不改；本机探针10个显示上下文不阻塞，只有当前python纯C上下文作为已知CUDA等待。分支复查无新增可删项，活动/未合分支保留。

## 九、执行记录与结论

| 记录 | 实际证据 | 结果与边界 |
| --- | --- | --- |
| E1自实现 | 8B protocol_evidence及[阶段报告](research/sd-stage-report-2026-10-15.md) | 四格数学门全过、共318830 token；BC30/60单mask目标未复现；不是作者代码复现 |
| E2五方法 | [paired_evidence_v1.json](../studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json)，`cells`和`pairs` | 11/11原生验收；4800次Spec prompt执行共809118 token全argmax通过；主方法τ见下表 |
| E3完整证据 | [direction_evidence_v1.json](../studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json)、[最终报告](research/4b-full-direction-controls-2026-10-16.md)、[PR22](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/22) | `complete=true`；六格各480完整。原资源失败Study仍passed=false；补齐Study passed=true，不改签旧失败 |
| 第三方与来源负结果 | 官方来源核查、第三方ORIGIN及审计JSON | 找不到核验入口不证明未开源；非官方首token门失败，不虚构τ、不重跑8B |

| 已验收主方法 | 4B BC30 pooled τ | 4B BC60 pooled τ |
| --- | --- | --- |
| ESP frozen mean | 1.402895 | 1.499079 |
| 输入零阶 | 1.422420 | 1.519109 |
| 输入一阶 | 1.511958 | 1.636340 |
| 输出零阶 | 1.433178 | 1.528922 |
| 输出一阶 | 1.435540 | 1.529836 |

表中每个值来自E2证据`cells[预算/方法].totals.pooled_tau`，共享AR作为诊断参考，不给AR虚构draft接受τ。逐prompt接受数/calls赢平输、AR首分叉及输出长度都在该文件，完整结果不被本表舍入替代。本批输入一阶整体最好；输出一阶相对输出零阶增量小，不能据接受数或调用数某一列单独宣布联合收益，更不能外推等输出速度。

E3有效pooled τ：template1.507777653、随机0为1.495188134、随机1为1.464349086、随机2为1.523884714、zero1.422420206、content1.364829262，均来自上述正式六格证据。随机2相对template逐promptτ267胜/47平/166负；三个随机种子均高于zero，故不支持本批收益只由e2−e1方向造成的解释。仍有逐prompt反例，content范数未匹配且输出长度不同；这不是推荐最佳随机种子作新方法，也不证明任意方向有效。零偏置复用和失败资源工作量单列，不能把补齐成功包装成原失败Study通过。

本轮不追加BC60方向矩阵：新增五个有效格已有2,400次prompt执行、405,073 token/276,523模型调用，三随机种子已否定本批e2−e1唯一性；BC60回答跨预算泛化，不能消除content范数混杂。失败完整partial另记20,977 token/13,854 calls，不拼接有效样本，细节见最终报告。CPU回归及PR22/dev闭环已完成。当前接续第八章：8B独立非官方复验→4B范数扫描。已有CPU缺口账本保留，不再追入口，不改prompt、预算或聚合追目标；0.6B、算子、Kalman和新方法归档。

最新CPU来源刷新见[官方路径重查](research/esp-official-recheck-2026-10-16.md)：出版版Table1目标1.63/1.74未变；Figure8十类与采样表八类标签不构成Table1样本/权重合同。本地固定480/13类不改；入口与OpenReview覆盖有限。该来源核查已结束，不再作为新执行门。

作者repo/commit/附件[追加追踪](research/esp-author-attachment-followup-2026-10-16.md)已完成本轮有界核查：主页commit未变、三个精确代码查询各0项、六个论坛ID命中均未核验出作者实现；OpenReview三个附件候选403，未取得代码。未取得可核验作者代码；本轮不再续搜，不把非官方执行冒称官方。

当前未解合同集中在[Table1协议缺口登记](research/esp-table1-protocol-gaps-2026-10-16.md)：逐项区分本地事实、作者所需证据、CPU对账与官方启动触发。本轮作者公开帖未提供可核验代码入口；短链未成功解析，OpenReview未刷新。PR36完成旧非官方全480原始证据复核；当前新复验独立记录version/Run，不覆盖旧结果。

[会议/HF公开渠道追加核验](research/esp-table1-protocol-gaps-2026-10-16.md)补齐作者账号公开资产范围：三种仓库API各返回0项，会议页面可见文本未提供代码；私有/组织资产及录屏内容未覆盖。此项仅历史来源覆盖记录，不再周期刷新。

本轮[P1输入/P2计数CPU对账](research/esp-table1-cpu-contract-reconciliation-2026-10-16.md)完成：四格原480首轮重渲染均token精确、无过滤（输入21–1500 token；80条源记录有多轮但仅取首轮）。BC30/60低于256的320/323条均为EOS停止；pooled相对prompt mean的差−0.029855/−0.022197主要是类间调用权重。完整旧trace前100 token的pooled τ为1.564682/1.647816，仍未达1.63/1.74；这是已观测前缀计数，不是100-token新生成。原主结果不变；作者qid/模板/thinking/revision、实际100/256停止合同、逐条原始计数/任务权重仍缺，不能唯一归因。现有对账不能唯一归因。当前下一步是已冻结的独立复验及范数曲线，保留输入与停止/聚合合同不变；原作者侧缺口如未来自然获得证据再核对，不主动循环追找。

### 夜间至本轮合入台账（UTC；均已合dev）

来源为本轮已取得的GitHub PR元数据；源码搜索类PR仅保留历史证据，12:44起不再续搜。PR36是旧全量证据复核，不是新增GPU执行；新独立复验见顶部快照。

| PR | 合入时间 UTC | merge commit | 内容/证据范围 |
| --- | --- | --- | --- |
| [#22](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/22) | 2026-10-15T20:02:48Z | `d792ca578` | 六格方向全480，含随机3seed和失败恢复边界 |
| [#23](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/23) | 2026-10-15T19:38:57Z | `b00ecc0a1` | 规范模板与中英文主计划 |
| [#24](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/24) | 2026-10-15T20:17:26Z | `617278abb` | 完成状态、GPU交接、远程分支清单 |
| [#25](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/25) | 2026-10-15T20:37:42Z | `0dc6568df` | 既有Table1差距与CPU优先缺口 |
| [#26](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/26) | 2026-10-15T23:52:56Z | `c467c21b9` | 出版版Table1与公开入口核查 |
| [#27](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/27) | 2026-10-16T00:02:06Z | `3eac53c9e` | 官方入口启动清单 |
| [#28](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/28) | 2026-10-16T00:19:34Z | `8e2132b33` | 作者commit与附件访问范围 |
| [#29](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/29) | 2026-10-16T00:49:08Z | `635a9a589` | 共同作者与历史refs核查 |
| [#30](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/30) | 2026-10-16T01:08:43Z | `1e734ff61` | 组织README候选核查 |
| [#31](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/31) | 2026-10-16T01:31:50Z | `eed894989` | release与arXiv源包核查 |
| [#32](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/32) | 2026-10-16T01:51:52Z | `a9e6a6653` | 源版本与公开资产核查 |
| [#33](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/33) | 2026-10-16T02:21:33Z | `d5f75ef02` | 发表渠道与OpenReview访问限制 |
| [#34](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/34) | 2026-10-16T02:36:24Z | `a3acf6795` | 已识别作者公开branch/release refs |
| [#35](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/35) | 2026-10-16T02:53:18Z | `c4f589a44` | 组织源码搜索线索 |
| [#36](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/36) | 2026-10-16T03:26:47Z | `6ce71665e` | 非官方四格全480原始证据复核，无新GPU矩阵 |
| [#37](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/37) | 2026-10-16T03:57:26Z | `776a2b4a2` | 协议缺口分层登记 |
| [#38](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/38) | 2026-10-16T04:18:15Z | `8d0f49cc4` | 作者HF公开资产及会议渠道范围 |
| [#39](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/39) | 2026-10-16T04:37:41Z | `437954bbe` | P1输入合同/P2计数聚合CPU对账 |
| [#40](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/40) | 2026-10-16T04:57:05Z | `aed4eee23` | 重复root剪枝机会及可证伪假设 |

文献阅读深度见第二章，旧v6的60条参考文献、定理条件和未核对项完整列于第十一章；没有因归并而宣称新增正文核验。关键未解项仍为作者实际输入/停止/τ权重、非官方实现与论文的差别、有限随机seed及内容幅度混杂；下一步先独立复验，再按已冻结8格范数扫描检验，不恢复入口搜索/算子链。

本轮新证据：[BC10全480复验](../studies/esp_table1_nonofficial_revalidation/docs/STUDY_REPORT.md)已验证本机旧结果逐条重复，后两预算尚未完成。PR44源字节修正、PR45范数准备、PR46三格读出/本检查点是当前执行支持，均不冒称作者官方代码或新方法；不因AR失配重跑，不改prompt或算子设置。项目计划继续只保留中英文两份，原冻结085722e章节引用不重签。

## 十、GitHub使用与计划变更

本文件是当前宏观计划入口，英文版逐章对应；历史Study独立计划已并入第十一章，其先于make冻结的合同按固定Git版本核验，数字/图在其`STUDY_REPORT.md`和evidence中。原始tokens/trace、完整Run、模型和cache不进Git。每次正式运行沿用原生记录的commit、source/input/model摘要、配置、实际环境、seed、命令、RunID、失败和结果路径，不重建同名Run或重签历史成功记录。

| 日期 | 变更与原因 | 影响范围 |
| --- | --- | --- |
| 2026-10-16 | 按Enmao新规范十章重写并补英文，导入RPipe固定来源规范/模板，纳入定向文献证据和当前结果 | 替换默认计划入口，旧v6历史稿保留；不是重新预注册，不改已冻结实验 |
| 2026-10-16 | 记录作者/自实现/第三方差异、正确性门与AR诊断、三随机方向及资源补齐状态 | 收紧结论，不扩大GPU预算或改prompt |
| 2026-10-16 | 安装依赖固定Transformers5.12.1，避免5.20缓存API漂移导致CPU CI失败 | 只固定已验证安装版本，不改科学缓存算法、不替换现有GPU环境 |

计划PR23与结果[PR22](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/22)均已合dev，对应远程分支均已删除，详见[清理记录](research/remote-branch-cleanup-2026-10-15.md)。规范正本、译本、来源与SHA见[standards README](research/standards/README.md)。旧v6更广的60篇参考文献仅为历史线索，不能当本次已全部审阅的证据。本文献表各作者原文、官方数据入口和实际使用的本仓报告即本次参考资料，关键事实已就近定位。

最新收尾：PR #41双CI通过合dev（5ef98ba），唯一中英文计划归并完成。随后已清27本地/1远端已合分支、撤销1废弃worktree登记，859份实验文件哈希未变；GPU活动树保留，开放PR/issue查询均0，完整名单及约每5个合PR的规则见[分支清理记录](research/remote-branch-cleanup-2026-10-15.md)。未合过时项先列清单，不删除实验数据。

范数扫描准备状态：脚本及9项CPU检查通过，未make/launch；8B运行中尝试提前make被自动审批以“可能重叠GPU/模型准备，需等8B队列结束”拒绝，整条命令未执行。准确命令和理由见[审批记录](research/remote-branch-cleanup-2026-10-15.md#自动审批记录范数study-make延后)，自然退出后再按串行顺序准备，不绕过。

### 归并验收与复放边界

全称改名由git mv单独提交7c609f9；60份子计划归入独立章节后按12:53明确授权git rm，原文/公式与固定Git来源保留。当前导航、报告生成器及工具读计划位置同步迁移；Study provenance.include不允许跨Study目录，故旧计划include替换为项目路径/章节/历史Git blob元数据，而不是伪装外部include。该元数据随study.yaml锁定，原生框架不会自动读取中央计划；新执行如当前8B复验另显式核验计划Git blob。

旧Run来源锁不重签；迁移后的Study声明/工具不应拿来冒充原执行源码。历史离线哈希读取固定Git blob并核对原换行SHA，复放完整旧Run仍需其原Git版本。正在执行的8B位于隔离工作树，未受此次主仓改动影响。科学源码src/spec_mtp、vendored src/rpipe、数据、原始结果JSON和权重均未修改。

本轮检查：链接/60章节/历史原字节/第三方证据专项10项通过；ruff通过，mypy56文件通过。完整CPU套件及PR远端CI结果在PR中列实际状态；未执行的检查不记通过。规范正本4文件字节不变，其3种DEM外部引用按来源README明确不可随clone解析，链接测试只豁免这3种已知外部路径。

## 十一、子课题与历史冻结计划的完整归档

以下60个独立章节归并原Study计划、专项协议、评审与历史草案。它们是历史原文，不构成恢复0.6B、FP32、算子机理或Kalman的执行授权；其“待执行/先exact”等表述按原时点保留，由本计划当前范围与正确性门覆盖。导航相对路径已随整合迁移；原字节、换行和Git blob身份见[归并清单](research/plan-consolidation-manifest.json)，需要核验历史签名时读取固定Git版本，不以当前计划重新签旧Run。

- [docs/research/esp-table1-bf16-reproduction-plan.md](#legacy-docs-research-esp-table1-bf16-reproduction-plan)
- [docs/research/publication-plan.md](#legacy-docs-research-publication-plan)
- [docs/research/research-plan-v6.md](#legacy-docs-research-research-plan-v6)
- [studies/_template/docs/PLAN.md](#legacy-studies--template-docs-plan)
- [studies/esp_table1_bf16_full/docs/PLAN.md](#legacy-studies-esp-table1-bf16-full-docs-plan)
- [studies/esp_table1_bf16_full_argmax/docs/PLAN.md](#legacy-studies-esp-table1-bf16-full-argmax-docs-plan)
- [studies/esp_table1_bf16_full_argmax/docs/THIRD_PARTY_PLAN.md](#legacy-studies-esp-table1-bf16-full-argmax-docs-third-party-plan)
- [studies/esp_table1_bf16_numerics/docs/PLAN.md](#legacy-studies-esp-table1-bf16-numerics-docs-plan)
- [studies/esp_table1_bf16_preflight/docs/PLAN.md](#legacy-studies-esp-table1-bf16-preflight-docs-plan)
- [studies/esp_table1_bf16_qwen_protocol/docs/PLAN.md](#legacy-studies-esp-table1-bf16-qwen-protocol-docs-plan)
- [studies/esp_table1_bf16_reduction_check/docs/PLAN.md](#legacy-studies-esp-table1-bf16-reduction-check-docs-plan)
- [studies/esp_table1_bf16_shape_check/docs/PLAN.md](#legacy-studies-esp-table1-bf16-shape-check-docs-plan)
- [studies/input_mask_baseline/docs/PLAN.md](#legacy-studies-input-mask-baseline-docs-plan)
- [studies/output_space_4b_holdout/docs/PLAN.md](#legacy-studies-output-space-4b-holdout-docs-plan)
- [studies/output_space_4b_input_control/docs/PLAN.md](#legacy-studies-output-space-4b-input-control-docs-plan)
- [studies/output_space_4b_recheck/docs/PLAN.md](#legacy-studies-output-space-4b-recheck-docs-plan)
- [studies/output_space_4b_recheck_v2/docs/PLAN.md](#legacy-studies-output-space-4b-recheck-v2-docs-plan)
- [studies/output_space_cross_task/docs/CANONICAL_PLAN.md](#legacy-studies-output-space-cross-task-docs-canonical-plan)
- [studies/output_space_cross_task/docs/EXECUTION_PLAN.md](#legacy-studies-output-space-cross-task-docs-execution-plan)
- [studies/output_space_cross_task/docs/FIRST_DELTA_PLAN.md](#legacy-studies-output-space-cross-task-docs-first-delta-plan)
- [studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md](#legacy-studies-output-space-cross-task-docs-kernel-execution-plan)
- [studies/output_space_cross_task/docs/KERNEL_PLAN.md](#legacy-studies-output-space-cross-task-docs-kernel-plan)
- [studies/output_space_cross_task/docs/LOCAL_FACTOR_PLAN.md](#legacy-studies-output-space-cross-task-docs-local-factor-plan)
- [studies/output_space_cross_task/docs/OWNER_PLAN.md](#legacy-studies-output-space-cross-task-docs-owner-plan)
- [studies/output_space_cross_task/docs/PLAN.md](#legacy-studies-output-space-cross-task-docs-plan)
- [studies/output_space_cross_task/docs/RARE_EVENT_PLAN.md](#legacy-studies-output-space-cross-task-docs-rare-event-plan)
- [studies/output_space_cross_task/docs/SAMPLED_DEPTH_PLAN.md](#legacy-studies-output-space-cross-task-docs-sampled-depth-plan)
- [studies/output_space_cross_task/docs/SAMPLING_CONTINUATION_PLAN.md](#legacy-studies-output-space-cross-task-docs-sampling-continuation-plan)
- [studies/output_space_cross_task/docs/SAMPLING_EXPANSION_PLAN.md](#legacy-studies-output-space-cross-task-docs-sampling-expansion-plan)
- [studies/output_space_cross_task/docs/SAMPLING_PLAN.md](#legacy-studies-output-space-cross-task-docs-sampling-plan)
- [studies/output_space_cross_task/docs/STATE_EXTENSION_PLAN.md](#legacy-studies-output-space-cross-task-docs-state-extension-plan)
- [studies/output_space_cross_task/docs/STATE_FACTOR_PLAN.md](#legacy-studies-output-space-cross-task-docs-state-factor-plan)
- [studies/output_space_extension/docs/NORM_KERNEL_PLAN.md](#legacy-studies-output-space-extension-docs-norm-kernel-plan)
- [studies/output_space_extension/docs/NUMERICAL_PLAN.md](#legacy-studies-output-space-extension-docs-numerical-plan)
- [studies/output_space_extension/docs/NUMERICAL_PLAN_REVIEW.md](#legacy-studies-output-space-extension-docs-numerical-plan-review)
- [studies/output_space_extension/docs/OPERATOR_PLAN.md](#legacy-studies-output-space-extension-docs-operator-plan)
- [studies/output_space_extension/docs/OPERATOR_PLAN_REVIEW.md](#legacy-studies-output-space-extension-docs-operator-plan-review)
- [studies/output_space_extension/docs/PHI_RMS_PLAN.md](#legacy-studies-output-space-extension-docs-phi-rms-plan)
- [studies/output_space_extension/docs/PLAN.md](#legacy-studies-output-space-extension-docs-plan)
- [studies/output_space_extension/docs/ROUNDING_FACTOR_PLAN.md](#legacy-studies-output-space-extension-docs-rounding-factor-plan)
- [studies/output_space_followup/docs/PLAN.md](#legacy-studies-output-space-followup-docs-plan)
- [studies/output_space_followup/docs/PLAN_REVIEW.md](#legacy-studies-output-space-followup-docs-plan-review)
- [studies/output_space_prerequisite/docs/PLAN.md](#legacy-studies-output-space-prerequisite-docs-plan)
- [studies/output_space_small_matrix/docs/CROSS_PROMPT_PLAN.md](#legacy-studies-output-space-small-matrix-docs-cross-prompt-plan)
- [studies/output_space_small_matrix/docs/PLAN.md](#legacy-studies-output-space-small-matrix-docs-plan)
- [studies/output_space_small_matrix/docs/SCALE_PLAN.md](#legacy-studies-output-space-small-matrix-docs-scale-plan)
- [studies/output_space_small_matrix/docs/TRANSFER_PLAN.md](#legacy-studies-output-space-small-matrix-docs-transfer-plan)
- [studies/output_space_small_matrix/docs/WINDOW_PLAN.md](#legacy-studies-output-space-small-matrix-docs-window-plan)
- [studies/qwen3_4b_bf16_directions/docs/PLAN.md](#legacy-studies-qwen3-4b-bf16-directions-docs-plan)
- [studies/qwen3_4b_bf16_directions/docs/RECOVERY_PLAN.md](#legacy-studies-qwen3-4b-bf16-directions-docs-recovery-plan)
- [studies/qwen3_4b_bf16_directions_recovery/docs/PLAN.md](#legacy-studies-qwen3-4b-bf16-directions-recovery-docs-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/GEOMETRY_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-geometry-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_ACTIVE_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-active-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_DIRECTION_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-direction-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_EARLY_TEMPLATE_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-early-template-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_RECIPROCAL_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-reciprocal-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_TEMPLATE_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-template-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/MASK_WINDOW_PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-window-plan)
- [studies/qwen3_4b_bf16_five_methods/docs/PLAN.md](#legacy-studies-qwen3-4b-bf16-five-methods-docs-plan)
- [studies/smoke/docs/PLAN.md](#legacy-studies-smoke-docs-plan)

<a id="legacy-docs-research-esp-table1-bf16-reproduction-plan"></a>

### docs/research/esp-table1-bf16-reproduction-plan.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[docs/research/esp-table1-bf16-reproduction-plan.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/docs/research/esp-table1-bf16-reproduction-plan.md)；原SHA256 `3da5be5ca14baeb826ac4a85f581187363aa30d9a5190f94fae69b3b14296833`。

#### ESP Table 1 reproduction — Qwen3-8B / BF16

当前执行口径（2026-10-15 Enmao 最终指示）：复现必须使用原作者官方代码，固定来源/commit 并按原字节迁入 `src`，保持原生入口；不得继续以本仓自实现替代。最新状态、公开来源核查、baseline 未执行清单与后续准入见[官方代码复现报告](research/esp-official-code-reproduction-2026-10-15.md)。尚未核验到作者实现，OpenReview 附件未能完整访问，本轮未启动新 GPU 运行。以下各节保留为历史自实现协议与执行记录，不作为新一轮运行授权；已有 Study 的冻结 PLAN/证据保持不变。

状态：2026-10-15 计划已立项，未声称复现成功。优先于所有其他研究。此前 4B 证据全部 FP32，已按工具/测试与证据两个主题提交，并推送工作分支；PR #7 的 Branch flow/cpu-checks 已通过并合 dev（8677376e1ac878a159a3095711eac43420d97bde），不合 main。0.6B 罕见事件/局部机理和 4B FP32 扩展暂停。

##### 1. 来源、目标与未公开细节

主来源：[ESP arXiv v2](https://arxiv.org/html/2603.17942v2)，§3.1/3.4/3.5、§4、Table 1/3、Appendix D/G.2。已读正文及本地 PDF 第 7 页表格。模型为 [Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)，权重优先 ModelScope；仅用于本 ESP 复现，禁止拿 8B 做其他研究。

Table 1 Qwen3-8B 的 ESP τ 目标：BC30=1.63，BC60=1.74。BC10 是额外协议核验，Table 1 没有其目标。本文表注为 256-token 输出/H100，§4 正文为 100-token/A100；本次遵循 Table 1 的 256 上限，EOS 则提前停止，逐条记录实际输出长度。不得把原仓库 100-token 静态配置冒充本复现。

论文未明确公开 precision、模型 immutable revision、Qwen thinking 开关、prompt 截断策略与 τ 跨任务聚合权重。BF16 是当前用户明确要求。采用非 thinking chat template、首轮原始 prompt、不截断且纳入全部 480 条；输出单列这些实现选择，不称其为已获作者确认的配置。RTX 5090D v2 24GB 与 H100 不同，torch/transformers 版本差异也记录。用户不要求测速，因此不测 S/R。

官方 SpecBench：[hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench)，固定提交 fd2c1cd7d2201ef71db4c5f4e455008f017967bf，data/spec_bench/question.jsonl。本地 480 条 JSON 对象及顺序与官方完全一致；远端 SHA256=4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04，本地 SHA256=7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21。只存在字节排版差异。保存原文件并绑定两哈希；不得为了对齐目标筛掉难样本。

2026-10-15 模型内容补核验：[官方SHA256证据](../studies/esp_table1_bf16_full_argmax/docs/modelscope-content-verification-v1.json)记录ModelScope master直接返回的Sha256。16个文件（含5个权重分片）均与原冻结模型清单及BC10 Run实际输入哈希一致。原冻结文件不修改、不重签；这证明核验时的官方内容一致，论文未给出immutable revision及历史下载渠道仍未确认，不冒称论文指定commit已验证。

##### 2. 当前协议（Qwen3 BC60 修订）

[Appendix G.2](https://arxiv.org/html/2603.17942v2#A7.SS2)，PDF第18页原句：

> For Qwen3 models, single mask token is used for BC=60 while for LLaMA3 model two mask tokens are used.

§4 的一般默认不能覆盖这条针对Qwen3的具体说明。[Table 3](https://arxiv.org/html/2603.17942v2#S4.T3)的单mask树宽[29]对应BC=2×(1+29)=60。Table 3的数值行是LLaMA，G.2语句在Figure8的附加结果语境中；Table1没有单独给出mask数。因此采用60/1为主配置是基于模型特定说明的协议判断，不冒称已经获得作者确认。60/2保留为预先声明的对照，不从结果中择优。

此前已执行的预检与数值控制以60/2为BC60格，属于对照结果，不能继续标为Qwen3 BC60主复现。原冻结PLAN/Run/证据保持原字节；修订预检采用新PLAN与新Run补60/1，并按冻结哈希复用原已执行三格，避免重跑已有数字。预检和全480报告均固定列60/1、60/2的τ及各自相对1.74的差值，标明AR质量，缺格写pending而不是0。

| 项目 | 本次值 |
| --- | --- |
| dtype / temperature | BF16 / T0，绝不新增 FP32 模型实验 |
| mean mask | prompt embedding mean 初始化；每个已生成 token 按 m←m+0.1(e−m) 更新 |
| BC / mask count | 主配置10/1、30/1、60/1；对照60/2 |
| tree | dynamic，Top-1 扩展，累计路径概率排序；父子同 token 替换为下一候选；pruning=true |
| implementation | native naive 动态树 attention/cache，保留原始模型与验证语义 |
| seed / outputs | seed123，max_new_tokens256，EOS 提前停止，warmup=false |
| input set | 全部 480 首轮，chat template enable_thinking=false；max_prompt_len40960，实际预检不丢任何条目 |
| correctness gate | 每个提交 token 等于产生它的验证前向 logits 的 argmax，全部 token 覆盖且必须全过；包括 prefill 首 token |
| AR diagnostic | 独立 AR 一致率及每条首分叉位置仅作报告指标，不阻塞 τ 执行或本次正确性验收 |
| numerical settings | BF16、默认归约及默认算子；不运行归约开关、单行 linear/RMSNorm、compact attention 或逐层追踪 |
| clocks | Study 局部禁用 decoding clock；零 timing 是原生字段占位，不是测量 |

BC=(k+1)(1+draft_nodes)，本次实际 draft budget=4/14/29，另保留60/2的19节点对照；检查每步真实 BC 不超过预算并记录树宽。论文 Algorithm 1 的 B 与正文 Eq7 表述存在歧义；遵循 Eq7 与当前 Top-1 实现并报告映射，不暗改预算来追目标。

主指标同时报告 pooled τ=sum(committed)/sum(decode_calls)、逐 prompt τ 的均值、逐 category pooled τ 后等权均值。committed 包含 bonus，不含 prefill；另列含 prefill 的实际 model calls、新 token 数与 accepted draft 数。尾部 EOS/长度裁剪口径明确，不混淆 accepted 与 τ。目标的近似程度报告原值/差值/按两位小数是否一致；不得预先把某个聚合方式定为作者确切口径。

##### 3. 执行与停止规则

1. 复用 native RPipe Study prepare→execute→collect→summarize→write→process；新 Study/version/Run，原失败和成功 Run 不重签。权重逐文件哈希、官方 ModelScope 文件名/尺寸与实际 BF16参数核验；离线缓存 immutable revision 未确认则明确记录。
2. 先新建小样本协议检查 Study（每类首条，BC10/1、BC30/1、BC60/1主配置与BC60/2对照）。检查 BF16装载、动态树、256上限、精度及原生门；这只是预检，不能代替 480 条 Table 1。
3. 按本次四格协议冻结全480 Study，串行执行BC10/1、BC30/1、BC60/1主配置及BC60/2附加对照。正确性门是数学无损自检：每个提交 token 都等于该次验证前向 logits 的 argmax，覆盖所有实际输出 token，并且必须全过；可记录 logits 差值，但不要求额外算子分析。与独立 AR 的一致率及逐条首分叉只作为报告指标，不阻塞 τ 执行或正确性验收。BF16树验证与逐token AR的前向形状不同，AR不一致本身不判定方法错误。数据、模型来源、BF16装载、树预算及自检覆盖仍须核验；来源/协议不符或OOM先修复，再用新Run，不切FP32、不用teacher纠正、不筛除反例。
4. OOM/其他任务占显存时保留失败、等资源或用新 Run 可逆验证，不停其他任务。单卡只启动一条实验调度链，不并发加载模型。
5. 全量完成后报告配置覆盖、精度、每任务/每 prompt计数、三种 τ聚合、BC30相对1.63与BC60相对1.74的差值、argmax自检覆盖/通过率、独立AR一致率及逐条首分叉。数值不符也保留可复核负结果；仅核对数据、配置和统计口径，不转入算子级数值排查，不按结果挑聚合或mask数。

四任务共用 24GB GPU。记录启动前空闲显存及运行容量采样（含其他进程，非精确 allocator peak）；不使用受干扰的墙钟或 timing 占位做性能比较。四格预检约 13×4×2×256=26624（其中旧三格已执行，新补单格上限6656） 生成 token 上界；全量四格含独立 AR 审计约 480×4×2×256=983040 上界。串行，没有额外 teacher/warmup。容量不足先暂停该 GPU 启动，继续证据/实现核对。

##### 4. 全480完成后的4B BF16配对

同批 prompt、同树/预算/长度/随机种子，比较 ESP frozen mean、输入零阶 updating mean λ=0.1、输入一阶 mean+gamma、输出零阶 final-normalized hidden 范数映射、输出一阶 final hidden+EMA velocity，另跑独立 AR。逐 prompt 列 accepted/model calls、每个对照的赢平输及 token exact。不得用此前 FP32 数字代替。

这里的 ESP frozen mean 是用户指定的后续基线，区别于 Table 1 updating mean。tied 权重直接用输出 hidden 的尺度映射，不做逆映射。4B工作分支已补齐按深度映射到branching/dynamic/pruned树owner的支持，单链默认行为保留；多兄弟、多深度、非首兄弟路径和BF16原生CLI均通过CPU回归，尚无真实4B BF16新结果。不能让方法使用不同树再配对，实际开跑合同见[4B PLAN](<研究计划 Denoising Speculative Decoding.md#legacy-studies-qwen3-4b-bf16-five-methods-docs-plan>)。4B 子实验开始前单独冻结 PLAN、beta/gamma、精度内部计算与实际 tied-storage 检查。按最新最终指示，先完成8B四格全480并写回三种τ、目标差值、AR一致率及verifier argmax自检，再启动4B BF16五方法加AR的同prompt配对；不因BF16相对独立AR的数值失配转入算子机理。不得提前与8B矩阵并发启动4B。

##### 5. 当前执行与归档（2026-10-15）

FP32收尾：PR #7 已经CI通过并合dev，merge8677376；main未操作。此前4B FP32及0.6B局部研究原位保留，本轮不扩展。

已完成的全480GPU执行链为[全480 argmax审计](../studies/esp_table1_bf16_full_argmax/README.md)，正式结果见[STUDY_REPORT](../studies/esp_table1_bf16_full_argmax/docs/STUDY_REPORT.md)。四格固定顺序为10/1、30/1、60/1、60/2；保持Qwen3-8B/BF16/T0/默认归约、λ=.1 updating mean、动态树+剪枝、256/EOS/seed123及全部480条。60/1为主，60/2仅附加对照，两者均报告相对1.74差值，不择优。

四格均完成全480并经新门重验为succeeded，原生process的来源、配置、seed及证据检查全过。pooled τ依次为1.399047、1.563378、1.647301、1.603729；BC30主配置距1.63为−0.066622，BC60单mask主配置距1.74为−0.092699，双mask对照为−0.136271。三种聚合均保留，按两位小数未复现论文目标，不以正确性门通过冒充数值复现成功。AR一致率依次为188/480、175/480、185/480、187/480，仅诊断。

Study局部观察器绑定每个选择行到对应验证forward的logits，记录argmax/选择值及forward/行号；覆盖prefill首token和全部实际提交token，EOS/预算裁剪后的未提交尾部不混入覆盖数。不增加模型前向，不改精度、算子、采样或树。argmax自检失败立即阻止该Run的execute完成；独立AR运行不安装此观察器。

四格旧AR门失败result均按SHA保留；按用户新授权仅重做summarize gate/write，execute_rerun=false，四项原始证据和执行来源锁字节未变，新validator SHA另列。队列退出后用同一门重做原生process，旧process结果也已归档。没有因AR失配重跑execute，没有改冻结科学配置、Run输入或来源锁。[原生process重验记录](../studies/esp_table1_bf16_full_argmax/docs/native-process-revalidation-v1.json)。下一阶段按第4节推进4B BF16五方法加AR；不再开8B算子机理链，也不以调参追目标数值。

无argmax记录的全量v1部分日志保留。新argmax Study v2首次四Run在模型加载前因ModelScope缓存接入失败，未执行解码；失败Run保留。补接此前同一已校验本地snapshot后以v3新Run执行，未改模型或实验设置。

永久停止BF16 ESP相对独立AR的全部算子级数值机理排查。既有[reduction报告](../studies/esp_table1_bf16_reduction_check/docs/STUDY_REPORT.md)、[shape报告](../studies/esp_table1_bf16_shape_check/docs/STUDY_REPORT.md)和测试已按主题提交推送，仅作历史归档；不再启动归约开关、单行linear/RMSNorm、compact attention或逐层BF16算子追踪。权重、缓存和完整Run留本机；全480结束后仅按第4节推进4B BF16五方法加AR，不测速、不恢复机理链、不合main、不force push。

##### 6. 执行授权与检查依据

依据2026-10-15 Enmao本任务长期授权，正常工作分支推送、PR、合dev、本地检查替代远端CI直接执行；不使用提问弹窗。报告注明实际验证命令、结果及覆盖范围；保留平台强制分支保护。删除数据/分支、强推、合main、花钱前暂停相应动作并写报告。自动审批拒绝记录具体命令、拒绝理由及执行状态，继续未受阻工作，交监督助手处理。[授权与审批记录](handover.md)。

2026-10-15原生验收门修订：用户明确要求summarize/process也以完整verifier argmax为门，AR仅诊断。在隔离工作树更新validation与原生T0审计元数据，当前GPU执行包保持冻结；已完成的旧AR门失败格只重新验收并归档原失败result，禁止重跑BC10 execute。执行来源与验收validator SHA分开记录；全四格原始execute全部结束后再按原顺序做4B，不恢复算子机理或测速。

</details>

<a id="legacy-docs-research-publication-plan"></a>

### docs/research/publication-plan.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[docs/research/publication-plan.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/docs/research/publication-plan.md)；原SHA256 `602c2cba074f8c78b0e2d54d731ca80745e6f018fedf3f96fdca747ef45719ad`。

#### 最小推送候选与本地提交清单

2026-10-14，本轮仅准备可评审材料，没有推送、创建新PR或合并。清单不是新的公开授权。当前分支feature/output-space-prerequisite包含实现和全部后续研究，直接推该分支会带入整段历史，不能称为最小功能PR。

##### 已核对的本地依赖

本机origin/dev为1b6d59c7f6507b75e3ff4e91b0a3142bcad5b8cd；origin/refactor/spec-rpipe为1219a1d88f10c2faee345b8d9bf6a42fd1e0110e。这是本机refs，不代表本轮远端状态核验。已有[PR6](https://github.com/DreamSoul-AI/denoising-speculative-decoding/pull/6)承载重构、RPipe0.3.1与Study来源门；本轮没有重新查询它的CI或合并状态。

最小实现候选依赖PR6那棵树。将来若获准提交功能PR，应先核对最新dev实际包含这些依赖，再从该dev建立干净feature分支，选取下表实现提交；现在没有创建分支、cherry-pick、rebase或改历史。不能把437f201直接接到本机旧dev，或者为绕过依赖合并PR6。

##### 候选顺序

| 顺序 | 精确提交 / 范围 | 作用 | 最小功能PR |
| --- | --- | --- | --- |
| 前置 | PR6树至1219a1d | 原生流程、来源门、重构依赖 | 属已有PR，不重复打包 |
| 1 | 437f20140ffc72a788814d16d3084d2677e66d60 | tied输出hidden零/一阶静态链、拒绝边界、四格前置验收 | 唯一必选功能提交 |
| 2 | 9ba5c3a | q171离线轨迹诊断 | 后续研究，暂缓 |
| 3 | 4a4ae8d | q172–174矩阵与匹配前缀 | 后续研究，暂缓 |
| 4 | 77fe411 | q173窗口与方向／范数对照 | 后续研究，暂缓 |
| 5 | 6a953bc | q173单槽历史替换 | 后续研究，暂缓 |
| 6 | 7a02501 | 跨prompt零结果 | 后续研究，暂缓 |
| 7 | a42313d | q175–178扩展 | 后续研究，暂缓 |
| 8 | f3b95ea | q178规范状态／严格重放及q179–182 | 后续研究，暂缓 |
| 9 | 本轮算子隔离、草案和推送准备提交 | q178 provider对照；未执行新Study；PR草稿 | 后续研究，暂缓，以本轮实际HEAD为准 |

功能候选437f201共33个文件，修改5个科学模块并附实际验收、测试和文档；父为1219a1d。诊断研究以最终完整依赖链评审，不建议任取最后一个研究提交：它导入此前helpers并引用前序Study。若单独做研究PR，应从已含功能提交的dev开始，重新整理最少的工具、测试和必要scalar证据，保持原始研究提交与本机数据可恢复；本轮未执行整理或改历史。

##### 可复核文件清单与排除范围

[候选manifest](research/publication-manifest.json)固定437f201各文件的Git blob、SHA256、字节数和提交依赖，按源码、测试、Study合同／验收、文档分组。哈希对应提交对象，不对应后来修改的工作树。清单保留来源与必要拒绝边界，不删除历史数据来制造最小范围。

排除后续研究提交、未执行跨任务Study、完整Run/shared/scripts/.tmp、权重、模型缓存、selected tensors、profiler Chrome trace、用户workspace、内部公司资料、临时环境日志。候选是33个提交文件，不能用git add .扩展。公开前仍需按实际待推tree核对来源许可、脱敏、依赖、目标dev、必需cpu-checks/Branch flow和模块Owner责任；本轮只是本地候选与说明草稿，没有新远端CI。此前270项、36格和安装验收是437f201记录的历史验证，不能写成新PR CI已通过。

##### PR说明草稿

[pr-draft.md](research/pr-draft.md)仅描述最小功能候选的最终行为与前置负结果。12prompt探索、当前kernel名称与未执行协议保留在研究报告，不混入功能交付或速度承诺。实际目标为dev，绝不直接main；不会以准备材料代替推送或合并授权。

</details>

<a id="legacy-docs-research-research-plan-v6"></a>

### docs/research/research-plan-v6.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[docs/research/research-plan-v6.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/docs/research/research-plan-v6.md)；原SHA256 `cde92995ae885650648cb0efaa9f19c83090545c03dcb9b78a9b15a8ad9291fb`。

---
title: "denoising speculative decoding: Training-Free Kalman-Filtered Output-Space Masks for Parallel Drafting"
subtitle: "denoising speculative decoding 研究计划（论文版 v6，草案）"
date: "2026-10-13"
---

> 历史探索稿（2026-10-13）。含未验证理论、当时待测设想和异条件历史结果，不代表当前执行或已证实贡献。当前计划已按新版模板重写为[中文](<研究计划 Denoising Speculative Decoding.md>) / [English](<Research Plan Denoising Speculative Decoding.md>)；本稿原文保留，Kalman/高阶等不因保留而恢复研究。

#### denoising speculative decoding: Training-Free Kalman-Filtered Output-Space Masks for Parallel Drafting

> **项目正式名称**：**denoising speculative decoding**（全小写）。代码仓库目录仍为 `denoise_spec_mtp`，仅作路径引用。
>
> **文档性质**：研究计划（论文体例），不是论文。
>
> **标注约定**：
> - 【实测】：仅限附录 C 的输入空间、树形实验；
> - 【待测】：本版主方法与卡尔曼变体，**均无实测**；
> - 【猜想】：未证明的陈述；
> - 【待作者确认】：需要作者拍板的事项。
>
> **相对 v5 的改动**：
> 1. 统一名称，并解释 denoising 的含义（§1.5）；
> 2. 新增卡尔曼自适应变体（§5.9）：状态空间模型、维度分解、预测-校正、离散代数 Riccati 稳态增益、基于新息的 Q/R 在线估计、代理观测 $R'>R$、槽位不确定性；
> 3. 新增 Algorithm 3、复杂度与数值稳定性分析；
> 4. 新增 AR / 差分 / 卡尔曼层级关系（§5.10）；
> 5. 新增命题 8–10、假设 H12–H13、实验 E16–E18、文献 [55]–[60]。

---

##### Abstract

投机解码的起草器大多需要训练，因此受训练分布约束，已有工作报告 EAGLE-3 在训练域外的接受长度显著下降 [44][45]。ESP [12] 不需要训练：同一次前向既验证上一轮草稿，又在 mask 位置起草，但它的 mask 是输入 embedding 的均值，不带预测信息。

我们提出 **denoising speculative decoding**。核心想法是把相邻位置的末层 hidden 看作“对下一 token 的预测”的**含噪轨迹**，用滤波估计其 $n$ 阶状态 $(h,\nabla h,\dots,\nabla^nh)$，再外推出未来位置的 mask。方法分三层：

- **固定增益版**：逐阶 EMA，即 α-β-γ 滤波；
- **主自适应变体**：卡尔曼滤波。用状态空间模型 $s_{t+1}=Fs_t+w$、$y_t=Hs_t+v$ 描述轨迹，增益由协方差递推给出；噪声协方差 $Q$、$R$ 按新息逐序列在线估计 [56][57][58][59]，全程不训练；
- **代理观测处理**：草稿前缀上的代理 hidden 赋更大的观测噪声 $R'>R$。

状态沿每个节点的前缀累积，块内只预测，所有节点的 mask 在一次前向中并行处理。tied 模型直接用 hidden 并做范数缩放，非 tied 模型经 LatentMAS 岭回归对齐 [34]。

理论结果：

- 任意 soft mask 下保持无损；
- v4 的固定 EMA 恰是“局部线性趋势、无观测噪声”模型的稳态卡尔曼滤波，且 $\beta$ 有闭式（命题 8）；
- 稳态增益在常增益滤波中均方最优，因此最优 $\beta$ 存在（命题 9）；
- 预测协方差给出各槽位的不确定性（命题 10）。

全部实验待测，包括与网格搜索固定 $\beta$ 的对照、跨数据集无需重调，以及跨域与训练式基线对比。

---

##### 1. Introduction

###### 1.1 背景
投机解码 [1][2] 用便宜的草稿加一次并行验证来减少目标模型的前向次数，同时保持输出分布不变。按起草器来源，现有方法分三类：

- **训练式**：Medusa [3]、Hydra [7]、EAGLE-1/2/3 [4][5][6]、MTP [9][10]，以及学习 mask/前瞻 token 的方法 [18][27][39][40][41]；
- **免训练（检索、n-gram、迭代）**：[8][17][19][20][21][29]；
- **免训练（embedding 探测）**：ESP [12]。

###### 1.2 定位：免训练意味着没有过拟合与迁移问题
训练式起草器是在某个数据分布上拟合目标模型行为的参数化模型。部署分布一旦偏离训练分布（领域、长度、模板），接受率就会下降。已有工作在跨域 [44]、长上下文 [45]、长推理链 [46] 等设置下报告了 EAGLE-3 的退化，并为此提出在线适配 [47] 或多起草器选择 [48]。

本法**不引入任何可训练参数**。起草完全由冻结的目标模型在 mask 位置完成，所用的全部量（hidden、$W_{\text{in}}$、$W_{\text{out}}$）都来自目标模型自身。因此：

- 不存在起草器对训练集的过拟合；
- 不存在训练集与部署集之间的迁移问题；
- 换模型无需重训，只需（非 tied 时）一次闭式预计算。

这一定位并不声称本法的绝对接受率高于域内的训练式方法。我们关心的是**跨域稳定性**，以及与免训练基线的比较（H1、H8）。

###### 1.3 问题
ESP 的 mask 是 prompt embedding 的均值。它是上下文的词袋统计，**不包含对下一 token 的预测**。所有槽位、所有节点共用同一个向量，模型只能靠注意力从前缀中自行推断 mask 位置该填什么。

###### 1.4 贡献

1. **前缀条件的高阶输出空间 mask**：用 $n$ 阶状态跟踪 hidden 轨迹，沿每个节点的草稿前缀做预测-校正，块内外推，一次前向并行处理。
2. **卡尔曼自适应变体（主自适应版本）**：在维度分解的状态空间模型上做卡尔曼滤波，用新息在线估计 $Q$、$R$，代理观测使用 $R'>R$，并由预测协方差 $P$ 给出槽位不确定性。完全免训练，逐序列自适应。
3. **统一层级**：在线 AR（线性预测）⊃ 固定系数 AR（牛顿差分）；常增益滤波（逐阶 EMA / α-β-γ）⊂ 稳态卡尔曼 ⊂ 自适应卡尔曼（§5.10）。
4. **映射**：tied 模型直接用 hidden 并做范数缩放；非 tied 模型加 LatentMAS 适配 [34]。
5. **理论**：
   - 无损性（命题 1）与加速比上界（命题 2）；
   - 统一性（命题 5）与偏差-方差权衡（命题 6、7）；
   - 固定 EMA 与稳态卡尔曼的对应及 $\beta^\star$ 闭式（命题 8）；
   - 稳态增益最优性（命题 9）；
   - 槽位不确定性（命题 10）。
6. **实验计划**（H1–H13），全部待测。

###### 1.5 名称：denoising 的含义
本文把末层 hidden 序列建模为
$$y_t=Hs_t+v_t,$$
即“可预测的平滑趋势 $Hs_t$”加上“与具体 token 相关的、近似不可预测的成分 $v_t$”。

估计 $s_t$ 就是从含噪轨迹中**去噪**：

- EMA 是固定增益的去噪器；
- 卡尔曼滤波是在给定 $Q$、$R$ 下的最优线性去噪器（命题 9）；
- 零阶增益 $\kappa_0<1$ 时，连锚点 hidden 本身也被平滑（v4 中 $\kappa_0=1$，不对锚点去噪）。

因此 “denoising” 指**对输出 hidden 轨迹做状态估计（滤波）去噪**，与扩散模型的 denoising 无关。

仓库目录名 `denoise_spec_mtp` 中 “denoise” 的原始含义是否就是这一点，【待作者确认】。

##### 2. Motivation

**(1) 输出端 hidden 带预测性，输入均值不带。** 末层 hidden $h_i$ 经 LM head 直接给出 $p_\theta(\cdot\mid x_{\le i})$，它在定义上就是**对下一 token 的预测**，同时编码了全部前缀。ESP 的均值 mask $\bar e=\frac1t\sum e(x_i)$ 只是输入 token 的平均，与“下一个是什么”无关。可解释性研究也表明，单个 hidden 可以解码出其后多个 token [36]。

因此，把“已经预测了下一 token”的表示放到 mask 位置，比放一个无预测性的均值更有理由帮助模型填出后续 token。这是动机，不是证据【待测】。

**(2) 一阶速度提供方向。** 零阶锚点只对应紧接着的一个 token。mask $j\ge2$ 需要更远的位置，一阶速度 $v$（hidden 差分的 EMA）给出 hidden 轨迹的局部方向，可以外推到更远的槽位。作者认为这一项有意义，所以作为主方法的一部分；它是否真能提高远槽位的接受率，由消融 E4 检验。

**(3) 前缀条件。** 链上节点 $i$ 的 mask 预测的是 $\tilde x_{1:i}$ 之后的 token。如果所有节点共用同一组 mask（仓库现状，见 §5.11），深节点的 mask 携带的就是“根之后”的信息，与其实际前缀错位。前缀条件让每个节点的 mask 与自己的前缀对齐。

**(4) 可得性。** 所需 hidden 都在上一轮块前向中已经算出，不增加前向次数。

**待测风险：soft 输入的分布偏移。** 末层 hidden 与输入 embedding 的统计分布不同，作为输入可能引起分布外激活。LatentMAS [34] 报告对齐能缓解这一问题（在推理任务上）。在本场景中这只是风险，由 E3 检验，不作为动机。

---


**(5) 高阶信息。** 单个 hidden 只给出“下一个 token”的预测。相邻位置 hidden 的一阶、二阶、三阶差分，刻画的是预测本身随位置的变化趋势：一阶对应方向，二阶对应趋势的转弯。如果这种变化在短窗口内足够平滑，高阶项可以改善远槽位 $j\ge2$ 的外推。

反过来，hidden 轨迹逐 token 跳变，带有大量与具体 token 相关、近似不可预测的成分。差分会放大这部分“噪声”（命题 6）。因此最优阶数是一个经验问题（H10），我们预期不会很高【猜想】。

##### 3. Related Work

文献编号 [1]–[43] 沿用 v3，[44]–[48] 为本版新增（标题已核对）。

- **投机解码与 ESP**：基础与综述 [1][2][14]；训练式 [3]–[7][9][10][16]；免训练 [8][17]–[23][29][30]；树验证 [11][15]；扩散起草 [24]–[26]；ESP 为主基线 [12]。
- **特征层与 hidden 起草（训练式）**：EAGLE [4]–[6] 用训练过的小网络在目标特征上自回归起草；Medusa/Hydra [3][7] 在 hidden 上接训练过的头；Hidden Transfer [35] 训练中间层到未来位置的变换。
  - 本法区别：同样利用目标 hidden，但不训练任何模块，起草仍由目标模型在 mask 位置完成。
- **mask/前瞻 token（训练式）**：PaSS [18]、BiTA [39]、ParallelSpec [40]、Samragh 等 [27]、SPACE [41]。这些方法学习 mask embedding；本法在推理时解析构造 mask。
- **起草器的分布偏移**：
  - Osprey [44] 报告在数学上训练的 EAGLE-3 起草器（Qwen3-8B），域内平均接受长度 5.06，在代码上只有 1.86。
  - OWL [45] 报告 EAGLE-3 在长上下文基准 LongSpecBench 上接受长度约 1.28，低于 PLD 等免训练方法，速度降至 0.81×。
  - P-EAGLE [46] 报告在短序列上训练的起草器在长推理轨迹上接受率最多下降 25%。
  - Online Speculative Decoding [47] 以查询分布偏移为动机，在线更新起草器。
  - HedgeSpec [48] 在多个领域专家起草器之间在线选择，以应对分布外查询。
  - 以上数字均为原文报告值，不是本仓库实测。它们支撑“训练式起草器存在迁移问题”这一定位；本法在同等设置下是否更稳，记为 H8【待测】。
- **连续/soft 输入**：Coconut [31]（训练）、Soft Thinking [32]、Mixture of Inputs [33]、LatentMAS [34]（免训练）、CoCoMix [42]、Prompt tuning [38]。它们用 soft 输入改变生成内容；本法只在不提交的 mask 位置使用 soft 输入，输出逐 token 不变。
- **hidden 解码与外推**：Future Lens [36]、Tuned Lens [37]、LCM [43]、FlowCast [13]。
- **新颖性**：尚未找到“前缀条件的输出空间外推 mask + ESP 式同前向验证”的工作，有待系统检索确认。

---

- **经典外推与滤波**：
  - 卡尔曼滤波 [49] 与常增益 α-β / α-β-γ 跟踪器 [50]（含 tracking index 的稳态增益分析）；
  - 指数平滑及其趋势扩展，即 Holt 线性趋势 [52]、Gardner 综述 [53]、指数平滑的状态空间框架 [54]；
  - 线性预测 [51]。
  
  牛顿前向/后向差分插值与外推是数值分析教材内容，本文只用其标准恒等式，不另引用（教材出处待补）。本法把这些经典工具用在 LLM 末层 hidden 的位置序列上，用于构造投机解码的 mask；目前未见这样的用法（待系统检索）。

- **卡尔曼滤波及其自适应**：
  - 稳态卡尔曼增益由离散代数 Riccati 方程决定 [49]；
  - α-β / α-β-γ 跟踪器的最优增益可由 Kalata 跟踪指数给出闭式 [50]，α-β-γ 情形的解析式见 Gray & Murray [55]；
  - 噪声协方差未知时的自适应估计包括：Mehra 的基于新息的方差辨识 [56][57]、Sage–Husa 的序贯先验统计估计 [58]、Mohamed–Schwarz 的新息协方差匹配 [59]。

  本法把这些方法用在 LLM hidden 轨迹上，构造投机解码的 mask，不训练任何参数。

##### 4. Problem Formulation

###### 4.1 记号

| 记号 | 含义 |
|---|---|
| $W_{\text{in}}=E,\ W_{\text{out}}=W\in\mathbb R^{V\times d}$ | 输入 embedding / LM head；tied 指 $W=E$ |
| $f_\theta$ | 冻结模型，输出逐位置末层 hidden $h$（经最终归一化）与 logits $Wh$ |
| $g(\cdot)$ | 贪心预测 $\arg\max p_\theta$ |
| $k$ | 链长 = 每个节点的 mask 数 |
| $r$ | 根，即已提交的最后一个 token，位置 $p$ |
| $\tilde x_{1:k}$ | 本轮链式草稿 |
| $\Phi$ | 输出到输入的映射 |
| $\bar\nu=\frac1V\sum_w\lVert E_w\rVert$ | 输入 embedding 平均行范数 |

###### 4.2 块、注意力掩码与位置编码（并行）

**块序列**（$N_B=(k+1)(k+1)$ 个位置）：
$$B=\big[\underbrace{n_0,n_1,\dots,n_k}_{\text{节点}}\ \big|\ \underbrace{M^{(0)}_{1..k}}_{\text{节点 0 的 mask}},\dots,\underbrace{M^{(k)}_{1..k}}_{\text{节点 }k\text{ 的 mask}}\big],\qquad n_0=e(r),\ n_i=e(\tilde x_i).$$

**索引**：节点 $n_i$ 在块内的索引为 $i$；$M^{(i)}_j$ 的索引为 $\iota(i,j)=(k+1)+ik+(j-1)$。

**注意力掩码** $\mathcal A\in\{0,-\infty\}^{N_B\times(p+N_B)}$：前缀 KV 共 $p$ 列，全部允许；块内规则为
$$\mathcal A[n_i,n_{i'}]=0\iff i'\le i;$$
$$\mathcal A[M^{(i)}_j,n_{i'}]=0\iff i'\le i;$$
$$\mathcal A[M^{(i)}_j,M^{(i')}_{j'}]=0\iff i'=i\wedge j'\le j;$$
$$\mathcal A[n_i,M^{(\cdot)}_{\cdot}]=-\infty\ \text{（节点不看任何 mask）}.$$

**位置编码**：
$$\mathrm{pos}(n_i)=p+i,\qquad \mathrm{pos}(M^{(i)}_j)=p+i+j.$$
$M^{(i)}_j$ 占据“前缀 $\tilde x_{1:i}$ 之后第 $j$ 个 token”的位置，与 ESP 的约定一致。不同节点的 mask 可以共享同一个 position id（例如 $M^{(0)}_2$ 与 $M^{(1)}_1$），但它们互不注意，各自的前缀不同，所以不冲突。

**并行性**：所有 $M^{(i)}$ 的**内容**都只依赖本轮前向之前已知的量（§5.1），因此 $N_B$ 个位置可在一次前向中并行处理。若下一轮接受了 $A$ 个草稿，就取 $M^{(A)}$ 位置的输出作为下一轮的草稿和状态。

**KV**：只保留前缀、$n_0$ 和 $n_{1..A}$。mask 与被拒节点的 KV 全部丢弃。

###### 4.3 接受长度与目标
贪心接受长度为
$$A=\max\{a\le k:\tilde x_i=g(\text{前缀},r,\tilde x_{<i})\ \forall i\le a\},\qquad\tau=A+1.$$
目标是 $\max\mathbb E[\tau]$，约束为输出与自回归一致。

---

##### 5. Method

###### 5.1 上一轮留下的量与代理观测
设上一轮接受到节点 $A'$。那一轮中节点 $A'$ 的第 $\ell$ 个 mask $M^{(A')}_\ell$ 有两个输出：本轮草稿 $\tilde x_\ell$，以及末层 hidden $\tilde h_\ell$（$\ell=1..k$）。

由于 $M^{(A')}_\ell$ 的位置就是本轮节点 $n_{\ell-1}$ 的位置，$\tilde h_\ell$ 可以作为“节点 $n_{\ell-1}$ 处 hidden”的**代理观测**，它预测的正是 $\tilde x_\ell$。

已提交位置的 hidden 是**精确观测**：它们在被接受节点上已经计算过，见 Alg.2 第 10 行。

###### 5.2 $n$ 阶状态与转移矩阵
**状态。** 状态为 $s=(s^{[0]},\dots,s^{[n]})\in\mathbb R^{(n+1)\times d}$，其中 $s^{[r]}$ 估计当前位置处轨迹的 $r$ 阶**前向**差分 $\Delta^r h$。

**转移。** 预测一步时，各阶前向差分满足 $\Delta^r h_{t+1}=\Delta^r h_t+\Delta^{r+1}h_t$，最高阶 $\Delta^n$ 视为常数（常 $n$ 阶差分模型，对应常加速度等模型）。因此
$$s_{j}=F\,s_{j-1},\qquad F=\begin{pmatrix}1&\gamma_1&&\\&\rho_1&\gamma_2&\\&&\ddots&\ddots\\&&&\rho_n\end{pmatrix}\otimes I_d.$$

- 标准情形 $\gamma_r=\rho_r=1$，即 $F=I+N$，$N$ 为上移位矩阵；
- $\gamma_r$ 为步长；$\rho_r\le1$ 为阻尼，类比阻尼趋势指数平滑 [53]。

**命题（牛顿外推，见命题 5(a)）。** 当 $\gamma=\rho=1$ 时，由二项式定理（$N^{n+1}=0$）得
$$F^j=\sum_{r=0}^n\binom{j}{r}N^r,\qquad \hat h_{t+j}=[F^js]^{[0]}=\sum_{r=0}^n\binom jr\,\Delta^rh_t.$$
这就是牛顿前向差分外推。

**用后向差分表示。** 若用历史观测的后向差分 $\nabla^r h_t$ 表示（$\nabla h_t=h_t-h_{t-1}$），则对 $n$ 次多项式轨迹有
$$\hat h_{t+j}=\sum_{r=0}^n\binom{j+r-1}{r}\nabla^r h_t,$$
即牛顿-格雷戈里后向公式。两种坐标的换算为 $\Delta^r=\sum_{m=0}^{n-r}\binom{r+m-1}{m}\nabla^{r+m}$（截断到 $n$ 阶），由 $\Delta=\nabla(1-\nabla)^{-1}$ 得到。

###### 5.3 观测校正：各阶 EMA（α-β-γ 推广）
**预测-校正。** 收到观测 $y$ 时（精确或代理），先预测再校正：
$$s^-=Fs,\qquad s\leftarrow s^-+K\otimes(y-s^{-[0]}),\qquad K=(\kappa_0,\kappa_1,\dots,\kappa_n)^\top.$$

- $n=1$ 时为 α-β 滤波，$n=2$ 时为 α-β-γ 滤波 [50]；
- 固定 $K$ 相当于稳态（常增益）卡尔曼滤波 [49]；
- $K$ 的每个分量都是作用于全部 $d$ 维的标量，不需要学习。

**各阶独立 β 的写法（等价参数化，主实现）。** 令 $\kappa_0=1$，即零阶直接取观测值（与 v4 锚点一致）。对 $r\ge1$ 逐阶做 EMA：
$$s^{[r]}\leftarrow\beta_r\,s^{-[r]}+(1-\beta_r)\big(s^{[r-1]}_{\text{new}}-s^{[r-1]}_{\text{old}}\big).$$
这里 $s^{[r-1]}_{\text{new}}-s^{[r-1]}_{\text{old}}$ 是低一阶状态在本步的实际变化，所以“各阶差分各自 EMA”。这一写法与 $K$ 形式的对应关系为：

- $n=1$ 时 $\kappa_1=1-\beta_1$，严格成立（命题 5(b)）；
- $n\ge2$ 时两者都是常增益线性滤波器，但具体增益对应关系复杂，只作参数化选择【实现取逐阶 EMA，与 $K$ 形式的数值差异作消融 E15】。

**两类观测分开设增益。**

- 已提交位置是精确观测，用增益 $K$（或 $\beta_r$）；
- 草稿前缀上的代理观测 $\tilde h$ 噪声更大，可以用更保守的增益 $K'$（或 $\beta'_r\ge\beta_r$）。

###### 5.4 沿前缀累积与块内外推
**沿前缀累积。** 设已提交状态为 $s^c$，它的最后一次校正用的是 $h_{\text{last}}=H[n_{A'}]$。节点状态为：
$$s^{(0)}=\mathrm{Corr}_{K'}\big(Fs^c,\ \tilde h_1\big),\qquad s^{(i)}=\mathrm{Corr}_{K'}\big(Fs^{(i-1)},\ \tilde h_{i+1}\big)\ (1\le i\le k-1),\qquad s^{(k)}=Fs^{(k-1)}.$$
链末节点 $k$ 没有代理观测，只做预测。

**块内外推。** 对节点 $i$：
$$M^{(i)}_j=\Phi\big([F_{\text{blk}}^{\,j-1}s^{(i)}]^{[0]}\big),\qquad j=1..k.$$
其中 $F_{\text{blk}}$ 可以与 $F$ 不同（例如另设阻尼 $\rho$），用来消融“块内是否递推、是否阻尼”。块内没有新观测，所以“逐步更新状态”就是 $s_j=F_{\text{blk}}s_{j-1}$。

**特例（命题 5）。**

- $n=0$：零阶-only，$M^{(i)}_j=\Phi(\tilde h_{i+1})$；
- $n=1$，$K=(1,1-\beta)$，$F_{\text{blk}}=\left(\begin{smallmatrix}1&\gamma\\0&\rho\end{smallmatrix}\right)$：即 v4 的一阶递推（$\rho=\beta'+(1-\beta')\gamma$）；
- $\rho=1$：冻结速度。

###### 5.5 一般线性预测器（对照形式）
**形式。**
$$\hat h_{t+1}=\sum_{l=0}^{L}a_l\,h_{t-l},\qquad a\in\mathbb R^{L+1},\ \text{各维共享标量系数}.$$
多步预测时把预测值迭代代入，等价于伴随矩阵 $C(a)$ 的幂。

**(i) 固定系数（由阶数决定）。** 取 $L=n$，令 $\nabla^{n+1}\hat h_{t+1}=0$，得
$$a_l=(-1)^l\binom{n+1}{l+1},\qquad l=0..n.$$
例如 $n=1$ 时为 $2h_t-h_{t-1}$，$n=2$ 时为 $3h_t-3h_{t-1}+h_{t-2}$。系数满足 $\sum_la_l=1$（仿射，保持常数轨迹）。这与 §5.2 的无平滑差分外推完全等价（命题 5(c)）。

**(ii) 在线最小二乘（免训练，逐序列测试时拟合）。** 在最近 $W$ 个已提交位置上拟合：
$$a^\star=\arg\min_a\sum_{\tau=t-W+1}^{t}\omega^{t-\tau}\Big\lVert h_\tau-\sum_l a_lh_{\tau-1-l}\Big\rVert^2+\mu\lVert a-a^{\text{Newton}}\rVert^2.$$

- 闭式解：解一个 $(L+1)\times(L+1)$ 的正规方程，可选约束 $\sum a_l=1$；
- 每步成本 $O(WL^2d)$，或递推最小二乘 $O(L^2d)$；
- 这是 Makhoul 意义下的线性预测 [51]：系数随序列在线估计，不使用任何训练数据。

**与差分形式的关系。** 两者都是线性时不变（或慢变）预测器：

- 固定系数预测器的伴随矩阵 $C(a^{\text{Newton}})$ 与 $F=I+N$ 相似，状态一个是延迟坐标、一个是差分坐标；
- 带 EMA 校正的差分滤波，在稳态下等价于某个无穷阶（指数衰减）的线性预测器。Holt 线性趋势与某类 ARIMA 等价是经典结论，具体出处待核对。

###### 5.6 映射 $\Phi$
与 v4 相同：

- tied 模型：$\Phi(h)=\bar\nu\,h/\lVert h\rVert$；
- 非 tied 模型：$\Phi(h)=\bar\nu\,hW_a/\lVert hW_a\rVert$，其中 $W_a=(W_{\text{out}}^\top W_{\text{out}}+\lambda I)^{-1}W_{\text{out}}^\top W_{\text{in}}$ [34]。

由于外推是线性的，非 tied 时可以先对状态各分量乘 $W_a$ 再外推，每轮只需 $(k+1)(n+1)$ 次 $d\times d$ 乘法。

###### 5.7 开销

| 项目 | 存储 | 每轮计算 |
|---|---|---|
| 已提交状态 $s^c$ | $(n+1)d$ | 每提交一个 token 做一次校正，$O(n d)$ |
| 节点状态 $s^{(0..k)}$ | $(k+1)(n+1)d$ | $O(k\,n\,d)$ |
| 块内外推 | — | $(k+1)k$ 个 mask，每个 $O(nd)$，合计 $O(k^2nd)$ |
| 非 tied 映射 | $d^2$（$W_a$） | $O((k+1)(n+1)d^2)$ |
| 线性预测 LS（可选） | $Wd$（最近 $W$ 个 hidden） | $O(WL^2d)$ |
| 块前向（参照） | — | 约 $2N(k+1)^2$ |

例：$k=4$，$n=3$，$d=4096$，非 tied 映射约 $20\cdot2d^2\approx6.7\times10^8$ FLOPs，块前向（8B）约 $2\cdot8\times10^9\cdot25=4\times10^{11}$，比值约 0.2%（估算，非实测）。

只需额外缓存“最近 $n+1$ 个已提交 hidden”或等价的状态，以及上一轮 $M^{(A')}$ 的 $k$ 个 hidden。

###### 5.8 算法
**Algorithm 1：$n$ 阶前缀条件 mask 构造**
```text
Input : committed state s^c ∈ ℝ^{(n+1)×d}; proxies h̃_1..h̃_k; F, F_blk; gains K′ (or β′_1..β′_n); Φ
Output: M⁽⁰⁾..M⁽ᵏ⁾ ∈ ℝ^{k×d}  (fixed before the forward)
 1  s ← s^c
 2  for i ← 0 to k do                                  ▷ prefix accumulation along the draft
 3      s⁻ ← F·s
 4      if i < k then s ← Corr(s⁻, h̃_{i+1}; K′)         ▷ s⁻ + K′⊗(h̃_{i+1} − s⁻[0]) or per-order EMA
 5      else s ← s⁻                                    ▷ chain end: predict only
 6      s⁽ⁱ⁾ ← s
 7  for i ← 0 to k do                                  ▷ independent → parallel
 8      u ← s⁽ⁱ⁾
 9      for j ← 1 to k do
10          M⁽ⁱ⁾_j ← Φ(u[0]);  u ← F_blk·u                ▷ s_j = F s_{j−1}
11  return M⁽⁰⁾..M⁽ᵏ⁾
```

**Algorithm 2：链式解码（贪心）**
```text
Input : prompt x_{1:t}; frozen f_θ; k; n; Φ; N_max
Output: Y (identical to greedy AR, Prop. 1)
 1  if untied: W_a ← solve(WᵀW + λI, WᵀE) in fp32
 2  (H, L) ← f_θ([e(x_{1:t}) | ē×k])                     ▷ prefill
 3  s^c ← init from H[t−n..t−1] (backward differences → Δ via §5.2)
 4  r ← argmax L[t−1]; x̃ ← argmax L[masks]; h̃_{1..k} ← H[masks]; Y ← [r]
 5  while |Y| < N_max and r ≠ EOS do
 6      M ← Alg1(s^c, h̃, F, F_blk, K′, Φ)
 7      (H, L) ← f_θ([e(r), e(x̃_{1..k}) | M⁽⁰⁾..M⁽ᵏ⁾]; 𝒜, pos as §4.2)
 8      A ← max{a : argmax L[n_{i−1}] = x̃_i ∀ i ≤ a}
 9      C ← [x̃_{1..A}, argmax L[n_A]]; truncate; Y ← Y ∥ C
10      for i ← 0 to A: s^c ← Corr(F·s^c, H[n_i]; K)      ▷ exact observations
11      h̃_{1..k} ← H[M⁽ᴬ⁾]; x̃ ← argmax L[M⁽ᴬ⁾]; r ← last(C)
12      keep KV of prefix + n_0..n_A
13  return Y
```
第 10 行：已提交状态在每个被接受节点处用精确 hidden 校正一次。bonus token 的 hidden 要等下一轮根节点才算出，因此有一步延迟。

###### 5.9 卡尔曼自适应变体（主自适应版本）
**状态空间模型**：
$$s_{t+1}=F s_t+w_t,\quad w_t\sim\mathcal N(0,\mathbf Q);\qquad y_t=Hs_t+v_t,\quad v_t\sim\mathcal N(0,\mathbf R);\qquad H=[1,0,\dots,0].$$
其中 $s_t\in\mathbb R^{(n+1)\times d}$，$y_t$ 为观测到的末层 hidden。

**维度分解假设**：取 $\mathbf Q=Q\otimes I_d$，$\mathbf R=R\,I_d$，其中 $Q\in\mathbb R^{(n+1)\times(n+1)}$，$R>0$ 为标量。这样完整协方差为 $P\otimes I_d$，$P\in\mathbb R^{(n+1)\times(n+1)}$，滤波沿 $d$ 维解耦为 $d$ 个同参数的 $(n+1)$ 维滤波器，并共享同一个 $P$ 和 $K$。

- **合理性**：
  - 计算上，$P$ 只有 $(n+1)^2$ 个元素，增益是标量向量，与 v5 的“各维共享标量增益”一致；
  - 统计上，跨维度相关被忽略，各维噪声尺度被假设相同。
- **已知不成立之处**：
  - 末层 hidden 经 RMSNorm 后乘以逐维可学习权重，各维尺度不同，且常有少数大幅值维度；
  - hidden 具有各向异性。
- **可选扩展**（均仍可解耦）：
  - 逐维：$\mathbf R=\mathrm{diag}(R_1..R_d)$，$\mathbf Q=\mathrm{blockdiag}(Q_m)$，得到 $d$ 个独立小滤波器，成本 $O(dn^2)$；
  - 分组：$G$ 组共享参数；
  - 预白化：先除以 RMSNorm 权重，再用同参数滤波。
- 三种取法的比较见 E17。

**预测-校正**（每维相同，写成 $(n+1)$ 维形式）：
$$\text{预测：}\ s^-=Fs,\qquad P^-=FPF^\top+Q;$$
$$\text{校正：}\ e=y-Hs^-,\quad S=HP^-H^\top+R,\quad K=P^-H^\top S^{-1},\quad s=s^-+K\otimes e,\quad P=(I-KH)P^-(I-KH)^\top+KRK^\top.$$
$P$ 的更新采用 Joseph 形式，以保持对称正定。

**稳态增益**：若 $(F,H)$ 可观测、$(F,Q^{1/2})$ 可稳，则 $P^-$ 收敛到离散代数 Riccati 方程的解
$$P^-_\infty=FP^-_\infty F^\top-FP^-_\infty H^\top(HP^-_\infty H^\top+R)^{-1}HP^-_\infty F^\top+Q,\qquad K^\star=P^-_\infty H^\top(HP^-_\infty H^\top+R)^{-1}.$$
$F=I+N$、$H=[1,0..]$ 构成可观测对，这是标准结论；具体收敛条件以教材为准，教材出处待补。

**与 α-β 的对应（Kalata）**：对“离散白噪声加速度”模型（$n=1$，$Q=\sigma_w^2GG^\top$，$G=(T^2/2,T)^\top$，$R=\sigma_v^2$），稳态增益 $K^\star=(\alpha,\beta/T)$ 由跟踪指数 $\lambda=\sigma_wT^2/\sigma_v$ 决定。常见引述形式为：
$$r=\frac{4+\lambda-\sqrt{8\lambda+\lambda^2}}{4},\qquad\alpha=1-r^2,\qquad\beta=2(2-\alpha)-4\sqrt{1-\alpha}.$$
α-β-γ 情形另有 $\gamma=\beta^2/(2\alpha)$ 等关系 [55]。

**核对说明**：上式经二手来源（Wikipedia “Alpha beta filter” 条目与 ControlSystems.jl 文档，均注明出自 Kalata 1984 [50] 与 Gray & Murray 1993 [55]）核对，**未读到原文，记为待核对**。另外，增益约定存在因子 2 的差异（$\gamma$ 与 $2\gamma$），以原文为准。

**新息自适应的 $Q$、$R$ 在线估计**（免训练、逐序列；下列公式为常见引述形式，具体细节以原文为准，待核对）：

- **新息协方差匹配**（Mohamed–Schwarz [59]）：在窗口 $W$ 内计算标量化的新息二阶矩
  $$\hat C_t=\frac1{Wd}\sum_{\tau=t-W+1}^t\lVert e_\tau\rVert^2.$$
  由此估计
  $$\hat R=\max\big(\hat C_t-HP^-H^\top,\ r_{\min}\big),\qquad \hat Q=K\hat C_tK^\top\ (+\epsilon I).$$
- **Sage–Husa 型递推**（[58]，带遗忘因子 $b\in(0,1)$，$\delta_t=(1-b)/(1-b^{t+1})$）：
  $$\hat R_t=(1-\delta_t)\hat R_{t-1}+\delta_t\big(\tfrac1d\lVert e_t\rVert^2-HP^-H^\top\big),$$
  $$\hat Q_t=(1-\delta_t)\hat Q_{t-1}+\delta_t\big(K\tfrac1d\lVert e_t\rVert^2K^\top+P-FP_{t-1}F^\top\big).$$
  每步投影回正定锥（特征值截断），并设下界。
- **Mehra 型**（[56][57]）：用新息序列的自相关检验和辨识 $Q$、$R$。新息白则模型一致，可作为诊断指标，也用于 E17。
- **初始化**：用 prompt 段的已提交 hidden 预热（只做校正、不起草）。也可由 prompt 段的 $\lVert\nabla^rh\rVert$ 给出 $Q$、$R$ 的矩估计，此时可借助命题 6：在独立噪声模型下，$\mathbb E\lVert\nabla^{n+1}y\rVert^2=\lVert\nabla^{n+1}p\rVert^2+\binom{2n+2}{n+1}\sigma^2d$。

**两类观测**：

- 已提交位置：精确 hidden，观测噪声 $R$；
- 草稿代理 $\tilde h$：观测噪声 $R'=R+R_\delta$，$R_\delta\ge0$，相当于 $R'>R$。

$R_\delta$ 可以免训练地在线估计。一个节点被接受后，它的精确 hidden $H[n_i]$ 就可以与上一轮的代理 $\tilde h_{i+1}$ 比较，因为两者对应同一位置：
$$\hat R_\delta\leftarrow\text{EMA}\Big(\tfrac1d\lVert H[n_i]-\tilde h_{i+1}\rVert^2\Big).$$
沿节点前缀累积时用 $R'$ 校正（§5.4），块内只预测。

**槽位不确定性（扩展）**：节点 $i$ 在块内第 $j$ 步的预测协方差为
$$P^{(i)}_j=F^{\,j-1}P^{(i)}F^{(j-1)\top}+\sum_{m<j-1}F^mQF^{m\top},\qquad \sigma^{2}_{i,j}=HP^{(i)}_jH^\top$$
（每维）。可能的用法：

- (a) 自适应草稿长度：取 $k_{\text{eff}}=\max\{j:\sigma^2_{0,j}\le\theta\}$，需配合注意力块的变长或截断；
- (b) 自适应 mask 温度：在 M2 期望 embedding 映射中令 $T_j\propto\sigma_{i,j}$；
- (c) 作为接受概率的特征。

以上均为**扩展**【待测】，不进入主结果。$\sigma^2$ 与真实接受率的关系是猜想 C13。

**Algorithm 3：卡尔曼 mask provider**
```text
State : s^c ∈ ℝ^{(n+1)×d}, P^c ∈ ℝ^{(n+1)×(n+1)}, Q, R, R_δ, innovation window 𝓦
Init(prompt hiddens y_1..y_t):
 1  s^c ← backward-difference init from y_{t−n..t}; P^c ← p_0·I; Q,R ← moment init (Prop. 6) or defaults
 2  for τ in warm-up positions: (s^c, P^c, e) ← KF(s^c, P^c, y_τ, R); push e to 𝓦
ObserveCommitted(y):                                   ▷ exact hidden of an accepted node
 3  (s^c, P^c, e) ← KF(s^c, P^c, y, R); push e to 𝓦
 4  (Q, R) ← AdaptQR(𝓦, P^c, K)                         ▷ covariance matching / Sage–Husa; clamp to PSD, R ≥ r_min
ObserveProxyError(y_exact, h̃_proxy):
 5  R_δ ← EMA(‖y_exact − h̃_proxy‖²/d)
Masks(proxies h̃_1..h̃_k):                              ▷ called before the block forward
 6  s ← s^c; P ← P^c
 7  for i ← 0 to k do
 8      s ← F·s; P ← F P Fᵀ + Q
 9      if i < k then (s, P, _) ← KF_update(s, P, h̃_{i+1}, R + R_δ)
10      (s⁽ⁱ⁾, P⁽ⁱ⁾) ← (s, P)
11  for i ← 0 to k do                                  ▷ parallel over nodes
12      u ← s⁽ⁱ⁾; Σ ← P⁽ⁱ⁾
13      for j ← 1 to k do
14          M⁽ⁱ⁾_j ← Φ(u[0]); σ²_{i,j} ← Σ[0,0]              ▷ σ² optional (extensions)
15          u ← F_blk·u; Σ ← F_blk Σ F_blkᵀ + Q
16  return M, σ²
KF_update(s, P, y, R):  e ← y − s[0]; S ← P[0,0] + R; K ← P[:,0]/S
                        s ← s + K⊗e; P ← (I−K hᵀ)P(I−K hᵀ)ᵀ + K R Kᵀ  (h = e_0, Joseph); return (s, P, e)
```
（KF = 预测 + KF\_update。）

**复杂度**：

- 协方差递推与 $d$ 无关，每步 $O(n^3)$；$n\le3$ 时可以忽略。
- 状态更新每步 $O(nd)$。
- 每轮：
  - 前缀累积 $O(k(nd+n^3))$；
  - 块内外推 $O(k^2(nd+n^3))$；
  - 自适应估计 $O(d+n^3)$。
- 逐维扩展把协方差部分乘以 $d$，变为 $O(dn^3)$。
- 与 v5 的固定增益版相比，额外开销只有 $n$ 维小矩阵运算。

**数值稳定性**：

- 状态与协方差用 fp32；
- $P$ 用 Joseph 形式更新，每步对称化 $P\leftarrow(P+P^\top)/2$；
- $R$、$Q$ 设下界并投影回正定锥；
- 高阶（$n\ge2$）下 $F$ 的幂会放大，块内可用阻尼 $\rho<1$；
- $S=P_{00}+R$ 为标量，不需要矩阵求逆。

###### 5.10 三种预测器的层级关系

| 层级 | 形式 | 系数 / 增益来源 | 假设 | 免训练 |
|---|---|---|---|---|
| 在线线性预测（小 AR） | $\hat h_{t+1}=\sum_{l\le L}a_lh_{t-l}$，$a$ 由逐序列在线（加权）最小二乘拟合 | 推理时逐序列拟合 | 局部平稳的线性动态 | 是 |
| 牛顿差分（固定 AR） | $a_l=(-1)^l\binom{n+1}{l+1}$；等价于 $F=I+N$、不平滑 | 由阶数 $n$ 唯一决定 | 局部 $n$ 次多项式、无噪声 | 是 |
| 常增益滤波（逐阶 EMA / α-β-γ） | $s\leftarrow Fs+K\otimes(y-[Fs]_0)$，$K$ 固定 | 手设或网格搜索 $\beta$ | 多项式趋势 + 噪声，增益固定 | 是（但需选参） |
| 稳态卡尔曼 | 同上，$K=K^\star(Q,R)$ 解自 DARE | 由 $(Q,R)$ 决定 | 线性高斯状态空间模型，$Q$、$R$ 已知 | 是 |
| 自适应卡尔曼 | 时变 $K_t$，$Q$、$R$ 由新息在线估计 | 逐序列在线估计 | $Q$、$R$ 未知、慢变 | 是 |

```text
 online AR (fit a_l per sequence) ──special case: fixed a_l = Newton──▶ Newton difference (fixed AR, no smoothing)
          ▲ steady-state Kalman ≡ infinite-order AR with geometrically decaying a_l (classical equivalence, ref. TBD)
 constant-gain filter (EMA / α-β-γ) ──K chosen optimally for (Q,R)──▶ steady-state Kalman ──Q,R estimated online──▶ adaptive Kalman
 Newton difference = difference-state filter without smoothing (β_r = 0, Prop. 5c); fixed EMA (n=1) = steady-state Kalman of local-linear-trend, R→0 (Prop. 8)
```

一句话概括三者：

- 线性预测器是推理时逐序列在线拟合的小 AR；
- 差分是局部多项式假设下系数固定的 AR；
- 卡尔曼是带最优（且可在线自适应）增益的状态空间版本，固定 EMA 是它的稳态特例（命题 8）。

###### 5.11 实现说明（仓库只读核查，2026-10-13）
**现状**：

1. **每个节点已有自己的 $k$ 个 mask 槽位，结构上已按前缀并行。**
   - `tree_attention.block_layout` 的块布局为 `[root | node_1..node_N | masks(root) | masks(node_1) | … ]`，每个 owner 有 $k$ 个 mask；
   - mask $j$ 注意 owner 的注意集（前缀 + 祖先 + owner 自身）、同 owner 的 mask $1..j-1$ 和自身；
   - 位置为 `rel[owner] + j + 1`；
   - 节点不注意 mask。
   
   这与 §4.2 的 $\mathcal A$ 和 pos 一致（链式即 `branches: [1]*k`）。下一轮种子取最深接受节点的 mask（`mask_indices_of(tree, k, n_star)`）。

2. **但 mask 的内容在所有 owner 之间完全相同。**
   - `decoding.py` 中 `masks = provider.masks()` 返回 $[k,d]$，随后执行 `masks.unsqueeze(0).expand(1 + num_nodes, k, -1)`，把同一组向量复制给每个 owner；
   - provider 状态只通过 `on_token_committed(new_embed, prev_embed)` 用**已提交** token 的**输入** embedding 更新，不看草稿前缀，也不接收 hidden。

3. **hidden**：`HFCausalLM.forward` 有 `output_hidden_states` 参数，但 `SpecDecoder` 不请求 hidden。末项是否经过最终归一化尚待核对。

**要改什么**（仅列清单，未改代码）：

| 改动 | 位置 | 说明 |
|---|---|---|
| provider 接口返回 $[1+N,k,d]$，输入为草稿前缀与代理 hidden | `mask_providers.py` | 新增 `masks_per_owner(draft_tokens, draft_hiddens, ...)`；保留旧接口供 ESP 使用 |
| 用逐 owner 张量替换 `expand` | `decoding.py`（块 embedding 拼接处） | 按 `block_layout` 的 owner 顺序 reshape |
| 前向请求末层 hidden，保存 $H[n_A]$ 与 $H[M^{(A)}]$ | `decoding.py`、`hf_causal.py` | 只取末层，避免保留全部层 |
| 状态更新改用 hidden（已提交节点的精确 hidden） | `decoding.py` 提交循环 | 替代 `on_token_committed(embed)` 或与之并存 |
| 非 tied 时预计算 $W_a$ 与 $\bar\nu$ | 新模块 | fp32，一次 |
| 注意力掩码与位置编码 | `tree_attention.py` | **无需改动**：结构只依赖树形状，efficient 缓存路径仍然适用 |

树形扩展时，节点前缀为其祖先路径，按祖先链折叠状态即可。

**对此前说法的纠正**：v3 §11 写的“所有 owner 共用同一组 mask”，就 mask 向量的**数值**而言准确；但没有说明注意力和位置**已经**按 owner（前缀）独立、天然并行。因此，实现前缀条件 mask 只需改 mask 的内容（provider 与拼接），不需要改注意力或位置编码。

---


**高阶实现补充（v5，未改代码）**：

- **缓存**：provider 维护已提交状态 $s^c\in\mathbb R^{(n+1)\times d}$（或最近 $n+1$ 个已提交 hidden 的环形缓冲），并保留上一轮 $H[M^{(A)}]$ 的 $k$ 个 hidden。
- **接口**：新增 `observe_committed(hidden)` 与 `masks_per_owner(proxy_hiddens) -> [1+N,k,d]`。
- **配置**：`order n`、`F/F_blk`（$\gamma_r,\rho_r$）、`K` 与 `K′`（或 $\beta_r,\beta'_r$）、`predictor ∈ {difference, linear_fixed, linear_ls}`。
- **需要改的文件**：`mask_providers.py`、`decoding.py`、`hf_causal.py`（只请求末层 hidden）；`tree_attention.py` 不变。
- **数值**：建议用 fp32 保存状态。高阶差分在 bf16 下抵消误差大，需在 E12 检查。

##### 6. Theoretical Analysis

###### 命题 1（soft mask 下的无损性）
**条件**：节点输入为离散 token embedding；注意力满足 $\mathcal A$（节点不看 mask）；$\mathrm{pos}(n_i)=p+i$；算术精确。

**结论**：对任意 mask 内容（包括逐节点不同的 soft 向量），

- (a) 贪心输出与自回归逐 token 一致；
- (b) 以 mask 分布 $q$ 作草稿分布、按 $\min(1,p/q)$ 接受并在拒绝时从残差分布重采样 [1][2]，输出与自回归采样同分布。

**证明**：节点 logits 与 mask 无关，等于自回归在前缀 $(\cdot,\tilde x_{1:i})$ 上的条件分布。草稿分布 $q$ 在本轮之前已经确定，所以标准投机采样引理逐位置适用。KV 只保留离散输入的节点。$\square$

浮点下以 exact\_match 检验。仓库现有的 $T>0$ 路径不是残差采样，(b) 需要新实现。

###### 命题 2（链式加速比上界）
$$S\le\frac{\mathbb E[\tau]\,c_1}{c_B+c_\Phi}\le\mathbb E[\tau]\le k+1.$$
其中 $c_1$ 为单 token 前向开销，$c_B$ 为块前向开销，$c_\Phi$ 为映射开销。

###### 命题 3（hidden 与 logit 外推等价）
外推是线性组合，而 $W$ 是线性映射，因此 $W\hat a_j$ 等于在 logits 上做同样外推的结果。在分布空间外推不等价。$\square$

###### 命题 4（tied 下岭回归对齐近似恒等）
若 $W_{\text{out}}=W_{\text{in}}=E$，则 $W_a=(E^\top E+\lambda I)^{-1}E^\top E$，且
$$\lVert W_a-I\rVert_2=\frac{\lambda}{\sigma_{\min}^2(E)+\lambda}.$$
当 $\lambda\ll\sigma_{\min}^2$ 时 $W_a\approx I$。所以在 tied 模型上岭回归映射几乎不起作用，直接用 hidden 加范数缩放即可。$\sigma_{\min}$ 的实际数值待测（E8 可顺带计算）。


###### 命题 5（统一：零阶、一阶、线性预测都是特例）
(a) **牛顿外推。** 当 $\gamma_r=\rho_r=1$ 时，$[F^js]^{[0]}=\sum_{r\le n}\binom jr s^{[r]}$。

*证明*：$F=I+N$，$N$ 幂零（$N^{n+1}=0$），且 $N$ 与 $I$ 可交换，由二项式展开即得。$\square$

(b) **v4 是 $n=1$ 的特例。** 取 $n=1$，$\kappa_0=1$，$\kappa_1=1-\beta$，$F=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$。则校正后零阶等于观测值，一阶为 $v\leftarrow\beta v+(1-\beta)(y-y_{\text{prev}})$，正是 v4 的速度 EMA。块内用 $F_{\text{blk}}=\left(\begin{smallmatrix}1&\gamma\\0&\rho\end{smallmatrix}\right)$，得到 v4 的 $\hat a_j=a+\gamma c_j(\rho)v$。$n=0$ 时为零阶-only。

*证明*：

- 预测：$s^{-[0]}=y_{\text{prev}}+v$，$s^{-[1]}=v$；
- 零阶校正：$s^{[0]}=s^{-[0]}+1\cdot(y-s^{-[0]})=y$；
- 一阶校正：$s^{[1]}=v+(1-\beta)(y-y_{\text{prev}}-v)=\beta v+(1-\beta)(y-y_{\text{prev}})$；
- 块内：$F_{\text{blk}}^{\,j}$ 的 $(0,1)$ 元为 $\gamma\sum_{s<j}\rho^s$。$\square$

(c) **固定系数线性预测器等价于无平滑差分外推。** 若各阶状态直接取历史观测的后向差分（无 EMA，$\beta_r=0$），则一步外推 $\hat h_{t+1}$ 等于 $a_l=(-1)^l\binom{n+1}{l+1}$ 的线性预测器。

*证明*：由 $\nabla^{n+1}\hat h_{t+1}=\sum_{m=0}^{n+1}(-1)^m\binom{n+1}m h_{t+1-m}=0$ 解出 $\hat h_{t+1}$。$\square$

###### 命题 6（一步外推的偏差-噪声放大分解）
**设定**：观测 $h_\tau=p_\tau+\varepsilon_\tau$。其中 $p$ 为确定性轨迹；$\varepsilon_\tau$ 独立、零均值、协方差 $\sigma^2I_d$。考虑无平滑 $n$ 阶外推（命题 5(c)）$\hat h_{t+1}=\sum_{l=0}^n a_lh_{t-l}$。

**结论**：
$$\mathbb E\lVert\hat h_{t+1}-p_{t+1}\rVert^2=\underbrace{\lVert\nabla^{n+1}p_{t+1}\rVert^2}_{\text{偏差}^2}+\underbrace{\sigma^2d\Big[\tbinom{2n+2}{n+1}-1\Big]}_{\text{噪声放大}}.$$

*证明*：

1. 由命题 5(c)，$p_{t+1}-\sum_la_lp_{t-l}=\nabla^{n+1}p_{t+1}$。
2. 因此 $\hat h_{t+1}-p_{t+1}=-\nabla^{n+1}p_{t+1}+\sum_la_l\varepsilon_{t-l}$。
3. 交叉项期望为 0；噪声项方差为 $\sigma^2d\sum_la_l^2$。
4. $\sum_{l=0}^n\binom{n+1}{l+1}^2=\sum_{m=1}^{n+1}\binom{n+1}m^2=\binom{2n+2}{n+1}-1$（范德蒙德恒等式）。$\square$

**数值**：噪声放大因子 $\binom{2n+2}{n+1}-1$ 在 $n=0,1,2,3$ 时分别为 $1,5,19,69$，按约 $4^{n+1}$ 增长。

**推论（最优阶数条件）**：阶数从 $n$ 增加到 $n+1$ 能降低一步 MSE，当且仅当
$$\lVert\nabla^{n+1}p\rVert^2-\lVert\nabla^{n+2}p\rVert^2>\sigma^2d\Big[\tbinom{2n+4}{n+2}-\tbinom{2n+2}{n+1}\Big].$$
所以只有当轨迹的高阶差分随阶数快速衰减、且噪声较小时，高阶才有益。

###### 命题 7（多步外推的方差增长）
同样设定下，$j$ 步无平滑外推可写为 $\hat h_{t+j}=\sum_{l=0}^n\ell_l(j)\,h_{t-l}$。其中 $\ell_l$ 是以节点 $\{0,-1,\dots,-n\}$ 为插值点的拉格朗日基函数在 $j$ 处的值。噪声方差为 $\sigma^2d\sum_l\ell_l(j)^2$。

由于每个 $\ell_l$ 是 $n$ 次多项式，方差随 $j$ 以 $O(j^{2n})$ 增长。

*证明*：$n$ 次多项式外推是拉格朗日插值在 $j$ 处取值，其余同命题 6。$\square$

**含义**：远槽位的高阶外推方差增长更快，这为块内阻尼 $\rho<1$ 或低阶块内外推提供了理由。

###### 猜想 C11（平滑的作用）
EMA 校正（$\beta_r>0$）降低噪声放大，代价是滞后偏差。存在使 $\mathbb E[A]$ 最大的 $(n,\beta_{1..n})$ 组合，且最优 $n$ 较小（$\le2$）。命题 6、7 只覆盖无平滑情形，带平滑的稳态误差分析（参照 [50] 的 tracking index）未完成。

###### 猜想 C12（hidden 噪声模型）
末层 hidden 的逐 token 变化中，“可预测趋势”占比有限，独立噪声假设只是近似。可以在 E13 中用已提交 hidden 的 $\lVert\nabla^r h\rVert$ 随 $r$ 的变化，经验估计命题 6 中偏差项与噪声项的相对大小。若 $\lVert\nabla^rh\rVert$ 随 $r$ 按 $\approx\sqrt{\binom{2r}{r}}$ 增长，说明以噪声为主。


###### 命题 8（固定 EMA 是稳态卡尔曼的特例；$\beta^\star$ 闭式）
**设定**：$n=1$，$F=\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$，$H=[1,0]$。取“局部线性趋势”模型 $Q=\mathrm{diag}(q_0,q_1)$，其中 $q_0>0$ 为水平噪声、$q_1>0$ 为斜率噪声，并令 $R\to0$（观测无噪声）。

**结论**：稳态卡尔曼增益为 $K^\star=(1,\ \kappa_1^\star)$，其中
$$\kappa_1^\star=\frac{\pi}{1+\pi},\qquad \pi=\frac{\phi+\sqrt{\phi^2+4\phi}}{2},\qquad\phi=\frac{q_1}{q_0}.$$
因此 v4 的固定 EMA（$\kappa_0=1$，$\kappa_1=1-\beta$）恰是该模型的稳态卡尔曼，且
$$\beta^\star=1-\kappa_1^\star=\frac1{1+\pi(\phi)}.$$
$\beta^\star$ 随信噪比 $\phi$ 单调递减：$\phi\to0$（斜率几乎不变）时 $\beta^\star\to1$，即长记忆；$\phi\to\infty$ 时 $\beta^\star\to0$。

**证明**：

1. $R=0$ 时，校正后位置方差为 0，所以稳态 $P=\mathrm{diag}(0,p)$。
2. 预测协方差 $P^-=FPF^\top+Q=\left(\begin{smallmatrix}p+q_0&p\\p&p+q_1\end{smallmatrix}\right)$。
3. 增益 $K=P^-H^\top/P^-_{00}=(1,\,p/(p+q_0))$。
4. 校正后斜率方差为 $p+q_1-p^2/(p+q_0)$；稳态要求它等于 $p$，得 $p^2-q_1p-q_1q_0=0$，正根 $p=q_0\pi$。
5. 代入得 $\kappa_1^\star=\pi/(1+\pi)$。
6. 校正后 $s^{[0]}=y$，$s^{[1]}=v+\kappa_1(y-y_{\text{prev}}-v)=(1-\kappa_1)v+\kappa_1\nabla y$，即 EMA，其中 $\beta=1-\kappa_1$。$\square$

**含义**：

- 网格搜索 $\beta$，等价于在“无观测噪声的局部线性趋势模型”族中搜索信噪比 $\phi$；
- $R>0$ 时最优 $\kappa_0<1$，即锚点本身也被去噪。这一自由度超出了 v4/v5 的固定 EMA 族，是卡尔曼版本可能带来增益的理论来源（H12）；
- $n\ge2$ 的逐阶 EMA 是否都能作为某个对角 $Q$ 的稳态卡尔曼，未证明，记为猜想 C14。

**与 Kalata 的关系**：Kalata [50] 的离散白噪声加速度模型（$Q$ 秩 1）在 $R>0$ 时给出 $\alpha<1$ 的闭式（§5.9，待核对）。本命题用的是另一种 $Q$ 结构（对角、$R\to0$），两者都是 α-β 族的稳态卡尔曼特例。

###### 命题 9（稳态增益的均方最优性，最优 β 存在）
在线性高斯模型下，给定 $(Q,R)$：

- 卡尔曼滤波在所有线性（实际上是所有可测）估计器中使预测均方误差最小 [49]；
- 在所有使误差动态稳定的常增益滤波 $K$ 中，$K^\star$（DARE 解）使稳态预测误差协方差最小（标准结论，教材出处待补）。

**推论**：对命题 8 的模型族，最优 $\beta^\star(\phi)$ 存在且唯一。若真实 hidden 轨迹近似符合某个 $(Q,R)$，则网格搜索得到的最优 $\beta$ 应接近 $\beta^\star(\hat\phi)$，可在 E16 中检验。

**意义**：最优 $\beta$ 由数据的噪声结构决定，不是超参数。自适应卡尔曼逐序列估计 $(Q,R)$，等价于逐序列选取 $\beta$，因此有望跨数据集免调参（H12）。

###### 命题 10（槽位预测方差的递推）
块内预测满足 $P^{(i)}_{j+1}=F_{\text{blk}}P^{(i)}_jF_{\text{blk}}^\top+Q$，因此
$$\sigma^2_{i,j+1}=[F_{\text{blk}}P^{(i)}_jF_{\text{blk}}^\top]_{00}+Q_{00}.$$
当 $F_{\text{blk}}=I+N$、$n\ge1$ 时，$[FPF^\top]_{00}=P_{00}+2P_{01}+P_{11}$。若 $P_{01}\ge-P_{11}/2$（例如 $P$ 由 Riccati 递推得到且位置与斜率误差非负相关），则 $\sigma^2_{i,j}$ 关于 $j$ 严格递增（$Q_{00}>0$ 时）。

**证明**：$F=I+N$ 的第 0 行为 $(1,1,0,\dots)$，代入二次型即得。$\square$

一般条件下的单调性未证明，记为猜想 C15。

###### 猜想（卡尔曼）

- **C13**：$\sigma^2_{i,j}$ 与逐槽接受率负相关，可用于自适应 $k$ 或温度。
- **C14**：$n\ge2$ 的逐阶 EMA 可以表示为某对角 $Q$、$R\to0$ 模型的稳态卡尔曼。
- **C15**：不加协方差符号条件时，命题 10 的单调性是否仍然成立。
- **C16**：由数据估计的 $(\hat Q,\hat R)$ 可预测最优阶数。具体做法是按命题 6 的推论，用 $\hat Q$ 的高阶分量与 $\hat R$ 之比判定增阶是否有益。

###### 猜想

- **C6（预测性）**：$\mathbb E[A]$ 满足 输出空间零阶 > ESP 均值。
- **C9（一阶）**：一阶外推在远槽位（$j\ge2$）上提高逐槽接受率。
- **C10（前缀条件）**：前缀条件优于共享状态，且增益随节点深度 $i$ 增大。
- **C7（分布偏移风险）**：soft 输入可能引起分布外激活；范数缩放能否缓解，待测。

---

##### 7. Hypotheses
记 $\Delta_x(\pi,\pi')$ 为逐 prompt 接受数之差。检验采用确认集上的配对 bootstrap 95% CI + Wilcoxon，对多个假设做 Holm 校正。

- **H1（主效应）**：$\Delta(\text{Ours},\text{ESP})>0$，在至少 2 个规模、至少 2 个数据切片上成立。
- **H2（零阶预测性）**：$\Delta(\text{零阶-only},\text{ESP})>0$。
- **H3（一阶）**：$\Delta(\text{零阶+一阶},\text{零阶-only})>0$，且增益集中在 $j\ge2$。
- **H4（前缀条件）**：$\Delta(\text{前缀条件},\text{共享状态})>0$，且逐节点深度的增益单调非减（描述性）。
- **H5（块内递推）**：在 $\rho\ne1$ 时，递推优于冻结。负对照为 $\gamma=1$ 或 $k=1$。
- **H6（映射）**：
  - tied：hidden + 范数缩放 ≥ 原始 hidden；
  - 非 tied：$W_a$ 映射 ≥ 直接 hidden；
  - 负对照：tied 上 $W_a$ 映射与恒等无差异。
- **H7（偏移判别）**：相对固定偏移基线 $\pi_{\text{off}}$ 有显著增益。
- **H8（跨域稳定性）**：定义域外保留率 $\mathrm{R}=\mathbb E[\tau]_{\text{OOD}}/\mathbb E[\tau]_{\text{ID}}$，其中 ID/OOD 以训练式基线的训练分布划分。假设 $\mathrm R_{\text{Ours}}>\mathrm R_{\text{EAGLE-3}}$，且 OOD 上 $\mathbb E[\tau]_{\text{Ours}}$ 与 EAGLE-3 的差距小于 ID 上的差距。由于本法免训练，其 $\mathrm R$ 只反映任务本身难度的差异。

- **H9（高阶增益）**：存在 $n\ge2$，使 $\Delta(n,\,n{=}1)>0$，且增益集中在 $j\ge2$ 的槽位。
- **H10（最优阶数）**：$\mathbb E[A]$ 关于 $n\in\{0,1,2,3\}$ 呈单峰；最优阶数 $n^\star$ 与命题 6 推论用经验 $\lVert\nabla^r h\rVert$ 估计出的阶数一致（描述性检验）。
- **H11（预测器形式）**：带 EMA 的差分滤波优于同阶无平滑外推（固定系数线性预测）；在线最小二乘线性预测与差分滤波的差异作双侧检验。
- **H12（自适应 ≥ 最优固定）**：在每个确认集上，自适应卡尔曼的 $\mathbb E[A]$ 不低于在**该数据集上**网格搜索得到的最优固定 $\beta$（非劣效界 $-\delta$，$\delta$ 待作者确认）；且在**另一数据集上调出的** $\beta$ 迁移过来时，自适应卡尔曼显著更优（跨数据集无需重调）。
- **H13（估计的 Q/R 预测最优阶数）**：由 $(\hat Q,\hat R)$ 和命题 6 推论预测的阶数 $\hat n$，与 E13 经验最优的 $n^\star$ 一致（逐数据集比较，描述性检验，配合 Spearman 相关）。

---

##### 8. Experimental Plan

###### 8.1 数据

- **主数据**：Spec-Bench [14] 480 条，切片为 first 26（选参）/ next 52 / last 52 / 全量（确认）。
- **跨域（H8）**：对每个训练式基线，按其训练数据定义 ID/OOD：
  - EAGLE 系列以 ShareGPT 类对话为训练分布（以所用权重的说明为准【待核对】），ID 取对话类子集；
  - OOD 取 Spec-Bench 中的数学、代码、检索增强、摘要等子类，以及长上下文集（如 LongSpecBench [45]，可得性待核对）；
  - 可选：自训一个只用数学数据的 EAGLE-3 起草器，复刻 [44] 的设置【需算力，待作者确认】。
- **模板**：聊天模板开/关作为一个因素。

###### 8.2 模型

| 模型 | tied | $d$ | 映射 |
|---|---|---:|---|
| Qwen3-0.6B | 是 | 1024 | hidden + 缩放 |
| Qwen3-1.7B | 是 | 2048 | hidden + 缩放 |
| Qwen3-4B | 是 | 2560 | hidden + 缩放 |
| Qwen3-8B | 否 | 4096 | $W_a$ 适配 |
| Qwen3-14B | 否 | 5120 | $W_a$ 适配 |

tied 与 $d$ 来自 HF config。EAGLE-3 的公开起草器是否覆盖这些目标模型尚【待核对】，H8 的模型选择以此为准。

###### 8.3 基线

- 自回归；
- ESP 均值；
- 输入空间外推（附录 B）；
- 固定偏移 $\pi_{\text{off}}$；
- 免训练：PLD、Lookahead [8]、STAND [21]；
- 训练式：EAGLE-3 [6]（公开权重），以及可选的 Medusa [3]/Hydra [7]。训练式基线只用于 H8 和参照。

###### 8.4 指标

- 主指标：$\sum A$ 与 $\mathbb E[\tau]$；
- 逐槽、逐节点深度的接受率；
- 跨域保留率 $\mathrm R$；
- 墙钟 $S$ 与 $c_\Phi/c_B$；
- exact\_match = 1 作为门槛；
- 机制指标：mask 输入的范数与激活统计（C7）。

###### 8.5 实验矩阵（$k\in\{1,\dots,4\}$，全部待测）

| 编号 | 类型 | 设置 | 假设 |
|---|---|---|---|
| E1 | 基线复现 | ESP 原设置（树形，附录） | — |
| E2 | 主对照 | Ours vs ESP vs 输入空间外推 | H1 |
| E3 | 映射 | 缩放开/关、原始 hidden、期望 embedding、$W_a$（$\lambda$ 扫描） | H6、C7 |
| E4 | 阶数 | ESP → 零阶-only → +一阶冻结 → +一阶递推，按槽 $j$ 分解 | H2、H3、H5 |
| E5 | 前缀条件 | 前缀条件 vs 共享 $(a^{(0)},v^{(0)})$，按节点深度 $i$ 分解 | H4 |
| E6 | 空间 | hidden vs logit（应相同）vs 分布 | 命题 3 |
| E7 | 偏移判别 | vs $\pi_{\text{off}}$ | H7 |
| E8 | 规模/tied | 0.6B–14B；tied 上 $W_a$ vs 恒等 | H6、命题 4 |
| E9 | 采样 | 残差投机采样，$T\in\{0.7,1\}$ | 命题 1b |
| E10 | 墙钟 | $S$、$c_\Phi/c_B$ | 命题 2 |
| E11 | 跨域 | ID/OOD × {Ours, ESP, PLD, EAGLE-3}，报告 $\mathbb E[\tau]$、$\mathrm R$、$S$ | H8 |
| E12 | 失败分析 | 冷启动、长上下文、OOD 激活、bf16 | 边界 |

**跨域表模板（E11）**

| 方法 | 训练 | ID $\mathbb E[\tau]$ | OOD $\mathbb E[\tau]$ | $\mathrm R$ | OOD $S$ |
|---|---|---:|---:|---:|---:|
| EAGLE-3 | 是 | 待测 | 待测 | 待测 | 待测 |
| PLD | 否 | 待测 | 待测 | 待测 | 待测 |
| ESP | 否 | 待测 | 待测 | 待测 | 待测 |
| Ours | 否 | 待测 | 待测 | 待测 | 待测 |


**高阶消融（全部待测）**

| 编号 | 类型 | 设置 | 假设 |
|---|---|---|---|
| E13 | 阶数 | $n\in\{0,1,2,3\}$，各阶 $\beta_r\in\{0,0.5,0.9,0.99\}$，在选参集上做网格或逐阶搜索；按槽位 $j$ 分解；同时记录经验 $\lVert\nabla^r h\rVert$ | H9、H10、命题 6、C12 |
| E14 | 预测器 | 差分滤波 vs 固定系数线性预测（$L=n$）vs 在线 LS（$W\in\{16,64\}$，$L\in\{1,2,3\}$） | H11 |
| E15 | 增益与阻尼 | 逐阶 EMA vs $K$ 形式；代理增益 $K'$ vs $K$；块内 $\rho\in\{1,0.9,0.7\}$ | C11、命题 7 |

**卡尔曼与自适应实验（全部待测）**

| 编号 | 类型 | 设置 | 假设 |
|---|---|---|---|
| E16 | 自适应 vs 固定 | 固定 $\beta$ 网格（各数据集各自最优 / 跨数据集迁移）vs 稳态卡尔曼（$Q$、$R$ 由 prompt 段矩估计）vs 自适应卡尔曼（协方差匹配 / Sage–Husa，$W\in\{16,64\}$，$b\in\{0.95,0.99\}$）；同时比较网格最优 $\beta$ 与 $\beta^\star(\hat\phi)$ | H12、命题 8/9 |
| E17 | 模型假设 | 维度共享 vs 逐维 vs 分组 vs 预白化；代理 $R'$ 固定倍数 vs 在线 $\hat R_\delta$；新息白度检验（Mehra 自相关） | §5.9、C12 |
| E18 | 阶数预测与不确定性 | 用 $(\hat Q,\hat R)$ 预测 $\hat n$，与 E13 对照；$\sigma^2_{i,j}$ 与逐槽接受率的相关；扩展：自适应 $k$、自适应温度 | H13、C13、C16 |

---

##### 9. Preliminary Results
v4 主方法没有实测。仓库已有数字为输入空间、树形、$k=1$，见附录 C，不作为本方法的证据。

##### 10. Limitations

1. 尚无实测数据。
2. 首轮 mask 只能用 ESP 均值（冷启动）。
3. $\tilde h_\ell$ 是代理 hidden：它来自上一轮另一前缀下的计算，当草稿与真实前缀不一致时会产生误差。
4. 链末节点 $k$ 没有代理，只能靠外推。
5. soft 输入的分布偏移是待测风险。
6. 本法的绝对接受率可能低于域内的训练式方法；定位依赖 H8 的结果。
7. 残差采样需要新实现。
8. 新颖性有待系统检索确认。

9. 命题 6、7 假设独立噪声且无平滑，与实际 hidden 轨迹不一定相符；带平滑的误差分析未完成。
10. 高阶差分对数值精度敏感（bf16）。
11. 卡尔曼变体依赖维度分解与线性高斯假设；自适应 $Q$、$R$ 估计在短序列上可能不稳定。

##### 11. Timeline & Venue（建议）

| 阶段 | 内容 | 时长 |
|---|---|---|
| P1 | 实现 §5.11 改动清单（需授权改代码）；核对 hidden 归一化；4B first 26 检查 | 2 周 |
| P2 | E2–E7 | 4 周 |
| P3 | E8–E11（含 EAGLE-3 跨域） | 4 周 |
| P4 | E12、附录树形、写作 | 3 周 |

目标会议：ICML / NeurIPS / ICLR，或 ACL / EMNLP（建议）。

##### 12. Open Questions for Author

1. 代理 hidden $\tilde h_\ell$ 的定义（位置对齐方式）是否认可？链末节点 $a^{(k)}$ 采用外推还是复用 $a^{(k-1)}$？
2. 前缀速度沿草稿折叠时，是否使用与已提交历史相同的 $\beta$？
3. H8 的 ID/OOD 划分，以及是否自训起草器以复刻 [44]？
4. 是否实现残差投机采样？
5. 模型范围、精度与预算。
6. 确认 “denoising” 的含义即“对输出 hidden 轨迹做滤波去噪”（§1.5），以及论文定位。

---

7. 高阶状态的主参数化选哪一种：逐阶独立 $\beta_r$（推荐）还是 $K$ 增益形式？代理观测是否用单独的增益 $K'$？
8. 阶数搜索预算：$n\le3$ 是否足够？在线 LS 线性预测进主文还是放附录？
9. 卡尔曼变体作为“主自适应变体”还是直接作为主方法？H12 的非劣效界 $\delta$ 取多少？
10. 维度分解采用共享、逐维还是预白化？


##### References

1. Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. <https://arxiv.org/abs/2211.17192>
2. C. Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv 2023. <https://arxiv.org/abs/2302.01318>
3. T. Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. <https://arxiv.org/abs/2401.10774>
4. Y. Li et al. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML 2024. <https://arxiv.org/abs/2401.15077>
5. Y. Li et al. *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees.* EMNLP 2024. <https://arxiv.org/abs/2406.16858>
6. Y. Li et al. *EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test.* NeurIPS 2025. <https://arxiv.org/abs/2503.01840>
7. Z. Ankner et al. *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding.* COLM 2024. <https://arxiv.org/abs/2402.05109>
8. Y. Fu et al. *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.* ICML 2024. <https://arxiv.org/abs/2402.02057>
9. F. Gloeckle et al. *Better & Faster Large Language Models via Multi-token Prediction.* ICML 2024. <https://arxiv.org/abs/2404.19737>
10. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv 2024. <https://arxiv.org/abs/2412.19437>
11. X. Miao et al. *SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification.* ASPLOS 2024. <https://arxiv.org/abs/2305.09781>
12. R. Goel et al. *Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing.* ICML 2026. <https://arxiv.org/abs/2603.17942>
13. D. J. Bajpai et al. *FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching.* ICLR 2026. <https://arxiv.org/abs/2602.01329>
14. H. Xia et al. *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding.* ACL 2024 Findings. <https://arxiv.org/abs/2401.07851>
15. Z. Chen et al. *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding.* NeurIPS 2024. <https://arxiv.org/abs/2402.12374>
16. M. Stern, N. Shazeer, J. Uszkoreit. *Blockwise Parallel Decoding for Deep Autoregressive Models.* NeurIPS 2018. <https://arxiv.org/abs/1811.03115>
17. A. Santilli et al. *Accelerating Transformer Inference for Translation via Parallel Decoding.* ACL 2023. <https://arxiv.org/abs/2305.10427>
18. G. Monea et al. *PaSS: Parallel Speculative Sampling.* ENLSP Workshop @ NeurIPS 2023. <https://arxiv.org/abs/2311.13581>
19. N. Yang et al. *Inference with Reference: Lossless Acceleration of Large Language Models.* arXiv 2023. <https://arxiv.org/abs/2304.04487>
20. Z. He et al. *REST: Retrieval-Based Speculative Decoding.* NAACL 2024. <https://arxiv.org/abs/2311.08252>
21. W. Song et al. *Accelerated Test-Time Scaling with Model-Free Speculative Sampling.* EMNLP 2025. <https://arxiv.org/abs/2506.04708>
22. J. Zhang et al. *Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding.* ACL 2024. <https://arxiv.org/abs/2309.08168>
23. F. Liu et al. *Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting.* NeurIPS 2024. <https://arxiv.org/abs/2404.18911>
24. J. K. Christopher et al. *Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion.* NAACL 2025. <https://arxiv.org/abs/2408.05636>
25. G. Li et al. *DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding.* arXiv 2025. <https://arxiv.org/abs/2510.02358>
26. S. Agrawal et al. *Structuring The Future: Diffusion LLM Speculative Decoding via Calibrated Draft Graphs.* arXiv 2025. <https://arxiv.org/abs/2509.18085>
27. M. Samragh et al. *Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential.* arXiv 2025. <https://arxiv.org/abs/2507.11851>
28. S. Kou et al. *CLLMs: Consistency Large Language Models.* ICML 2024. <https://arxiv.org/abs/2403.00835>
29. G. Oliaro et al. *SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications.* NeurIPS 2025. <https://arxiv.org/abs/2411.04975>
30. H. Xia et al. *SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration.* ICLR 2025. <https://arxiv.org/abs/2410.06916>
31. S. Hao et al. *Training Large Language Models to Reason in a Continuous Latent Space* (Coconut). COLM 2025. <https://arxiv.org/abs/2412.06769>
32. Z. Zhang et al. *Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space.* arXiv 2025. <https://arxiv.org/abs/2505.15778>
33. Y. Zhuang et al. *Text Generation Beyond Discrete Token Sampling* (Mixture of Inputs). arXiv 2025. <https://arxiv.org/abs/2505.14827>
34. J. Zou et al. *Latent Collaboration in Multi-Agent Systems* (LatentMAS). ICML 2026 Spotlight. <https://arxiv.org/abs/2511.20639>；代码 <https://github.com/Gen-Verse/LatentMAS>
35. P. Wu et al. *Parallel Decoding via Hidden Transfer for Lossless Large Language Model Acceleration.* arXiv 2024. <https://arxiv.org/abs/2404.12022>
36. K. Pal et al. *Future Lens: Anticipating Subsequent Tokens from a Single Hidden State.* CoNLL 2023. <https://arxiv.org/abs/2311.04897>
37. N. Belrose et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv 2023. <https://arxiv.org/abs/2303.08112>
38. B. Lester, R. Al-Rfou, N. Constant. *The Power of Scale for Parameter-Efficient Prompt Tuning.* EMNLP 2021. <https://arxiv.org/abs/2104.08691>
39. F. Lin et al. *BiTA: Bi-Directional Tuning for Lossless Acceleration in Large Language Models.* arXiv 2024. <https://arxiv.org/abs/2401.12522>
40. Z. Xiao et al. *ParallelSpec: Parallel Drafter for Efficient Speculative Decoding.* arXiv 2024. <https://arxiv.org/abs/2410.05589>
41. H. Yi et al. *Generation Meets Verification: Accelerating LLM Inference with Smart Parallel Auto-Correct Decoding* (SPACE). ACL 2024 Findings. <https://arxiv.org/abs/2402.11809>
42. J. Tack et al. *LLM Pretraining with Continuous Concepts* (CoCoMix). arXiv 2025. <https://arxiv.org/abs/2502.08524>
43. LCM team. *Large Concept Models: Language Modeling in a Sentence Representation Space.* arXiv 2024. <https://arxiv.org/abs/2412.08821>
44. F. Bie et al. *Osprey: Target-agnostic Pre-training Makes Stronger Drafters in Speculative Decoding.* EMNLP 2026（arXiv 评论：Accepted at EMNLP 2026）. <https://arxiv.org/abs/2609.09338>
45. J. Lee et al. *OWL: Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs.* arXiv 2025. <https://arxiv.org/abs/2510.07535>
46. M. Hui et al. *P-EAGLE: Parallel-Drafting EAGLE with Scalable Training.* arXiv 2026. <https://arxiv.org/abs/2602.01469>
47. X. Liu et al. *Online Speculative Decoding.* arXiv 2023（发表渠道待核对）. <https://arxiv.org/abs/2310.07177>
48. H. Liu et al. *Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs* (HedgeSpec). ICLR 2026（arXiv 评论）. <https://arxiv.org/abs/2510.20064>

[44]–[48] 的标题与第一作者于 2026-10-13 用 arXiv API 核对。[44][45][46] 中引用的数字取自原文 HTML 正文，不是本仓库实测。


49. R. E. Kalman. *A New Approach to Linear Filtering and Prediction Problems.* Journal of Basic Engineering, 1960. <https://doi.org/10.1115/1.3662552>
50. P. R. Kalata. *The tracking index: A generalized parameter for α-β and α-β-γ target trackers.* 22nd IEEE Conference on Decision and Control, 1983. <https://doi.org/10.1109/CDC.1983.269580>
51. J. Makhoul. *Linear prediction: A tutorial review.* Proceedings of the IEEE, 1975. <https://doi.org/10.1109/PROC.1975.9792>
52. C. C. Holt. *Forecasting seasonals and trends by exponentially weighted moving averages.* International Journal of Forecasting, 2004（1957 年报告的重刊）. <https://doi.org/10.1016/j.ijforecast.2003.09.015>
53. E. S. Gardner Jr. *Exponential smoothing: The state of the art.* Journal of Forecasting, 1985. <https://doi.org/10.1002/for.3980040103>
54. R. J. Hyndman, A. B. Koehler, R. D. Snyder, S. Grose. *A state space framework for automatic forecasting using exponential smoothing methods.* International Journal of Forecasting, 2002. <https://doi.org/10.1016/S0169-2070(01)00110-8>

[49]–[54] 的标题、渠道与年份于 2026-10-13 用 Crossref API 核对。牛顿差分外推的教材出处待补。“阻尼趋势”的原始出处（通常归于 Gardner & McKenzie）未核对，正文只引用了综述 [53]。

55. J. E. Gray, W. J. Murray. *A derivation of an analytic expression for the tracking index for the alpha-beta-gamma filter.* IEEE Transactions on Aerospace and Electronic Systems, 1993. <https://doi.org/10.1109/7.220956>
56. R. K. Mehra. *On the identification of variances and adaptive Kalman filtering.* IEEE Transactions on Automatic Control, 1970. <https://doi.org/10.1109/TAC.1970.1099422>
57. R. K. Mehra. *Approaches to adaptive filtering.* IEEE Transactions on Automatic Control, 1972. <https://doi.org/10.1109/TAC.1972.1100100>
58. A. P. Sage, G. W. Husa. *Algorithms for sequential adaptive estimation of prior statistics.* 1969 IEEE Symposium on Adaptive Processes (8th) Decision and Control, 1969. <https://doi.org/10.1109/SAP.1969.269927>
59. A. H. Mohamed, K. P. Schwarz. *Adaptive Kalman Filtering for INS/GPS.* Journal of Geodesy, 1999. <https://doi.org/10.1007/s001900050236>
60. P. R. Kalata. *The Tracking Index: A Generalized Parameter for α-β and α-β-γ Target Trackers.* IEEE Transactions on Aerospace and Electronic Systems, 1984. <https://doi.org/10.1109/TAES.1984.310438>（[50] 为同名 1983 CDC 会议版；跟踪指数公式以本期刊版为准，原文待核对）

[55]–[60] 的标题、渠道与年份于 2026-10-13 用 Crossref API 核对。说明：

- §5.9 中 Kalata 的 α-β 闭式公式只经二手来源核对；
- Sage–Husa 与协方差匹配的具体更新式按常见引述形式书写，原文细节待核对；
- Joseph 形式与 DARE 收敛条件的教材出处待补。
---

##### 附录 A：《研究计划规范》六方面对照

| 方面 | 位置 | 状态 |
|---|---|---|
| 文献调研 | §3、References（60 篇） | 已核对；新颖性待验证 |
| 数据 | §8.1 | 跨域划分待核对 |
| 模型 | §4.1、§8.2–8.3 | tied 已核对；全部待测 |
| 算法 | §4.2、§5.1–5.10、§6、§5.11 | 已形式化；代码未改 |
| 指标 | §4.3、§8.4、§7 | 待实施 |
| 实验设计 | §7 H1–H13、§8.5 E1–E18 | 全部待测 |

##### 附录 B：输入空间外推（对照）
状态在输入 embedding 空间维护：
$$\hat v^e\leftarrow\beta\hat v^e+(1-\beta)(e_{\text{new}}-e_{\text{prev}}),\qquad m_j=\hat e+\gamma c_j(\rho)\hat v^e.$$

- 长记忆下 $\hat v^e\approx d_1$，退化为固定偏移（C1）。
- 输入空间的前缀条件版本可以**精确**计算：草稿 token 的 $e(\tilde x_i)$ 在前向前已知，所以可以沿草稿折叠 $\hat e,\hat v^e$。这可以作为“前缀条件”效应在输入空间的对照。
- 仓库中 `extrapolate_ema: true` 部分实现了递推（$\beta'=\beta$）。

**树形扩展**：节点前缀为其祖先路径，命题 1 推广为“节点只注意祖先”。

##### 附录 C：仓库已有实测（输入空间，树形 $[14]$，$k=1$）
来源：`docs/history-extrapolation.md`。

条件：Qwen3，fp32，贪心，单 seed；所有格子 exact\_match = 1。**与 v3 输出空间方法无关。**

| 模型 | 切片 | 冻结均值 | 更新均值 | 输入空间一阶外推（选定） |
|---|---|---:|---:|---:|
| 0.6B | first 26 / next 52 / last 52 | 726 / 1550 / 1598 | 701 / 1473 / 1468 | 723 / 1553 / 1575 |
| 1.7B | first 26 / next 52 / last 52 | 778 / 1747 / 1723 | 757 / 1639 / 1626 | 780 / 1709 / 1719 |
| 4B | first 26 / next 52 / last 52 | 680 / 1462 / 1476 | 704 / 1479 / 1525 | 777 / 1686 / 1723 |
| 4B | 全量 480 | 12809 | — | 15009 |

$\beta$ 扫描、$\gamma$ 扫描、$[7,2]$ 树的数字见 v2 附录 C。


</details>

<a id="legacy-studies--template-docs-plan"></a>

### studies/_template/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/_template/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/_template/docs/PLAN.md)；原SHA256 `abc881fbcdb43e8ccbd622541a551e76e79c10a4af8a037428dba27e600060d1`。

#### Study Plan: my_study

复制后填写本轮事实，先确认计划再运行。遵循 [Study 指南](../studies/README.md)，结果填 [STUDY_REPORT](../studies/_template/docs/STUDY_REPORT.md)。本模板没有实验数字或执行授权。

##### 1. 研究问题与对照

写清要回答的问题、现有结论、主指标与对照。AR 对照、ESP、EMA 或 alignment 只在本轮问题需要时选择；未实现的外部 baseline 不写成已支持。保留论文解释差异的链接，不自动宣称完整复现。

##### 2. Study / Experiment / Run

- Study：填写新目录与名称。
- Experiment：填写 axes；没有 axes 时说明单组合。
- Run：每组合 × seeds；列出总数和新的 version。
- 声明：记录基底 → fixed → 轴/seed 的覆盖关系，make 后核对实际 config/index。

##### 3. 变量轴

| **因素** | **字段** | **取值** | **比较目的** |
| --- | --- | --- | --- |
| 待填写 | 待填写 | 待填写 | 待填写 |

seeds 单独写，不能用并发数代替。列出已有正式接受数的格子，默认不重跑。

##### 4. 固定条件与来源

| **项** | **必须记录的内容** |
| --- | --- |
| 源码 | 本仓库提交、RPipe ORIGIN、recipe、专项锁、额外参考输入 |
| 模型/tokenizer | 名称、Hub/实际 revision、权重来源和 digest；training-free 权重语义 |
| 数据/prompt | 文件与来源、切片、聊天模板、tokenization、输入 digest |
| 解码 | BC、mask 数/初始化、树模式/pruner、β/λ/γ、接受/回退和 KV/cache 合同 |
| 预算/停止 | temperature、seed、max_new_tokens、EOS、context-limit、prompt 数 |
| 精度与环境 | 实际 dtype、TF32/混合精度、attention backend、依赖、设备、随机性开关 |
| 判定 | exact_match_rate 必须为 1；主指标及各深度诊断、τ 与速度的独立口径 |

按目标原配方填写，不以示例 YAML 的 float32/T=0 替代研究条件。不为提速换成不能接收 4D 树 mask 的 backend。来源/配方变化先换 version，再显式 make/remake，不能重新签收旧成功产物。

##### 5. 高效率排班与失败处理

先以 `make --round 1` 串行测量权重、KV、树、mask 和进程开销；记录实际容量后才用 `system.config.vram_bytes` 小步扩大独立 Run 并发，不套用视觉训练估计。每个 Run 独占状态/产物/logger，wait 组末等所有进程退出，不重复启动 launch。

速度对照使用同任务量并串行进行，记录预热、同步、backend、实际生成长度与整体墙钟。失败/OOM 保留 ID、退出码和日志；后续独立重算换 version。解码 checkpoint/resume 未验收，不能把 `resume: latest` 当作已支持的 KV/mask/RNG 恢复。

```powershell
$env:PYTHONUTF8 = '1'
python -m spec_mtp make studies/<name> --num-gpus 1 --init-gpu 0 --round 1
python -m spec_mtp launch studies/<name> --num-gpus 1 --init-gpu 0
```

###### 5.1 时长预估

make 后、launch 前填 est；同一 wait 的预估取最慢一条，整轮求和。报告填实际值，缺失明确标记，不能拿估计冒充实测。

| **wait** | **factors** | **mode** | **seed** | **Run ID** | **est (s)** |
| --- | --- | --- | --- | --- | --- |
| 待填写 | 待填写 | eval | 待填写 | make 后填写 | 待测 |

##### 6. 阶段与成功标准

默认 prepare/execute 经 recipe 和专项 Algorithm 负责构造/原始观测；collect 计算指标，summarize 在成功前检查 exact-match/证据，write 核对来源后由库定稿。Study process 只终验当前 index，不扫描旧 Run。

- 计划因素 × seed 全部存在、Run succeeded，原始 token 与 AR 按原合同一致，exact_match_rate == 1。
- 来源、配置、输入、权重/runtime 与接受/cache 证据完整；子集或混合来源不能通过完整终验。
- 同设备旧/新数值前置门通过，保留所有失败反例；不挑选通过重复或放宽阈值。
- 人写报告按 Experiment 给 mean/std/min/max/n、单位、条件和结论，附真实解码结果图；NUMBERS 不是结论。

##### 7. 执行范围与验证边界

填写允许的模型/设备/格子/预算和停止条件，分别列出静态、CPU tiny、真实 GPU、安装、恢复及并行验证。短探针不替代完整矩阵；未执行项和本机原始日志不随 clone 提供的边界必须保留。

</details>

<a id="legacy-studies-esp-table1-bf16-full-docs-plan"></a>

### studies/esp_table1_bf16_full/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/esp_table1_bf16_full/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_full/docs/PLAN.md)；原SHA256 `0d8ac40e958037c9aef53bf212dc3157a23ee98277134eb28de016e2f3e10c0a`。

#### ESP Table 1 reproduction — Qwen3-8B / BF16

状态：2026-10-15 计划已立项，未声称复现成功。优先于所有其他研究。此前 4B 证据全部 FP32，已按工具/测试与证据两个主题提交，并推送工作分支；PR #7 的 Branch flow/cpu-checks 已通过并合 dev（8677376e1ac878a159a3095711eac43420d97bde），不合 main。0.6B 罕见事件/局部机理和 4B FP32 扩展暂停。

##### 1. 来源、目标与未公开细节

主来源：[ESP arXiv v2](https://arxiv.org/html/2603.17942v2)，§3.1/3.4/3.5、§4、Table 1/3、Appendix D/G.2。已读正文及本地 PDF 第 7 页表格。模型为 [Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)，权重优先 ModelScope；仅用于本 ESP 复现，禁止拿 8B 做其他研究。

Table 1 Qwen3-8B 的 ESP τ 目标：BC30=1.63，BC60=1.74。BC10 是额外协议核验，Table 1 没有其目标。本文表注为 256-token 输出/H100，§4 正文为 100-token/A100；本次遵循 Table 1 的 256 上限，EOS 则提前停止，逐条记录实际输出长度。不得把原仓库 100-token 静态配置冒充本复现。

论文未明确公开 precision、模型 immutable revision、Qwen thinking 开关、prompt 截断策略与 τ 跨任务聚合权重。BF16 是当前用户明确要求。采用非 thinking chat template、首轮原始 prompt、不截断且纳入全部 480 条；输出单列这些实现选择，不称其为已获作者确认的配置。RTX 5090D v2 24GB 与 H100 不同，torch/transformers 版本差异也记录。用户不要求测速，因此不测 S/R。

官方 SpecBench：[hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench)，固定提交 fd2c1cd7d2201ef71db4c5f4e455008f017967bf，data/spec_bench/question.jsonl。本地 480 条 JSON 对象及顺序与官方完全一致；远端 SHA256=4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04，本地 SHA256=7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21。只存在字节排版差异。保存原文件并绑定两哈希；不得为了对齐目标筛掉难样本。

##### 2. 当前协议（Qwen3 BC60 修订）

[Appendix G.2](https://arxiv.org/html/2603.17942v2#A7.SS2)，PDF第18页原句：

> For Qwen3 models, single mask token is used for BC=60 while for LLaMA3 model two mask tokens are used.

§4 的一般默认不能覆盖这条针对Qwen3的具体说明。[Table 3](https://arxiv.org/html/2603.17942v2#S4.T3)的单mask树宽[29]对应BC=2×(1+29)=60。Table 3的数值行是LLaMA，G.2语句在Figure8的附加结果语境中；Table1没有单独给出mask数。因此采用60/1为主配置是基于模型特定说明的协议判断，不冒称已经获得作者确认。60/2保留为预先声明的对照，不从结果中择优。

此前已执行的预检与数值控制以60/2为BC60格，属于对照结果，不能继续标为Qwen3 BC60主复现。原冻结PLAN/Run/证据保持原字节；修订预检采用新PLAN与新Run补60/1，并按冻结哈希复用原已执行三格，避免重跑已有数字。预检和全480报告均固定列60/1、60/2的τ及各自相对1.74的差值，标明AR质量，缺格写pending而不是0。

| 项目 | 本次值 |
| --- | --- |
| dtype / temperature | BF16 / T0，绝不新增 FP32 模型实验 |
| mean mask | prompt embedding mean 初始化；每个已生成 token 按 m←m+0.1(e−m) 更新 |
| BC / mask count | 主配置10/1、30/1、60/1；对照60/2 |
| tree | dynamic，Top-1 扩展，累计路径概率排序；父子同 token 替换为下一候选；pruning=true |
| implementation | native naive 动态树 attention/cache，保留原始模型与验证语义 |
| seed / outputs | seed123，max_new_tokens256，EOS 提前停止，warmup=false |
| input set | 全部 480 首轮，chat template enable_thinking=false；max_prompt_len40960，实际预检不丢任何条目 |
| audits | 每个 ESP 输出与该 BF16 模型独立 AR 逐 token 核验；报告所有不一致 |
| clocks | Study 局部禁用 decoding clock；零 timing 是原生字段占位，不是测量 |

BC=(k+1)(1+draft_nodes)，本次实际 draft budget=4/14/29，另保留60/2的19节点对照；检查每步真实 BC 不超过预算并记录树宽。论文 Algorithm 1 的 B 与正文 Eq7 表述存在歧义；遵循 Eq7 与当前 Top-1 实现并报告映射，不暗改预算来追目标。

主指标同时报告 pooled τ=sum(committed)/sum(decode_calls)、逐 prompt τ 的均值、逐 category pooled τ 后等权均值。committed 包含 bonus，不含 prefill；另列含 prefill 的实际 model calls、新 token 数与 accepted draft 数。尾部 EOS/长度裁剪口径明确，不混淆 accepted 与 τ。目标的近似程度报告原值/差值/按两位小数是否一致；不得预先把某个聚合方式定为作者确切口径。

##### 3. 执行与停止规则

1. 复用 native RPipe Study prepare→execute→collect→summarize→write→process；新 Study/version/Run，原失败和成功 Run 不重签。权重逐文件哈希、官方 ModelScope 文件名/尺寸与实际 BF16参数核验；离线缓存 immutable revision 未确认则明确记录。
2. 先新建小样本协议检查 Study（每类首条，BC10/1、BC30/1、BC60/1主配置与BC60/2对照）。检查 BF16装载、动态树、256上限、精度及原生门；这只是预检，不能代替 480 条 Table 1。
3. 按本次四格协议冻结全480 Study，预检后串行执行主配置与双mask对照。预检失配不隐藏，也不替代全量：全量仍保留native exact_match_rate=1门，失败格τ仅标为诊断值并列差值，不宣称成功。若实际数据/模型来源或协议校验失败、OOM则先修复来源/容量，再用新Run；不得改FP32、teacher强制纠正、丢反例或放宽门。
4. OOM/其他任务占显存时保留失败、等资源或用新 Run 可逆验证，不停其他任务。单卡只启动一条实验调度链，不并发加载模型。
5. 全量完成后报告配置覆盖、精度、每任务/每 prompt计数、三种 τ聚合、对目标差值与失败边界。数值不符也输出可复核负结果，再按首个可证伪差异推进，不宣称强行成功。

四任务共用 24GB GPU。记录启动前空闲显存及运行容量采样（含其他进程，非精确 allocator peak）；不使用受干扰的墙钟或 timing 占位做性能比较。四格预检约 13×4×2×256=26624（其中旧三格已执行，新补单格上限6656） 生成 token 上界；全量四格含独立 AR 审计约 480×4×2×256=983040 上界。串行，没有额外 teacher/warmup。容量不足先暂停该 GPU 启动，继续证据/实现核对。

##### 4. 后续 4B BF16 配对（ESP 复现之后）

同批 prompt、同树/预算/长度/随机种子，比较 ESP frozen mean、输入零阶 updating mean λ=0.1、输入一阶 mean+gamma、输出零阶 final-normalized hidden 范数映射、输出一阶 final hidden+EMA velocity，另跑独立 AR。逐 prompt 列 accepted/model calls、每个对照的赢平输及 token exact。不得用此前 FP32 数字代替。

这里的 ESP frozen mean 是用户指定的后续基线，区别于 Table 1 updating mean。tied 权重直接用输出 hidden 的尺度映射，不做逆映射。当前 output provider 只支持 unpruned static chain；后续若统一采用 dynamic/pruned，必须先最小扩展并验证 owner/proxy 对齐，不能让方法使用不同树再配对。4B 子实验开始前单独冻结 PLAN、beta/gamma、精度内部计算与实际 tied-storage 检查。本阶段仅排队，不先跑 4B。


##### 本Study全量合同

全部480条首轮，不跳类别/困难样本，BF16/T0/256/EOS/seed123，mean mask按Eq5 λ=.1更新。四格固定顺序：BC10/1辅助、BC30/1主（目标1.63）、BC60/1主（29节点、目标1.74）、BC60/2对照（19节点、同列相对1.74差值）。动态树+剪枝，不将单mask切成另一种实现。两种BC60必须全量执行并同时报告，拒绝按结果择优。

原设置预检已出现BF16数值失配；完整480仍按本次协议执行以落实全量证据，native exact_match_rate=1门不放宽，失配格仅列诊断τ及差值，整体不宣称复现成功。若来源/容量/协议本身不符则先处理再新Run，不能用teacher修正或切FP32。检查点/实际BF16数值策略与此前native预检相同，不混用reduction/shape干预结果。其他研究和4B五方法暂停。

排班：预检退出且空闲显存约19GiB以上才串行启动一条launch链；round1，容量采样包含其他任务、不做测速/墙钟比较。四格含fresh AR上限983040生成token，无warmup/teacher；每Run退出释放权重后再下一格，不自动retry。命令：python -m spec_mtp make studies/esp_table1_bf16_full --round 1；python tools/launch_once.py launch studies/esp_table1_bf16_full --round 1 --console shared。所有新的failed/succeeded Run保留，不重签旧Run。

正式报告保留每prompt计数、精度/AR质量、输出与AR的SHA256及首分叉摘要；完整tokens/native Run留本机ignored路径。公开报告不包含完整480条原生Run/权重/缓存。τ列pooled、prompt均值、category等权均值并各给预定目标差值；不从三种聚合中事后挑接近目标者。

</details>

<a id="legacy-studies-esp-table1-bf16-full-argmax-docs-plan"></a>

### studies/esp_table1_bf16_full_argmax/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/esp_table1_bf16_full_argmax/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_full_argmax/docs/PLAN.md)；原SHA256 `88f524f6f33eda74c43792826a8a80f91f550261d3ce3943b7fc595e2d005e4c`。

#### ESP Table 1 reproduction — Qwen3-8B / BF16

状态：2026-10-15 计划已立项，未声称复现成功。优先于所有其他研究。此前 4B 证据全部 FP32，已按工具/测试与证据两个主题提交，并推送工作分支；PR #7 的 Branch flow/cpu-checks 已通过并合 dev（8677376e1ac878a159a3095711eac43420d97bde），不合 main。0.6B 罕见事件/局部机理和 4B FP32 扩展暂停。

##### 1. 来源、目标与未公开细节

主来源：[ESP arXiv v2](https://arxiv.org/html/2603.17942v2)，§3.1/3.4/3.5、§4、Table 1/3、Appendix D/G.2。已读正文及本地 PDF 第 7 页表格。模型为 [Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)，权重优先 ModelScope；仅用于本 ESP 复现，禁止拿 8B 做其他研究。

Table 1 Qwen3-8B 的 ESP τ 目标：BC30=1.63，BC60=1.74。BC10 是额外协议核验，Table 1 没有其目标。本文表注为 256-token 输出/H100，§4 正文为 100-token/A100；本次遵循 Table 1 的 256 上限，EOS 则提前停止，逐条记录实际输出长度。不得把原仓库 100-token 静态配置冒充本复现。

论文未明确公开 precision、模型 immutable revision、Qwen thinking 开关、prompt 截断策略与 τ 跨任务聚合权重。BF16 是当前用户明确要求。采用非 thinking chat template、首轮原始 prompt、不截断且纳入全部 480 条；输出单列这些实现选择，不称其为已获作者确认的配置。RTX 5090D v2 24GB 与 H100 不同，torch/transformers 版本差异也记录。用户不要求测速，因此不测 S/R。

官方 SpecBench：[hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench)，固定提交 fd2c1cd7d2201ef71db4c5f4e455008f017967bf，data/spec_bench/question.jsonl。本地 480 条 JSON 对象及顺序与官方完全一致；远端 SHA256=4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04，本地 SHA256=7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21。只存在字节排版差异。保存原文件并绑定两哈希；不得为了对齐目标筛掉难样本。

##### 2. 当前协议（Qwen3 BC60 修订）

[Appendix G.2](https://arxiv.org/html/2603.17942v2#A7.SS2)，PDF第18页原句：

> For Qwen3 models, single mask token is used for BC=60 while for LLaMA3 model two mask tokens are used.

§4 的一般默认不能覆盖这条针对Qwen3的具体说明。[Table 3](https://arxiv.org/html/2603.17942v2#S4.T3)的单mask树宽[29]对应BC=2×(1+29)=60。Table 3的数值行是LLaMA，G.2语句在Figure8的附加结果语境中；Table1没有单独给出mask数。因此采用60/1为主配置是基于模型特定说明的协议判断，不冒称已经获得作者确认。60/2保留为预先声明的对照，不从结果中择优。

此前已执行的预检与数值控制以60/2为BC60格，属于对照结果，不能继续标为Qwen3 BC60主复现。原冻结PLAN/Run/证据保持原字节；修订预检采用新PLAN与新Run补60/1，并按冻结哈希复用原已执行三格，避免重跑已有数字。预检和全480报告均固定列60/1、60/2的τ及各自相对1.74的差值，标明AR质量，缺格写pending而不是0。

| 项目 | 本次值 |
| --- | --- |
| dtype / temperature | BF16 / T0，绝不新增 FP32 模型实验 |
| mean mask | prompt embedding mean 初始化；每个已生成 token 按 m←m+0.1(e−m) 更新 |
| BC / mask count | 主配置10/1、30/1、60/1；对照60/2 |
| tree | dynamic，Top-1 扩展，累计路径概率排序；父子同 token 替换为下一候选；pruning=true |
| implementation | native naive 动态树 attention/cache，保留原始模型与验证语义 |
| seed / outputs | seed123，max_new_tokens256，EOS 提前停止，warmup=false |
| input set | 全部 480 首轮，chat template enable_thinking=false；max_prompt_len40960，实际预检不丢任何条目 |
| audits | 每个 ESP 输出与该 BF16 模型独立 AR 逐 token 核验；报告所有不一致 |
| clocks | Study 局部禁用 decoding clock；零 timing 是原生字段占位，不是测量 |

BC=(k+1)(1+draft_nodes)，本次实际 draft budget=4/14/29，另保留60/2的19节点对照；检查每步真实 BC 不超过预算并记录树宽。论文 Algorithm 1 的 B 与正文 Eq7 表述存在歧义；遵循 Eq7 与当前 Top-1 实现并报告映射，不暗改预算来追目标。

主指标同时报告 pooled τ=sum(committed)/sum(decode_calls)、逐 prompt τ 的均值、逐 category pooled τ 后等权均值。committed 包含 bonus，不含 prefill；另列含 prefill 的实际 model calls、新 token 数与 accepted draft 数。尾部 EOS/长度裁剪口径明确，不混淆 accepted 与 τ。目标的近似程度报告原值/差值/按两位小数是否一致；不得预先把某个聚合方式定为作者确切口径。

##### 3. 执行与停止规则

1. 复用 native RPipe Study prepare→execute→collect→summarize→write→process；新 Study/version/Run，原失败和成功 Run 不重签。权重逐文件哈希、官方 ModelScope 文件名/尺寸与实际 BF16参数核验；离线缓存 immutable revision 未确认则明确记录。
2. 先新建小样本协议检查 Study（每类首条，BC10/1、BC30/1、BC60/1主配置与BC60/2对照）。检查 BF16装载、动态树、256上限、精度及原生门；这只是预检，不能代替 480 条 Table 1。
3. 按本次四格协议冻结全480 Study，预检后串行执行主配置与双mask对照。预检失配不隐藏，也不替代全量：本Study按最新用户要求将AR一致率独立报告，不作为τ执行门；argmax自检是必须全过的验收。原生AR门的失败状态另外保留，不篡改结果。若实际数据/模型来源或协议校验失败、OOM则先修复来源/容量，再用新Run；不得改FP32、teacher强制纠正、丢反例或放宽门。
4. OOM/其他任务占显存时保留失败、等资源或用新 Run 可逆验证，不停其他任务。单卡只启动一条实验调度链，不并发加载模型。
5. 全量完成后报告配置覆盖、精度、每任务/每 prompt计数、三种 τ聚合、对目标差值与失败边界。数值不符也输出可复核负结果，再按首个可证伪差异推进，不宣称强行成功。

四任务共用 24GB GPU。记录启动前空闲显存及运行容量采样（含其他进程，非精确 allocator peak）；不使用受干扰的墙钟或 timing 占位做性能比较。四格预检约 13×4×2×256=26624（其中旧三格已执行，新补单格上限6656） 生成 token 上界；全量四格含独立 AR 审计约 480×4×2×256=983040 上界。串行，没有额外 teacher/warmup。容量不足先暂停该 GPU 启动，继续证据/实现核对。

##### 4. 后续 4B BF16 配对（ESP 复现之后）

同批 prompt、同树/预算/长度/随机种子，比较 ESP frozen mean、输入零阶 updating mean λ=0.1、输入一阶 mean+gamma、输出零阶 final-normalized hidden 范数映射、输出一阶 final hidden+EMA velocity，另跑独立 AR。逐 prompt 列 accepted/model calls、每个对照的赢平输及 token exact。不得用此前 FP32 数字代替。

这里的 ESP frozen mean 是用户指定的后续基线，区别于 Table 1 updating mean。tied 权重直接用输出 hidden 的尺度映射，不做逆映射。当前 output provider 只支持 unpruned static chain；后续若统一采用 dynamic/pruned，必须先最小扩展并验证 owner/proxy 对齐，不能让方法使用不同树再配对。4B 子实验开始前单独冻结 PLAN、beta/gamma、精度内部计算与实际 tied-storage 检查。本阶段仅排队，不先跑 4B。


##### 本Study全量合同

全部480条首轮，不跳类别/困难样本，BF16/T0/256/EOS/seed123，mean mask按Eq5 λ=.1更新。四格固定顺序：BC10/1辅助、BC30/1主（目标1.63）、BC60/1主（29节点、目标1.74）、BC60/2对照（19节点、同列相对1.74差值）。动态树+剪枝，不将单mask切成另一种实现。两种BC60必须全量执行并同时报告，拒绝按结果择优。

原设置预检已出现BF16数值失配；完整480仍按本次协议执行以落实全量证据，AR一致率与逐条首分叉独立报告，失配不阻止τ矩阵执行；每个提交token等于其验证前向argmax的自检必须全过。若来源/容量/协议本身不符则先处理再新Run，不能用teacher修正或切FP32。检查点/实际BF16数值策略与此前native预检相同，不混用reduction/shape干预结果。其他研究和4B五方法暂停。

排班：预检退出且空闲显存约19GiB以上才串行启动一条launch链；round1，容量采样包含其他任务、不做测速/墙钟比较。四格含fresh AR上限983040生成token，无warmup/teacher；每Run退出释放权重后再下一格，不自动retry。命令：python -m spec_mtp make studies/esp_table1_bf16_full --round 1；python tools/launch_once.py launch studies/esp_table1_bf16_full --round 1 --console shared。所有新的failed/succeeded Run保留，不重签旧Run。

正式报告保留每prompt计数、精度/AR质量、输出与AR的SHA256及首分叉摘要；完整tokens/native Run留本机ignored路径。公开报告不包含完整480条原生Run/权重/缓存。τ列pooled、prompt均值、category等权均值并各给预定目标差值；不从三种聚合中事后挑接近目标者。

##### 2026-10-15 τ主线与argmax审计修订

停止compact attention、单行RMSNorm及逐层算子归因；既有reduction/shape报告和FP32标注已提交推送，后续不再扩展。此Study v2保持v1所有模型、输入、默认归约、树、mask和采样设置，只增加观察：在SpecDecoder.generate范围内绑定选择行到最近一次forward的logits存储，记录forward序号/行号/argmax/选择token，验证prefill首token及每个实际提交token。EOS/预算截断后的未提交尾部不冒充覆盖；同时核验其选择值，但τ仍按原生裁剪后committed计算。不增加模型forward，不改权重、算子、精度、候选或采样；AR运行不安装观察器。每prompt自检落JSONL并flush，完整自检嵌入原始stats。

v1启动后新增自检要求，已处理的日志缺少argmax，不能事后证明覆盖；保留旧Run/冻结合同及日志，以新version/Run重新执行完整四格。只停止本任务旧launch/run进程，不终止其他项目。

原生AR严格门保持其历史语义，因此失配Run可能在summarize失败；launch_once关闭自动retry但继续下一格，τ汇总读取已完成execute/collect观测，不把这种原生失败伪称成功。报告分别给出原生生命周期、AR一致率、逐条首分叉和argmax全覆盖，通过argmax自检也不宣称与独立AR逐token一致。BC60单mask/29为主，双mask/19仅附加对照。四格480条全覆盖与argmax自检全过后才进入4B BF16配对。

</details>

<a id="legacy-studies-esp-table1-bf16-full-argmax-docs-third-party-plan"></a>

### studies/esp_table1_bf16_full_argmax/docs/THIRD_PARTY_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/esp_table1_bf16_full_argmax/docs/THIRD_PARTY_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_full_argmax/docs/THIRD_PARTY_PLAN.md)；原SHA256 `d6ceece7973d504235a29e8f89c177d543f1b0e9bcfba38222f72f024ee29821`。

#### 非官方第三方ESP原字节验收

2026-10-15，按Enmao授权尝试HF dataset `Harshvardhan-Mestha/esp-reproduction-artifacts`，commit `ad76f8da29f3bfc7faa20488193c6aa83a9f708c`。它不是作者官方代码，不能替代此前作者来源核查结论。全部4个code/入口文件原字节放src，另保留README及原结果JSON，ORIGIN逐文件SHA及省略的非执行海报/logbook列清单。完整Git clone留本机，不改入口或修正算法。

##### 一、兼容边界

原run.sh会安装依赖；原runner硬编码FP16、10个示例prompt、BC60/2及测速。本任务不执行这些动作。外部验收工具仅调用原`ESPDecoder.generate_esp`，加载已有ModelScope Qwen3-8B BF16/eager权重，采用原Table1 SpecBench480冻结manifest和数据，T0贪心、max256、BC10/30/60均单mask、seed123。实际输入token必须与原manifest逐token一致，不改prompt或模板。

原类仍保留其原始flat候选、draft softmax temperature=1、无KV复用等算法；未改成论文动态树剪枝。它是第三方原算法在共同外部设定下的兼容性/正确性验收，不冒称完全同协议Table1复现。计时仅在第三方模块作用域替换为常量，不测AR/TPS、不改全局时钟；原source字节不变。权重/model snapshot、数据、prompt和source/工具/协议全部绑定SHA。

##### 二、数学门与停止

先只对固定第一条q121，在三个BC串行验收，原max256不变，每格最多24次模型前向。通过前置数学门才能讨论完整480；如原算法失败，不重写或换候选/数据修结果。

验收只读同一次verification forward的logits，核对原算法即将追加的真实前缀：第一个token对应prompt末行，后续token对应其前一已追加draft的位置。不能把draft自身行或包含未提交候选的末mask行当作正确前缀的预测。每个计划提交token必须等于对应前缀父行argmax，出现第一个不等即拒绝并在实际追加前停止，保留位置、margin、前向计数和私有token证据。独立AR不新增或作门，永不追BF16算子机理。

原算法按最后self-match索引追加之前全部draft，因此会检查真实计划追加的prefix，不只检查其报告n_acc个位置。source报告的model calls可能不含proposal forward，另外逐前向计数；失败格valid_tau为空，不能用上游declared_tau冒充tokens/实际调用，也不能给出论文差值。

如果发生兼容错误或预算不足，同样保留失败，不自动重跑或改源码。其余预算仍可在同一模型上独立做相同固定q121前置验收，最多72次前向。完整480未执行就明确标注，第三方仓库既有结果单列为外部结果，不作为本机测量。

##### 三、资源与交付

8B只用于此Table1尝试。单进程顺序跑三个预算，加载前实际nvidia-smi无其他GPU驻留Python且free≥20GiB，各预算再核对GPU驻留作业；CPU进程不查、不登记SHA，不起背景等待或重叠队列，不抢占/杀外部作业。权重优先复用ModelScope缓存，不改CUDA torch或另建venv。

报告含基线/论文差距与不可比设置，全部数字给来源路径，明确非官方、未跑480或失败边界。源代码原样保留，标量结果/报告按主题commit/push/PR合dev，合后删除相应远程工作分支。本地分支、cache、权重、完整token和失败证据不删，不合main、不强推、不花钱、不弹窗。

</details>

<a id="legacy-studies-esp-table1-bf16-numerics-docs-plan"></a>

### studies/esp_table1_bf16_numerics/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/esp_table1_bf16_numerics/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_numerics/docs/PLAN.md)；原SHA256 `c1d91a12ec50c6ee1d9fd26151e8754e23067478cd5e4ef28150d43944878538`。

#### ESP Table 1 BF16 numerical gate isolation

仅服务当前8B ESP复现，不启动其他方法/4B。前置BC10的13条只有6条exact，q81最早分叉在新token index9（40233 vs20397）。先在q81、BC10、动态树/剪枝、lambda=.1、T0、32-token上限下，固定checkpoint/prompt与原生算法；比较2×2控制：allow_bf16_reduced_precision_reduction True/False × deterministic False/True。所有权重/模型activation保持BF16；正常HF RMSNorm/softmax的内部高精度归约是原生实现，不切FP32模型。全部禁用解码计时/预热。

CUBLAS_WORKSPACE_CONFIG=:4096:8 对四格一致，recipe设置在模型首次CUDA计算之前。首格重复原条件用于核对该环境配置和长度变化是否改变早期分叉；不是重签旧Run。其余格单因子与交互比较，不能把组合控制结果叫单kernel归因。Run内记录实际flags且成功/失败均恢复进程先前设置。若最早分叉未消失，不能把deterministic或reduction解释为充分修复；若消失仍只是一条短样本，下一步必须扩大前置13条/三BC。

原生exact_match_rate==1门保持；失败输出/Run不删，最大上限4×1×2×32=256生成token，另计checkpoint哈希成本。共享24GB卡只在preflight三个Run退出/显存释放后串行启动，一条launch链。无FP32、无速度、无teacher强制接受或筛样本。冻结前置raw SHA256/同tokenmanifest/本地16文件权重哈希；该对照不能替代Table1全480。


v1仅make，未启动GPU；CPU注入失败测试发现patch.object不支持CUDA后台属性删除恢复。v2改为显式try/finally赋值恢复并使用新version/Run，旧pending目录保留，不作为已执行结果。

</details>

<a id="legacy-studies-esp-table1-bf16-preflight-docs-plan"></a>

### studies/esp_table1_bf16_preflight/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/esp_table1_bf16_preflight/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_preflight/docs/PLAN.md)；原SHA256 `d19405874a3b628648f52c5bb817fbe0a66d2173fc3669595de90ca5b1931ba8`。

#### ESP Table 1 reproduction — Qwen3-8B / BF16

状态：2026-10-15 计划已立项，未声称复现成功。优先于所有其他研究。此前 4B 证据全部 FP32，已按工具/测试与证据两个主题提交，并推送工作分支；PR #7 等待 CPU CI，通过后仅合 dev。0.6B 罕见事件/局部机理和 4B FP32 扩展暂停。

##### 1. 来源、目标与未公开细节

主来源：[ESP arXiv v2](https://arxiv.org/html/2603.17942v2)，§3.1/3.4/3.5、§4、Table 1、Appendix D。已读正文及本地 PDF 第 7 页表格。模型为 [Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)，权重优先 ModelScope；仅用于本 ESP 复现，禁止拿 8B 做其他研究。

Table 1 Qwen3-8B 的 ESP τ 目标：BC30=1.63，BC60=1.74。BC10 是额外协议核验，Table 1 没有其目标。本文表注为 256-token 输出/H100，§4 正文为 100-token/A100；本次遵循 Table 1 的 256 上限，EOS 则提前停止，逐条记录实际输出长度。不得把原仓库 100-token 静态配置冒充本复现。

论文未明确公开 precision、模型 immutable revision、Qwen thinking 开关、prompt 截断策略与 τ 跨任务聚合权重。BF16 是当前用户明确要求。采用非 thinking chat template、首轮原始 prompt、不截断且纳入全部 480 条；输出单列这些实现选择，不称其为已获作者确认的配置。RTX 5090D v2 24GB 与 H100 不同，torch/transformers 版本差异也记录。用户不要求测速，因此不测 S/R。

官方 SpecBench：[hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench)，固定提交 fd2c1cd7d2201ef71db4c5f4e455008f017967bf，data/spec_bench/question.jsonl。本地 480 条 JSON 对象及顺序与官方完全一致；远端 SHA256=4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04，本地 SHA256=7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21。只存在字节排版差异。保存原文件并绑定两哈希；不得为了对齐目标筛掉难样本。

##### 2. 冻结协议

| 项目 | 本次值 |
| --- | --- |
| dtype / temperature | BF16 / T0，绝不新增 FP32 模型实验 |
| mean mask | prompt embedding mean 初始化；每个已生成 token 按 m←m+0.1(e−m) 更新 |
| BC / mask count | 10/1、30/1、60/2 |
| tree | dynamic，Top-1 扩展，累计路径概率排序；父子同 token 替换为下一候选；pruning=true |
| implementation | native naive 动态树 attention/cache，保留原始模型与验证语义 |
| seed / outputs | seed123，max_new_tokens256，EOS 提前停止，warmup=false |
| input set | 全部 480 首轮，chat template enable_thinking=false；max_prompt_len40960，实际预检不丢任何条目 |
| audits | 每个 ESP 输出与该 BF16 模型独立 AR 逐 token 核验；报告所有不一致 |
| clocks | Study 局部禁用 decoding clock；零 timing 是原生字段占位，不是测量 |

BC=(k+1)(1+draft_nodes)，本次实际 draft budget=4/14/19；检查每步真实 BC 不超过预算并记录树宽。论文 Algorithm 1 的 B 与正文 Eq7 表述存在歧义；遵循 Eq7 与当前 Top-1 实现并报告映射，不暗改预算来追目标。

主指标同时报告 pooled τ=sum(committed)/sum(decode_calls)、逐 prompt τ 的均值、逐 category pooled τ 后等权均值。committed 包含 bonus，不含 prefill；另列含 prefill 的实际 model calls、新 token 数与 accepted draft 数。尾部 EOS/长度裁剪口径明确，不混淆 accepted 与 τ。目标的近似程度报告原值/差值/按两位小数是否一致；不得预先把某个聚合方式定为作者确切口径。

##### 3. 执行与停止规则

1. 复用 native RPipe Study prepare→execute→collect→summarize→write→process；新 Study/version/Run，原失败和成功 Run 不重签。权重逐文件哈希、官方 ModelScope 文件名/尺寸与实际 BF16参数核验；离线缓存 immutable revision 未确认则明确记录。
2. 先新建小样本协议检查 Study（每类首条，BC10/30/60）。检查 BF16装载、动态树、256上限、精度及原生门；这只是预检，不能代替 480 条 Table 1。
3. 若全部 token exact，另建全 480 三格 Study 串行执行。若发现 mismatch，保留原始 token 与首分叉，先做同精度最小复现；不得改为 FP32、以 teacher 强制纠正、丢反例或放宽原生 exact_match_rate=1 门。
4. OOM/其他任务占显存时保留失败、等资源或用新 Run 可逆验证，不停其他任务。单卡只启动一条实验调度链，不并发加载模型。
5. 全量完成后报告配置覆盖、精度、每任务/每 prompt计数、三种 τ聚合、对目标差值与失败边界。数值不符也输出可复核负结果，再按首个可证伪差异推进，不宣称强行成功。

四任务共用 24GB GPU。记录启动前空闲显存及运行容量采样（含其他进程，非精确 allocator peak）；不使用受干扰的墙钟或 timing 占位做性能比较。预检约 13×3×2×256=19968 生成 token 上界；全量三格含独立 AR 审计约 480×3×2×256=737280 上界。串行，没有额外 teacher/warmup。容量不足先暂停该 GPU 启动，继续证据/实现核对。

##### 4. 后续 4B BF16 配对（ESP 复现之后）

同批 prompt、同树/预算/长度/随机种子，比较 ESP frozen mean、输入零阶 updating mean λ=0.1、输入一阶 mean+gamma、输出零阶 final-normalized hidden 范数映射、输出一阶 final hidden+EMA velocity，另跑独立 AR。逐 prompt 列 accepted/model calls、每个对照的赢平输及 token exact。不得用此前 FP32 数字代替。

这里的 ESP frozen mean 是用户指定的后续基线，区别于 Table 1 updating mean。tied 权重直接用输出 hidden 的尺度映射，不做逆映射。当前 output provider 只支持 unpruned static chain；后续若统一采用 dynamic/pruned，必须先最小扩展并验证 owner/proxy 对齐，不能让方法使用不同树再配对。4B 子实验开始前单独冻结 PLAN、beta/gamma、精度内部计算与实际 tied-storage 检查。本阶段仅排队，不先跑 4B。


本 Study 只做每类首条共13条三预算预检；不代表Table1全量。完整计划见 docs/research。同卡串行，起始空闲21408MiB（含其他任务2628MiB）；非性能测量。

</details>

<a id="legacy-studies-esp-table1-bf16-qwen-protocol-docs-plan"></a>

### studies/esp_table1_bf16_qwen_protocol/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/esp_table1_bf16_qwen_protocol/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_qwen_protocol/docs/PLAN.md)；原SHA256 `782d0022f811ca80ca716de8b8cf89007df16aa758bfaec0dcc5fb3f59442418`。

#### ESP Table 1 reproduction — Qwen3-8B / BF16

状态：2026-10-15 计划已立项，未声称复现成功。优先于所有其他研究。此前 4B 证据全部 FP32，已按工具/测试与证据两个主题提交，并推送工作分支；PR #7 的 Branch flow/cpu-checks 已通过并合 dev（8677376e1ac878a159a3095711eac43420d97bde），不合 main。0.6B 罕见事件/局部机理和 4B FP32 扩展暂停。

##### 1. 来源、目标与未公开细节

主来源：[ESP arXiv v2](https://arxiv.org/html/2603.17942v2)，§3.1/3.4/3.5、§4、Table 1/3、Appendix D/G.2。已读正文及本地 PDF 第 7 页表格。模型为 [Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)，权重优先 ModelScope；仅用于本 ESP 复现，禁止拿 8B 做其他研究。

Table 1 Qwen3-8B 的 ESP τ 目标：BC30=1.63，BC60=1.74。BC10 是额外协议核验，Table 1 没有其目标。本文表注为 256-token 输出/H100，§4 正文为 100-token/A100；本次遵循 Table 1 的 256 上限，EOS 则提前停止，逐条记录实际输出长度。不得把原仓库 100-token 静态配置冒充本复现。

论文未明确公开 precision、模型 immutable revision、Qwen thinking 开关、prompt 截断策略与 τ 跨任务聚合权重。BF16 是当前用户明确要求。采用非 thinking chat template、首轮原始 prompt、不截断且纳入全部 480 条；输出单列这些实现选择，不称其为已获作者确认的配置。RTX 5090D v2 24GB 与 H100 不同，torch/transformers 版本差异也记录。用户不要求测速，因此不测 S/R。

官方 SpecBench：[hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench)，固定提交 fd2c1cd7d2201ef71db4c5f4e455008f017967bf，data/spec_bench/question.jsonl。本地 480 条 JSON 对象及顺序与官方完全一致；远端 SHA256=4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04，本地 SHA256=7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21。只存在字节排版差异。保存原文件并绑定两哈希；不得为了对齐目标筛掉难样本。

##### 2. 当前协议（Qwen3 BC60 修订）

[Appendix G.2](https://arxiv.org/html/2603.17942v2#A7.SS2)，PDF第18页原句：

> For Qwen3 models, single mask token is used for BC=60 while for LLaMA3 model two mask tokens are used.

§4 的一般默认不能覆盖这条针对Qwen3的具体说明。[Table 3](https://arxiv.org/html/2603.17942v2#S4.T3)的单mask树宽[29]对应BC=2×(1+29)=60。Table 3的数值行是LLaMA，G.2语句在Figure8的附加结果语境中；Table1没有单独给出mask数。因此采用60/1为主配置是基于模型特定说明的协议判断，不冒称已经获得作者确认。60/2保留为预先声明的对照，不从结果中择优。

此前已执行的预检与数值控制以60/2为BC60格，属于对照结果，不能继续标为Qwen3 BC60主复现。原冻结PLAN/Run/证据保持原字节；修订预检采用新PLAN与新Run补60/1，并按冻结哈希复用原已执行三格，避免重跑已有数字。预检和全480报告均固定列60/1、60/2的τ及各自相对1.74的差值，标明AR质量，缺格写pending而不是0。

| 项目 | 本次值 |
| --- | --- |
| dtype / temperature | BF16 / T0，绝不新增 FP32 模型实验 |
| mean mask | prompt embedding mean 初始化；每个已生成 token 按 m←m+0.1(e−m) 更新 |
| BC / mask count | 主配置10/1、30/1、60/1；对照60/2 |
| tree | dynamic，Top-1 扩展，累计路径概率排序；父子同 token 替换为下一候选；pruning=true |
| implementation | native naive 动态树 attention/cache，保留原始模型与验证语义 |
| seed / outputs | seed123，max_new_tokens256，EOS 提前停止，warmup=false |
| input set | 全部 480 首轮，chat template enable_thinking=false；max_prompt_len40960，实际预检不丢任何条目 |
| audits | 每个 ESP 输出与该 BF16 模型独立 AR 逐 token 核验；报告所有不一致 |
| clocks | Study 局部禁用 decoding clock；零 timing 是原生字段占位，不是测量 |

BC=(k+1)(1+draft_nodes)，本次实际 draft budget=4/14/29，另保留60/2的19节点对照；检查每步真实 BC 不超过预算并记录树宽。论文 Algorithm 1 的 B 与正文 Eq7 表述存在歧义；遵循 Eq7 与当前 Top-1 实现并报告映射，不暗改预算来追目标。

主指标同时报告 pooled τ=sum(committed)/sum(decode_calls)、逐 prompt τ 的均值、逐 category pooled τ 后等权均值。committed 包含 bonus，不含 prefill；另列含 prefill 的实际 model calls、新 token 数与 accepted draft 数。尾部 EOS/长度裁剪口径明确，不混淆 accepted 与 τ。目标的近似程度报告原值/差值/按两位小数是否一致；不得预先把某个聚合方式定为作者确切口径。

##### 3. 执行与停止规则

1. 复用 native RPipe Study prepare→execute→collect→summarize→write→process；新 Study/version/Run，原失败和成功 Run 不重签。权重逐文件哈希、官方 ModelScope 文件名/尺寸与实际 BF16参数核验；离线缓存 immutable revision 未确认则明确记录。
2. 先新建小样本协议检查 Study（每类首条，BC10/1、BC30/1、BC60/1主配置与BC60/2对照）。检查 BF16装载、动态树、256上限、精度及原生门；这只是预检，不能代替 480 条 Table 1。
3. 按本次四格协议冻结全480 Study，预检后串行执行主配置与双mask对照。预检失配不隐藏，也不替代全量：全量仍保留native exact_match_rate=1门，失败格τ仅标为诊断值并列差值，不宣称成功。若实际数据/模型来源或协议校验失败、OOM则先修复来源/容量，再用新Run；不得改FP32、teacher强制纠正、丢反例或放宽门。
4. OOM/其他任务占显存时保留失败、等资源或用新 Run 可逆验证，不停其他任务。单卡只启动一条实验调度链，不并发加载模型。
5. 全量完成后报告配置覆盖、精度、每任务/每 prompt计数、三种 τ聚合、对目标差值与失败边界。数值不符也输出可复核负结果，再按首个可证伪差异推进，不宣称强行成功。

四任务共用 24GB GPU。记录启动前空闲显存及运行容量采样（含其他进程，非精确 allocator peak）；不使用受干扰的墙钟或 timing 占位做性能比较。四格预检约 13×4×2×256=26624（其中旧三格已执行，新补单格上限6656） 生成 token 上界；全量四格含独立 AR 审计约 480×4×2×256=983040 上界。串行，没有额外 teacher/warmup。容量不足先暂停该 GPU 启动，继续证据/实现核对。

##### 4. 后续 4B BF16 配对（ESP 复现之后）

同批 prompt、同树/预算/长度/随机种子，比较 ESP frozen mean、输入零阶 updating mean λ=0.1、输入一阶 mean+gamma、输出零阶 final-normalized hidden 范数映射、输出一阶 final hidden+EMA velocity，另跑独立 AR。逐 prompt 列 accepted/model calls、每个对照的赢平输及 token exact。不得用此前 FP32 数字代替。

这里的 ESP frozen mean 是用户指定的后续基线，区别于 Table 1 updating mean。tied 权重直接用输出 hidden 的尺度映射，不做逆映射。当前 output provider 只支持 unpruned static chain；后续若统一采用 dynamic/pruned，必须先最小扩展并验证 owner/proxy 对齐，不能让方法使用不同树再配对。4B 子实验开始前单独冻结 PLAN、beta/gamma、精度内部计算与实际 tied-storage 检查。本阶段仅排队，不先跑 4B。


##### 本Study合同

这是修订预检PLAN。原esp_table1_bf16_preflight/docs/PLAN.md保持历史冻结字节；G.2原句与60/1主、60/2对照以上已明确。固定原13条首条/category、BF16/T0、256/EOS、seed123、lambda=.1 updating mean、动态树剪枝。新Run只补原未运行的BC60/1，draft budget29；原10/1、30/1、60/2三格按reused_preflight.json的冻结来源/哈希复用。不得再跑已有数字，不能把复用三格冒充新Run成功。预检总表固定四行，BC60双方分别报告pooled/prompt/category τ与1.74差值、AR质量，不择优、不因失配删除。

原生BF16精度策略不调整；实际归约/确定性记录在Run。原设置与reduction/shape控制的结果不混合为一张同设置表。独立AR审计与native exact_match_rate=1门保留；失败τ为诊断值。下一全量四格由独立Study冻结全部480且同时列两种BC60，预检失配作为已知风险，不替代全量结果。若来源不符/OOM才先处理该失败边界。

GPU排班：只启动本Study一条launch链，单Run串行，当前约21GiB空闲；不停止其他任务。容量记录包含背景，不做速度/墙钟比较。新格13×2×256=6656生成token上界，无warmup/teacher。命令：python -m spec_mtp make studies/esp_table1_bf16_qwen_protocol --round 1；python tools/launch_once.py launch studies/esp_table1_bf16_qwen_protocol --round 1 --console shared，使用native scheduler的retry_failed=False，不修改vendored RPipe。

</details>

<a id="legacy-studies-esp-table1-bf16-reduction-check-docs-plan"></a>

### studies/esp_table1_bf16_reduction_check/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/esp_table1_bf16_reduction_check/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_reduction_check/docs/PLAN.md)；原SHA256 `28d3aa91fd0c0ddf901332814825e4b12df5b50bf48e45e2fcb60d277abc2fa2`。

#### ESP Table 1 BF16 reduction-control expansion

当前8B仅用于ESP复现，其他研究暂停。原13条/三BC分别只有6/6/5条与AR完全相同；q81最早分叉index9。q81新2×2控制的reduction=False/deterministic=False格32token已exact，AR输出与原设置相同。此Study预先固定原13条、BC10/30/60、256上限/EOS、T0/eager、dynamic/pruning、mean_prompt λ=.1 updating，与原始preflight同权重/tokenmanifest/source。唯一科学运算设置控制为allow_bf16_reduced_precision_reduction=False；deterministic=False，与原设置一致。CUBLAS_WORKSPACE_CONFIG=:4096:8固定，q81真值原设置控制已确认32token轨迹不受该环境变量影响。

全模型BF16，ESP状态也维持原生BF16，不转FP32；HF标准RMSNorm/softmax内部float32归约保持原实现。actualflags写入Run，原生fresh AR逐token门保持==1；三格先串行，禁止自动retry。三格全部通过才启动另一个新version全480三BC，任何不一致先保留反例并检查，不筛样本/追τ/使用teacher强制修正。即使这13条通过仍不是Table1成功，不把backend选择称为作者已确认配置。

13×3×2×256=19968生成token上界，没有warmup/teacher/计时；GPU只在numerics四格全部退出后使用。共享24GB，峰值采样包括背景，不比较墙钟。冻结前置raw哈希/权重文件/数据源/选定token；旧失败Run和未执行v1目录保留，新Run不重签。若失败，不切FP32或进入4B五方法。

</details>

<a id="legacy-studies-esp-table1-bf16-shape-check-docs-plan"></a>

### studies/esp_table1_bf16_shape_check/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/esp_table1_bf16_shape_check/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/esp_table1_bf16_shape_check/docs/PLAN.md)；原SHA256 `0e16b11ffb543a5f634b6de244dd26b7700865c84b2636ad364aa2856f7e423f`。

#### ESP BF16 shape control — q81 / BC10 / 96 tokens

只服务8B ESP Table1复现的token exact门；不启动新mask方法/4B/Kalman。reduction=False/deterministic=False虽使q81前32token一致，原13条256长度BC10仅4/13 exact，q81在index74再次分叉。最小假设：相同线性投影随批量行数选择不同BF16 GEMM归约路径，导致树batch与AR单行边界差异。

两格固定同checkpoint/q81/input/seed123，BF16、T0/eager、BC10单mask、dynamic/pruning、λ=.1 updating mean、96上限/EOS。归约禁用/确定性false两格一致。唯一干预：第二格把模型内nn.Linear按单行调用其原始forward再拼回原形，同时用于ESP与AR，不改变权重/激活dtype、RMSNorm/attention/RoPE、树/cache/mask/接受语义。第一格原native路径用于核对256→96前缀一致性。父级LLM forward调用数保持原统计；内部GEMM调用变多，因此不能用此对照做速度或算力结论。

单行linear是显式数值约定控制，尚未宣称作者实现或通用修复；若AR输出也改变，完整列出，不能对旧AR强制对齐。只比较各自新鲜AR并保留所有失配。若此干预也未通过，线性批量行数不是充分解释，继续定位attention/norm的形状边界；不调参追τ。即使通过，下一步必须13条/三BC，再全480。

在当前reduction_check三个Run退出后串行执行，不自动retry。2×1×2×96=384生成token上界；无warmup/teacher/timing。原始失败与来源保留，新version/Run。所有patch用ExitStack在成功/异常恢复；CPU测试检查BF16的批量拆分/行置换一致性、dtype/权重不变与异常恢复。

</details>

<a id="legacy-studies-input-mask-baseline-docs-plan"></a>

### studies/input_mask_baseline/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/input_mask_baseline/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/input_mask_baseline/docs/PLAN.md)；原SHA256 `1e56b2024b6dedcbc8867471bc55db2308a303c356fee879c5fda65d6c6700dc`。

#### Study Plan: input_mask_baseline

##### 一、问题与证伪条件

宏观依据：[研究提取](research/rpipe-study-extraction.md) 与 v6 §5.11 / E1–E2。真实预训练模型上，现有 frozen_mean / gamma_last 与 AR 能否逐 token 一致，并留下可靠的接受/cache、来源和阶段证据？任何 token mismatch、权重/runtime 漂移或缺证据均否决本轮流程质量。接受数增大不是门，不检验 H1–H13 或新颖性。

##### 二、因素与固定条件

四 Experiment 的轴为完整 algorithm.config 补丁：ESP frozen_mean、ESP gamma_last、AR、alignment frozen_mean；基底保留共同 eval 与 version，每 case × seeds=[123] 为四个 Run。make 后核对父 config 的递归合并，避免丢失共同预算。新 version=input-mask-baseline-v031-ms-20261014-v4。

- 数据：本仓库 datasets/spec_bench/question.jsonl，translation skip=10/take=1，q171；第一回合、Qwen 非 thinking 聊天模板、max_prompt_len=2048。独立单位为一 prompt，未用于选参，不重跑历史前 10 条/类的正式格子。
- 模型：Qwen3-0.6B，按用户2026-10-14明确要求从ModelScope显式下载并准备新缓存到 shared/model/modelscope；Run保持local_files_only=true，不在执行期下载或替换snapshot。实际权重/tokenizer/config 哈希由构造/执行与终验锁定；master 不是固定 revision，不能宣称已核对远端版本。
- 原协议：float32、eager、cuda，记录实际 TF32/matmul/随机性条件；T=0、max_new_tokens=100、EOS 停止、seed123、AR 对照与逐 prompt 观测开启。没有统一 FP32 化其他模型；此处沿历史真实 0.6B 原配方。
- mask：named mask 原定义；一 mask、static [14]、pruning、BC30 不变。gamma_last 保留 β=.9/λ=.1/γ=1；frozen_mean 保留均值初始化、不更新。training-free 权重和 mask autograd 边界不变。
- alignment：四个 hindsight 步，冻结均值，计数与各层 cos 分开报告；不是 Figure 2 的 100 条 Dolly 实验。

##### 三、来源、阶段与成功条件

执行库固定 v0.3.1 / 7a15c46，唯一 Python 例外为 Windows torch-first；完整 ORIGIN/许可证保留。Study 的两份声明/recipe/阶段 Python 与 PLAN 显式进入 provenance，专项 manifest 补齐 src/spec_mtp；数据、模型/tokenizer、输入、配置和运行条件进入 raw 证据。文件新增/删除/修改与终验漂移必须拒绝。源码/输入变更先换 version，再 make/remake，不能签收旧成功 Run。

默认 prepare 构造输入与模型，execute 调用原 Algorithm；collect 写 CSV/Tracker，summarize/write 拒绝失配后才由库定稿，process 的 Study scope 只终验当前四个 case×seed。所有 Run succeeded、ESP exact_match_rate=1、来源一致且 complete/专项质量同时通过，才称前置门通过。AR 需真实 output/summary，alignment 需真实 sums/counts；二者不伪造投机 exact-match 指标。

主门为正确性/证据，accepted 是所有深度的原截断前 draft 数；深度接受、new_tokens、committed、decode_calls、τ 分别列出。速度只记录条件，不排名。n=1 不支持统计稳定性或机制优越性。

##### 四、运行、资源和停止规则

先通过源码导入、离线 Toy/Stub 模板门、隔离 tiny/失败回归、36 格原轨迹对照，再执行这四个真实 Run。GPU 串行 --round 1，每 Run 独占状态与产物；记录本机显存/主存/线程/I/O 条件与峰值，未校准不扩大并发。性能期间不跑竞争任务，不拿训练预算装箱默认值作依据。

已有缓存可用时，将本机 .cache/modelscope/models/Qwen--Qwen3-0.6B 复制到本 Study shared/model/modelscope/models 下，保留 snapshots/master 与原文件；不提交权重。然后：

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = 'src'
$env:OMP_NUM_THREADS = '1'
python -m spec_mtp make studies/input_mask_baseline --num-gpus 1 --init-gpu 0 --round 1
python -m spec_mtp launch studies/input_mask_baseline --num-gpus 1 --init-gpu 0 --console shared
python -m spec_mtp process studies/input_mask_baseline
python -m spec_mtp status studies/input_mask_baseline
python -m spec_mtp logs studies/input_mask_baseline
python -m spec_mtp report studies/input_mask_baseline
```

保留 make 的 est 与实际 wall，不把估计当实测。任何 OOM、数值/来源门失败停止扩大并保留证据；缺资源先报告具体条件。允许的规模仅四 case、单 prompt、seed123、原100-token预算。无并发、无调参、无新增输出空间实现、无完整矩阵或解码恢复验收。

##### 五、报告与边界

结果、原始输出与图保留在本 Study，报告区分执行完成、流程质量与研究假设支持；n=1 前置门不证明接受率提升、跨域泛化或论文复现。raw runs/shared/.tmp 仅本机，正式精选 JSON/图/报告随 Git。旧失败和历史数字不改写。

##### 六、2026-10-14 新设备执行准备

v0.3.1 升级及下游 report 质量写回修复改变源码来源，采用新 version / Run ID；原 v1 无真实成功格子，不复用历史正式输入空间格子。Qwen3-0.6B、q171、float32/eager/CUDA、seed123、100-token、EOS、mask 配方与四 case 不变。设备为 RTX 5090 D v2，torch 2.11.0+cu130、CUDA13、Transformers5.12.1。资源清单不等于真实模型串行校准。

检查本项目缓存、用户 ModelScope 默认缓存及常见 Hugging Face 缓存，未找到计划所需 Qwen3-0.6B snapshot。不同规模 Qwen 或图像模型不替代本模型；local_files_only=true 保持。先完成无权重展开与明确离线缺资源核验，不 launch 真实 Run，不补造 token、接受数或峰值。提供原 snapshot 后，先复核来源/模型文件，再串行校准并按当前 PLAN 执行四格；若需要下载新 snapshot，须明确实际来源与原缓存协议的差异，不能悄然替换。

##### 七、ModelScope 显式准备（2026-10-14）

用户回复“记得在modelscope下载”，授权准备Qwen3-0.6B。此前v2的缺缓存事实与四个未执行Run保留；采用v3身份，不用新snapshot冒充旧设备缓存。显式snapshot_download在make/Run之前执行，记录provider、实际缓存snapshot、模型/tokenizer/config文件哈希；可变master不伪称固定revision。下载完成后离线解析，同一snapshot用于四格，原q171/FP32/eager/T0/seed123/预算100/EOS与mask配方不改。

先单进程仅加载模型并测资源，再按原生round1串行四格。校准不更换科学精度或额外生成正式格子；记录显存/主存/线程/I/O、实际环境与外部负载，速度不排名。任一correctness/source/evidence门失败保留原Run并停止扩大，不自动放宽门或复跑旧ID。

加载校准已完成（仅加载，不生成正式token）：596049920参数，FP32/eager/cuda:0；CUDA allocator峰值2384724992 bytes，RSS采样峰值5139087360 bytes，torch intra-op线程1。运行前同时冻结snapshot文件哈希与加载校准JSON。load-only的cuDNN benchmark默认为false，没有cuDNN计算；正式Run由原生System按deterministic=false设置benchmark=true，并在raw记录实际标志，不把校准默认值覆盖到科学配置。

第一格frozen_mean用原生run-one作为正式运行同时测解码资源，不额外生成或重复此格。通过后原生launch读取round1计划，跳过第一格成功result并串行运行其余三格；不另建runner。外部0.5秒采样记录进程RSS/线程/I/O及nvidia-smi设备memory.used/利用率，设备采样峰值含背景占用且可能漏掉短峰，不冒称allocator精确峰或性能优势。实际allocator加载峰和解码采样峰分开。

##### 八、B-008纠错验收（2026-10-14）

v3四格均Run成功：两ESP exact_match=1，AR真实输出，alignment原始sums/counts完整；Study因嵌套配置轴的期望因素误比较未通过。v3证据和终验失败完整保留，不能标成整轮质量通过。修复仅将期望因素从冻结基底/声明按原生合并与Experiment分组独立推导，不改decoder/Data/Model、轴、输入、精度、预算或数值门。通过CPU嵌套轴/子集回归与完整工程门后以v4重新执行同四case，禁止给v3成功Run重新签来源。资源测量沿已验证加载器；v4继续串行并保存原始观测，与v3逐项对照不包括墙钟。

</details>

<a id="legacy-studies-output-space-4b-holdout-docs-plan"></a>

### studies/output_space_4b_holdout/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_4b_holdout/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_4b_holdout/docs/PLAN.md)；原SHA256 `1ef5c46f3c3a824e45ffff9f354c4def5f3c7fc281cd1ad19c44df030603ff29`。

#### Qwen3-4B matched output-space recheck

Frozen before new model forwards,2026-10-15 after1274bb2. User directs primary model4B, suspend0.6B rare/single-position analysis, no speed tests/push/PR/merge. Located old4B input-space first26 single-slot680/704/777, two-depth[7,2]621/704 and full48012809/15009; actual prior output0/output1 raw/result/config not found in stated search. History inventory distinguishes evidence from user recollection; this validates a new matched chain, not exact reproduction of unlocated positive output-space numbers.

##### Fixed input and comparisons

SpecBench category file order skip2/take2, categories=['summarization', 'translation'], 4 prompts, manifest IDs/rendered tokens frozen before make. Seed123, per-prompt generator123+index, nonthinking first-turn template, no shortening/drop/replacement. Qwen3-4B existing14-file local snapshot whose paths/sizes match official ModelScope listing; reuse via shared/model/modelscope junction, local_files_only=true. Actual SHA/config/index/tied-storage/model digest/dtype/backend/source/runtime gates; remote immutable revision and original download origin remain unconfirmed. Source670c744e0535dcad unchanged.

Four native Runs: matched input frozen-mean ESP, output0, output1 and fresh independent AR. All methods same static unpruned[1,1], k2/BC9, FP32/eager/CUDA, max100/EOS/T0. Outputbeta.9/beta_prime.99 and mapping/state semantics unchanged; no tuning. The input method is current research ESP frozen mean, not historical selected EMA or single-slot[14]; those numbers are background only. Actual tied weight storage mandatory. Declare H1 zero-input accepted total>0 and one-input accepted total>0; any nonpositive contrast contradicts that bounded gain prediction, not a correctness failure. Separate one-zero contrast and per-prompt/category signs. No superiority/population assertion from these fixed prompts.

Count accepted drafts before budget/EOS clipping; count main generation model calls as prefill+decode for spec and ar_calls for AR. Internal AR audit calls separate, not added to main-path call comparison. Pair by exact qid/input/output and seed; every spec output must equal own AR AND separate AR Run token-for-token. Retain complete depth/accepted/cache/call traces and all negative/tie rows. Source/input/model/context/OOM/nonfinite/AR failure stops expansion, preserve failed Runs; never relax gate or change precision for a good result.

##### No benchmarking and resource schedule

eval.warmup=false. Study-only UntimedEval replaces decoder module clock with constant zero during execute, without touching global time or scientific src; native-required timing fields are explicit disabled zero sentinels, no latency/throughput/speedup measurement or conclusion. Native collect/report compatibility fields may be zero, and formal report excludes them. Wrapper tests prove clock restoration/token/state/call identity. Build same HF model directly onCUDA via device_map to avoid a full16GB FP32 host copy; same wrapper/config/actual model digest, no offload/quantization. Study-local adapters/hash frozen and restore scoped construction hooks.

SingleGPU/round1 serial. First Run is capacity calibration (same planned prompts, no repeated successful cells), then remaining three; .5s RSS/available-RAM/globalGPU sampling for OOM risk only. Est first26~15min/all4 (planning, not throughput); stop resource failure rather than kill another task. Budget4x4x100 plus3x4x100 internal AR audits = 2800new tokens, zero teacher/warmup, no single-window interventions. Realized EOS lower bound reported. Model loading/hashing/native finalization may be substantial but no timing ranking.

##### Follow-up and archival

After first26 all quality gates pass, execute separately versioned predeclared holdout: summarization/translation skip2/take2 (qid243244163164), four new prompts, same four Runs/configs; regardless positive/negative primary signs, no selection by outcomes. Its purpose is limited out-of-development paired confirmation, not full matrix or mechanistic analysis. Old0.6B reports/raw/weights/cache/commits remain historical, no new rare/window work. Archive status and unresolved questions with links, not deletion/moving raw. Formal tables/paired count plots/provenance/report/local commit; raw Runs/model/cache local only. No remote, new PR or merge.

</details>

<a id="legacy-studies-output-space-4b-input-control-docs-plan"></a>

### studies/output_space_4b_input_control/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_4b_input_control/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_4b_input_control/docs/PLAN.md)；原SHA256 `256422682fbfe1e2a86f334653c3be279c10263dc77c0c6d7aa190df58e3d5b3`。

#### Fixed historical input-EMA matched-chain control

Frozen2026-10-15 after first26 and fixed4-new-prompt reports, before any new EMA forwards. Primary input/zero/one accepted238/329/321 and calls1955/1866/1876; new4 accepted51/39/47 and calls189/201/194, all independentAR. Historical selected input EMA beta=.9999/lam=.1/gamma1 improved wider[14]/[7,2] input baselines, but its strength on this chain is unknown. This control addresses that comparison gap without retuning outputs or searching prompts/parameters.

Two native EMA-only Runs: reuse exactly original first26 and fixed4 manifests/seed ordering/model snapshot/runtime. Same FP32/eager/T0/EOS/max100/static unpruned[1,1]/k2/BC9; input EMA levelorder0, beta.9999, lam.1, step_scale1, normalizefalse, extrapolate_emafalse, update=true from current named mean_plus_gamma implementation. No other beta/gamma/lam tested. Frozen mean/output0/output1/fourthAR successful Runs are reused by hashes and never rerun. eval.num_eval_prompts26 is upper limit; second dataset has exactly4 rows, no filtering/truncation. Require exact original input IDs/count/order and full output equal own freshAR AND preceding separateAR, same actualmodel/source/runtime/flags; keep all signs and failures.

H2 bounded strong-baseline persistence: both outputs accepted totals exceed the fixed selected input EMA in each group. Report signs per prompt and group, not just pooled30; a nonpositive contrast rejects that bounded group's persistence prediction. A weaker EMA than frozen mean remains a result, not proof it is a stronger method here. Prior EMA selection used first26, so these are comparative development outcomes, not held-out population significance.

Budget2*(26+4)*100=6000 new generatedtokens including ownAR audits, zero teacher/warmup; no new output or independentAR generation. Same tested Study-only directCUDA placement/disabled decoderclock/native phases, 0 time sentinels not measurements; count main prefill+decode separately from audit ar_calls. Serial singleGPU/round1/.5s capacity sampling, no timing rank. Stop/preserve source/model/input/context/OOM/nonfinite/AR/Run gate failure, no replacement or same-success re-signing. Weights/cache/fullRuns local; formal paired counts/complete semantic evidence/report/local commit. No scientificsrc/default change, 0.6B rare/window work, push, PR or merge.

</details>

<a id="legacy-studies-output-space-4b-recheck-docs-plan"></a>

### studies/output_space_4b_recheck/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_4b_recheck/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_4b_recheck/docs/PLAN.md)；原SHA256 `e7e34debe59388340c71cee2e922f9a1778b2e629689eac39a2c86e64fa28a63`。

#### Qwen3-4B matched output-space recheck

Frozen before new model forwards,2026-10-15 after1274bb2. User directs primary model4B, suspend0.6B rare/single-position analysis, no speed tests/push/PR/merge. Located old4B input-space first26 single-slot680/704/777, two-depth[7,2]621/704 and full48012809/15009; actual prior output0/output1 raw/result/config not found in stated search. History inventory distinguishes evidence from user recollection; this validates a new matched chain, not exact reproduction of unlocated positive output-space numbers.

##### Fixed input and comparisons

SpecBench category file order skip0/take2, categories=all13, 26 prompts, manifest IDs/rendered tokens frozen before make. Seed123, per-prompt generator123+index, nonthinking first-turn template, no shortening/drop/replacement. Qwen3-4B existing14-file local snapshot whose paths/sizes match official ModelScope listing; reuse via shared/model/modelscope junction, local_files_only=true. Actual SHA/config/index/tied-storage/model digest/dtype/backend/source/runtime gates; remote immutable revision and original download origin remain unconfirmed. Source670c744e0535dcad unchanged.

Four native Runs: matched input frozen-mean ESP, output0, output1 and fresh independent AR. All methods same static unpruned[1,1], k2/BC9, FP32/eager/CUDA, max100/EOS/T0. Outputbeta.9/beta_prime.99 and mapping/state semantics unchanged; no tuning. The input method is current research ESP frozen mean, not historical selected EMA or single-slot[14]; those numbers are background only. Actual tied weight storage mandatory. Declare H1 zero-input accepted total>0 and one-input accepted total>0; any nonpositive contrast contradicts that bounded gain prediction, not a correctness failure. Separate one-zero contrast and per-prompt/category signs. No superiority/population assertion from these fixed prompts.

Count accepted drafts before budget/EOS clipping; count main generation model calls as prefill+decode for spec and ar_calls for AR. Internal AR audit calls separate, not added to main-path call comparison. Pair by exact qid/input/output and seed; every spec output must equal own AR AND separate AR Run token-for-token. Retain complete depth/accepted/cache/call traces and all negative/tie rows. Source/input/model/context/OOM/nonfinite/AR failure stops expansion, preserve failed Runs; never relax gate or change precision for a good result.

##### No benchmarking and resource schedule

eval.warmup=false. Study-only UntimedEval replaces decoder module clock with constant zero during execute, without touching global time or scientific src; native-required timing fields are explicit disabled zero sentinels, no latency/throughput/speedup measurement or conclusion. Native collect/report compatibility fields may be zero, and formal report excludes them. Wrapper tests prove clock restoration/token/state/call identity. Build same HF model directly onCUDA via device_map to avoid a full16GB FP32 host copy; same wrapper/config/actual model digest, no offload/quantization. Study-local adapters/hash frozen and restore scoped construction hooks.

SingleGPU/round1 serial. First Run is capacity calibration (same planned prompts, no repeated successful cells), then remaining three; .5s RSS/available-RAM/globalGPU sampling for OOM risk only. Est first26~15min/all4 (planning, not throughput); stop resource failure rather than kill another task. Budget4x26x100 plus3x26x100 internal AR audits = 18200new tokens, zero teacher/warmup, no single-window interventions. Realized EOS lower bound reported. Model loading/hashing/native finalization may be substantial but no timing ranking.

##### Follow-up and archival

After first26 all quality gates pass, execute separately versioned predeclared holdout: summarization/translation skip2/take2 (qid243244163164), four new prompts, same four Runs/configs; regardless positive/negative primary signs, no selection by outcomes. Its purpose is limited out-of-development paired confirmation, not full matrix or mechanistic analysis. Old0.6B reports/raw/weights/cache/commits remain historical, no new rare/window work. Archive status and unresolved questions with links, not deletion/moving raw. Formal tables/paired count plots/provenance/report/local commit; raw Runs/model/cache local only. No remote, new PR or merge.

</details>

<a id="legacy-studies-output-space-4b-recheck-v2-docs-plan"></a>

### studies/output_space_4b_recheck_v2/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_4b_recheck_v2/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_4b_recheck_v2/docs/PLAN.md)；原SHA256 `4a92e2c3fcddd4760a6823c66da536120fed0811d09741b174a8e18d3f06eeac`。

#### Qwen3-4B matched output-space recheck

Frozen before new model forwards,2026-10-15 after1274bb2. User directs primary model4B, suspend0.6B rare/single-position analysis, no speed tests/push/PR/merge. Located old4B input-space first26 single-slot680/704/777, two-depth[7,2]621/704 and full48012809/15009; actual prior output0/output1 raw/result/config not found in stated search. History inventory distinguishes evidence from user recollection; this validates a new matched chain, not exact reproduction of unlocated positive output-space numbers.

##### Fixed input and comparisons

SpecBench category file order skip0/take2, categories=all13, 26 prompts, manifest IDs/rendered tokens frozen before make. Seed123, per-prompt generator123+index, nonthinking first-turn template, no shortening/drop/replacement. Qwen3-4B existing14-file local snapshot whose paths/sizes match official ModelScope listing; reuse via shared/model/modelscope junction, local_files_only=true. Actual SHA/config/index/tied-storage/model digest/dtype/backend/source/runtime gates; remote immutable revision and original download origin remain unconfirmed. Source670c744e0535dcad unchanged.

Four native Runs: matched input frozen-mean ESP, output0, output1 and fresh independent AR. All methods same static unpruned[1,1], k2/BC9, FP32/eager/CUDA, max100/EOS/T0. Outputbeta.9/beta_prime.99 and mapping/state semantics unchanged; no tuning. The input method is current research ESP frozen mean, not historical selected EMA or single-slot[14]; those numbers are background only. Actual tied weight storage mandatory. Declare H1 zero-input accepted total>0 and one-input accepted total>0; any nonpositive contrast contradicts that bounded gain prediction, not a correctness failure. Separate one-zero contrast and per-prompt/category signs. No superiority/population assertion from these fixed prompts.

Count accepted drafts before budget/EOS clipping; count main generation model calls as prefill+decode for spec and ar_calls for AR. Internal AR audit calls separate, not added to main-path call comparison. Pair by exact qid/input/output and seed; every spec output must equal own AR AND separate AR Run token-for-token. Retain complete depth/accepted/cache/call traces and all negative/tie rows. Source/input/model/context/OOM/nonfinite/AR failure stops expansion, preserve failed Runs; never relax gate or change precision for a good result.

##### No benchmarking and resource schedule

eval.warmup=false. Study-only UntimedEval replaces decoder module clock with constant zero during execute, without touching global time or scientific src; native-required timing fields are explicit disabled zero sentinels, no latency/throughput/speedup measurement or conclusion. Native collect/report compatibility fields may be zero, and formal report excludes them. Wrapper tests prove clock restoration/token/state/call identity. Build same HF model directly onCUDA via device_map to avoid a full16GB FP32 host copy; same wrapper/config/actual model digest, no offload/quantization. Study-local adapters/hash frozen and restore scoped construction hooks.

SingleGPU/round1 serial. First Run is capacity calibration (same planned prompts, no repeated successful cells), then remaining three; .5s RSS/available-RAM/globalGPU sampling for OOM risk only. Est first26~15min/all4 (planning, not throughput); stop resource failure rather than kill another task. Budget4x26x100 plus3x26x100 internal AR audits = 18200new tokens, zero teacher/warmup, no single-window interventions. Realized EOS lower bound reported. Model loading/hashing/native finalization may be substantial but no timing ranking.

##### Follow-up and archival

After first26 all quality gates pass, execute separately versioned predeclared holdout: summarization/translation skip2/take2 (qid243244163164), four new prompts, same four Runs/configs; regardless positive/negative primary signs, no selection by outcomes. Its purpose is limited out-of-development paired confirmation, not full matrix or mechanistic analysis. Old0.6B reports/raw/weights/cache/commits remain historical, no new rare/window work. Archive status and unresolved questions with links, not deletion/moving raw. Formal tables/paired count plots/provenance/report/local commit; raw Runs/model/cache local only. No remote, new PR or merge.

##### Activation v2 after the preserved collect failure

First26 v1 input Run completed execute with26/26 exact AR and zero timing sentinels, accepted238/decode_calls1929, but Study collect/__init__ exported a module rather than callable run; final status failed. No scientific acceptance of that Run. Preserve v1 declarations/locks/raw/results unchanged. V2 new Study and algorithm.version repairs only four phase exports to existing from .run import run pattern; model/config/inputs/output algorithms and source670c744e0535dcad unchanged. Add phase-entry callable regression before new make. New first input Run is a fresh calibration under v2, not re-signing v1. The not-yet-made holdout receives the same callable source repair before its first freeze/forward. All successful cells are still never rerun.

</details>

<a id="legacy-studies-output-space-cross-task-docs-canonical-plan"></a>

### studies/output_space_cross_task/docs/CANONICAL_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/CANONICAL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/CANONICAL_PLAN.md)；原SHA256 `0536c43a5e6809e3ac6c451cd401018f7371b8b4d79d0f5dc430303e78ac7d39`。

#### First cross-task H2 canonical follow-up

Predeclared 2026-10-14 after native and fixed-window cross-task outcomes, before any canonical forward. Separately versioned diagnostic, not an adaptive cell appended to the native batch; all existing PLAN/EXECUTION_PLAN/checker/raw/hypothesis bytes remain unchanged.

Selection exactly follows the prior follow-up criterion: first file-order feasible unit with accepted-changing transfer isq241, entry716, order1<-order0 changes11→12. No search for best gain or alternative root. Test whether that contrast persists under canonical committed anchor/v at every actual root>716; incoming proxies retained, true KV root verified. Native controls use original runtime/defaults/seed123, ModelScope/snapshot/source and original masks from the completed diagnostic.

Four generations: native order1 self, native one<-zero, canonical self, canonical one<-zero; budget400 tokens. Two native controls must reproduce previous complete AR/semantic trace and all captured tensors. One teacher-forced full independent AR prefix792 positions (692 prompt+100 new) with final hidden only; exact-root cosine>=.99999 against baseline sampled roots. Reuse native canonical_states() and synchronized_transfer(); no new EMA/decoder implementation. Teacher packing differs and is not exact real arithmetic. Canonical self may change; compare its graft against it, not only native baseline. Full output AR equality and original through716 semantics mandatory.

After716 replace only anchor/v with per-prefix CPU-FP32 chronological native observe_committed states; preserve incoming proxies at each patch, but later forward-generated proxies may change. Do not force previous roots/owners. Four generations plus1 teacher, no warmup/other auxiliary model forward. Source/snapshot/oldraw tensor SHA/newtool/oldhelpers/currentplan/model/runtime before/after; load large1.9GB raw via read-only mmap, not a full eager copy. SerialGPU/OMP-MKL1/.5s sampling; priornative~6.7GBRSS/7.3GiBGPU, teacher792-position logits adds memory, calibrate actual/stop on pressure/OOM/hash/nonfinite/output/cos/scope/root failure.

If canonical contrast remains+1 with stable self, this supports bounded persistence and excludes committed numeric microdifferences as necessary for this local contrast under synchronization. Vanishing contrast is a bounded counterexample; changed self constrains attribution. Do not infer pure proxy mediation, default gain or other prompt generality. Original hypotheses_v1.json H2=not_tested is retained as batch history; follow-up result is separate. Full weights/cache/tensors/Run local only; no source/default/remote/PR/deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-execution-plan"></a>

### studies/output_space_cross_task/docs/EXECUTION_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/EXECUTION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/EXECUTION_PLAN.md)；原SHA256 `7051210ee105a6b449a8bf0562e2efd7a75c10e1357d31c3f8bfbaa3ad271040`。

#### q241–q244 real execution addendum

2026-10-14 after4c9ad8e, before first real make. Current user authorizes the bounded real cross-task batch. Original PLAN/CHECKER_PROTOCOL draft bytes remain unchanged; their draft-only status is historical. This addendum activates exactly their fixed native/window/H1/H3 protocol, no new hypothesis or prompt/window search.

Version output-space-summarization-241-244-20261014-v1, same rendered IDs692/647/595/901 and11-file local ModelScope snapshot/source670c744e0535dcad. Four native cells input/output0/output1/AR, T0/EOS/max100, beta.9/beta_prime.99/static[1,1]/k2, original native runtime defaults. Only this new Study version/provenance changes, no scientific/default recipe changes.

Before first make verify snapshot/dataset/rendered IDs, reuse shared model via junction, singleGPU/round1/OMP-MKL1. Run the first frozen_mean_chain cell serially for capacity calibration, record .5s sampled RSS/GPU/available RAM; proceed with one launch chain for remaining cells only after first gate/resource check. No completed cell reruns or reuse of older Run directories. Native token budget1600+1200 audit+72 unchanged warmup; diagnostics20x100=2000, no auxiliary model forwards. Stop/preserve OOM/nonfinite/hash/output/root/scope/context failures; do not shorten prompts or relax gates.

Native source/factor/raw/AR/Study/process gates→report→checker export→frozen window selection→unchanged diagnose_study_windows→checker verdict. Preserve actual positive/null/infeasible and legitimate depth2 witnesses; H2 remains untested until another frozen canonical protocol. Execution evidence/full scalar traces formal; raw tensors/cache/weights/completeRun local only. Parallel independent work is planning/coding only; GPU diagnostics for q178 wait until native/window GPU processes exit.

No push, new PR, merge or deletion. Existing workspace preserved.

</details>

<a id="legacy-studies-output-space-cross-task-docs-first-delta-plan"></a>

### studies/output_space_cross_task/docs/FIRST_DELTA_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/FIRST_DELTA_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/FIRST_DELTA_PLAN.md)；原SHA256 `bda13e540bb5bfa0b2853fc3dac85cdd8d999dbace65fff959d27830b40f5350`。

#### First local acceptance change versus final full-path direction

Frozen after the completed expansion and its STUDY_REPORT update, before inspecting/scoring first differences,2026-10-15. This is an explanatory test on already observed final outcomes, not a new held-out stochastic sample. No new model forwards/draws/inputs/window selection.

H7: the sign of the first chronological semantic self/graft difference's accepted-node-count delta predicts the sign of the final accepted-count delta. Use all10 new completed sampled units plus2 old development rows separately; do not choose a helpful root or omit cancellation/nulls. Compare TRACE_KEYS only, because mask norm/hidden diagnostic metadata can differ before any accepted-path effect. Shared prefixes imply the first divergent trace record has the same actualroot; require this, do not force later alignment.

Verify per-cell accepted equals sum of trace accepted_nodes before final budget/EOS clipping. Preserve actual common roots and each side's unique roots. Reconstruct final delta exactly as common-root node-count difference + graft-only accepts - self-only accepts; this is a trace accounting identity, not tensor causal mediation or a forced baseline path. Report firstgain→finalloss, firstloss→finalgain and firsteffect→netzero counterexamples. Identical traces must imply zero delta; missing/truncated/duplicate-root records fail, never default to zero.

Complete script with tests for sign reversal, cancellation, preserved-direction controls, first-record prefix mismatch and damaged counts. Bind original sampled reports, expansion analysis, tool/plan/source before/after; formal structured comparisons only. Then update report with actual H7 verdict and new/old scope. No source/default/old evidence edit, remote/PR/merge/delete.

</details>

<a id="legacy-studies-output-space-cross-task-docs-kernel-execution-plan"></a>

### studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md)；原SHA256 `65519e3e0dbe7aaf8ae391afb6b1a02637eb02016e2214e33fd65cd63705c578`。

#### Fixed q_proj protocol execution addendum

2026-10-14 after a50f9e0, before measurements. KERNEL_PLAN.md remains the original draft byte-identical. Current authorization permits the reversible local experiment described there, independently of provider norm controls.

Execute exactly the six planned canonical q178 generations: native self/graft read-only captures of source blocks85/86, plus source85 owner0 slot1 row4 q_proj singleton-FP32 or CPU-FP64-rounded output substitutions, each self/graft. Reuse previous canonical states/donors; no teacher forward. Four cloned-cache exact block replay forwards only. All selected logits/final hiddens in those replays must reproduce captured values bit-for-bit; restore a fresh cloned cache, never original live cache. Pre-forward cache snapshots hold independent storage and their KV tensors remain unchanged after replay.

Profile layer0 q_proj packed[1,9,d] and selected singleton[1,1,d] for each source85 path; correlate aten::mm/addmm CPU IDs to actual CUDA kernels, retain shapes and names even if equal. Direct numerical comparisons additionally cover captured block86. High-precision CPU linear uses exactly represented FP32 activations/weights/bias, casts output only back toFP32. It changes one operation, not model precision. No operator/root/layer search.

Native capture cells must match prior canonical full semantic traces/selected samples and independent AR. At patch assert actual source85, row4, untouched other rows, finite replacement, one application and removable hooks. Complete variant AR/scope/state gates remain mandatory; native paths need not hold after substitutions. Budget600 new tokens plus4 exact block auxiliary forwards; source/plan/model/11-file snapshot/tool/helpers/oldraw/native config/inputs bound before/after, strict deterministic process settings, serialGPU/.5s resources. Preserve failures and no source/default/remote/data deletion.

Separately q241–244 receives a runnable evidence-export/verdict script and deterministic synthetic minimal reproductions. This turn does not launch its real-model batch. H1/H3 real results remain unmeasured; H2 needs a later frozen canonical follow-up. This explicitly limits the next-step claims.

</details>

<a id="legacy-studies-output-space-cross-task-docs-kernel-plan"></a>

### studies/output_space_cross_task/docs/KERNEL_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/KERNEL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/KERNEL_PLAN.md)；原SHA256 `662bb5e126e48dbfdf5e8a5ad69d8039f7fa842c8e7796eea468f5c1a41c54b7`。

#### Transformer proxy-rounding / CUDA packing protocol draft

Unexecuted protocol; separate from the completed provider NormTwoOps controls. Those controls replace provider math on identical inputs, and do not observe all operations that create transformer proxy hiddens. Do not attribute pos88/root89 committed delta to the profiled provider kernel.

##### Question and fixed unit

Q178, original root85 owner0 second-slot graft, strict FP32 canonical anchor/v controls, same ModelScope snapshot/source/seed126/AR112. Capture source blocks85 and86 for self and one<-zero, including both owner0 mask rows and exact/draft rows. Self/graft source inputs differ semantically, so their hidden delta is not an estimate of rounding error. Numerical counterfactuals instead compare each source block with itself using exactly the same inputs, cache and model weights.

Predeclare operator target: layer0 q_proj linear output, selected owner0 mask slot1 row of block85. This fixed choice tests a narrow GEMM/packing hypothesis and avoids searching for a kernel with a favorable outcome. A null at this target leaves later layers, RMSNorm, attention and hidden store rounding unresolved.

##### Capture, identical-input replay and attribution

1. Read-only hooks retain inputs_embeds, positions, complete4D attention and every pre-forward KV layer tensor; clone before mutable cache pruning, verify storage independence. Persist shapes/dtypes/strides, hashes and exact layer0 q_proj input row/weight/bias/output. No hooks edit science src. Retain source row/true owner from the decoder and remove hooks in finally. Two complete strict canonical generations reproduce old self8/27 and graft9/26, independent AR and old selected windows. If capture alters them, stop.
2. Restore a fresh cloned pre-forward cache for each auxiliary replay. Native exact block replay must reproduce all selected logits/hiddens bit-for-bit. Profile the original q_proj on its actual packed input, and the identical selected row via singleton FP32 linear; associate CPU operator external IDs, dimensions and concrete CUDA launch names. A name can remain equal across shapes: record it, do not invent a different kernel variant.
3. Replay only that same row with CPU FP64 linear from exactly represented FP32 input/weight/bias, round output to FP32. Compare native packed FP32, singleton GPU FP32 and CPU FP64-rounded output. The GPU packing comparison fixes dtype and arithmetic operator; the FP64 comparison changes arithmetic rounding. CPU FP64 is a local reference, not exact full-model truth. Other rows/layers are unchanged.
4. Two interventions during complete canonical decoding: replace only the predeclared row of the predeclared source-block q_proj output with singleton FP32 output, or CPU FP64-rounded output. Run each with self and graft; retain a native original-value self/graft pair from capture. At every patch assert all other output rows bit-equal, same inputs/weights/state, actual block/row, and no candidate/owner forcing. Immediate proxy hidden/mask/logit differences and later traces are outcomes, not fixed to old values.
5. If captured native packed versus singleton differs and the scoped replacement changes the self/graft contrast, this demonstrates sensitivity to the q_proj operation's packing/backend realization at that input. It does not isolate an internal CUDA instruction or explain every path-dependent hidden delta. If FP64 replacement alone changes the contrast, it shows local arithmetic-rounding sensitivity. If neither changes it, reject necessity of that layer0/source-row operation for this fixed contrast; preserve all other open operations.

##### Budget and gates

Six generations (capture native self/graft+four interventions)x100=600 tokens, at most4 auxiliary exact block model forwards (2 sources per path); direct linear replays2 sources x3 modes with reused weights, no new full model or FP64 model. One GPU process, OMP/MKL1, strict runtime recorded, .5s resources and fresh output directory. Extra capture cache increases memory: calibrate first self capture serially; stop/preserve on pressure or cloning failure. Hash source/model/11 snapshot files/input/native/previous raw/tool/this protocol before/after. AR equality, native/capture/replay gates, legal depth2 flags and patch cleanup required. CUPTI unavailability preserves the kernel-name gap, no installation or guessed attribution.

No code for this protocol has been run or added as a scientific method. No new default, push, merge or deletion. Draft choices are intentionally narrow and falsifiable; a later operator change requires a new version before measurement.

</details>

<a id="legacy-studies-output-space-cross-task-docs-local-factor-plan"></a>

### studies/output_space_cross_task/docs/LOCAL_FACTOR_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/LOCAL_FACTOR_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/LOCAL_FACTOR_PLAN.md)；原SHA256 `3fb58d846c188137ac70bc6d39f270922bfe79ca009a2de1acb07593a74122c2`。

#### q241 fixed-block state/proxy timing experiment

Frozen after4a02551 before new forwards, 2026-10-15. Explain the observed first divergence atroot720; selection is post-observation, not a new independent sample. Decoder constructs the candidate tree from the preceding mask logits before current provider mapping. Therefore test source719 and current720 separately, not a mislabeled current-state intervention on already constructed candidates.

Two canonical self/graft full generation controls, same q241 seed123/entry716/budget100/default runtime, must reproduce prior full AR/trace/stats/all captured tensors. Capture fresh-cache frames from canonical self at719 and720; record both paths at actual716→717→719→720. Clone actual input/positions/attention/KV before each selected block, and capture logits/final hidden. Do not fabricate unvisited718. No teacher, no new Run or old re-signing.

At each fixed frame, cross immediate committed state (canonical self versus native self captured at that same root) with incoming proxy (canonical self versus canonical graft captured at that same root). Recompute only6 mask embeddings using unchanged native OutputHiddenMaskProvider, holding FP32 native scale/order1/beta.9/beta_prime.99. Hold root/node embeddings, positions, attention and every KV entry. Execute all4 cells3 times with fresh cloned cache:24 block forwards total. Native-equivalent cell must exactly reproduce full saved logits/hidden; repeated outputs must be bit-identical. All root/node logits must be bit-identical across mask-only changes because tree attention prevents future masks leaking into verification. Saved cache, states/proxies and first3 input rows stay unchanged. Report if any factor is itself ineffective rather than replacing it with arbitrary noise.

Source719 owner0 rows3/4 seedroot720. Report first/second AR margins, generated seed IDs, mask-input change and final proxy hidden; root720 candidate acceptance comes from the previously fixed tree, not its current masks. Root720 interventions test immediate verification equivalence and subsequent seed differences, not full downstream generations. Check source owner/positions against original actual trace and record all four steps' candidates, decisions and state/proxy differences. This separates mapping effects on a fixed cache/tree context; it does not claim effects generalize to the graft context or uniquely partition full path effects. Keep nulls and backend numerical differences explicit.

Budget200 newtokens plus24 block forwards, zero teacher/warmup. SingleGPU/OMP-MKL1/.5s memory calibration; stop/preserve OOM, nonfinite, cache/root/scope/output/byte/runtime drift. Bind raw reports/tensors/plan/tool/old helpers/model/11-file ModelScope snapshot before/after; full frames/cache/tensors local only. No science/default edits, push, PR or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-owner-plan"></a>

### studies/output_space_cross_task/docs/OWNER_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/OWNER_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/OWNER_PLAN.md)；原SHA256 `da5faf4eaccd41f97f098dfc5f3630e0780c87b5bd59c05e2a46eaff954b36f8`。

#### q241 actual owner1 fixed-frame state/proxy control

Frozen after6cca8fc before new forwards/draws, 2026-10-15. Next different actual owner issource717 owner1, selected from the already measured716→717→719 chain, not by favorable margins. Owner1 masks rows5/6 atpositions719/720 seedroot719; do not score owner0 rows3/4 as the next-path seeds.

Two q241 seed123/T0 canonical self/graft full generation controls must match prior AR/trace/stats/all captured tensors; capture canonical-self source717 input/positions/attention/cache/logits/hidden, fixed root/tree/KV. Reuse unchanged synchronized_transfer and prior teacher states, zero teacher. Four S/P cells (canonical versus native self committed state, canonical self versus graft proxies), three fresh-cache repeats per cell,12 block forwards. Only six mask embeddings change; preserve root/node verification outputs and saved cache, position, attention, state/proxy; native-equivalent cell exact, repeats exact. Compare actual owner1 seed logits with previous graft next-root719 seed, even if they fail to recover a correct greedy candidate. No arbitrary noise or alternative owner search.

Conditional acceptance at this other owner uses the old actual verification root719 logits from the saved fixed-frame experiment, native sampler T=.8/1, seeds200..263, same64 verifier draws under each S/P cell, exact candidate softmax masses. Distinguish changed candidate/seed from a restored hit. Root719/root720 contexts and mass differences differ, not paired evidence that an owner index itself caused the effect.

Budget200 full-generation tokens +12 direct blocks +128 conditional verifier draws, zero teacher/warmup. The sampled-path study remains separate. Bind old native/canonical reports+tensors/new plan/tool/old local helpers/scientific sources/11 snapshot/model/runtime before/after; .5s singleGPU monitoring. Raw cache/frame tensors local only; no source/default edits, old re-sign, remote/PR/merge/delete.

</details>

<a id="legacy-studies-output-space-cross-task-docs-plan"></a>

### studies/output_space_cross_task/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/PLAN.md)；原SHA256 `07abd854854816d51c4770939e14e095e6f9931dbe62698528f6e58d19475df9`。

#### Cross-task falsification Study draft

Draft 2026-10-14, scientific source670c744e0535dcad after f3b95ea. This round prepares the Study and tokenizer input manifest only; no make, launch, native Run, source lock or result exists. Do not interpret expanded configs as measured results. Inputs and gates must be verified again before the first launch, then retain this contract byte-identical or use a new version if the contract changes.

##### Evidence motivating the questions

Twelve translation development units q171–q182: pooled input/output0/output1 accepted60/45/47. Natural order1 gains occur atq173/q179, loss atq182; the remaining nine have equal pooled accepted, sometimes differing depth paths. Fixed bidirectional transfer changes accepted on4 units and not on8, including terminal/unused-owner controls. Q178 strict canonical states preserve8→9; q182 has a later valid second-slot restoration under equal source/root anchor/v but lacks a full canonical-state control. Seed changes alone do not imply accepted changes. This is development evidence, not random sampling or significance.

The executed provider arithmetic/norm controls in the extension preserve q178 full self/graft paths across all three operator substitutions. Transformer-created proxy hidden rounding remains open, covered by KERNEL_PLAN.md; this draft does not claim it is ruled out.

##### Predeclared falsifiable hypotheses

H1 cross-task occurrence: at least one of four fixed summarization units has a natural output-order semantic difference or a fixed one-slot transfer semantic difference. If all four complete with none, reject this bounded occurrence prediction; do not replace prompts/windows after seeing nulls. Equal total accepted with different depth/owner/KV paths counts as semantic difference but must be distinguished from net gain.

H2 committed-state necessity: if a new unit has an accepted-count transfer change, it persists under a separately predeclared canonical anchor/v self-versus-graft comparison at the same fixed entry. A disappearing change with stable canonical self falsifies this local necessity exclusion; changed self constrains the comparison. Native and canonical results must be separate. The current cross-task budget does not include adaptive canonical Runs: retain the first eligible result and draft a new frozen protocol/version before such a run, rather than expanding this batch opportunistically. Q178 supports one instance, q179/q182 remain unresolved.

H3 first-slot-only explanation: any legal second-depth change while its first slot stays the same correct AR token falsifies it. Already contradicted onq178/q182; this batch tests whether that counterexample appears in another task. Never score second-depth margins when the first token is wrong or EOS removes the next target. Retain first-only gains/losses and no-change units.

H4 numeric-artifact explanation: q178 effect requires provider EMA/slot or norm rounding. Existing S64/N64/N32 controls contradict necessity of the post-entry provider stages in those fixed cells; initial root85 mapping was left native. The narrower untested transformer hypothesis is defined in KERNEL_PLAN.md; a kernel name alone is insufficient to accept it.

##### Fixed inputs and conditions

Summarization file order skip0/take4, q241/242/243/244. DatasetSHA7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21; rendered input lengths692/647/595/901 with exact ID/template hashes in prompts_manifest.json. Same local ModelScope11-file Qwen3-0.6B snapshot, tied actual weights, FP32/eager/CUDA, nonthinking, T0/EOS/max100, static unpruned[1,1]/k2/BC9, beta.9/beta_prime.99, original native deterministicfalse. Seeds123+i; algorithm version output-space-summarization-241-244-draft-v1. No prompt shortening; any input mismatch, unsupported context or source drift stops the batch and is reported.

Four new native Runs: input frozen mean, output0, output1, independent AR. Native make→launch→process→report, all source/model/input/factor/raw/AR/Study gates, full output token equality required. No successful old Run is rerun or re-signed.

After native success, freeze a selection manifest using existing choose_window(), identical to the q175–q182 procedure: earliest shared acceptance with preceding common rejection; otherwise first continuing shared rejection; otherwise continuing unused-owner control; no feasible continuing root is retained explicitly. Capture full original-root union from entry; use actual visited roots and owners after transfer. Two baseline observations, both donor directions and order1 original-value placebo per feasible unit, owner0 second slot[0,1] once only; independent AR and through-entry semantic equality hard gates. Original baseline/placebo must reproduce full native trace. Reuse measured helpers without editing their bytes. The manifest must supply rendered_prompts/question_id for diagnose_study_windows.py; native rendered IDs must equal the preflight IDs, not just qids.

##### Metrics, budget and decisions

For each unit report native accepted by depth, calls, all actual semantic trajectories, legal two-slot target margins, matched-state anchor/v equality, source owner and proxy/seed differences, gains and losses, infeasible/noncommon roots. Separate H1 occurrence, accepted net gain and H3 depth-only effects. No power/significance claim from n=4, no win-rate estimate from the prior12; no default retuning or speed conclusion.

Serial singleGPU/round1/OMP-MKL1. Native budget1600 new tokens+1200 internal AR audit+72 existing warmup; diagnostics at most20x100=2000 new tokens, no auxiliary transformer forward. Initial inputs are longer than the translation calibration, so first native Run calibrates actual host/KV/GPU usage before continuing. Prior6.7GB RSS/~6.3GiB globalGPU is context, not this batch's measurement. Sample.5s; retain est/actual and explicit background-memory limitation. Stop/preserve OOM, nonfinite, output/scope/source/hash/root failure, context shortening or resource pressure. No replacement prompt or relaxed gate.

Full weights/cache/Run/profiler traces/tensors local only; formal summaries and selected scalar provenance can be reviewed separately. No push/merge/delete. Next decision after this bounded batch: accept/reject the occurrence prediction, retain every positive/null, and choose a frozen canonical or transformer-operator follow-up based on the declared criterion. No full matrix is implied.

</details>

<a id="legacy-studies-output-space-cross-task-docs-rare-event-plan"></a>

### studies/output_space_cross_task/docs/RARE_EVENT_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/RARE_EVENT_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/RARE_EVENT_PLAN.md)；原SHA256 `414eacd84fbca42458ed8258c6692033075c1b5e776d2515b1a2c533ddb5f11b`。

#### Actual owner1 rare-event detectability boundary

Frozen aftera5ede78 before new draws,2026-10-15. Use old actual root719 verification logits, source717 owner1 candidates11/323, no new model forward or altered temperature. T=.8 and1, fresh seeds300..4395,4096 independent-generator single-token native CUDA select_token calls per temperature (8192 verifier draws total). Reuse one realized draw bank across four S/P factor cells. Keep all sampled token IDs and RNG-consumption gates; do not turn zero observations into probability zero.

Bind actual CUDA-FP32 softmax masses and CPU-FP64 reference separately, same-backend reconstruction mandatory. Compute explicitly idealized Bernoulli detectability from mass p: P(at least one inN)= -expm1(N*log1p(-p)), trials for95% one-event detection=ceil(log(.05)/log1p(-p)), and one-sided95% upper bound afterzero/N= -expm1(log(.05)/N). Validate0<=p<=1/N>0 endpoints and stable tiny-p math. This is a theoretical probability-model power calculation, not proof that native finite-precision CUDA sampling implements extremely small probabilities exactly.

Budget8192 native verifier draws, zero model forward/teacher/generated continuation tokens; no billions-draw brute force, boosted logits or rare event replacement. Source/helper/report/raw/plan/tool hashes before/after, OMP-MKL1 and .5s resource monitoring. Negative repeat may be stable at this budget while detectability remains negligible; leave native sampler rare-event numerical reachability and complete-path accumulation unverified. Formal scalars/draws only; no source/default/old evidence edits, remote, PR, merge or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-sampled-depth-plan"></a>

### studies/output_space_cross_task/docs/SAMPLED_DEPTH_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/SAMPLED_DEPTH_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/SAMPLED_DEPTH_PLAN.md)；原SHA256 `58c7651aec38eee35962c5c81078f0bd5663d30fa73846be34be7f4e5bb7784d`。

#### Sampled legal second-depth counterexample check

Frozen after the six-unit H8 report update, before inspecting new selected tensors or depth events,2026-10-15. Existing accepted totals are known; explanatory analysis on a fixed measured batch, not held-out population inference. No new model forwards/draws/window search.

H10 bounded first-slot-only prediction: among all originally captured shared self/graft roots of the six new sampled units, no legal second-slot acceptance flip exists with the same correct first candidate and bit-identical current committed anchor/velocity. A legal event requires both first draft candidates equal the corresponding fresh sampled AR token at root+1, node1 accepted by both actual paths, a non-EOS first target, second target present, and node2 membership differing. Preserve source-root/owner and second candidates, first/second targets, current state/proxy equality/max deltas and exact root-hidden equality. Same first candidate alone without both accepted or remaining second target is not a legal second-depth event. Root/node causal context may differ after earlier path changes; equal current state is not a full canonical or equal-KV factor control.

Score all captured actual root intersections from original six-target rules; retain unvisited/exclusive roots and invalid first-slot/EOS/budget controls, no substitute windows. Evaluate root+1/root+2 in full AR output IDs (includes input prefix), not generated-only offsets. Bit-equal current anchor/v legal events falsify H10; legal events with nonzero state microdifference constrain but do not falsify its equal-state version. If none captured, report not falsified in this finite capture, not a proof of first-slot-only full-path mechanism. Current H8 outcomes must not be used to drop negative units.

Bind diagnosis/tensor SHA, selection/protocol/tool/current source before/after. Pure predicate tests cover node1 prerequisites, second-node flip, EOS/missing target and no-flip control. Scalar/full path observations formal, raw/tensors local; no science/default/old helper/Run edit, remote/PR/merge/delete.

</details>

<a id="legacy-studies-output-space-cross-task-docs-sampling-continuation-plan"></a>

### studies/output_space_cross_task/docs/SAMPLING_CONTINUATION_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/SAMPLING_CONTINUATION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/SAMPLING_CONTINUATION_PLAN.md)；原SHA256 `648c34a0e71e2a00501d2d0c99a254bc30a3c7da78917a7d8985494b7fc90d26`。

#### Additional prompts and seeds, native rare-event detection stress

Frozen after985c7c1,2026-10-15 before new draws or generation. Prior rare candidates were0/4096, not a first rare hit. The earlier source719 owner0 first-slot restoration and the H7 sign-reversal counterexamples are separate evidence.

##### Full sampled paths

Fixed six new units: q242/126, q244/126, q245/126, q245/127, q246/126, q246/127. New prompts are the next two summarization entries in file order (skip4/take2), never selected by model outcomes. Render with the same native tokenizer/chat template (enable_thinking=false); preserve all IDs, no shortening or replacing oversized prompts. Old q242/244 rendered IDs must equal native AR inputs. Bind dataset, tokenizer/model snapshot, input manifest, helpers and old source before/after.

H8 bounded sign persistence: q242/126 graft-self>0 and q244/126 graft-self<0, extending the prior three seeds. Either zero or opposite effect falsifies that bounded prediction. New q245/246 are exploratory replication, all positive/null/negative/infeasible results retained. No population significance or tuning.

Same FP32/eager/CUDA/native runtime, ModelScope weights, T=.8/max100/EOS/static[1,1]/beta.9/beta_prime.99. Seven generations per feasible unit: fresh independent sampled AR, output0/output1, two baseline observations, order1 self/graft. Reuse unchanged choose_window/frozen_targets/observe/transfer. Lock selection from native traces before hidden/logits capture, six roots maximum, single[0,1] once; complete actual paths retained. Require AR equality, native observer/full self stats and semantic trace replay, scope, before-entry identity and cache/source/model/runtime gates. Infeasible stays explicit, never scored as zero. Budget42 generations/4200new, zero teacher/warmup; existing12 units reused separately, not rerun. Test first-delta accounting again without claiming new tensor causal decomposition.

##### Conditional rare-event control and stress

Keep actual old owner1 source717 candidate11/323 and verifier root719 fixed. No transformer forwards, changed logits or artificial candidate embedding. Original T=.8 and1 each16384 fresh native select_token draws with seeds10000..26383, same draw bank for four S/P cells. Add explicitly changed-condition T=2 with131072 seeds10000..141071, not a default or full-path temperature change. Total163840 draws; no adaptive sample increase or optional stopping. Repeated seeds across temperatures are paired, not independent groups. Retain every draw locally, full hash, first-hit seeds, counts, CUDA-FP32 mass and CPU-FP64 reference.

Zero-forward preflight before choosing this stress budget measured T=2 mass11=3.130066761514172e-5 and323=3.7709174648625776e-5; ideal probability of at least one at131072 exceeds.98 for each. H9 bounded stochastic prediction: observe at least one hit for each candidate in this changed distribution. Zero contradicts that prediction at its stated ideal miss risk, not a universal sampler impossibility proof. Require preflight mass consistency and >=.98 ideal detection before sampling. Compute95% exact Clopper-Pearson marginal intervals; compare native masses descriptively, no independent-cell test because banks are shared. Original-condition zero bounds remain insufficient for1e-11..1e-9 mass. T=2 hits cannot establish original T=.8/1 tail reachability or useful complete-path acceptance.

##### Schedule and gates

Serial singleGPU/OMP-MKL1, .5s process RSS/available RAM and globalGPU monitor, no concurrent model/sampler tasks. Previous6new-unit estimate~110seconds and rare~120seconds is planning only. First new prompt calibrates no shortening; stop/preserve OOM, nonfinite, context/source/input/runtime/AR/scope mismatch. Raw weights/cache/tensors/draws stay local.tmp; formal scalar/provenance/semantic comparisons only. New tools and protocol exact-byte Git attributes before execution; no old tool/plan/Run/scientific src/default edits, push, PR, merge or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-sampling-expansion-plan"></a>

### studies/output_space_cross_task/docs/SAMPLING_EXPANSION_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/SAMPLING_EXPANSION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/SAMPLING_EXPANSION_PLAN.md)；原SHA256 `9486949089770147504b702e601855f07a2a80fea7b421dc42b5713761b61556`。

#### Four-prompt three-seed sampled-path expansion

Frozen aftera5ede78 before new forwards,2026-10-15. Keep old q241/T=.8 seeds123,124 as development evidence, never rerun/re-sign them. Add onlyq241 seed125 andq242/q243/q244 seeds123,124,125:10 new units,12 total plotted with old rows distinguished. T=.8/max100/EOS/FP32/eager/static[1,1]/beta/beta_prime/default runtime and exact rendered IDs unchanged. Do not replace difficult/null units or force source719.

Per new unit seven generations: independent sampled AR, native output0/output1, two observations and order1 self/graft. AR output equality, no shortened context, observer/native full stats+semantic equality, original-value placebo and pre-entry/single-vector transfer gates mandatory. Reuse existing choose_window and frozen_targets (entry+first3+last2), selection based only on actual native traces and written before hidden/logit capture. Infeasible units remain explicit, not null effects. Same reference sampler/generator per seed, true sampled prefixes/owners retained; after transfer do not force original paths.

Budget70 generations/7000 generated new tokens including10 AR audits, zero teacher/warmup. Existing q241 rows are separately bound source reports/raw hashes; no pooling old units into a new held-out claim. Report every per-unit accepted/depth/calls/trace delta and sign, same-seed outputs, across-seed AR differences, and whether first local changes cancel/reverse over the complete path. No random-prompt population inference, speed comparison or stable-effect claim from3seeds. AtT>0 first-slot draft remains top1 while verification samples; different sampled windows lack a canonical equal-state decomposition.

ModelScope cached11 files/hash/model/source/baseline files/tool/plan/old helpers before/after; serialGPU/OMP-MKL1/.5s monitor, previous~5.5GBRSS/~7.1GiB globalGPU is context, not this measurement. First fresh unit calibrates memory; stop/preserve source/input/output/numerical/context/OOM failure, never relax AR gate or rerun completed old units. Formal scalar/full semantic traces only; raw/tensors/weights/cache local. No source/default/old contract edits, push, PR, merge or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-sampling-plan"></a>

### studies/output_space_cross_task/docs/SAMPLING_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/SAMPLING_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/SAMPLING_PLAN.md)；原SHA256 `4b371bab1a53cabac17f749e0b1cdf83b07faff703dfe1e85378cd389e8999ca`。

#### q241 nonzero-temperature and conditional-sampling protocol

Frozen after6cca8fc before new generation/draws, 2026-10-15. No scientific/default/config/old helper/Run edits. Two complementary experiments must stay distinct: conditional stochastic acceptance at the old fixed source719 context, and genuinely sampled complete paths with different prefixes/windows.

Full-path q241/T=.8 seeds123,124, max100/EOS same recipe/default runtime. Each unit independently generates AR with the native shared-generator select_token sampler, then native output0/output1 with the same fresh seed; require exact output equality against that AR. Select entry only using unchanged choose_window on the two realized traces, freeze selection JSON before observing hidden/logits. Keep infeasible entries; no searching a gain or forcing719. Capture entry plus first3/last2 selected roots (at most6); re-observe two baselines and run order1 original-value self and order1<-order0 once each, preserving AR and full through-entry semantics. Drafts remain deterministic top1 even atT>0; verification multinomial is stochastic. Budget14 total generations/1400 newtokens (including two AR audits), zero teacher/warmup. All gate failures retained, no relaxed exactness or replacement seed. The two fresh AR paths must be reported different/equal, never asserted beforehand.

Conditional sampling: saved actual verification root720 logits held fixed; old source719 four S/P factor cells supply draft top1 candidates. Temperatures .8 and1, seeds200..263,64 coupled one-token verifier draws per temperature. Use CUDA native select_token(logits,T,fresh Generator) once per seed, same draw under all S/P conditions. Compare candidate==sampled-verifier-token and exact softmax candidate mass, not greedy AR margin or an invented sampled continuation. The same seed/RNG-state bank is reused at owner1; verifier tokens are drawn separately for its different context, then held across its factor cells. No model forward/new generation tokens; include RNG-active/state and draft-vs-verifier distinction in regression tests. It is a conditional Monte Carlo test on an old context, not64 sampled prompts or source719 reached under a new sampled trajectory.

The full-path windows may differ from719 and lack canonical equal-state controls. A nonzero graft effect there only establishes sampled-path occurrence, not pure proxy mediation or a general mechanism. Current prefix/seed/RNG/temperature/output/call counts and all nulls must remain visible. Bind plan/tool/old sources/helpers/native inputs/snapshot/local raw/model before/after, ModelScope cached only. SingleGPU/OMP-MKL1/.5s resources; stop/preserve OOM/nonfinite/hash/input/output/scope/context failures. Formal scalar/traces only, raw/cache/weights local. No push/PR/merge/delete.

</details>

<a id="legacy-studies-output-space-cross-task-docs-state-extension-plan"></a>

### studies/output_space_cross_task/docs/STATE_EXTENSION_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/STATE_EXTENSION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/STATE_EXTENSION_PLAN.md)；原SHA256 `65247ceb024d9199e56d66823976c05f30adba703577fde4f0c1bb4e1a64ef97`。

#### Fixed q242/q243 two-seed state interaction extension

Frozen after4a02551 before new forwards, 2026-10-15. Test whether q241 self-interaction-1 appears universally in a small fixed extension. Select next two original file-order prompts q242 seeds124,127 andq243 seeds125,128, keeping same100-new-token T0/default recipe and rendered inputs; no outcome-based prompt replacement. Seeds are administrative replay probes: T0 select_token uses argmax, output_hidden init/mapping has no RNG draw, so these are not independent stochastic samples. Retain equality or differences rather than count four independent samples.

Per unit observe output0/output1 against original native full AR/trace/stats. Original fixed roots658/600 are retained. Capture at most6 targets chosen before new logits: entry, first3 original selection roots afterentry, last2 original selection roots afterentry; deduplicate and retain missing/new roots explicitly. Baseline controls prove new seeds reproduce the existing greedy paths or stop with a failed quality gate. One teacher on the full original AR prefix, final hidden only, cosine>=.99999 for captured exact roots; canonical states use unchanged chronological CPUFP32 helper.

Per unit reuse observed order1 baseline as native-self, generate native one<-zero plus anchor-only self/graft, velocity-only self/graft and both-synchronized self/graft:7 transfers plus2 baseline generations,8 logical factor cells. Actual KV root synchronization starts only after the fixed entry; other state/proxies at each single-coordinate patch unchanged. No force of native future root/owner path. All outputs must equal original independent AR; original-value self and native-beforeentry semantics mandatory; every captured synchronized coordinate checked with actual tensors and all actual-root scope records checked.

H5: count interaction Y11-Y10-Y01+Y00 per self/graft, retain zero/nonzero/sign and controls. Universal q241-like nonzero interaction across these fixed units is falsified by any complete zero cell; zero is not a global linearity proof. Changes across administrative seeds with otherwise identical greedy inputs require a numerical reproducibility investigation, not a seed benefit claim. H6 count/path equivalence and H3 legal second-depth witness remain descriptive checks, excluding wrong first, EOS, missing next AR target and unaligned source/root.

Budget3600 generated tokens across36 generations plus4 teacher forwards (two747 and two695positions), zero warmup. .5s monitoring, singleGPU/OMP-MKL1, cached ModelScope only. Sources/model/inputs/snapshot/oldhelper/selection/tool/plan before/after and native runtime gates. Stop on resource/OOM/nonfinite/scope/cosine/output/hash drift; preserve every null/counterexample, no relaxing gate or new prompts. Formal scalar/full semantic traces only; raw/cache/weights/complete Run local. No scientific source/default/old contract edits, push, PR or deletion.

</details>

<a id="legacy-studies-output-space-cross-task-docs-state-factor-plan"></a>

### studies/output_space_cross_task/docs/STATE_FACTOR_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_cross_task/docs/STATE_FACTOR_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_cross_task/docs/STATE_FACTOR_PLAN.md)；原SHA256 `4b9a74e2fd5c7fe46e2941b53265f2843abee9bce4d9946d89a5d4e972abf4b6`。

#### q241 state-factor counterexample protocol

Frozen after c4cb011, before new generations. Reuse the already observed q241/seed123/entry716, 100-token AR and canonical teacher states. The selection is explanatory, not held-out confirmation. No new root/prompt/seed/default or native Run; old plans/tools/outputs remain unchanged.

Six generations: native self/graft audit controls, anchor-only self/graft, velocity-only self/graft. Reuse saved native and both-synchronized cells for the full 2x2 anchor/velocity by 2 donor table. At every actual KV root>716 overwrite only the named committed coordinate; assert the other coordinate and incoming proxies remain bit-identical at that patch. Later proxies/state/path may change. Capture the previously frozen targets only; actual new roots are reported as unobserved. Reuse transfer/decoder, native runtime, cached ModelScope/hash binding and teacher; no teacher/auxiliary forward/warmup; budget600 new tokens. Native controls must reproduce all old captured tensors and full semantic traces; AR/full through716 trace equality and cache-root gates mandatory.

H5: count effects of anchor and velocity synchronization are additive separately for self/graft. Compute interaction Y11-Y10-Y01+Y00; nonzero falsifies this local additivity hypothesis. Also retain all traces and same-root margins: equal counts never imply equivalent paths. H6: equal total acceptance implies path equivalence is already contradicted by the two old canonical cells; quantify the gains/losses and first divergence instead of treating another run as held-out validation.

Structured comparison uses chronological prefix divergence, common-root decisions, lost/new roots, true source owner, first/second AR margins and captured state/proxy/seed tensor deltas. Same-position canonical state equality can exclude differing committed anchor/v at that position as necessary for a local self/graft difference, but upstream proxy histories can retain earlier influences. State-factor effects are interventions, not a unique percentage decomposition of native committed microdifferences. Stop on any input/hash/source/output/nonfinite/root/cleanup failure, preserve evidence. Serial GPU, OMP/MKL1 and .5s resource sampling. No push, PR, source/default edits, deletion or weights/raw cache in Git.

</details>

<a id="legacy-studies-output-space-extension-docs-norm-kernel-plan"></a>

### studies/output_space_extension/docs/NORM_KERNEL_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_extension/docs/NORM_KERNEL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/NORM_KERNEL_PLAN.md)；原SHA256 `441c33bf69bba47f46aecbde39027d5ed149418437a2bf783e952f7bdc9489bc`。

#### q178 CUDA norm implementation contrast

Predeclared 2026-10-14 after a50f9e0. This extends OPERATOR_PLAN without editing it. Same source/model/strict runtime/seed126/canonical states/root85 and independent AR112. The current user explicitly requests local kernel controls; no remote action/default change.

Fixed FP32 GPU alternative: sqrt(sum(slots * slots, dim=-1, keepdim=True)), versus native vector_norm. Identical slot values, dtype and scale/multiply/divide; the alternative changes the implementation graph, including intermediate square rounding, so it is not a pure instruction-only kernel experiment. Save both actual profiler launch names and CPU operator external-ID correlation. Reference each norm to CPU FP64 norm rounded toFP32 on held inputs. No guessed kernel names or speed claim.

Eight complete canonical generations: native self/graft controls, entry-only alternative self/graft, post-entry-only alternative self/graft, entry+post alternative self/graft. At entry85 replace only the existing donor/self vector[0,1], computed from that donor's original frozen provider state; other masks remain original. Post-entry variants replace only norm mapping for all subsequent actual provider calls with real incoming proxies and canonical anchor/v. Self changes constrain interpretation. Controls must reproduce prior canonical full paths and selected tensors. All cells require complete independent AR, pre/through85 semantic equality, real KV-root checks and patch cleanup. No forced owner/candidate/path.

Frozen-input operator comparisons use both baseline donor states at85 and both canonical states at86, four inputs; native recomputation must match saved masks. Profile native and alternative on canonical-self root86, once each; save traces locally. Log slots/native norm/alternative norm/reference errors and returned mask deltas. Actual matching inputs can yield unchanged results; do not drop ineffective cells.

Budget8x100=800 new tokens, zero teacher/auxiliary model forwards/warmup, bounded four-state mapping comparisons and two profiles. SerialGPU/OMP-MKL1/.5s sampling; prior~5.2GB RSS/global6.3GiB, stop on OOM/nonfinite/hash/output/scope/root/unsupported-op failure. Bind this plan/newtool/oldhelpers/oldraw/model snapshot/nativeconfig/inputs before/after. Fresh .tmp output; raw tensors/traces/cache/weights never Git.

Decision: stable self/graft paths exclude necessity of these tested norm graph differences at entry85 and/or subsequent calls for this fixed contrast. Changed contrast shows local sensitivity under that stage substitution, not unique attribution to one CUDA instruction. This cannot isolate transformer-generated proxy hidden; fixed q_proj protocol runs separately. Retain every local loss/gain and legal second-slot flag.

</details>

<a id="legacy-studies-output-space-extension-docs-numerical-plan"></a>

### studies/output_space_extension/docs/NUMERICAL_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_extension/docs/NUMERICAL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/NUMERICAL_PLAN.md)；原SHA256 `cb41fb5759365d5f82c45d87c822524777d7d5bf3cbbc308a0f4bde3c3d74c49`。

#### q178 fixed precision and canonical committed-state controls

Predeclared 2026-10-14 before new controls, version q178-numerical-history-20261014-v1. Original q178 seed126 FP32/eager/T0/static[1,1]/k2/BC9/100tokensEOS, fixedroot85 owner0 second-slot graft, original independentAR112 total positions/36new tokens. No scientific src/default/nativeRun changes. Previous single-slot tool/observer/compare helpers remain byte-identical.

##### Questions and fixed cells

Q1: do strict FP32 deterministic repeats give the same masks/logits/state/acceptance, and do the tiny same-position differences from altered block/owner paths persist? Q2: if later committed anchor/velocity are synchronized to one canonical AR-prefix reference while actual proxy/seed/owner history is retained, does the 8→9 effect persist? Candidate rank is the immediate T0 acceptance rule; history is a possible upstream cause, not an alternative rule. Do not force an owner, candidate or old trajectory.

Strict process-only settings before CUDA initialization: CUBLAS_WORKSPACE_CONFIG=:4096:8, FP32/eager, TF32 matmul/cudnn off, float32 matmul precisionhighest, deterministic algorithmsenabled with warn_onlyfalse, cudnn deterministictrue/benchmarkfalse. Record requested/effective settings, do not edit nativeconfig/defaults. First two strict original output0/1 observations must reproduce old full AR and semantic trace; capture union of original roots>=85, by visited subset. Use their actual root85 donor tensors (not assuming old floats equal across runtime modes).

Then fixed strict cells: order1<-order0 twice, order0<-order1 twice, order1 original-value placebo once. Each must match independent AR and pre/through85 semantic prefix; repeats compare actual trace and all captured tensor values, not only pooledaccepted. Compare saved old-path state deltas separately; no expectation that determinism makes differently packed floating computations bit-equal.

One additional strict full112-position teacher-forced AR forward, final normalized hidden only, no generation. Verify teacher exact-root hidden versus observed baselines at all captured positions (cosine>=.99999). Reuse native OutputHiddenMaskProvider.observe_committed over those chronological exact hiddens to form per-order canonical anchor/velocity for each prefix lengthp (hidden throughp-1). Teacher packing differs from native blocks: this is an explicit numerical synchronization intervention, not a claim of exact math or a production recipe.

Four canonical cells: output0 self-value/no-graft control, output1 self-value/no-graft control, order1<-order0 and order0<-order1. Keeproot85 original state/masks and graft definitions; for every actual provider call root>85 overwrite only committed anchor/velocity with the corresponding canonical same-AR-prefix state, leave incoming proxies unchanged. Read actual currentroot from the real mutable KV cache length after native pruning, check again at modelprehook; never infer post-control roots from a baseline trace. Clone tensors and remove patch/hooks even on failure. Canonical self-value cells may have different later semantics; retain changes rather than require a false placebo equality. Compare graft versus its canonical same-order self control, not just original baseline, at true matchedprefixes. If a new root lacks an original capturedstate, canonical state still exists from the complete112-position teacher; no invented old-state evidence.

Budget11 decoder generations x100=1100new tokens (2 strict baselines+4 repeats+1 strictplacebo+4canonical), one112-position teacherforward, no other auxiliary query or warmup. Source670c744e0535dcad, model/snapshot11-file hashes, old native/raw/transfer tensors and this PLAN/new tool/helpers bind before/after. New .tmp/q178-numerical-20261014-v1, never overwrite prior evidence. SerialGPU/OMP-MKL1; prior same-model RSS~6.6GB/globalGPU~6.3GiB capacity, sample.5s/recordrealcalls. Stop/preserve if unsupported deterministic kernel, hash/output/teacher/scope/actualroot/nonfinite/OOM failure; no precision/contract change to rescue a result. Complete AR equality and actual source-owner/trace evidence required. No pushes/merges/deletion/weightcacheexport.

##### Interpretation boundaries

Identical repeats plus persistent path-dependent microdeltas supports floating differences from changed packing/commit paths rather than random noise; it does not identify every low-level CUDA kernel. Canonical synchronization preserving gains would rule out those committed-state microdeltas as necessary for the effect in these fixed cells, while still not isolating numeric differences carried inside proxy hiddens. Canonical controls changing acceptance must be reported and constrain causal claims. A vanished gain bounds robustness, not the entire method. Ranking versus history are separated by immediate seed/margin/accepted depth and later actual-owner propagation, with no global EMA claim.

All cells/results and limitations go into STUDY_REPORT; PLAN's predeclaration remains immutable and results are saved separately. New prompt batch executes separately with original runtime defaults; no concurrent GPU work.

</details>

<a id="legacy-studies-output-space-extension-docs-numerical-plan-review"></a>

### studies/output_space_extension/docs/NUMERICAL_PLAN_REVIEW.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_extension/docs/NUMERICAL_PLAN_REVIEW.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/NUMERICAL_PLAN_REVIEW.md)；原SHA256 `7c84e30a2af76019581fe1f3657bd74c2fc52767d805e1ffed6ace678b3e0060`。

#### q178 numerical PLAN review

Results review2026-10-14; original NUMERICAL_PLAN.md and native PLAN.md are retained byte-identical, not re-signed. Evidence: numerical_diagnosis_v1.json and numerical_analysis_v1.json.

Q1 locally supported: strict FP32 repeats produce bit-identical selected logits/hiddens/masks/state and full semantic paths, while altered-versus-baseline path microdeltas persist. First observed common divergence is root89: anchor for exact position88 comes from accepted row2 of block86 in the original path, versus row0 of block88 in the altered path (the latter verified by tensor). This locates a deterministic block/KV/owner packing difference; no individual CUDA kernel is identified.

Q2 locally supported: canonical self/no-graft controls remain8/27; canonical grafts remain9/26 and match strict graft full semantic paths. At same actual AR-prefix root>85, canonical anchor/v are bit-equal across same-order control and graft; incoming proxies are left intact. Thus those committed-state microdeltas are not necessary for this effect under the declared canonical intervention. Teacher/canonical packing itself differs; self controls constrain interpretation, and numeric differences carried in proxy hiddens remain unisolated.

Immediate root86 second-slot rank crossing and subsequent owner1/2 history are different stages, not mutually exclusive causes. No forced-owner or fixed-candidate counterfactual was run. A future causal refinement could hold a specified proxy/owner-output vector under canonical states at one declared root, with self-value and AR gates; this is a proposal, not an executed result or changed default. Current newprompt generality evidence belongs to output_space_followup. No need to change precision/default/source from these diagnostic outcomes.

</details>

<a id="legacy-studies-output-space-extension-docs-operator-plan"></a>

### studies/output_space_extension/docs/OPERATOR_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_extension/docs/OPERATOR_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/OPERATOR_PLAN.md)；原SHA256 `add72cf85dfc1a6cd4b9369fb3aa9b8258b549ad7a01f16ace80008a8a77e60f`。

#### q178 provider rounding and norm-backend isolation

Predeclared 2026-10-14 after f3b95ea, before this diagnostic. Existing NUMERICAL_PLAN and successful native contracts stay byte-identical. This is a diagnostic extension of q178, not a new default or a new scientific implementation.

##### Fixed interventions

Restrict to order1 at the existing root85, seed126, unchanged independent AR112 positions/36 new tokens. Reuse canonical committed states and original root85 self/donor vectors from the hash-bound previous strict evidence. Do not regenerate a teacher, change model precision, force candidates/owners or search roots. At root85 and earlier use the native provider byte-for-byte; only subsequent actual provider calls can change. Every call still receives its actual incoming proxy tensors. The canonical wrapper reads root from the real KV cache and replaces anchor/v at every actual root>85.

Eight generations: native self and one<-zero, then each of three operator variants with self and one<-zero. S64 computes simulated EMA/slots from the exact same FP32 inputs in CPU FP64, casts completed slots to FP32, and retains native GPU norm/scale/multiply/divide. N64 retains native GPU FP32 slots and replaces only the norm scalar with CPU FP64 vector_norm rounded to FP32. N32 retains those slots and substitutes CPU FP32 vector_norm, leaving scale/multiply/divide on GPU. Thus S64 bounds provider arithmetic rounding; N32 holds input dtype fixed while switching norm backend; N64 changes reduction precision. They do not isolate hidden-rounding inside transformer proxies or distinguish every instruction inside a kernel. No combined S64+N64 cell; interaction remains unchecked.

The repeated native self/graft must match previous canonical full AR/semantic trace and selected masks/state/hidden/logits at frozen roots. All variant cells must equal independent AR, preserve semantics through85 and preserve provider state/incoming proxies during the mapping itself. Later proxies/paths may change and must be retained. Self controls may change and must constrain attribution. At roots86/89/108 of the previously saved canonical self/graft, replay all mappings on identical saved provider inputs and report norm/output error versus native. Exact native-formula replication is a hard gate.

Profile exactly one native mapping of canonical-one-self root86 (2x1024 norm per owner) using installed torch.profiler CPU+CUDA. Save Chrome trace locally and extract concrete kernel names correlated with norm CPU operator via external IDs. If CUPTI/kernel events are unavailable, preserve that fact and do not infer names from CPU timing. Profiling is location evidence; N32/N64 are operator interventions. A norm kernel name plus changed margin is not proof that its internal reduction tree uniquely caused the old transformer hidden microdelta.

Budget8 generations x100=800 tokens, no teacher/auxiliary transformer forwards/warmup; six saved-state mapping inputs x4 modes plus one profiled native mapping. Single CUDA process/OMP-MKL1, strict settings identical to prior numerical tool, snapshot/source/model/old tensor/old report/native config/inputs/PLAN/tool/helpers bound before/after. Prior RSS~4.9GB/global GPU~5.7GiB plus244.8MB CPU evidence; sample.5s. Stop and preserve on mismatch/nonfinite/hash/root/OOM/unsupported deterministic op; profiler unavailability alone records a bounded gap. New output directory only. Full traces/scalars formal, raw tensor and profiler trace remain .tmp. No scientific src/default edits, push, merge, deletion or cache/weight export.

##### Decision rules

Compare graft versus its same-mode self, not pooled against a single native baseline. If S64 alone removes/restores the graft difference while self stays fixed, provider slot rounding is necessary for that fixed contrast under the intervention. If N32 alone changes it, norm backend implementation can matter despite identical input dtype; if N64 alone changes it, sensitivity to norm rounding is demonstrated. Stable paths across all three constrain necessity of these stages for this fixed q178 effect, not all numeric sensitivity or all prompts. Full trace/margin differences still matter when accepted counts stay equal. Next hidden-rounding/backend experiment must capture identical Q/K/V or norm inputs inside the model and intervene on one operation, with self/AR gates; current provider experiment cannot make that claim.

</details>

<a id="legacy-studies-output-space-extension-docs-operator-plan-review"></a>

### studies/output_space_extension/docs/OPERATOR_PLAN_REVIEW.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_extension/docs/OPERATOR_PLAN_REVIEW.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/OPERATOR_PLAN_REVIEW.md)；原SHA256 `7b6762b4812736d140f1c56591cf8fcb2a7f77d59511ff0ce617a93d41764e29`。

#### q178 provider operator result review

2026-10-14. OPERATOR_PLAN.md and the archived measured tool source remain byte-identical. CLI usage-only docstring repair is checked by execution-AST equality; the original report hash is not re-signed. Eight outputs match independent AR; native self/graft repeat previous canonical selected tensors and full semantic traces exactly. Each S64, N64 and N32 self remains8/27; each graft remains9/26, with full paths equal to its corresponding native case. Provider differences up to5.960464477539063e-8 and small downstream seed/logit changes exist, so these are effective interventions, not identical implementations.

Profiler records84 CUDA launches and3 aten::linalg_vector_norm CPU calls, correlated via external IDs to the actual NormTwoOps reduce_kernel symbol recorded in operator_diagnosis_v1.json. That identifies the provider norm launch. The CPU-FP32 backend substitution and CPU-FP64-rounded norm substitution bound its necessity for this fixed contrast; they do not identify the internal reduction instruction or isolate the transformer kernels generating proxy hidden.

The post-root85 provider-rounding necessity explanation is narrowed by stable paths; root85 entry mapping was left native and remains an untested stage, while transformer hidden rounding and operator interactions remain open. Only order1 self/one<-zero was tested; reverse/order0 and combined stage replacements were not added. KERNEL_PLAN in the new cross-task draft gives a fixed-row linear packing/high-precision protocol, explicitly unexecuted. New Study is draft only. No default or scientific source change follows.

</details>

<a id="legacy-studies-output-space-extension-docs-phi-rms-plan"></a>

### studies/output_space_extension/docs/PHI_RMS_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_extension/docs/PHI_RMS_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/PHI_RMS_PLAN.md)；原SHA256 `b5231de753f37521e59590d3279ece1182a785757f8252ca28d6b9a096dc1879`。

#### q178 entry Phi and fixed post-attention RMSNorm controls

Predeclared 2026-10-14 after4c9ad8e, while cross-task native GPU queue runs. Execute only after that queue and its transfer diagnostics exit. Same strict q178 root85/seed126/ModelScope/source/AR112/canonical states. No root/layer/row search or new default.

Eight generations: native self/graft controls, Phi64 self/graft, RMSNorm singleton-FP32 self/graft, RMSNorm reference64 self/graft. Controls reproduce old full canonical semantic trace/selected tensors. All variants preserve independent AR and through85 semantics; actual future owners/paths are not forced. Budget800 tokens, no teacher/auxiliary model forwards/warmup, four frozen RMSNorm direct comparisons plus paired native/singleton CUDA profiles.

Phi64: compute the donor's native FP32 simulated owner0 slot1 exactly (EMA/slot construction not changed), hold the native FP32 embedding-norm scale, compute full norm+scale multiplication+division in CPU FP64, cast completed vector once toFP32. Replace only root85[0,1], other mask vectors and subsequent provider mapping remain native; canonical anchor/v after85 retained. Report vector bit-equality/max/L2 first; an unchanged vector is numerically ineffective and cannot establish rounding robustness. Do not recompute scale or add arbitrary noise.

RMSNorm: fixed model.layers[0].post_attention_layernorm, source85 row4 only. This is a different operator stage from q_proj. Native capture retains same complete input/output plus shape/stride. Singleton mode directly calls module.forward on that row to avoid hook recursion; reference64 uses exact FP32 input/weight and FP32-represented epsilon, CPU FP64 square/mean/rsqrt/product, then casts once back. All other output rows remain bit-identical. Record nonzero/equal substitution, final masked hidden/proxy/seed changes and full actual paths. Direct comparisons use each native self/graft frozen input, not their semantically different inputs subtracted as rounding error. Bind installed RMSNorm source file in addition to runtime version, tool/common/plan/oldraw/source/model/11file snapshot/native config/inputs.

SingleGPU/OMP-MKL1, .5s resources. Capacity from q178 prior~5.4GB RSS/~5.8GiB globalGPU is context, not new measurement. Stop/preserve OOM/nonfinite/hash/scope/root/output/unsupported-kernel failure; no gain tuning. Raw tensors/Chrome traces/weights/cache/fullRuns local only; scalar and semantic evidence formal. No source/default edits, remote push, PR or deletion.

Interpretation: stable trajectories under effective Phi mapping or RMSNorm substitutions bound necessity of these numerical realizations for this fixed contrast, not necessity of the operator itself or all layers/contexts. Ineffective singleton or Phi cells remain explicit equivalence controls; do not pool them with effective numerical interventions. Cross-task H1/H3 results are separate, H2 still requires a later frozen canonical protocol.

</details>

<a id="legacy-studies-output-space-extension-docs-plan"></a>

### studies/output_space_extension/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_extension/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/PLAN.md)；原SHA256 `61743ea13904291ceb770172a23a7a51af96c88133e68a94584e184ce06a0bab`。

#### Four-prompt extension and fixed single-slot intervention plan

Predeclared 2026-10-14 on parent7a02501. Version output-space-q175-178-20261014-v1. Same scientific source670c744e0535dcad, model/snapshot/default gains/decoder/sampler/cache. Existing native Study machinery and observe()/transfer()/compare_window() are reused. No code in scientific src is changed.

##### Existing q173 evidence and hypotheses

Evidence sources: output_space_small_matrix/docs/trajectory_diagnosis.json, window_diagnosis_v1.json, transfer_diagnosis_v1.json, cross_prompt_diagnosis_v1.json and cross_prompt_path_v1.json. These are development observations, not independent confirmation. q173's 29 common output0/1 roots differ in acceptance only at94/97; q171's36, q172's12 and q174's11 have no output-order acceptance difference. Full q173 mask/logit measurements exist only at the prior five fixed transfer roots, not every earlier root.

| qid | intervention root | root minus input length | first common hit | output0/1 accepted | transplant vector L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 171 | 63 | 0 | 64 | 10/10 | .09785780 |
| 172 | 42 | 3 | 43 | 1/1 | .05879461 |
| 173 | 83 | 14 | 84 | 1/3 | .05366028 |
| 174 | 41 | 9 | 42 (terminal) | 1/1 | .06347270 |

Q173 did not have the largest injected vector difference, and late window position alone is insufficient (q174 terminates; q171 has more remaining roots but no hit transfer). At q17394/97, baseline order0/1 target margins have opposite signs, deltas about+4.00275/+5.11003. The order1<-order0 transplant removes both hits with same committed anchor/v at five measured roots; reverse order0<-order1 restores94 but not97. Q171/172 transfers change margins by at most.42729/.59764 without sign crossing, and all semantic trajectories remain equal. Q174 has no continuing owner1 propagation after EOS.

H1 (rank-boundary hypothesis): historical proxy perturbation can alter candidate directions without acceptance change; q173's local gain requires specific candidate rankings to cross the correct-target top1 boundary. Mere nonzero velocity, displacement amplitude or propagated seed difference is insufficient. In new prompts report natural baseline margin sign opposition at all common visited roots, and actual matched acceptance/sign changes after fixed transfers. Natural acceptance differences with no transfer would bound this single-entry sufficiency, not invalidate the T0 ranking identity.

H2 (owner-path history hypothesis): the first common accepted-owner transition allows the preceding second slot to influence later first-seed proxies. A donor-like first seed may persist along owner0-only blocks, while later accepted-owner transitions can mix it back into recipient-specific history/velocity; this could explain q173 reverse transfer restoring94 but not97. Measure actual source owners, donor-first-seed equality/L2 at common roots, and alignment of equality loss with subsequent accepted roots. This is observational mediation after one intervention, not an independent test of every intermediate slot; retain exceptions.

H3 (terminal/unused-owner boundary): if no continuing accepted-owner path uses the injected slot, a null acceptance result is expected and is not evidence against all historical effects. Explicitly classify terminal/no-shared-hit/unused-owner controls instead of combining them with continuing propagation cases.

These hypotheses were written before observing new q175–q178 outputs. Do not retune or pick another window after seeing new margins.

##### Fixed new inputs and native matrix

Select contiguous next translation records, file order skip14/take4, q175/176/177/178. Freeze text, datasetSHA and independent tokenizer-rendered input IDs in prompts_manifest.json before make. Same locally prepared ModelScope Qwen3-0.6B 11-file snapshot, tied weights, FP32/eager/CUDA, nonthinking, T0/EOS, max_new_tokens100, unpruned static[1,1]/k2/BC9, beta.9/beta_prime.99. Seed123 with per-prompt generators123/124/125/126; these are four prompts, not stochastic seed replications.

Four native Experiments/Runs (input frozen mean, output0, output1, independent AR), each containing four prompts, new version/IDs. Run make/launch/process/report, round1 single GPU. Gate exact_match1, no context shortening, input/model/source/factor/provenance/raw/Study quality all pass. No warmup changes. At most1600 new tokens in native runs plus1200 internal AR-audit tokens and at most72 unchanged warmup tokens. Do not rerun already successful Run IDs or replace a null prompt.

##### Predeclared transfer selection and diagnostics

For each prompt using native output0/1 semantic traces only: choose the earliest common acceptance root whose immediately preceding common visited root is rejected by both; intervene at that preceding root. If none, choose the earliest common root rejected by both that has a later root. If still none, choose the earliest common nonterminal root and label unused-owner control. If no continuing common root exists, report no feasible slot-use window rather than replace the prompt. This is an explicitly planned fallback, not hidden selection by logits.

Freeze the resulting root/selection reason and complete union of all baseline roots at/after it in window_selection_v1.json before any hidden/margin observation. Each original-order observation captures only targets it actually visits; changed paths in transfers retain unvisited/new roots. No forced baseline owner or extra window search. Intervene only once at owner0 second slot[0,1]. Cross-order first-slot equality is measured, not a universal requirement; recipient first seed and all unselected tensor coordinates must remain unchanged at the intervention.

Two unchanged baseline observations plus order1<-order0, order0<-order1 and order1 original-value placebo for each feasible prompt, using frozen donor tensors. Budget at most20 diagnostic generations x100=2000 new tokens, no teacher/isolated-prefix/warmup forwards. Existing observe/transfer restore hooks/provider on failure and require unchanged baseline/placebo semantic paths, correct actual seed owner, only selected coordinate replaced, unchanged pre/through-intervention semantic trace and complete independent-AR tokens. Post-intervention semantic changes are retained, not rejected merely because a hit appears/disappears. Record per-root margin/state and all losses/gains, including changed/noncommon paths. Compare anchor/v with recipient at all matched roots, report bit/L2 differences rather than assume equality; zero-order velocity remains0.

Bind source/model/snapshot/runtime, native inputs/config/raw, original execution JSON, this PLAN, frozen selection, runner and unchanged helpers before/after. Stop on source drift, hash/token/scope/ownership/nonfinite/OOM/gate failure and preserve it, without swapping prompts, re-signing old outputs or tuning defaults. New .tmp/extension-native-20261014 and .tmp/extension-transfer-20261014-v1 hold local logs/tensors. Formal scalar/full-trace JSON and reports live here; do not export weights/raw tensors.

##### Capacity and completion

Same-model prior sampled RSS~6.1GB/device-global~5.8GiB guides serial capacity. OMP/MKL1, single GPU, no competing launch chain. Sample process-tree RSS/threads/IO, device-global memory and available RAM at.5s. Record actual wall times; heuristic make est is separate and observer synchronization precludes performance ranking. No full matrix or concurrency experiment.

Complete when four native Runs and available fixed transfers pass correctness/scope/provenance, all four prompt outcomes including unfeasible/negative cases and hypothesis boundaries are in STUDY_REPORT with a small combined matrix, and appropriate tests/static/docs checks/local commit are done. No pushes/merges/deletions. These eight accumulated development prompts (q171–q178) remain a small nonrandom sample, not significance or universal EMA attribution.

</details>

<a id="legacy-studies-output-space-extension-docs-rounding-factor-plan"></a>

### studies/output_space_extension/docs/ROUNDING_FACTOR_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_extension/docs/ROUNDING_FACTOR_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_extension/docs/ROUNDING_FACTOR_PLAN.md)；原SHA256 `9124baecac7c7b3dfa1f044b9d4263335ecd47c730b4dceb596a95c1b96d12aa`。

#### q178 operator arithmetic factor protocol

Frozen after c4cb011 before new direct arithmetic comparisons. No model generation, prompt/layer/row search or default change. Reuse q178 entry85 FP32 donor slots, native scale and native self/graft layer0 post_attention_layernorm row4 captures. Bind old tensor/report, installed normalization implementation, tools/plan, source and cached weight bytes; read only the needed gamma from safetensors, no model loading.

Phi: 2x2x2 factors (norm32 native GPU versus CPU64 exact-denominator, scale*slot32 versus64, division32 versus64), cast final vector once to32. Native000 must reproduce stored entry exactly, all high111 is compared against the stored CPU64 mapping, including any CPU/GPU backend disagreement. Hold FP32 simulated slots and FP32 scale; no reinterpretation as full EMA precision. Retain all eight effects, non-additive interactions and native/all64 endpoint matching, not an automatic sole-cause label.

RMSNorm: all32 native square, mean-plus-epsilon, rsqrt, input*inverse and gamma product must reproduce saved packed output row4 exactly. Enumerate the 32 mixed precision factor cells on the same native self/graft input, holding the FP32-represented epsilon and FP32 gamma; low stages round input/output to32, high stages to64. Compare isolated factors and full64 against the stored CPU64 intervention; retain all changed-coordinate counts/max/L2, intermediate scalar values and backend disagreement. Packed versus singleton all32 differences are an explicit control. Arithmetic factors change precision and implementation; profiler documents executed kernels but cannot identify a hardware instruction as the unique cause. No downstream generation effect claimed beyond the prior eight-cell evidence.

No model/teacher/auxiliary forward or new tokens. CPU/GPU direct arithmetic with OMP/MKL1, saved profiles local only. Reject nonfinite/input mutation/shape/native mismatch/hash/source failure; save a new version and never overwrite old evidence. Formal scalar findings plus small tests; tensors/model/cache/Chrome traces remain local. No remote/PR/deletion.

</details>

<a id="legacy-studies-output-space-followup-docs-plan"></a>

### studies/output_space_followup/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_followup/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_followup/docs/PLAN.md)；原SHA256 `e407142572ae457e725c294a9b52c102e1e36549ead24c7bb17dceb84d422ded`。

#### q179–q182 small follow-up plan

Predeclared 2026-10-14 on a42313d before new outputs. Version output-space-q179-182-20261014-v1. Hypotheses and existing facts from output_space_extension/docs/PLAN.md and STUDY_REPORT.md are retained: q178 has both single-slot transfers accepted8→9 with mixed losses/gains; root86 valid second slot flips under bit-identical committed state, later state changes by anchor~3.05e-5/velocity~2.38e-6. Ranking is the immediate acceptance rule; owner history can change future seeds, so they are different causal stages rather than exclusive explanations. Separate q178 numerical controls test sensitivity without altering default recipes.

Select next four translation records, file order skip18/take4, q179/180/181/182. Freeze datasetSHA, text and independent renderedIDs before make. Same ModelScope Qwen3-0.6B local 11-file snapshot, actual tied weights, FP32/eager/CUDA, nonthinking, T0/EOS/max100, static unpruned[1,1]/k2/BC9, beta.9/beta_prime.99. Source670c744e0535dcad; seed123 and per-prompt generators123/124/125/126. Original runtime deterministicfalse and defaults stay unchanged for native Runs.

Four new native Runs: matched input mean/output0/output1 and independent AR, each four prompts. New version/IDs, native make/launch/process/report and complete source/input/model/raw/factor/Study/AR gates. No context shortening, no rerunning old successful IDs or replacement prompt after a null. At most1600 native new tokens+1200 internal AR audit tokens+72 unchanged warmup tokens. Single GPU round1/OMP-MKL1, no concurrent launch chains. Prior sampled RSS~6.6GB/GPU-global~6.3GiB guides capacity; sample resources at.5s, not allocator/speed rankings.

Transfer selection exactly follows prior choose_window(): earliest eligible common acceptance with a preceding shared rejection; otherwise earliest continuing common rejection; otherwise continuing common unused-owner control; no continuing common root means explicit infeasible, prompt retained. Freeze window_selection_v1.json and runner/PLAN/execution hashes before any hidden/logits observation. Complete baseline root union from that position; each baseline captures its visited subset. Reuse existing observe(), transfer(), compare_window() and choose_window() without editing measured helpers. No forced old owner, root search or gain tuning.

Two original baseline observations and both one-slot directions plus original-value order1 placebo per feasible prompt, max20 generations x100=2000 diagnostic tokens, no auxiliary model forwards. Only owner0 second slot[0,1] once; unchanged remaining coordinates and pre/through-intervention semantics, true seed ownership and independent full AR output required. Original baselines/placebo full semantic equality required. Post-transfer path/state differences and adverse hits preserved. Measure valid first/second slot margins with second AR path flag, same-position recipient anchor/v bit equality and L2, donor-seed comparisons and actual/noncommon roots. Unvisited/new roots are explicit; no fabricated hidden for new roots outside frozen targets.

Hypothesis scope: test whether any new prompt has natural output-order differences or transplant effects beyond q173/q178; preserve all4 and local losses, not only pooledaccepted. Ranking threshold crossing without shared first-token ancestor is invalid for depth2. Nulls cannot exclude other windows; positive effects cannot establish a global method advantage or pure proxy causation when state changes. Four more prompts make12 accumulated development units, not random-seed repetition, significance or cross-task generalization.

Stop/preserve on source/hash/output/ownership/scope/nonfinite/OOM or native-gate failure; do not relax a gate or silently re-sign. New .tmp/followup-native-20261014 and .tmp/followup-transfer-20261014-v1 hold raw/local artifacts; formal scalar/full traces in this Study. Bind source670/model/runtime/snapshot11 files/nativeconfig/inputs/raw/PLAN/selection/tool/helpers before/after. No scientific source/default edits, pushes/merges/deletions; full weights/cache/tensors not in Git. Completion: native and feasible diagnostics checked, all results/limits in PLAN's retained contract plus STUDY_REPORT, appropriate tests/static/docs, local commit.

</details>

<a id="legacy-studies-output-space-followup-docs-plan-review"></a>

### studies/output_space_followup/docs/PLAN_REVIEW.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_followup/docs/PLAN_REVIEW.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_followup/docs/PLAN_REVIEW.md)；原SHA256 `40f4492c630bf74aee3ac8b38d40384ebeb9ee6105edd1837fac8931990d2edc`。

#### q179–q182 PLAN result review

Results review2026-10-14. PLAN.md remains the predeclared frozen execution contract; this file records resulting decisions without re-signing successful Runs.

Generality observation expanded: q179's natural output1 accepted9 versusoutput0=8, whileq182 has3 versus4. Both directions onq179 give10; q182 output1<-output0 restores a second-depth hit3->4, reverse remains4. Q180/181 have changedseed but no accepted change. Preserve all gains and losses, no stable order1 advantage or new default inferred.

Ranking hypothesis requires legal second-slot paths, not only first-slot margins: q182root47 original second margin+.481377/-.039644, one<-zero+.334106 under same anchor/v at that root. This is a later owner-history example unlike q178's immediate root86 directsecond-slot graft. Q179 has multiple first-slot oppositions and mixed transfer hits/losses, with post-path microstate differences; no clean all-history mediation proof.

The separate q178 canonical/strict controls show committed-state microdeltas need not produce or preserve its9-count outcome, but no equivalent synchronization control was run onq179/q182. Scope12 existing translation prompts, one fixed window perunit/T0/same model; no statistical or cross-task claim. Proposals for later study: predeclare another-task batch, or one proxy-specific canonical-state intervention. They are not executed or a change to default recipe. Current boundedbatch is complete; no furtherroot/scale search was appended.

</details>

<a id="legacy-studies-output-space-prerequisite-docs-plan"></a>

### studies/output_space_prerequisite/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_prerequisite/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_prerequisite/docs/PLAN.md)；原SHA256 `a37509bc64abf3fe5cbc952b0db16f40e06f2000fcfd9e1586ac598e857b55fe`。

#### Output-space prerequisite Study Plan

##### Question and scope

Extracted from research-plan-v6 sections 5.1–5.8/5.11 and E2/E3: can normalized final-hidden masks, conditioned on each draft prefix, use the existing simultaneous verifier without changing the greedy AR tokens? Compare output order0/order1 against frozen_mean and independent AR under the same two-slot chain. This is the first output-space numerical prerequisite, not a Kalman/full-paper or superiority test. Parameters are predeclared, never chosen by acceptance.

##### Fixed protocol

Qwen/Qwen3-0.6B from the already downloaded ModelScope snapshot (same file hashes as input_mask_baseline), local_files_only=true; tied weights required and verified, no untied fallback. SpecBench q171 translation skip10/take1, first turn, nonthinking template; FP32/eager/CUDA, seed123, T0, budget100, stop_on_eos=true. Four cases: frozen_mean chain, output_order0 chain, output_order1 chain, independent AR; each n=1. New version output-space-q171-20261014-v1. Tree static [1,1], k=2, pruning=false, actual BC=9; this differs from the earlier BC30 single-slot tree, so those older accepted counts are not a matched comparison. Reuse q171 as the predeclared development prompt, not a new held-out accuracy sample.

Output state FP32: order0 anchor; order1 anchor plus velocity. Exact observation y: anchor=y, velocity=beta*v+(1-beta)*(y-old_anchor), beta=.9. Node proxy observation uses beta_prime=.99 (conservative fixed choice, not estimated), chain-end has no proxy and only predicts h+v. Each owner starts from a private simulation state; slot j uses anchor+(j-1)*v, gamma=rho=1. Prompt warms exact state in position order. Prefill still uses mean-input masks, produces root token and k normalized final-mask hiddens; no extra target forward. The exact state ends at last prompt hidden, not the generated root: root hidden is unavailable until next block. After verification, observe root and accepted nodes in order; bonus hidden is observed next round. Proxy source is the preceding accepted owner's mask rows. All owner masks are fixed before the forward. Mapping Phi(h)=mean row norm of input embedding table * h/||h||; zero/nonfinite norms fail. Model weights never update.

##### Implementation mapping and evidence

Existing decoder/provider loop remains for old methods. Output provider in structure/algorithm handles state and per-owner content only; HFCausalLM exposes final-normalized hidden from a temporary final-norm hook with try/finally removal (no all-layer storage). Decoder requests it only for the new method; attention/position/tree/cache algorithms remain unchanged. Config API supports explicit spec, no new task/runner. Collect/summarize/write/process reuse current raw correctness/source/evidence gates. Tests verify hook equality/logits, cleanup, state math, owner separation, rejection/acceptance/KV, T0 and shared-seed sampled equality, unsupported untied/nonchain rejection. Existing36-cell strict baseline and full regression must remain equal.

Native make/launch/process/report, round1 single GPU; no concurrent experiment. Prior load/serial resource calibration establishes capacity for same model; sample the new decoder RSS/thread/I/O and device memory, do not claim exact allocator peak from sampling or speed ranking. Model/input/PLAN/source manifests freeze before make. New source never signs an old succeeded Run; historical input baseline remains immutable evidence and is not rerun. Any token mismatch, source/input drift, missing observation, OOM or gate failure stops expansion; preserve failures, change version for a corrected execution.

##### Success and interpretation

Execution: all four new Run statuses succeeded. Quality: exact_match=1 for all three speculative cases and independent full AR IDs equal, Run+Study source/evidence/complete gates pass, final hidden hook test proven, state/owner/observed position trace valid. Record accepted/depth/calls/new/committed separately, no speed pass threshold. n=1 cannot show mechanism advantage or support H1–H13. No full matrix, untied mapping, general trees, order>=2, adaptive Kalman, tuning, concurrent/recovery validation, merge or release.

##### Sources and reading

Local research v6 method/implementation sections read in full for this stage; fixed RPipe7a15c46 research.md read. Knowledge09078015 four research/PyTorch skill references and guides were read in prior stage; no installation/dependency added. Qwen downloaded config declares tie_word_embeddings=true; actual tensor storage equality is required. Local Transformers5.12.1 Qwen3Model/LlamaModel.forward norm->last_hidden_state inspected; validate against output_hidden_states[-1] on tiny. Official references: https://huggingface.co/Qwen/Qwen3-0.6B and https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook . No novelty claim or new literature review asserted.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-cross-prompt-plan"></a>

### studies/output_space_small_matrix/docs/CROSS_PROMPT_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_small_matrix/docs/CROSS_PROMPT_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/CROSS_PROMPT_PLAN.md)；原SHA256 `24aaaa5aa2157bfe0c256af1c7a176946c9c246364976dbaec93a124c54d3b7d`。

#### Fixed cross-prompt single-slot replication plan

Predeclared 2026-10-14 before new hidden/logit observation or transfers. Version output-space-cross-q171-172-174-20261014-v1. Reuse observe() from diagnose_output_windows.py and transfer() from diagnose_proxy_transfer.py without editing them, scientific src, defaults or original Runs. New runner only selects saved native inputs and calls those functions; no decoder/sampler/cache/scheduler implementation.

##### Selection and falsification

Use all three other available translation prompts: q171/seed123 from output_space_prerequisite; q172/seed123 and q174/seed125 from output_space_small_matrix. The original output0/1 trajectories match in each prompt, although input-chain acceptance differs. These are prior failures of an order1 acceptance advantage, not new held-out population samples or stochastic repeats (T0).

Selection rule, using only retained semantic traces: find the first root where both output orders accept a draft; choose its immediately preceding common visited root, where both reject all drafts. The fixed intervention roots are q171:63 (first common hit64), q172:42 (hit43), q174:41 (hit42). Replace owner0 second-slot vector [0,1] once with the saved opposite-order vector from a new unchanged baseline observation. Keep first slot and every other coordinate/state untouched. Do not select after looking at direction/logits or add another root if results are null. q174's hit is terminal with bonus EOS, so it cannot demonstrate later continuing proxy propagation; retain this boundary/negative control, not an implied general falsification of all history effects.

For each prompt: two unchanged output0/1 baseline observations, followed by order1<-order0, order0<-order1 and order1<-order1 original-value placebo. Baselines must reproduce original complete tokens/semantic trace. Observe all declared baseline roots from the intervention to termination (union across the two orders); retain visited/unvisited roots when actual post-intervention paths change. Report all baseline failures and gains/losses, not just acceptance count. Targets are selected before hidden observation. No forced traversal along baseline acceptance.

##### Bindings and limits

Bind this PLAN, both existing helper hashes, new runner, two original execution JSONs, snapshot, native config/decode_inputs/decode_observations and source670c744e0535dcad before/after. Same prepared ModelScope Qwen3-0.6B 11-file snapshot; weights are local-only and actual model digest/runtime must agree. FP32/eager/CUDA, T0, tied model, EOS, budget100, static unpruned[1,1]/k2/BC9, native gains unchanged. Per-prompt generator seeds as above; input IDs and independent AR must match the exact native evidence index (q171 index0, q172 index0, q174 index2).

Budget15 generations x100=1500 new tokens: six unchanged observations, six real transfers and three placebos, no auxiliary model forward/teacher/warmup. Expected native lengths47/14/13; each replay must equal independent AR completely, without shortening/context-limit. Model call count must equal1+actual decode_calls. Baseline trace and placebo must reproduce originals exactly; pre/through-intervention trace must match in transfer cases. At actual forward assert selected call is the fixed root, donor equals declared saved tensor, and nonselected coordinates are bit-identical. Existing functions restore hooks/provider on failure. Record forward counts/resources and all full actual traces.

Compare captured committed anchor and velocity with recipient baseline at every common declared root: bit equality and max absolute/L2 deltas; inequality is a measured result, not automatically a gate failure or license to silently realign. Order0 velocity must remain zero. Incoming proxy and seed changes are measured separately, with first-seed target margins and baseline accepted-node lengths at matched roots. The first seed at intervention must be unchanged; actual source root/owner must match actual returned trace. Skipped/new roots remain explicit and do not become matched-state evidence.

Use one new .tmp/output-cross-prompt-20261014-v1 directory with baseline/transfer tensors and failure report; refuse existing output. Scientific full-source validation is retained, not rerun as new evidence. Three prompt versions/seeds are not a full matrix. Serial single GPU, OMP/MKL1. Prior same-model observed RSS~6.1GB/GPU-global~5.7GiB guides capacity; sample RSS/threads/device-global memory/available RAM/I/O at.5s, approximate and includes background. Stop on hash/output/scope/ownership/nonfinite/OOM failure and preserve it, no replacement case or retuning. No push, merge, data deletion, new default or original Run re-sign.

##### Conclusions allowed

Test whether q173-like hit loss/restoration appears under this fixed intervention outside q173. Changes in proxy/margin without acceptance change count as null for hit transfer, not a success. A null does not disprove every other root intervention; a changed hit does not prove global history mediation or stable method advantage. Explicitly report q174 termination boundary and existing matched-order baselines. Save scalar/full-trace evidence in this Study and local tensors ignored; results and scope go back to STUDY_REPORT with a local commit.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-plan"></a>

### studies/output_space_small_matrix/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_small_matrix/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/PLAN.md)；原SHA256 `cfa06142c9380a3b6f4ecaa7367263919bb235a5152807faa8ae288e6fa0fa24`。

#### Output-space small-matrix Study Plan

##### Question and selection

Read the latest output_space_prerequisite report and offline_diagnosis before selecting this stage. q171 showed input-chain accepted17 versus output-order0/order1=10/10, despite different owner-mask statistics and identical output-order verification trajectories. Does this descriptive pattern persist on three further predeclared prompts, and at which matched committed-prefix positions do the greedy trajectories differ? This reuses the working implementation; not a mechanism/novelty/Kalman test.

Selected by contiguous dataset order, without observing new outputs: translation skip11/take3, question_id172,173,174; first turn German->English, nonthinking Qwen template. [Prompt manifest](../studies/output_space_small_matrix/docs/prompts_manifest.json) freezes exact text/reference and dataset hash. q171 is excluded and its raw/source/PLAN remain untouched. n=3 prompt units, not three independent seed replicates or a held-out population sample.

##### Fixed model/method/budget

Same ModelScope Qwen3-0.6B snapshot and 11 file hashes, actually tied weights, FP32/eager/CUDA, T0, EOS stop, max_new_tokens100 per prompt, max_prompt_len2048. Source670c744e0535dcad on local9ba5c3a, upstream7a15c46 v0.3.1, existing torch2.11.0+cu130/Transformers5.12.1 maintained. No algorithm/source changes. Native Run seed123; per-prompt generators use seed+index123/124/125 (T0 gives no sampling selection).

Four Experiments, one Run each with all three prompts: frozen_mean input chain, output_order0, output_order1, independent AR. All spec methods: unpruned static[1,1], k2, BC9, efficient, same prefill input mean. Output order1 beta=.9/beta_prime=.99/gamma=rho=1 and mean embedding-row-norm mapping unchanged. No tuning/sweep, adaptive Kalman, untied/general tree or full matrix. Measured output budget at most4*3*100=1200 new tokens, plus at most900 tokens for three per-prompt AR audits; existing warmup8 per case and internal audit remain, up to56 additional warmup tokens. No replacement of a low-acceptance prompt.

##### Design, expected information and diagnosis

Report all per-prompt accepted/depth1/depth2/calls/new/committed and pooled sums. Quality passes only if each speculative output and its internal AR equal the independent AR under identical prompt IDs and full budget, with no context-limit shortening. Acceptance improvement is not a pass threshold. Preserve negative/zero/mixed observations, do not prefer the best case.

For each same-prompt pair (input/order0/order1), align trace rows by absolute root_position and verify cache_before and committed prefix are equal. Compare accepted-node lengths and committed segments at these common positions; list noncommon positions, which arise because different acceptance lengths skip different future roots. Do not compare step number across different generated-prefix positions. Compare output0/order1 owner-mask difference statistics at matched roots, but do not call this full vector/logit agreement. Missing draft probabilities, full mask vectors and exact/proxy-hidden pairs prevent causal attribution: any mechanism explanation remains a hypothesis. This stage narrows whether q171 was the only observed pattern before instrumenting heavier diagnostics.

##### Execution, provenance and stop

New version output-space-q172-174-20261014-v1 and new Run IDs, separate Study. Freeze PLAN/prompt/model manifest/declarations/recipe/phase files and SpecMTP source before make; actual input/model/runtime evidence recorded by existing stages. Source/freeze/gate check before execution; same architecture/science code as the just-passed270tests/36 strict cells and wheel/sdist checks, so do not repeat those hardware experiments or alter code to create new evidence.

Native make --num-gpus1 --init-gpu0 --round1; launch same queue, then process/status/logs/report. Existing verified serial capacity of same model6GB device-global sampled peak guides round1; sample new process RSS/thread/I/O/device memory and available host RAM, distinguish heuristic est from measured Flow/launch wall. OMP/MKL1, no competing experiment. Any OOM/token mismatch/source drift/missing raw/gate failure stops expansion and preserves failed Run. Successful old IDs never re-signed; corrected execution must newversion.

##### Completion and limits

Execution complete if four new Runs succeed. Quality if all12 prompt outputs, full AR equality, current source/input/Study gates and independent prompt identity mapping pass. Save per-prompt paired trajectory diagnosis with actual raw hashes, NUMBERS/report/chart and redraw tool. Formal selected evidence local; full raw/log/modelcache ignored. No push, PR merge, data deletion or publication.

Three contiguous prompts are a small feasibility/development matrix; not a generalization, significance or method-superiority result. Negative evidence can motivate a predeclared proxy-vs-exact hidden diagnostic, but does not license immediate large-scale runs, retuning or declaring all output-space methods ineffective.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-scale-plan"></a>

### studies/output_space_small_matrix/docs/SCALE_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_small_matrix/docs/SCALE_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/SCALE_PLAN.md)；原SHA256 `54eaf6cb1e6950ab713935f8fa31c8d291c2c000d5ddc6bd957ffa930b50090c`。

#### q173 root83 direction/norm counterfactual plan

Predeclared2026-10-14 after the window diagnosis: at root83, frozen-input seed mask norm=.1912895292, output-order0/1 seed mask norm=.9250697494 (4.84x); output0/order1 first vectors are exactly equal. Their correct-target margins are+.63981 versus-5.51714. The first-slot target is7602, same true prefixAR[:83]. Direction and norm differ simultaneously; original observations do not separate them.

Fixed 2x2: input/output recorded seed-vector direction crossed with the two observed norms, not a searched grid. Use q173 root83 only, seed input from selected_tensors.pt bound by the window diagnosis SHA. No new generated tokens, decoder replay, prompts, gain/norm optimization or scientific-source change. Four teacher-forced prefix+mask forwards maximum (two observed combinations reproduce the window isolated checks, two are counterfactuals), FP32/eager/CUDA/current runtime, same verified ModelScope snapshot. At most84 positions each, serial/threads1; version output-space-q173-scale-20261014-v1. Copy neither weights nor originalRun, delete nothing.

Require source/model/snapshot/runtime/input hash consistency, raw first vectors as saved, nonzero finite norms, observed combinations identical top1 and target-margin agreement<=1e-3 with recorded isolated logits. Preserve all four outcomes, including failures. Report target probability, top1/top2 and margin, distinguishing these isolated mask-position predictions from full-decoder accepted counts: counterfactual queries do not establish a stable method improvement and are not retroactive selection of a new default scale. No push/merge/full matrix.

The follow-up may separate a local direction effect from a local norm effect, including interaction. It cannot show generalization, select global beta/mapping, or explain every trajectory. The original3prompt matrix, q171 negative result and locked science recipe remain unchanged.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-transfer-plan"></a>

### studies/output_space_small_matrix/docs/TRANSFER_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_small_matrix/docs/TRANSFER_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/TRANSFER_PLAN.md)；原SHA256 `f90669213dbdde85a625a087386764376daa9be8bb2a005cf24f98a0bf780b8b`。

#### q173 single-slot historical-proxy transfer plan

Predeclared 2026-10-14, before the three intervention replays. Version: output-space-q173-transfer-20261014-v1. Builds on the retained matrix, WINDOW_PLAN, SCALE_PLAN and post hoc structural diagnosis; it does not replace an original Run or add a formal acceptance cell.

##### Question and fixed intervention

At saved root83, output0/1 owner0 slot0 vectors are identical. Their slot1 vectors have the same observed norm (about .92507) but cosine .99831766. Both original paths accept one draft at root84 and next visit root86. Only exchange the returned owner0 slot1 vector at root83 (zero-based indices [0,1]), using the saved opposite-order current_masks tensor. Leave all other vectors, committed state, logits, decoder, cache, gains and defaults intact. This tests whether a single earlier mask can propagate through proxy history to later candidates without a direct slot0 velocity term. It cannot isolate all earlier velocity history or establish a general method advantage.

Three serial cases, no additional selections or reruns for an unfavorable result:

| recipient | root83 slot1 donor | role |
| --- | --- | --- |
| output_order1 | saved output_order0 | one-to-zero transfer |
| output_order0 | saved output_order1 | zero-to-one transfer |
| output_order1 | saved output_order1 | original-value placebo |

Replay the unchanged SpecDecoder directly with a scoped provider patch, following existing tools/profile_cpu.py and window tools. Do not use the unchanged-window observer's baseline acceptance path for post-intervention seed ownership. Infer the preceding selected owner from actual root progress, then check against actual returned trace. Observe only predeclared roots83/84/86/94/97 if visited; a root skipped by a changed acceptance path is recorded as unvisited, not forced or silently substituted. Capture first-seed margin, actual seed source owner/position, incoming proxies/anchor/velocity, current masks and mask hidden rows. Report all three cases, including null and adverse effects. No search over seed, root, scale or donor.

##### Bindings, budget and execution gates

Bind scientific source670c744e0535dcad, unchanged original execution_v1.json, original Run config/decode_inputs/decode_observations, WINDOW_PLAN and observer JSON, this PLAN, tool/helper hashes and selected_tensors SHA47166a3b41e0c00ed39110593fe7f83b20323c2e3714fe0355f2a43aaa9dc14c. Reuse the same locally prepared ModelScope Qwen3-0.6B 11-file snapshot, verify all hashes before/after, actual model digest and runtime equality. Tied model, FP32/eager/CUDA, unpruned static[1,1], k2/BC9, beta .9/beta_prime .99, T0/EOS, seed124, max_new_tokens100. No download, src or default recipe edit.

Budget at most three decoder generations x100 new tokens, no auxiliary teacher/prefix forwards and no warmup. Each generation must reproduce all saved independent-AR tokens (33 for this q173) without context shortening. Instrumentation forwards must equal1+actual decode_calls. Select the intervention call from the original root83 index; at actual forward assert it is root83 and exactly one provider call was modified. Before/through root83 verify full semantic verification/commit/KV trace and original root83 masks. Nonselected tensor coordinates must remain bit-identical, the replacement must equal its predeclared donor, and the patch/hooks must be restored even on failure. Placebo must reproduce original full tokens/semantic trace. Post-intervention semantic changes are permitted in the two transfer cases and must be retained.

Use a new .tmp/output-transfer-20261014-v1 output directory; refuse existing outputs. Preserve failed output and all earlier data, no push/merge/delete. Formal scalar JSON/report stays in this Study; local tensor payload stays ignored. Sample process-tree RSS/threads, device-global GPU memory, available RAM and I/O at .5s; these are approximate peaks with background occupancy, not allocator or speed rankings. Prior observed capacity is about6.1GB RSS and5.7GiB device-global GPU memory; serial execution only. Stop on binding/replay/ownership/scope gate failure, nonfinite tensor, OOM or resource failure; preserve failure and do not change contract to pass.

##### Interpretation boundary

If an exchanged second slot changes the next proxy and subsequent first-seed direction/margin while the first seed at83 is unchanged, this supports a local historical propagation route. If only immediate proxy changes or the later94/97 margins/acceptances persist, record that this single early intervention does not explain those later gains. Neither outcome identifies all hidden mediation or the contribution of current velocity in later slots. No stable benefit, global EMA causal claim, new optimized default or performance ranking follows from this one q173 window.

</details>

<a id="legacy-studies-output-space-small-matrix-docs-window-plan"></a>

### studies/output_space_small_matrix/docs/WINDOW_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/output_space_small_matrix/docs/WINDOW_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/output_space_small_matrix/docs/WINDOW_PLAN.md)；原SHA256 `f6e966c71706fe13c07295e2a73037b300295e9048506b4e470c45f9d5689a7b`。

#### q173 matched-window diagnostic plan

##### Purpose and predeclaration (2026-10-14)

The fixed three-prompt matrix is retained. Diagnose q173 at root83 (input-chain advantage) and root94/97 (order1 versus order0 advantage), selected from the saved paired-prefix analysis before examining hidden/logit values. No other windows, gain search or replacement prompt. Version output-space-q173-window-20261014-v1 identifies an instrumented diagnostic replay, linked to the original four native Run IDs; it is not counted as new acceptance evidence or used to re-sign them.

Existing tools/profile_cpu.py measures the unchanged SpecDecoder directly; follow the same tool pattern, with read-only hooks around the existing forward/provider. No new decoder, sampler, cache routine, Run scheduler or scientific src edit. Tool source hash, this PLAN hash, original execution JSON/raw hashes, source670c744e0535dcad and ModelScope11-file hashes bind the measurement. Temporary tensors/logs go to a new .tmp/output-windows-20261014 directory; formal scalar evidence/report stays in this Study. No push, merge or data cleanup.

##### Fixed replay and budget

Same Qwen3-0.6B ModelScope snapshot, FP32/eager/CUDA, unpruned[1,1]/k2/BC9, order0/order1 gains, T0, EOS, budget100. Replay q173 for input/order0/order1 serially once per case, using the saved rendered input and original per-prompt generator seed124. No warmup or timing comparison; hooks may synchronize and distort wall time. Existing runtime flags must match. Full output and semantic verification/commit/KV trace must reproduce the corresponding saved matrix case exactly; otherwise preserve failure and do not interpret the measurements.

At most300 decoded new tokens across three replays (the saved outputs have33 each). One teacher-forced causal forward over the saved AR sequence supplies exact normalized hidden references; it is not a new AR generation. For each of3cases*3windows, at most1 additional causal prefix+actual first seed-mask forward checks that recorded seed logits have the same top1 as an isolated correct-prefix replay. Max10 auxiliary forwards, max103 positions per forward, singleGPU/threads1, no concurrent experiment. Capacity uses the prior same-model serial calibration and fresh device/RAM check, recording new sampled resources separately.

##### Observation timing and quantities

Current-block masks seed the NEXT tree. Acceptance at root p uses mask logits from the preceding block's deepest accepted owner's masks (first mask input at position p), not the masks being constructed at root p. Record seed source block root/owner, absolute mask positions and input vectors, normalized hidden/logits, as well as current root exact hidden and output-provider committed anchor/velocity/proxies. The original root verifier and cache keep logic remain untouched.

At each selected window, compute first-slot target token versus best alternative logit margin, top1/top2 IDs/probabilities and target probability; the target is the independently saved AR token at p+1. Compare seed-mask norm, proxy-versus-exact root hidden cosine and relative L2 error. Record second-slot candidate too, but label its AR comparison valid only if the first draft token matches the AR next token; an invalid draft prefix is not an exact hidden observation. Save full selected tensors locally to make scalar calculations reproducible, without publishing weights or full vocabulary logits.

The auxiliary prefix+mask input is AR[:p] followed by the actual stored seed mask at position p. It has the same true prefix as the preceding accepted owner, which is independently checked from the original block layout. Record differences in original versus isolated target margins; require equal top1 IDs and normalized-hidden cosine>=.99999. Root exact hidden versus teacher-forced AR hidden also requires cosine>=.99999. These are diagnostic consistency gates, not acceptance-improvement thresholds.

##### Interpretation and stop

All three replay/token/trace/source/runtime/input and window-completeness gates must pass. Verify hook removal and provider method restoration, forward counts (1+decode_calls per replay; no observer-induced target calls), and model digest unchanged. Stop on any mismatch, OOM, missing capture, source/file drift or consistency gate failure; no silent correction or retry on an old diagnostic output directory.

A seed margin crossing0 at the same verified prefix explains a greedy candidate/acceptance difference under these fixed observations. It does not by itself establish that EMA velocity is optimal, that proxy-error magnitude causes the gain, or that output-space masks outperform input masks. Keep both advantage and disadvantage windows and all failures. Do not retune, expand the matrix, change src or declare Kalman supported.

</details>

<a id="legacy-studies-qwen3-4b-bf16-directions-docs-plan"></a>

### studies/qwen3_4b_bf16_directions/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_directions/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_directions/docs/PLAN.md)；原SHA256 `0ef4df6b1856c89e10b71d87a5cea16abfcbecff6391adbc086455cc0d5b4cf7`。

#### 固定480条BC30方向对照

2026-10-16，接续Enmao监督指示：四窗口不足以回答方向是否特殊，先完成BC30全480，再按实际资源工作量决定BC60。只比较指定方向控制，不发明新方法，不改变prompt或主实验数据，不测速。

##### 一、固定条件

使用`qwen3_4b_bf16_five_methods`已验收的480条SpecBench，顺序、首轮non-thinking模板、input_ids、EOS、seed123+index、max_new_tokens256完全相同。ModelScope Qwen3-4B BF16/eager，T0、双mask、动态树剪枝、BC30/draft budget9，level为原输入updating mean、λ0.1、γ1、normalize=false、extrapolate_ema=false。模型/source/输入内容校验后才forward。8B、测速、算子机理和新prompt筛选不在本轮。

##### 二、六组与既有方法的关系

| 组 | 完整解码定义 | 执行 |
| --- | --- | --- |
| template | v固定为原prompt e2−e1，m_i=updating mean+i*v，每次提交只更新mean，v始终不变 | 新完整480 |
| random0/1/2 | CPU独立Generator种子0/1/2采样标准正态方向，FP32归一至同prompt e2−e1范数再转BF16；每次提交只更新mean，v始终不变 | 三个新完整480 |
| zero | v恒零，完全等价于已有输入零阶updating mean，不是frozen mean | 复用已验收30/input_zero的480条，不重跑 |
| content | 原输入一阶β0.9999的初始化从原序列首个user内容token起的后缀计算，保留完整prompt mean；随后按原生EMA更新v及mean | 新完整480 |

template与随机组是严格固定偏置，和四窗口干预的读出方向定义一致；不能用已有原生input_first（β0.9999且有更新）冒充template。content沿用先前控制定义：后缀包括原末尾模板，不改模型输入，不另行范数匹配。其余mean/anchor/树/cache/验证/停止语义由原生decoder保留。

四窗口原生input_first与template接受数相同不意味着全480相同；全量报告明确列template对原input_first的配对，检验差异。原五方法和共享AR全部复用旧正式结果，保留原source_digest；新Study的源码摘要会因已迁入第三方参考文件而不同，核对科学decoder/provider/模型文件逐字节相同，不重签旧结果。

##### 三、指标与停止

主τ=Σdecode committed/Σdecode_calls，另报逐promptτ均值、category pooled等权均值；accepted包含预算/EOS裁剪前接受数，主calls=prefill+decode。逐question_id报告各组相对template的accepted、calls和τ胜/平/负；另列与旧五方法的全部配对，输出长度和AR首分叉并列，不声称同输出测速收益。

硬门：每个提交token等于该次验证前向正确owner logits的argmax，全覆盖/hash/owner链由原生validation核验。独立AR仅复用旧参考作诊断，不新增AR GPU、不作失败门。各新格完整480并原生Run和Study process通过后才能称完成；pending不补零、不当平局。科学门失败、来源不符、OOM均保留新Run失败，不自动重试或修prompt。

##### 四、执行与资源

走原生SpecMTP/RPipe make/launch，Study专用provider作用域只替换指定方向并复用原native untimed输入合同。五个新增Run单队列`--round 1`串行，禁止第二队列。加载前只看GPU0 nvidia-smi实际驻留外部Python与空闲显存≥14GiB，各prompt再次检查驻留作业；不读取CPU进程/脚本SHA，不抢占不杀进程。外部实验出现则该格保留停止状态，GPU释放后用新version/Run继续，不覆写失败。

每格写原始token/trace/argmax和方向初值标量证据，完整Run/model/cache留本机。执行前提交冻结声明、Study代码与输入来源manifest。CPU测试覆盖固定v跨提交不漂移、mean与零阶完全一致、随机同范数/种子、content保留完整mean及原生tiny BF16硬门。没有GPU测速计时；只记录总forward/token工作量供BC60决策，未测墙钟不得填性能数。

首轮只声明BC30五个新Run，BC60在其完成后依据资源负担与信息量再写补充冻结计划，不提前暗中筛选优胜方向。阶段报告持续列实际状态、来源与局限；主题commit/push、PR合dev，合后删远程分支。保留全部本地分支/数据，不合main、不强推、不弹窗。

</details>

<a id="legacy-studies-qwen3-4b-bf16-directions-docs-recovery-plan"></a>

### studies/qwen3_4b_bf16_directions/docs/RECOVERY_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_directions/docs/RECOVERY_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_directions/docs/RECOVERY_PLAN.md)；原SHA256 `ff430b2b472938ba430ea1411d691535eebf25de3f96c6883415177fbf8e00e9`。

#### 外部GPU准入中断后的补齐协议

2026-10-16，原BC30队列的random0 Run `be910fe1fa042b28`在完成46条后遇到真实nvidia-smi外部Python PID65348。按冻结协议在下一prompt开始前停止；并非数学门或OOM失败。46条原始token/trace保留，原生argmax验证9,028个token通过，证据见[resource_interruption_v1.json](../studies/qwen3_4b_bf16_directions/docs/resource_interruption_v1.json)。该格没有有效全480τ，不将46条或旧四窗口拼成全量。

原队列已自动进入random1，不中断它，不另起GPU队列、不杀外部进程。先让原队列到终态，逐项核实哪些因素完整成功、哪些因资源中断失败。已成功template全480和旧zero全480绝不重跑。

队列退出后，仅对未完整成功的指定因素建立新recovery Study：换algorithm.config.version、新Run ID，完整480从头执行，不覆盖旧Run、不删数据、不宣称支持KV checkpoint恢复。不更改数据、prompt、模型、dtype、控制公式、seed、停止或正确性门。复用原control函数原文及全部科学/source/input引用；资源等待的明确差异见下段，另有Study路径、失败因素与新version差异。原partial只作为保留的失败证据，不参与最终有效格的重复计数。

队列终态补充：随机1/2也因真实GPU进程中断、content加载前未获准入，原队列已自然exit1，无任务被人工停止。重复短时占用暴露原assert式资源处理会浪费完整prompt。因此补齐Study保留`control()`完整原文及科学56文件，只将两个资源拒绝调用点改为同一Run在加载前/prompt边界sleep5秒后再查，等待期间不做forward、不重启。完整driver不再声称逐字节相同，新增资源helper源码差异由新Study冻结；详情见[补齐执行PLAN](<研究计划 Denoising Speculative Decoding.md#legacy-studies-qwen3-4b-bf16-directions-recovery-docs-plan>)。这不改变方向构造、数据或数学门，不是新方法。

补跑前仍只看GPU实际驻留Python与可用显存，free≥14GiB后单队列round1串行；如果仍有外部占用，不启动第二队列，也不因CPU进程阻塞。新队列同样不自动重试失败格，后续根据真实终态再处理。

最终报告保留原Study的失败/终验结果，不将其改写成成功。每个指定方向只采纳唯一完整native成功Run：优先原成功格；只对原失败因素采用预先声明的补齐Run，不按τ挑选。补齐Study自身native process必须通过；组合验收逐因素核对480条、seed、参数、模型/runtime、原生argmax、科学源码与控制实现字节一致性。若任何方向仍缺失，整个方向目标仍不完成。零偏置继续明确复用原生输入零阶，不重新签收旧Run。

恢复会多产生失败前的真实计算工作量，报告单列，不作测速或性能数据。读出只在组合完整后给出最终六组矩阵与结论；PR合dev及远程分支删除仍待全部规定格通过，不因为补跑而缩小到已成功子集。

</details>

<a id="legacy-studies-qwen3-4b-bf16-directions-recovery-docs-plan"></a>

### studies/qwen3_4b_bf16_directions_recovery/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_directions_recovery/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_directions_recovery/docs/PLAN.md)；原SHA256 `45c24df0cdca96ba2e8f31753ac8c281fddd88c04937f07780e2a9387187b394`。

#### BC30全480资源补齐计划

2026-10-16，原单队列已自然结束，无运行任务被中断。原template全480成功；random0/1/2分别在46/50/9条遇真实GPU外部Python而终止，content在加载前未获准入，均未产生有效全480τ。原Run、失败result和partial token/argmax全部保留。

本Study仅补四个未完整因素，全部480从头跑，新version/Run，不能把partial拼成全量。不重跑成功template或旧zero，不修改原Study或失败记录。模型、数据、prompt、T0、BF16/eager、BC30双mask、动态树、seed123+index、max256/EOS、mean/方向/EMA规则与[原冻结协议](<研究计划 Denoising Speculative Decoding.md#legacy-studies-qwen3-4b-bf16-directions-docs-plan>)完全相同；同范数随机0/1/2是固定方向，content是原后缀初始化后native EMA，内容范数不额外匹配。

唯一执行差异是资源控制：原实现在发现外部GPU进程时直接失败，重复浪费完整prompt。新版本加载前及每个prompt开始前仅查询真实nvidia-smi，满足原14GiB初始空闲及无外部GPU Python条件后执行；不满足则本进程sleep5秒后再查，保留当前Run、模型和已完成prompt，不重启、不做GPU前向、不枚举CPU进程、不抢占或杀其他作业。等待状态写本机`.tmp/full-direction-resource-wait.json`。睡眠不进入测速，decoder模块计时仍为常量。

只在prompt边界等待，不宣称KV/checkpoint恢复；OOM、来源或数学门错误仍原样失败，不自动重试。单队列round1，原队列已确认terminal exit1后才允许启动本队列。当前GPU任务不被文档或后续研究计划模板对齐打断；模板补全使用附加文档，不改已冻结执行PLAN字节。

科学包56文件、旧输入/权重合同及source_refs原字节保持；`control()`完整方向公式与原Study AST/字节保持，资源等待helper和两个调用点是已披露的新Study来源差异。开跑前CPU回归验证不满足准入会等且恢复、不把CPU进程当GPU、控制函数一致，并运行既有tiny BF16控制门。

四格各完整480、native Run和本Study process通过才算补齐成功；原Study保留资源失败终验，不假装其通过。最终跨Study只选原template成功格和本Study四个预定因素，zero引用旧input_zero；每因素只有一个完整有效Run，不按τ选结果。组合验收逐项核对数据/模型/runtime/参数/seed、480条全argmax和原生来源，任何一格缺失则目标不完成。全量报告输出三种τ、相对template及原五方法的accepted/calls/τ配对胜平负，并保留AR诊断、输出长度和资源浪费工作量。

BC60仍等BC30完整后按实际token/forward工作量决定；8B/HF第三方不再推进，新方法/调prompt/测速/算子机理均不做。结果主题提交推工作分支，PR22完整后ready/合dev/删远程分支，不合main、不强推、不删本地数据、不弹窗。

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-geometry-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/GEOMETRY_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_five_methods/docs/GEOMETRY_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/GEOMETRY_PLAN.md)；原SHA256 `86b5b92e283c059e0abae35db3b86aa06f54e630ea6ebd195a9cc6ab231efe00`。

#### 4B BF16 mask 几何与接受深度诊断

2026-10-15 Enmao 新指示：分析输出零/一阶为何落后输入一阶，先用既有结果做 CPU 分析，再按需要补最小 4B 观测。本文件是新诊断协议，不改首次 make 的 PLAN、配方、来源或成功 Run；本轮不是 ESP 原作者代码复现，不使用 8B。

##### CPU 分析（零模型 forward）

- 读取全部 480 prompt、两个 BC、五方法的原始 decode_observations.json，并绑定 paired_evidence_v1.json 的原始 SHA；不重新执行 summarize 或签收 Run。
- 核对逐深度 attempt/accept、完整 trace 与总计数。深度条件接受率只使用真实 attempt 分母；按生成 root 索引分箱 [0,32)、[32,64)、[64,128)、[128,192)、[192,256)，报告每次验证块接受至少一个/两个候选的比例，不能冒充该位置的深度条件接受率。
- 用 τ = 1 + accept_d1/calls + accept_d2/calls − trim_loss/calls 分解差值；trim_loss=accepted+decode_calls−committed。它是计数恒等式，不是跨不同输出路径的因果分解。
- 检查输出 mask 的逐块 min/max 范数与记录的目标 r，核对固定 state/proxy 下速度项的作用槽位、EMA 更新、状态是否被预测污染。原日志没有 hidden 向量，不从范数或 token IDs 编造余弦分布。

##### 最小 GPU 观测（须经非抢占准入）

固定选择原清单 coding、math_reasoning、summarization、translation 各第一条，选择不依赖结果；BC30，输入一阶/输出零阶/输出一阶，原 BF16/T0/eager/默认设置，max_new_tokens=64/EOS，共 12 个短诊断，生成上限 768 token、模型 forward 上限 768。复用已核验的 ModelScope Qwen3-4B，单进程串行，不下载新权重、不跑独立 AR forward、不测速。

启动前两次检查共享 GPU 显存与 Python 作业；没有其它 GPU/未知模型作业且可用显存至少 18 GiB 才加载。已有其它作业则不加载、记录 deferred；不抢占、不终止其它进程。每个短诊断前再核查新作业。桌面图形进程与已确认的 CPU CI 单列；这些检查是观测准入，不冒称跨进程的原子全局预约。

前两次准入因发现未知 Python 作业延期，未加载模型。随后核验量化任务 `download_layers.py`：只调用下载和文件哈希，不加载模型；SHA256 `10689bac808d0e5fa1958d306508f459d3c4637dc92a51bf5b6c1bb915545026`。另核验 `build_official_container_v2.py`：仅准备容器、编译扩展，Docker create 没有 GPU/device 参数；SHA256 `b22cbe9814bcfcf85e9e6caade2a60a2258d2bb83f5e1d856124753bf8976e44`。仅路径与 SHA 同时匹配时按已核验 CPU 准备作业放行，变化或其他未知作业仍延期。保留前两次延期证据。

第三次加载成功后，逐格准入发现新 CPU watcher 即退出，尚无生成；第四次在初检延期。为识别 Windows venv launcher 的短暂无命令行子进程，将已核验 CPU CI watcher/release-ci、显式排除 GPU 的 RPipe 测试及其 Python 后代按父 PID 链分类；不关联的未知模型作业仍拒绝，并有回归覆盖。所有延期/退出记录保留，不将模型加载计为生成结果。

使用未修改的 SpecDecoder/HFCausalLM/真实 provider，只增加只读 hook，记录实际输入 mask、原生 final-normalized hidden、provider 的原始 state/proxy。只处理 root 与实际接受路径上的 owner；拒绝分支没有真实延续，不混入目标统计。槽 j 的目标为该 owner 之后第 j+1 个实际生成 token 的 embedding，末尾无目标者跳过并计数。另比较真实 owner hidden 与其预测的下一 token embedding，区分原始 hidden、范数缩放后的实际 BF16 mask、真实 embedding。

验证全部提交 token 的原生 verifier-argmax 审计、输出为原完整 Run 的前缀、hook 未增加 forward。独立 AR 前缀一致率只从已有共享 AR 读出，不作为失败门。完整向量/原始 token 仅本机保存，公开逐点几何标量、分位数、计数及 SHA。几何分析在 CPU 完成；不改变算子/归约开关，不分析 BF16 算子数值分叉。

##### 假设与可证伪边界

1. H1：输出范数映射错误或偏离目标 r；用全部旧 trace 的范围及新实际 mask 范数检查。半径正确不证明方向或输入分布正确。
2. H2：last-hidden 与输入 embedding 的方向/分布不兼容；比较 raw hidden、actual mask、真实 owner hidden、目标 embedding 及输入 mask。tied 权重不等价于 hidden 到 embedding 的逆映射；余弦相关不等于接受因果。
3. H3：速度直接影响的槽与第二深度实际机会有限；联合源码槽位恒等式、深度条件接受率和每块深度收益检查。固定 state 下首槽不变不能推出整个后续轨迹完全无影响。
4. H4：代理 hidden 对真实 owner hidden 的误差掩盖速度收益；只在有真实 owner 的路径计算，两方法不同路径及共享前缀子集单列，不将拒绝 owner 当真值。
5. H5：输入/输出 β、初始化与阶数作用定义不匹配；报告实际速度/anchor 范数与槽间角度，不能把不同固定配方的差异全部归于空间。

当前只观测，不改默认配方或调参，不启动大矩阵、teacher 或算子机理。GPU 不获准时交付完整 CPU 证据、实现检查与未完成几何缺口；获准则按上述固定预算补采。结果写入 docs/research，并保留所有负结果与局限，按主题提交和 PR 合 dev。

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-active-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_ACTIVE_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_five_methods/docs/MASK_ACTIVE_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_ACTIVE_PLAN.md)；原SHA256 `229e04dd5f4a3b8eb5b46d388b8e60f578da7dc87512a21a86ee680c11c0494d`。

#### 4B 第二深度机会条件窗口

2026-10-15，在首批24格完成后追加的条件诊断，非原先无条件窗口的扩容或替换。首批4个下一树的真实d1均非Top-1，不能直接检验“有真实第二深度机会时速度是否有益”。首批的全部负结果保留，代码冻结版本为 ce7c329，不重跑。

##### 固定选择与预算

只选 q121/q401，各读取原已完成输出零阶 Run 的 trace，从头找到第一对相邻块，其中后一块 `accepted_nodes == [1]`。选择前一块作为 mask 干预窗口，分别固定 root_index=3/19，下一验证 root_index=4/20。该规则利用原控制的接受结果，是**条件选择**，不能据此估计全480 win rate或τ，也不能与首批4个无条件样本混合成随机样本。

沿用 MASK_WINDOW_PLAN 的同历史shadow、原生输入一阶控制及四输出局部公式，不调参数。2窗口×6格，每格两次局部forward。原控制前缀重建加干预总forward最多64，超预算即停。只跑4B BF16/T0，已有魔搭权重/eager/default，不跑8B、teacher、新AR、测速或算子链。

开跑前确认非抢占准入，每上下文复核，未知作业延期，不抢/杀其他任务。新目录 `.tmp/4b-mask-active-v1`，原始文件/分支/数据不删除。新工具版本记录独立SHA，原测量工具 ce7c329 的 Git blob字节作为首批来源，不能拿新版本重新签收旧数据。

##### 门与读出

控制前缀、同块owner argmax、原控制下块轨迹门沿首批。额外核验控制下一树实际有第二层、实际首候选命中且只接受1个，拒绝把没有机会的窗口冒充活跃深度。保留所有正负/平、候选排名、宽度、实际接受深度和数学argmax审计。独立AR只沿用旧报告诊断，不新增forward。

CPU后续直接读取保存的GPU实际树，不要求CPU软最大值重建与GPU树逐项一致。候选覆盖反事实使用剪枝后的严格/并列排名区间，区分必然覆盖、可能覆盖及真实验证接受，不追查跨设备排序数值原因。

若核验失败则保留失败条件并停止，不换prompt/窗口补成功。完成后与首批分表报告、写入阶段报告并按PR合dev，不合main、强推、删分支或数据。

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-direction-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_DIRECTION_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_five_methods/docs/MASK_DIRECTION_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_DIRECTION_PLAN.md)；原SHA256 `c824223f7a2c173c8cebf5f5971caca760ae34177cee42fbe82a54d266aa38c6`。

#### 固定数据下的模板方向对照

2026-10-15，按Enmao明确指定的四类对照执行。不提出新主方法，不改原五方法定义、数据、prompt、chat模板或任何input token。五方法BC30/60各480条加一次AR的11/11完整验收已在原Study完成，本协议仅是局部分析对照。

##### 一、固定上下文与八格

仍使用已冻结的输出零阶轨迹q121/R1、q401/R0、q241/R1、q161/R2，四处真实prefix/cache/tree/位置和注意力固定，下一验证块沿原输出零阶规则。8格/窗口共32格，总forward上限96，不扩CPU筛选、不生成完整新矩阵。

| 格 | 当前块输入mask的速度 |
| --- | --- |
| zero | 原输出零阶控制，不是输入零偏置 |
| input_first | 原输入一阶同历史控制 |
| input_template | 原v=e2−e1，固定于原prompt最前两token的embedding差分 |
| input_random_0/1/2 | 3个固定随机种子0/1/2，各用同一个随机方向服务全部4个prompt；范数匹配原e2−e1，固定偏置、不随内容更新 |
| input_mean | 零速度，保留原updating mean；即输入零阶，不是frozen mean |
| input_content | 仅把速度初始化移到原用户正文最前两个token，从该处起按原β处理prompt后缀，再正常更新已提交token；mean仍由完整prompt初始化并以原λ更新 |

随机方向用独立CPU Generator的标准正态向量归一化到原差分范数，再转实际BF16；记录实际范数比/与原方向余弦，范数比误差必须<1%。该CPU采样不是FP32模型前向，不搜索种子。正文位置通过原SpecBench第一turn和原tokenizer offset定位，重新渲染仅用于核对与冻结input_ids逐token完全一致；实际模型输入始终取原manifest。四处正文起始均为index3，若数据/渲染/位置不符则停止，不改prompt修正结果。

内容初始化保留自己的原生幅度，未额外归一化到模板范数，不能把其方向与幅度效应分开。它也不是只读出时换一个固定内容方向，而是从内容差分开始重新执行原生速度EMA，mean完全不变。其余β=.9999/λ=.1/γ=1、BC30/双mask/动态树剪枝不变。

##### 二、判定与证据

报告每格第一/第二目标排名、树宽及实际下一块接受数，并分别相对零偏置、原模板偏置和原生输入速度统计赢平输。3个随机种子分开列，不选最优种子。目标是检验本批提升是否仅限e2−e1方向；若随机/内容也提升则否定该局部“只有原方向”版本，不能直接外推全480。

当前owner argmax/接受路径不变；输出零阶控制续接旧Run；全部局部提交选择等于对应验证前向argmax。模板/native/零偏置的已测控制逐项对照前批，不重签旧Run。独立AR仍只报告，不新增AR前向。原数据SHA、manifest SHA、内容起始、随机种子、实际速度范数/余弦与工具/协议/model/source/private logits SHA均保留。

##### 三、串行资源与范围

单次命令`--template-control --early-template --direction-controls`，只用实际nvidia-smi GPU驻留Python与显存准入，初始free≥14GiB；CPU进程不查询、不读源码登记SHA。未知GPU作业/容量不足保留deferred，不抢占，不杀外部进程，不起等待或重叠队列。

仅ModelScope缓存Qwen3-4B BF16/T0/eager/default，不跑8B或测速，不进入算子机理。控制/数学/来源/有限值/OOM/预算失败保留停止，不自动重跑或换精度。源码默认配方与原成功Run不改，权重/cache/完整token/logits留本机；正式标量及报告按主题commit/push/PR合dev。

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-early-template-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_EARLY_TEMPLATE_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_five_methods/docs/MASK_EARLY_TEMPLATE_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_EARLY_TEMPLATE_PLAN.md)；原SHA256 `62f1e118711822047af8c82030ef16a0144b40848c9bfb98384c013197a21ce4`。

#### 4B 早期同前缀模板对照

2026-10-15。Enmao最新要求准入只看实际GPU计算进程及可用显存，CPU Python不再检查或登记SHA。PR #19已完成晚窗口/条件窗口24格模板控制，不重跑已有格。本次沿CPU定位的早期候选前移窗口，其他科学源码、默认配方和旧证据不改。

##### 一、窗口与四格

仍以原BC30输出零阶Run重建真实前缀，固定同一当前树/cache/已验证历史，四格为输出零阶、输入原生速度、输入固定模板首差分、输入零速度。三种输入控制共用同一已更新mean，β=.9999、λ=.1、γ=1；zero mean不是frozen mean。复用原工具`--template-control --early-template`。

| prompt | 当前输出零阶root | 控制下一root | 原输入两路径首接受差异root |
| --- | --- | --- | --- |
| q121 | 1 | 2 | 2 |
| q401 | 0 | 1 | 1 |
| q241 | 1 | 3 | 3 |
| q161 | 2 | 4 | 3 |

前三处选原控制中到达候选差异root的前驱；q161输出零阶轨迹跳过root3，保留固定root2→4作为反向候选的邻近对照，不声称精确复现原输入root3。候选来自已有结果的条件选择，不是随机样本，原路径与本轮控制路径不同。所有位置在开跑前固定，不按本轮结果换窗口。

每格读当前mask两行logits、原生建下一树，再验证一个块，报告目标排名、树宽、实际接受数、完整seed logits是否等于原生输入速度。当前owner argmax必须保持，输出零阶续接旧Run，所有局部提交选择均须等于验证logits的argmax。独立AR不新增前向、不作阻塞门。每格两次干预前向，加前缀重建，总forward上限64；不沿干预路径继续生成。

##### 二、资源准入与执行

只查询GPU0的`nvidia-smi --query-compute-apps=pid,process_name`和`--query-gpu=memory.free,memory.used`；排除本进程PID后，如有GPU驻留Python则延期。Windows WDDM会列出桌面程序，桌面进程不当作Python实验；其显存仍计入实际free。CPU进程不查CIM、不读源码、不登记SHA。显存门为初始空闲≥14GiB（4B BF16约8GB权重加临时KV/树与余量），加载后只查有无其他GPU Python，保留查询记录。这是观测准入，不冒称全局原子排他。

只启动一个串行命令，不启动后台或重叠等待队列。GPU忙则保留0-forward的deferred；不杀其他作业。GPU空闲且容量通过立即加载已核验ModelScope缓存。单个输出目录只创建一次，失败记录不覆盖；质量、来源、控制、预算或OOM失败即保存停止，不自动重跑，不换精度、不追算子。

仅Qwen3-4B BF16/T0/eager/default，不跑8B、FP32模型或测速。来源/工具/本协议/原始观测/私有logits绑定SHA，标量结果及报告按主题commit、push、PR合dev；完整权重/cache/logits/token留本机，不删数据和分支、不强推、不合main。

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-reciprocal-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_RECIPROCAL_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_five_methods/docs/MASK_RECIPROCAL_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_RECIPROCAL_PLAN.md)；原SHA256 `f8d78aecb23a1a8dca7b9c379c6ec28d3c00f0723dacb4e6f30fb89441e0fa1f`。

#### 4B 早期同前缀双向mask替换

2026-10-15，按Enmao追加指令直接做GPU因果验证，不再扩CPU筛选。已完成的早期模板16格保留，不重签。本次补输出一阶，并从原输入一阶路径执行反向替换，科学src和配方不改。

##### 一、固定设计

两个origin分批串行：原BC30输出零阶Run、原BC30输入一阶Run。各固定q121/R1、q401/R0、q241/R1、q161/R2，实际必须存在该root；不按本轮结果换位置。每origin每prompt三格：同历史下的输出零阶、输入一阶、输出一阶，仅替换当前验证块各owner的mask构造，prefix/cache/tree/位置/注意力不改。输入使用原β=.9999/λ=.1/γ=1；输出使用原β=.9/β′=.99与范数映射。

输出origin复用原输出零阶capture，并行只读更新输出一阶shadow。输入origin重建原输入一阶路径，以原生last_hidden_only观察接口收集已验证prompt/committed hidden，按相同真实历史更新两种输出shadow；proxy只取上一个真实验证块最深owner的mask hidden。shadow不写回生成状态、不做独立AR或teacher前向。

每格当前owner argmax及当前接受路径必须与origin原控制相同；使用该格当前mask logits建下一树，执行一个下一验证块，未来mask保持origin原规则（输入origin继续原输入mask，输出origin继续原输出零阶mask）。每个提交选择必须等于对应验证前向argmax。native控制mask逐元素等于重建块的实际mask，控制续接旧Run，来源和预算均为硬门。

##### 二、指标与边界

每origin12格，上限64次总forward。记录目标排名、下一树宽、实际下一块接受数、当前argmax/本地数学自检及旧轨迹诊断。CPU读出默认相对输出零阶；报告另外逐origin明确其native控制：输出origin为zero，输入origin为input_first。保留所有正负与无收益结果，不将两个origin不同真实prefix和树混成同一对照。

输出origin的下一root固定2/1/3/4；输入origin固定2/1/3/3，q161两条原路径不同，不能宣称跨origin位置完全相同。选点来自前批已有结果，有条件选择偏差。若接受数随构造改变，支持该固定状态下的局部因果作用；不能据此分配全480差距的因果份额。没有改prefill本体，不将root0 decode称作prefill替换。

##### 三、执行与停止

使用`--template-control --early-template --reciprocal`，输入origin另加`--input-origin`，两个新目录顺次执行，只有一个串行命令，无后台/重叠队列。准入只查GPU0实际nvidia-smi计算进程和空闲显存，初始≥14GiB、无其他GPU驻留Python即运行；CPU进程不检查、不登记SHA。未知GPU Python、容量不足则保留deferred，不抢占或杀外部作业。

仅魔搭缓存Qwen3-4B BF16/T0/eager/default，不跑8B、FP32模型、测速或算子机理。控制、数学门、来源、非有限、OOM、预算失败均保留并停止，不自动重跑。工具/协议/model/source/raw/private logits绑定SHA；完整权重/cache/token/logits留本机，标量与报告按主题提交、推工作分支、PR合dev；不合main、不强推、不删数据和分支。

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-template-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_TEMPLATE_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_five_methods/docs/MASK_TEMPLATE_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_TEMPLATE_PLAN.md)；原SHA256 `75ab6681098d72304b40f1935f2d4e622aad7723a7f8f01922e6b9910ea92196`。

#### 输入速度模板差分最小对照

2026-10-15，接续已合dev的PR #18。138个保存输入速度与模板初始差分最大相对误差约0.095%，但代数接近不证明候选或接受相同。本次直接检验这个替代解释，不改变默认配方、科学src、旧Run或原PLAN。

##### 一、冻结问题与四格

沿用相同输出零阶真实前缀/cache/当前树和BC30双mask、9节点、Qwen3-4B BF16/T0/eager/default。每格的mean都先用原生输入一阶provider按完整prompt及同一真实已提交前缀更新，λ=.1不变，只在读出mask时替换速度：

| 名称 | 局部mask |
| --- | --- |
| zero | 原输出零阶，复核旧Run续接 |
| input_first | 原输入一阶同历史，β=.9999、γ=1 |
| input_template | 相同已更新mean，速度直接设为prompt前两token的embedding差分，不含后续内容 |
| input_mean | 相同已更新mean，速度置零；不冻结mean，不冒充ESP frozen mean |

这是一次局部速度替换，不把修改后的状态写回原生生成历史；下一验证块仍使用原输出零阶mask控制规则。沿用原工具和树读出，无新框架。模型/forward/mask为BF16，原生输出内部EMA沿用既有实现，不做换精度或舍入机理实验。

##### 二、窗口、指标与证伪

1. 无条件4窗口固定q121/q401/q241/q161，仍取原输出零阶第一个root_index≥8，实际R9/8/8/8。四格共16对照，总forward上限96。
2. 条件2窗口另表q121/q401，原控制第一次下一块仅accepted_nodes=[1]的前驱，实际R3/19。四格共8对照，总forward上限48；不能与无条件样本合算总体胜率。
3. 每格重新读同一当前块的mask logits，再原生建树并验证一个块；报告首/次目标排名、d1/d2宽度、真实下一块接受数、与input_first的完整两行seed logits是否相同。保存完整logits于本机，公开只保留标量和SHA。
4. 若template改变任一窗口接受数，则否定“该窗口冻结模板可替代原生输入速度”的强版本；排名或logits变化但接受相同，仍须保留差异。所有窗口相同也只能支持有限窗口，不能外推全480。
5. input_template相对input_mean的接受差异用于检验固定偏置是否在该窗口有效；不预设固定偏置一定有效，不据此宣称内容速度无价值。控制input_first与前批同位置结果逐项比对，差异即报告并停止追加，不追算子。

##### 三、准入和停止

使用原工具`--template-control`，条件批另加`--active-depth`，分开新输出目录。先提交协议/工具/测试，再非抢占检查；共享GPU忙则保留deferred，绝不杀进程或抢占。权重继续复用ModelScope缓存，绑定source/model/tool/plan/raw SHA。

新出现的RPipe `watch-version-final.py`已读源码核验，仅调用已检查的CPU远端CI查询脚本；准入按脚本SHA `f33abb9086f09bbf917118b20a5fc10cd3c13cca8ec0690b6d0178ef4a399763`识别及其子进程。未知进程仍拒绝。排队最多20次、每次间隔30秒，只重试0-forward的deferred，任何执行失败不自动重跑。

零forward资源停止记录保留后补充核验`watch-version-dev.py`，同样只读远端CI，SHA `b69fd1fe0c67a405afee52be9599cd79cb06459b5abc23b17860ea9014d2ce92`。新版本额外记录未知PID，控制公式、窗口和预算不变；人工确认0-forward后可新目录重新准入，不重用失败目录或自动重试执行失败。

再次延期所见PID 53808/64588对应`watch-publication.py`，已读源码核验仅查询远端PR/CI，SHA `4f9b8c486aff8b93cc18116100e9013290e469fe8ed362639b4ebf790bf2dece`。停止本任务无模型子进程的等待队列后才修订识别并提交，再用未用过的目录续接；不操作外部查询或实验进程。

实际队列停止检查首次因子进程仍在而拒绝，后续新旧队列短暂重叠，目录存在门阻止重复尝试；已取消重复队列，原队列因目录保护退出，全部0-forward，记录不删除。后续只单次串行准入。第16次的PID 34924/63696是量化CPU校准数据准备：已读两个脚本，只有tokenizer/datasets采样及CPU tensor保存，没有模型/CUDA前向；`watch_calibration.py` SHA `dc3f6db96ca5698f19feba0a7e6a23c2f5bd7602bbf29b38cb78c03c8b29e845`，`prepare_official_calibration.py` SHA `669a03a0e26231c9e7fbb0bd1d261c892dbe2f57da1164abd06123f94c324baf`。仅识别已核验CPU作业，不操作该作业。

第17次新增PID 44856/55352对应RPipe `watch-publication-main.py`，已读源码，仍是同一只读CI查询器，SHA `336458e7229ec727348604c9142ec1a63283d2a77f00c81f235f34b24cdb3350`。这只是外部CPU资源识别，不授权或执行外部仓库的main合并。

全部提交token必须等于对应验证前向argmax，当前owner argmax不变，输出零阶控制续接旧Run。独立AR仅旧诊断，不新增AR前向。任一数学门、来源、有限值、预算或控制失败即保存停止；不跑8B、不测速、不删数据、不改默认配方、不钻算子机理。按长期授权正常提交/推工作分支/PR合dev，保留分支，不合main。

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-mask-window-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/MASK_WINDOW_PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_five_methods/docs/MASK_WINDOW_PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/MASK_WINDOW_PLAN.md)；原SHA256 `25a06da4c932f42184fbe092e03938f6b123f45f58193a56266a7c11d9f25bf7`。

#### 4B 输出 mask 固定窗口对照

2026-10-15，接续 Enmao 对输出侧差距的诊断要求。PR #17 的全量深度/位置分解与 12 格只读几何已完成。本协议新增 CPU 向量分解及有限窗口干预，不改原科学源码、配方、PLAN 或 Run，不跑 8B、FP32 模型、测速或算子数值链。

##### 一、CPU 先行

1. 绑定已有 geometry.json、vectors.pt 和原始 Run SHA。对同 owner 的原始槽差 v 做径向/切向分解，计算统一范数映射保留的相对移动量，检验“归一化消掉大部分速度”的具体版本。这里只看原始向量的几何运算，不追踪 BF16 算子。
2. 全 480 条 BC30/60 的三方法按上一块接受深度分组，报告下块 d1 接受率与下一树 seed owner 深度。它是条件统计，不作跨路径因果结论。
3. 对照 decoder：mask 槽位输入对应 owner+1/owner+2，槽位 logits 则预测 owner+2/owner+3。下一树以 deepest accepted owner 的 bonus 为 root。固定当前树时，mask 不参与当前 owner 的验证注意力；不能把当前 mask 改动误算为当前接受率收益。

##### 二、GPU 最小固定窗口

沿用预先选定的 q121/q401/q241/q161，各从已完成输出零阶 Run 的原设置重建前缀，只取第一个 root_index >=8 的验证块，最多重建到 32 新 token。选择与接受结果无关。4 个上下文、5 种输出局部干预加输入一阶同历史控制，共24格，最多 128 次总模型 forward，不生成完整轨迹或矩阵。输入控制用于直接核对同一窗口内的两端差距，不以不同原生成轨迹冒充控制。

同一个输出零阶真实前缀、cache、当前树、真实已验证 hidden 历史和代理观测上，并行维护 order1 shadow state（β=.9、β′=.99），不把 shadow 预测写入真实历史。局部 2×2：速度从槽0还是槽1开始、逐槽固定半径还是保留槽间径向幅度。另设原输出零阶对照。位移系数固定为 1，不搜索参数。

| 名称 | 预测槽（j=0,1） | 范数映射 |
| --- | --- | --- |
| zero | 原 order0 anchor | 原 r·a/范数(a) |
| input_first | 原 β=.9999、λ=.1 mean+gamma，gamma=1，按相同输出零阶真实 token 历史更新输入 provider | 原输入配方，不缩放 |
| first | 原 order1 a+j·v | 原逐槽固定半径 |
| shifted | a+(j+1)·v | 原逐槽固定半径 |
| radial | 原 order1 a+j·v | 统一用本 owner 的原 a 范数作分母 |
| both | a+(j+1)·v | 统一用本 owner 的原 a 范数作分母 |

order1 的 a/v 由原生 shadow.masks_per_owner 对同一代理作纯推演。深度0/1令 a=对应代理，深度2沿原规则 a+=v；只改变表中局部槽公式与映射。保持原 BF16/T0/eager/默认归约和双 mask BC30/9 节点。

每格先在同一当前块/cache 上前向，核对 root/真实接受 owner 的 argmax 与原控制完全一致，且控制重放与原输出零阶 Run 对齐。随后用 deepest owner 的两行 mask logits 原生 build_tree 生成下一树，记录目标 token 的原始 rank、剪枝/预算后是否入树、d1/d2 宽度。目标来自已有输出零阶轨迹，只是诊断，不是 teacher 强制纠正。

再用该格下一树和同一真实前缀 cache 做一个原生验证块，mask 内容使用相同控制规则，只读 owner logits 的 argmax 路径。报告实际接受深度、数学 argmax 自检、与原控制及旧轨迹是否一致。每个干预仅两次局部前向；不沿干预结果继续生成。独立 AR 不新增 forward、不作门。

准入已核验的 CPU 准备作业包括量化容器 build_v3（无 Docker GPU/device 参数，SHA bc206af76763f1364fe2565be1be8cf2294342aaa414f3b263cbf115dafcceec）、RPipe watch-first/second 的远端 CI 查询及其子进程。工具按实际脚本 SHA 检查并记录计数，未知作业继续拒绝。检查只针对资源准入，不运行或修改这些外部脚本。

##### 三、准入、停止与证据

复用已核验的 ModelScope Qwen3-4B，加载前两次非抢占检查和内容 SHA 检查，每个上下文再查。若未知 GPU/模型作业存在或初始可用显存不足18 GiB，保留 deferred，不抢占或杀其他作业。此为观测准入，不冒称全局原子锁。

模型/source/输入不符、控制前缀或当前 owner argmax 不符、forward 超128、非有限或 OOM 则保留证据并停止，不切精度、不追算子分叉。未获准就交付 CPU 结果与明确几何缺口，不编造实验。

正式报告和可公开标量放 docs/research，完整 cache/权重/logits/token IDs 留本机 .tmp。保留全部零收益/负收益。固定窗口可区分局部作用，不能说明全 480 条 τ 会改善。工具、协议、模型、来源与原始证据绑定 SHA，按主题提交、PR 合 dev，保留分支和数据。

</details>

<a id="legacy-studies-qwen3-4b-bf16-five-methods-docs-plan"></a>

### studies/qwen3_4b_bf16_five_methods/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/qwen3_4b_bf16_five_methods/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/qwen3_4b_bf16_five_methods/docs/PLAN.md)；原SHA256 `e0cf97834d9e095585fc941ac7dd76a6ca9b7712bde1cc0c6440b750cb617816`。

#### Qwen3-4B BF16 five-method paired comparison

状态：执行协议与配对阅读器就绪，owner映射/BF16范数验收及CPU回归已完成；本PLAN作为首次make的冻结候选。实际make/运行状态另记STUDY_REPORT，冻结后不回写本PLAN。前置ESP Table1四格全480、argmax与原生process已完成；PR #10经CI合dev（fafed614）。论文τ目标未达到，完整负结果保留。本Study不再使用8B，不恢复算子机理。

##### 问题与固定范围

检验4B/BF16下输出零阶和一阶的接受收益，并在同一批480条SpecBench首轮prompt、同一设置下比较五方法加独立AR。不能用旧4B FP32或26条链式结果冒充本轮。

主预算BC30，另预先固定BC60复核；两者均用两个mask、naive动态树和重复剪枝（节点预算9/19），不按结果选预算。选择两mask是延续既有4B输出零/一阶定义，使第二mask保留显式速度外推；它不是8B Table1单mask主协议的新复现。所有五方法使用相同树/预算/长度/种子，不能给输出方法换成更窄单链。

| 方法 | 固定实现与参数 |
| --- | --- |
| ESP frozen mean | prompt embedding mean，update=false；用户指定基线，区别于论文updating mean |
| 输入零阶 | prompt mean初始化，embedding EMA λ=.1；直接复用ESP updating mean |
| 输入一阶 | mean_plus_gamma原配方：level=order0、λ=.1、β=.9999、γ=1、normalize=false、extrapolate_ema=false |
| 输出零阶 | final-normalized hidden，order=0，按r=mean_v norm(W_v)缩放；保留既有代理状态定义 |
| 输出一阶 | 同一hidden映射，order=1，已验证历史β=.9、代理历史β′=.99；沿用既有4B配方，不事后调参 |
| 独立AR | 同一模型/输入/seed/停止规则，仅执行一次480条，供两个预算全部方法离线配对 |

共11个Run：1个AR，5方法×2预算。Spec Run关闭重复内部AR；完整输出与独立AR逐token对比在统一阅读器完成，保留所有首分叉。没有额外teacher、warmup或速度基准。

##### 模型、输入与精度

Qwen/Qwen3-4B，优先复用ModelScope既有检查点；冻结实际文件SHA、ModelScope可核验来源、model digest与实际tied-storage指针检查。不得用8B或未确认权重代替。权重、forward及输入mask为BF16，eager、默认归约、deterministic=false；不做FP32模型实验。复用原生内部计算：输出EMA/范数映射保留FP32状态，输入EMA保留其原生dtype，最终mask转模型BF16；明确记录，不能称所有内部算子均为BF16，不另开精度对照。

数据沿用Table1官方SpecBench固定480条及原顺序，skip=0/take=all、首轮、non-thinking chat template、max_prompt_len=40960，不截断/丢样本。用4B tokenizer生成并冻结prompt manifest，核对qid/category/顺序与文本来源；不能未经核验直接复制8B token IDs。T0、seed123（逐prompt 123+index）、max_new_tokens256、EOS停止、warmup=false；所有Run设置相同。

##### 开跑前必须完成

1. 输出provider仍生成原k+1个深度代理状态，按[0]+tree.depths映射至实际root/候选owner；同深度兄弟共享同一已有代理预测。forward后仅真实root/accepted路径更新已验证hidden，bonus延后，拒绝分支不写历史；不增加forward，不改候选/树算法或默认配方。
2. 原生验收按实际配置的树规模核对owner数，按模型dtype的量化精度核对实际BF16 mask范数，保留非有限/零范数和篡改拒绝。提交token的对应forward argmax门不得放宽。
3. 回归覆盖多兄弟/多深度、动态剪枝、被接受的非首兄弟、EOS/预算裁剪、mask state不被代理污染、旧单链行为及实际BF16原生CLI验收；检查所有调用者、证据验证和文档。
4. 校验模型/数据、source pin及无测速适配器，完成来源manifest后再冻结本PLAN与recipe/配置。源码或输入变化必须新版本/新Run，已有成功数字不重跑或重签。

##### 指标、正确性与停止规则

按qid和实际输入/模型/seed配对，逐prompt报告accepted（裁剪前）、committed/new_tokens、主model calls（prefill+decode）、pooled与prompt τ、独立AR exact和首分叉。对五方法所有两两对照分别报告accepted较高和calls较低的赢/平/输，另列相对frozen mean、updating mean及output1-output0；两个指标不强行合成单一胜负，不拿不同输出长度隐藏EOS影响。两个预算的正负/平局全部保留，单seed不声称总体优越或测速收益。

正确性门：全部实际提交token等于对应验证forward logits的argmax，包含prefill首token且覆盖输出全长；独立AR逐token一致率仅诊断，不阻止收齐矩阵，不触发BF16算子机理。来源/配置/计数/cache/owner/范数或argmax证据失败、OOM/非有限立即保留失败格并处理真实问题，不能切FP32、teacher纠正、删除反例或以局部成功替代全量。

##### 排班、产物与验收

共用单张24GB GPU，一次只开一条round1/num-gpus1调度链，先独立AR，再BC30五方法、BC60五方法，组间确认进程退出。开跑前核对其他GPU任务，不能杀其他任务；无预热、无decode计时、无latency/throughput结论，原生计时字段仅零占位。

报告需包含每个Run真实状态、实际dtype/参数/输入与来源SHA、完整逐prompt赢平输和AR差异、argmax覆盖及失败边界；原始tokens/Run/权重/cache留本机。正常工作分支提交/推送、PR和合dev按长期授权执行，不合main、不强推、不删除数据/分支、不花钱。

##### 首次make前预检

模型内容14文件共8,060,927,256字节，全部匹配魔搭直接SHA256及此前4B FP32检查点；内容已核验，不称历史下载来源或immutable revision已确认。离线snapshot_download成功，缓存revision身份警告保留。4B tokenizer实际生成全部480条，qid/category/顺序及token IDs与Table1输入一致，最长1500，不截断。模型配置声明qwen3/2560 hidden/36层/151936 vocab/tied=true；实际GPU存储共享仍由每Run首forward前的adapter检查，不能拿配置声明代替实测。

已完成CPU原生六Run tiny BF16检查、全套CPU兼容性与mypy56文件，以及新Study九项协议/恢复/loader和phase导出回归。11格配置已用原生expander核对；此处不宣称真实4B结果。配对阅读器复用原生verify_evidence、source/Study冻结门及输入manifest检查，每Run核对实际tied-storage、dtype、model digest和runtime；逐prompt保留全部20组两两对照的接受/calls赢平输、输出长度和AR首分叉。仅11格与native process全部完成才生成不可覆盖的正式paired_evidence_v1.json，未完成格不补零，不以pending冒充结果。新增回归覆盖正负平、调用数反向判赢、缺失/错位拒绝、AR长度分叉、pending完整性及已验收证据的读出格式。

</details>

<a id="legacy-studies-smoke-docs-plan"></a>

### studies/smoke/docs/PLAN.md

<details>
<summary>历史计划全文（原文件已归并）</summary>

来源：[studies/smoke/docs/PLAN.md](https://github.com/DreamSoul-AI/denoising-speculative-decoding/blob/aed4eee23ea5e6b9772b388eec9aa50a38452089/studies/smoke/docs/PLAN.md)；原SHA256 `9b61ffa71abcea1e1d923d93639652a35bb9e20b6a38c62bbd5c01c1e7666560`。

#### Study Plan: smoke

##### 1. 目标与范围

固定 RPipe v0.3.0 (`804b3823ab6d2af46cbda1a65688ca8b76c60b22`)，验证随机 tiny 的 losslessness 和实际六阶段证据门，保持 ESP/EMA/AR/alignment 的 training-free 语义。历史验收见 [STUDY_REPORT](../studies/smoke/docs/STUDY_REPORT.md)，下面的复跑合同不把旧成功签收为新结果，不启动真实模型矩阵。

##### 2. Study / Experiment / Run

`experiment_config.yaml` 是基底，`fixed` 覆盖同名条件，唯一轴为 `algorithm.config.variant = [esp_static, esp_dynamic, ema_velocity]`。三个 Experiment × seeds=[123] 为三个 Run。无独立 Experiment 文件夹，当前 index 是唯一终验集合。

声明中的 `rpipe-v030-20261013-delivery` 是历史交付版本。新实测复制到隔离 `.tmp/` 目录，改 Study 名与 `fixed.algorithm.config.version`，核对实际 config/index 后 make/launch；不得追加写入旧 Run。

##### 3. 固定条件与数值门

| **项** | **原配方 / 验收条件** |
| --- | --- |
| 模型 | 随机 tiny LLaMA，vocab=256、hidden=64、intermediate=128、layers=2、heads=4、KV heads=2、position limit=1024 |
| 输入 | synthetic，四个长度 32 的 prompt，data seed=0；模型/Run seed=123 |
| 计算 | CPU，eager，随机 tiny 的实际 dtype float32；记录 TF32、matmul 精度与依赖，不自行更改 |
| 解码 | 三个既有 variant 的 mask/树/EMA 配方保持，32 新 token、T=0、stop_on_eos=false、AR 与逐 prompt 证据开启 |
| 质量 | 每个 Run exact_match_rate=1；完整 token、接受/回退、KV cache 和输入/权重/runtime 来源可复验 |
| 数字 | 旧 tiny accept_d1=27/31/8；对应 all-depth accepted=28/31/8；深度统计和总接受口径分开 |

上述固定计数仅适用于记录的模型/input/依赖/设备条件。同设备旧新数值比较保留原始 token/完整轨迹；改变环境后重新建立参考，不能用调温度、FP32 化真实模型或挑选重复来消除反例。τ 与速度不是质量门，smoke 不做性能排名。

##### 4. 阶段、来源和失败边界

prepare 通过 recipe 和 builder 检查来源/输入并构造；execute 保存真实观测；collect 计算指标；summarize 拒绝 exact-match 和证据失败；write 复验后由 RPipe 定稿；process 的 Run scope 用默认职责，Study scope 只验当前 index 的因素/seed、来源和质量。

覆盖 source 新增/删除/修改、输入/配置漂移、raw 缺失、真实 token mismatch、不完整子集/seed、混合来源和 Study scope。安装验证分别说明 wheel/sdist/console scripts/实际 CLI 链，不能只报 import 或退出码。专项包锁与 RPipe provenance 配合，细则见 [指南](../studies/README.md#4-来源冻结与复跑)。

##### 5. 排班与命令

三个 CPU Run 串行，make 使用 `--round 1`，使用现有科学依赖和 `OMP_NUM_THREADS=1` 的小规模验证条件；不叠加资源竞争的测试或计时任务。新目录确认后使用 `python -m spec_mtp make <copy> --round 1`、`launch <copy>`，再 process/status/logs/report。运行前记录新 version、Run ID 和输入；失败及旧产物保留。

| **排班** | **任务** | **est / actual** |
| --- | --- | --- |
| 串行，三个独立 Run | 每个 variant × seed=123 | 新复跑在 make 后填 est，执行后填 actual；历史证据未保留统一墙钟，不补造 |

CUDA 仅另行授权的随机 tiny 串行门；实际 dtype、TF32、attention 与输入必须按原配方记录。没有真实模型容量测量前不增加 GPU 并发；不会将训练 `resume: latest` 当作解码恢复。

##### 6. 成功与报告

所有声明组合的 Run succeeded、exact-match=1、证据/来源一致且 Study `complete` 与质量门通过，才能报告完整通过。报告按三个 Experiment 给指标与 n=1 的统计边界，附真实历史结果图和独立 Run/失败引用，保留静态/CPU/GPU/安装的区分。正式参考 JSON、图与报告随 Git；原始 runs/shared/.tmp 日志只在本机。

##### 7. 历史迁移计划（2026-10-13 原记录）

以下保存当时的迁移执行计划；其中初始 version 后续使用 final03/delivery 等新版本复验，实际提交、Run、失败和数字以报告/JSON 为准。


目标：固定 RPipe 804b3823，保留 ESP/EMA/AR/alignment 的 training-free 语义，验证实际六阶段、来源冻结与拒绝成功边界。非大模型研究矩阵，不重跑历史正式接受数。

1. 同一设备和依赖上保存 c3263c0 工作区改动前的 555 格 fingerprint，再与当前来源对照；固定 tiny LLaMA、prompt、temperature 和 seed 不变。
2. 导出旧链 b1f6191 的 src，使用 [validate_migration.py](../studies/smoke/validate_migration.py) 比较三种 smoke variant × T={0,0.7,1} × 四个固定 prompt。被动 Python line trace 读取接受路径/cache，不改旧源码；比较完整 token、接受/回退与 cache 轨迹。T=0 历史 tiny accept_d1 为 27/31/8。
3. 新 version rpipe-v030-20261013 走 make→launch→process/status/logs/report；AR/alignment 与失败边界由真实 CLI 回归验证。CPU 串行，不做吞吐排名。
4. wheel/sdist 各在源码目录外安装至隔离 .tmp，复用已有科学依赖，核对 import、console script、ORIGIN/LICENSE 和实际 tiny 阶段链。
5. Windows 双导入入口验证；CUDA 只允许随机 tiny 三 variant 串行门，不下载真实模型，不扩大并发，不宣称大模型/恢复已验收。

所有新实测采用新 Run ID/目录；失败日志留 .tmp（不会随 clone 提供）。正式结果和条件记 [STUDY_REPORT.md](../studies/smoke/docs/STUDY_REPORT.md)。执行有不一致时保留反例，不放宽阈值。

</details>


<a id="norm-continuation-20261016"></a>
### 2026-10-16 范数首格恢复与七格续跑登记

首格 template/0.5 的 480 条 execute 已完整结束，Run `6ee3f749ee20c450` 因 Study collect 包未导出 run 而失败；调用同一原生 collector 并用既有 revalidate_run 工具恢复验收，80,940 个提交 token 的 verifier argmax 门通过，原始文件 SHA 全部不变，不重跑 execute。失败 result 与恢复证据保留于 `.tmp/norm-collect-recovery-20261016/`。这不是 AR exact 门放宽。

旧 v2-capacity 的冻结入口不改、不重签。剩余七格采用独立 continuation Study/新 version，仅移除已完成首格，原顺序、SpecBench480、prompt token IDs、4B BF16/T0、BC30 双 mask、种子123、范数倍率和全部科学实现不变；复用原 NormEval，修正阶段包导出并使用19:53显示进程白名单分类。CPU 汇总逐格绑定旧首格或新七格来源与数学门，不把旧 Study 七个未执行 Run 记为成功。对照仍复用零/一倍端点；不重复任何已有格。新增入口回归覆盖四个 phase 包及 callable。

当前8B BC60 `a8e6bba1dd1d80e2` 已自然通过原 RAM 门并在运行，优先完成，不暂停、不重复开。续跑七格在它结束后单队列接续；准入保留实测 host peak+2GiB 与实际空闲显存，图形/远程桌面不阻塞、不杀进程。完成后统一报告三种 τ、配对胜平负、来源、入口故障及恢复边界，攒到完整里程碑 PR。


续跑准备实况：七格原生make通过，新首格d288934359a7ede1。单一等待队列60480在BC60自然退出前不载模，只读入账器78720已确认原首格成功；失败旧watcher输出另存保留。四阶段native dispatcher和来源映射正/负回归通过，Ruff/mypy通过，完整套件仅两项工具文档门失败，补齐后89项定向检查通过。首格三种τ与配对详情见[续跑报告](../studies/qwen3_4b_bf16_bias_norm_continuation/docs/STUDY_REPORT.md)。本地/远端仅main、dev和仍载BC60的已合活动分支；本工作分支未合，均保留，无新增删除。

V4 operational update: documentation-only helper SHA change; superseded unexecuted seven IDs retained with metadata backup. Current first Run `6719b3ae2c089133`, serial waiting queue `64880`, read-only ledger `50944`. Earlier60480/78720 CPU processes exited before replacement; BC60 was never interrupted. Native make/source bindings renewed only for the new version, not old completed evidence.
