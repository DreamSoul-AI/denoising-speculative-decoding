**去噪推测解码 研究计划**
更新日期：2026年11月6日。唯一当前研究计划，包含当前成果与结论。按 Owner 最新方向，先核验已有 Kalman-filter 与 EMA 是否正确、可用，再尽快收束现有方案与历史 Run；工程整理服务于这条研究主线。现有方案的核验不以等待新算法为前置。工程实施见 [开发计划](<开发计划 Development Plan 去噪推测解码.md>)。

# 一、研究目标与优先级
核心问题是：在固定模型、输入和验证协议下，已有 EMA 或 Kalman 类状态估计是否正确使用可获得的信息，能否相对可信基线提高多 token 接受效率，并转化为实际计算成本收益？本阶段核验和收束已有方案，不另开算法方向。

1. **先验证正确性与可用性**：辨明 EMA、各具体 Kalman 递推与静态仿射 K，核对状态归属、时间对齐、重置和数值边界，先排除实现错误与不可比协议。
2. **尽快给现有方案结论**：利用固定来源，分开实现正确、接受效率有效、实际加速和证据不足。失败方案按已验证范围关闭，未完成 Run 如实登记，不无限扩展搜索。
3. **当前队列只保最高价值工作**：EMA 既有正结果的适用范围、Kalman 关键正确性与未定比较、可信 ESP 基线及必要证据工具。低后续投入价值支线进入 [研究成果归档](<研究成果归档 Research Archive 去噪推测解码.md>)，保留负结果与来源，不删除旧 Run 或改签完成状态。

文献基础与阅读范围沿用 [文献调研](<文献调研 Literature Review 去噪推测解码.md>)，机制和代价见 [通用调研](<通用调研 General Research 去噪推测解码.md>)。本次未新增文献检索；真实CUDA试跑按新Run单列。

# 二、当前成果与结论
下表区分固定历史结果与本次新Run。历史行未重新执行原Run或重签原生审计；本次真实CUDA试跑单列，按新源码/模型/输入/协议及完整提交链验收，不冒充旧480条复验。

| **方案与范围** | **已有结果** | **当前结论与处置** |
| --- | --- | --- |
| 输入 EMA，Qwen3-4B BF16 五方法固定480条，seed123，T0，256/EOS，BC30/60 | input_first 为以order0 EMA作anchor的一阶velocity外推（method=ema_velocity、level=order0），β=.9999、λ=.1，无verify/Kalman；pooled τ 为1.511957584/1.636339710，高于 input_zero 的1.422420206/1.519108823；配对 accepted 胜:负364:81、374:70。11个历史Run已完成，10个speculative Run的809,118个提交token argmax审计通过，计时关闭 | **已有可运行实现及该协议内接受效率优势**。保留优先基线；单seed、不同后续轨迹及无计时不能证明普遍优势、纯空间因果或加速。[历史证据归位](<运行记录 Run Record/2026-10-31 历史证据归位.md>) |
| 本次Qwen3-4B真实CUDA试跑，SpecBench qid81–88、32 token/noEOS、naive K2/BC30 | 两臂16项、512个提交token来源/T0审计全过，8对输出token均相同。accepted总数ESP113、EMA86，EMA胜:负:平0:8:0，配对均差−3.375；同步解码墙钟均值ESP0.680310725秒、EMA0.810950813秒，配对均差+0.130640088秒 | **本次预设联合假设不获支持**：8条均更少接受且更慢。只关闭这次有界试跑，不抹掉旧480条局部正结果；本次只取writing子集，输出预算与停止条件不同，不构成旧完整480条的直接复验、独立确认或普遍速度结论。[本次运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) |
| 合成 EMA 与 observation 联合，确认seed21/22/23，192 prompt对 | 相对较弱EMA trajectory为102:71，p=.022；相对更强ESP为88:94，p=.71 | 已在合成预定主比较中确认优于EMA自身，但未优于更强ESP，不作为超越强单侧的成功方案。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>) |
| 标量trajectory-only Kalman，合成发现与确认分离 | 发现117:59:16，p=1.5e-5；确认101:83:8，p=.210 | **收益未获独立确认**，该配方停止扩展；不推广为所有Kalman模型无效。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>) |
| OnlineVerifyKalman，合成及真实模型CUDA BF16/eager | 合成确认97:86:9，p=.460；真实Qwen3-4B/0.6B各32条、16 token、k1[4]，argmax通过，但accepted胜:负:平2:24:6 / 1:28:3，τ从1.361→1.218 / 1.452→1.216 | **该协议内明确负结果**，不以同配方扩大实验；执行正确性与收益失败分列。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>) |
| AlignedResidualKalman，真实4B/0.6B小规模比较 | 两模型均5:9:18，p=.424；τ分别1.306→1.294、1.335→1.327 | 未超过ESP，作为有价值负证据收束，残差对齐未证明收益。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>) |
| 8B历史命名frozen-Kalman（实际冻结对角K残差修正），office-qwen3-8b-v1，128条Dolly、64 token、T0/noEOS、k1[4] | 对ESP为51:41:36，p=.348141；accepted配对均差+.1953125，CI[-.0546875,.4453125]；pooled τ 1.343552→1.350754，argmax通过，未测速度 | **冻结校正收益未获确认**。固定原源码为hybrid_gain、mode=frozen、observe=last；校准后无在线K/P更新，不能支持真正在线滤波有效。只作为既有静态校正对照收束。[原记录](<运行记录 Run Record/2026-11-03 Qwen3-8B同预算与冻结卡尔曼.md>) |
| 静态ridge/仿射残差K与k_affine_residual | 旧合成确认124:58:10，p=1.1e-6；真实0.6B 128条50:40:38，p=.343，未获收益。当前postcommit-final-norm-v1仅为合成工程夹具，160对数据与真实收益未执行 | **静态拟合正结果不证明Kalman有效**。保留必要诊断对照，不优先扩新矩阵。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>)、[实际声明](../studies/k_affine_residual/study.yaml) |
| 输出hidden hybrid Kalman | 4B两项p=.448/.562，0.6B τ 1.485→1.279；root MSE改善未转化为接受收益 | 按原协议关闭负消融，退出当前实验队列，历史记录归档。[开发记录及运行入口](<开发记录 Development Record/2026-11-02 输出hidden混合卡尔曼.md>) |
| Qwen3-8B observation扫描及按长度墙钟 | 均为**ESP**，不是EMA/Kalman。扫描同128条探索数据，g=.5相对g=0 accepted均差−13.1796875，p=6.388e-29，平均墙钟+.473827秒；另一次按长度比较trajectory相对full在64/256/512 token平均少.096467/.474111/1.148640秒 | 这些条件下加入该观测不划算；不把同批探索的配置选择当独立确认，不把收益归给EMA/Kalman。[扫描](<运行记录 Run Record/2026-11-04 Qwen3-8B observation增益扫描.md>)、[墙钟](<运行记录 Run Record/2026-11-04 Qwen3-8B 墙钟按长度.md>) |
| ESP非官方8B固定480条复验 | BC30/60 pooled τ 1.563377504/1.647301264，未达论文目标1.63/1.74；完整提交链239,137 token argmax审计通过 | 保留可信的**本仓协议基线**，论文目标未复现，不为追数字重开长矩阵。[历史证据归位](<运行记录 Run Record/2026-10-31 历史证据归位.md>) |

EMA五方法原件固定于Git b8b20b5ada7fd860894114cbf224ce749af430cf的 studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json，SHA256为8e85ac6ea9984047abf0ffd5d6653b28b6aadd74aa876139fa10465dc5bbf274。本拍核到原件字节，不等于重新核验其全部原生Run。8B frozen身份另据固定原HEAD 29fc9f815fa83aae72272f4e96a831425d3f5d8c的studies/trajectory_observation/structure/office_qwen8b.py与src/spec_mtp/masks.py核对：离线校准对角K后不作在线K/P更新，观测z仍随提交变化，不表示输出mask恒定。早期first26调参结果仅作发现线索，不充当独立确认。

**当前总判断**：EMA已有局部可行和接受收益证据，速度与外推范围仍未定；Kalman类实现确实存在，但多个已测配方为负或未确认，8B历史命名冻结变体实为静态K修正，收益也未获确认。先查清实现和协议是否忠实，再逐方案结案；名称、静态K拟合或工程测试不能替代研究判断。

# 三、先核验机制，再作方案结案
## （一）方案身份与数学合同
实现见 [providers.py](../src/spec_mtp/structure/algorithm/mask/providers.py) 与 [fit.py](../src/spec_mtp/structure/algorithm/mask/fit.py)，分别核验，不能混称一个Kalman方法。

| **实现** | **需核验的合同** | **可得结论边界** |
| --- | --- | --- |
| EMA provider | v=βv+(1−β)(e_new−e_prev)，anchor/order0、步长外推、范数、block-local与实际β/λ | 平滑与外推是否按协议执行；没有协方差模型，不据此声称Kalman最优性 |
| KalmanMaskProvider | 标量随机游走P⁻=P+Q、K=P⁻/(P⁻+R)、level与Joseph后验、噪声自适应/floor、冻结/reset | 递推是否忠实且数值有效；共享标量噪声假设另作研究判断 |
| OnlineVerifyKalman / AlignedResidualKalman | OnlineVerify逐维P/Q/R；Aligned标量F/P/Q/R与F裁剪；观测和提交token时序 | 排除错位、未来信息和跨prompt泄漏，不等于完整embedding协方差模型 |
| HybridGainKalman / OutputHiddenHybridKalman | 对角K参数递推、x/innovation、校准先验、在线/冻结/scratch/bonus与root对齐 | 实际变体与Run逐一对应，不同状态语义不共享收益结论 |
| 静态scalar/diag/affine/ridge K | 拟合输入/目标、隔离单位、正则与冻结时点 | 只回答回归校正合同，不证明Kalman滤波有效 |

逐方案列状态、观测、F/H（适用时）、P/Q/R/K、更新时点、owner/bonus槽位、重置与冻结条件，以独立参考式核对。未满足线性/噪声等假设时明确模型限制，递推存在不等于最优性成立。本轮CUDA已复现并修复13个方差/噪声字段接纳NaN/Inf，以及OnlineVerify/Aligned在合法forget=1下除零。该端点沿已有scalar语义关闭Q/R自适应（包括初值低于floor时保持），P与F仍更新；这是显式no-adaptation约定，不是自适应权重在forget→1时的连续极限。旧finite且forget=.95的研究配置不因这次端点修复自动改签或失效。

## （二）当前P0验证范围
本轮以CUDA FP64独立参考核EMA闭式加权历史与last/order0 anchor，标量Kalman的Joseph后验、自适应floor/冻结，以及OnlineVerify逐维和Aligned标量递推。forget=1单独检查Q/R保持、P/F继续更新，包含合法below-floor初值；重初始化清除所测状态。Hybrid按独立回归更新式区分first、last与frozen，冻结K/P不冻结观测或输出mask。测试与实际设备、版本从本轮运行记录定位，范围限声明输入与模式，不等于完整状态空间证明。

时序边界：OnlineVerify和Aligned按每轮root prior的**首个提交**更新一次；后续同轮commit不新增该后验，OnlineVerify的z因此保首commit，不是末bonus。Hybrid first/last分别在首commit/finish更新，last使用最后实际提交，EOS或预算截断时不保证它是完整轮bonus。真实CUDA解码回归覆盖prefill、逐commit、截断/EOS、finish与完整T0链。保留这些旧语义，不直接改成另一算法；递推吻合并不证明状态空间/噪声假设成立、滤波最优或研究收益。

## （三）按信息价值验证
| **优先级** | **行动与预算边界** | **验收与停止** |
| --- | --- | --- |
| P0 实现与身份 | 对齐源码、固定配置与Run；CUDA小张量独立参考验证更新、冻结/reset、有限值与因果顺序，复用既有测试层 | 记录实际tensor/device、dtype/backend及来源。差异定位到合同；发现错误先修复并新Run验证，旧结果不重签 |
| P1 历史Run收束 | 逐项核第二章最高价值来源与完成度，保原样本、对照、单位、计数和失败判据；先读原件，不重跑成功项 | 每方案分列正确性已验/待验、收益支持/不支持/不足、速度已测/未测、Run完整/部分/未知及来源；负结果也可结案 |
| P2 只补影响决策的缺口 | P0通过后如缺执行证据，先选已有缓存模型、一个既定配置、两臂、最多8个冻结prompt×32新token，单项CUDA诊断预算上界5分钟 | 这是准入诊断而非收益确认；T0门/状态/来源一致才算可运行。超预算记partial，不扩表、不退CPU |
| P3 必要的收益判定 | 仅当未定结果会改变保留/关闭决定时，为该既有方案冻结最小比较：最强可信基线、隔离确认数据、主指标、样本单位、区间/检验、多重比较、时长与总预算 | 明确选择先写回本文对应课题再执行；不事后缩样本、改门或挑臂求通过。证据不足记未能判断，不无限续跑 |

运行前确认torch.cuda.is_available()为真，日志写实际CUDA设备、dtype/backend、环境与资源占用；GPU忙时等待或缩小规模。CPU仅用于快速纯合同/IO单元检查。复用本地模型来源清单，不下载模型或覆盖CUDA torch。新执行以新version/Run冻结源码、输入、配置和模型，日志与制品留显式外部根。

# 四、重点课题与保留决策
## （一）EMA：核清既有收益边界
既有主张是：固定协议下EMA相对input_zero或可信ESP基线提高接受效率。先核正证据实际β、λ、anchor、更新顺序、表精度及数据身份，分开first26调参与固定480条比较。已有结果支持已测条件下可用，不能单靠τ推断速度。

当前有界试跑协议为ema-esp-4b-cuda-admission-v1：唯一假设是在Qwen3-4B BF16/eager CUDA、SpecBench原始JSONL前8条（qid81–88，首轮输入）、32新token且不因EOS停止时，既有EMA相对ESP提高接受数并减少同步generate墙钟。两臂均用K2/dynamic/pruned/BC30/naive，EMA保β=.9999、λ=.1、order0、step_scale=1、normalize=false、extrapolate_ema=false、update=true；ESP为mean_prompt、λ=.1、update=true。逐qid交替臂顺序，两次2-token预热不计入配对；生成前后CUDA同步，加载/哈希/输入渲染另记，不纳入解码墙钟。模型完整来源清单、渲染输入与token、配置、当前源码及临时执行脚本冻结到新外部Run；GPU诊断上界300秒，超时保partial。

验收先要求16项全部完成、完整来源首尾校验和T0提交链审计通过，再判本次8对的accepted均差（EMA−ESP）>0且同步解码墙钟均差<0。任一方向不满足即本次假设不获支持；缺项或质量失败不作收益判定。不以8条短输出证明普遍加速或复现旧SpecBench固定480结果，不加臂、不选优重跑、不扩Study。Kalman observation-only是另一个GPU覆盖缺口，本拍不随这项EMA试跑接卡。

该试跑已完整执行，来源与T0门通过，两个效果方向均失败，按预先规则结为“本次假设不获支持”。不加臂或改β求正结果。accept计数与本次实际接受前缀相符；同步墙钟含provider构造和generate内存记录，加载、哈希、渲染、独立审计及落盘另计。

| **待判结论** | **已有GPU证据** | **下一项GPU安排与边界** |
| --- | --- | --- |
| Kalman所测递推/时序是否按声明执行 | full模式独立参考、first/last/frozen及真实loop均已核；native CUDA smoke12项通过 | 下一拍优先OnlineVerify/Aligned的observation-only四场景（forget=.8/1、首commit与reset），预计小于30秒；现有临时脚本须重绑实际HEAD/源摘要后运行，不自动接旧矩阵 |
| EMA在不同输出预算下是否仍具接受/速度优势 | 旧480条只有保存的GPU接受正证据；本次8×32真实GPU为负，不能声称普遍收益 | 后续候选仅固定相同8条/两臂，把长度改256以隔离预算因素，预算300秒，先在下一拍冻结一个假设；未执行，不将旧记录当计时证据 |
| 在线Kalman的真实模型收益是否可确认 | OnlineVerify/Aligned已有记录级真实GPU负结果；8B frozen实际为静态K且收益未确认 | 不重复扩样同一负配方。现行原生重验与来源完整度仍缺，只有材料/实现变化会改变结案决定时才另设最小GPU验证 |

滤波模型假设与最优性还需相应推导和可识别性论证，不能用补跑GPU消除这一理论缺口。以上队列不是并行新Study，本拍只执行已冻结的8×32假设。


若材料不足以收束实用性，必要比较保同模型、输出协议、树/预算与资源，分开accepted、τ、完整时长和状态成本。已观察过的SpecBench固定480及先前选过配置属于探索/选择材料，不改标独立留出。接受收益而无墙钟收益时明确结为接受有效、速度未受益，不继续扩大搜索。

## （二）Kalman：逐配方验真
先完成第三章数学和时序核验，再按第二章结案。OnlineVerify、aligned residual、output-hidden等已测负结果不重复扩样；若真实实现错误使旧比较无法回答问题，保留旧记录并明确原因，新验证单独冻结。8B frozen按静态对角K修正结案，不作为在线滤波正证据；其校准/评价隔离与来源仍须完整登记。没有信息增益就保留未获确认并停止投入。

静态K=[A,b]与postcommit-final-norm-v1只作拟合改善/解码受益的诊断对照。160对与跨集真实收益未执行，不能写成Kalman成功，不因夹具可运行就启动矩阵。

## （三）Study取舍
当前主动队列只保EMA、已有Kalman核验/结案、可信ESP基线及必要证据支撑。五方法/input-mask材料服务EMA身份和对照；trajectory/observation探索约束解释，不作为独立扩展队列。输出hidden、失败top1/bonus和已停止BF16机理支线进入 [研究成果归档](<研究成果归档 Research Archive 去噪推测解码.md>)。

七格范数、11格独立配对、旧72项模型矩阵、旧长矩阵和原1500步训练不自动恢复，退出当前排程；旧pending/partial保持真实状态，不伪造全Run完成。特定材料直接影响结案时只读所需来源或按第三章登记最小验证，不整批接卡。模板、smoke、来源校验和测试基础设施是工程支持，不登记为算法收益Study。

本拍完成研究队列与记录归属清理。Study源码、第三方来源、数据、旧Run及source lock保留；物理删除/归并前核真实消费者和取证依赖。归档基于低后续投入价值，负结果本身仍是有价值证据。

# 五、历史Run收束与判读合同
| **材料** | **当前状态与收束** |
| --- | --- |
| 五方法11 Run与ESP完整复验 | 五方法固定H02原件保存11个Run ID、终态、配置、source/model/validator摘要与六类文件SHA；本次已核原位置未找到这11个Run目录。保历史完成事实，不冒充当前原生重验。ESP仍按各自原记录定位 |
| OnlineVerify / Aligned旧比较 | 指定旧输出根当前不在，32条真实模型负结果目前属运行记录级证据，原终态、逐token与模型/输入摘要未重新核得；eager按历史记载而非本拍新核身份解释。保负结果及缺口，不为补通过重跑同配方 |
| 8B冻结K旧比较 | 原header/summary、三臂各128条JSONL、fit/slice与日志可读，原source HEAD、prompt摘要和qid序列吻合；stored argmax布尔不等于完整token链重验。固定原源码有校准/eval文本去交集门，但fit自身缺model/source/校准文本哈希，不能事后补签；收益仍未确认 |
| 未完成范数/配对/续跑 | 登记原计划、实际完成/未完成和退出当前队列理由；排程取消不等于旧Run成功 |
| 历史来源缺口 | 四项cross-task原tool/plan/runtime精确字节仍缺，受影响读端严格拒绝、保unknown，不回填、不用当前helper替代；不阻塞不依赖这些文件的EMA/Kalman核验 |

逐Run核源码/声明、输入/模型/tokenizer、配置/seed、设备/dtype/backend、计划与实收样本、终态、质量门、指标及来源回执。只生成新收束摘要，不覆盖旧Run。原执行与失败从 [运行记录](<运行记录 Run Record/INDEX.md>) 定位，开发决定归日期开发记录。

T0门是每个实际提交token对应验证前向argmax，覆盖prefill、bonus、EOS及预算裁剪后完整链。独立AR一致率只作诊断，T0不推广为T>0分布保证。旧full/argmax协议保各自判据，不替换失败门。BF16相对独立AR的算子机理排查仍永久停止。

τ为排除prefill的解码提交数/decode_calls，零调用记缺失；逐prompt均值与合并计数之比分列。accepted与committed可能裁剪位置不同，保冻结口径。配对固定qid、输入、模型、停止和预算，保胜平负、缺行与不可比项，不把同prompt多树/长度/臂当独立样本。

实际加速需同质量、同输出条件完整计时，冻结预热、CUDA同步、加载边界、臂顺序与资源。缺计时不填零，host-clock、cProfile及关闭计时占位不是速度证据。工程tiny模型、小张量、重绘与计数复核只证明对应合同，不产生新真实模型收益结论。

# 六、停止与下一步
按本轮P0已核范围收束高价值Run；未覆盖的observation-only等模式与真实权重正确性保持未验，不把局部参考通过推广为全部Kalman已正确。本次EMA 8×32假设已按负结果结案；下一拍先补observation-only的GPU状态参考，再按上述队列选择唯一会改变研究判断的实验，不扩算法或研究矩阵。正确且收益支持则保留；正确但收益不支持则负结果结案；实现错误先修复后独立验证；缺来源则受影响部分未知并冻结。

研究判断、比较和预算改变时更新本文，不另建平行计划。归档不承接当前工作；新算法方向另行决定，不成为现有核验的等待条件。工程优先修直接阻塞正确性、证据链与结案的缺口，其余结构/测试迁移留开发待办。
