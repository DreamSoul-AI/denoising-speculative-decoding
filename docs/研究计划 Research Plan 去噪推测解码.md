**去噪推测解码 研究计划**
更新日期：2026年11月7日。唯一当前研究计划，包含当前成果与结论。工程清理已收尾，当前按 Owner 授权回到 training-free speculative decoding 的论文研究，以 denoising / Kalman filter / EMA 为基本要件。R65独立校准与R66合作者原代码隔离均获本切片正证据；R67已完成同一K跨实现对齐，R68–R74已逐项记录拟合条件、provider/anchor、frontier、计时与现有Kalman的正负结果，R64、历史结案与来源缺口保持原范围。工程实施见 [开发计划](<开发计划 Development Plan 去噪推测解码.md>)。

# 一、研究目标与优先级
宏观＝围绕 training-free 的 speculative decoding 发一篇论文，以 denoising、卡尔曼滤波、EMA 为基本要件。

介观＝搞清楚合作者 speculative_mtp（d91ed75）为什么能 work、我们这边为什么不 work。先在临时目录（D:\Grok Bot\tmp\sd）隔离跑通合作者原代码，用同一模型、同一数据、同一 GPU，按源码重新校准 K，作为新实验记录；再和我们的实现逐项对照差异（配置、mask provider、EMA 和 Kalman 公式、frontier gain、解码/验证/KV、评测口径），每项用最小 CUDA 实验定位导致差距的原因，结论写进研究计划。合并进我们的分支先不做，也不改合作者代码。

宏观与介观沿用 Owner 和巡检给定的原文；当前微观R75–R77已完成本拍留出、组合与质量预算检查，R68–R74保持各自已测范围，R67已完成同一K对照，R65/R66原冻结协议与结果保持，不以本拍小结论改写上两层。

1. **本拍R78–R79完整答案协议待验证**：先按第四章第二十节准入，再在固定独立近期切片执行；mask/EMA已对齐部分不重复派发，在线frontier仅在新合同冻结后进入。
2. **此前R75–R77有界执行已收束**：32题冻结K复核通过预设稳定门，16题直接组合两法均负，四题质量批全部无EOS、语义未闭合；结果见第二章及第十七至十九节。三批已退出，不改原码/生产或扩研究树。
3. **此前R68–R74逐因素对照已收束**：R67同K对照已通过，R68已完成固定401对的ridge对照，点方向通过但区间跨无效值；R69固定λ=.3的两条校准输入加入对照亦已完成，R69b配对边界对照已完成，R70/R71 provider与anchor、R72常量/完整frontier、R73计时口径与R74现有Kalman对照均已完成；本小时队列收束，不自动加臂/扫参/扩长度。每项先冻结协议、再CUDA执行并记录正负结果，临时输出只在D:/Grok Bot/tmp/sd的本拍子目录，原代码保持原字节。
4. **按已验证范围判断可用性**：分开 EMA、各具体 Kalman 递推与静态仿射 K；受测状态、时序、重置与数值合同已核，未覆盖范围不写成通过。
5. **保持既有结案**：分开受测实现可运行、接受效率、实际速度与证据不足。32/256 writing 联合收益候选关闭，未完成 Run 保留原状态，不通过延长预算或扫参数追正结果。
6. **保留有取舍价值的证据**：EMA 历史局部正结果的适用范围、Kalman 各配方已测负结果与未知、可信 ESP 基线和必要来源。低后续投入价值支线进入 [研究成果归档](<研究成果归档 Research Archive 去噪推测解码.md>)，保留负结果与来源，不删除旧 Run 或改签完成状态。

文献基础与阅读范围沿用 [文献调研](<文献调研 Literature Review 去噪推测解码.md>)，机制和代价见 [通用调研](<通用调研 General Research 去噪推测解码.md>)。当前结论沿用已有文献阅读与实际运行证据。

# 二、当前成果与结论
**当前总判断**：EMA已有历史局部接受正证据，但writing子集32/256及本次R64幅度规则微观消融均未支持各自预设联合收益，实际BF16末状态符合此前已声明舍入合同；不推广为全部EMA无效。Kalman类实现确实存在，但多个已测配方为负或未确认，8B历史命名冻结变体实为静态K修正，收益也未获确认。R65本仓独立校准与R66原d91ed75静态仿射去噪均在各自拟合留出切片取得CUDA稳态接受/速度正证据，见下表；它不是前述Kalman递推或旧8B冻结K的再确认。所测合同与收益结论分别处置；名称、静态K拟合或工程测试不能替代研究判断。

下表区分固定历史结果与近期有界运行。历史行未重签原生审计；近期真实CUDA试跑按各自源码/模型/输入/协议与完整提交链验收，不冒充旧480条复验。已退出投入的详细支线结果由 [研究成果归档](<研究成果归档 Research Archive 去噪推测解码.md>) 承接。

**R68–R74切片对齐结论**：同一K的运行时、本次外生轨迹provider递推与这批计时排序已对齐。R72完整仿射映射比最佳常量接受多9、速度1.102720×，比ESP为1.114408×；正证据属于按原码新校准的静态frontier残差修正。R71原EMA-last与R74现有online标量Kalman相对ESP分别0.904454×/0.913915×，各自联合收益不获支持。ridge、两条校准输入与prefill标签的条件效应分别记录，不相加解释全部历史差距。这些速度排除校准/加载、保留同等T0审计插桩，输出不全同；原480表cfg/K/CSV、语义质量及完整校准摊销仍缺，不冒充原表复现。

**R75–R77当前结论**：32条本次拟合/近期选择留出上，冻结完整K相对ESP为1.084824×、相对同dense路径MSE最佳常量为1.096119×，两个预设95%稳定门通过。固定16题上，直接叠加EMA-last或现有online Kalman相对fresh full分别0.878205×/0.904080×，两个联合收益假设均不获支持。452项正式运行/65536提交全T0通过；质量20项均触512预算、无EOS而不可评价，128预算24条匿名AI文本核查亦全unknown，因此当前没有语义等质或完整答案端到端加速结论。全部三模型批已实际退出，不扩参数、长度或新Study。

| **方案与范围** | **已有结果** | **当前结论与处置** |
| --- | --- | --- |
| R77：4数学题×5臂，512/EOS有界质量覆盖 | 20项/10240提交完整T0；全部512预算无EOS。每臂correct0/wrong0/not_evaluable4，保全分母；128预算翻译/摘要24匿名AI核查全unknown | **执行完成，语义结论未闭合**。自定义数值代理不等于官方GSM8K分数，报告passed仅表示评分合同成功；不能把0/4未取得答案解释为100%答错或质量通过。见运行记录第八十一节 |
| R76：同冻结full K上分别叠加EMA-last/Kalman，固定16题 | 144项/18432提交完整T0；相对fresh full接受−10.25/−7.0625，速度0.878205×/0.904080×，均0胜16负0平 | **两个预设联合假设不获支持**；95%速度区间[.856334,.901877]/[.883266,.924745]，97.5%区间仍低于1。更换provider改变ESP训练特征分布，仅检验冻结K直接组合；不乘32题批速度推combo/ESP。见运行记录第八十节 |
| R75：四任务各8题的冻结K拟合留出复核 | 288项/36864提交全T0；ESP/bias/full实际接受43.28125/43.84375/51.46875，平均τ1.516491/1.527499/1.681752；合并τ1.511905/1.523238/1.673806 | **点方向及预设稳定门均通过**。full/ESP1.084824×，95%[1.065705,1.103849]，接受+8.1875；full/bias1.096119×[1.080285,1.112069]，接受+7.625，两者31/0/1。历史480已见，仅相对本次拟合/近期选择留出；不同输出/同审计的固定预算速度，语义未证。见运行记录第七十九节 |
| R74：现有online标量Kalman与ESP | esp接受41.875/τ1.499941/秒3.138798/相对ESP1.000000×；kalman接受34.5/τ1.371887/秒3.434455/相对ESP0.913915×；48项完整T0通过 | **H74未获预注册联合收益支持：接受均差-7.375，墙钟+0.295656秒，速度0.913915×；探索性95%速度[0.867172,0.958170]、接受[-11.25,-3.75]。** CUDA state/innovation张量、Python host P/Q/R/K标量，prompt递推与同步成本均在generate内；不是合作者静态ridge K。[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第七十八节 |
| R73：同一R68保存72项计时的口径核对 | 内部prefill+decode与strict sync generate都排K03<K10<ESP，8qid符号同向8/8；无新生成/模型/拟合 | K03对K10速度1.044472/1.044511×，未发生排序翻转；仅排除本批该计时口径的排序解释，不归因为异步单因。[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第七十七节 |
| R72：同401对最佳常量残差与完整frontier gain | esp接受41.875/τ1.499941/秒3.175949/相对ESP1.000000×；bias接受44.25/τ1.551044/秒3.142639/相对ESP1.010600×；full接受53.25/τ1.725151/秒2.849897/相对ESP1.114408×；72项完整T0通过 | **H72按预注册点方向通过：接受均差+9，墙钟-0.292742秒，速度1.102720×；探索性95%速度[1.053091,1.159804]、接受[5.375,13]。** 同401对的最佳常量与完整条件映射比较，两者保原dense路径；不是固定full截距去A，也不是部署优化。[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第七十六节 |
| R71：EMA last/order0 anchor单因素，同8qid×128×3 | ESP/last/order0接受41.875/33.125/44.125，72项完整T0通过 | **H71未获预注册联合收益支持；last−order0接受-11、墙钟+0.402614秒、速度0.887095×。探索性95%速度[0.849912,0.930506]、接受[-14.125,-7.875]。** last/order0相对ESP分别0.904454×/1.019569×；anchor子对照与ESP收益分开。[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第七十五节 |
| R70：原代码与本仓provider逐状态CUDA核对 | ESP masks1032/1032，EMA-last masks/velocity/anchor/ref_norm各1032/1032存储位全同，forward/newgen/fit均0 | **两个机制子假设通过**，只限此外生ESP轨迹，不代表EMA收益或全部输入等价。[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第七十四节 |
| 输入 EMA，Qwen3-4B BF16 五方法固定480条，seed123，T0，256/EOS，BC30/60 | input_first 为以order0 EMA作anchor的一阶velocity外推（method=ema_velocity、level=order0），β=.9999、λ=.1，无verify/Kalman；pooled τ 为1.511957584/1.636339710，高于 input_zero 的1.422420206/1.519108823；配对 accepted 胜:负364:81、374:70。11个历史Run已完成，10个speculative Run的809,118个提交token argmax审计通过，计时关闭 | **已有可运行实现及该协议内接受效率优势**。保留优先基线；单seed、不同后续轨迹及无计时不能证明普遍优势、纯空间因果或加速。[历史证据归位](<运行记录 Run Record/2026-10-31 历史证据归位.md>) |
| Qwen3-4B已完成32-token CUDA试跑，SpecBench qid81–88、32 token/noEOS、naive K2/BC30 | 两臂16项、512个提交token来源/T0审计全过，8对输出token均相同。accepted总数ESP113、EMA86，EMA胜:负:平0:8:0，配对均差−3.375；同步解码墙钟均值ESP0.680310725秒、EMA0.810950813秒，配对均差+0.130640088秒 | **本次预设联合假设不获支持**：8条均更少接受且更慢。只关闭这次有界试跑，不抹掉旧480条局部正结果；本次只取writing子集，输出预算与停止条件不同，不构成旧完整480条的直接复验、独立确认或普遍速度结论。[本次运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) |
| 同8条256-token CUDA比较，模型/输入/参数与32-token回合相同 | 16项来源/T0、4096实际提交及8项EMA末状态门通过。原口径accepted ESP543、EMA509，均差−4.25；同步墙钟均值6.748489612/7.001628888秒，EMA−ESP为+0.253139275秒，接受与更快胜负均2:6 | **256预算联合假设仍不获支持**。EMA原accepted含1个未提交尾接受（实际提交中508），不事后改主指标。仅1对256输出相同，故这是同输入/预算下各自轨迹的实测比较，不是严格同输出加速证明；不自动续512或扫β。[本次运行](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) |
| R69b：同8cal/λ=.3，仅在324共同decode对后追加8个prefill标签 | R65保存值无损FP64→FP32，共同324 X/Y存储位零差；72项完整T0/9216提交通过。without/with接受51.875/53，τ1.694045/1.723797，同步秒2.898836/2.848934 | **H69b点方向通过**：with−without接受+1.125、墙钟−.049902秒、速度1.017516×，探索性95%速度[1.003445,1.036423]、接受[0,2.375]，WLT5/2/1。仅支持这8标签的加入效应，不能将多个条件效应相加归因为全部历史差距。[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第七十三节 |
| R69：固定λ=.3，只加入107/247两条校准输入，324→401对 | ESP/cal8/cal10接受均值41.875/51.875/53.25，τ1.499941/1.694045/1.725151，同步秒3.122478/2.835645/2.794639；72项完整T0/9216提交通过 | **H69按预注册点方向通过，区间仍含无效值**。cal10−cal8接受+1.375、墙钟−.041006秒、速度1.014673×；探索性95%速度[.992050,1.043003]、接受[0,2.875]，W/L/T6/1/1。加入两条的效果限此切片，不称一般校准样本量规律；两规模都比ESP有正点收益。[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第七十二节 |
| R68：固定原401对，仅ridge=.3/10，ESP/K03/K10各8qid×128×3 | K03的6,556,160个FP32字与原K逐位重现；72项完整T0/9216提交通过。ESP/K03/K10接受均值41.875/53.25/50.375，τ1.499941/1.725151/1.675156，同步墙钟3.138899/2.781261/2.905058秒 | **H68按预注册点方向通过，证据仍不确定**。K03相对K10接受+2.875、墙钟−0.123797秒、速度1.044511×；探索性95%速度区间[0.997400,1.088691]、接受差[−0.625,6.5]，W/L/T=4/3/1。两K相对ESP为1.128589×/1.080495×，本批没有支持“ridge10完全不work”；不据点值称λ单独解释全部历史差距。[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第七十一节 |
| R67：固定R66同一K，两端同8qid×128token各3次 | 48项完整T0/来源及6144提交通过；24/24完整输出、实际接受、完整tree/owner/裁剪trace均同，全部实际forward行argmax与frontier摘要亦同。CUDA复核1,794对frontier的5,382对BF16 placeholder/probe/head输入存储逐位相同。两端接受均值53.25、τ1.725151 | **H67获本切片支持**。同K时，本仓临时adapter与未改原d91ed75运行时离散轨迹一致；R65/R66原不同K的接受差不再构成本切片实现失败证据。后续λ/校准输入/配对边界的条件效应已由R68/R69/R69b分别记录，不能相加推广为全部成因。墙钟仅诊断，当前不判断算法速度差。[实际运行](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第七十节 |
| R66：原d91ed75代码，10cal×64token、同8eval×128token，原ESP/full各3次 | 原fit401对、校准内原网格选λ=.3，再全10cal拟合，评价零在线更新；48项完整来源/T0及6144提交通过。ESP/full接受均值41.875/53.25、τ1.499941/1.725151、同步generate墙钟3.243978/2.893222秒 | **H66获本切片支持**：full相对原ESP1.121234×，接受+11.375、τ+0.225210、墙钟−0.350756秒；速度探索性95%区间[1.071959,1.173249]，接受差[7.875,14.75]，接受8胜0负0平。输出相同仅6/24；无AR计时/语义评分。R65/R66的ESP在24/24配对输出和接受数相同，但两批K校准/配对/选λ不同，未证明full差距原因；不是原480条正表精确复现。[原运行](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第六十九节 |
| R65：8条校准×64token，另8条评价×128token，ESP／bias／full各3次 | 332对仅校准拟合，评价训练对0，K全程冻结；72项来源/完整T0及9216提交通过。qid级3次中位数后ESP/bias/full接受均值41.875/43.625/50.125，τ均值1.499941/1.538010/1.674861，同步墙钟3.208102/3.189781/2.936466秒 | **H65获本切片支持**。full相对ESP加速1.092504×，接受均差+8.25、τ差+0.174920、墙钟差−0.271636秒；速度探索性95%区间[1.060176,1.133017]、接受差[5.75,11.125]，8/0/0接受胜/负/平。bias为1.005744×且速度区间跨1，不据点值称确定收益。全输出相同full/bias相对ESP仅9/24、12/24；未测语义质量或AR计时，不称严格同输出/相对AR加速。共同final-norm取证开销已计入基线；不是原表复现、独立全局测试或Kalman证明。[实际运行](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)第六十八节 |
| R64：8条非writing输入×128token，ESP／原幅度EMA／范数重标定EMA，各3次 | 72项来源/完整T0及9216主提交通过；每qid/arm先取3次中位数后，ESP/raw/norm的已提交接受均值43.875/40.25/40.25，同步墙钟均值3.073649/3.259394/3.224180秒。raw相对ESP加速比0.943012，接受均差−3.625；raw相对norm为0.989196，接受均差0 | **H64不获支持**。原幅度既未超ESP，也未取得相对范数重标定的联合优势；norm亦未超过ESP。raw-vs-ESP速度比95%探索性区间[0.916053,0.973508]、接受均差[-6.75,-0.75]；raw-vs-norm为[0.972331,1.005006]及[-0.75,0.625]，不称norm确定胜出或两者等效。三repeat在本次输出/接受计数一致，raw-vs-ESP/norm完整输出相同分别18/24、15/24；非全同输出、无计时AR或语义质量评分，非独立留出。不追加本批实验。[实际运行](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) |
| 合成 EMA 与 observation 联合，确认seed21/22/23，192 prompt对 | 相对较弱EMA trajectory为102:71，p=.022；相对更强ESP为88:94，p=.71 | 已在合成预定主比较中确认优于EMA自身，但未优于更强ESP，不作为超越强单侧的成功方案。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>) |
| 标量trajectory-only Kalman，合成发现与确认分离 | 发现117:59:16，p=1.5e-5；确认101:83:8，p=.210 | **收益未获独立确认**，该配方停止扩展；不推广为所有Kalman模型无效。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>) |
| OnlineVerifyKalman，合成及真实模型CUDA BF16（旧比较backend/EOS未核） | 合成确认97:86:9，p=.460；真实Qwen3-4B/0.6B各32条、16 token、k1[4]，argmax通过，但accepted胜:负:平2:24:6 / 1:28:3，τ从1.361→1.218 / 1.452→1.216 | **该协议内明确负结果**，不以同配方扩大实验；执行正确性与收益失败分列。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>) |
| AlignedResidualKalman，真实4B/0.6B小规模比较 | 两模型均5:9:18，p=.424；τ分别1.306→1.294、1.335→1.327 | 未超过ESP，作为有价值负证据收束，残差对齐未证明收益。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>) |
| Kalman observation-only工程合同，OnlineVerify/Aligned × forget=.8/1 | 四场景CUDA通过：FP32实际状态匹配独立FP64参考，观测自有状态与动态ESP基底区分，首commit更新、后续commit不重复更新及reset符合受测合同 | **所测合同已完成**，不再列为待验任务。该Run无真实模型收益/计时对照，不能从四项通过推出Kalman有效或BF16全域正确。[运行证据](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) |
| 8B历史命名frozen-Kalman（实际冻结对角K残差修正），office-qwen3-8b-v1，128条Dolly、64 token、T0/noEOS、k1[4] | 对ESP为51:41:36，p=.348141；accepted配对均差+.1953125，CI[-.0546875,.4453125]；pooled τ 1.343552→1.350754，argmax通过，未测速度 | **冻结校正收益未获确认**。固定原源码为hybrid_gain、mode=frozen、observe=last；校准后无在线K/P更新，不能支持真正在线滤波有效。只作为既有静态校正对照收束。[原记录](<运行记录 Run Record/2026-11-03 Qwen3-8B同预算与冻结卡尔曼.md>) |
| 静态ridge/仿射残差K与k_affine_residual | 旧合成确认124:58:10，p=1.1e-6；真实0.6B 128条50:40:38，p=.343，未获收益。当前postcommit-final-norm-v1仅为合成工程夹具，160对数据与真实收益未执行 | **静态拟合正结果不证明Kalman有效**。保留必要诊断对照，不优先扩新矩阵。[联合检查](<运行记录 Run Record/2026-11-01 联合mask检查.md>)、[实际声明](../studies/k_affine_residual/study.yaml) |
| Qwen3-8B observation扫描及按长度墙钟 | 均为**ESP**，不是EMA/Kalman。扫描同128条探索数据，g=.5相对g=0 accepted均差−13.1796875，p=6.388e-29，平均墙钟+.473827秒；另一次按长度比较trajectory相对full在64/256/512 token平均少.096467/.474111/1.148640秒 | 这些条件下加入该观测不划算；不把同批探索的配置选择当独立确认，不把收益归给EMA/Kalman。[扫描](<运行记录 Run Record/2026-11-04 Qwen3-8B observation增益扫描.md>)、[墙钟](<运行记录 Run Record/2026-11-04 Qwen3-8B 墙钟按长度.md>) |
| ESP非官方8B固定480条复验 | BC30/60 pooled τ 1.563377504/1.647301264，未达论文目标1.63/1.74；完整提交链239,137 token argmax审计通过 | 保留可信的**本仓协议基线**，论文目标未复现，不为追数字重开长矩阵。[历史证据归位](<运行记录 Run Record/2026-10-31 历史证据归位.md>) |

EMA五方法原件固定于Git b8b20b5ada7fd860894114cbf224ce749af430cf的 studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json，SHA256为8e85ac6ea9984047abf0ffd5d6653b28b6aadd74aa876139fa10465dc5bbf274。固定原件的字节核验不等于其全部原生Run重新验证。8B frozen身份另据固定原HEAD 29fc9f815fa83aae72272f4e96a831425d3f5d8c的studies/trajectory_observation/structure/office_qwen8b.py与src/spec_mtp/masks.py核对：离线校准对角K后不作在线K/P更新，观测z仍随提交变化，不表示输出mask恒定。早期first26调参结果仅作发现线索，不充当独立确认。

## 当前阶段结案决定
| **对象** | **已经能够判定** | **当前处置** |
| --- | --- | --- |
| EMA-vs-ESP：同8条writing、32/256预算 | 两次来源/T0门通过；256的实际末状态符合声明精度合同。两预算的联合收益假设均不获支持，结果限各自协议 | **关闭256-token预算扩展候选**；该切片不作为默认加速方案，不自动续512、扫β、加臂或改状态精度。保留EMA为有条件基线，旧480条局部正结果不被改签 |
| R64微观幅度规则消融 | 新四类8条切片中，raw对ESP接受更少且更慢，对norm没有联合优势；质量仅覆盖完整验证链T0 | **本批结束**，两种现有幅度规则都不据此进入默认加速候选；不扫β、延长长度或挑子集追正。保留逐qid反例、完整正负对照与初始化几何；不推广全部EMA/Kalman无效 |
| Kalman observation-only四场景 | 两类provider在所测输入、精度与时序下的工程合同通过；本Run没有真实模型收益比较 | **关闭这项受测合同覆盖缺口**，不重复派发。收益仍按具体旧配方的负/未确认证据判读，不写成整个Kalman方法已有效 |
| 历史正/负/未确认材料 | 原协议内的结果和已核来源能够保留；缺失原件、未覆盖域与理论假设不能据当前测试补齐 | 保留其条件与unknown，受影响读端严格拒绝；不通过补跑新数据、改门或重签，把旧Run补成已验证 |

# 三、已核实现合同与证据边界
## （一）方案身份与数学合同
实现见 [providers.py](../src/spec_mtp/structure/algorithm/mask/providers.py) 与 [fit.py](../src/spec_mtp/structure/algorithm/mask/fit.py)，分别核验，不能混称一个Kalman方法。

| **实现** | **声明的合同** | **可得结论边界** |
| --- | --- | --- |
| EMA provider | v=βv+(1−β)(e_new−e_prev)，anchor/order0、步长外推、范数、block-local与实际β/λ | 平滑与外推是否按协议执行；没有协方差模型，不据此声称Kalman最优性 |
| KalmanMaskProvider | 标量随机游走P⁻=P+Q、K=P⁻/(P⁻+R)、level与Joseph后验、噪声自适应/floor、冻结/reset | 递推是否忠实且数值有效；共享标量噪声假设另作研究判断 |
| OnlineVerifyKalman / AlignedResidualKalman | OnlineVerify逐维P/Q/R；Aligned标量F/P/Q/R与F裁剪；观测和提交token时序 | 排除错位、未来信息和跨prompt泄漏，不等于完整embedding协方差模型 |
| HybridGainKalman / OutputHiddenHybridKalman | 对角K参数递推、x/innovation、校准先验、在线/冻结/scratch/bonus与root对齐 | 实际变体与Run逐一对应，不同状态语义不共享收益结论 |
| 静态scalar/diag/affine/ridge K | 拟合输入/目标、隔离单位、正则与冻结时点 | 只回答回归校正合同，不证明Kalman滤波有效 |

方案解释按状态、观测、F/H（适用时）、P/Q/R/K、更新时点、owner/bonus槽位、重置与冻结条件定位，不把身份表视为全部模式已验证。未满足线性/噪声等假设时明确模型限制，递推存在不等于最优性成立。当前实现已拒绝13个方差/噪声字段的非有限值，并修复OnlineVerify/Aligned的合法forget=1除零，实际复现与验证见日期记录。该端点沿已有scalar语义关闭Q/R自适应（包括初值低于floor时保持），P与F仍更新；这是显式no-adaptation约定，不是自适应权重在forget→1时的连续极限。旧finite且forget=.95的研究配置不因这次端点修复自动改签或失效。

## （二）已验证合同范围
已保存的CUDA FP64独立参考覆盖EMA闭式加权历史与last/order0 anchor，标量Kalman的Joseph后验、自适应floor/冻结，以及OnlineVerify逐维和Aligned标量递推。forget=1单独检查Q/R保持、P/F继续更新，包含合法below-floor初值；重初始化清除所测状态。Hybrid按独立回归更新式区分first、last与frozen，冻结K/P不冻结观测或输出mask。测试与实际设备、版本从对应运行记录定位，范围限声明输入与模式，不等于完整状态空间证明。

时序边界：OnlineVerify和Aligned按每轮root prior的**首个提交**更新一次；后续同轮commit不新增该后验，OnlineVerify的z因此保首commit，不是末bonus。Hybrid first/last分别在首commit/finish更新，last使用最后实际提交，EOS或预算截断时不保证它是完整轮bonus。真实CUDA解码回归覆盖prefill、逐commit、截断/EOS、finish与完整T0链。保留这些旧语义，不直接改成另一算法；递推吻合并不证明状态空间/噪声假设成立、滤波最优或研究收益。

## （三）剩余事项与执行边界
| **事项** | **当前状态** | **停止与继续条件** |
| --- | --- | --- |
| 本阶段受测实现合同 | EMA末状态、Kalman observation-only及既有full/时序合同已经按各自范围验证 | 无具体阻断合同矛盾则停止补验；扩大精度/输入域不自动成为新任务 |
| 历史Run来源与结案 | 正、负、未确认结果及原件缺口按第五章分别保留 | 只核可定位的已有材料；缺失维持unknown，不拿新运行替代旧原件，不为结案凑满矩阵 |
| 例外合同补验 | 只有已声明合同与实现/证据的具体矛盾，且会阻断当前结论时才成立 | 先固定刺激、来源和单一判据，执行范围与次数以当前授权为准；无具体矛盾不运行。理论、泛化或资料缺失不能自动触发GPU |
| 本步 R64 幅度规则消融 | 72项已完成，预设联合假设不获支持；与已关闭 writing 32/256 扩展区分 | 原三臂、8条输入、128token与3次重复保持，本批结束；未测语义质量/AR，不追长度或扫参数 |

未来确有会改变方案取舍的研究问题时，先更新本文对应范围、对照、预算与停止规则，再按当前授权执行；关闭候选不因旧命令或计划自动恢复。涉及数值时确认CUDA可用并记录实际设备、dtype/backend与环境，GPU忙时等待或缩规模，CPU仅做快速纯合同/IO检查。复用本地模型来源，不下载或覆盖CUDA torch；新Run独立冻结来源，不覆盖旧Run。

# 四、重点课题与保留决策
## （一）EMA：收益边界已按本切片收束
本课题的已完成对照为同8条writing子集qid81–88，Qwen3-4B BF16/eager、naive K2/dynamic/pruned/BC30，ESP λ=.1与既有EMA β=.9999/λ=.1/order0，其余固定；分别使用32与256新token/noEOS。完整来源/T0门通过后，以EMA−ESP的accepted均差>0且同步generate墙钟均差<0为联合判据。数字、输出一致性、尾接受口径和处置统一见第二章；冻结协议、模型/输入/源码身份、逐case与计时成本见 [日期运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)。

256的8×5实际末状态域对声明舍入参考误差全0（BF16状态/mask maxULP0、anchor位相等、FP32 ref_norm差0）；相对理想FP64的velocity相对L2偏差2.89%–3.24%、mask为2.78%–2.98%。这是末状态与精度合同结论，不是理想递推等价、逐步实测或性能差的因果解释。

16条256轨迹保留各自旧32前缀仅作诊断。writing子集与已观察的480条均不改标独立留出，当前比较不构成旧480完整复验，也不推广为全部EMA无效。后续不自动扩展该候选，停止规则由第二章结案表维护。

## （二）Kalman：逐配方验真
第三章列出的受测合同已按对应范围完成，不再作为重复待跑任务；收益按第二章逐配方结案。OnlineVerify、aligned residual、output-hidden等已测负结果不重复扩样；若真实实现错误使旧比较无法回答问题，保留旧记录并明确原因，新验证单独冻结。8B frozen按静态对角K修正结案，不作为在线滤波正证据；其校准/评价隔离与来源仍须完整登记。没有信息增益就保留未获确认并停止投入。

静态K=[A,b]与postcommit-final-norm-v1只作拟合改善/解码受益的诊断对照。160对与跨集真实收益未执行，不能写成Kalman成功，不因夹具可运行就启动矩阵。

## （三）Study取舍
既有队列保EMA条件性基线、已有Kalman方案结案、可信ESP基线及必要来源收束；当前按R68起逐项预注册协议执行；R65/R66/R67已收束。五方法/input-mask材料服务EMA身份和对照；trajectory/observation探索约束解释，不作为独立扩展队列。输出hidden、失败top1/bonus和已停止BF16机理支线进入 [研究成果归档](<研究成果归档 Research Archive 去噪推测解码.md>)。

七格范数、11格独立配对、旧72项模型矩阵、旧长矩阵和原1500步训练不自动恢复，退出当前排程；旧pending/partial保持真实状态，不伪造全Run完成。特定材料直接影响结案时只读所需来源或按第三章登记最小验证，不整批接卡。模板、smoke、来源校验和测试基础设施是工程支持，不登记为算法收益Study。

研究队列与记录归属按上述处置维护。Study源码、第三方来源、数据、旧Run及source lock保留；物理删除/归并前核真实消费者和取证依赖。归档基于低后续投入价值，负结果本身仍是有价值证据。

## （四）本拍 R64：EMA 外推幅度规则的探索性消融
**本步研究依据**：为 training-free 去噪起草选择幅度规则，提供论文前期消融或负结果；不训练模型，不声称新颖性或 Kalman 最优性已成立。依据为 ESP v2 [§3.1–3.3](https://arxiv.org/html/2603.17942v2) 的冻结模型 probing/均值更新，以及本仓既有 EMAVelocityMaskProvider。当前实际重读前者指定正文，EMA 幅度规则是本仓实现，不归于 ESP 作者。

**微观假设 H64**：保留 EMA 原幅度的外推，在本切片相对范数重标定 EMA 和 ESP 都获得更多已提交 speculative token，并缩短同步解码墙钟。原幅度步长为 v，重标定步长为 r·v/||v||，r 是差分范数 EMA；只有 normalize 改变。幅度不预先解释为可靠置信度。β=.9999 且首差分初始化可能主导有限长度状态，BF16 还影响实际递推；单次消融不能证明噪声模型或滤波最优性。

| **微观冻结项** | **协议与验收** |
| --- | --- |
| 数据与预算 | SpecBench 现有原件，固定 qid101/102（reasoning）、111/112（math）、121/122（coding）、241/242（summarization），只取首轮；每条128新token、T0、noEOS、seed123。它们来自已看过的480条，不称独立留出或跨域确认；不根据结果剔除输入 |
| 模型与三臂 | 本地 Qwen3-4B，BF16/eager，冻结权重；K2、dynamic/pruned、BC30、naive。ESP 为 mean_prompt/λ=.1/update=true；EMA 两臂均 order0、β=.9999、λ=.1、step_scale=1、extrapolate_ema=false、update=true，只分别 normalize=false/true。不增加 AR、Kalman 或其它臂，不改生产实现 |
| 重复与顺序 | 三臂各2token预热一次并排除；3次正式重复，repeat为外层、qid为中层，三臂顺序按(qid序号+repeat) mod 3循环轮换，共72case。3次计时是同prompt的重复，不能当24个独立样本 |
| 计时与口径 | CUDA同步夹住 provider构造＋generate/同等内存证据记录；不含加载、哈希、渲染、落盘、独立审计或计时外几何诊断。每qid/arm先取3次中位数，跨8qid计算配对墙钟差；相对基线加速比为基线中位时长之和/该臂中位时长之和。归一化新增归约/除法/标量分支的成本计入，不以墙钟差独断信息质量 |
| 接受与质量 | primary accepted为每轮min(accepted_nodes数,committed_tokens数)之和，排除被预算裁掉的尾接受；逐轮与raw forward实际commit核对，legacy accepted另列。全部72case须完整128token、来源/evidence/完整提交链T0通过。报告三臂输出一致率及重复一致性；输出不同仅比较同输入/预算下各自轨迹，不称严格同输出加速。T0不替代数学答案/代码/摘要的语义质量评分 |
| 主判据与不确定性 | H64获本切片支持须全部质量门过，且raw分别对normalized和ESP的qid级已提交接受中位数均差>0、同步墙钟中位数均差<0。报告两个子对照各自结果，不能只选有利者。CUDA按8qid配对bootstrap10000次报告95%区间，重复不扩大独立样本量；区间仅作小样本探索性不确定性，不称总体显著性或独立确认 |
| 计时外机制诊断 | 72case全部结束后、关闭模型前，用同prompt独立重放初始化，记录实际v的范数、r、与首差分余弦及两种mask的最大差异；不声称采到了全部实际中间态。若初始化mask相同，仅记“初始无差异”，后续干预是否相同仍未知，不据此否定后续变化或从计时推机制；个别零差不删样本，主判据不变 |
| 来源、预算与停止 | 新Run与临时材料仅用户指定D盘临时根round64，完整冻结当前源码、计划/协议/driver、模型清单、数据和渲染tokens。先确认CUDA可用并记录RTX5090 D v2；忙则等待或缩规模前重新冻结协议，不回退CPU。整批执行预算600秒，失败保partial、不盲重试；正负/无差异均停止，不续512、扫β、改精度、重跑旧writing或旧矩阵 |

**预注册与实测状态**：计划先于正式GPU提交1553b6a，执行前仅收紧初始化诊断的范围后提交56873c8；实际Run冻结后者clean及原计划/协议字节，不回签当前结果正文。72项均完成，H64=False；第二章维护结论，既有运行记录第六十七节保存实际命令、原件和未覆盖项。初始化八条的两种mask最大差异均非零（0.001953125–0.005859375），v与首差分方向的FP32余弦估计约1；这仅为初态证据，不能归因为退化成因或证明幅度是置信度。进程退出后收到的层级澄清只修正当前宏观/介观原文及“微观假设”称谓，未改变H64/参数/判据。生产源码、测试、CI、工程90/300秒门及旧Run不改。

## （五）本拍 R65：独立校准的首 mask 隐藏状态残差修正
**依据与身份**：合作者提供的 [固定源码](https://github.com/jasonchengqs/speculative_mtp/commit/d91ed75145801d9263281d2938a7d0da5841b2bf) 中，frontier_gain 以已知 bonus 的输入 embedding 与原 placeholder 之差，修正已接受最深节点的首 mask 最终归一化隐藏状态，再重算该位置的起草 logits。本拍实际阅读 frontier_gain、fit_frontier_gain、decoding、state_dump 及计时调用方；源码可读，原正表的最终 cfg/K/逐 prompt CSV 未提交。本拍是新校准与新评价，不冒充原表复现，不复制私有来源到本仓。它是冻结 LLM 权重的离线 ridge 仿射残差校正，有数据校准成本，不是协方差递推 Kalman，也不宣称无参数估计。

**微观假设 H65**：仅用固定校准输入拟合一次 K，冻结后在与校准分离的本切片，相对 ESP 同时增加实际已提交接受数、缩短 CUDA 同步解码墙钟，并保持完整提交链 T0。仅 bias 为必要消融，不以它较弱替代 ESP 主对照；不由本假设改写宏观或介观。

| **微观冻结项** | **协议与验收** |
| --- | --- |
| 数据分离 | 同一 SpecBench 原件、首轮、四类各2条。校准 qid103/104、113/114、123/124、243/244，各64新token；评价105/106、115/116、125/126、245/246，各128新token。校准与评价集合严格不交；这些材料来自历史已见480条，只称本次拟合的留出，不称全局未见测试集或跨域确认；不得按结果剔除或换输入 |
| 模型与共同路径 | 现有本地 Qwen3-4B，BF16/eager，冻结权重；T0、noEOS、seed123、默认已有聊天渲染参数。ESP mean_prompt/λ=.1/update=true，K2、dynamic/pruned、BC30、naive。复用当前完整验证链宿主，三臂同样捕获 final norm，只有校正臂多算 correction 和 LM head；不改生产源码 |
| 拟合与时序 | 校准只运行 ESP。u=[e(bonus)−本 forward 首 placeholder;1]，y=下一 decode forward 同位置实际 root 的 final-norm hidden−该 frontier 首 mask hidden；包含 prefill→首 decode 的配对和后续相邻 decode，校正也从已知 prefill bonus 开始。它区别于原离线 recorder 仅取 decode 对的口径。跨 prompt 不串 pending，末尾未获下一次标签的 pending 丢弃；512个校准提交不等于512个训练对。CUDA FP64 解 (UᵀU+diag(10,…,10,0))K=UᵀY，偏置不惩罚，K 转FP32后冻结。禁止原 fit 脚本的全数据重拟合；评价不收集标签、不更新 K |
| 三臂 | ESP 无校正；bias 只加同一校准残差 Y 的均值 b0（不是完整 K 的截距行）；full 为 z′=z+uK。只替换下一轮起草的首 mask logits，bonus 已知后执行，其余起草行和验证 logits 保持原路径。不把静态 K 命名为 Kalman 已有效 |
| 重复与计时 | 拟合后每臂2token预热一次并排除；正式3次重复，repeat外层、qid中层，按(qid序号+repeat) mod3轮换三臂，共72case。CUDA同步夹住 provider构造、generate与同等内存证据记录；不含模型加载、校准、拟合、渲染、哈希、落盘或独立审计。校准/拟合成本另报，有正收益时再算摊销，不隐藏训练外成本 |
| 接受与质量 | primary accepted逐轮取min(accepted_nodes数,实际committed_tokens数)，legacy另列；τ沿宿主合同按解码阶段提交token（排除prefill）/解码调用报告，不混用。72case全部须128token、来源/evidence/完整提交链argmax审计通过；报告重复一致性、三臂完整输出一致率。T0不证明独立AR逐字一致或语义质量，输出不全同则不得称严格同输出加速 |
| 主判据与分析 | 每qid/arm先取3次中位数，再以8qid配对：full−ESP接受均差>0且墙钟均差<0、全部质量门过才支持H65。速度比为ESP中位时长之和/full中位时长之和。CUDA FP64按qid配对bootstrap10000次、seed6501，报告探索性95%区间；重复不扩大独立样本量，bias仅次级消融，不以单格或子集追正 |
| 来源与停止 | 新round65外部Run冻结当前clean提交、原计划/协议/driver、模型全清单、输入与K；torch.cuda.is_available须真并记录RTX5090 D v2。研究批总预算900秒，原工程90/300秒门保持；失败保partial，不盲重试。正负均停止，不扫ridge/长度/模型，不重开旧矩阵或旧EMA候选 |

**预注册与结果**：0aef28f先登记假设，bb7333c明确prefill采/用时序，d80cfd0在正式GPU前明确τ排除prefill；实际Run冻结d80cfd0/clean。唯一正式进程于UTC2026-11-06 17:37:57—17:42:43退出0，内部271.335160秒/900秒；72项全部完成，H65=True。统计在CUDA按既定8qid/3次中位数、bootstrap10000/seed6501执行；三臂重复的输出和实际接受均一致。原件、逐qid结果与完整源绑定见既有运行记录第六十八节，不把结果文档提交回签成新模型运行。

**成本与解释边界**：校准生成512token用13.305518秒，332对ridge拟合0.161637秒；K为2561×2560、6,556,160系数，序列化26,237,389字节。按本切片0.271636秒/条的稳态节省，仅校准生成+拟合的计算摊销下界为50条128-token请求；未计加载、渲染、哈希、落盘与冷启动传输，完整冷启动盈亏点仍未测，不能宣称单次请求含校准也变快。CUDA从保存校准对独立重算K与bias均零差。校准MSE下降只作拟合诊断，不当作评价收益。LLM冻结但K需数据拟合；本拍不支持“无校准”或“零额外参数”表述。

**本批停止与下一步**：保持本K、输入、负结果与正结果原件，不扫ridge、延长长度或换模型追正。该正证据可进入论文探索性主表/消融；17:48 Owner更新介观，下一步优先原d91ed75代码隔离，再做共同条件下的逐项对照；原冻结K跨切片候选不自动执行，语义质量和完整冷启动成本缺口保留。这里的处理同时包含残差校正与单行LM-head重算，未设零K重算head臂，不能单独归因于某个矩阵分量或滤波机理；bias是独立合理常量消融。已有k_affine_residual的postcommit合成协议、合作者原表及本次新frontier协议分别保留，不把本结果替换旧160对未执行状态或四项拒绝原件。

## （六）R66：合作者原代码隔离运行预注册
**研究问题与微观假设 H66**：不改d91ed75145801d9263281d2938a7d0da5841b2bf原代码，按其原StateRecorder、fit_frontier_gain和FrontierGain重新校准K后，原full相对原ESP在同一评价切片获得实际接受增加与同步墙钟下降，完整T0通过。这是原程序的新校准执行，不是原正表精确复现，也不是跨实现差距的因果结论。R65说明本仓隔离重实现也能在所测条件获得正结果，不能把旧EMA负结果等同于同方法复现失败。

| **冻结项** | **协议与验收** |
| --- | --- |
| 原码与执行边界 | 只在D:/Grok Bot/tmp/sd/round66运行已逐Git blob及SHA256核验的小源码闭包，原字节不改，不合并到本仓；外层只配置/派发/只读取证，不替换算法函数、公式或验证/KV绑定。先冻结原码清单、当前计划、driver/config、模型、真实输入和子进程命令；GPU不可用或不足则等待，不回退CPU |
| 数据与最小准入差异 | 模型仍同一本地Qwen3-4B/同模型清单，GPU仍RTX5090 D v2，评价8qid105/106/115/116/125/126/245/246复用R65实际渲染tokens，各128token、T0/noEOS/seed123。原fit入口按i%10选choose=4、report=9，8个校准prompt会缺report组而失败；因此只加两条明确校准qid107/247，冻结顺序为103/104/113/114/123/124/243/244/107/247，各64token。不得重复/伪造prompt填组，10cal与8eval文本/tokens均不交 |
| 原校准流程 | 原ESP的StateRecorder只保存decode相邻对，不补prefill对。将仅10cal按冻结顺序保存为单个states.pt，交给未改原fit脚本（不依赖多文件glob排序）：初fit8、choose序号4的qid123、report序号9的qid247；原LAMBDAS=(.1,.3,1,3,10,100)仅在校准内部按其cov@14选择，再依原程序全10cal重拟合。内部report cov是重拟合前校准诊断，真正评价是未传入fit的另外8qid。不要把它写成固定ridge10或独立内部report确认 |
| 两臂与重复 | 原ESP mean_prompt/λ=.1/update=true，num_masks2、dynamic/pruned/BC30/naive，BF16/eager。原full仅打开frontier_gain enabled/path，online=false；K生成后原字节冻结，评价不拟合。两臂各2token预热排除，3次重复，repeat外层/qid中层，按(qid序号+repeat) mod2轮换，共48评价case |
| 原路径T0与口径 | 原decoder无完整trace导出，使用其既有recorder.record回调的只读代理与HF输出只读hook观测真实验证logits、accepted_nodes、未裁commit及history；原reset/record/export/close仍原样委托，不改返回。按实际128-token结果裁尾，每一提交绑定真实验证row argmax；prefill单独记录，实际接受逐轮min(A,C)，τ排除prefill。原stats与严格新口径分别保存，不能以AR一致替代T0；若无法闭合则记录未通过，不降低门 |
| 计时、判据与停止 | 对照原esp_mtp.test中decoder在prompt循环外构造的边界，主墙钟用CUDA同步夹住原generate及同等只读取证；每case原decoder/gain构造与K加载/传输另计setup，不混入稳态主指标。外层为换臂各case新实例，online=false且原generate/reset不变，区别于原main缓存一个controller，不能把setup也说成已摊销。模型加载、哈希、校准/原fit另记，原内部异步计时仅诊断。每qid先取3次中位数，H66须全部48项来源/T0/128token过且full−ESP接受均差>0、墙钟均差<0；GPU paired bootstrap10000/seed6601以8qid为单位。整批900秒/外层960秒，无自动重试，失败保partial；正负均停，不手动扫参、放大模型/长度 |
| 对照解释与后续 | R66与R65同时不同于cal8→10、原内选λ/全cal重拟合、配对是否含prefill及实现，故禁止把两批结果差归因于某一差异。先只判原full对原ESP；后续再冻结共同输入/校准，逐项最小CUDA对照配置、mask provider、EMA/Kalman公式、frontier gain、解码/验证/KV与评测口径，未执行项不写已解释。无AR计时/语义质量或原表制品时保持缺口 |

**结果与身份**：4e1c967预注册原代码隔离，4368281在任何正式GPU前对照原esp_mtp.test收紧计时边界；实际Run冻结4368281/clean。UTC18:02:09—18:05:53，calibration、未改原fit脚本、evaluation串行三个owner及父进程均退出0；内部224.395558秒/900秒，48case全部完成。原27个blob逐SHA256/Git对象字节核验保持，外层只在decoder实例注入原样委托的recorder观察层，未全局重绑原类/算法/验证/KV；原model/decoder/recorder/gain及fit脚本实际执行，不冒称原main的整表配置/AR/logger也已复现。17:48新介观不回写R65原冻结目标。

**质量与成本**：CUDA FP64按8qid/3次中位数、bootstrap10000/seed6601分析，full−ESP墙钟探索性95%区间[−0.492507,−0.210694]秒、τ差[0.159548,0.291328]；重复输出/接受均一致。full与原ESP完整输出仅2/8独特qid相同，T0不代语义质量或独立AR逐字证明。原calibration阶段36.231521秒、原fit阶段8.772394秒（含各自加载/验证等）；controller fresh setup均值ESP0.000067/full0.007273秒另列，不混入generate主墙钟。K为2561×2560/6,556,160系数，原序列化26,226,324字节。校准、参数与冷启动成本不隐去，未声称单次请求含校准也变快；原main缓存controller与本外层换臂新实例的部署成本不同。

**第一项实际对照**：R65与R66在相同8个评价输入的ESP三次结果，24/24完整输出和实际接受数相等。这是所测基线路径一致证据，不证明全量数值等价，也不能把full的1.093×/1.121×之差归因于实现。原内部report cov@14 .642857→.904762仅为全cal重拟合前的校准诊断，不是最终K的独立测试。

## （七）已定位的源码差异与下一项最小对照
以下只列实际阅读/结构核对与本次观测，不把未运行的单项因果实验记为完成。原路径相对固定d91ed75的denoise_mtp/src；原文件和本仓源码的完整SHA、行范围及5个函数AST摘要在round66/source-comparison.json，实际版本与原件由运行记录定位。

| **项目** | **已核事实** | **仍待最小CUDA对照** |
| --- | --- | --- |
| 配置与ESP | 两边mean_prompt及m←m+λ(e−m)一致；K2/dynamic-pruned/BC30/λ=.1一致，配置容器接口不同。真实8qid ESP输出与接受24/24相同 | 已完成本切片基线对齐，保留其它配置范围 |
| EMA与Kalman身份 | 速度/范数EMA递推相同，但原EMA用末embedding anchor；本仓另有H64采用的order0 anchor。输入mask外推与正表frontier静态ridge不是同一方法；本仓P/Q/R Kalman也不能与静态K混称 | R70保存轨迹两provider逐位一致；R71只换anchor后last/ESP0.904454×，R74现有Kalman/ESP0.913915×，各自本切片联合收益不获支持，不推广至全部EMA/Kalman |
| frontier与拟合 | 两边均在已知bonus后修正首mask final-norm hidden并重算draft head；R65 cal8/固定10/含prefill，R66 cal10/原网格选择.3/仅decode相邻对 | R67同K运行时对齐；R68固定401对分离λ已完成，R69/R69b分别记录校准输入与prefill标签条件效应，R72完整映射优于同dense路径最佳常量 |
| 动态树与KV | 预算、top-k pruning、dynamic tree、crop、keep五函数仅去注解/docstring后的AST相同；均保prefix+root+accepted path | R67完整tree/accepted路径与全部行argmax已同；全量浮点状态不属于本次观测 |
| 控制器与评测 | 原入口prompt循环外加载K；R66将fresh setup独立记，primary仅generate+同等只读取证。R65驻GPU K且provider构造在primary | 跨批墙钟仍不作单因归因；R73在同R68保存72项上统一比较两计时口径，排序均K03<K10<ESP，无反转 |

**本项已执行**：R67同K对照及R68–R74逐项冻结对照均已完成，正负结果与未覆盖范围见第二章及各协议状态。固定401对ridge、校准输入/prefill、provider/anchor、frontier、评测口径与现有Kalman各自收束；不重复派发、不把临时实现并入生产。下一拍未执行候选仅按第六章另行明确数据、语义质量与预算。

## （八）R67：固定同一K的跨实现CUDA对照
**微观假设 H67**：固定R66原K、真实输入与模型后，未改d91ed75原解码器/FrontierGain与本仓当前宿主加R65临时FrontierPolicy，在本切片具有完全相同的输出token、实际接受和规范化解码trace。它是实现一致性假设，不是再次拟合、证明Kalman或比较两批不同K的速度。不改生产实现、合作者源码、EMA/Kalman参数、校准数据、λ或KV路径。

| **冻结项** | **协议与验收** |
| --- | --- |
| 原件与范围 | R66 gain.pt原字节SHA17b62c5ff2df63df0d4201a66158b5d1be00c39f984fb45c37394c8a13da2e66，原λ=.3/401校准对/2561×2560 FP32矩阵；仅加载，不拟合/在线更新。R66原27源码blob、R65临时adapter与R66临时只读RecorderTap各复制原字节并登记来源，本仓生产源码快照另核；临时实现与私有原源码不入Git、不合入生产分支 |
| 数据与模型 | R66实际8条评价qid105/106/115/116/125/126/245/246、首轮渲染tokens逐字节复用，不重新渲染或挑题；本地同一Qwen3-4B与14文件清单、BF16/eager、RTX5090 D v2。T0/noEOS/seed123、128新token、K2/dynamic-pruned/BC30/naive、ESP mean_prompt/λ=.1/update=true，两端均只开full frontier静态K，online=false |
| 新执行 | 本仓端、原代码端两个独立owner子进程串行，实际确认前者退出再启动后者，不共用模型/KV/模块导入状态；两端各1次2token预热排除，各8qid×3次正式重复，共48case/6144正式提交。重复用于复现性，不当24个独立prompt；不新增ESP/bias或其它算法臂，不重跑校准 |
| 共同观测 | 本仓保留宿主原完整T0记录，原端沿原recorder实例只读委托及HF forward hook采真实提交链。两端以只读profile观察原build_tree返回值，保存tree tokens/parents/depths；不重绑原树构建/验证/KV函数或改变模型返回。final-norm/head只读hook记录实际首mask修正的BF16 head输入摘要及argmax等边界，避免保存全词表logits；摘要相同只证明所记录边界，不宣称所有浮点状态相同 |
| 规范化trace | prefill独立commit；每decode pass按实际forward保存root_position、完整tree tokens/parents/depths、accepted_nodes、实际裁剪后committed_tokens及实际commit的owner/argmax链。仅排除计时/仪器专有字段；终止pass也保完整accepted路径并另外按min(accepted_nodes数,实际committed_tokens数)计实际接受。不能为凑一致删掉不同的树/终止事件 |
| 主判据 | 48case全部128token、来源/模型/K不变与完整T0通过；24对同qid/repeat的全部输出、实际接受和规范化trace均逐项完全相同，才支持H67在本切片成立。各端内部3次重复也逐项报告。任何差异均反驳严格一致；保原件、定位首个不同的token/pass/字段，再区分输入、K、frontier或验证边界，不靠AR一致替代T0 |
| 计时与解释 | 两端CUDA同步计generate与同等只读取证，controller/provider/K加载传输setup另记，模型加载/哈希/落盘/审计不混入。进程顺序固定，观察与宿主内部记账成本未完全等价，因此墙钟/τ仅诊断，不设速度胜负门，不将差值归因于算法。若本仓同K复现原输出/接受，可排除本切片的相应运行时差异解释；不能直接证明R65→R66收益差由单独λ或校准数量导致 |
| 执行边界 | 新Run仅D:/Grok Bot/tmp/sd/round67，先冻结当前clean提交、计划/协议/driver/依赖原件、K/输入/模型与环境；CUDA可用并记录设备，满载等待、不回退CPU。整批900秒/外层960秒，原工程90/300秒门不改。失败保partial、不自动重试；正负或无差异都结束，不扫参/长度/模型，不重开旧矩阵/EMA支线 |

**结果与身份**：2bbdb85先于正式GPU预注册，实际Run冻结该clean提交。UTC2026-11-06 18:36:58—18:40:16（上海11月7日02:36:58—02:40:16），本仓端、原代码端及父进程均实际退出0，内部197.672061秒/900秒；48case全部通过128token、来源和完整T0。24/24对输出、实际接受及包含完整树的规范化trace相同，两端各自3次重复一致；没有首个离散分歧。两端模型配置/精度/TF32设置相同，K和全部生产/原源码字节不变，训练对/在线更新均为0。

**补充CUDA边界验证**：1,794对匹配frontier中，placeholder、final-norm probe、送入LM head的BF16输入共5,382对张量，原文件hash与读取的向量指纹闭合，按int16存储位比较均完全相同；逐frontier head argmax、全部实际forward行argmax也一致。该结论覆盖这些选中边界，未保存的全部中间隐藏状态保持未测。独立JSON/整数审计从真实forward argmax和完整树重走验证链，未将driver的true字段当作唯一证据。实际接受两端各1278，按qid3次中位数后均值53.25、τ1.725151；主判据支持H67。

**计时诊断与判读**：本仓/原端generate均值2.944377/2.893154秒，setup0.000192/0.006852秒，均为qid内3次中位数再跨8qid平均。顺序固定、宿主内部记账及K加载方式不同，墙钟不用于本次算法速度胜负推断。本次没有新ESP臂，R66原速度正结论仍绑定其原Run；R67回答同K移植一致性。下一步优先分离K的拟合因素，不能由一致性把R65/R66原差距单独归因于λ、训练样本数或是否含prefill。实际命令、逐qid值和不可变原件见既有运行记录第七十节。

## （九）R68：固定原401对的ridge单因素CUDA验证
**微观假设 H68**：保持R66原401个校准对、最终拟合流程及原推理端相同，ridge=.3相对10在当前8条评价切片同时增加实际接受数、缩短同步generate墙钟。ESP仅作共同背景对照。LLM权重冻结，离线K需拟合；研究对象是这一已出现的正则差异，而非重新搜索最优λ。

| **冻结项** | **协议与验收** |
| --- | --- |
| 原件与精度 | 只读R66 states.pt原字节SHA5921ad00a7f855ddc1133d1b8bdfbce7a384b2242a9e8ce39ccb7682ee25de0b、原K SHA17b62c5ff2df63df0d4201a66158b5d1be00c39f984fb45c37394c8a13da2e66、固定d91ed75的27源码闭包和原ridge函数；原码不改，不运行新λ网格或回写旧Run |
| 401对重建 | 依原10prompt输入索引拆fit8/choose序号4=qid123/report序号9=qid247，各组分别调用原frontier_pairs（组内按category/qid字符串排序），再严格cat(fit,choose,report)。BF16模型embedding→FP32，X=[emb(bonus)−placeholder;1]、Y=FP32(z_true−z_probe)；只decode相邻对，不补prefill。原ridge内才将X/Y转FP64、bias不惩罚，最终K转FP32。保存同序X/Y及prompt/配对索引，数量须401 |
| 拟合准入 | CUDA直接调用未改原ridge函数，先λ=.3重算并以int32存储位核对原K完全相同；失败保原差异并停止评价，不放宽容差/重排数据追通过。λ=10复用同一X/Y、同原函数，须有限FP32且形状2561×2560。K03评价继续读原K逐字节副本，不把新torch.save的封装字节差异当数值差异；K10新原件在评价前冻结 |
| 评价三臂 | 原ESP、原decoder+K03、原decoder+K10；只切frontier_gain enabled/path，online=false，K不更新。Qwen3-4B同14文件模型清单、BF16/eager、RTX5090 D v2、同R66真实8qid105/106/115/116/125/126/245/246输入tokens、128新token/T0/noEOS/seed123，K2/dynamic-pruned/BC30/naive、mean_prompt/λ=.1。当前8qid为历史已见材料，本拍是探索性敏感性对照 |
| 重复与边界 | 每臂2token预热1次排除，3次正式重复，repeat外层/qid中层，按(qid序号+repeat) mod3循环轮换，共72case/9216提交。原decoder/gain构造及K加载/传输setup单列，CUDA同步夹住原generate和相同只读T0取证；加载、401对重建/拟合、哈希/渲染/落盘/独审另计。两K前后文件SHA/实际GPU值不变，evaluation原gain_pairs/refits均0 |
| 主判据与质量 | 72项完整128token、来源和完整验证前向argmax链通过；实际接受Σmin(A,C)，τ排除prefill，legacy另列。每qid先取3次中位数，mean(接受K03−K10)>0且mean(墙钟K03−K10)<0才支持H68。报告相对ESP的两臂结果、输出一致率及重复一致性；T0不替代语义质量或独立AR逐字证明 |
| 分析与停止 | CUDA FP64以8qid配对bootstrap10000/seed6801，报告探索性95%区间，不把重复当独立n；K03/K10主比较与各自ESP背景对照分列。单个研究批总900秒/外层960秒，失败保partial、不自动重试，正负均停；原工程90/300秒门保持，不扫λ/模型/长度。重建及输出仅D:/Grok Bot/tmp/sd/round68，TEMP/TMP和缓存指向该根 |

**结果**：7433d90预注册后，UTC19:20:03—19:24:44在RTX5090 D v2完成唯一正式批次，两owner及父进程退出0，72项完整T0通过。K03位重现零差、两K全程冻结；H68预设点方向通过，但8qid配对区间跨无效值，保持探索性判断。K03/K10与ESP完整输出相同分别6/24、15/24；两K之间6/24，不称严格同输出或语义质量已验证。三repeat接受/输出一致，λ并非当前切片所有题都同向（4胜3负1平）。原件与拟合成本见既有运行记录第七十一节；下一项固定λ=.3检验两条校准输入的加入。

## （十）R69：仅加入两条校准输入的CUDA对照
**微观假设 H69**：固定ridge=.3、原推理端及配对定义，加入qid107/247这两条校准输入（324→401对）相对原8条校准子集，在同一评价切片增加实际接受并缩短同步generate墙钟。只检验这两条已命名输入的整体加入效应，不称一般样本量增长必然带来收益。

| **冻结项** | **协议与验收** |
| --- | --- |
| 数据与单因素 | 只读R68已冻结同序FP32 X/Y（pairs.pt SHA7413a7baa52dccdfc8123762d0b60eb2c27e346390f07f5a48902bef978aeb61）及原401条metadata。删qid107的35行[202,237)、qid247的42行[359,401)，保留[0,202)+[237,359)的324行，顺序和值均不改，不重新按i%10切组。保留原8个cal qid，与eval8不交；group标签只作原始来源角色，所有保留行均用于训练 |
| 拟合与来源 | 原ridge函数CUDA FP64、固定λ=.3、bias不惩罚、输出FP32，只拟合新cal8 K一次；cal10继续用经R68位重现的原K17b62…逐字节副本。原401对、R68fit/admission/complete原件均绑定，不重生成校准轨迹、不选择λ、不增加prefill对。Gram/目标是同一和平方目标，加入两条同时改变其数据内容和权重，当前不分拆这两种数据效应 |
| 评价与口径 | 原ESP/cal8/cal10三臂，除gain path外同原decoder，online=false。原R66实际8条tokens、Qwen3-4B BF16/eager/RTX5090 D v2、128token/T0/noEOS/seed123、K2/dynamic-pruned/BC30/naive与mean_prompt/λ=.1保持。各2token预热排除，3rep Latin循环，共72case；controller/K载入setup另计，主计时CUDA同步generate+同等只读T0 |
| 主判据 | 72项source/完整T0/128门全过，K前后文件SHA/实际GPU值不变，原gain_pairs/refits均0。qid内3rep中位数后，mean(接受cal10−cal8)>0且mean(墙钟cal10−cal8)<0才支持H69；ESP仅背景。实际接受Σmin(A,C)、τ排除prefill；输出一致率/重复一致性和原legacy分别报 |
| 分析与停止 | CUDA FP64按8qid配对bootstrap10000/seed6901，报告探索性区间；同一历史已见8qid用于敏感性消融，不称全球未见测试或参数确认。新根D:/Grok Bot/tmp/sd/round69，所有TEMP/TMP/cache/输出在此；总900秒/外层960秒，无自动重试，正负均停，不将拟合因素与EMA/Kalman/其它参数混改，原工程门保持 |

**结果**：56575bd事前预注册，UTC19:28:14—19:32:45唯一批次退出0；324稳定子序列与两K冻结通过、72项完整T0通过。H69点方向通过，但速度和墙钟区间含无差异，保持探索性判断。cal8/cal10相对ESP速度1.101153×/1.117310×；输出相同12/24、6/24，未测语义质量。下一项先分离prefill标签加入，再验证mask provider/EMA anchor；不据此泛化样本量效果或改判据。

## （十一）R69b：仅加入8个prefill标签的CUDA对照
**微观假设 H69b**：固定原8条校准输入、ridge=.3与原推理端，在324个普通decode相邻对后加入8个真实prefill→首decode标签，提高本评价切片的实际接受并缩短同步墙钟。该项分离配对边界，不重新生成校准轨迹或改变校准prompt。

| **冻结项** | **协议与验收** |
| --- | --- |
| 两组原件与准入 | 基底为R69已冻结324对子矩阵（pairs.pt SHA1a20790779c5a0ee8e7f9854701d1fb2a648c698d38c5c9f8b80e49b96a55425）与无prefill的K5ccfa1dd…；另一来源为R65真实calibration_pairs.pt（SHAea67d63501916c73dde20dd82af08c3a6a9399e19667950041445bb2f1eac574）。按qid与R65 from_forward=原source_pass+1一一对应324普通对，bonus/to_forward也须同。CUDA须验证R65 FP64值无损转FP32且共同324 X/Y存储位全同，失败停止此项，保差异、不补造或容差放宽 |
| 唯一变项 | 保持原324行及其顺序，新矩阵在末尾按原cal qid103/104/113/114/123/124/243/244顺序附8个from_forward=0真实标签，合计332；这区别于R65按prompt交错的原332顺序，作为新Run声明。只调用原ridge函数CUDA FP64、λ=.3、bias不惩罚→FP32，旧324 K继续原字节。共同prompt/数据及配对类型外的全部公式与推理保持 |
| 评价与边界 | 原ESP/without_prefill/with_prefill三臂，原decoder+FrontierGain只切path、online=false；原8eval tokens/128/T0/noEOS/seed123、Qwen3-4B BF16/eager/RTX5090 D v2、K2/dynamic-pruned/BC30/naive、mean_prompt/λ=.1保持。每臂2token预热排除，3rep Latin共72case；同步generate与setup分记，两K原件/实际GPU值不变、source/完整T0全部通过 |
| 主判据与停止 | qid内3次中位数后，mean(接受with−without)>0且mean(墙钟with−without)<0才支持；ESP背景、实际接受Σmin(A,C)、τ排除prefill。CUDA FP64配对bootstrap10000/seed6951，以8qid为单位报告探索性区间。新根D:/Grok Bot/tmp/sd/round69b，总900/外层960秒，无自动重试；失败/正/负均留证，已有R65/R66身份不回签，原源码不改 |

**结果**：8e435a8事前预注册，UTC19:35:54—19:40:28唯一正式批退出0；R65值无损转FP32，共同324 X/Y均逐位相同；新332与旧324两K冻结，72项完整T0通过。H69b点方向通过，接受差区间下界为0；所有三repeat输出/接受一致。两K均有相对ESP正点收益，但只在本切片、该λ与这8prefill标签下成立，不能把不同条件下的效应简单相加，也不是Kalman有效性证据。

## （十二）R70：保存轨迹上的mask provider与EMA公式CUDA核对
**独立子假设**：H70-ESP＝原ESP与本仓ESP在同一实际输入/提交链上的全部mask存储位相同；H70-EMA＝原EMA与本仓level=last在同一链上的mask、velocity、anchor及ref_norm存储位相同。两项各自验收，属于机制实现核对，不产生新的接受率/速度收益。

固定原d91ed75的27文件与本仓R67已冻结115源，核实际module origins；用同一Qwen3-4B BF16/eager/CUDA0 embedding，原R66 r0 ESP的8条真实128-token T0链为外生轨迹。每provider对子分别核初始化及每commit后状态：8×129=1032。ESP k2/mean_prompt/λ=.1/update=true；EMA β=.9999/step1/normalize=false/extrapolate=false/update=true，本仓level=last。mask/velocity/anchor BF16用int16位比较，ref_norm FP32用int32比较，另核有限值及向量指纹；host独有order0 level不在本项对照。

只做embedding lookup和逐token provider递推；actual model-adapter/HF forward计数必须0，新生成/拟合为0，模型加载/重放全段置fork_rng内恢复CPU/CUDA随机状态。原件/计划/协议/driver/模型清单先冻结，总300秒，外层有界，原工程90/300秒门保持。输出和TEMP/TMP/cache仅D:/Grok Bot/tmp/sd/round70。两个1032位门各自全过才记对应子假设通过；失配保首个状态和数据，不修改原算法凑一致。范围是已存ESP轨迹，不是EMA自身生成效果或全部输入的等价证明。

**状态**：已完成，两项子假设分别通过，各1032状态及全部字段存储位相同；forward/newgen/fit为0，随机状态恢复、owner实际退出0。独立IO审计复核2064事件/5160字段指纹；范围不升级为EMA收益。见运行记录第七十四节。

## （十三）R71：只切换EMA anchor的CUDA收益对照
**微观假设 H71**：同一本仓宿主和其余EMA参数下，原公式对应的last anchor相对order0 anchor增加实际接受并缩短同步generate墙钟。ESP提供背景对照；即使last胜order0，也须另看是否超过ESP，不能混称EMA获得加速。

固定R66实际8条tokens、Qwen3-4B BF16/eager/RTX5090 D v2、128/T0/noEOS/seed123/K2/dynamic-pruned/BC30/naive；三臂ESP mean_prompt/λ=.1/update，EMA两臂β=.9999/λ=.1/raw step1/normalize=false/extrapolate=false/update=true，唯一设置差为level=last/order0。level包括anchor初始化与逐token平滑规则的固定差异，velocity递推/精度不变。调用现有工厂/类，不改生产公式、不加载K或校准材料。

每臂2token预热排除，3rep Latin循环共72/9216正式提交，qid内取3次中位数。主门全部source/完整T0/128通过，mean(接受last−order0)>0且mean(同步墙钟last−order0)<0；ESP子对照分别报告。CUDA FP64按8qid配对bootstrap10000/seed7101，重复不扩n，输出一致率、实际min(A,C)接受/τ与legacy另列。provider构造setup在外，初始化/逐token更新成本在generate内；加载/hash/IO/独审不含主墙钟。

新根D:/Grok Bot/tmp/sd/round71、总900/外层960秒，源/模型/输入/计划/协议/driver事前冻结，失败保partial无自动重试，正负均停。当前历史已见8qid是有界机制消融，语义质量和独立AR仍未测；不扫β/长度、不以本项判断全部EMA或Kalman。**状态**：72项完整T0/9216提交通过。H71未获预注册联合收益支持；last−order0接受-11、墙钟+0.402614秒、速度0.887095×。探索性95%速度[0.849912,0.930506]、接受[-14.125,-7.875]。last/order0相对ESP分别0.904454×/1.019569×；anchor子对照与ESP收益分开。8qid不以重复扩n；正负均收束，不扫β/长度。完整数据见运行记录第七十五节。

## （十四）R72：同401对的常量残差与完整frontier gain对照
**微观假设H72**：固定原401校准对和原ridge .3完整K后，完整仿射修正相对仅常量残差同时增加实际接受、缩短同步generate墙钟。ESP为独立背景对照。检验的是bonus条件化完整映射相对最佳常量拟合的净效果，不是保持完整拟合截距而仅去掉A。

校准原401 FP32 X/Y仅从冻结R68原件读取，不加入评价标签。CUDA FP64对Y求均值后转FP32得b0，bias K的全部A行为0、末行为b0；GPU验证全部401个X@Kbias逐字等于b0。full直接复用原17b62…K.3字节；两校正臂都用未改原FrontierGain完整dense u@K和head路径，不声称常量部署已优化。报告两者校准MSE仅作诊断，不代替留出收益。

评价继续固定8条R66 tokens、Qwen3-4B BF16/eager/RTX5090 D v2、128/T0/noEOS/seed123/K2/dynamic-pruned/BC30/naive/ESP mean_prompt λ=.1；ESP/bias/full各2token预热排除、3rep Latin循环共72/9216正式提交。provider/K加载setup另列，同步generate含实际校正与LM head开销。全部完整T0与来源/128门通过且full−bias实际min(A,C)接受均差>0、墙钟均差<0才支持H72；ESP子对照另报。qid内取3次中位数，8qid CUDA FP64配对bootstrap10000/seed7201，重复不扩n，legacy/τ/输出一致率另列。新根D:/Grok Bot/tmp/sd/round72，900秒/外层960，失败保partial不自动重试；不新增模型、长度或参数扫。**状态**：H72按预注册点方向通过：接受均差+9，墙钟-0.292742秒，速度1.102720×；探索性95%速度[1.053091,1.159804]、接受[5.375,13]。72项完整T0/9216正式提交全部通过；正负均结束，不自动扩样或扫参。见运行记录第七十六节。

## （十五）R73：同一批保存CUDA计时的评测口径核对
只分析R68原72个正式case，不新生成、不加载模型、不bootstrap。事前问题：K03与K10按内部prefill_time+decode_time与严格CUDA同步generate两种口径的qid级均值排序，是否严格反转；tie单独记录。冻结原独审c8c2822e…及各result SHA，不与其它Run混合。每qid/arm先取3次中位数再跨8qid均值，CUDA FP64计算和/差/median/ratio/排序，记录三臂全排序、qid级同/反/平及72个原始差额；缺失/非正内部阶段或墙钟拒绝，差额负值保留，不截断、不加入setup或挑样本。

计时差额仅称边界/同步与Python观察和记账等合计，不能全部归为CUDA异步；阶段内部和也不是端到端速度。新根D:/Grok Bot/tmp/sd/round73、分析上限60秒，CUDA可用及实际RTX5090设备写明，原source/计时不改。结果只回答已有记录的口径敏感性，不是第二次模型性能运行。**状态**：已完成；K03−K10内部−.123663秒、同步−.123797秒，两种口径均K03<K10<ESP，8qid符号同向8/8，主排序未翻转。保存72项旧记录CUDA FP64分析，无新模型/生成/bootstrap，见运行记录第七十七节。

## （十六）R74：现有标量自适应Kalman相对ESP的同切片对照
**微观假设H74**：本仓既有KalmanMaskProvider默认online配方相对ESP，同时增加实际接受并缩短同步generate墙钟。它有P/Q/R/K递推，区别于合作者静态ridge K；不把静态K正结果直接转记为Kalman成功。

只调用现有实现：online/forget=.95/r_min=1e-6/q_min=1e-8/q_scale=.01，不扫参数、不改公式。记录实际mask/state/innovation张量为CUDA；P/Q/R/K为实现内Python host标量，包含innovation到标量的CUDA同步，全部prompt初始化/逐token递推成本计入generate。记录有限的末态k/q/r/p及实际provider身份，要求k在[0,1]、p非负、q/r正。ESP mean_prompt/λ=.1/update=true，其他设置固定为R66同8输入、Qwen3-4B BF16/eager/RTX5090 D v2、128/T0/noEOS/seed123/K2/dynamic-pruned/BC30/naive。

两臂各2token预热排除、3rep Latin循环共48/6144正式提交。主门全部完整T0/来源/128通过，kalman−ESP的qid级实际接受均差>0且同步墙钟均差<0；qid内3次中位数、8qid CUDA FP64配对bootstrap10000/seed7401，τ/legacy/输出一致率另报。新根D:/Grok Bot/tmp/sd/round74，总900/外层960秒，失败保partial不自动重试。只有本小时剩余预算足以在20:10Z前结束才启动，否则如实记未执行；不重开旧writing32/256、旧Kalman配方或长矩阵。**状态**：H74未获预注册联合收益支持：接受均差-7.375，墙钟+0.295656秒，速度0.913915×；探索性95%速度[0.867172,0.958170]、接受[-11.25,-3.75]。48项完整T0/6144正式提交全部通过；正负均结束，不自动扩样或扫参。见运行记录第七十八节。

## （十七）R75：冻结K在扩大拟合留出上的CUDA复核
**微观假设H75**：保持现有完整frontier K与最佳常量K冻结，扩大到未参与本次校准/近期选择的32条输入后，full分别相对ESP和最佳常量都增加实际已提交接受并缩短同步generate墙钟。两子对照均报告；“稳住”另外要求两个速度比95%配对区间下界>1且接受差下界>0，不用单个有利子集代替总结果。历史480题曾经被使用，因此这是相对于本次拟合与近期选择的预注册留出复核，不称全局未见/跨数据集确认。

| **冻结项** | **协议** |
| --- | --- |
| 数据 | 原SpecBench question.jsonl SHA7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21；按预先题号规则固定translation161–168、summarization248–255、math_reasoning401–408、rag481–488，各8条。与cal10、R64八题、R66–74八题及writing81–88共34题在qid、原文本、完整渲染与token序列均不交；原480的其它切片不算额外独立数据。不看模型输出换题 |
| 输入与参考 | D:/Grok Bot/tmp/sd/round75/inputs.json SHA4ad97bc22eafce04bd1b5c6f68e02a2c14425153479f0de11592eefc7d31ba09，32题总12401输入token、最长1351，不截断/不silent skip。CPU只做tokenizer与IO预处理，R66原18题重渲染逐token相同；实际template_kwargs={}。实际输入/Latin遍历顺序固定为math_reasoning401–408→translation161–168→summarization248–255→rag481–488，与输入字节同序。参考独立quality-references.json SHAfd1dd0506673ebcb598fe2239961e3cee3cf8fbcce9401fd522ba4ea5217b02e，不进入模型输入 |
| 模型与三臂 | 同一本地Qwen3-4B、14成员manifest20ff496548750bd139b280a544bcab83424f7bcf691b22c7d25c41c52a120167、BF16/eager、CUDA0 RTX5090 D v2；共同当前host与已冻结R65 FrontierPolicy，所有臂同等捕获/审计。ESP mean_prompt/λ=.1/update=true；bias用R72最佳常量K f9374de94a7a7f8c359182c022eb7dec0cfbb45f7c266a3d9eb36a4342ed945c；full用R66按原码新校准的K17b62c5ff2df63df0d4201a66158b5d1be00c39f984fb45c37394c8a13da2e66。两校正臂均走dense u@K及LM head，常量部署未优化；不refit/选λ/更新K，原码与生产源不改 |
| 预算与顺序 | K2/dynamic-pruned/BC30/naive、128/T0/noEOS/seed123。每臂2token预热一次排除，repeat外/qid中、按(qid序号+repeat)mod3循环顺序，共288正式case/36864实际提交。qid内取3次中位数，4个任务各8题固定构成，不将重复当独立题 |
| 计时与门 | fresh provider构造/K加载setup另列，同步generate含prompt初始化、逐token状态更新、实际校正/LM head及同等完整T0取证；模型/哈希/渲染/落盘/分析另计。每case完整来源/128预算/实际forward argmax提交链通过，K原字节与GPU值前后不变。accepted=Σmin(A,C)，legacy、排prefill的平均与合并τ、输出一致率、重复一致性、全部失败均报告 |
| 统计 | 32qid为配对单位，在每个任务内8qid重采样、保持4任务等权的CUDA FP64分层配对bootstrap10000/seed7501；主比为baseline中位墙钟之和/full之和，接受/墙钟为qid配对均差。报告整体及四任务，不按结果删题；两子对照点方向及95%区间分别列，不把保留同T0当成语义等质 |
| 执行边界 | 本批1200秒、外层1260秒；确认torch.cuda.is_available且记录设备后执行，不CPU替代。此前只用旧q105做五臂各8token的CUDA适配准入（≤180/外层240秒）：完整T0、实际base placeholder/bonus与首slot校正的调用归属、有限值与K冻结须过，正式32题不得用于适配调参。失败保partial不自动重试或缩题后冒称完成。所有命令TEMP/TMP先设且保持D:/Grok Bot/tmp，源码缓存/临时材料/输出仅本会话子目录，原工程90/300秒门保持 |

**状态**：正式CUDA已完成，288项/36864全T0，H75两点判据及95%稳定目标均通过，full/ESP1.084824×、full/bias1.096119×；来源/命令/原件见运行记录第七十九节。首版纯元数据预检发现调度EXPECTED/protocol与已冻输入顺序不一，在GPU前只统一顺序并冻结v2，原题号/tokens/K/判据/预算不变，旧注册文件与candidate保留，不回签。数据预处理回执data-semantic-audit.json SHAf031baecf4b2dad6b84ea9d919f4787ebb240582306b8c85d9678deee064f068。本批只回答冻结K的留出稳健性，不追原480正表或重开旧矩阵。

## （十八）R76：冻结完整K之上分别叠加EMA与Kalman
**微观假设H76-EMA/H76-Kalman**：在冻结full K上，仅把base provider分别换成原EMA-last或现有online Kalman，是否相对fresh full+ESP同时提高实际接受并缩短同步generate。两个假设各自验收，无论R75正负都按原定小批执行，不只选较有利组合；不新增参数/长度搜索。

为给质量和交付留预算，事前固定R75每类首4条：161–164、248–251、401–404、481–484，共16；inputs-r76.json SHAdb869a0f0c477d484bd492b2fbb74c9f2786c6610518ebdbd9af3cddf98f415e。实际输入/Latin遍历顺序同样为math_reasoning→translation→summarization→rag、各类前4，与冻结inputs-r76字节同序。三臂fresh full、full+EMA-last、full+Kalman全部同host/FrontierPolicy与原17b62 K，不用R75跨批墙钟替代共同基线。EMA β=.9999/λ=.1/step1/normalize=false/extrapolate=false/update=true/level=last；Kalman online/forget=.95/r_min=1e-6/q_min=1e-8/q_scale=.01。其它模型/树/128/noEOS/T0/seed与R75相同，三臂各2token预热排除、3rep Latin共144正式case/18432提交。K是按ESP placeholder特征拟合；更换provider改变placeholder/innovation分布，本实验只回答冻结K直接组合的效果，不据失败否定再校准组合，也不事后重拟合追正。

provider构造setup在外，prompt初始化/EMA递推/Kalman实际Python标量P/Q/R/K与CUDA level/innovation同步及K校正均计入generate；末态finite与设备按实际记。完整来源/T0/预算/K冻结为门，两个候选分别须接受均差>0、墙钟均差<0才支持对应点假设。16qid按四任务各4分层CUDA配对bootstrap10000/seed7601，qid内median3；报告95%探索区间，并给两个方法比较的97.5%保守区间，不把挑中一法称整体成功。输出一致率/legacy/平均与合并τ全列，语义判断另列。总900/外层960秒、输出round76，失败保partial不重试；21:10Z后不新派模型任务，已执行结果不填假完成。

**状态**：144项/18432全T0完成，两个叠加假设均不获支持；EMA/full0.878205×、KF/full0.904080×，各0/16/0。实际证据见运行记录第八十节。本比较不把静态K重新命名为Kalman，也不修改合作者或生产算法。

## （十九）R77：有真实数值参考的小规模语义检查
同128性能批先保存每题生成token与原始decoded文本，严格区分思考段和最终答案。RAG参考混有多要点/多答案，翻译/摘要的词面相似不当语义正确；不统一any-alias判对、不伪造judge评分、不以AR一致或T0替代语义。

独立质量小批固定math_reasoning401–404，输入inputs-r77.json SHA2973aee0b118dda77045bc96470b11b2ea97764702c255d3f5eb4f08c0692c31，真实参考####金数依次1000/5/306/7。比较ESP、同dense最佳常量、full、full+EMA-last、full+Kalman，统一同模型/原模板与参数，max512/EOS=true/seed123、每臂每题1次共20case，不加预热、不进入128速度主表。预算480/外层540秒，不据结果再延长；这是本拍新增的质量覆盖，不恢复旧writing长度候选。完整实际提交链T0覆盖变长与首EOS裁剪，若少于512须实际EOS终止，终态未知不补零。

事前数值代理：gold只解析唯一行首####纯数值、Decimal去千分逗号精确比；参考[GSM8K原数据处理](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py)的####约定，预测抽取为本仓自定义，不称官方GSM8K评分。预测只取生成</think>之后的finaltail；无闭合、空尾、预算截断记not_evaluable且保留全部4题/20case分母。若有boxed只接受唯一可解析纯数值，分式/多box不静默回退；无box才取finaltail最后合法decimal，支持正规千分逗号。EOS完成且数值可解析时记correct/wrong；同时报告最终答案出现率、未完成/不可解析数和包含全部分母的数值EM，原文本保留供独立复核解析歧义。q404参考rationale有文字笔误，题干与最终gold7已独立核算一致，原件不改。

另在预定两道翻译161/162、两道摘要248/249上，对已有输出做匿名、基于源文/参考的有限文本核查，记录覆盖/不支持内容/截断，明确是AI文本审阅而非验证过的人工或模型judge评分；无可判断最终答案保unknown。质量四题不足以证明语义非劣，开放任务未有可信评分器的范围保未测。所有正负、截断及未知写现有运行记录和本文，不增加研究文件树。**状态**：20项/10240提交执行及独审完成，实际均512且无EOS；五臂各NE4/4，评分规则通过而语义结论未闭合，未延长。128预算24匿名AI核查亦全unknown。见运行记录第八十一节。

## （二十）R78–R79：完整最终答案的独立质量协议
**微观问题与预注册**：R77在默认思考/T0下20项均无EOS，不能判语义质量。本批只检验显式非思考模板能否形成可评分的完整答案，以及冻结完整K相对ESP的数值答案表现；不是把R77改签为通过。Qwen3官方模型卡说明可用`enable_thinking=False`关闭思考，并警示思考模式下greedy可能退化；这只是协议选择依据，不是此前未完成原因的因果证明。[直接来源](https://huggingface.co/Qwen/Qwen3-4B#switching-between-thinking-and-non-thinking-mode)

| **冻结项** | **协议与验收** |
| --- | --- |
| 数据与隔离 | R78开发准入固定q401，两臂各1次；R79正式固定数学qid409–416、ESP/full两臂各3次，共48项，按qid序号与repeat轮换臂顺序。原SpecBench480的SHA为7dd26882f31ada100c2aa7447019b5349bddc55697021acbc5643fdc8740aa21；原题干不加system、答案格式或提示，原gold仅评分读取。409–416的qid/题干与本次K校准及近期32题不交，但历史480已见，不能称全局盲测。重复不扩大独立n=8 |
| 模型与处理 | 同本地Qwen3-4B/BF16/eager、RTX5090 D v2 cuda:0；同ESP mean_prompt/λ=.1/update、k2/BC30/dynamic/pruning/naive，full复用R66新校准冻结K 17b62c5f…且不重拟合。只在渲染阶段显式enable_thinking=False；生成仍temperature0/seed123/max256/stopEOS，实际模型/Tokenizer EOS并集。模板把空闭合think放入输入prefix，评分不得伪造生成的think闭合。无预热，不以本批声明稳定速度 |
| 完整答案准入 H78 | q401两臂均须实际终端EOS、非空可解析finaltail、完整来源/T0、实际close/退出0；答案正确与协议可评分分列。任一不满足即停止正式R79，不改预算/题目/采样追通过 |
| 数值代理与 H79 | 输入空闭合think须逐字核对；生成再次开启think则要求实际闭合，否则NE。实际EOS仅末位，生成不得新开角色；唯一合法boxed十进数优先（多框、分数框或畸形不回退），否则finaltail最后合法十进数；Decimal精确比较原gold的唯一####标量。所有8题分母保留，缺EOS/格式不合法为not_evaluable，不能填wrong或剔除；正确/错误/NE与两臂配对输出分别列。H79仅为16个qid×arm均可评分且重复稳定的协议可执行性，不是语义非劣或普遍质量结论；自定义代理不是官方GSM8K评分 |
| 计时与停止 | 同步generate实际墙钟/输出长度描述，按每qid/arm三次中位数，不与旧128token速度拼接，不据小样本正确数声称非劣。R78内部180/父进程240秒，R79内部900/父进程960秒；准入失败或正式失败保partial，正负均不自动延长/扫参。原工程90/300秒门不变 |
| 来源与执行 | 新外部round78/79冻结clean HEAD、当前计划、驱动/协议、原输入/refs、模型14件清单及旧adapter/K；TEMP/TMP恒为D:/Grok Bot/tmp，全部输出/cache在D:/Grok Bot/tmp/sd。torch.cuda.is_available须真，日志设备具名，模型全部close与来源末核、父进程实际退出后才收束；合作者原码与本仓生产字节不改 |

**当前状态**：预注册，尚未运行。下一项差分只考虑原FrontierGain.online的有界因果更新合同：其256对触发与向离线K收缩重拟合不同于本仓P/Q/R标量Kalman；先读源码、另冻结判据，不能把无refit的短请求当自适应收益。

## （二十一）原FrontierGain.online：先核因果更新合同
**未排除的具体差异**：原d91ed75的`FrontierGain`累计XᵀX/XᵀY，每256个已获标签的pair解一次FP64正规方程；有离线K₀时最小化‖XK−Y‖²+λ‖K−K₀‖²（截距行不惩罚）。`reset()`只丢pending及本prompt applied，保K/累计统计/计数，原decoder在prompt循环外持有。它与本仓标量随机游走KalmanMaskProvider的P/Q/R递推不同；R70的ESP/EMA逐状态零差不覆盖该自适应机制。原源码只读：[固定版本](https://github.com/jasonchengqs/speculative_mtp/blob/d91ed75145801d9263281d2938a7d0da5841b2bf/denoise_mtp/src/spec/frontier_gain.py)。

**CUDA代数合同预注册**：只重放R68已保存401对X/Y（SHA7413a7ba…）与原K₀（17b62c5f…），不载LLM、不生成新token。用明确标注的identity head/norm把固定创新送原未改`update→label`，online=true/ridge=.3/refit_every256。验收：前255次label不改K，第256次恰1refit，后145只累积；update不消耗未来label、同pending只消耗一次，reset丢未获标签且保累计；独立CUDA FP64批量带prior方程与原顺序计算的FP32 K最大绝差≤2e−5、相对L2≤5e−5，XᵀX/XᵀY最大绝差≤2e−8，截距惩罚为0；原源码闭包/输入/计划SHA与clean HEAD始末一致。内部150/外层210秒，设备具名，失败保未知，不改阈值。只检验已知校准数据上的调用/代数合同，不是新真实在线收益或无泄漏泛化证明。

**当前状态**：未执行。若合同成立且时间足够，再预注册同初始K、同原类的static/online真实CUDA对照；每repeat重置总状态、每prompt只reset pending，必须实际触发refit并把累计/求解时间计入generate。状态跨prompt耦合，不能把qid当独立bootstrap单位；没有refit不算适应收益，不以本仓默认Kalman负结果代替该方法测量。

# 五、历史Run收束与判读合同
| **材料** | **当前状态与收束** |
| --- | --- |
| EMA五方法11 Run | 固定H02报告字节与已登记SHA吻合，11条身份元数据可核。已登记的checkout候选路径及固定归档同路径未取得66个Run成员，H02本体没有执行根字段，外部原件位置仍unknown。validator原字节已从H01明确原运行根取得且匹配同一SHA，但不据此补齐H02实际调用环境、其余源码/配置或模型链；保历史完成事实，不冒充原生重验 |
| ESP完整复验H01 | 沿报告明确execution_root取得3个原Run，18个Run声明成员、59个宿主成员及16个Study声明成员（不含RPipe）均匹配保存SHA。实际source_refs及其3个引用原件均命中运行摘要，关闭其与上游protocol的原字节绑定；固定Git报告仍保独立归档身份。报告生成器原字节绑定仍未闭合；权重/tokenizer、内嵌RPipe源码及重新执行原生质量门未核 |
| OnlineVerify / Aligned旧比较 | 指定旧输出根当前不在，32条真实模型负结果目前属运行记录级证据，原终态、逐token与模型/输入摘要未重新核得。原记录段未写明backend或EOS停止设置，两者保持unknown，不据当前默认值反推。保负结果及缺口，不为补通过重跑同配方 |
| 8B冻结K旧比较 | 指定10份原制品与3个固定Git源码对象字节均与已登记SHA吻合，原source HEAD、prompt摘要和qid序列可核。stored argmax布尔不等于完整token链重验；fit可被旧流程复用，header/source HEAD不构成每份fit的绑定。fit自身model/source/校准文本哈希仍缺，不能事后补签；收益仍未确认 |
| 未完成范数/配对/续跑 | 登记原计划、实际完成/未完成和退出当前队列理由；排程取消不等于旧Run成功 |
| 历史来源缺口 | 四项cross-task原tool/plan/runtime精确字节仍缺，受影响读端严格拒绝、保unknown，不回填、不用当前helper替代；不阻塞不依赖这些文件的EMA/Kalman核验 |

固定Git对象与实际运行原件按各自已登记摘要核对，不以归档字节替代运行字节。实际原件命中时只关闭对应来源边，缺失或不匹配继续保留unknown，不转换换行或用语义相同的JSON补齐。来源审计不改变第二章的方案处置。

以下命名缺口全部保持 **unknown / still-refuse**，只约束依赖这些来源的操作。H02定位与H01两项生成器绑定分别登记，不混为四项cross-task原件，也不影响第二章已冻结的研究处置。

| **缺口** | **所需原身份** | **已知定位边界** | **关闭条件** |
| --- | --- | --- | --- |
| H02执行根与运行链 | 实际根值未登记 | H02报告及固定旧PLAN均不提供实际根。11个checkout候选根未命中只限已查位置，不从H01推根 | 取得真实旧运行头、派发回执或来源清单，明确实际根并对应11 Run及66成员保存摘要 |
| H01生成器与复用汇总器 | analyzer `cbb00e6edd5418a033f41e3472c0979803849af8fd5cdba422ad9b8de0c55363`；aggregation `5db751c8d9973ffc5d29ef28120bef5e9addea11c383e5f473ea9da3aa8af519` | 已知phase入口为56字节再导出，SHA为8e5e1829…且不匹配analyzer。两个原成员/producer提交均未定位 | 分别取得匹配两摘要的原始源码及真实成员/版本关联。阶段入口、validator或当前readout不替代生成器 |
| 运行矩阵 `studies/output_space_cross_task/docs/execution_v1.json` | `708bf37bc5160ab6cc2c0787dd89e773b0fd60e518fdd9b272a6b114756dc254` | diagnosis.source_binding.matrix_sha256只声明原身份。已取固定归档be50f497…与目标708bf37…不同 | 保留原矩阵未改字节并命中预期SHA，连同其真实producer/原成员绑定 |
| 原生导出器 `studies/output_space_cross_task/check_hypotheses.py` | `0f8d644e049eb6f9a7cf2e80b6d43de96ec9731023bd61bc55421363c659ccf5` | 原Run source packet声明该SHA，保存export索引的producer_sources为空 | 原导出器字节命中SHA，并具备正式读端要求的原Git/member身份 |
| 跨任务工具 `tools/diagnose_study_windows.py` | `ba72f70420e6309a412f1deaf1f2fd28b9c02767ccc71b4cc1248b6235710948` | diagnosis.source_binding.tool_sha256有声明，cross索引没有绑定原工具 | 原工具字节命中SHA，并有独立支持的原成员/版本关联 |
| 原计划 `studies/output_space_cross_task/docs/PLAN.md` | `07abd854854816d51c4770939e14e095e6f9931dbe62698528f6e58d19475df9` | 旧Run与diagnosis的声明一致，实际原计划字节未取得 | 原计划未改字节命中SHA及原成员/版本。不重建换行或拿当前中文计划补签 |

原报告中的声明字段、可读的归档和已命中的其他来源边均不能替缺失原件。cross原路径报告也与保存索引绑定的固定报告字节不同，声明可读不代表报告身份已闭合。只在出现明确原定位或匹配原件时核对应边，不按文件名猜源、重扫既查位置或为缺失来源补跑GPU。正式拒绝证据保原提交身份，本拍未重跑拒绝driver。

逐Run核源码/声明、输入/模型/tokenizer、配置/seed、设备/dtype/backend、计划与实收样本、终态、质量门、指标及来源回执。只生成新收束摘要，不覆盖旧Run。原执行与失败从 [运行记录](<运行记录 Run Record/INDEX.md>) 定位，开发决定归日期开发记录。

T0门是每个实际提交token对应验证前向argmax，覆盖prefill、bonus、EOS及预算裁剪后完整链。独立AR一致率只作诊断，T0不推广为T>0分布保证。旧full/argmax协议保各自判据，不替换失败门。BF16相对独立AR的算子机理排查仍永久停止。

τ为排除prefill的解码提交数/decode_calls，零调用记缺失；逐prompt均值与合并计数之比分列。accepted与committed可能裁剪位置不同，保冻结口径。配对固定qid、输入、模型、停止和预算，保胜平负、缺行与不可比项，不把同prompt多树/长度/臂当独立样本。

实际加速需同质量、同输出条件完整计时，冻结预热、CUDA同步、加载边界、臂顺序与资源。缺计时不填零，host-clock、cProfile及关闭计时占位不是速度证据。工程tiny模型、小张量、重绘与计数复核只证明对应合同，不产生新真实模型收益结论。

# 六、停止与下一步
既有候选执行处置按第二章结案表，来源缺口及关闭条件按第五章台账；本小时R68–R74已按第四章冻结协议收束，R65/R66/R67亦保持原结果；R75–R77三批已按冻结协议收束，后续优先制定能获取完整最终答案的独立质量协议，再决定更广数据或匹配再校准组合的研究价值；当前不自动扩长度、改思考配置或追加模型，不自动接旧矩阵或复跑已通过项。只核明确原定位，无定位继续unknown/still-refuse，已通过合同不重复派发，不把新研究替代旧原件或重签旧结案。

研究问题、对照或预算变化时更新本文，不另建平行计划。新算法不是已有结案的等待条件。工程只处理直接阻塞当前结论或来源保全的具体缺口，其余事项见开发计划。
