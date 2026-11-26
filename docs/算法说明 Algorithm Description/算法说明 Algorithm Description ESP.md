# 算法说明 Algorithm Description ESP

## 当前 input 接口与 BC 构树顺序

2026年11月27日确认。当前只考虑 input，保留 ESP 的公共 block、树注意力、候选选择、验证与缓存处理。探测槽统一读取当前共享水平向量 M，不按槽或树深度额外读出 M+hV。只有探测向量的状态更新发生变化。

| **结构** | **固定增益** | **Kalman 增益** |
| --- | --- | --- |
| 零阶水平 | EMA，按原 ESP 规则对齐的基线 | 零阶 Kalman |
| 水平与趋势 | Holt | 水平—趋势 Kalman |

EMA 与 ESP 的水平递推在 prompt 均值初值、更新 token、顺序／时点和权重一致时对齐，本轮不重复列为两个独立算法臂。原 ESP 来源版本和数值状态仍记录，历史精度或调度差异不冒充算法收益。差分 EMA 若 V 不进入 M 更新或实际读出，对探测无作用，退出当前四臂；M+hV 的显式外推保留为后续另一项改动。

正式提交的 token 给出 input embedding z，依位置顺序更新一次。正常候选节点始终输入自己的 hard token embedding，附属槽输入 soft M。M 更新不重算旧 logits，不修改旧候选。额外 probe 融合、output、Jason 组合、树预算重分配和 Q/R 在线更新均不纳入当前比较。

### 预算先于候选分配

先固定 block 总位置预算 B、每个正常节点附属槽数 k，再得到候选容量 $N=\lfloor B/(1+k)\rfloor-1$，其中 root 也配 k 个槽。BC30、k=2 时：1 个 root＋9 个 hard 候选＋20 个 soft 槽＝30 个新增位置。缓存历史前缀不重复计入。

容量不是候选池大小。先从 probe 分布提出候选，按累计探测分数和 Top-1 深入扩展约束保留预算内连通树。各层留几个由评分及策略决定，7＋2 是附录例子，不是 BC30 固定层宽，也不保证全局真实概率最优子树。原树节点用 hard embedding 验证，附属 soft M 探测后续候选。[论文 §3.3—3.4](https://arxiv.org/html/2603.17942v2#S3.SS3) · [附录 D](https://arxiv.org/html/2603.17942v2#A4)

当前沿用选定基线的 k、BC 与层宽配置，不联合优化每节点槽数。历史单槽 BC30／branches=[14] 的实验与双槽教学例子分开。EMA 与 Holt 的详细更新见 [EMA 算法说明](<算法说明 Algorithm Description EMA与Holt.md>)，Kalman 与校准见 [卡尔曼算法说明](<算法说明 Algorithm Description Kalman.md>)。下文保留论文机制、验证合同和历史候选，后续研究表述不覆盖本节的当前范围。



飞书学习笔记和外部协作文档需要对应访问权限，仓内算法与计划可独立阅读。
