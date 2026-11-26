# 算法说明 Algorithm Description EMA与Holt

## 当前 input 算法：EMA 与 Holt

2026年11月27日确认。当前只考虑 input，保留 ESP 的公共 block、树注意力、候选选择、验证与缓存处理。探测槽统一读取当前共享水平向量 M，不按槽或树深度额外读出 M+hV。只有探测向量的状态更新发生变化。

| **结构** | **固定增益** | **Kalman 增益** |
| --- | --- | --- |
| 零阶水平 | EMA，按原 ESP 规则对齐的基线 | 零阶 Kalman |
| 水平与趋势 | Holt | 水平—趋势 Kalman |

EMA 与 ESP 的水平递推在 prompt 均值初值、更新 token、顺序／时点和权重一致时对齐，本轮不重复列为两个独立算法臂。原 ESP 来源版本和数值状态仍记录，历史精度或调度差异不冒充算法收益。差分 EMA 若 V 不进入 M 更新或实际读出，对探测无作用，退出当前四臂；M+hV 的显式外推保留为后续另一项改动。

正式提交的 token 给出 input embedding z，依位置顺序更新一次。正常候选节点始终输入自己的 hard token embedding，附属槽输入 soft M。M 更新不重算旧 logits，不修改旧候选。额外 probe 融合、output、Jason 组合、树预算重分配和 Q/R 在线更新均不纳入当前比较。

### 变量与更新

位置 t 指正式提交 token 的次序，不是模型 forward 次数。$z_t=E(y_t)$ 是正式 token 的 input embedding。M 是平滑水平，V 是每 token 趋势。两者都不是模型权重。

EMA 使用固定新观测权重 α：

$$M_t=(1-\alpha)M_{t-1}+\alpha z_t.$$

Holt 先预测当前位置，再校正水平，最后更新趋势。β 表示旧趋势保留权重：

$$M_t^-=M_{t-1}+V_{t-1},\qquad r_t=z_t-M_t^-,$$

$$M_t=M_t^-+\alpha r_t,\qquad V_t=\beta V_{t-1}+(1-\beta)(M_t-M_{t-1}).$$

趋势更新等价为 $V_t=V_{t-1}+\alpha(1-\beta)r_t$。新 M 估计的是当前位置，不是从当前 z 再向前走一步。趋势通过旧 V 进入 M 的预测，当前槽只读 M 也能受到趋势影响。

差分 EMA 则平滑 $z_t-z_{t-1}$。若水平仍独立按零阶 EMA 更新、探测仍只读 M，V 不影响实际输入，因此不属于当前有效的一阶臂。Holt 和差分 EMA 的差别不能仅归为最后一行，水平更新是否带入旧 V 也需说明。

EMA 可解释为指数加权水平估计。Holt 是局部水平加趋势的指数平滑结构，也有创新状态空间表述；这提供建模来源，不保证 embedding 轨迹符合假设或提高接受量。[指数平滑与 Holt](https://otexts.com/fpp3/exponential-smoothing.html) · [状态空间框架](https://robjhyndman.com/papers/hksg.pdf)

### 初始化、calibration 与 warm-up

| **概念** | **作用与数据范围** |
| --- | --- |
| 初始化 | 为每条新样本确定起始 M，Holt 还需 V，Kalman 还需 P。它是起始状态及其产生规则，不等同于某一种数据处理方式 |
| 离线 calibration | 尽量用较大样本、多个数据集或类别估计固定增益、Q/R 和初始状态／协方差的规则或参数。保留各来源身份、分布和数据划分，不能把已看过的 eval 当新拟合集 |
| 当前样本 warm-up | 用该样本推理前已可取得的 prompt、prefill 或明确记录的 preview 信息预热状态。preview 若有额外模型调用或信息须单列，不使用尚未产生的未来 verified 标签 |

calibration 与 warm-up 可以组合。离线学习的可以是根据当前 prompt 产生 M0、V0、P0 的规则，不必给所有样本同一向量。需比较的初始化包括沿用 ESP 的 prompt 均值、当前样本顺序预热，以及离线校准规则。先固定初始化比较四臂，再固定算法比较初始化，不一次混改。

Kalman 开启后，状态结构仍为零阶或水平—趋势，修正权重改由协方差计算，见 [卡尔曼算法说明](<算法说明 Algorithm Description Kalman.md>)。历史外推、旧实现及 output 讨论保留在下文，按原日期和版本阅读，不作为当前四臂的读出规格。



飞书学习笔记和外部协作文档需要对应访问权限，仓内算法与计划可独立阅读。
