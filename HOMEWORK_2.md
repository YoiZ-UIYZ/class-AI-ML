# ResNet

约定：块为 $x_{l+1} = x_l + F(x_l, W_l)$，shortcut 为无参数恒等映射；$\delta_l = \partial L/\partial x_l$ 为列向量，$J_l = \partial F(x_l, W_l)/\partial x_l$。

## 1. 理论推导

**前向展开**（核心：深度方向是加法，不是乘法）

$$
x_{L} = x_{l} + \sum_{i=l}^{L-1} F(x_i, W_i)
$$

**对浅层输入的梯度**（求和项不能丢，每个 $x_i$ 都依赖 $x_l$）

$$
\delta_l = (I + J_l)^{\top}\,\delta_{l+1}
\qquad\Longrightarrow\qquad
\frac{\partial L}{\partial x_l}
= \frac{\partial L}{\partial x_L}
\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i, W_i)\right)
$$

两项含义完全不同：第一项从 $L$ 直达 $l$，**不经过任何权重层，增益恒为 $1$**，与深度无关；第二项是 Jacobian 连乘，仍可能消失，但只是"额外"通路。普通网络只有第二项，所以深层梯度必然病态。

**权重梯度**：$W_l$ 只在 $F$ 里出现，恒等分支无参数、贡献严格为 $0$，链式法则只剩一条通路

$$
\frac{\partial L}{\partial W_l}
= \left(\frac{\partial F(x_l, W_l)}{\partial W_l}\right)^{\top}
\frac{\partial L}{\partial x_{l+1}}
$$

取 $F = W_l\,\sigma(x_l)$ 时，即 $\partial L/\partial W_l = \delta_{l+1}\,\sigma(x_l)^{\top}$。

**与 $\partial L/\partial x_{l+1}$ 的关联**：形式上完全一样 —— ResNet 没改"上游梯度 × 局部导数"这条规则，差别全在 $\delta$ 本身多了 $+I$。把

$$
\delta_{l+1} = \left[\prod_{j=l+1}^{L-1}(I + J_j)\right]^{\top}\delta_{L}
$$

代回权重梯度，自然分成"从 $L$ 直达"与"逐层衰减"两条通路。

一句话：它不改变规则，只改变上游梯度的构成 —— 从纯连乘变成 $1+$ 连乘。

**失效条件**：原始 ResNet 输出端带 $\mathrm{ReLU}$ → "$1+$" 被打折；projection shortcut 使前向变为

$$
x_{L} = \left(\prod_{j} W_s\right) x_{l} + \cdots
$$

衰减重新回归。故严格成立需 **pre-activation**（He et al., 2016），这才是 1000 层可训的关键。

## 2a. Highway vs ResNet

$$
\text{Highway:}\quad y = T \odot H(x, W_H) + (1 - T) \odot x
\qquad\qquad
\text{ResNet:}\quad y = F(x) + x
$$

**本质：门控的凸组合 vs 无门的加法。**

| 维度 | Highway | ResNet |
|---|---|---|
| 旁路系数 | 可学习、数据依赖、逐元素，$T \in (0,1)$ | 恒为 $1$，无参数 |
| 梯度通路 | $\prod_{i}(1 - T_i)$，门开大则旁路被关，**衰减仍在** | $1 + \prod_{j}(I + J_j)$，**不可关闭** |
| 参数与初始化 | 多一套 $W_T$；靠 $b_T$ 取负实现"近恒等" | shortcut 零参数；靠 BN 天然近恒等 |
| 尺度 | 凸组合，保尺度，弱依赖 BN | 范数随深度增长，**强耦合 BN** |

决定性差异在梯度：Highway 的门控允许网络自己关掉旁路，ResNet 把系数钉死为 $1$，通路不可关闭。工程上 ResNet 赢在少参数、少敏感超参、一行实现。

## 2b. DenseNet vs ResNet

$$
\text{ResNet:}\quad x_{l+1} = x_{l} + F(x_{l})
\qquad\qquad
\text{DenseNet:}\quad x_{l} = H_{l}\big([x_0, x_1, \dots, x_{l-1}]\big)
$$

**本质：聚合算子从"加和"换成"拼接"。** ResNet 是"合并同类项"（信息被压缩、叠加、覆盖，$L$ 条边）；DenseNet 是"保留全部证据"（block 内完全图，$L(L+1)/2$ 条边，每层原样看到所有历史特征）。

- **优点**：梯度到任意浅层都是短路径，且特征未被混合，缓解更彻底；growth rate $k$ 小，参数效率高一个量级（DenseNet-121 约 $8\mathrm{M}$ / $25.0\%$ vs ResNet-152 约 $60\mathrm{M}$ / $23.0\%$）；小数据集正则好、多尺度天然融合。
- **缺点**：显存 $O(L^2 k)$ 是硬伤（输入通道 $k_0 + k\cdot l$ 线性涨）；FLOPs 不高但 memory-bound，墙钟时间不占优；通道强依赖前序所有层，**层不能删、剪枝困难**；工程生态不如 ResNet，工业界骨干仍是 ResNet 系。

**共同主线**：三者都在解决"梯度如何跨层无损传递"，趋势是越来越不信网络自学"该传多少" —— 交给门控（Highway）→ 钉死为 $1$（ResNet）→ 连合并都省了全保留（DenseNet）。
