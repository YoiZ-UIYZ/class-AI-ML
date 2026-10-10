# 奇异值分解(SVD)
$$ A = UΣV^T $$

对于矩阵 $A \in R^{m \times n}$,  都可以分解为 $UΣV^T$ 形式, 其中 $U \in R^{m\times m}, V \in R^{n \times n}$ 都是正交矩阵, $Σ \in R^{m\times n}$ 只有 $(i, i)$ 非 0, 且满足 $σ_1 \ge σ_2 \ge \cdots \ge σ_r \gt 0, σ_{r+1} = \cdots = σ_{min(m,n)} = 0$.
$σ_i$ 是 $A$ 的奇异值. $U, V$ 的列向量 $u_i, v_i$ 分别为左右奇异向量.

## 求奇异值与奇异向量
令 $B = A^TA$, $B = A^TA = VΣU^TUΣV^T = VΣ^2V^T = VΣ^2V^{-1}$,

由于 $B = PΛP^{-1}$
$\implies Σ = \sqrt{Λ}$,  $V$ 是 $P$的单位正交化
$U = AVΣ^{-1}$
因此求 $A$ 的奇异值和左右奇异向量只需要求 $A^TA$ 的特征值与特征向量

### 样例
对于 
$$ A = \begin{bmatrix}
3&0\\
4&5
\end{bmatrix} $$
求出 $A^TA$
$$ A^TA = \begin{bmatrix}
3&4\\
0&5
\end{bmatrix} \begin{bmatrix}
3&0\\
4&5
\end{bmatrix} = \begin{bmatrix}
9+16&0+20\\
0+20&0+25
\end{bmatrix} = \begin{bmatrix}
25&20\\
20&25 \end{bmatrix} $$
求 $A^TA$ 的特征值 $λ_i$
$$ \begin{vmatrix}
25-λ&20\\
20&25-λ
\end{vmatrix} = 
(25-λ)^2 - 400 = 0 $$
$$ 25-λ = \pm20, \implies λ_1 = 45, λ_2 = 5 $$
	得到奇异值 $σ_1 = \sqrt{45}, σ_2 = \sqrt{5}$.
	$$ Σ = \begin{bmatrix}
	3\sqrt5&0\\
	0&\sqrt5
	\end{bmatrix} $$
求 $A^TA$ 的单位特征向量
	对于 $λ_1 = 45$: 
		$$A^TA - 45I = \begin{bmatrix}
		-20&20\\
		-20&20
		\end{bmatrix} $$
		解 $-20a+20b=0$ 得 $a=b$, 取 $v = (1,1)^T$, 单位化得 
		$$v_1 = \frac{1}{\sqrt2} \begin{bmatrix}1\\1\end{bmatrix}$$
	对于 $λ_2 = 5$:
		$$A^TA - 5I = \begin{bmatrix}
		20&20\\
		20&20
		\end{bmatrix} $$
		解 $a+b=0$ 得 $a=-b$, 取 $v=(1,-1)^T$, 单位化得
		$$ v_2 = \frac{1}{\sqrt2} \begin{bmatrix}1\\-1\end{bmatrix} $$
	$$ V = \frac{\sqrt2}{2} \begin{bmatrix}
	1&1\\
	1&-1
	\end{bmatrix} $$
求 $U = AVΣ^{-1}$ 得
$$ U = \frac{\sqrt{10}}{10}\begin{bmatrix}
1&3\\
3&-1
\end{bmatrix} $$


# 贝叶斯过程
## 贝叶斯定理
$$ p(θ|D) = \frac{p(D|θ)\cdot p(θ)}{p(D)} $$
$p(θ|D)$: 后验, 更新后的信念, 在 $D$ 成立的前提下 $θ$ 成立的概率.
$p(D|θ)$: 似然, 在 $θ$ 成立的前提下, $D$ 成立的概率.
$p(θ)$: 先验, 更新前的信念, 在看到数据前对 $θ$ 的判断.
$p(D)$: 证据, $D$ 成立的总共概率.

## 贝叶斯更新
不断用新数据修正对未知参数的信念, 先验 + 数据 --> 后验, 把后验代入到下一轮先验不断迭代计算.

### 样例
#### 广告点击率的 Beta-Bernoulli 贝叶斯更新

##### 设定

设每次广告展示的结果为

$$
Y_i=
\begin{cases}
1, & \text{用户点击}\\
0, & \text{用户没点击}
\end{cases}
\qquad
Y_i\mid\theta\ \sim\ \mathrm{Bernoulli}(\theta)
$$

其中 $\theta\in(0,1)$ 为未知点击率，$Y_1,\dots,Y_n$ 在给定 $\theta$ 下条件独立。

**似然**（$n$ 次展示中 $k$ 次点击）：

$$
p(D\mid\theta)=\prod_{i=1}^{n}\theta^{y_i}(1-\theta)^{1-y_i}
=\theta^{k}(1-\theta)^{\,n-k}
$$

**先验**（以 $\mathrm{Beta}(2,2)$ 表征"点击率约 0.5 但极不确定"）：

$$
p(\theta)=\frac{1}{B(\alpha,\beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1},
\qquad \alpha=2,\ \beta=2
$$

二者形状一致（均为 $\theta^{(\cdot)}(1-\theta)^{(\cdot)}$），故为共轭对。

---

##### 第一次迭代

观测数据 $D_1$：$n_1=10,\ k_1=3$。由贝叶斯公式

$$
p(\theta\mid D_1)=\frac{p(D_1\mid\theta)\,p(\theta)}{p(D_1)}
$$

分子仅保留 $\theta$ 的相关部分：

$$
p(D_1\mid\theta)p(\theta)\ \propto\ \theta^{3}(1-\theta)^{7}\cdot\theta^{2-1}(1-\theta)^{2-1}
=\theta^{\,4}(1-\theta)^{\,8}
$$

该式恰为 Beta 分布的核（$\alpha-1=4,\ \beta-1=8$），故

$$
\theta\mid D_1\ \sim\ \mathrm{Beta}(5,9)
$$

证据（归一化常数）为

$$
p(D_1)=\int_0^1 p(D_1\mid\theta)p(\theta)\,d\theta
=\frac{B(5,9)}{B(2,2)}=9.3240\times10^{-4}
$$

后验均值

$$
E[\theta\mid D_1]=\frac{5}{5+9}=\frac{5}{14}\approx 0.3571
$$

---

##### 第二次迭代

将第一次迭代所得后验 $p(\theta\mid D_1)$ 作为第二次的先验。观测数据 $D_2$：$n_2=10,\ k_2=5$。

$$
p(\theta\mid D_1,D_2)=\frac{p(D_2\mid\theta)\cdot p(\theta\mid D_1)}{p(D_2\mid D_1)}
$$

**上一轮后验在此作为因子显式参与**。分子为

$$
p(D_2\mid\theta)\,p(\theta\mid D_1)
\ \propto\ \Big[\theta^{5}(1-\theta)^{5}\Big]\cdot\underbrace{\Big[\theta^{4}(1-\theta)^{8}\Big]}_{\text{上一轮后验的核}}
=\theta^{\,9}(1-\theta)^{\,13}
$$

指数相加：$4+5=9,\ 8+5=13$。该式仍为 Beta 核，故

$$
\theta\mid D_1,D_2\ \sim\ \mathrm{Beta}(10,14)
$$

此处的归一化常数为**条件证据** $p(D_2\mid D_1)$（非 $p(D_2)$）：

$$
p(D_2\mid D_1)=\frac{B(10,14)}{B(5,9)}=5.6247\times10^{-4}
$$

后验均值

$$
E[\theta\mid D_1,D_2]=\frac{10}{10+14}=\frac{10}{24}\approx 0.4167
$$

---

##### 汇总

| 阶段    | 本轮数据         | 先验（＝上一轮后验）           | 后验                     | $E[\theta\mid D]$ |
| ----- | ------------ | -------------------- | ---------------------- | ----------------- |
| 初始    | —            | —                    | $\mathrm{Beta}(2,2)$   | 0.5000            |
| 第一次迭代 | $n=10,\ k=3$ | $\mathrm{Beta}(2,2)$ | $\mathrm{Beta}(5,9)$   | 0.3571            |
| 第二次迭代 | $n=10,\ k=5$ | $\mathrm{Beta}(5,9)$ | $\mathrm{Beta}(10,14)$ | 0.4167            |


# 交叉熵

**自信息**: 一个概率为 $p(x)$ 的事件发生时携带的信息量, 定义为
$$ I(x) = -\log p(x) $$
概率越小的事件, 发生时的信息量越大.

**信息熵**: 随机变量 $X$ 的整体不确定性的**期望信息量**
$$ H(p) = E[I(X)] = \sum_{x}p(x)I(x) = -\sum_{x}p(x) \log{p(x)} $$

**交叉熵**: 用模型分布 $q$ 去描述真实分布 $p$ 时, 所付出的**平均编码代价**
$$ H(p,q) = -\sum_{x} p(x)\log{q(x)} $$

**KL 散度**: 用分布 $q$ 近似 $p$ 所损失的信息:
$$ D_{\mathrm{KL}}(p\Vert q)=\sum_{x}p(x)\log\frac{p(x)}{q(x)} $$
$$
\begin{aligned}
D_{\mathrm{KL}}(p\Vert q)
&=\sum_{x}p(x)\big(\log p(x)-\log q(x)\big)\\
&=\sum_{x}p(x)\log p(x)-\sum_{x}p(x)\log q(x)\\
&=-H(p)+H(p,q)
\end{aligned}
$$
$$ H(p,q) = H(p)+D_{\mathrm{KL}}(p\Vert q) $$

### 样例
考虑三分类问题（猫、狗、鸟），输入一张**猫**的图片。

$$
p=\begin{bmatrix}1\\0\\0\end{bmatrix}
\quad\text{（真实标签，one-hot）}
$$

$$
q_A=\begin{bmatrix}0.7\\0.2\\0.1\end{bmatrix}\ \text{（模型 A，预测较好）},
\qquad
q_B=\begin{bmatrix}0.1\\0.6\\0.3\end{bmatrix}\ \text{（模型 B，预测较差）}
$$

真实分布为 one-hot，其熵为

$$
H(p)=-(1\cdot\ln 1+0\cdot\ln 0+0\cdot\ln 0)=0
$$

即真实标签不存在任何不确定性。两个模型的交叉熵损失为

$$
H(p,q_A)=-\Big[1\cdot\ln 0.7+0\cdot\ln 0.2+0\cdot\ln 0.1\Big]=-\ln 0.7\approx 0.356675
$$

$$
H(p,q_B)=-\Big[1\cdot\ln 0.1+0\cdot\ln 0.6+0\cdot\ln 0.3\Big]=-\ln 0.1\approx 2.302585
$$

由于 $p$ 为 one-hot，交叉熵退化为"只取真实类别的预测概率的负对数"：

$$
H(p,q)=-\ln q_{\text{真实类别}}
$$

**结论**：模型 A 给真实类别 0.7 的概率，损失 $0.357$；模型 B 只给 0.1，损失 $2.303$，是前者的 6.5 倍。交叉熵确实惩罚了偏离真实的预测。