# 公共数学符号与张量约定

> 适用范围：全书及对应实验。更新日期：2026-11-15。
> [阅读入口](../../README.md) · [术语表](glossary.md) · [参考来源](references.md) · [写作规范](../authoring-guide.md)

本表用于查阅记号。章节仍须在首次使用处解释变量、形状与成立条件，并通过图示、手算和代码讲清概念。本项目采用花书的标量、列向量、矩阵记法；代码中的批次布局另行写明，避免将一个样本的公式直接套在批次上。[花书记号表](https://www.deeplearningbook.org/contents/notation.html)与[第2章§2.1～2.2](https://www.deeplearningbook.org/contents/linear_algebra.html)是主要记号依据。

## 1. 对象、形状与运算

| 对象 | 本书写法 | 含义及形状 |
| --- | --- | --- |
| 实数、整数与自然数 | $\mathbb{R},\mathbb{Z},\mathbb{N}$ | 计数的取值范围另写，例如 $B\geq 1$，不依赖自然数是否含0的不同约定 |
| 标量 | $x,b,\eta,\lambda$ | 单个数；学习率常记作 $\eta$，正则化系数常记作 $\lambda$ |
| 向量 | $\mathbf{x}\in\mathbb{R}^{d}$ | 默认列向量；第 $j$ 个元素是 $x_j$ |
| 矩阵 | $\mathbf{W}\in\mathbb{R}^{m\times d}$ | 第 $r$ 行第 $j$ 列元素为 $W_{rj}$ |
| 高阶张量 | $\mathcal{X}$ | 同时列出形状及每个轴的含义；秩指轴数时明确说“张量阶数” |
| 数据集 | $\mathcal{D}=\{(\mathbf{x}^{(i)},y^{(i)})\}_{i=1}^{N}$ | $N$ 个带目标的样本；训练、验证、测试集加对应下标 |
| 全部可训练参数 | $\boldsymbol{\theta}$ | 概念上可展平成参数向量；实现可为多个参数张量的集合 |
| 模型与预测 | $f(\mathbf{x};\boldsymbol{\theta}),\hat y$ | 分号区分输入与参数；帽号表示预测或估计，须说明具体对象 |
| 转置 | $\mathbf{W}^{\top}$ | 交换矩阵的行与列；$m\times d$ 变为 $d\times m$ |
| 全1向量、单位矩阵 | $\mathbf{1}_{B},\mathbf{I}_{d}$ | 形状分别为 $B$ 和 $d\times d$ |
| 内积 | $\mathbf{x}^{\top}\mathbf{v}$ | 两向量同为 $d$ 维，结果是标量 |
| 逐元素乘法 | $\mathbf{x}\odot\mathbf{v}$ | 相同形状的对应元素相乘，与矩阵乘法区分 |
| 欧氏范数 | $\lVert\mathbf{x}\rVert_2$ | $\sqrt{\sum_{j=1}^{d}x_j^2}$；矩阵范数须另行注明 |
| 相等与近似 | $=,\approx$ | 近似来自舍入、数值方法或统计估计时，说明误差来源 |

数学中的一维列向量在代码中可以存成形状为 `(d,)` 的一维数组；它没有显式的行轴或列轴。涉及转置、外积或广播时，要使用明确的二维形状，不能声称一维数组的 `.T` 会自动变成列矩阵。

## 2. 索引与维度名称

| 标记 | 固定用途 | 例子 |
| --- | --- | --- |
| $i=1,\ldots,N$ | 数据集中的样本索引 | $\mathbf{x}^{(i)}$ 是第 $i$ 个样本 |
| $b=1,\ldots,B$ | 当前批次中的位置 | 批次第 $b$ 行可对应全数据集索引 $i_b$ |
| $j=1,\ldots,d$ | 输入特征索引 | $x_j^{(i)}$ |
| $r=1,\ldots,m$ | 输出分量索引 | $z_r$ |
| $c=1,\ldots,C$ | 类别索引 | $p_c$ 为第 $c$ 类概率 |
| $[\ell]$ | 网络层编号 | $\mathbf{h}^{[\ell]}$；方括号上标不是幂 |
| $t=1,\ldots,T$ | 序列时间步或 token 位置 | $\mathbf{h}_t$；具体序列含义在相关章节解释 |
| $k=0,1,\ldots$ | 优化更新步 | $\boldsymbol{\theta}_k$ |

$N$ 为指定数据集的样本数，$B$ 为实际批次大小，$d$ 为输入维度，$m$ 为输出维度，$C$ 为类别数。最后一个批次不足预设大小时，求平均必须使用它的实际样本数。代码通常从0开始索引，正文计算通常从1开始；对照示例要说明 `x[0]` 对应 $x_1$。如果同时出现层、样本与时间，分别写成 $\mathbf{h}_t^{[\ell](i)}$ 并在章内解释，避免用同一个上标承载几种含义。

## 3. 单样本列向量与行批次

单样本的仿射层写作：

$$
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b},
\qquad
\mathbf{x}\in\mathbb{R}^{d},\quad
\mathbf{W}\in\mathbb{R}^{m\times d},\quad
\mathbf{b},\mathbf{z}\in\mathbb{R}^{m}.
$$

把一个批次的样本逐行排成矩阵，每行是某个列向量的转置：

$$
\mathbf{X}=\begin{bmatrix}
(\mathbf{x}^{(i_1)})^{\top}\\
\vdots\\
(\mathbf{x}^{(i_B)})^{\top}
\end{bmatrix}\in\mathbb{R}^{B\times d}.
$$

各样本共用同一组参数，所以批次输出为：

$$
\mathbf{Z}=\mathbf{X}\mathbf{W}^{\top}
+\mathbf{1}_{B}\mathbf{b}^{\top}
\in\mathbb{R}^{B\times m}.
$$

这里 $\mathbf{X}\mathbf{W}^{\top}$ 的维度为 $(B\times d)(d\times m)$；偏置项是 $(B\times1)(1\times m)$，表示每行加相同偏置。代码 `Z = X @ W.T + b` 的广播要逐项对应这一步。某些教材或实现把权重存成 $d\times m$，那里的 `X @ W` 与本书数学权重相差转置；引用时须先写出映射。多层网络可以沿用相同约定，把第 $\ell$ 层输入、权重和输出的形状列在章内。

## 4. 损失与归约

单样本损失使用小写 $\ell$，例如 $\ell_i=\ell(\hat y^{(i)},y^{(i)})$。批次平均损失默认是：

$$
\mathcal{L}_{\mathcal{B}}(\boldsymbol{\theta})
=\frac{1}{B}\sum_{b=1}^{B}\ell_{i_b}(\boldsymbol{\theta}).
$$

需要加正则化时，单独声明目标：

$$
\mathcal{J}_{\mathcal{B}}(\boldsymbol{\theta})
=\mathcal{L}_{\mathcal{B}}(\boldsymbol{\theta})
+\lambda R(\boldsymbol{\theta}).
$$

$\mathcal{B}$ 表示本批次的样本索引集合，$R$ 是已定义的正则化函数。正文将 $\mathcal{L}$ 用作含正则化总目标时，必须先明确改用的定义。每段训练代码都应解释损失最终是否为标量，以及它沿哪些轴求和或平均。

| 归约名称 | 含义 | 需要说明的细节 |
| --- | --- | --- |
| `none` | 保留未归约损失 | 输出可能逐样本，也可能逐元素；写出实际形状 |
| `sum` | 对指定元素求和 | 元素数量改变时损失与梯度的尺度也会改变 |
| `mean` | 对指定元素求平均 | 明确分母是样本数、全部元素数，还是有效位置数 |

标量回归的教学推导可以选用 $\ell_i=\tfrac12(\hat y^{(i)}-y^{(i)})^2$，其中 $\tfrac12$ 用于简化导数。若代码调用普通均方误差，而其定义没有 $\tfrac12$，应调整实现或说明梯度相差2倍。多输出回归需声明是否先对 $m$ 个输出求平均；序列损失需声明 padding 是否排除、分母是否为有效 token 数。不得仅凭接口参数名 `mean` 推断具体数学目标。

跨多个批次统计整轮平均损失时，用批次实际样本数加权；不能把大小不同的批次均值直接等权平均。分类交叉熵中的“对类别求和”和“对样本求平均”是两个不同的轴操作，须分别写清。

## 5. 导数、梯度与 Jacobian

标量函数 $f:\mathbb{R}^{d}\to\mathbb{R}$ 的梯度取列向量约定：

$$
\nabla_{\mathbf{x}}f
=\begin{bmatrix}
\partial f/\partial x_1\\
\vdots\\
\partial f/\partial x_d
\end{bmatrix}\in\mathbb{R}^{d}.
$$

向量函数 $\mathbf{f}:\mathbb{R}^{d}\to\mathbb{R}^{m}$ 的 Jacobian 采用“输出为行、输入为列”的布局：

$$
\mathbf{J}_{\mathbf{f}}(\mathbf{x})\in\mathbb{R}^{m\times d},
\qquad
(\mathbf{J}_{\mathbf{f}})_{rj}=\frac{\partial f_r}{\partial x_j}.
$$

若 $\mathbf{z}=\mathbf{f}(\mathbf{x})$、$\mathcal{L}$ 为标量，则链式法则为 $\nabla_{\mathbf{x}}\mathcal{L}=\mathbf{J}_{\mathbf{f}}^{\top}\nabla_{\mathbf{z}}\mathcal{L}$。这说明梯度为何与输入维度相同；具体推导在第02章和第08章展开。矩阵参数的梯度与该参数形状相同，其每个元素是对应偏导；不能把框架的参数梯度数组和整层 Jacobian 混为一谈。微积分来源见[MIT-CALC-B 与 D2L-CALC](references.md)。

## 6. 概率、随机变量与对数

| 写法 | 本书含义 |
| --- | --- |
| $X$ 与 $x$ | 大写斜体 $X$ 表示随机变量，小写 $x$ 表示其取值；粗体 $\mathbf{X}$ 仍用于上文的批次矩阵，章内不得混用 |
| $P(X=x)$ | 离散随机变量取某值的概率 |
| $p_X(x)$ | 连续随机变量的概率密度；密度值可以大于1，某个点的概率不能直接用密度值代替 |
| $p(y\mid\mathbf{x};\boldsymbol{\theta})$ | 参数固定时，给定输入的条件分布；竖线表示条件，不是除法 |
| $X\sim p$ | $X$ 按分布 $p$ 取值 |
| $\mathbb{E}_{X\sim p}[g(X)]$ | 对指定分布取期望；省略下标时需在近处说明分布 |
| $\operatorname{Var}(X)$ | $\mathbb{E}[(X-\mathbb{E}[X])^2]$，存在性条件在相关章节说明 |
| $\log x$ | 自然对数；信息量单位为 nat，使用以2为底的对数时显式标 $\log_2$ 并用 bit |

无歧义处可将概率质量函数和密度统称为 $p$，但每次引入新分布时必须说明变量是离散还是连续。对观测数据固定、把参数作为自变量时，$p(\mathcal{D};\boldsymbol{\theta})$ 被视为似然；它通常不是关于参数的归一化概率分布。把联合似然拆成乘积前，说明采用的独立或条件独立假设。

离散分布的熵、交叉熵和 KL 散度统一写作：

$$
H(p)=-\sum_c p_c\log p_c,
\qquad
H(p,q)=-\sum_c p_c\log q_c,
$$

$$
D_{\mathrm{KL}}(p\Vert q)
=\sum_c p_c\log\frac{p_c}{q_c}
=H(p,q)-H(p).
$$

$\sum_c$ 的范围必须由类别或状态空间定义；上述分布均归一化。按极限约定 $0\log0=0$；若 $p_c>0$ 但 $q_c=0$，相应交叉熵与 KL 项为正无穷。KL 的方向不能交换。连续情形改为积分并单独解释，不能默认把离散熵的所有性质照搬过去。概率与信息论依据为[花书第3章](https://www.deeplearningbook.org/contents/prob.html)。

## 7. 逐章维护检查

每章从本表选出实际用到的符号，整理为章内短表。新增公共符号时，同步本表、术语表和对应实验；局部变量只在当前章节定义即可。引用外部公式时先统一布局、下标和对数底数，再说明与原来源的对应关系。后续序列、注意力和生成模型的专用符号在相关章实现时追加，不在基础章节提前要求读者掌握。