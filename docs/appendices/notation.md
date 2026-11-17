# 公共数学符号与张量约定

> 适用范围：全书及对应实验。更新日期：2026-11-17。
> [阅读入口](../../README.md) · [术语表](glossary.md) · [参考来源](references.md) · [写作规范](../authoring-guide.md)

本表用于查阅记号。章节仍须在首次使用处解释变量、形状与成立条件，并通过图示、手算和代码讲清概念。本项目采用花书的标量、列向量、矩阵记法；代码中的批次布局另行写明，避免将一个样本的公式直接套在批次上。[花书记号表](https://www.deeplearningbook.org/contents/notation.html)与[第2章§2.1～2.2](https://www.deeplearningbook.org/contents/linear_algebra.html)是主要记号依据。

## 1. 对象、形状与运算

| 对象 | 本书写法 | 含义及形状 |
| --- | --- | --- |
| 实数、整数与自然数 | $`\mathbb{R},\mathbb{Z},\mathbb{N}`$ | 计数的取值范围另写，例如 $`B\geq 1`$，不依赖自然数是否含0的不同约定 |
| 标量 | $`x,b,\eta,\lambda`$ | 单个数；学习率常记作 $`\eta`$，正则化系数常记作 $`\lambda`$ |
| 向量 | $`\mathbf{x}\in\mathbb{R}^{d}`$ | 默认列向量；第 $`j`$ 个元素是 $`x_j`$ |
| 矩阵 | $`\mathbf{W}\in\mathbb{R}^{m\times d}`$ | 第 $`r`$ 行第 $`j`$ 列元素为 $`W_{rj}`$ |
| 高阶张量 | $`\mathcal{X}`$ | 同时列出形状及每个轴的含义；秩指轴数时明确说“张量阶数” |
| 数据集 | $`\mathcal{D}=\{(\mathbf{x}^{(i)},y^{(i)})\}_{i=1}^{N}`$ | $`N`$ 个带目标的样本；训练、验证、测试集加对应下标 |
| 全部可训练参数 | $`\boldsymbol{\theta}`$ | 概念上可展平成参数向量；实现可为多个参数张量的集合 |
| 模型与预测 | $`f(\mathbf{x};\boldsymbol{\theta}),\hat y`$ | 分号区分输入与参数；帽号表示预测或估计，须说明具体对象 |
| 转置 | $`\mathbf{W}^{\top}`$ | 交换矩阵的行与列；$`m\times d`$ 变为 $`d\times m`$ |
| 全1向量、单位矩阵 | $`\mathbf{1}_{B},\mathbf{I}_{d}`$ | 形状分别为 $`B`$ 和 $`d\times d`$ |
| 内积 | $`\mathbf{x}^{\top}\mathbf{v}`$ | 两向量同为 $`d`$ 维，结果是标量 |
| 逐元素乘法 | $`\mathbf{x}\odot\mathbf{v}`$ | 相同形状的对应元素相乘，与矩阵乘法区分 |
| 欧氏范数 | $`\lVert\mathbf{x}\rVert_2`$ | $`\sqrt{\sum_{j=1}^{d}x_j^2}`$；矩阵范数须另行注明 |
| 相等与近似 | $`=,\approx`$ | 近似来自舍入、数值方法或统计估计时，说明误差来源 |

数学中的一维列向量在代码中可以存成形状为 `(d,)` 的一维数组；它没有显式的行轴或列轴。涉及转置、外积或广播时，要使用明确的二维形状，不能声称一维数组的 `.T` 会自动变成列矩阵。

## 2. 索引与维度名称

| 标记 | 基础章节默认用途 | 例子 |
| --- | --- | --- |
| $`i=1,\ldots,N`$ | 数据集中的样本索引 | $`\mathbf{x}^{(i)}`$ 是第 $`i`$ 个样本 |
| $`b=1,\ldots,B`$ | 当前批次中的位置 | 批次第 $`b`$ 行可对应全数据集索引 $`i_b`$ |
| $`j=1,\ldots,d`$ | 输入特征索引 | $`x_j^{(i)}`$ |
| $`r=1,\ldots,m`$ | 输出分量索引 | $`z_r`$ |
| $`c=1,\ldots,C`$ | 类别索引 | $`p_c`$ 为第 $`c`$ 类概率 |
| $`[\ell]`$ | 网络层编号 | $`\mathbf{h}^{[\ell]}`$；方括号上标不是幂 |
| $`t=1,\ldots,T`$ | 序列时间步或 token 位置 | $`\mathbf{h}_t`$；具体序列含义在相关章节解释 |
| $`k=0,1,\ldots`$ | 优化更新步 | $`\boldsymbol{\theta}_k`$ |

在本基础约定中，$`N`$为数据集样本数，$`B`$为批次大小，$`d`$为输入维度，$`m`$为输出维度，$`C`$为类别数。专章可使用领域常见局部记号，但须在章内重新定义含义、形状和范围；例如LoRA秩$`r`$、检索候选数$`k`$、上下文预算$`B`$和每图patch数$`N`$不能机械套用本表。批次不足预设大小时，平均仍用实际分母。代码通常从0开始，正文通常从1开始，示例必须说明对应关系；层、样本、时间与优化更新编号在同一作用域内不得混用。

## 3. 单样本列向量与行批次

单样本的仿射层写作：

```math
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b},
\qquad
\mathbf{x}\in\mathbb{R}^{d},\quad
\mathbf{W}\in\mathbb{R}^{m\times d},\quad
\mathbf{b},\mathbf{z}\in\mathbb{R}^{m}.
```

把一个批次的样本逐行排成矩阵，每行是某个列向量的转置：

```math
\mathbf{X}=\begin{bmatrix}
(\mathbf{x}^{(i_1)})^{\top}\\
\vdots\\
(\mathbf{x}^{(i_B)})^{\top}
\end{bmatrix}\in\mathbb{R}^{B\times d}.
```

各样本共用同一组参数，所以批次输出为：

```math
\mathbf{Z}=\mathbf{X}\mathbf{W}^{\top}
+\mathbf{1}_{B}\mathbf{b}^{\top}
\in\mathbb{R}^{B\times m}.
```

这里 $`\mathbf{X}\mathbf{W}^{\top}`$ 的维度为 $`(B\times d)(d\times m)`$；偏置项是 $`(B\times1)(1\times m)`$，表示每行加相同偏置。代码 `Z = X @ W.T + b` 的广播要逐项对应这一步。某些教材或实现把权重存成 $`d\times m`$，那里的 `X @ W` 与本书数学权重相差转置；引用时须先写出映射。多层网络可以沿用相同约定，把第 $`\ell`$ 层输入、权重和输出的形状列在章内。

## 4. 损失与归约

单样本损失使用小写 $`\ell`$，例如 $`\ell_i=\ell(\hat y^{(i)},y^{(i)})`$。批次平均损失默认是：

```math
\mathcal{L}_{\mathcal{B}}(\boldsymbol{\theta})
=\frac{1}{B}\sum_{b=1}^{B}\ell_{i_b}(\boldsymbol{\theta}).
```

需要加正则化时，单独声明目标：

```math
\mathcal{J}_{\mathcal{B}}(\boldsymbol{\theta})
=\mathcal{L}_{\mathcal{B}}(\boldsymbol{\theta})
+\lambda R(\boldsymbol{\theta}).
```

$`\mathcal{B}`$ 表示本批次的样本索引集合，$`R`$ 是已定义的正则化函数。正文将 $`\mathcal{L}`$ 用作含正则化总目标时，必须先明确改用的定义。每段训练代码都应解释损失最终是否为标量，以及它沿哪些轴求和或平均。

| 归约名称 | 含义 | 需要说明的细节 |
| --- | --- | --- |
| `none` | 保留未归约损失 | 输出可能逐样本，也可能逐元素；写出实际形状 |
| `sum` | 对指定元素求和 | 元素数量改变时损失与梯度的尺度也会改变 |
| `mean` | 对指定元素求平均 | 明确分母是样本数、全部元素数，还是有效位置数 |

标量回归的教学推导可以选用 $`\ell_i=\tfrac12(\hat y^{(i)}-y^{(i)})^2`$，其中 $`\tfrac12`$ 用于简化导数。若代码调用普通均方误差，而其定义没有 $`\tfrac12`$，应调整实现或说明梯度相差2倍。多输出回归需声明是否先对 $`m`$ 个输出求平均；序列损失需声明 padding 是否排除、分母是否为有效 token 数。不得仅凭接口参数名 `mean` 推断具体数学目标。

跨多个批次统计整轮平均损失时，用批次实际样本数加权；不能把大小不同的批次均值直接等权平均。分类交叉熵中的“对类别求和”和“对样本求平均”是两个不同的轴操作，须分别写清。

## 5. 导数、梯度与 Jacobian

标量函数 $`f:\mathbb{R}^{d}\to\mathbb{R}`$ 的梯度取列向量约定：

```math
\nabla_{\mathbf{x}}f
=\begin{bmatrix}
\partial f/\partial x_1\\
\vdots\\
\partial f/\partial x_d
\end{bmatrix}\in\mathbb{R}^{d}.
```

向量函数 $`\mathbf{f}:\mathbb{R}^{d}\to\mathbb{R}^{m}`$ 的 Jacobian 采用“输出为行、输入为列”的布局：

```math
\mathbf{J}_{\mathbf{f}}(\mathbf{x})\in\mathbb{R}^{m\times d},
\qquad
(\mathbf{J}_{\mathbf{f}})_{rj}=\frac{\partial f_r}{\partial x_j}.
```

若 $`\mathbf{z}=\mathbf{f}(\mathbf{x})`$、$`\mathcal{L}`$ 为标量，则链式法则为 $`\nabla_{\mathbf{x}}\mathcal{L}=\mathbf{J}_{\mathbf{f}}^{\top}\nabla_{\mathbf{z}}\mathcal{L}`$。这说明梯度为何与输入维度相同；具体推导在第02章和第08章展开。矩阵参数的梯度与该参数形状相同，其每个元素是对应偏导；不能把框架的参数梯度数组和整层 Jacobian 混为一谈。微积分来源见[MIT-CALC-B 与 D2L-CALC](references.md)。

## 6. 概率、随机变量与对数

| 写法 | 本书含义 |
| --- | --- |
| $`X`$ 与 $`x`$ | 大写斜体 $`X`$ 表示随机变量，小写 $`x`$ 表示其取值；粗体 $`\mathbf{X}`$ 仍用于上文的批次矩阵，章内不得混用 |
| $`P(X=x)`$ | 离散随机变量取某值的概率 |
| $`p_X(x)`$ | 连续随机变量的概率密度；密度值可以大于1，某个点的概率不能直接用密度值代替 |
| $`p(y\mid\mathbf{x};\boldsymbol{\theta})`$ | 参数固定时，给定输入的条件分布；竖线表示条件，不是除法 |
| $`X\sim p`$ | $`X`$ 按分布 $`p`$ 取值 |
| $`\mathbb{E}_{X\sim p}[g(X)]`$ | 对指定分布取期望；省略下标时需在近处说明分布 |
| $`\operatorname{Var}(X)`$ | $`\mathbb{E}[(X-\mathbb{E}[X])^2]`$，存在性条件在相关章节说明 |
| $`\log x`$ | 自然对数；信息量单位为 nat，使用以2为底的对数时显式标 $`\log_2`$ 并用 bit |

无歧义处可将概率质量函数和密度统称为 $`p`$，但每次引入新分布时必须说明变量是离散还是连续。对观测数据固定、把参数作为自变量时，$`p(\mathcal{D};\boldsymbol{\theta})`$ 被视为似然；它通常不是关于参数的归一化概率分布。把联合似然拆成乘积前，说明采用的独立或条件独立假设。

离散分布的熵、交叉熵和 KL 散度统一写作：

```math
H(p)=-\sum_c p_c\log p_c,
\qquad
H(p,q)=-\sum_c p_c\log q_c,
```

```math
D_{\mathrm{KL}}(p\Vert q)
=\sum_c p_c\log\frac{p_c}{q_c}
=H(p,q)-H(p).
```

$`\sum_c`$ 的范围必须由类别或状态空间定义；上述分布均归一化。按极限约定 $`0\log0=0`$；若 $`p_c>0`$ 但 $`q_c=0`$，相应交叉熵与 KL 项为正无穷。KL 的方向不能交换。连续情形改为积分并单独解释，不能默认把离散熵的所有性质照搬过去。概率与信息论依据为[花书第3章](https://www.deeplearningbook.org/contents/prob.html)。

## 7. 逐章维护检查

每章从本表选出实际用到的符号，整理为章内短表。新增公共符号时，同步本表、术语表和对应实验；局部变量只在当前章节定义即可。引用外部公式时先统一布局、下标和对数底数，再说明与原来源的对应关系。已讲专章的符号、轴语义和局部复用见以下索引；按学习进度查阅，无需提前掌握全部记号。

## 8. 专章符号与轴语义索引

公共符号是基础默认约定，不是全书每个字母只能有一个含义。专章可以采用领域常见局部记号，但必须在章内短表定义形状和作用域。同一段公式不可悄悄换含义；从另一章搬公式或代码时，先核对映射，再做代换。

### 8.1 网络更新、图像与时间状态

| 主题 | 符号与含义 | 必须核对的形状或边界 | 定义所在章 |
| --- | --- | --- | --- |
| 更新与优化器 | $`\mathbf g_k`$为本次参数梯度，$`\eta_k`$为学习率；$`\mu`$为动量系数，$`\beta_1,\beta_2`$为Adam指数平均系数 | 梯度与对应参数同形；Adam计数从1开始，调度与代码数组可能从0开始，不能混用偏差校正指数 | [第09章§3、§5～7](../chapters/09-training-and-optimizers.md) |
| 梯度缓冲与计算图 | $`\nabla_{\boldsymbol\theta}\mathcal L`$是当前标量目标的参数梯度；多条路径贡献相加 | 框架的.grad还可能跨多次反传累计；新建计算图不等于清除旧缓冲 | [第08章](../chapters/08-backpropagation.md)、[第09章](../chapters/09-training-and-optimizers.md) |
| 正则化 | $`\lambda`$为已声明惩罚或衰减系数；概率掩码与缩放以章内定义为准 | L2损失项、优化器权重衰减与Dropout是不同操作；统计均值的层次必须明确 | [第10章](../chapters/10-generalization.md) |
| 卷积布局 | 行批次图像使用 $`B\times C_{\mathrm{in}}\times H\times W`$，卷积核有输出/输入通道与两个空间轴 | 本章代码为NCHW；通道、空间位置和批次不可互换；padding、stride、dilation各自改变索引规则 | [第11章§3～6](../chapters/11-convolutional-networks.md) |
| 残差与归一化 | 残差形式为 $`\mathbf x+F(\mathbf x)`$；$`\gamma,\beta`$可表示归一化的仿射参数 | 相加必须同形或明确投影；BN按通道跨指定批次/空间轴统计，LN按指定特征轴统计；这里的$`\beta`$不是DPO系数 | [第12章](../chapters/12-residual-and-normalization.md) |
| 时间状态 | $`\mathbf h_t`$为隐藏状态，LSTM另有细胞状态$`\mathbf c_t`$及输入/遗忘/输出门 | 单样本状态为列向量；代码须说明批次、时间、特征轴；$`t`$是序列位置，不是优化更新$`k`$ | [第13章§3～6](../chapters/13-sequence-models.md) |

### 8.2 嵌入、注意力与可见关系

| 对象 | 记号与形状 | 轴与作用域说明 | 定义所在章 |
| --- | --- | --- | --- |
| 离散ID与嵌入 | 嵌入表按词元ID选行，输出连续向量 | 字符、码点、UTF-8字节、子词ID和向量坐标不是同一对象；重复ID的梯度需累加到同一行 | [第14章](../chapters/14-tokenization-and-embeddings.md) |
| 单头查询/键/值 | 行序列 $`\mathbf Q\in\mathbb R^{n_q\times d_k}`$，$`\mathbf K\in\mathbb R^{n_k\times d_k}`$，$`\mathbf V\in\mathbb R^{n_k\times d_v}`$ | $`\mathbf Q\mathbf K^\top`$形状为$`n_q\times n_k`$；每个查询沿键轴归一化，乘$`\mathbf V`$后为$`n_q\times d_v`$ | [第15章§3～5](../chapters/15-attention.md) |
| 多头与模型宽度 | $`H`$为头数，$`d_h`$为每头宽度；本微型模型$`d=Hd_h`$ | 代码注意力张量使用$`[B,H,T,d_h]`$；$`H`$在图像章又可能是高度，依章内表区分 | [第16章§3～6](../chapters/16-transformer.md)、[第21章](../chapters/21-tiny-language-model.md) |
| mask | 可见矩阵规定查询能读哪些键；padding与因果限制分别定义 | 数学上的允许矩阵与加性$`0/-\infty`$偏置不是同一种数据；布尔True含义须按所用API核对，不能跨接口猜测 | [第15章](../chapters/15-attention.md)、[第16章](../chapters/16-transformer.md) |
| KV缓存 | 每层K、V各为$`[B,H,S,d_h]`$；$`S`$是历史长度，$`C`$是本次新块长度 | 新块局部行$`r`$使用绝对位置$`S+r`$，允许键位置$`j\le S+r`$；其logits仅有新$`C`$行 | [第25章§3～4](../chapters/25-language-model-inference.md) |

第25章的$`\mathbf V_{\mathrm{cache}}`$是值矩阵，$`V`$也可能是词表大小；字体、上下标和形状共同消除歧义。第16章的位置编码与第21/25章的学习位置嵌入是不同构造，不能仅因都使用位置编号就假设外推或缓存协议相同。

### 8.3 潜变量、近似推断与扩散时间

| 主题 | 专用记号 | 查阅要点 | 定义所在章 |
| --- | --- | --- | --- |
| 自编码器与PCA | 编码为低维表示$`\mathbf z`$，解码为重建$`\hat{\mathbf x}`$；PCA方向为正交向量 | 表示坐标可以不唯一；中心化参数只由训练数据估计；低维不自动等于更小磁盘文件 | [第17章](../chapters/17-representation-learning.md) |
| VAE | $`p_\theta(\mathbf x\mid\mathbf z)`$为条件生成分布，$`q_\phi(\mathbf z\mid\mathbf x)`$为近似后验，$`\boldsymbol\epsilon\sim\mathcal N(\mathbf0,\mathbf I)`$为重参数化噪声 | $`\theta,\phi`$分别指生成器和推断网络；ELBO的KL方向为近似后验到先验；最小化负ELBO与最大化ELBO符号相反 | [第18章§3～5](../chapters/18-vae-and-gan.md) |
| GAN | $`G`$、$`D`$分别为生成器与判别器，随机噪声经$`G`$产生样本 | 两个网络优化的目标和梯度边界分别声明，不能把所有loss相加后对两者同时反传 | [第18章](../chapters/18-vae-and-gan.md) |
| 扩散 | $`\beta_t`$为第$`t`$步前向方差，$`\alpha_t=1-\beta_t`$，$`\bar\alpha_t=\prod_{s=1}^{t}\alpha_s`$ | 乘噪声时使用$`\sqrt{\beta_t}`$；扩散时间$`t`$与训练步不同；$`\beta_t`$也不是Adam系数或DPO正则系数 | [第19章§3～5](../chapters/19-diffusion-models.md) |
| 扩散分布与噪声网络 | $`q`$表示固定前向加噪，$`p_\theta`$表示学习的反向分布，$`\boldsymbol\epsilon_\theta(\mathbf x_t,t)`$预测条件噪声 | 单时刻前向边缘、完整Markov轨迹和学习反向链分别定义；简化噪声MSE不等于完整似然目标 | [第19章](../chapters/19-diffusion-models.md) |

### 8.4 语言目标、低秩适配、偏好与采样

| 主题 | 专用记号与布局 | 必须保留的协议 | 定义所在章 |
| --- | --- | --- | --- |
| 语言模型 | 输入ID为$`[B,T]`$，logits为$`[B,T,V]`$；$`V`$为词表大小 | 词表轴log-softmax；输入位置预测右移目标；回答、提示、补齐和终止符是否计分分别声明 | [第20～22章](../chapters/20-language-model-pretraining.md) |
| 序列概率与NLL | $`\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid x,y_{<t})`$ | 完整回答概率用求和；平均NLL/PPL的分母为声明的有效目标数，不能与序列和分数互换 | [第20章](../chapters/20-language-model-pretraining.md)、[第22章](../chapters/22-instruction-tuning.md)、[第24章](../chapters/24-preference-alignment.md) |
| 两种掩码 | attention mask控制上下文可见性；loss mask控制哪些输出目标计损失 | 忽略提示输出损失不等于删除提示输入；提示末位置可能需要监督首回答目标；ignore_index只在对应损失接口有意义 | [第22章§3～5](../chapters/22-instruction-tuning.md) |
| LoRA | 第23章采用$`\mathbf W_0\in\mathbb R^{d\times k}`$，$`\mathbf A\in\mathbb R^{r\times k}`$，$`\mathbf B\in\mathbb R^{d\times r}`$，$`\Delta\mathbf W=s\mathbf B\mathbf A`$，$`s=\alpha/r`$ | 此处$`k`$是输入宽度、$`d`$是输出宽度、$`r`$是增量秩上界，需映射到基础节的输入/输出维度命名；$`s`$不是量化比例尺 | [第23章§3～7](../chapters/23-parameter-efficient-tuning.md) |
| 量化 | 第23章用$`\delta_i`$记量化步长，第25章用$`s_r`$记行比例尺；$`q_{ij}`$为离散整数，$`\widehat{\mathbf W}`$为重建权重 | 这些$`q`$不是概率分布；NF4码本、均匀整数、零点和尺度元数据必须区分；存储dtype与计算dtype分别报告 | [第23章§8～9](../chapters/23-parameter-efficient-tuning.md)、[第25章§6](../chapters/25-language-model-inference.md) |
| DPO | $`\pi_\theta`$为当前策略，$`\pi_{\mathrm{ref}}`$为固定参考，$`y_w,y_l`$为一对候选；$`\Delta_\theta`$为扣除参考偏向的对数概率差，$`m_\theta=\beta\Delta_\theta`$ | $`\beta>0`$是本目标的正则/间隔尺度；奖励$`r`$不是概率；候选回答含同样的任务终止事件，分数用回答对数概率和 | [第24章§3～6](../chapters/24-preference-alignment.md) |
| 采样 | $`\tau>0`$为温度，$`k`$为top-k候选数，$`p_0`$为top-p累计阈值 | 与优化步$`k`$及模型概率$`p`$区分；筛选顺序、重归一化、并列规则、随机种子和停止事件均须固定 | [第25章§5](../chapters/25-language-model-inference.md) |

### 8.5 检索、评价与图文匹配

| 主题 | 专用记号与形状 | 归一化、分母与边界 | 定义所在章 |
| --- | --- | --- | --- |
| 检索 | $`\mathbf u,\mathbf v_j\in\mathbb R^d`$为查询/片段列向量；片段按行堆为$`\mathbf V\in\mathbb R^{M\times d}`$；$`k,m`$为检索候选数与上下文片段数 | 点积与余弦需区分；第26章的$`B`$是上下文预算，不是当前批次；预算须给计量单位 | [第26章§2～6](../chapters/26-retrieval-augmented-generation.md) |
| 引用区间 | 原文区间采用$`[s,e)`$左闭右开坐标并绑定文档ID | 字符、字节、token坐标不能混用；区间匹配与事实支持是不同事件 | [第26章§3、§7](../chapters/26-retrieval-augmented-generation.md) |
| 分类与校准 | TP/FP/FN/TN需先定义正类；$`p_i`$为正类概率，$`c_i`$为预测类别信心，$`B_b`$为校准桶的样本集合 | F1、Brier和ECE采用本章已声明的定义；桶集合$`B_b`$不是批大小；空分母与分桶边界必须声明 | [第27章§3～4](../chapters/27-model-evaluation.md) |
| 配对组统计 | $`n`$为样本数，$`G`$为组数，$`n_g`$为组大小，$`d_g`$为同组两模型正确率差 | 等权组均值只在各组同样大小时等于等权样本均值；按组共同重抽两模型结果，不把组内变体当独立样本 | [第27章§5](../chapters/27-model-evaluation.md) |
| 视觉patch | 图像尺寸为$`H_I\times W_I\times C_I`$，patch边长$`P_I`$，patch数$`N=H_IW_I/P_I^2`$ | 第28章的$`N`$是每图patch数，不是全数据样本数；视觉序列$`[B,N,d]`$与图像级检索向量$`[B,d_e]`$不同 | [第28章§3](../chapters/28-multimodal-representations.md) |
| 图文对比 | 单位列向量$`\mathbf u_i,\mathbf v_j\in\mathbb R^{d_e}`$；批次行矩阵$`\mathbf U,\mathbf V`$形成$`\mathbf S=\mathbf U\mathbf V^\top/\tau`$ | 行为图像、列为文字；两方向分别沿候选轴归一化，对查询平均后再对方向平均；此$`\mathbf S`$不是自注意力矩阵 | [第28章§4～5](../chapters/28-multimodal-representations.md) |
| 检索命中 | $`\mathcal R_q`$为查询相关候选集合，$`\operatorname{TopK}(q)`$为返回前K项 | Hit@K统计至少命中一个相关项的查询比例，不等于找回全部相关项的比例；多正例需明确相关集合 | [第28章§6～7](../chapters/28-multimodal-representations.md) |

沿专章索引查阅时，优先核对输入/输出轴、局部字母含义与归约分母。索引给出查找入口，具体推导、图示、手算和代码检查仍在相应章节。


### 8.6 连续视觉前缀与项目证据

| 主题 | 符号及边界 | 查阅入口 |
| --- | --- | --- |
| 视觉条件生成 | $`P`$个视觉位置、宽$`d_v`$；连接器映射到$`d_\ell`$，输出$`B\times P\times d_\ell`$。文字输入$`T`$与视觉前缀合计$`P+T\le64`$；问题含$`Q`$字节时，首回答目标在0基位置$`P+Q-1`$ | [第29章§3～6](../chapters/29-vision-language-models.md) |
| 项目训练与质量 | $`U`$为更新步，$`\tau_j`$为第$`j`$项应用门槛；$`Q=\bigwedge_j(M_j\ge\tau_j)`$为逻辑与。此$`Q`$与视觉章的问题长度没有关系，工程核对与应用门槛分别报告 | [第30章§3～5](../chapters/30-capstone.md) |
