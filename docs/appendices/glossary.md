# 公共术语表

> 更新日期：2026-11-15。本表给出阅读时的简短释义；核心概念的完整图文、推导、例子与练习由相应章节承担。
> [阅读入口](../../README.md) · [公共符号](notation.md) · [参考来源](references.md) · [章节计划](../chapter-plan.md)

## 1. 首次出现与维护规则

每章首次把一个术语作为学习内容使用时，写“中文名称（英文名称，常用缩写）”，随后用一句话解释它解决什么问题，并给小例子。已经教过的前置术语可简短回顾并链接到实际存在的已完成章节。只有缩写时也要给出全称，例如“中央处理器（central processing unit，CPU）”。同章后续可以使用中文名或已经定义的缩写，不应在同一对象上反复切换译名。

阅读路线图可以提到尚未教授的名称，但必须标明“后续章节解释”，不把它用于当前推导或习题的必要前提。术语出现在本附录不代表对应教程已完成；下表“主讲章”表示编排位置，真实完成状态以章节计划为准。章节完成后，将具体定义所在的真实文件链接补充到相应术语附近。

## 2. 第00章：学习与环境

| 术语 | 英文 | 简短释义与使用边界 |
| --- | --- | --- |
| 人工智能 | artificial intelligence，AI | 研究机器完成需要智能的任务的广泛领域；本教程聚焦其中的数据学习路线 |
| 机器学习 | machine learning，ML | 依据数据和学习规则改善任务表现的方法；任务、数据和评价标准都需明确 |
| 深度学习 | deep learning，DL | 使用多层表示进行学习的方法，神经网络是本书主线；层数本身不保证效果 |
| 模型 | model | 将输入映射到输出的计算规则及其参数；名称相同的模型可以具有不同参数 |
| 训练 | training | 用训练数据和目标函数更新模型参数的过程 |
| 推理 | inference | 使用给定参数计算预测或生成结果的过程；本教程的工程语境与概率推断的广义含义区分 |
| 张量 | tensor | 实验代码中带形状的多维数值数组；数学对象与存储布局在第01章进一步区分 |
| 运行环境 | runtime environment | 操作系统、解释器、依赖、设备及其版本的组合，决定代码如何执行 |
| 虚拟环境 | virtual environment | 为项目隔离解释器包依赖的目录，不等于虚拟机或独立操作系统 |
| 中央处理器 | central processing unit，CPU | 执行通用程序和数值运算的处理器，是本机教学实验的默认设备 |
| 图形处理器 | graphics processing unit，GPU | 适合部分并行计算的设备；有GPU不等于当前框架已支持加速 |
| 随机种子 | random seed | 初始化伪随机数生成器状态的值；固定它不能独自保证跨硬件结果完全一致 |

主要阅读依据：[DL-INTRO、D2L-INSTALL、D2L-DATA](references.md)。具体设备是否可用以[环境报告](environment.md)及本次实验实测为准。

## 3. 第01章：线性代数与张量

[第01章完整图文与实验](../chapters/01-linear-algebra.md)。

| 术语 | 英文 | 简短释义与使用边界 |
| --- | --- | --- |
| 标量 | scalar | 单个数，如一个温度或一个损失值 |
| 向量 | vector | 有顺序的一组分量；本书公式默认列向量 |
| 矩阵 | matrix | 按行列排列的数值对象，可表示数据批次或线性变换 |
| 形状 | shape | 各轴长度按顺序组成的描述，如 `(B, d)`；维度相同不代表轴含义相同 |
| 广播 | broadcasting | 按明确的轴兼容规则扩展逐元素运算的语义，须检查实际结果形状 |
| 内积 | inner product / dot product | 同维实向量对应分量乘积之和，得到标量 |
| 矩阵乘法 | matrix multiplication | 对共享维度求和的运算，通常不满足交换律 |
| 范数 | norm | 满足特定性质的大小度量；需说明使用的范数类型 |
| 线性变换 | linear transformation | 保持加法与数乘的映射；加上非零偏置后的映射为仿射变换 |
| 批次 | batch | 一次共同参与计算的一组样本；本书代码默认把样本放在第一个轴 |

主要阅读依据：[DL-LA、MIT-LA-1、MIT-LA-3、D2L-LA](references.md)。

## 4. 第02章：导数与梯度

[第02章完整图文与实验](../chapters/02-calculus.md)。

| 术语 | 英文 | 简短释义与使用边界 |
| --- | --- | --- |
| 导数 | derivative | 单变量函数的局部变化率；可导性是使用条件 |
| 偏导数 | partial derivative | 多变量函数只改变一个自变量、固定其他自变量时的局部变化率 |
| 梯度 | gradient | 标量函数相对各输入分量的偏导组成的向量；本书取列向量布局 |
| 方向导数 | directional derivative | 沿指定方向的局部变化率，讨论单位距离变化时方向需归一化 |
| Jacobian矩阵 | Jacobian matrix | 向量函数的各输出对各输入的偏导矩阵，必须声明行列布局 |
| 链式法则 | chain rule | 按函数依赖关系组合局部导数的规则，支持复合函数求导 |
| 有限差分 | finite difference | 通过微小输入扰动近似导数的数值方法，存在截断与舍入误差 |
| 自动微分 | automatic differentiation，AD | 根据程序中的基础运算及链式法则计算导数，与符号推导或有限差分区分 |

主要阅读依据：[MIT-CALC-B、D2L-CALC、D2L-AD](references.md)。反向传播的完整计算流程在第08章解释。

## 5. 第03章：概率与信息论

[第03章完整图文与实验](../chapters/03-probability.md)。

| 术语 | 英文 | 简短释义与使用边界 |
| --- | --- | --- |
| 随机变量 | random variable | 将随机结果映射成数值的变量；与某次观测到的具体取值区分 |
| 概率分布 | probability distribution | 描述随机变量各取值或取值区间的概率规律 |
| 条件概率 | conditional probability | 在已知某事件或变量取值条件下的概率，须满足相应定义条件 |
| 独立 | independence | 联合分布可分解为边缘分布乘积的性质；不等于“不相关”的所有情形 |
| Bayes公式 | Bayes' rule | 用先验、似然和证据联系条件概率的公式，分母须为正 |
| 期望 | expectation | 按概率加权的平均，存在时可写为求和或积分 |
| 方差 | variance | 随机变量偏离其期望的平方的期望，度量波动大小 |
| 似然 | likelihood | 固定观测数据、把参数作为自变量时的概率模型表达式 |
| 熵 | entropy | 分布不确定性的一种量化方式；对数底数决定单位 |
| 交叉熵 | cross-entropy | 用一个分布评价另一分布对其编码或预测代价的量，方向与支持条件要明确 |
| KL散度 | Kullback–Leibler divergence | 两分布之间的非负差异量，通常不对称，不能当成数学距离 |

主要阅读依据：[DL-PROB、D2L-PROB](references.md)；记号与零概率处理见[公共符号](notation.md)。

## 6. 第04章：数值计算与优化

[第04章完整图文与实验](../chapters/04-numerical-optimization.md)。

| 术语 | 英文 | 简短释义与使用边界 |
| --- | --- | --- |
| 浮点数 | floating-point number | 用有限位数近似表示实数的计算机格式，精度与可表示范围有限 |
| 上溢与下溢 | overflow / underflow | 结果超出可表示范围，或极小数在有限精度下丢失；与数学函数本身的值区分 |
| 数值稳定性 | numerical stability | 数值方法对有限精度误差的控制能力；等价公式的实现表现可能不同 |
| softmax | softmax | 将一组实数映射为归一化正数的函数；需要指定归一化轴 |
| log-sum-exp | log-sum-exp | 对指数和取对数的运算，可通过减去最大值等变形稳定计算 |
| 梯度下降 | gradient descent，GD | 沿负梯度方向迭代调整参数的方法，学习率与目标性质影响其表现 |
| 学习率 | learning rate | 控制一次参数更新尺度的超参数，不等同于梯度本身 |

主要阅读依据：[DL-NUM、D2L-STABILITY、D2L-SOFTMAX](references.md)。

## 7. 第05章：机器学习与基线

[第05章完整图文与实验](../chapters/05-machine-learning.md)。

| 术语 | 英文 | 简短释义与使用边界 |
| --- | --- | --- |
| 监督学习 | supervised learning | 用输入及对应目标学习预测规则的方法 |
| 特征与标签 | feature / label | 模型可用的输入信息与任务目标；目标也可以是连续数值 |
| 参数与超参数 | parameter / hyperparameter | 参数由学习规则更新；超参数指定模型或训练方案，如学习率 |
| 损失函数 | loss function | 对预测与目标之间的误差进行量化的函数，归约方式属于其定义的一部分 |
| 训练集 | training set | 用于拟合模型参数的数据 |
| 验证集 | validation set | 用于选择超参数与方案的数据；反复使用可能使选择过程对它过拟合 |
| 测试集 | test set | 留给最终评价的数据，不能参与常规调参或拟合预处理步骤 |
| 数据泄漏 | data leakage | 训练或选择过程使用了实际预测时不可获得的信息，使评价失真 |
| 线性回归 | linear regression | 以输入的仿射函数预测连续目标的基线模型 |
| 逻辑回归 | logistic regression | 常用sigmoid把线性得分变为二分类概率的模型，名称中“回归”不改变其分类用途 |
| 泛化 | generalization | 模型在未用于拟合的新样本上的表现；需说明数据分布和评价方式 |
| 过拟合与欠拟合 | overfitting / underfitting | 分别指过度适配训练数据、或连训练规律也未充分拟合的现象，结合训练与验证证据判断 |
| 基线 | baseline | 为比较改进提供可复现参照的简单方法或模型 |

主要阅读依据：[DL-ML、D2L-LINEAR、D2L-SOFTMAX](references.md)。

## 8. 模型与后续主题名称

这些名称只用于理解路线，不作为第00～05章的推导前提，详细定义、数学、图示与实验在标明章节实现。

| 名称 | 英文或缩写 | 主讲章与当前阅读范围 |
| --- | --- | --- |
| 多层感知机 | multilayer perceptron，MLP | [第07章](../chapters/07-multilayer-perceptron.md)；由多个可学习层与非线性运算组成的基础网络 |
| 反向传播 | backpropagation | [第08章](../chapters/08-backpropagation.md)；按链式法则沿计算图累加梯度 |
| 卷积神经网络 | convolutional neural network，CNN | 第11章；用于局部结构等任务的网络，后续解释 |
| 循环神经网络 | recurrent neural network，RNN | 第13章；逐步处理序列并维护状态的网络，后续解释 |
| token与分词 | token / tokenization | 第14章；模型处理文本的单位及划分过程，后续解释 |
| 注意力 | attention | 第15章；按输入相关性组合信息的机制，后续解释 |
| Transformer | Transformer | 第16章；以注意力等模块构成的网络架构，后续解释 |
| 大语言模型 | large language model，LLM | 第20～27章；本教程中围绕文本概率建模及其使用的模型，规模与能力边界在对应章解释 |
| 检索增强生成 | retrieval-augmented generation，RAG | 第26章；将检索结果作为生成输入的系统方法，后续解释 |
| 多模态 | multimodal | 第28～30章；结合文本、图像等不同类型信息的建模主题，后续解释 |

后续章节实施时添加原始论文、大学课程和官方文档的精确来源；本表的名称预告不替代那些章节的引用核验。

## 9. 神经网络补充术语

### 第06章：阈值分类与更新

完整图文与实验见[神经元与感知机](../chapters/06-perceptron.md)。

| 术语 | 英文 | 本教程含义 |
| --- | --- | --- |
| 感知机 | perceptron | 用仿射分数和硬阈值分类，并按标签与分数的违例逐样本更新参数 |
| 激活函数 | activation function | 将一个单元的预激活分数转成输出的函数，具体形式由模型规定 |
| 线性可分 | linear separability | 在当前表示中存在严格分开全部有限训练样本的仿射分隔器 |
| 符号间隔 | signed functional margin | 本章定义为标签乘分数；参数缩放会改变它，不能直接当作几何距离 |
| 增广向量 | augmented vector | 在输入末尾加常数1，将偏置与权重合并到同一个内积 |
| 一轮与一次更新 | epoch / parameter update | 一轮扫描全部样本；一次更新实际修改参数，二者的计数不同 |

### 第07章：非线性表示与前向

完整图文与实验见[多层感知机](../chapters/07-multilayer-perceptron.md)。

| 术语 | 英文 | 本教程含义 |
| --- | --- | --- |
| 隐藏层 | hidden layer | 输入与最终输出之间的内部表示，数值可以逐项查看 |
| 前向传播 | forward propagation | 按计算依赖从输入求得各层及最终输出，不自动更新参数 |
| 整流线性单元 | rectified linear unit，ReLU | 取输入与0的较大者；分段线性，整体通常不可合并为一个仿射函数 |
| 双曲正切 | hyperbolic tangent，tanh | 输出在负1与正1之间的平滑激活，定义与导数见本章 |
| 表达能力 | representational capacity | 模型结构允许表示哪些函数，需与参数学习和新数据上的表现区分 |
| 参数量 | parameter count | 独立可训练数值位置的数量；不由批次大小决定 |


### 第08章：图与反向求导

完整图文与实验见[反向传播与自动微分](../chapters/08-backpropagation.md)。

| 术语 | 英文 | 本教程含义 |
| --- | --- | --- |
| 计算图 | computational graph | 记录变量与运算依赖的有向图，数值由本次前向给出 |
| 反向传播 | backpropagation | 以链式法则沿反向依赖累加标量损失对各变量的梯度 |
| 局部导数 | local derivative | 对当前运算直接输入求导；共享依赖还需汇合所有路径贡献 |
| 叶张量 | leaf tensor | 在当前跟踪图中由用户直接构造或按接口规则视为叶的张量；参数通常是叶 |
| 向量与Jacobian的乘积 | vector-Jacobian product，VJP | 给定输出侧权重后求该标量组合对输入的梯度，列向量写作Jacobian转置乘权重 |
| 梯度缓冲区 | gradient buffer | 参数的.grad累计位置；建立新图不会自动清除旧值 |


### 第09章：训练与更新状态

完整图文与实验见[训练循环与优化器](../chapters/09-training-and-optimizers.md)。

| 术语 | 英文 | 本教程含义 |
| --- | --- | --- |
| 小批次 | mini-batch | 一次梯度估计所用的样本组；批次大小、更新数和看过样本数须分别报告 |
| 动量 | momentum | 将当前梯度与历史缓冲结合的更新方法，需注明采用的缓冲约定 |
| Adam | adaptive moment estimation | 用梯度及其平方的指数平均并作初始偏差校正的逐坐标更新方法 |
| AdamW | decoupled weight decay | 在Adam更新中将参数衰减与梯度历史分开处理 |
| 学习率调度 | learning-rate schedule | 按更新步或轮改变学习率的规则，需明确计数与端点 |
| 全局范数裁剪 | global gradient norm clipping | 按全部待裁剪梯度合并后的范数统一缩放，不等于逐分量截断 |
| 训练检查点 | training checkpoint | 为继续同一训练保存模型、优化器、计数、采样及随机状态等上下文 |
