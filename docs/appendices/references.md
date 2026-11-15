# 参考来源目录与引用约定

> 官方页面核验日期：2026-11-15。条目是已核验的阅读来源，不代表相应教程章节已完成。
> [阅读入口](../../README.md) · [公共符号](notation.md) · [术语表](glossary.md) · [写作规范](../authoring-guide.md)

## 1. 如何使用与引用

教程以 Goodfellow、Bengio、Courville 的《Deep Learning》（2016）为数学和神经网络主线。大学课程帮助补足几何直觉和推导，D2L帮助连接数学与代码。正式章节在关键定义、图、推导附近用 `[1]` 等标号，章末按出现顺序列完整条目；本页代号用于跨章查找来源，不能用代号代替具体引用。

每条章末引用至少给出作者或授课者、作品名称、版本或年份、具体章节/小节/讲次和官方链接。使用本页尚未细化的整章条目时，写作阶段继续核对实际采用的小节。官方网页的链接可访问性与其内容真实性是两项检查，不能只给课程首页。转载或改编图前核对使用条件；本项目优先依据原理重绘真正的SVG，并在图注中说明依据。标明“依据某章重绘”不等于拥有整页扫描或大段转载的许可。

以下条目未包含第三方镜像。课程与图书页可能更新，后续章节记录实际采用版本。环境安装命令与框架API须再查本机版本相应的官方文档，不因教材有旧代码就原样复制。

## 2. 花书：数学与机器学习主线

作者均为 Ian Goodfellow、Yoshua Bengio、Aaron Courville；图书为 *Deep Learning*，MIT Press，2016。在线版由作者公开提供。

| 代号 | 具体定位与官方链接 | 本教程用途 |
| --- | --- | --- |
| DL-NOTATION | [Notation](https://www.deeplearningbook.org/contents/notation.html) | 标量、向量、矩阵、概率、导数和数据集记号的核对依据 |
| DL-INTRO | [第1章 Introduction](https://www.deeplearningbook.org/contents/intro.html) | 第00章的学习背景、层次表示与全书范围 |
| DL-LA | [第2章 Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)，§2.1 Scalars, Vectors, Matrices and Tensors；§2.2 Multiplying Matrices and Vectors；§2.5 Norms | 第01章的数据对象、矩阵乘法与范数 |
| DL-PROB | [第3章 Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html) | 第03章的条件概率、期望、常见分布及信息论；具体小节在章内引用时再定位 |
| DL-NUM | [第4章 Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html)，§4.1 Overflow and Underflow；§4.3 Gradient-Based Optimization | 第04章的数值稳定性与优化基础 |
| DL-ML | [第5章 Machine Learning Basics](https://www.deeplearningbook.org/contents/ml.html) | 第05章的学习算法、训练与泛化、超参数与验证集；具体小节在章内引用时再定位 |

本项目的列向量及字体约定与花书对齐；样本、层、时间与更新步的组合索引，以及批次损失的统一写法属于本项目明确采用的教学约定，不声称花书原文具有相同排版。

## 3. MIT：几何直觉与多元微积分

| 代号 | 授课者、课程与具体页面 | 本教程用途 |
| --- | --- | --- |
| MIT-LA-1 | Gilbert Strang，MIT 18.06 *Linear Algebra*，Spring 2010站点，[Lecture 1: The geometry of linear equations](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-1-the-geometry-of-linear-equations/index.html) | 第01章用行方程、列组合与矩阵形式理解线性方程组 |
| MIT-LA-3 | Gilbert Strang，同一课程，[Lecture 3: Multiplication and inverse matrices](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-3-multiplication-and-inverse-matrices/) | 第01章矩阵乘法的不同视角；逆矩阵为按需拓展 |
| MIT-CALC-B | Denis Auroux，MIT 18.02SC *Multivariable Calculus*，Fall 2010，[Unit 2, Part B: Chain Rule, Gradient and Directional Derivatives](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/2.-partial-derivatives/part-b-chain-rule-gradient-and-directional-derivatives/)；Sessions 32～38 | 第02章链式法则、梯度、方向导数；按具体使用内容注明session |

18.06站点标作Spring 2010，但这些Strang讲课视频实际录制于Fall 1999；引用时保留这个区别。不要据站点年份断言录像拍摄于2010，也不要编造视频时间戳。18.02SC的上表链接是具体教学单元，后续用某一session的图或推导时再细化到该session页面。

## 4. D2L：数学到代码的桥梁

作者为 Aston Zhang、Zachary C. Lipton、Mu Li、Alexander J. Smola；作品为 *Dive into Deep Learning*。以下核验页面显示在线版本1.0.3，使用其PyTorch阅读视图；在线内容会更新，代码依赖仍以本仓库锁定版本和本机实测为准。

| 代号 | 官方具体页面 | 本教程用途 |
| --- | --- | --- |
| D2L-NOTATION | [Notation](https://d2l.ai/chapter_notation/index.html) | 核对机器学习常见记号；与本项目批次约定有差异时解释映射 |
| D2L-INSTALL | [Installation](https://d2l.ai/chapter_installation/index.html) | 第00章的环境准备概念，安装步骤按本机条件调整 |
| D2L-DATA | [§2.1 Data Manipulation](https://d2l.ai/chapter_preliminaries/ndarray.html) | 第00～01章数组、形状与基础运算 |
| D2L-LA | [§2.3 Linear Algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html) | 第01章线性代数与张量实现的对应 |
| D2L-CALC | [§2.4 Calculus](https://d2l.ai/chapter_preliminaries/calculus.html) | 第02章导数和梯度 |
| D2L-AD | [§2.5 Automatic Differentiation](https://d2l.ai/chapter_preliminaries/autograd.html) | 第02章导数检查与第08章自动微分的实践补充 |
| D2L-PROB | [§2.6 Probability and Statistics](https://d2l.ai/chapter_preliminaries/probability.html) | 第03章概率与采样例子 |
| D2L-LINEAR | [§3.1 Linear Regression](https://d2l.ai/chapter_linear-regression/linear-regression.html) | 第05章线性预测与平方损失的衔接 |
| D2L-SOFTMAX | [§4.1 Softmax Regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html) | 第04～05章softmax与多分类交叉熵 |
| D2L-STABILITY | [§5.4 Numerical Stability and Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) | 第04章数值稳定性，以及后续初始化主题；仅按已教前置内容选读 |

章节可要求读者选读相关内容，不把D2L全书安装、整书依赖或其他设备路线作为本机基础练习的必要条件。

## 5. 新增来源与现代主题

第06章起按章节任务增补具体教材部分；Transformer、LLM、多模态采用对应大学课程、原始论文及官方实现。尚未逐章核验的论文不在本目录冒充已采用来源。新增来源时记录实际采用的版本、具体定义或图的位置、官方链接和核验日期，并解释它补充了花书未覆盖的哪部分内容。

本目录仅维护共享来源身份和用途。原理是否解释充分、公式是否正确、实验是否可复现，仍须在对应章节独立验收。