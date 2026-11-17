# 参考来源目录与引用约定

> 整理日期：2026-11-17。来源核验日期按各章记录，主要为2026-11-15，29～30章为2026-11-17。这里提供查阅索引，具体定义与实验范围见章末。
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

第06章起的教材部分、大学课程、原始论文及官方实现已按章节核验，下列索引补齐后续主线。新增来源仍需记录版本、具体定位、链接和核验日期，并解释它补充了花书未覆盖的内容。

本目录仅维护共享来源身份和用途。原理是否解释充分、公式是否正确、实验是否可复现，仍须在对应章节独立验收。

## 6. 已实现章节补充来源

| 来源 | 精确入口与用途 |
| --- | --- |
| MIT 6.041SC，Fall 2013，John Tsitsiklis | [Lecture 2：Conditioning and Bayes' Rule](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-i/lecture-2/)、[Lecture 3：Independence](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-i/lecture-3/)；第03章的条件概率、Bayes与独立 |
| MIT 18.335J，Spring 2019，Steven G. Johnson | [Week 1 / Lecture 2](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/week-1/)、[Week 2 / Lecture 3](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/week-2/)；第04章的浮点表示、求和与稳定性动机 |
| Cornell CS4780，Fall 2018 | [Lecture 3：The Perceptron](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote03.html)；第06章的更新、偏置增广与收敛证明 |
| Stanford CS231n 官方课程笔记 | [Neural Networks, Part 1](https://cs231n.github.io/neural-networks-1/)；第07章的逐层组织、前向计算与表达能力，页面未固定课程年份，按2026-11-15核验版本引用 |
| scikit-learn 官方实践 | [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) §12.1～12.2；第05章预处理隔离与数据泄漏原则，实验无需安装该包 |

以上入口在相应章节实现时核验；定义与数值例子仍以各章的具体引用、推导和实验为准。


## 7. 花书主线的后续章节

以下条目的作者仍为Goodfellow、Bengio、Courville，作品为2016年《Deep Learning》。这些部分已在对应教程章实际采用，不能继续只列数学与第5章基础。

| 代号 | 精确定位与官方链接 | 教程位置与用途 |
| --- | --- | --- |
| DL-MLP | 第6章，§6.3隐藏单元、§6.5.1～6.5.4与§6.5.6～6.5.7计算图、链式法则和反向传播：[官方正文](https://www.deeplearningbook.org/contents/mlp.html) | 第07～08章；第08章来源1给出逐小节定位 |
| DL-REG | 第7章，§7.1参数范数惩罚、§7.8早停、§7.12 Dropout：[官方正文](https://www.deeplearningbook.org/contents/regularization.html) | 第10章来源1；正则化定义与适用条件 |
| DL-OPT | 第8章，§8.1、§8.1.3、§8.3.1～8.3.2、§8.5.3～8.5.4：[官方正文](https://www.deeplearningbook.org/contents/optimization.html) | 第09章来源1；优化与学习的区别、基本更新和自适应优化 |
| DL-CONV | 第9章，§9.1卷积与互相关、§9.2局部连接/参数共享/等变/感受野、§9.3池化：[官方正文](https://www.deeplearningbook.org/contents/convnets.html) | 第11章来源1 |
| DL-SEQ | 第10章，§10.1时间展开、§10.2 RNN、§10.7长期依赖、§10.10门控、§10.11优化；§10.2.1教师强制、§10.2.3有向序列建模：[官方正文](https://www.deeplearningbook.org/contents/rnn.html) | 第13章来源1、第21章来源1 |
| DL-PRACTICE | 第11章，§11.1性能度量、§11.5调试：[官方正文](https://www.deeplearningbook.org/contents/guidelines.html) | 第10章来源2、第27章来源1；基线、诊断与评价范围 |
| DL-LANGUAGE | 第12章，§12.4.1语言模型与n-gram、§12.4.2神经语言模型、§12.4.3高维输出：[官方正文](https://www.deeplearningbook.org/contents/applications.html) | 第14章来源1、第20章来源1 |
| DL-AE | 第14章，§14.1欠完备自编码器与PCA关系、§14.2.2及§14.5去噪目标：[官方正文](https://www.deeplearningbook.org/contents/autoencoders.html) | 第17章来源1 |
| DL-REP | 第15章，§15.4分布式表示：[官方正文](https://www.deeplearningbook.org/contents/representation.html) | 第14章来源2 |
| DL-VARIATIONAL | 第19章，§19.1与§19.4近似推断和变分族：[官方正文](https://www.deeplearningbook.org/contents/inference.html) | 第18章来源1；ELBO与近似后验 |
| DL-GENERATIVE | 第20章，§20.10.2～20.10.4生成器、VAE和GAN：[官方正文](https://www.deeplearningbook.org/contents/generative_models.html) | 第18章来源2 |

## 8. 网络训练、视觉与时间状态

| 代号 | 已采用来源与定位 | 教程位置 |
| --- | --- | --- |
| D2L-BACKPROP | D2L在线1.0.3，§5.3.1～5.3.4前向、计算图、反向及依赖：[具体章节](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html) | 第08章来源2 |
| D2L-OPTIMIZERS | D2L在线1.0.3，§12.5[小批次SGD](https://d2l.ai/chapter_optimization/minibatch-sgd.html)、§12.10[Adam](https://d2l.ai/chapter_optimization/adam.html)、§12.11[学习率调度](https://d2l.ai/chapter_optimization/lr-scheduler.html) | 第09章来源2、4、8；调度端点仍以本实验定义为准 |
| ADAMW | Loshchilov、Hutter，*Decoupled Weight Decay Regularization*，2017初稿、ICLR2019修订，arXiv v3：[原始论文](https://arxiv.org/abs/1711.05101) | 第09章来源6、第10章来源5；衰减与损失L2惩罚须区别 |
| CS231N-TRAIN | Stanford CS231n，*Neural Networks, Part 3*，Loss/accuracy curves、Hyperparameter optimization、Evaluation；网页无固定授课年份，按原章访问版本：[大学笔记](https://cs231n.github.io/neural-networks-3/) | 第10章来源4 |
| CS231N-CONV | Stanford CS231n，*Convolutional Neural Networks*，Convolutional Layer下Local Connectivity、Spatial arrangement、Parameter Sharing：[大学笔记](https://cs231n.github.io/convolutional-networks/) | 第11章来源6；D2L补充见§7.2互相关、§7.3填充步幅、§7.4多通道、§7.5池化，精确链接留在章末 |
| RESNET-BN-LN | He等，*Deep Residual Learning for Image Recognition*，2015预印本：[ResNet论文](https://arxiv.org/abs/1512.03385)；Ioffe、Szegedy，*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*，2015：[BN论文](https://arxiv.org/abs/1502.03167)；Ba、Kiros、Hinton，*Layer Normalization*，2016：[LN论文](https://arxiv.org/abs/1607.06450) | 第12章来源1、5、6；课程桥梁为D2L1.0.3 §8.6.1～8.6.2[残差块](https://d2l.ai/chapter_convolutional-modern/resnet.html)、§8.5.1～8.5.2[批量归一化](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) |
| D2L-RNN-GATES | D2L在线1.0.3，§9.4[RNN](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html)、§9.7尤其§9.7.1.2[BPTT与截断](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html)、§10.1[LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html)、§10.2[GRU](https://d2l.ai/chapter_recurrent-modern/gru.html) | 第13章来源2～5；GRU候选重置位置与PyTorch实现的差别在章内说明 |

## 9. 文本表示、注意力与Transformer

| 代号 | 已采用来源与定位 | 教程位置 |
| --- | --- | --- |
| BPE | Sennrich、Haddow、Birch，*Neural Machine Translation of Rare Words with Subword Units*，ACL2016，§3.2、算法1，论文印刷页1717～1718：[官方论文PDF](https://aclanthology.org/P16-1162.pdf) | 第14章来源4；初始字符、词尾、频次合并与冻结应用 |
| CS224N-WORD2VEC | Stanford CS224N，Winter2023，*Note 1: Introduction and Word2Vec*，公开draft：[具体笔记PDF](https://web.stanford.edu/class/cs224n/readings/cs224n_winter2023_lecture1_notes_draft.pdf)；D2L1.0.3 §15.1.1～15.1.4：[one-hot、词向量、skip-gram与CBOW](https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html) | 第14章来源3、5；本实验采用完整softmax简化目标 |
| CS224N-ATTENTION | Stanford CS224N，Winter2019，课程笔记Part VI §2 *Attention Mechanism*：[具体笔记PDF](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf) | 第15章来源1；固定上下文到动态读取的动机 |
| TRANSFORMER | Vaswani等，*Attention Is All You Need*，2017，arXiv v7；§3.1～3.5、图1/2，特别§3.2.1缩放点积与§3.2.3应用类型：[原论文v7](https://arxiv.org/html/1706.03762v7) | 第15章来源3、第16章来源1 |
| D2L-TRANSFORMER | D2L在线1.0.3，§11.1[Q/K/V](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html)、§11.3[分数函数](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)、§11.6尤其§11.6.3[位置编码](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)、§11.7.1～11.7.5[Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) | 第15～16章及第21章 |
| GPT2-BERT | Radford等，*Language Models are Unsupervised Multitask Learners*，2019，§2.1目标、§2.3归一化位置：[原始报告](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)；Devlin等，*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*，2018预印本/NAACL2019，§3与§3.1：[原论文](https://arxiv.org/abs/1810.04805) | 第16章来源4、5；区分模型家族与本章教学块 |
| CS224N-A4 | Stanford CS224N，Spring2024 Assignment4，*Self-Attention, Transformers, and Pretraining*，§1～2数学练习与预训练组织：[大学作业PDF](https://web.stanford.edu/class/cs224n/assignments/a4_spr24_student_handout.pdf) | 第15章来源4、第21章来源4；不沿用其GPU预算 |

## 10. 表示学习与生成模型

| 代号 | 已采用来源与定位 | 教程位置 |
| --- | --- | --- |
| CS229-PCA | Andrew Ng，*Part XI: Principal Components Analysis*，6页讲义，未标固定授课年份，pp.2～6：[官方讲义](https://see.stanford.edu/materials/aimlcs229/cs229-notes10.pdf)；Carlos Guestrin，CS229 Fall2021，*Lecture14: Dimensionality Reduction / PCA*，slides12～17：[官方课件](https://cs229.stanford.edu/notes2021fall/lecture14-pca.pdf) | 第17章来源2、3；预处理、投影、重建和特征值关系 |
| VAE | Kingma、Welling，*Auto-Encoding Variational Bayes*，2013初稿/2014会议版本，arXiv v11，§2.2～2.4、§3、附录B：[原论文v11](https://arxiv.org/html/1312.6114v11) | 第18章来源3；ELBO、重参数化与对角高斯KL |
| GAN | Goodfellow等，*Generative Adversarial Nets*，2014，§3～4：[原论文v1](https://arxiv.org/html/1406.2661v1) | 第18章来源4；minimax、非饱和目标与理想理论边界 |
| DDPM | Ho、Jain、Abbeel，*Denoising Diffusion Probabilistic Models*，NeurIPS2020，arXiv v2，§2、§3.2/3.4，式(2)/(4)/(6)/(7)/(11)/(14)、算法1/2：[原论文](https://arxiv.org/abs/2006.11239) | 第19章来源1；前向边缘、后验与噪声参数化 |
| CS236-DIFFUSION | Stefano Ermon，Stanford CS236 Fall2023，第16讲 *Score Based Diffusion Models*：[该讲官方PPTX](https://deepgenerativemodels.github.io/assets/slides/lecture16-2023-comp.pptx) | 第19章来源2；不以课程主页替代实际讲义 |

## 11. 语言数据、监督微调与偏好优化

| 代号 | 已采用来源与定位 | 教程位置 |
| --- | --- | --- |
| D2L-LM | D2L在线1.0.3，§9.3.1语言模型、§9.3.2困惑度、§9.3.3序列划分：[具体章节](https://d2l.ai/chapter_recurrent-neural-networks/language-model.html) | 第20章来源2、第21章来源2 |
| NEURAL-LM | Bengio、Ducharme、Vincent、Jauvin，*A Neural Probabilistic Language Model*，JMLR3，2003，pp.1137～1155：[期刊原文](https://jmlr.org/papers/v3/bengio03a.html) | 第20章来源5；联合学习词表示与条件序列概率的经典背景 |
| LM-DEDUP | Lee等，*Deduplicating Training Data Makes Language Models Better*，ACL2022，§4.1精确重复子串、§4.2近似整篇匹配及评估重叠：[官方论文PDF](https://aclanthology.org/2022.acl-long.577.pdf) | 第20章来源3；本机只做整篇哈希与已知组，不冒称实现大规模去重 |
| INSTRUCTGPT | Ouyang等，*Training language models to follow instructions with human feedback*，2022，§3.1三阶段路线、§3.2数据、§3.5训练：[原始论文](https://arxiv.org/abs/2203.02155) | 第22章来源1、第24章来源2；本章SFT/DPO小实验不等于复现完整InstructGPT |
| CS224N-SFT-PREF | Archit Sharma，Stanford CS224N Spring2024，第10讲 *Prompting, Instruction Finetuning, and DPO/RLHF*，基于Jesse Mu讲义：[官方PDF](https://web.stanford.edu/class/cs224n/slides/cs224n-spr2024-lecture10-prompting-rlhf.pdf) | 第22章使用PDF40页SFT、53页限制；第24章使用PDF60页比较建模、62页KL、70页总结。均为从1起算的PDF页，与课件内标签63/66/74区分 |
| CHAT-TEMPLATE | Hugging Face Transformers，*Chat templates*，角色消息序列化与生成提示；在线版本按原章核验记录：[官方说明](https://huggingface.co/docs/transformers/chat_templating) | 第22章来源3；教学模型使用原创显式文本模板，不依赖该库 |
| LORA | Hu等，*LoRA: Low-Rank Adaptation of Large Language Models*，2021，arXiv v2，§4.1低秩增量/初始化/合并、§4.2目标层/成本、§7.1目标矩阵实验：[原论文v2](https://arxiv.org/html/2106.09685v2)；Microsoft官方Linear行为：[loralib源码](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py) | 第23章来源1、3 |
| QLORA | Dettmers等，*QLoRA: Efficient Finetuning of Quantized LLMs*，2023，§2量化/LoRA/内存、§3 NF4/双重量化/分页优化器、§4层覆盖：[原论文v1](https://arxiv.org/html/2305.14314v1) | 第23章来源2；NF4、双重量化和分页优化器是论文组成，本机均匀四位CPU替代不等于完整QLoRA |
| DPO | Rafailov等，*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*，arXiv v3，§3、§4式(4)～(7)、附录A.1/A.2/A.4：[原论文v3](https://arxiv.org/html/2305.18290v3) | 第24章来源1；比较假设、KL理想策略、配分函数消去与梯度 |

## 12. 推理、检索、评价与多模态

| 代号 | 已采用来源与定位 | 教程位置 |
| --- | --- | --- |
| CS336-INFERENCE | Stanford CS336 Spring2025，第10讲 *Inference*，`arithmetic_intensity_of_inference`、`compute_transformer_stats`、`throughput_and_latency`、`quantization`：[官方讲义源码](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py) | 第25章来源1；预填充、解码、缓存与资源口径 |
| KV-CACHE | Hugging Face Transformers4.50.0，*Caching*，KV追加、mask形状与位置：[固定版本说明](https://huggingface.co/docs/transformers/v4.50.0/en/cache_explanation)；GenerationConfig温度与top-k/top-p：[固定版本接口](https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/text_generation) | 第25章来源2、4；原生PyTorch适配器不依赖Transformers |
| NUCLEUS | Holtzman等，*The Curious Case of Neural Text Degeneration*，ICLR2020，arXiv v2，§3.1～3.3：[原论文v2](https://arxiv.org/abs/1904.09751v2) | 第25章来源3；top-p、top-k与温度 |
| LLM-INT8 | Dettmers等，*LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*，NeurIPS2022，arXiv v2，§3：[原论文v2](https://arxiv.org/abs/2208.07339v2) | 第25章来源5；教学逐行量化未实现论文的异常值混合精度与整数内核 |
| CS224V-RAG | Monica Lam，Stanford CS224V 2024，Lecture1，PDF第23页（零基22）Reading/RAG和grounding：[大学具体讲义](https://web.stanford.edu/class/cs224v/lectures_2024/l-introduction.pdf) | 第26章来源1 |
| RAG-DPR | Lewis等，*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*，NeurIPS2020，arXiv v4，§2.1～2.4：[RAG原论文](https://arxiv.org/html/2005.11401v4)；Karpukhin等，*Dense Passage Retrieval for Open-Domain Question Answering*，EMNLP2020，arXiv v3，§2～3：[DPR原论文](https://arxiv.org/html/2004.04906v3) | 第26章来源2、3；区分隐文档边缘化、神经双塔与本机字符TF-IDF/简化拼接 |
| TFIDF | scikit-learn1.9.1核验版本，用户指南§8.2.3.4，平滑IDF、sublinear TF与归一化：[具体官方说明](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) | 第26章来源4；本机用NumPy实现，未把该包作为训练依赖 |
| SQUAD-METRICS | Rajpurkar等，*SQuAD: 100,000+ Questions for Machine Comprehension of Text*，EMNLP2016，§6.1、PDF第7页：[官方论文PDF](https://aclanthology.org/D16-1264.pdf) | 第27章来源2；EM、词袋F1与归一化的具体任务定义 |
| HELM | Liang等，*Holistic Evaluation of Language Models*，arXiv v2，§4.3准确性、§4.4校准/不确定性、§4.9效率：[原文v2](https://arxiv.org/html/2211.09110v2) | 第27章来源3；不是本项目全面复现HELM的声明 |
| CS109-BOOTSTRAP | Jerry Cain等，Stanford CS109 Winter2024，Lecture19 *Sampling and the Bootstrap*，2024-02-23：[大学具体讲义](https://web.stanford.edu/class/archive/cs/cs109/cs109.1244/lectures/19_sampling_bootstrap.pdf) | 第27章来源4；有放回重抽基础，本实验另外声明按组配对与小样本限制 |
| CLIP | Radford等，*Learning Transferable Visual Models From Natural Language Supervision*，ICML2021，§2.3与图3、§2.4、§3.1、§6：[原论文](https://arxiv.org/abs/2103.00020) | 第28章来源1；图文目标、编码器、零样本迁移及限制 |
| VIT | Dosovitskiy等，*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*，ICLR2021，arXiv v2，§3.1、式(1)：[原论文v2](https://arxiv.org/abs/2010.11929v2) | 第28章来源3；patch投影、位置与视觉序列 |
| CS231N-MULTIMODAL | Ranjay Krishna，Stanford CS231n Spring2025，第16讲 *Multi-Modal Foundation Models*，2025-05-27，PDF16～20、24页：[大学具体讲义](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf)；CLIP的VisionTransformer和forward：[官方源码](https://github.com/openai/CLIP/blob/main/clip/model.py) | 第28章来源2、4；本机训练原创11840参数模型，没有导入CLIP权重或运行自然语言文字塔 |

框架函数与实现差异继续由各章引用版本化官方接口，不将教材版本、库版本和检查点协议混为一个版本号。上述索引为查找主来源提供入口，各章的图文、推导、实际配置与真实失败结果仍是理解本项目实现范围的依据。


## 13. 视觉语言生成与综合项目

以下条目于2026-11-17在对应章节核验；以各章的原理、接口、实际训练与失败证据为具体范围。

| 代号 | 具体来源与用途 | 教程位置 |
| --- | --- | --- |
| LLAVA | Liu、Li、Wu、Lee，2023，Visual Instruction Tuning，arXiv v2，§4.1视觉映射、§4.2回答目标与两阶段：[原论文](https://arxiv.org/html/2304.08485v2)；[接入源码](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py)、[连接器](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py) | [第29章](../chapters/29-vision-language-models.md)；原创小模型借鉴接口职责，未复现LLaVA权重或性能 |
| BLIP2 | Li、Li、Savarese、Hoi，2023，BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models，§2.2、§3.1与§3.3：[原论文v3](https://arxiv.org/html/2301.12597v3) | 第29章；Q-Former、查询压缩与连续前缀的拓展，未实际训练完整Q-Former |
| POPE | Li等，2023，Evaluating Object Hallucination in Large Vision-Language Models，§2.2、§4、§5：[原论文v3](https://arxiv.org/html/2305.10355v3) | 第29章；对象幻觉、正负例及指标，本项目未执行自然图像标准基准 |
| CS224N-REPORT | Stanford CS224N，Project Report Instructions，2025-02-27，§1第2～4页方法、实验与误差分析：[课程PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/project/CS224N_Final_Project_Report_Instructions_2025.pdf) | [第30章](../chapters/30-capstone.md)；综合项目的报告组织 |
| TORCH-CHECKPOINT | PyTorch 2.15，Saving and Loading Models中General Checkpoint：[官方教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)；[Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html) | 第30章；参数、优化器与随机状态恢复及跨平台边界 |
| MODEL-CARDS | Mitchell等，Model Cards for Model Reporting，FAT*2019：[原论文v2](https://arxiv.org/abs/1810.03993v2) | 第30章；预期用途、评价条件和局限；未声称执行真实人群分组评价 |
