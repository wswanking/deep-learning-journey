# deep-learning-journey

基于《Deep Learning》（花书）的中文教程，循序渐进学习数学基础、神经网络、典型模型、LLM与多模态，并结合Windows本机开展小模型训练实践。

## 正式阅读目录

- [第00章：学习路线与本机实验环境](docs/chapters/00-learning-and-environment.md)
- [第01章：线性代数与张量](docs/chapters/01-linear-algebra.md)
- [第02章：导数、梯度与链式法则](docs/chapters/02-calculus.md)
- [第03章：概率、统计与信息论](docs/chapters/03-probability.md)
- [第04章：数值计算与优化入门](docs/chapters/04-numerical-optimization.md)
- [第05章：机器学习问题与线性基线](docs/chapters/05-machine-learning.md)
- [第06章：神经元与感知机](docs/chapters/06-perceptron.md)
- [第07章：多层感知机与前向传播](docs/chapters/07-multilayer-perceptron.md)
- [第08章：计算图、反向传播与自动微分](docs/chapters/08-backpropagation.md)
- [第09章：训练循环与优化算法](docs/chapters/09-training-and-optimizers.md)
- [第10章：泛化、正则化与实验方法](docs/chapters/10-generalization.md)
- [第11章：卷积神经网络](docs/chapters/11-convolutional-networks.md)
- [第12章：残差连接、归一化与迁移学习](docs/chapters/12-residual-and-normalization.md)
- [第13章：序列建模与 RNN、LSTM、GRU](docs/chapters/13-sequence-models.md)
- [第14章：文本表示、词嵌入与分词](docs/chapters/14-tokenization-and-embeddings.md)
- [第15章：注意力机制](docs/chapters/15-attention.md)
- [第16章：Transformer 与模型家族](docs/chapters/16-transformer.md)
- [第17章：表示学习与自编码器](docs/chapters/17-representation-learning.md)
- [第18章：生成模型：VAE 与 GAN](docs/chapters/18-vae-and-gan.md)
- [第19章：扩散模型入门](docs/chapters/19-diffusion-models.md)
- [第20章：语言模型目标、数据与预训练](docs/chapters/20-language-model-pretraining.md)
- [第21章：从零训练微型语言模型](docs/chapters/21-tiny-language-model.md)
- [第22章：指令微调与数据构造](docs/chapters/22-instruction-tuning.md)
- [第23章：参数高效微调：LoRA 与 QLoRA](docs/chapters/23-parameter-efficient-tuning.md)
- [第24章：偏好学习与对齐](docs/chapters/24-preference-alignment.md)
- [第25章：LLM 推理、采样与效率](docs/chapters/25-language-model-inference.md)
- [第26章：检索增强生成](docs/chapters/26-retrieval-augmented-generation.md)
- [第27章：模型评估与失败分析](docs/chapters/27-model-evaluation.md)
- [第28章：多模态表示与 CLIP](docs/chapters/28-multimodal-representations.md)

更多章节见[全部章节开发与草稿索引](docs/draft-index.md)；后续各章持续并行完善，骨架已建立。31章任务和验收依据见[章节计划](docs/chapter-plan.md)。

## 项目入口

- [项目上下文与任务约定](AGENTS.md)
- [写作、数学、图文与排版规范](docs/authoring-guide.md)
- [公共数学符号](docs/appendices/notation.md) · [术语表](docs/appendices/glossary.md)
- [参考来源目录](docs/appendices/references.md) · [导航约定](docs/navigation.md)
- [本机硬件调研](docs/appendices/environment.md) · [CPU框架实测](docs/appendices/framework-validation.md)
- [训练依赖与复现](requirements/README.md)
- [HTML/PDF构建说明](docs/build-guide.md)
- [基础工具链验收](docs/appendices/build-validation.md) · [逐章验收记录](docs/appendices/chapter-validation.md)

## 教程要求

核心原理配图文，优先依据经典教材、优秀大学课程和原始论文。所有教学配图使用真正SVG，PDF保留矢量；数学符号标准，推导、形状、手算与代码一致；概念充分解释，按前置知识衔接。每章以Markdown为内容源，同源生成HTML和PDF。

## 当前进度

| 内容 | 状态 |
| --- | --- |
| INF-01～03 | 环境调研、导出工具、公共符号/术语/导航已完成 |
| 第00～28章 | 已验收；逐章CPU实验与HTML/PDF通过 |
| 后续章节 | 持续并行完善；实际进度见章节计划 |
| CPU训练环境 | Python 3.11.5、NumPy 2.4.6、PyTorch 2.15.0+cpu实测通过 |

可提前推进后续章，前置关系用于阅读和整合检查。每章独立验收后提交并同步远程；默认全书仅选择已完成章。

## 一键生成HTML / PDF

需要Node.js 22.13+；PDF使用已有Edge或Chrome。从仓库根目录执行：

```powershell
npm ci --ignore-scripts

# 已完成正式章节汇编
& ./scripts/build.ps1 -Format all

# 单章；未完成章会显示草稿标记
& ./scripts/build.ps1 -Format all -Chapter 00

# 独立排版样张
& ./scripts/build.ps1 -Format all -Sample
```

全书输出为build/html/deep-learning-journey.html和build/pdf/deep-learning-journey.pdf；单章为chapter-00.html/.pdf。图和数学字体已内嵌，PDF保留矢量、中文、书签和页码。源码附件的携带方式见构建说明。build/和node_modules/不进入Git。

## 本机实践与格式检查

先按训练依赖说明准备项目 .venv，再运行：

```powershell
& ./.venv/Scripts/python.exe -B scripts/cpu-smoke.py --with-frameworks
& ./scripts/check-environment.ps1 -PythonPath ./.venv/Scripts/python.exe
npm test

# 导出后的桌面/窄屏、锚点、SVG边界、离线资源和PDF结构检查
node scripts/check-publication.mjs --name chapter-00 --capture
```

检查工具将报告放在build/qa/；每章仍需目视检查实际分页和图文。后续训练规模以CPU、小数据、小模型为默认，正式运行前复查资源。
