# 第14章实验：token、BPE与嵌入

~~~powershell
& ./.venv/Scripts/python.exe -B examples/14-tokenization-and-embeddings/run.py
~~~

从仓库根目录执行，依赖项目现有NumPy和CPU PyTorch。默认输出build/experiments/14-tokenization-and-embeddings：

- results.json：版本/Unicode数据库、17项检查、完整分词探针/合并计数、训练损失曲线、相似度及资源记录。
- byte-bpe.json：固定256字节基表、16轮有序合并、正规化/边界/并列规则。
- embedding.pt：12词映射、两张12×8参数表与训练协议；入口保存后恢复并核验logit逐项一致。
- figures/：实际BPE长度曲线与嵌入训练SVG。

发布所有矢量图：

~~~powershell
node scripts/generate-chapter14-figures.mjs
& ./.venv/Scripts/python.exe -B examples/14-tokenization-and-embeddings/run.py --figure-dir docs/assets/14-tokenization-and-embeddings
~~~

## 协议与边界

语料和18个训练对完全原创，使用仓库许可证。三轮字符BPE手算词典为abab×3、abac×2、cab×1，原子词尾为单独符号；合并计数9/4/3，长度29→20→16→13。这与byte BPE分开。

byte BPE在正文列出的六条中英文训练字符串及其频次上学习16次合并；初始保留所有256字节，最终272。无正规化、无预切分，可跨文档内空格，不跨文档；并列按整数符号对升序。六条冻结探针含组合重音和未见扩展汉字，严格拼字节还原原文；不宣称是生产tokenizer或真实语义评测。

嵌入词表[猫、狗、兔、苹果、香蕉、梨、跑、跳、睡、甜、酸、熟]。前3个中心各配跑/跳/睡，后3个中心各配甜/酸/熟，共18个完整softmax训练对。两张12×8独立表共192参数，0.1倍标准正态初始化，种子14；float64 CPU单线程，Adam学习率0.05、默认betas/eps、无衰减，完整批次200步，每20步记录。

这仅演示学习目标与梯度，未划分语义验证/测试集、未搜索超参数，不将训练曲线或人工共现近邻当作泛化成绩。语料结构由作者规定，因此同组近邻有教学意义、没有自然语言能力证明。

## 已运行

2026-11-15，17项检查两次通过，入口成功退出，无数值警告；两次合并配置、ID、完整曲线、近邻和全部权重逐项一致。Python3.11.5、NumPy2.4.6、PyTorch2.15.0+cpu；Windows标准输出显式UTF-8，避免GBK无法打印组合重音/扩展汉字。首个成功运行26.719秒（导入12.498秒），复跑15.993秒；Python跟踪峰值136.10MiB。实际耗时与Python跟踪分配在JSON中；跟踪峰值不等于完整进程RSS，不含大量原生张量内存。

- byte BPE加权训练token数225→107，272个token；六条探针无损往返与配置恢复通过。
- one-hot/查表前向、重复索引梯度、PAD梯度与均值mask、ID重编号及上下文手算通过。
- 全softmax手导/NumPy/自动微分梯度一致，192参数训练损失2.4958838351→1.0986321332，下确界log3=1.0986122887。
- 猫的前两个中心近邻为狗/兔，苹果为梨/香蕉；仅比较参与中心输入训练的6行。
- 检查点恢复后的18对logit完全一致。

[完整原理、图示与10题解析](../../docs/chapters/14-tokenization-and-embeddings.md)。
