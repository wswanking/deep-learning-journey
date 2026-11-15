# 第20章CPU实践：语料、next-token目标与困惑度

[章节正文](../../docs/chapters/20-language-model-pretraining.md) · [原创数据说明](data/README.md)

## 运行

在正式仓库根目录执行：

~~~powershell
& ./.venv/Scripts/python.exe -B examples/20-language-model-pretraining/run.py
~~~

依赖项目锁定环境：Python3.11.5、NumPy2.4.6、PyTorch2.15.0+cpu；无需下载数据或GPU。入口使用CPU、float64、单线程和seed20。结构探针只读导入第21章model.py，因此需要一并保留该文件；不加载或修改21基座检查点。

结果写入build/experiments/20-language-model-pretraining，包括results.json、corpus-manifest.json、清理划分后的JSONL、两个NPZ计数模型和SVG。输出均可重建，不提交模型缓存。只想核对清理/分窗可阅读corpus.py。

## 固定协议

- 自制40条JSONL，固定NFC/换行/行末空白清理，删除2重复、1空白、1过短，保留12组36篇。
- 12组按SHA256(20261115|group_id)排序，8/2/2组分到训练/验证/测试；文档24/6/6，有效后续字节目标971/239/245。组与整篇哈希不相交；公共模板仍共享，没有实现全量模糊去重。
- 词表固定UTF8 byte0..255；没有BOS/EOS。首字节给定，不计首字节或文档结束概率；仅评分文档内相邻后续字节。
- 窗口最多64输入位置，取65原始字节后移位；每窗上下文重置，目标不重复。输入右padding为0但mask=False、目标=-100。真实NUL字节0保留有效。
- unigram/bigram只从训练目标计数拟合；候选α=0.01/0.1/1，最低验证NLL选择后冻结并评价测试。计数模型无需优化器、学习率或训练轮数。
- 随机11408参数TinyDecoderLM探针只验因果/右padding接口，没有训练，不报告其语言能力。

## 实测与复现

2026-11-15正式CPU环境：21项检查通过。首次20项运行9.564秒（含导入4.894秒）、Python跟踪峰值141.75MiB；补NUL核验后9.146秒。跟踪峰值不是RSS，不含大量原生张量内存。耗时随负载变化。

| 模型 | 选中α | 验证PPL | 测试PPL | 测试bits/预测字节 |
| --- | --- | --- | --- | --- |
| unigram | 0.1 | 16.32029844 | 17.04305942 | 4.09111243 |
| bigram | 0.01 | 4.46559982 | 5.80906888 | 2.53830694 |

这是高度模板化的小计数实验，不能作为真实语言泛化或大模型能力证明。bits分母为声明的有效后续字节数。

21项断言覆盖清理、组/哈希隔离、重复冲突拒绝、坏行划分泄漏、移位分窗、文档边界、真实NUL、有效目标数、NumPy/PyTorch maskedCE（误差8.88e-16）、padding梯度0、复制未移位作弊、PPL手算/全局归约/非上界、公共因果性（误差0）与右padding（1.65e-17）、评估不改计数、两个模型NPZ保存恢复和独立泄漏玩具反例。

复制策略的错误未移位CE为5.26e-7，正确移位约20；另一个独立玩具的违规测试计数把PPL从256降至8.001，不是正式成绩。不要将这类低数字解释为能力提高。

复跑到单独目录：

~~~powershell
& ./.venv/Scripts/python.exe -B examples/20-language-model-pretraining/run.py --output-dir build/experiments/20-language-model-pretraining/repeated
~~~

固定环境应得到相同语料清单、候选验证结果、所有计数及概率和正式指标；时间/内存字段不要求相等。不同平台浮点库可能有极小差异。NPZ保存计数、概率和α，入口重新加载并逐元素验证。

## SVG与导出

~~~powershell
node scripts/generate-chapter20-figures.mjs
& ./.venv/Scripts/python.exe -B examples/20-language-model-pretraining/run.py --figure-dir docs/assets/20-language-model-pretraining
& ./scripts/build.ps1 -Format all -Chapter 20
~~~

3幅原理图由Node生成，2幅实际结果图来自run.py；图形、文字、曲线均为真正矢量，无嵌入位图。独立publication与PDF审计结果见章节末验收记录。

## 文件

- corpus.py：固定清理、精确去重、组划分、移位窗口与collate。
- run.py：计数拟合/评估/恢复、NumPy/PyTorch交叉核验、失误反例及结果图。
- data/raw.jsonl：原创数据源；清理规则与许可见同目录README。
