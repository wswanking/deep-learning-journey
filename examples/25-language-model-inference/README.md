# 第25章：CPU推理、采样与缓存实验

从仓库根目录运行：

```powershell
& ./.venv/Scripts/python.exe -B examples/25-language-model-inference/run.py
```

先使用第21章入口生成 `build/experiments/21-tiny-language-model/base-model.pt`；已有正确指纹的文件可直接复用。其他可信兼容检查点可通过 `--checkpoint` 指定，但不能套用正文既有数字。

- 环境：Windows，Python3.11.5、NumPy2.4.6、PyTorch2.15.0+cpu，float32，CPU单线程。
- 只读21基座，35328参数；无新增训练、优化器或学习率。
- 来源SHA256：

```text
326e1a5e405fc3280aae20e57f3eadd7ef0d7f54df3cb35af6a87bc679068e5a
```

## 实际做什么

`inference.py` 不改21模型，在25目录实现等长无padding缓存适配器、采样分布和逐行int8权重存储。缓存使用绝对位置与显式块因果mask，超过64位置拒绝；训练、变长批次和滑动窗口未实现。

`run.py` 先检查47位置的逐步/分块logits及greedy一致，再比较B/P=(1/8,1/24,1/40,4/24)。每条路线一次prefill和15次decode，表示16个输出分布；输入是相同固定随机字节，种子2503，无采样或提前终止。暖身5次、31次计时、路线顺序交替；耗时不含加载、分词、网络或排队。

采样手算用4候选分布和20000次固定种子频率；模型样例固定16字节并保存原始bytes。量化真实存int8，但每次计算反量化为float32，不是整数加速内核。量化副本状态字典恢复检查不等于训练恢复实验。

## 本机真实结果与范围

2026-11-15：分块/逐步logits最大误差3.81e-6/5.48e-6，greedy一致；B1/P24总中位重算5.889ms、缓存4.442ms，缓存解码约3649字节/秒。不同机器和并行负载会改变速度，不能按位置工作比外推加速。

张量存储141312→70400字节；量化探针logit最大误差0.16646、CE3.760314→3.763218。浮点重建实现总中位7.731ms，没有展示加速；其输入探针内容不同且单独测量，不能用于精确内核排名。全部检查与计时2.188秒，含导入5.093秒；峰值进程内存未测。

结果和每次原始计时：`build/experiments/25-language-model-inference/results.json`。图源 `figures.py` 从JSON生成对应章节的5幅真实SVG。只重画图可导入 `write_figures(result, folder)`，无需重跑或修改模型。


## 同输入的补充量化基准

在仓库根目录运行：

```powershell
& ./.venv/Scripts/python.exe -B examples/25-language-model-inference/run.py --matched-quantization-only
```

同一固定39字节、B1/P24、16个输出分布，无缓存；各暖身5次，再31次交替。完整原始记录写入build/experiments/25-language-model-inference/matched-quantization.json。此次浮点中位5.736ms，int8存储/float32重建7.160ms；源参数不变。结果限定于本机小模型和这个未融合实现；两份计时记录拥有独立时间戳，原缓存基准不以补充测量替换。
