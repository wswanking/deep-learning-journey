# 第22章：共用基座上的SFT对照

## 运行

先执行第21章预训练，生成其`base-model.pt`，再从仓库根目录运行：

```powershell
& ./.venv/Scripts/python.exe -B examples/22-instruction-tuning/run.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/22-instruction-tuning/run.py
```

本次验收实际先复制为冻结副本，然后使用：

```powershell
& ./.venv/Scripts/python.exe -B examples/22-instruction-tuning/run.py --base-checkpoint build/experiments/22-instruction-tuning/base-used.pt --skip-figures
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/22-instruction-tuning/run.py --base-checkpoint build/experiments/22-instruction-tuning/base-used.pt --skip-figures
```

默认800更新；`--check-data-only`仅检查数据/mask/梯度，不执行SFT。`--skip-figures`不更新图，默认完整命令会从报告生成五幅SVG。无需下载或额外训练框架。

## 基座与结果追踪

只读复用`examples/21-tiny-language-model/model.py`的TinyDecoderLM/load_checkpoint，以及固定UTF8字节0..255。词表256、context64、宽32、4头、2层、FFN64、Dropout0，共35,328参数；本章两组全参数更新。

预训练600步终点的基座文件与本章`base-used.pt`的SHA256：

```text
326e1a5e405fc3280aae20e57f3eadd7ef0d7f54df3cb35af6a87bc679068e5a
```

基座参数状态SHA256：

```text
60530df67c1c9013e573b57a044c1e996e09b455216d7671ee59835198aca582
```

结果在`build/experiments/22-instruction-tuning/`。schema1检查点`response_only.pt`与`all_tokens.pt`包含共用模型配置、词表元数据、基座状态指纹、数据哈希、模板版本和选中更新数；可由21的load_checkpoint读取。元数据补入后再次核对所有参数未改变。报告记录最终检查点文件SHA256，便于后续章节绑定。

回答监督检查点SHA256：

```text
77d94287f5b8d4f1df3a1aaa2c937ec026c1bf1e42c9c9380985faee61e75201
```

## 数据与监督协议

原创JSONL在data目录，由`sft_data.py`确定性构造；默认运行只读取并验证，不覆盖已有数据。15个不同字母对，两个顺序×两任务×两措辞共120条，按group分72/24/24，不跨集共享同一字母对。复制与反转回答为两个不同字母。

模板为`[指令]\n{instruction}\n[回答]\n{response}\n`，总31或37bytes，输入30或36。byte0仅在attention_mask=False处当padding；LF10是本任务终止，不是通用EOS。回答监督只计两个回答字节+终止LF，前prefix_length-1个右移标签设-100。全词对照则监督全部有效下词标签。空监督与超长样例报错，不静默截断。

API：load_splits()->(splits,metadata)，collate(records,context_length,objective)->input_ids/attention_mask/labels/encoded；render_prompt(record)供推理。后续23可只读复用，不修改本章数据或模型。

## 实训与完整评价

Python3.11.5、PyTorch2.15.0+cpu、CPU、float32、单线程。两组同基座、同批次顺序（种子2201），AdamW lr0.0015、betas0.9/0.999、eps1e-8、decay0，batch16，800更新。每50更新按同一验证回答NLL选模，分别最佳250/750；两组训练目标不同，不直接比较各自训练loss。模型冻结后才测试。

2026-11-15 23:17（Asia/Shanghai）实训通过：

| 方法 | 训练整答 | 测试整答 | 测试格式 | 测试回答NLL |
| --- | --- | --- | --- | --- |
| 基座 | 0/72 | 0/24 | 0/24 | 9.97313 |
| 回答监督 | 71/72 | 6/24 | 24/24 | 0.94879 |
| 整段监督 | 72/72 | 3/24 | 24/24 | 1.62512 |

测试是24条但只有3个底层字母对，相关变体不是24个独立任务。两组格式改善却缺乏组合泛化，不代表通用指令能力或统计显著差异。JSON保留全部测试与训练生成，含未终止、错误任务/字母等失败。

机制检查：手算CE0.4581453659370775；忽略logits梯度0但prompt位置嵌入梯度范数1.025067；首答对齐、每条3监督字节、实际右padding、全忽略拒绝、单条/填充批次一致、初态与批次hash一致、检查点恢复均通过。

训练和评价42.737秒、含导入50.723秒，不含随后SVG/文档及元数据整理；JSONL大小13320字节不是进程内存，峰值未测量。改变基座、数据或预算后需重新记录结果。


2026-11-17整合复核：已修复共享模型导入导致的同名figures模块优先级，实际验证入口导入本章绘图文件并从已有实测报告重建五幅SVG。原训练、权重来源与15日数字保留。


## Padding检查的数值精度

训练仍使用原float32协议。批量与单条不同形状的计算存在舍入差异，确定性和单线程不保证两者逐位相同。当前检查同时要求float32最大绝对差小于1e-5、同权重float64交叉检查小于1e-11，并要求同形状下改变padding内容时真实token输出逐位相同。结果JSON记录各精度误差，不能只放宽float32阈值后省略语义检查。详见[PyTorch数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。

第21章生成基座后，可用Linux虚拟环境运行正反例回归：

```bash
.venv-linux/bin/python -B tests/padding.test.py
```

该测试须拒绝故意移除全部attention约束（因果及padding mask）以及只在float32批量输出注入偏差的模型。
