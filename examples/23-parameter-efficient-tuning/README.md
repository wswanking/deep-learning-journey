# 第23章：LoRA与明确标界的CPU均匀int4替代

[完整教程](../../docs/chapters/23-parameter-efficient-tuning.md)

## 运行

在仓库根目录，先按第21章入口生成其基座，再执行：

~~~powershell
& ./.venv/Scripts/python.exe -B examples/23-parameter-efficient-tuning/run.py
~~~

Linux / WSL：

~~~bash
.venv-linux/bin/python -B examples/23-parameter-efficient-tuning/run.py
~~~

Python3.11.5、NumPy2.4.6、PyTorch2.15.0+cpu；CPU float64单线程。无需新增依赖。只读导入21的model/tokenizer与22的sft_data，不覆盖它们。需保留22原始数据文件。

默认build/experiments/23-parameter-efficient-tuning输出results.json、full-ft.pt、两种adapter.pt及一个故意不匹配基座的拒绝加载检查文件。模型/构建输出不作为源代码提交。

## 协议与真实结果

原始35328参数基座来自21；22原创复制/反转模板，15个无序字母对按组9/3/3，样本72/24/24。只监督两回答字母+LF共3个byte。数据与基座SHA256、模板/分词规则均进入报告；LF仅本任务停止约定。

三方法顺序训练：full FT、q/v LoRA、所有Linear权重均匀int4冻结+q/v LoRA。rank4/alpha8，4个32×32目标矩阵新增1024个元素。未训练浮点和均匀量化基座也报告。所有训练800updates、batch16、AdamW lr.0015、betas(.9,.999)、eps1e-8、decay0，无dropout；A初始化seed2300，采样seed2201。每50步按验证回答NLL选择，所有配置冻结后才测测试/贪心完整回答。

| 方法 | 最佳步 | 测试回答NLL | 测试完整正确 | 训练完整正确 |
| --- | --- | --- | --- | --- |
| 冻结浮点基座 | 0 | 9.973132 | 0/24 | 0/72 |
| 仅均匀int4 | 0 | 9.978869 | 0/24 | 0/72 |
| full FT | 250 | 0.947816 | 6/24 | 71/72 |
| LoRA | 750 | 1.552767 | 0/24 | 20/72 |
| 均匀int4+LoRA | 750 | 1.607355 | 0/24 | 22/72 |

2026-11-15首轮34项通过，23.407秒含导入1.505秒；训练及各自检查约6.781/5.739/6.332秒。耗时含评估/恢复等，不是稳定吞吐基准。没有测RSS/GPU显存。训练方法都格式有效24/24，但LoRA完整答案0/24；保留失败，未按测试改rank/lr。统一学习率、q/v覆盖、小基座、单种子不能推出普遍算法排名。

## 四位替代与QLoRA的边界

adapters.PackedInt4Linear使用逐行对称均匀−7..7、码q+8（码0未用）、两码打包uint8、float64尺度，每次前向解量化到float64。零行尺度1，奇数权重补半码后裁掉。所有原始Linear权重量化，embedding/norm保留浮点。冻结偏置以buffer存储。它没有NF4、双重量化、分页优化器或低位加速核，**不是完整论文QLoRA**。完整NF4后端未在本锁定环境安装/验证，未执行/不报告其显存或速度。

量化仅压缩持久存储，计算有额外解量化矩阵；张量账本不含激活/临时权重/框架/分配器/RSS。训练末的持久张量合计fullFT1104.227KiB、LoRA308.063KiB、替代133.563KiB。

## 文件与核验

- adapters.py：LoRALinear、packed int4、注入、冻结指纹、合并副本、adapter-only保存/恢复、字节账。
- run.py：手算/NumPy核对、三方法训练、验证选择、完整生成、元数据。
- figures.py：3幅原理SVG+3幅真实结果SVG，无位图。

34项核验涵盖手算前向/梯度、初始零增量、双零停滞、行列方向、冻结输入梯度、量化打包/零行/奇数形状、no_grad反例、参数数、有限梯度、冻结grad=None且状态未变、merge误差<1e-11、仅8个A/B张量、bitexact恢复、同批次hash、原始基座未改、错误基座hash拒绝。

adapter文件存准确基座文件hash、目标层、rank/alpha、数据/词表/模板及量化配方；不是独立完整模型，也不是可继续同轨迹训练的完整优化器断点。load_adapter先检查基座hash再注入并严格验证键/形状/dtype。

## 独立复现与导出

~~~powershell
& ./.venv/Scripts/python.exe -B examples/23-parameter-efficient-tuning/run.py --output-dir build/experiments/23-parameter-efficient-tuning/repeated --skip-figures
& ./scripts/build.ps1 -Format all -Chapter 23
node scripts/check-publication.mjs --name chapter-23 --capture
~~~

Linux / WSL：

~~~bash
.venv-linux/bin/python -B examples/23-parameter-efficient-tuning/run.py --output-dir build/experiments/23-parameter-efficient-tuning/repeated --skip-figures
& ./scripts/build.ps1 -Format all -Chapter 23
node scripts/check-publication.mjs --name chapter-23 --capture
~~~

报告保留原始曲线、测试每条生成及误例；固定环境下指标/模型张量应一致，时间/序列化容器字节不要求跨平台相同。2026-11-17修复相对CLI输出路径记录异常后，34项完整复跑25.614秒；与首轮的全部轨迹/选择/指标/逐例生成及三个模型张量逐项一致。HTML/PDF17页/100公式/6SVG/37链接，publication1440/390与PDF审计通过；17页与6图已逐页/放大查看。
