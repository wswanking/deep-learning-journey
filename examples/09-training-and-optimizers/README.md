# 第09章 CPU 实验

这份实验把训练循环、优化器、学习率、批次与检查点恢复放在同一组可核对协议中。数据由 `y=sin(2x)+0.1x` 自行生成，无需下载或 GPU。

## 运行

在仓库根目录执行：

```powershell
& ./.venv/Scripts/python.exe -B examples/09-training-and-optimizers/run.py
# 重建与实际日志一致的纯SVG：
& ./.venv/Scripts/python.exe -B examples/09-training-and-optimizers/run.py --write-figures
# 仅使用已有JSON重建图，不重新训练：
& ./.venv/Scripts/python.exe -B examples/09-training-and-optimizers/figures.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/09-training-and-optimizers/run.py
# 重建与实际日志一致的纯SVG：
.venv-linux/bin/python -B examples/09-training-and-optimizers/run.py --write-figures
# 仅使用已有JSON重建图，不重新训练：
.venv-linux/bin/python -B examples/09-training-and-optimizers/figures.py
```

默认输出到 `build/experiments/09-training-and-optimizers/`，可用 `--output-dir` 指定。输出为真实 `results.json` 与第93次更新后的检查点 `step-93-checkpoint.pt`。仅加载自己可信的检查点。

## 协议

- CPU、float64、PyTorch 计算线程1，启用确定性算法；已在 Python3.11.5 / PyTorch2.15.0+cpu / NumPy2.4.6 实际运行。
- 64个等距训练点与63个中点验证数据，彼此不重叠；仅插值，无独立测试集。
- `Linear(1,16) -> Tanh -> Linear(16,1)`，49参数，损失为平均平方误差。
- 初始参数种子20261115；独立采样器种子17，每轮洗牌后不放回取批次。
- 九种配置处理相同3840个样本；B16更新240次，B64更新60次。Adam系数0.9/0.999，eps1e-8；所有组裁剪数据梯度全局范数到1。
- Adam耦合衰减显式 `decoupled_weight_decay=False`；AdamW解耦，正衰减组lambda0.01，包含偏置。训练曲线是数据MSE，不含正则项。
- 预热余弦20步升到0.01，再降到0.001；实际逐步学习率写入JSON。

## 实际核对与边界

2026-11-15运行通过33项核对：49参数、九组有限并下降、给定梯度的SGD/Momentum/Adam/AdamW手算、梯度累积、全局裁剪、不等批次样本加权、过大学习率二次函数发散、零衰减Adam/AdamW等价、保存后精确恢复。

检查点含模型、优化器、下一更新编号、调度配置、采样排列/游标/独立RNG、全局torchRNG和日志。恢复到240步后最终参数最大绝对差0，完整历史相同。跨平台/版本不承诺位级一致。

脚本没有给所有优化器做超参数搜索。本次Momentum终点较好，预热余弦反而比恒定AdamW差，完整保留失败证据。B64比较同时改变更新次数，不能只归于噪声。计时排除导入与图生成；392字节仅为参数数值载荷，不是进程峰值内存。

`figures.py` 用真实数值输出原生SVG路径、文字与形状；二维二次目标的优化路径是独立演示，不冒充MLP参数投影。图中未嵌入位图或滤镜。
