# 第07章实验：固定权重MLP与XOR前向

对应[第07章](../../docs/chapters/07-multilayer-perceptron.md)。完整XOR四点按固定2→2→1 ReLU网络求值，使用NumPy、手算基准和PyTorch模块相互核对。本章不训练参数。

## 运行

```powershell
& ./.venv/Scripts/python.exe -B examples/07-multilayer-perceptron/run.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/07-multilayer-perceptron/run.py
```

从仓库根目录执行，依赖项目NumPy和CPU PyTorch。本机Python 3.11.5、NumPy 2.4.6、PyTorch 2.15.0+cpu，CPU、float64、1个PyTorch线程。构造种子7，全部初始参数被明确的固定值覆盖；无外部数据或随机样本。

报告为 `build/experiments/07-multilayer-perceptron/results.json`，同时依据实际表示和解析函数生成三幅纯SVG。只核对数值可加 `--skip-figures`，报告路径可用 `--output` 指定。成功打印 `PASS`，不一致以错误退出。

## 核对内容

- 完整四点的预激活、隐藏表示、分数、sigmoid概率与预测。
- PyTorch模块与NumPy及逐样本计算一致，置换输入行只置换输出行。
- 2→2→1带偏置网络有9个参数；批次维度不进入参数量。
- 移除ReLU后与有效仿射层一致；相同固定参数的真值表正确率为3/4。
- 另一组非退化仿射矩阵验证合并公式；三种激活与各自解析定义一致。

默认实际分数为 `[-0.5,0.5,0.5,-0.5]`，概率约 `[0.37754,0.62246,0.62246,0.37754]`，四点全对。它证明构造的网络能表达这张真值表，不代表训练成功或数据泛化。

本例无训练/验证/测试划分、优化器、学习率、训练轮数和模型检查点。一次本机数值检查约0.793秒，含导入2.735秒；输入数组载荷64字节不是进程内存，进程峰值内存未测量。图源为[figures.py](figures.py)，来源与推导详见章节正文。
