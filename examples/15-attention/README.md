# 第15章 注意力 CPU 实验

在仓库根目录执行：

```powershell
& ./.venv/Scripts/python.exe -B examples/15-attention/run.py
& ./.venv/Scripts/python.exe -B examples/15-attention/run.py --write-figures
# 仅从已有真实日志重建原生SVG：
& ./.venv/Scripts/python.exe -B examples/15-attention/figures.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/15-attention/run.py
.venv-linux/bin/python -B examples/15-attention/run.py --write-figures
# 仅从已有真实日志重建原生SVG：
.venv-linux/bin/python -B examples/15-attention/figures.py
```

依赖现有NumPy与CPU PyTorch，无下载。默认输出 `build/experiments/15-attention/results.json`，可用 `--output-dir` 改变。`--write-figures` 才会改正文图片。

## 对象与协议

这是算子实践，不是语言模型训练：给定Q/K/V、可训练参数0，优化器、训练步数、数据划分和检查点均不适用。不要把23项核对称作模型训练成功。

- 可完整手算的两查询/三键/值宽2示例，权重为(.5,.25,.25)与(.25,.5,.25)，输出(1.5,2)、(1,3)。另屏蔽中间键，检查重新归一化。
- 两个长度4的批次，真实长度4/2，键宽3、值宽2；因果允许mask与有效键取交集。允许True，与PyTorch函数式SDPA一致，与MultiheadAttention的禁止True不同。
- 修改未来键+1000、值-1000，检查前两位置输出差和到未来键值的梯度均0；去掉mask的真实对照泄漏约1000.3545。
- padding键不影响有效查询；无效查询输出另外置0，不假装padding-key mask自动完成全部处理。
- NumPy入口检查形状/有限值；每条计算查询必须保留至少一个键，全遮挡提前报错。Torch手写核心在本实验有效输入上与其及官方SDPA核对。
- 独立标准正态缩放模拟：种子20261115，键宽2/8/32/128/256，每配置512查询、每查询8键，真实测量分数方差、熵nat和最大权重。
- 联合重排K/V输出不变；相同V、不同权重的输出反例。没有把权重热图等同因果解释。

2026-11-15 WindowsCPU、Python3.11.5、NumPy2.4.6、PyTorch2.15.0+cpu，float64、线程1、确定性算法。23项核对通过，核心计算约0.0263秒（排除导入/图生成）。手算QKV载荷128字节，不代表过程峰值内存。

官方SDPA的误差为4.44e-16（手算）、1.11e-16（合并mask），跨后端/版本数值尾差可能变化。五幅图由 `figures.py` 生成，只有原生SVG路径、文字和形状。
