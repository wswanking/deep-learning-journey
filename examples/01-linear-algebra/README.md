# 第01章实验：线性代数与张量

[对应章节](../../docs/chapters/01-linear-algebra.md) · [环境说明](../../requirements/README.md)

本实验把手算结果与 NumPy、PyTorch 的 CPU float64 结果逐项对照。输入是源码中公开的小数组，无需下载数据，无训练与检查点。固定种子为0；只有 Linear 的未使用初始化涉及随机数，随即被指定权重覆盖。

从仓库根目录执行：

```powershell
& ./.venv/Scripts/python.exe -B examples/01-linear-algebra/run.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/01-linear-algebra/run.py
```

默认 JSON 输出为 `build/experiments/01-linear-algebra/results.json`。也可以指定输出位置：

```powershell
& ./.venv/Scripts/python.exe -B examples/01-linear-algebra/run.py --output build/experiments/01-linear-algebra/repeated.json
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/01-linear-algebra/run.py --output build/experiments/01-linear-algebra/repeated.json
```

共18项检查，均检查数值及相关形状：

1. 对象的形状、轴数与元素数。
2. 一维向量和显式行/列矩阵。
3. 逐元素积、内积与标量/二维输出。
4. 手算矩阵乘法。
5. 列向量仿射计算。
6. 行批次仿射计算。
7. `nn.Linear` 权重存储与公式对应。
8. 偏置广播与显式矩阵加法相同。
9. 不合法广播在两库中均抛出预期类别的异常。
10. 合法广播产生错误的所有样本两两差，以及修正后的结果。
11. 归约、`keepdim` 与每行中心化。
12. reshape 不等于 transpose。
13. 批次矩阵乘法与最后两轴转置。
14. 向量1、2、无穷范数。
15. Frobenius、逐行2范数及矩阵2范数的区别。
16. 二维变换与教材图示坐标一致。
17. 固定例子的线性可加性与齐次性。
18. 非零偏置破坏线性的反例。

异常演示是预期行为，捕捉成功记为检查通过；意外接受错误形状会导致失败退出。数值使用 `rtol=1e-10`、`atol=1e-12`，并单独核对形状，避免广播掩盖数值比较中的错误。

报告记录实际 Python/NumPy/PyTorch 版本、设备、输入输出、预期异常类别、执行时间和 Python 分配跟踪峰值。**跟踪峰值不等于整个进程或 PyTorch 原生内存**，不能据此推断训练内存需求。启动导入时间计入总耗时；不同机器、缓存状态和其他程序负载会影响它，不以速度作为验收条件。

复现通过条件是18项全部通过，关键固定结果为：

- 矩阵积 `[[-2, 4], [-2, 13]]`。
- 行批次线性层输出 `[[4.5, -2], [-3.5, 6]]`。
- 错误广播形状 `(3,3)`、平均平方差 `4/3`；修正后形状 `(3,1)`、值0。
- 向量 `[-3,4]` 的1、2、无穷范数为7、5、4。
- `diag(3,4)` 的 Frobenius 范数为5，矩阵2范数为4。

若导入失败，先核对自己调用的是项目 `.venv` 解释器，再按环境说明准备依赖；不要把依赖缺失当作公式错误。导出 HTML/PDF 是另一项格式验收，由章节维护者按[构建说明](../../docs/build-guide.md)执行。

实际运行：2026-11-15 21:34（Asia/Shanghai），Python 3.11.5、NumPy 2.4.6、PyTorch 2.15.0+cpu，18/18通过，无警告。含导入总6.177秒，Python跟踪峰值105.54 MiB；该值不等于进程总内存。固定仿射层共8个参数（6权重、2偏置）。
