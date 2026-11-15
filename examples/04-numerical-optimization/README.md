# 第04章实验：稳定概率计算与学习率对照

对应[第04章](../../docs/chapters/04-numerical-optimization.md)，用固定解析函数验证浮点误差、softmax/LSE的稳定等价变形和梯度下降的步长条件。

## 运行

从仓库根目录执行：

```powershell
& ./.venv/Scripts/python.exe -B examples/04-numerical-optimization/run.py
```

依赖NumPy与CPU版PyTorch；本机Python 3.11.5、NumPy 2.4.6、PyTorch 2.15.0+cpu。脚本强制CPU与1个PyTorch线程。格式对比含float32和float64，概率与优化用float64。无外部数据、随机采样或网络访问。

输出为 `build/experiments/04-numerical-optimization/results.json`，并以[矢量生成程序](figures.py)从当前配置生成 `docs/assets/04-numerical-optimization/` 中四幅SVG。只重跑数值检查可加 `--skip-figures`；报告路径可用 `--output` 指定。

## 数值检查

- 通过 `finfo` 与 `nextafter` 查询浮点间距及上下界，验证float32的非结合加法与平方根消去现象。
- 比较正负极值分数的朴素softmax预期失败与稳定输出，检查每行归一化、共同偏移不变性，并与框架接口核对。
- 比较先softmax后log与直接log-softmax，展示普通概率下溢后对数信息的区别。
- 固定二维目标 `0.5*(theta1**2+4*theta2**2)`、初值 `[2,1]`，分别使用0.03、0.2、0.45和0.55的学习率更新40步。
- 每个点的解析梯度与自动微分核对，每个坐标与闭式解核对；检查慢速收敛、收敛、振荡收敛和发散的定量结果。

预期NaN与无穷值作为报告字符串记录，JSON仍符合标准。成功打印 `PASS`；公式、形状、有限性或预期现象不符时以错误退出。默认运行不应把朴素路径的预期失败当成整个实验失败。

## 实测及边界

2026-11-15默认CPU运行通过。40步目标分别约0.174964、3.5337e-8、3.5337e-8与4.32046e6；其中0.55第一步反而下降，后续发散。图04-3的发散面板只画前6步并标明，图04-4保留全部40步。

一次运行纯数值检查约0.075秒、含导入1.966秒，进程峰值驻留内存约198.2 MiB；不含SVG生成，具体每次以报告为准。解析构造无数据许可问题，训练划分、随机种子和模型检查点均不适用。本例不能替代真实模型的数据评价，也不能把0.5当作其他目标的通用学习率界限。
