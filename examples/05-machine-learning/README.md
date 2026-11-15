# 第05章实验：机器学习问题与线性基线

[对应章节](../../docs/chapters/05-machine-learning.md) · [CPU环境](../../requirements/README.md)

使用固定种子20261115的本项目合成数据，在CPU float64单线程下训练回归、二分类和三分类。无需外部数据、第三方数据许可或网络。实验用显式公式梯度更新，并以NumPy公式和PyTorch自动微分交叉核验。

```powershell
& ./.venv/Scripts/python.exe -B examples/05-machine-learning/run.py
```

默认输出在 `build/experiments/05-machine-learning/`：results.json、三个任务各自的模型JSON，以及figures/中三幅真正SVG。模型JSON包含权重、偏置、训练均值和标准差，保存后实际重建并核对验证集预测。前两幅静态教学图另由生成器重建：

```powershell
node scripts/generate-chapter05-figures.mjs
& ./.venv/Scripts/python.exe -B examples/05-machine-learning/run.py --figure-dir docs/assets/05-machine-learning
```

独立复跑并保留原报告：

```powershell
& ./.venv/Scripts/python.exe -B examples/05-machine-learning/run.py --output-dir build/experiments/05-machine-learning/repeated
```

每项任务按60%/20%/20%随机划分。预处理只拟合训练；模型候选仅接收训练/验证；候选为lr=0.05/0.2、零初始化、各300步完整批次梯度下降、每10步记录。选择所有记录中验证损失最低的快照，冻结参数与预处理之后，才评估测试一次。常数基线的预测值/类别概率也仅由训练目标拟合。没有正则化和动量，不保存优化器状态；JSON用于恢复冻结模型推理。

18项检查包括三种NumPy/手写/AD梯度，手算损失与极端logit稳定性，三份互斥完整划分和训练统计来源，验证损失超过训练拟合常数，模型保存恢复，XOR失败及预处理泄漏反例。完整过程不以测试分数达到某个阈值作为通过条件。

首次实测：2026-11-15，Windows build26200、Python3.11.5、NumPy2.4.6、PyTorch2.15.0+cpu。18/18通过；含导入7.413秒，其中导入4.856秒，Python跟踪峰值约104.68 MiB（非进程总内存，未包含完整原生张量）。

| 任务 | 选定lr/步 | 测试损失 | 测试辅助指标 |
| --- | --- | --- | --- |
| 回归 | 0.2/10 | MSE 0.022478 | RMSE 0.149926 |
| 二分类 | 0.2/130 | BCE 0.490591 | 准确率75.00% |
| 三分类 | 0.2/300 | CE 0.041107 | 准确率100.00%（60个合成点） |

回归原始坐标系拟合权重约2.495793、偏置约-0.386282。XOR四点的零初始化线性模型返回BCE=log2、准确率50%；这验证声明的失败实验，不代表任意线性模型在XOR上都只能分对一半。

每次JSON记录真实版本、生成数据哈希、集合ID、预处理统计、所有候选曲线、选中模型、三份指标、常数基线及资源。固定种子保证这份生成流程的输入可追溯，跨版本/硬件允许微小浮点差异。耗时、内存和时间戳不是重现判据。正文图的背景预测区域由SVG矩形构成，没有位图热图；黑线来自冻结模型的真实线性得分，不以生成机制替代拟合结果。
