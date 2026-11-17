# 第08章实验：计算图、反向传播与自动微分

[对应章节](../../docs/chapters/08-backpropagation.md) · [CPU环境](../../requirements/README.md)

从仓库根目录执行，无外部数据和网络要求：

```powershell
& ./.venv/Scripts/python.exe -B examples/08-backpropagation/run.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/08-backpropagation/run.py
```

默认报告为 `build/experiments/08-backpropagation/results.json`，第四幅结果图在figures/。发布配图重建：

```powershell
node scripts/generate-chapter08-figures.mjs
& ./.venv/Scripts/python.exe -B examples/08-backpropagation/run.py --figure-dir docs/assets/08-backpropagation
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
node scripts/generate-chapter08-figures.mjs
.venv-linux/bin/python -B examples/08-backpropagation/run.py --figure-dir docs/assets/08-backpropagation
```

独立复跑使用 `--output-dir build/experiments/08-backpropagation/repeated`。

固定CPU float64、单线程、种子8。9参数ReLU手算例全部预激活远离0；3→4→2 tanh网络的26参数全部进行中心差分与手写/AD对照，主扰动1e-5，另扫描1e-2到1e-8。2→3→1网络在32个合成训练样本上比较两套更新轨迹：13参数、学习率0.1、完整批次梯度下降200步，无动量/正则化。

损失统一定义为：输出分量平方差先求和，乘1/2，再对样本平均。它与默认多输出MSE平均归约不同。训练曲线只用于实现一致性检查，未划分验证/测试，不作泛化声明；不保存或声称验证了未要求的断点续训。

14项检查涵盖完整手算前向/反向/更新、共享参数两条路径、26参数三方梯度、偏置广播归约、grad缓冲6→12→6、autograd.grad仅返回结果、同一平方旧图再次反向的预期错误、保留图后的累积、detach断路、no_grad、修改保存值的预期错误、VJP、动态图实际分支和两种训练轨迹。

首次实测2026-11-15：Windows build26200、Python3.11.5、NumPy2.4.6、torch2.15.0+cpu；14/14通过。ReLU损失2→0.13005；平滑网络手写/AD最大绝对梯度误差1.11e-16，差分1e-5的最大绝对误差3.03e-11；200步训练损失0.175330→0.003560，轨迹参数最大差1.11e-16。

含导入7.432秒，其中导入5.311秒；Python跟踪峰值103.43 MiB，不包含完整原生张量或进程RSS。耗时和资源不作为跨机器一致性标准。JSON保留逐项结果、全部差分扫描、每10步训练记录和真实版本。数学小例子与实测值的解释均在正文中。

失败检查捕捉的是特定已说明的操作和异常类别，不锁死版本相关错误字符串。平方旧图需要保存中间值，因此二次反向被拒绝；不把这个结果夸大成所有简单旧图必然都会拒绝。
