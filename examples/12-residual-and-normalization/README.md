# 第12章：残差、归一化与迁移 CPU 实验

在仓库根目录执行：

```powershell
& ./.venv/Scripts/python.exe -B examples/12-residual-and-normalization/run.py
& ./.venv/Scripts/python.exe -B examples/12-residual-and-normalization/run.py --write-figures
# 仅从已有真实JSON重建纯SVG：
& ./.venv/Scripts/python.exe -B examples/12-residual-and-normalization/figures.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/12-residual-and-normalization/run.py
.venv-linux/bin/python -B examples/12-residual-and-normalization/run.py --write-figures
# 仅从已有真实JSON重建纯SVG：
.venv-linux/bin/python -B examples/12-residual-and-normalization/figures.py
```

依赖现有项目环境PyTorch和NumPy，不下载数据。输出默认在 `build/experiments/12-residual-and-normalization/`，可使用 `--output-dir` 指定。

## 三个受控实验

1. 残差：输入2→8、三个两层8→8块、四分类头，总492参数。同种子普通/残差共享初始权重，只改变是否执行x+F(x)。三种子11/22/33，源四象限任务，独立训练/验证各256点，Adam0.01全批次200更新。两种模型均可学会，残差不是每个种子都泛化更好。
2. BN/LN：小张量手算统计轴、前向有偏方差与运行无偏方差、train/eval、换批次、冻结参数仍变BN缓冲、每通道单值限制。使用float64；训练模型本身没有归一化，避免把其收益和残差混合。
3. 迁移：源残差模型迁移到32训练/256验证点的斜线二分类。替换后474参数，冻结仅18参数更新，微调与从零全部474更新。各100全批次更新，头学习率0.02，未冻结骨干0.001，固定终点，不用验证集选最优步。三个种子中冻结与微调均未超过从零，完整保留负结果。源预训练另有成本，不能称为总计算预算相等。

数据为本仓库自行合成的Uniform[-2,2]^2，源与目标数据生成种子101/102/201/202，无测试集。目标任务为x1+0.5*x2>=0。实验用于机制，不声称复制ResNet/ImageNet或通用视觉迁移收益。

## 输出与已执行事实

2026-11-15 Windows CPU、Python3.11.5、NumPy2.4.6、PyTorch2.15.0+cpu，float64、线程1、确定性算法。15次训练和张量演示约4.267秒，不含导入/图生成，40项核对通过。

`results.json` 保存完整曲线、环境、种子、各模式数值、冻结参数最大差和资源范围。`target-finetuned-seed11.pt` 仅为自己的推理权重；重新加载后logits最大误差0，不是精确续训检查点。全模型差异跨版本/平台不保证位级一致。

源权重纯数值3936字节、目标3792字节，不含优化器和框架，不是峰值内存。没有下载外部权重或处理第三方数据许可。

`figures.py` 用原生SVG文字、路径和形状生成六幅图，曲线源于真实日志。默认运行不改正文图，只有 `--write-figures` 明确请求时才重建。
