# 第18章 VAE与GAN CPU实践

在仓库根目录执行：

```powershell
& ./.venv/Scripts/python.exe -B examples/18-vae-and-gan/run.py
& ./.venv/Scripts/python.exe -B examples/18-vae-and-gan/run.py --write-figures
# 仅从已有真实结果重建SVG：
& ./.venv/Scripts/python.exe -B examples/18-vae-and-gan/figures.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/18-vae-and-gan/run.py
.venv-linux/bin/python -B examples/18-vae-and-gan/run.py --write-figures
# 仅从已有真实结果重建SVG：
.venv-linux/bin/python -B examples/18-vae-and-gan/figures.py
```

现有CPU PyTorch/NumPy即可，无下载。默认输出 `build/experiments/18-vae-and-gan/results.json`，可指定 `--output-dir`。默认运行不改正文图。

## 数据与协议

原创二维等权高斯混合中心(-2,0)/(2,0)，真实标准差0.2；训练512、验证256，数据种子101/102，无测试集。模型种子11/22/33，固定终点不按验证选优。

- VAE148参数、潜维1、观测高斯标准差0.25。Adam0.01，600步、批次64。重建NLL按坐标求和含常数再按样本平均，KL解析算。验证每点一次固定噪声，seed550，不是精确似然。
- 压力VAE只把KL系数从1改100，优化加权目标，不等同标准负ELBO；标准ELBO甚至可能变差，保留这一证据。
- 普通GAN的G82参数、D65参数、潜维2，Tanh隐藏层；Adam0.003，betas(0.5,0.999)，1000轮，每轮1D+1G、批次64。G用非饱和目标，D用两个BCE均值的和。
- 受限G(z)=c只有2参数，故意不依赖噪声，真实对抗训练后所有生成点相同；这是容量失败反例，不是普通GAN自发坍塌频率。
- D步detach假样本；G步冻结D参数但保留输入求导，实际核对三种梯度边界。

每个生成配置抽256样本。距离模式中心<=0.75记接近；每模至少5%样本记覆盖。该指标不衡量模式内方差，完整输出保留坐标标准差和点图，不能冒称通用质量指标。图固定第一模型种子11，不挑最好结果。

## 已运行结果

2026-11-15 WindowsCPU/Python3.11.5/NumPy2.4.6/PyTorch2.15.0+cpu，float64、线程1、确定性算法；12次训练和手算核对约15.465秒，不含导入或SVG生成，51核对通过。

标准VAE与普通GAN均双模但不完美；压力VAE KL<.001且样本挤在中心。普通GAN有模式内方差不足，常量G全部点同一位置，靠近比例虽100%却仅一模。正文给所有种子结果和边界。

`vae-seed11.pt` 为自己的可信推理权重，重新加载固定解码误差0，不是精确续训检查点。VAE纯权重1184字节，普通GAN两网1176字节，不含框架/优化器/激活，不是进程峰值内存。

六幅SVG由 `figures.py` 原生生成，包含ELBO手算、概率计算路径、交替训练、真实点图和曲线，无位图嵌入。
