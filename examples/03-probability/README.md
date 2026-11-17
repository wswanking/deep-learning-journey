# 第03章：概率、统计与信息论 CPU 实验

使用合成离散分布核对联合/条件概率与 Bayes，固定种子采样估计期望和方差，比较有序 Bernoulli 数据的似然，以及离散熵、交叉熵和双向 KL。无需数据下载、模型训练、PyTorch 或 GPU。

从正式仓库根目录运行：

```powershell
& ./.venv/Scripts/python.exe -B examples/03-probability/run.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/03-probability/run.py
```

默认原始 JSON 保存到 `build/experiments/03-probability.json`，同时显示在终端。该目录已由项目忽略，不将机器路径与每次耗时作为手工维护源文件。输出位置和种子可以显式指定：

```powershell
& ./.venv/Scripts/python.exe -B examples/03-probability/run.py `
    --seed 20261115 `
    --output build/experiments/03-probability.json
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/03-probability/run.py `
    --seed 20261115 `
    --output build/experiments/03-probability.json
```

依赖：项目 Python 3.11 与 NumPy；本章实测版本见[章节正文](../../docs/chapters/03-probability.md)。环境安装与固定版本依据根目录 requirements 的实测记录，不在此安装全局软件。

`checks.status` 和每个检查名称描述实际核对范围。若某项不符合预设关系，程序抛出错误并以非零状态退出。采样误差允许 0.03 的绝对范围，用于这组固定种子和10,000个样本的教学检查；它不证明所有随机种子都满足该界限，也不要求误差随样本数逐项下降。

所有信息量用自然对数，单位 nat。函数验证输入分布非负、归一化且形状相同；不悄悄重归一化或给零概率加 epsilon。`p>0,q=0` 时交叉熵与 KL 在数学上为正无穷。标准 JSON 没有无穷数，因此输出使用字符串 `"+inf"` 或 `"-inf"`；这些不是有限测量值。`p=0` 的项按定义跳过，包含 `p=q=0`。

`sample_array_bytes` 只统计10,000个 float64 样本的数组载荷，**不是进程总内存**。程序记录实际耗时，不做训练速度基准；进程峰值内存未测量。数据划分、模型参数、优化器、轮数、检查点对本次数值模拟不适用，JSON 明确标记这些项。

四幅 SVG 的可编辑源文件在 `docs/assets/03-probability/`；公式曲线和柱形位置可从同源生成程序重建：

```powershell
node scripts/generate-chapter03-figures.mjs
```

生成图使用正文中的解析公式和固定概率，不把某次采样曲线伪装为理论分布。图中不包含位图或外部图片。
