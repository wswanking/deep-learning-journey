# CPU 框架环境与第00章复测记录

> 实测：2026-11-15 21:31（Asia/Shanghai）。Windows 64位 / CPU，项目独立环境。
> [历史环境调研](environment.md) · [原始复测 JSON](framework-validation.json) · [依赖与复现](../../requirements/README.md)

## 1. 本次做了什么

在正式仓库创建独立的 .venv，使用现有 Python 3.11.5 安装 NumPy、Matplotlib 和官方 CPU 版 PyTorch。完整版本保存在 requirements/cpu-lock.txt，包之间的依赖检查通过。未修改全局 Python 环境、Git 代理或系统执行策略。

| 项目 | 实测版本或结果 |
| --- | --- |
| 解释器 | Python 3.11.5，项目 .venv/Scripts/python.exe |
| 数值与绘图 | NumPy 2.4.6、Matplotlib 3.11.2 |
| 自动微分 | PyTorch 2.15.0+cpu |
| 设备 | CPU，2个计算线程 |
| CUDA 构建 / 可用性 | null / false；本次使用 CPU wheel |
| pip 依赖检查 | No broken requirements found |

20:24的 environment-snapshot.json 保留首次调研时“未安装框架”的原始事实；它与本次新建项目环境有不同采样时间。其他可选框架、视觉/LLM依赖和加速后端仍需在对应章节准备与验证。

## 2. 真实执行与结果

从项目根目录执行：

```powershell
& ./.venv/Scripts/python.exe -B scripts/cpu-smoke.py --with-frameworks
& ./.venv/Scripts/python.exe -m pip check
& ./scripts/check-environment.ps1 -PythonPath ./.venv/Scripts/python.exe
```

标准库、NumPy、PyTorch 三个分支均为 passed，得到相同矩阵乘积：

$$
\begin{bmatrix}1&2\\3&4\end{bmatrix}
\begin{bmatrix}5&6\\7&8\end{bmatrix}
=\begin{bmatrix}19&22\\43&50\end{bmatrix}.
$$

训练数据为项目生成的5个确定性样本，目标为 $y=2x+1$。模型为 $wx+b$，共2个参数；全部样本用于训练，无独立评估集。固定零初始化，学习率0.1，200步全量梯度下降；PyTorch 使用 float64 和自动微分。

| 指标 | 本机结果 |
| --- | --- |
| 初始 / 最终 MSE | 9.0 / $1.9721522630525296\times10^{-32}$ |
| 权重 / 偏置 | 2.0 / 0.9999999999999998 |
| 标准库有限差分最大误差 | $1.2580017028085422\times10^{-9}$ |
| PyTorch 200步训练耗时 | 约0.0227秒 |
| PyTorch 导入及该分支总耗时 | 约1.746秒 |

耗时来自本次运行，会受系统负载与缓存影响；JSON分别记录纯训练和含导入的分支耗时。最终损失支持已知直线规则的数值链路可运行，无法提供泛化或大模型性能证据。

## 3. 资源与边界

21:27复测时系统可用物理内存约0.74 GiB，总物理内存约15.73 GiB。可用内存会变化，默认使用小数组、CPU和少量线程；正式训练前重查资源，重型实验按实测情况串行运行。没有测量进程峰值内存，系统可用内存不代表进程开销。

环境检查完整完成且 issues 为空。本轮修正了成功分支未显式返回0的问题，避免可选命令的退出状态遗留；修正后已实际重跑并取得成功退出状态。

锁文件已与实际安装、导入、运行和依赖检查对照；尚未在第二台机器复建此 Python 环境。文档工具链有独立锁文件，见[构建说明](../build-guide.md)。
