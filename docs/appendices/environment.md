# Windows 本机环境调研

> 本报告保留首次调研时点；后续项目.venv安装与实际CPU验证见[框架复测记录](framework-validation.md)。

> 任务：INF-01；状态：已完成调研与标准库 CPU 最小验证。
> 实测时间：2026-11-15 20:24（Asia/Shanghai）。
> 原始记录：[environment-snapshot.json](environment-snapshot.json)。
> 正式仓库：`D:\code\workspace\deep-learning-journey`。

调研结论：当前适合以 **CPU、小数据、小模型** 开始教学。Python 可用，标准库数值计算和微型回归训练已验证；选定 Python 环境尚未安装 NumPy、PyTorch 等训练依赖，不能据此声称 PyTorch 训练环境已就绪。安装、框架验证和依赖锁定属于后续环境准备工作，不与本次调研结果混淆。

## 1. 硬件实测

容量使用 GiB（$1\,\mathrm{GiB}=2^{30}$ 字节），原始字节数保存在 JSON 中。可用内存和磁盘空间是采样值，会随运行状态变化。

| 项目 | 实测结果 | 对教程的影响 |
| --- | --- | --- |
| 系统 | Windows 11 专业版，64 位，10.0.26200 / Build 26200 | 使用 Windows PowerShell/Python 路径 |
| CPU | Intel Core i9-13900H，14 核、20 逻辑处理器 | 基础数值实验和小网络以 CPU 为默认设备 |
| 物理内存 | 16 GiB；系统可见约 15.73 GiB | 控制模型、批次、上下文和数据缓存 |
| 可用内存 | 正式采样约 1.16 GiB；此前采样约 0.70～0.74 GiB | 当前先跑极小实验；扩大实验前重新检查 |
| 显卡 | Intel Iris Xe Graphics | 不将本机视作 NVIDIA CUDA 训练设备 |
| 显卡驱动 | 31.0.101.4502 | 仅记录已读到的版本，未开展驱动更新或其他加速后端验证 |
| 显存字段 | CIM 返回 AdapterRAM = 1 GiB | 该字段不能代表集成显卡实际可用显存，不据此制定 GPU 预算 |
| C 盘 | 可用约 37.73 GiB | 避免把大数据、权重和下载缓存默认堆在系统盘 |
| D 盘 | 可用约 562.76 GiB | 教程、后续数据和模型缓存优先按项目规划放在 D 盘 |

未在 CIM 中发现 NVIDIA 显卡；`nvidia-smi` 在 PATH 及两个标准位置均未找到，`nvcc` 未在 PATH 找到，`CUDA_PATH` 未设置。本次未安装 CUDA。由于 PyTorch 尚未安装，`torch.cuda.is_available()` **未执行**，原始记录为 null，不能解释成该函数已返回 false。

Intel 的其他加速路线（例如 XPU/DirectML）未验证，不纳入默认实验方案。本次没有全盘搜索全部软件安装目录；“未在 PATH 找到”和“全机未安装”严格区分。

## 2. Python、包管理器与开发工具

| 项目 | 实测状态 |
| --- | --- |
| Python | CPython 3.11.5，64 位，可执行与导入标准库 |
| 解释器路径 | `C:\Users\22515\AppData\Local\Programs\Python\Python311\python.exe` |
| Python Launcher | `py -0p` 列出上述 Python 3.11 |
| pip | 23.2.1，对应上述 Python |
| venv | 标准库模块可导入；项目 `.venv` 尚不存在，未创建环境 |
| NumPy / PyTorch / torchvision | 此解释器环境中均未安装 |
| Matplotlib / Jupyter / psutil | 此解释器环境中均未安装 |
| Transformers / Datasets / Accelerate | 此解释器环境中均未安装 |
| uv / conda | 未在 PATH 发现；uv 用户常见位置和 Conda 几个常见位置亦未发现 |
| Git | 2.45.1.windows.1 |
| Node.js / npm | v24.19.0 / 12.0.2 |
| PowerShell | 本次脚本在 7.6.9 实测通过 |

`python3` 在 PATH 中指向 WindowsApps 的执行别名，不将该别名当作独立可用解释器。本次查询和验证使用真实的 `python.exe`，并通过 `python -m pip` 绑定相同解释器，避免装到其他 Python 环境。

包状态只对应选定的解释器，没有推断其他未知环境的依赖。包管理器和文档工具的 PATH 检查也不等于全机安装清单。

## 3. 下载与网络

Windows 当时启用了 `127.0.0.1:7890` 本地代理。未发现 HTTP_PROXY、HTTPS_PROXY、ALL_PROXY 环境变量，Git 未配置匹配此地址的 HTTP 代理。本次仅在 HTTP 检查中显式使用该代理，未修改系统或 Git 全局设置。

正式记录的 HTTPS HEAD 探测结果如下，每次请求最多 12 秒：

| 入口 | 直连 | 经现有本地代理 |
| --- | --- | --- |
| GitHub 仓库页面 | HTTP 200 | HTTP 200 |
| PyPI 的 NumPy 索引 | HTTP 200 | HTTP 200 |
| PyTorch CPU wheel 索引 | HTTP 200 | HTTP 200 |
| Hugging Face 首页 | 12 秒内未完成 | HTTP 200 |

此前同轮检查中，GitHub 直连也曾超时；这与早先克隆发生重置/超时一致，说明直连结果会变化。不能把一次 HTTP 200 解释为后续 Git 推送、全部依赖或大模型下载均已验证。

本次只验证所列公开入口可达，没有安装包、下载数据/权重、进行账户认证或验证 Git 推送权限。后续若某个模型仓库需授权，应单独检查。

## 4. CPU 最小验证与实际结果

运行入口：[cpu-smoke.py](../../scripts/cpu-smoke.py)。本次使用 Python 标准库，不需要第三方包、模型、数据下载或 GPU。

验证包括：

1. 计算两个 $2\times2$ 矩阵的乘积，得到 $[[19,22],[43,50]]$。
2. 用中心有限差分对照手写的线性回归梯度，最大绝对误差约 $1.26\times10^{-9}$，小于 $10^{-6}$ 验收阈值。
3. 在 5 个确定性样本上对 $y=2x+1$ 进行 200 步梯度下降，学习率为 0.1。
4. 检查结果有限且最终均方误差小于 $10^{-12}$。

| 指标 | 本机实测 |
| --- | --- |
| 初始 MSE | 9.0 |
| 最终 MSE | 约 $1.97\times10^{-32}$ |
| 学到的权重 | 2.0 |
| 学到的偏置 | 0.9999999999999998 |
| 标准库 CPU 验证 | 通过 |
| NumPy / PyTorch 计算验证 | 未运行：对应包尚未安装 |

脚本整体的 `passed` 表示已执行的基础检查通过；可选依赖的 `not_installed` 仍保留在各自字段，不能被整体状态覆盖。本次极小示例的耗时不是训练性能基准，不能用于预测 LLM 训练时间。

安装依赖后，可使用同一脚本的 `--with-frameworks` 实际验证 NumPy 矩阵乘法及 PyTorch CPU 自动微分训练；导入/计算失败会返回失败状态和非零退出码。本次没有把这条未来验证计为已完成。

## 5. 教程实验的初始规模

下表是依据实际资源制定的**起步配置建议**，不是已完成的训练或性能承诺。正式实验前重新读取可用内存，按逐章实测结果调整。

| 阶段 | 起步配置 | 当前安排 |
| --- | --- | --- |
| 数学与线性模型 | 2×2 矩阵、5～1000 个合成样本 | 标准库最小示例已通过；逐章再引入 NumPy |
| MLP | 1～2 个隐藏层，32～64 隐藏单元，batch 8～16 | 安装 CPU PyTorch 后先从最小配置验证 |
| CNN | 28×28 或 32×32 小图，少量通道，batch 4～8 | 不从大分辨率、大数据集开始 |
| RNN / 注意力 | 序列长 16～32，隐藏维度 32～64，batch 1～4 | 先人工/合成小例子，再小数据训练 |
| 微型语言模型 | 1～2 层，宽度 32～64，上下文 32～64，batch 1～2，目标参数量不超过约 100 万 | 参数量还须按实际词表计算；CPU 路径，先短运行验证 |
| VAE / GAN / 扩散 | 二维合成数据或微小图像与网络 | 先理解损失和训练流程，逐项记录实际成本 |
| SFT / LoRA / 多模态 | 玩具模型或小型冻结模块上的局部训练 | 预训练模型、量化、其他加速后端均需另行核验 |

训练初始建议 `num_workers=0`、CPU 线程约 2～4；不要按 20 个逻辑处理器直接分配大量数据加载进程。以上是减少并发和内存占用的初始策略，实际最佳值尚未基准测试。

当前可用内存只有约 1 GiB，扩展至框架训练前应先复测。建议基础实验启动时争取至少约 3 GiB 可用内存，这只是保守的操作目标，不能替代实际内存监测。无须安装 CUDA 来完成 CPU 路线。大模型全量训练、QLoRA、复杂视觉语言模型训练不作为当前本机的默认可运行项。

## 6. 初始依赖方案与构建准备

完整方案见 [requirements/README.md](../../requirements/README.md)：

- 调研与标准库 CPU 最小验证：仅 Python 标准库与 Windows PowerShell/CIM，无第三方 Python 依赖。
- 基础教程：拟采用独立 `.venv`，先装 NumPy、Matplotlib 和 CPU 版 PyTorch；用脚本验证后再锁定实测版本。
- 视觉和 LLM：按章节需要再引入 torchvision、Transformers 等，避免一开始安装完整大模型工具栈。
- 依赖候选文件不是锁文件；本次未安装依赖，也未声称候选版本组合已通过验证。

文档构建相关实测：中文字体微软雅黑、黑体、宋体的字体文件存在；Pandoc、Quarto、XeLaTeX、Tectonic 和 Graphviz 的 `dot` 未在 PATH 发现。HTML/PDF 工具链尚未建立，字体存在也不代表公式、矢量图或分页已验证。该工作归 INF-02。

## 7. 重跑方法

在仓库根目录的 PowerShell 中：

```powershell
# 最小 CPU 数值计算，不安装第三方依赖。
python -B scripts/cpu-smoke.py

# 同时检查已经安装的 NumPy/PyTorch；缺失包单独标记。
python -B scripts/cpu-smoke.py --with-frameworks

# 硬件、依赖、CPU 检查，以及直连/现有 Windows 代理的网络探测。
& ./scripts/check-environment.ps1 -CheckNetwork

# 如需指定解释器：
& ./scripts/check-environment.ps1 -PythonPath 'C:\Users\22515\AppData\Local\Programs\Python\Python311\python.exe'

# 生成一份新的本地报告；build/ 已被忽略，不混入教程源文件。
& ./scripts/check-environment.ps1 -CheckNetwork -OutputPath build/environment/local-report.json
```

脚本默认优先选用项目 `.venv`，其次选 PATH 中真实的 Python；不会执行 WindowsApps 的商店别名。只有带 `-CheckNetwork` 才发出网络请求。可用 `-ProxyUrl http://127.0.0.1:7890` 明确指定本次代理，不写入全局配置。

默认拒绝覆盖已有输出报告，确需更新时显式传 `-Force`。CIM 或 Python 检查失败时生成 `partial` 状态并以非零码退出，不把查询失败的资源写成零。缺少可选依赖、文档工具或 NVIDIA 工具则如实记录。网络状态表示入口可达性，失败入口不等于整台机器不可用。

## 8. 验收与后续工作

| INF-01 要求 | 本轮证据 | 状态 |
| --- | --- | --- |
| 核对实际硬件与资源 | JSON 中 OS、CPU、内存、GPU/驱动、磁盘字段 | 已完成 |
| 核对 CUDA、Python、包管理器及框架状态 | NVIDIA 工具、CUDA_PATH、解释器和包清单 | 已完成；明确缺失和未验证项 |
| 检查必要网络访问 | 4 个公开入口、直连/代理两条路径 | 已完成 |
| 提供环境记录与重跑入口 | 本附录、原始 JSON、两个脚本 | 已交付并实测 |
| 至少完成 CPU 最小示例 | 矩阵、有限差分、200 步线性回归 | 已通过，使用标准库 |
| 提供初始依赖与实验规模方案 | 第5～6节及 requirements 文件 | 已完成规划 |

INF-01 的“已完成”指**调研任务**完成。PyTorch 安装/验证、性能基准、教程第00章、INF-02 导出和 INF-03 公共术语/符号仍待完成。

本次仅新增/更新本地项目文件，没有修改系统配置、安装包、创建虚拟环境或下载模型。下一步是 INF-02 的构建工具链方案与样张验证；实际章节实验开始前应先按依赖方案建立并验证 CPU 训练环境。

## 9. 官方参考

1. [Python 3.11 venv 文档](https://docs.python.org/3.11/library/venv.html)：独立环境、Windows 路径和直接调用虚拟环境解释器的方法。
2. [PyTorch 本机安装说明](https://pytorch.org/get-started/locally/)：Windows 与 CPU 安装路线；具体版本以实施安装时的官方选择器和实际 wheel 为准。
3. [NumPy 安装说明](https://numpy.org/install/)：pip/虚拟环境安装与导入验证。
