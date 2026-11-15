# 初始依赖方案：Windows / CPU

> INF-01 交付；2026-11-15。这里区分已验证的基础环境与尚未安装的候选训练依赖。
> 详细实测见 [本机环境报告](../docs/appendices/environment.md)。

## 1. 已验证的最小环境

Python 3.11.5（64 位）、pip 23.2.1、PowerShell 7.6.9。环境调研和标准库 CPU 示例不需要第三方 Python 包：

```powershell
python -B scripts/cpu-smoke.py
& ./scripts/check-environment.ps1
```

两个脚本本机执行通过。当前选择的 Python 尚无 NumPy、Matplotlib、PyTorch，项目虚拟环境尚未创建。

## 2. 计划采用的基础训练环境

- 用现有 Python 3.11 创建独立项目 `.venv`。
- 数值与矢量绘图：`cpu.in` 中的 NumPy、Matplotlib。
- 自动微分与训练：`torch-cpu.in` 中的 CPU 版 PyTorch，使用官方 CPU wheel 索引。
- 优先使用 Windows 64 位预编译 wheel，不为教学基础环境自行编译 CUDA 或 PyTorch。
- 图像相关依赖和 LLM 依赖等到相关章节再添加，并核对版本兼容性。

这两份 `.in` 是候选包清单，**不是锁文件，也未在本机安装验证**。版本在实际安装、CPU 验证通过后记录锁定，不能把未执行的安装命令写成成功记录。

以下是后续准备环境时使用的命令，本次没有执行：

```powershell
py -3.11 -m venv .venv

# 直接调用环境解释器，不要求修改系统执行策略或激活脚本。
& ./.venv/Scripts/python.exe -m pip install --upgrade pip
& ./.venv/Scripts/python.exe -m pip install -r requirements/cpu.in
& ./.venv/Scripts/python.exe -m pip install -r requirements/torch-cpu.in

# 安装后的验收：
& ./.venv/Scripts/python.exe -B scripts/cpu-smoke.py --with-frameworks
& ./scripts/check-environment.ps1 -PythonPath ./.venv/Scripts/python.exe

# 仅在实际验证通过后生成实测依赖记录：
& ./.venv/Scripts/python.exe -m pip freeze |
    Set-Content -LiteralPath requirements/cpu-lock.txt -Encoding utf8
```

如果安装命令的网络请求失败，应重新核对现有代理，并只为该次 pip 调用显式传入代理参数。不要将本机某次代理地址硬写进共享依赖文件，不修改全局代理或降低 TLS 校验。

后续锁定清单应说明 Python 版本、Windows/CPU 平台和 PyTorch 官方 CPU 索引；单独的版本列表不能代替安装来源说明。

## 3. 暂不预装的依赖与待验证项

| 范围 | 后续安排 |
| --- | --- |
| torchvision | CNN 数据与视觉实验需要时添加，先核对与 torch 的匹配关系 |
| transformers / datasets / accelerate | LLM 章节按模型和内存预算添加，并做实际导入/运行验证 |
| Jupyter | 可选，教程默认保留可独立运行的脚本入口 |
| CUDA / NVIDIA GPU 路线 | 当前硬件未发现 NVIDIA 设备，不作为默认基础环境 |
| XPU / DirectML / 量化后端 | 未验证，不承诺本机可用 |
| HTML/PDF 工具链 | INF-02 已完成：项目本地 npm 依赖，见下一节 |

本次只验证标准库 CPU 路线。表中的候选依赖和安装流程均不代表框架环境已经可运行。

## 4. 已验证的文档构建依赖

文档工具链独立于 Python 训练环境，使用 Node.js 22.13+、已有 Edge/Chrome 和项目本地 npm 包。实际版本由根目录 package.json、package-lock.json 锁定；本机 Node.js 24.19.0、npm 12.0.2、Edge 156.0.4314.42 已验证。

```powershell
npm ci --ignore-scripts
& ./scripts/build.ps1 -Format all -Sample
npm test
```

锁文件已在独立空目录重建通过，13项检查通过。文档依赖的安装不代表 NumPy/PyTorch 已安装。详见[构建说明](../docs/build-guide.md)和[验收记录](../docs/appendices/build-validation.md)。

## 5. 官方依据

- [Python venv](https://docs.python.org/3.11/library/venv.html)
- [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)
- [NumPy 官方安装说明](https://numpy.org/install/)
