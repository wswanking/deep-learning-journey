# Windows / CPU 训练依赖

> 实际安装和验证日期：2026-11-15。默认使用项目独立环境。
> [框架实测记录](../docs/appendices/framework-validation.md) · [原始结果](../docs/appendices/framework-validation.json) · [历史硬件调研](../docs/appendices/environment.md)

## 1. 已验证环境

Windows 11 64位、Python 3.11.5、NumPy 2.4.6、Matplotlib 3.11.2、PyTorch 2.15.0+cpu。CPU数组运算、float64自动微分和200步小回归训练实际通过，pip check无依赖冲突。项目环境位于仓库 .venv，不需要激活脚本或修改执行策略。

requirements/cpu-lock.txt 是本次成功安装后的完整版本记录，含 pip/setuptools；cpu.in 与 torch-cpu.in 保留顶层依赖意图。初次调研时框架未安装的历史JSON继续保留。

## 2. 新建或重建环境

从仓库根目录执行；使用专供本项目的环境：

```powershell
py -3.11 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/cpu-lock.txt
```

默认包索引提供普通依赖，PyTorch官方CPU索引提供带 +cpu 标识的wheel。锁定范围是Python 3.11 / Windows x64 / CPU；其他系统、Python版本或加速设备需另行核对wheel。锁文件已与实际环境及pip check对照，尚未在第二台机器复建。

开发阶段重新选择版本时，可使用原始顶层清单；重选后必须重新验收和更新锁文件：

```powershell
& ./.venv/Scripts/python.exe -m pip install --only-binary=:all: -r requirements/cpu.in
& ./.venv/Scripts/python.exe -m pip install --only-binary=:all: -r requirements/torch-cpu.in
```

下载失败时先核对网络配置，必要时仅为该次调用传入代理；代理地址不写入共享文件，不修改全局代理或降低TLS校验。

## 3. 验收方法

```powershell
& ./.venv/Scripts/python.exe -m pip check
& ./.venv/Scripts/python.exe -B scripts/cpu-smoke.py --with-frameworks
& ./scripts/check-environment.ps1 -PythonPath ./.venv/Scripts/python.exe
```

检查JSON中的baseline、numpy、torch三个分支分别为passed；缺少可选包时该分支记为not_installed，不能仅凭总状态判断框架就绪。本次CPU wheel的CUDA接口不可用。

标准库基线无需第三方包，可单独执行：

```powershell
py -3.11 -B scripts/cpu-smoke.py
```

依赖版本变更并完成验证后再更新记录：

```powershell
& ./.venv/Scripts/python.exe -m pip freeze --all | Set-Content -LiteralPath requirements/cpu-lock.txt -Encoding utf8
```

## 4. 后续按章节准备的依赖

| 范围 | 安排 |
| --- | --- |
| torchvision | 视觉实验需要时添加，核对与torch匹配关系 |
| transformers / datasets / accelerate | LLM章节按模型和内存预算添加，实际导入和运行 |
| Jupyter | 可选；教程保留独立脚本入口 |
| CUDA / XPU / DirectML / 量化后端 | 按硬件和框架实测验证，当前未作为默认路线 |
| 大型数据、权重与缓存 | 放入明确的本机输出目录，避免加入Git |

本机硬件报告仅发现Intel Iris Xe；其他加速后端尚未验证。扩大实践规模前复查内存。

## 5. 文档构建依赖

文档构建使用本地Node依赖、已有Edge/Chrome，锁定于package.json和package-lock.json：

```powershell
npm ci --ignore-scripts
& ./scripts/build.ps1 -Format all -Sample
npm test
```

它与Python训练环境分别安装；单章及全书导出见[构建说明](../docs/build-guide.md)。

## 6. 官方依据

- [Python 3.11 venv](https://docs.python.org/3.11/library/venv.html)
- [PyTorch安装与CPU平台选择](https://pytorch.org/get-started/locally/)
- [NumPy安装](https://numpy.org/install/)
