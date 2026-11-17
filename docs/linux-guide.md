# Linux / Windows WSL 构建与验收

本项目保留 Windows PowerShell 入口，同时支持 Linux Bash。Windows 中的 Linux 环境使用 WSL2；以下命令在 **Linux/WSL 终端**执行，不是在 PowerShell 中执行。无需 GPU，不自动修改系统配置、安装系统软件或下载浏览器。

## 1. 环境准备

- Node.js 22.13+ 和 npm，文档依赖使用现有 `package-lock.json`。
- Python 3.11+；标准库实验不需要额外依赖，训练环境需要 `venv` / `pip`。
- PDF 与页面验收需要已有 Chrome、Edge 或 Chromium，以及中文字体，例如 Noto Sans CJK。
- HTML 构建不需要浏览器。缺浏览器或字体时，请先用所选发行版的正式安装方式准备，再执行 PDF 构建。

推荐将项目克隆到 WSL 的 Linux 文件系统（如 `~/code/deep-learning-journey`）。也可从 `/mnt/d/code/workspace/deep-learning-journey` 使用已有源码，但 Windows 与 Linux **不能共用 Python 虚拟环境**；脚本使用 `.venv-linux`，不覆盖 Windows `.venv`。若共享 `node_modules`，切换系统后重新运行 setup。Windows 和 WSL 是不同运行时，WSL 检查结果不证明 Windows 原生环境也通过。

```bash
bash scripts/setup.sh
bash scripts/check-environment.sh
bash scripts/build.sh --format all --sample
bash scripts/build.sh --format all --chapter 00
bash scripts/build.sh --format all
```

构建、环境与验收脚本从任意当前目录调用均可；路径含空格时须加引号。`.sh` 固定 LF 换行，既可 `bash scripts/build.sh`，也可 `./scripts/build.sh`。

## 2. 浏览器与字体

所有发布检查共用同一发现逻辑：显式 `--browser` 优先，其次 `BROWSER_EXECUTABLE`，然后已有 Edge / Chrome，最后 Linux PATH 中的 Chromium / Chrome / Edge。显式指定无效时立即报错，不偷偷切换。

```bash
export BROWSER_EXECUTABLE=/usr/bin/chromium
bash scripts/build.sh --format all --sample
node scripts/check-publication.mjs --name typesetting-sample --capture
# 单次覆盖也适用于 build、check-publication、check-github-math
bash scripts/build.sh --format pdf --chapter 00 --browser /usr/bin/chromium
fc-list :lang=zh family
```

字体列表应有中文字体。PDF 会嵌入实际使用的字体，缺字或分页仍需目视检查，自动矢量检查不等于中文字体验收。不同字体/浏览器版本可能导致页数不同。

## 3. 可选 CPU 训练环境

只导出教程时跳过本节。此命令联网安装项目本地依赖，不改系统 Python；下载可能较大，先确认内存和磁盘。

```bash
PYTHON=python3 bash scripts/setup.sh --cpu
bash scripts/check-environment.sh --with-frameworks --require-frameworks
.venv-linux/bin/python -m pip check
.venv-linux/bin/python -B scripts/cpu-smoke.py --with-frameworks
```

Linux 使用 `requirements/cpu.in` 和官方 CPU 索引的 `requirements/torch-cpu.in` 解析可用 wheel，**不套用 Windows 专用 cpu-lock.txt**。这不是跨平台固定版本锁；安装后实际版本保存到 `build/environment/linux-freeze.txt`。Python/架构与 wheel 不兼容时安装失败，不回退 CUDA 或源码编译。正式复现应保存本机 freeze 与验收记录，未验证平台不得宣称通过。

环境检查默认仅输出 JSON，不联网、不训练、不写文件。`--with-frameworks` 加入已有 CPU smoke；`--require-frameworks` 要求 baseline / NumPy / PyTorch 全通过，缺包不得当作框架就绪。`--output path.json` 仅创建新文件，拒绝覆盖。容器 `/proc` 可能反映宿主机内存，应另外核对容器限制；历史采样不代表实时可用资源。

## 4. 一键验收

```bash
bash scripts/verify.sh
```

按顺序执行回归测试、Markdown 检查、环境采样、标准库 CPU 实验、样张及全31章 HTML/PDF 构建和 1440/390px 页面检查；任一步失败即返回非零退出码。不自动安装依赖，不自动联网检查 GitHub。

需要核对远端 GitHub 公式时另行执行（需要网络）：

```bash
node scripts/check-github-math.mjs --ref main --chapter 01 --browser /usr/bin/chromium
```

输出仍在 `build/html`、`build/pdf`、`build/qa`，不加入 Git。检查报告是自动验收证据，新内容仍需目视复核中文、公式和分页。[本次 Linux 验收范围](appendices/linux-validation.md)区分通过项和未测项。
