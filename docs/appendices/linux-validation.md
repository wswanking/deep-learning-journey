# Linux 构建验收记录

日期：2026-11-17。基线：`ab053fd`。本次未改章节正文与 Windows PowerShell 脚本。

## 本地云端 Linux

Debian 13、Node.js 24.22.1、npm 11.9.0、Python 3.12.14；不是用户 Windows/WSL 本机。

- `bash scripts/setup.sh`：锁文件安装成功（20个包），没有安装全局软件。
- Linux 专项12项测试通过，包括 Bash 语法/LF、错误参数、从其他工作目录调用、浏览器发现/显式选择、环境 JSON 防覆盖、已导出 PYTHON 时仍验收新 venv。
- 完整测试曾有2项浏览器 DOM 测试受当前执行环境限制：Chromium 的本地 socket 创建返回 `Operation not permitted`。审批后仍复现；不当作通过，不移除测试或降低门禁。
- Markdown 源文件81份检查零错误；新增文档单独检查。
- 样张 HTML：3幅 SVG、20处公式；全31章 HTML：154幅 SVG、3337处公式，构建通过。
- 环境 JSON 与标准库 CPU smoke 通过（200步回归、梯度有限差分）；历史 Windows 环境报告保持不变。
- 本地 PDF、页面自动验收受浏览器权限限制，未通过。本地未安装可选 CPU 框架。

## GitHub Actions

新增 `Linux publication acceptance`，在 main 推送、PR 或手动触发时，准备 Node 24、Python 3.12、Chromium 和 Noto CJK 字体，运行同一 `verify.sh`；随后运行 `setup.sh --cpu` 验证项目独立 CPU 环境。CI 独立安装浏览器仅限 runner，用户本地脚本不自动下载浏览器。

代码提交 `f3c7b1285d4c7cb2287b8a2dfe5448d2ac168eb8` 的 [Actions 运行 44099300523](https://github.com/wswanking/deep-learning-journey/actions/runs/44099300523) 已实测成功（2026-11-17）：

- Ubuntu 26.04.1 runner，Node 24，Python 3.12.16，独立 Chromium 与 Noto CJK 字体。
- 73/73项回归通过；83份 Markdown 源格式检查零错误。
- 样张5页 PDF、全31章422页 PDF，均零位图对象，书签和结构通过。
- 样张及全书1440/390px检查通过：无页面水平溢出、断锚、公式错误、SVG裁切、意外外部请求。
- `PYTHON=python3 bash scripts/setup.sh --cpu` 在独立 `.venv-linux` 安装并验收成功；NumPy 2.5.5、Matplotlib 3.11.2、PyTorch 2.15.0+cpu，pip check及baseline/NumPy/PyTorch三个分支通过。
- [运行产物](https://github.com/wswanking/deep-learning-journey/actions/runs/44099300523/artifacts/14335197222)包含页面检查、PDF结构报告与实际Linux依赖freeze；下载受GitHub访问权限和保留期限约束。

该远端实测补齐了本地因权限受阻的浏览器验收，不改变前述本地限制。后续提交以各自 Actions 结果为准。未实测用户 WSL、Windows 原生、GPU、其他架构；未逐章重跑所有训练实验，也未重新检查 GitHub 远端31章公式。不能把自动 PDF 验收等同于全书逐页目视复核。
