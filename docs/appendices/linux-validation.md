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

本记录初次提交时远端运行尚未完成；应以对应提交的 Actions 结果为准。工作流产物包含页面检查和 PDF 结构报告、实际 Linux 依赖 freeze。未实测用户 WSL、Windows 原生、GPU、其他架构；未逐章重跑所有训练实验，也未重新检查 GitHub 远端31章公式。不能把自动 PDF 验收等同于全书逐页目视复核。
