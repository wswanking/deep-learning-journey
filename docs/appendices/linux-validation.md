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


## 第二轮：全工程脚本与实验实跑

用户进一步要求扫描全部工程，不限构建入口。本节补充并更新上一轮的“未逐章重跑”范围；原始Windows证据、源SVG不覆盖。

- 清点97个仓库脚本：30章节入口、37辅助模块、9静态图生成器、15工具/共享模块、4测试文件、2原生PowerShell脚本。95个通过语法检查，2个PS1保留但未在Linux原生执行；静态扫描本身不等于全面兼容性证明。
- README主构建区及全部30章实践README补齐Bash示例；修复30章硬编码Edge、02/04章Linux峰值RSS、11/14/20图生成工作目录依赖。
- 第一轮全实验CI真实发现24章DPO训练损失未下降，没有跳过或放宽断言。隔离复现后，统一将学习率从0.0005降为0.0001，并用完整36对训练批次替代每步抽12对，仍固定300步；三种配置的原22项检查全部通过。协议调整只参考训练损失，没有依据测试集选配置。正文已明确历史数值/SVG属于原Windows协议，新结果读取本次JSON。
- 独立复核还发现24章图生成页脚硬编码历史指标，已改为通用解释；表格继续从当前JSON读取。新增不同指标输入的回归测试，Linux专项现15项本地通过。不会用新的Linux曲线覆盖已提交历史SVG。
- [代码d55a7cb的完整Actions 44104896665](https://github.com/wswanking/deep-learning-journey/actions/runs/44104896665)已成功：75项当时的回归、84份Markdown检查、样张5页与全书422页PDF、1440/390px检查通过；独立CPU环境安装及pip check通过。
- 该运行在隔离源码副本完成55个步骤：脚本清单、9生成器、00章CPU smoke、01～30章全部实验、10个独立figures入口、综合实践重放与严格质量门槛、重生成全书HTML/PDF及页面验收。辅助模块通过章节实际调用，不把空跑当覆盖。
- 综合报告7页PDF、6幅SVG、零位图，1440/390px无溢出或裁切；重放工程检查24项通过。模型质量仍是`quality_not_met`，严格模式按预期返回2，并非业务质量达标。
- [成功运行产物](https://github.com/wswanking/deep-learning-journey/actions/runs/44104896665/artifacts/14338622907)保留55步清单、独立日志、各章JSON与出版检查，不上传模型权重。后续图注修复及记录提交仍由相同CI重新验证，查看对应提交状态，不把旧提交成功冒充新提交成功。

本地还在隔离临时CPU环境复现并验证21/22/24～30章训练（30章本地不导出）；浏览器导出证据来自上述Actions。仍未验证用户Windows/WSL原生、GPU、其他CPU架构、在线GitHub公式页及全书逐页目视效果。Windows PowerShell脚本没有删除或改写；Linux资源检测不实现Windows注册表/CIM或其网络代理诊断选项。


## 2026-11-20：第22章padding精度复核

README三条Bash命令修正提交的[CI 44679792523](https://github.com/wswanking/deep-learning-journey/actions/runs/44679792523)通过出版和CPU环境检查，但在第22章`padding changed real-token outputs`失败，不能称整套验收通过。

本地用同版PyTorch2.15.0+cpu、NumPy2.5.5从21章重新训练基座，复现8个样本中1个超出旧`atol=2e-6, rtol=1e-5`逐元素阈值，float32最大绝对差5.1856041e-6，阈值比最大1.4836。1/2/4线程结果相同；同权重float64差仅4.4408921e-15。这是批量/单条形状造成的浮点舍入证据，不是通过增加线程得到的随机修复。

修复仅调整验收设计，不改变模型、mask、训练精度或预算：float32绝对误差须小于1e-5，同时增加float64小于1e-11的严格语义检查、同形状padding内容扰动逐位不变检查；记录误差。真实训练基座正例通过，移除全部因果及padding attention约束、仅float32批量注入1e-3偏差的两个负例均被拒绝。第22章完整默认800步本地通过。新测试加入全实验CI，完整runner从55步增加为56步；本段为已完成本地验证，远端结果仍须查看对应提交的Actions，不能沿用历史成功。
