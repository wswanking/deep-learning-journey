# 第30章综合实践：微型文本与证据实验

这是本章独立CPU项目。完整说明见[第30章](../../docs/chapters/30-capstone.md)，固定配置见[config.json](config.json)。所有语料为本仓库原创合成教学数据，无需下载外部模型或数据。

## 运行

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe examples/30-capstone/run.py --mode rebuild
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python examples/30-capstone/run.py --mode rebuild
```

一次运行会从随机初始化训练35,328参数的字节模型600步，另执行237步保存后恢复至600步核验，再进行800步回答区间SFT；验证选点、固定评估、TF-IDF检索和规则引用、真实神经读者问答、SVG与独立Markdown/HTML/PDF报告均由实际结果生成。依赖使用仓库既有Python虚拟环境、Node依赖及Microsoft Edge；不会安装依赖。模型计算为CPU单线程float32，检索矩阵为float64。

```powershell
.\.venv\Scripts\python.exe examples/30-capstone/run.py --mode replay
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python examples/30-capstone/run.py --mode replay
```

replay先校验冷构建的源与产物清单，再加载本章自己的模型重做推理，核对全部36例语言任务原始字节、规则轨迹与神经证据问答相同。写入replay-results.json、replay-index.npz、replay-manifest.json、replay-artifact-verification.json；保留冷构建的模型、results.json、SVG及报告。源码或输入改变时先报错，应重新rebuild并记录新的实验，不能修改旧摘要后声称原样重现。

加`--skip-export`只跳过独立报告HTML/PDF，仍生成Markdown/SVG和证据清单。它仅影响rebuild；replay始终保留原报告。skip-export时目录若残留旧HTML/PDF，不属于本次出版，当前清单也不收录这些旧文件。加`--require-quality`使应用门槛未达标时退出码为2；默认工程核对全部通过返回0，具体质量另见quality_review，不能用退出码0代替质量达标。

配置是可读的固定教学协议；模型、训练和解码预算由程序完整核对，不能只改JSON便假定底层训练函数随之变化。review_thresholds的数值可改变应用审核要求，但当前是已知前章结果之后的回顾性示例要求，不是新盲测预注册。

## 输入与复用边界

- 第21章：原创48篇双语形状/颜色语料、分组划分、TinyDecoderLM、字节编解码和训练/恢复函数。
- 第22章：原创a..f两不同字母复制/反转数据、回答损失mask和SFT函数；使用本章随机初始化重训的基座，不读原22权重作起点。
- 第26章：原创12篇虚构设施文档、6验证题、16测试题、精确TF-IDF索引及受限规则读者。
- 第27章：24条既有测试与12条新字母诊断的固定记录、严格EM、格式/UTF8/停止和NLL评估。
- 28/29章视觉分支不在本项目默认输入中，没有声称本次训练视觉模型。

本章只读复用这些代码和原始数据。全部新模型、索引、训练日志、推理结果和报告落到`build/experiments/30-capstone/`。运行前后核对已有21/22权重及21/22/26/27报告指纹相同。重跑rebuild会更新30目录自己的实验产物；需要保留多个版本时，先另存整个30输出目录和对应源码版本。

## 产物

| 文件 | 含义 |
| --- | --- |
| base-model.pt | 本章独立预训练的推理检查点 |
| step-237-training.pt | 含优化器、步数、随机状态的恢复核验检查点 |
| sft/response_only.pt | 800步预算内按回答验证NLL选择的SFT检查点 |
| training-record.json | 曲线、数据划分、恢复数值和训练时间 |
| index.npz | 本章选择配置下的矩阵、idf和词表 |
| results.json | 全量预测、失败分类、指标与质量门槛 |
| report.md / report.html / report.pdf | 同一Markdown和6幅纯SVG生成的独立报告 |
| report-publication.json | 1440/390视口及PDF矢量结构检查 |
| manifest.json | 源文件和产物SHA256清单，排除自身循环摘要 |
| artifact-verification.json | 对清单实际逐文件复核的状态与数量 |

SHA256证明文件内容匹配；它不证明分组独立、语义独立或外部泛化。state_sha256只摘要参数，文件摘要还包含序列化和元数据。报告生成程序为[report_text.py](report_text.py)、[render-report.mjs](render-report.mjs)，可编辑纯矢量源为[figures.py](figures.py)。

## 实测结论

2026-11-17独立冷构建：预训练终点验证CE=0.9473926544 nat/byte；600步主路线与237+363步恢复路线最终参数、历史和采样状态相同；SFT最低回答验证NLL在250步。SFT既有24例严格正确6例、新字母12例0例；始终复制基线分别12例和6例。规则12个可回答问题与4个无证据问题全对；基座及SFT各自12道问题，无证据/完整证据均EM=0。

这些结果支持工程流程可运行、受限规则可靠取回当前小资料的事实；不支持微型模型理解一般指令或证据。五个示例应用门槛三项未满足，总结为`quality_not_met`。完整运行耗时、文件指纹与出版验收以实际JSON及第30章最终记录为准。

## 最终实现与验收记录（2026-11-17）

- 冷构建汇总时间：2026-11-17T19:35:11.538351+08:00；22项工程核对通过。数值流程18.9685419秒，其中主预训练5.1176781秒、独立恢复核验4.7674888秒、SFT7.6227307秒。计时从NumPy/PyTorch导入及线程配置之后开始，包含辅助模块导入、检查、训练、评估和检查点I/O，不含SVG/报告生成及导出；不是完整进程壁钟基准。
- 独立重放与质量门槛汇总时间：2026-11-17T19:37:50.332103+08:00；24项核对通过，重放数值流程0.5843162秒。36例语言输出的原始字节、规则检索轨迹和两种神经读者的证据问答全部与冷构建一致；冷构建模型、结果、图和报告未变。`--require-quality`实际返回退出码2，默认重放返回0，两者均保留`quality_not_met`。
- 冷构建清单33个源文件、16个产物逐项SHA256复核相符；重放清单33个源文件、6个产物复核相符。6项既有章节保护文件均未改变。清单与核验记录排除自身，避免循环摘要。
- 章节导出14页、27个数学表达式、6幅纯SVG。1440/390像素视口没有横向页面溢出、公式错误、重复ID、失效锚点、未渲染粗体、裁切或外部请求；PDF独立审计0位图、12种嵌入字体、32个链接（12个内部链接），全部14页及6幅图实际目视通过。
- 独立项目报告7页、6幅同源SVG，1440/390视口通过；PDF独立审计0位图、4种嵌入字体、11个链接（8个内部链接），全部7页实际目视通过。报告PDF SHA256如下；完整出版摘要另在report-publication.json，独立字体文本审计在report.audit.json。

```text
cc42bdcb4e7f3b6e6f997208b9f07175ee05cf2ae4c5f064c926c2211d16e5dd
```

- 六幅SVG均为900宽viewBox，没有image、filter或foreignObject；曲线有更新步、nat/byte单位与实线/虚线区分，保留可编辑生成器。
- 交叉审查修复了可调整门槛时质量图末句硬编码的问题，现按quality_review生成；额外将“全部门槛通过”的构造记录渲染到独立QA目录，核对图文显示quality_met。跳过导出的报告措辞分支亦实际核对，不把旧HTML/PDF当作本次验收。
- 质量失败与边界：SFT既有任务6/24、新字母0/12；基座/SFT神经证据问答均0/12；规则小资料回答/拒答满分。三个神经应用门槛未通过，不支持开放问答、自然照片或真实人群泛化。未测峰值进程内存、稳态吞吐及多轮延迟分位数。
- 源码已统一UTF8无BOM、LF、无末尾空白并保留单一末尾换行；生成JSON、SVG和报告Markdown直接使用LF。源码、配置、字节数据、依赖锁、出版配置或所记录CSS改变后，旧清单会如实拒绝重放，应重新建立实验记录。
