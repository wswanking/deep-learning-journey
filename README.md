# deep-learning-journey

深入学习-LLM-多模态教程项目。以《Deep Learning》（花书）为基础，循序渐进学习数学基础、神经网络、典型模型、LLM 和多模态，并结合 Windows 本机开展小模型训练实践。

## 项目入口

- [项目上下文与任务约定](AGENTS.md)
- [章节计划与待完成需求](docs/chapter-plan.md)
- [教程写作、数学、图文与排版规范](docs/authoring-guide.md)
- [章节模板](docs/templates/chapter-template.md)
- [本机环境调研报告](docs/appendices/environment.md)
- [HTML/PDF 构建说明](docs/build-guide.md)
- [构建验收记录](docs/appendices/build-validation.md)
- [初始训练依赖方案](requirements/README.md)

## 教程要求

核心原理配图文并注明经典教材、大学课程或原始论文来源；所有配图使用真正的矢量图（SVG，PDF 导出保留矢量），保证放大和打印时的清晰度；数学符号规范，推导与概念解释完整；按前置知识逐步推进。章节以 Markdown 为内容源，同源生成 HTML 和 PDF。

## 当前进度

| 内容 | 状态 |
| --- | --- |
| 需求、规范、模板与31章编排 | 已建立 |
| INF-01 本机环境调研 | 已完成；标准库 CPU 验证通过 |
| INF-02 HTML/PDF 一键构建 | 已验收；5页样张、13项检查通过 |
| INF-03 公共符号、术语与导航约定 | 待完成 |
| 正式31章与对应训练实验 | 全部待完成 |
| NumPy/PyTorch 训练依赖 | 尚未安装；默认采用 CPU 小模型路线 |

## 一键生成 HTML / PDF

Windows 本机需要 Node.js 22.13+，PDF 使用已有的 Edge 或 Chrome。首次从仓库根目录准备项目依赖，然后生成独立排版样张：

```powershell
npm ci --ignore-scripts

& ./scripts/build.ps1 -Format html -Sample
& ./scripts/build.ps1 -Format pdf -Sample

# 同时生成两种输出：
& ./scripts/build.ps1 -Format all -Sample
```

输出为 build/html/typesetting-sample.html 和 build/pdf/typesetting-sample.pdf。图和数学字体已内嵌，样张 HTML 可离线打开；PDF 保留矢量图、中文、公式、书签和页码。

当前没有已完成的正式章节，使用 -Sample 只检验排版，不计入教程进度。正式章节登记、单章草稿和全书汇编方法见构建说明。build/ 和 node_modules/ 不进入 Git。

## 环境检查与开发验证

```powershell
python -B scripts/cpu-smoke.py
& ./scripts/check-environment.ps1 -CheckNetwork
npm test
```

CPU 示例当前使用 Python 标准库，未安装的可选框架会单独标注。训练依赖安装和逐章实验仍按计划推进。后续先完成 INF-03，再从第00章逐章实现。
