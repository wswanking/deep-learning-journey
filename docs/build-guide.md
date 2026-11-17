# Markdown 一键生成 HTML 与 PDF

> INF-02 已在本机验收。配图使用真正的 SVG，数学公式离线渲染，PDF 构建自动检查位图对象。
> [验收记录](appendices/build-validation.md) · [章节计划](chapter-plan.md)

## 1. 首次准备

本机实测：Windows 11、Node.js 24.19.0、npm 12.0.2、Microsoft Edge 156.0.4314.42。Node 要求为 22.13 或更新版本；PDF 使用已有的 Edge，找不到时尝试 Chrome，不自动下载浏览器。

从仓库根目录安装项目依赖：

```powershell
npm ci --ignore-scripts
```

依赖固定在 package.json 和 package-lock.json，不安装全局软件。已在独立空目录用锁文件和本机缓存实际验证 npm ci 成功。之后的 Markdown 渲染、数学字体和 PDF 输出不依赖 CDN；初次安装依赖需要网络。

中文使用本机微软雅黑等字体；本机字体已核实，PDF 已验证嵌入中文字体。其他电脑的字体或浏览器不同，版式可能略有差异，应重新检查。

## 2. 立即构建排版样张

可使用独立样张检查工具链；正式章节进度以章节计划为准：

```powershell
# HTML
& ./scripts/build.ps1 -Format html -Sample

# PDF（同时保留对应 HTML）
& ./scripts/build.ps1 -Format pdf -Sample

# 两种输出一起生成
& ./scripts/build.ps1 -Format all -Sample

# 单份样张：layout / vectors
& ./scripts/build.ps1 -Format pdf -Sample -Chapter layout
```

主要输出：

| 范围 | HTML | PDF |
| --- | --- | --- |
| 两份样张合订 | build/html/typesetting-sample.html | build/pdf/typesetting-sample.pdf |
| 单份 layout 样张 | build/html/sample-layout.html | build/pdf/sample-layout.pdf |

本机合订样张为5页，单份 layout 样张为3页。两份样张不是正式教程章节，不计入31章的完成进度。生成目录 build/ 已被 Git 忽略。

如果已有浏览器未在默认位置，可显式指定：

```powershell
& ./scripts/build.ps1 -Format pdf -Sample -BrowserExecutable 'C:\path\to\chrome.exe'
```

构建启动独立的无头浏览器，不读取用户已有浏览器会话。

## 3. 正式章节与全书

编排清单位于 [book.json](book.json)，正式章节登记在 chapters，排版样张登记在 samples。正式章的实际源文件和标题确定后，再登记，例如：

```json
{
  "id": "00",
  "title": "学习路线与本机实验环境",
  "source": "docs/chapters/00-learning-and-environment.md"
}
```

上述是实际第00章的登记格式。真实源文件登记后可显式预览单章：

```powershell
& ./scripts/build.ps1 -Format html -Chapter 00
& ./scripts/build.ps1 -Format pdf -Chapter 00
```

单章若尚未在章节计划中完成，会显示“草稿预览”标记。这解决了先导出检查、后验收勾选的顺序问题。

默认全书仅汇编章节计划中已勾选完成、且已在 book.json 登记的章节，并要求编号升序：

```powershell
& ./scripts/build.ps1 -Format html
& ./scripts/build.ps1 -Format pdf
```

第00～30章均已验收，默认全书包含全部31章。完成清单为空时仍明确拒绝生成空书；草稿和样张不混入正式汇编。新增或修改章节后按统一标准重新验收并构建。

单章文件名为 chapter-00.html/.pdf，全书为 deep-learning-journey.html/.pdf。每次构建另存 .build.json，记录输入章节、公式/矢量图数量；PDF 还记录浏览器、页数、位图检查及书签状态。

## 4. Markdown、公式与矢量图约定

- 使用标准标题、列表、引用、代码块和表格；一个章节恰有一个一级标题。
- 行内公式用 $...$，独立公式用单独成行的 $$...$$；公式由 KaTeX 渲染为 HTML/MathML 和字体/矢量内容，不做截图。
- 非法或不支持的公式直接让构建失败。不要依赖把错误公式变成正文错误提示的默认行为。
- 图片必须是仓库内的 .svg 文件，使用相对路径；SVG 必须有 viewBox。
- 拒绝内嵌位图、foreignObject、滤镜和外部资源等无法保证当前矢量输出约定的元素。
- SVG 内部标识符按图隔离，避免两张图的箭头或裁剪定义相互覆盖。
- Mermaid 代码块当前不自动转图；应先导出纯 SVG，再使用 Markdown 图片引用。未预渲染的 Mermaid 会明确报错，不会混入成品。
- 原始 HTML 默认按文本处理；当前没有网页脚本、视频或交互图的构建支持。

样张矢量图为本项目原创，可用以下命令从解析函数和示意矩阵重新生成：

```powershell
node scripts/generate-sample-figures.mjs
```

生成脚本和 SVG 都保存在仓库；解析曲线与热图不冒充训练实测结果。

## 5. 链接、目录与分页

- 自动生成阅读目录、章节导航和唯一标题锚点；同名小节通过后缀区分。
- 同一输出内的 Markdown 链接改写为内部跳转，包括中文锚点。
- 缺少本地文件或锚点会阻止构建。外部网页链接保留，不在构建时抓取其内容。
- 未选入当前输出的本地文件会作为源文件附件复制到 build/html/sources/，不会偷偷扩展成正式章节。
- 分享包含本地附件链接的 HTML 时，应同时携带 sources/；单纯图文样张的图和数学字体已内嵌，可直接离线打开。
- PDF 内部目录/章节链接已验证；PDF 指向本地源附件的链接需要保留本地 HTML 输出目录，跨电脑分享 PDF 时以内部正文和外部公开链接为主。
- PDF为A4，含页码、目录、书签和可访问结构；表头重复、代码换行、标题避免与后文分离。打印隐藏网页底部前后导航，避免导航单独产生尾页。
- 打印前检查公式、代码、表格和图是否水平溢出；过长内容应拆分，不能依靠缩小整页掩盖问题。

## 6. 构建成功的含义与验收

构建按顺序进行：检查章节选择和文件 → 渲染公式/高亮/矢量图 → PDF 打印布局检查 → PDF 位图对象/书签检查 → 写入正式输出。

失败不会用半成品覆盖上一版。临时文件只写入 build/.work/ 下的独立目录，结束后按已核对的目录范围清理；不清理用户源文件。

运行自动检查：

```powershell
npm test
```

当前13项覆盖：完成清单过滤、草稿、编号、跨章锚点、断链缺图、位图伪装、错误公式、代码中的美元符号、离线数学资源、矢量 PDF 门禁、失败不覆盖、SVG 内部引用及实际多章汇编。

自动成功仍不能代替新章节的目视验收。每次新增或显著修改内容后，实际生成 HTML/PDF，查看中文、公式、图、代码、表格及分页；检查矢量对象和放大效果。当前样张的完整验收依据见记录。

### 6.1 每章排版检查入口

先实际导出对应单章，再运行：

```powershell
node scripts/check-publication.mjs --name chapter-00 --capture
```

将编号换为当前章。检查1440/390px页面宽度、锚点/ID、严格公式、未渲染粗体、SVG内容边界、离线请求和PDF矢量结构；结果及可选目视截图保存在build/qa/。截图仅供排版检查。源图与交付PDF继续遵守矢量要求。工具通过后仍需逐页查看PDF，确认字体、代码、图表和分页。

## 7. 依赖与官方接口

| 依赖 | 本机锁定版本 | 用途 |
| --- | --- | --- |
| markdown-it | 15.0.3 | Markdown 解析 |
| @mdit/plugin-katex | 1.2.0 | 数学分隔符解析；使用项目自己的严格数学渲染器 |
| katex | 0.19.1 | 数学显示与离线字体 |
| highlight.js | 11.12.0 | 代码高亮 |
| @xmldom/xmldom | 0.9.12 | SVG XML 检查 |
| playwright-core | 1.65.1 | 调用本机浏览器输出 PDF |
| pdf-lib | 1.17.1 | 检查 PDF 位图、页数和文档结构 |

- [markdown-it 官方说明](https://markdown-it.github.io/markdown-it/)
- [KaTeX Node.js 接口](https://katex.org/docs/node)
- [Playwright PDF 接口](https://playwright.dev/docs/api/class-page#page-pdf)
- [PDF-LIB PDFDocument 接口](https://pdf-lib.js.org/docs/api/classes/pdfdocument)
