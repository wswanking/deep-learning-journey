import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import MarkdownIt from 'markdown-it';
import { katex as mathPlugin } from '@mdit/plugin-katex';
import katexEngine from 'katex';
import hljs from 'highlight.js';
import { DOMParser } from '@xmldom/xmldom';
import { launchBrowser } from './browser.mjs';
import { PDFDocument, PDFName, PDFRawStream } from 'pdf-lib';

const require = createRequire(import.meta.url);
export const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const escape = value => String(value).replace(/[&<>"']/g, c => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));
const slug = value => value.normalize('NFKC').toLowerCase()
  .replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-');

function sourcePath(root, base, relative) {
  const resolved = path.resolve(base, decodeURIComponent(relative));
  if (path.relative(root, resolved).startsWith('..') || path.isAbsolute(path.relative(root, resolved))) {
    throw new Error('本地资源超出仓库范围：' + relative);
  }
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error('缺少本地文件：' + relative + '（引用位置：' + path.relative(root, base) + '）');
  }
  const real = fs.realpathSync(resolved);
  if (path.relative(fs.realpathSync(root), real).startsWith('..')) {
    throw new Error('资源链接指向仓库外：' + relative);
  }
  return resolved;
}

export function selectDocuments(config, plan, { sample = false, chapter } = {}) {
  const entries = sample ? config.samples : config.chapters;
  if (!Array.isArray(entries)) throw new Error('docs/book.json 缺少章节列表');
  const complete = new Set([...plan.matchAll(/^### \[x\] (\d{2})[：:]/gm)].map(m => m[1]));
  const ids = new Set();
  for (const entry of entries) {
    if (!/^[a-z0-9-]+$/.test(entry.id) || ids.has(entry.id) ||
        typeof entry.source !== 'string' || typeof entry.title !== 'string') {
      throw new Error('编排配置含无效或重复的章节：' + entry.id);
    }
    if (!sample && !/^\d{2}$/.test(entry.id)) throw new Error('正式章节编号须为两位数字');
    ids.add(entry.id);
  }
  const selected = chapter ? entries.filter(entry => entry.id === chapter) :
    entries.filter(entry => sample || complete.has(entry.id));
  if (!selected.length) {
    throw new Error(chapter ? '未登记此章节：' + chapter :
      '尚无已完成章节可汇编。请使用 -Sample 验证排版；正式正文完成后再登记并勾选章节。');
  }
  if (!sample && !chapter) {
    const sorted = [...selected].sort((a, b) => a.id.localeCompare(b.id));
    if (sorted.some((entry, index) => entry !== selected[index])) {
      throw new Error('正式全书章节必须按编号升序编排');
    }
  }
  return selected.map(entry => ({ ...entry, draft: !sample && !complete.has(entry.id) }));
}

export function validateSvg(text, label = 'SVG') {
  if (/<!DOCTYPE|<!ENTITY/i.test(text)) throw new Error(label + ' 不应包含 DTD/实体');
  const document = new DOMParser({
    onError: (level, message) => { throw new Error(label + ' XML 解析失败：' + message); },
  }).parseFromString(text, 'image/svg+xml');
  const root = document.documentElement;
  if (!root || root.localName !== 'svg' || root.namespaceURI !== 'http://www.w3.org/2000/svg' ||
      !root.hasAttribute('viewBox')) throw new Error(label + ' 缺少 SVG 根元素或 viewBox');
  for (const node of Array.from(document.getElementsByTagName('*'))) {
    const name = node.localName.toLowerCase();
    if (['image', 'foreignobject', 'script', 'filter', 'feimage'].includes(name)) {
      throw new Error(label + ' 含位图或无法保证矢量输出的元素：' + name);
    }
    for (const attribute of Array.from(node.attributes || [])) {
      if (/^on/i.test(attribute.name)) throw new Error(label + ' 含事件脚本');
      const resourceUrls = [...attribute.value.matchAll(/url\(([^)]+)\)/gi)];
      const hasExternalUrl = resourceUrls.some(match =>
        !match[1].trim().replace(/^["']|["']$/g, '').startsWith('#'));
      if ((attribute.localName === 'href' && !attribute.value.startsWith('#')) || hasExternalUrl) {
        throw new Error(label + ' 含外部/非内嵌矢量资源');
      }
    }
    if (name === 'style' && /@import|url\(/i.test(node.textContent)) {
      throw new Error(label + ' 样式中含外部资源');
    }
  }
  return text.replace(/<\?xml[^>]*\?>/g, '').trim();
}

function svgWithIds(svg, prefix) {
  const ids = new Set([...svg.matchAll(/\bid=(["'])(.*?)\1/g)].map(match => match[2]));
  return svg.replace(/\bid=(["'])(.*?)\1/g, (_, quote, id) => 'id=' + quote + prefix + id + quote)
    .replace(/url\(\s*(["']?)#([^)'"\s]+)\1\s*\)/g, (match, quote, id) =>
      ids.has(id) ? 'url(' + quote + '#' + prefix + id + quote + ')' : match)
    .replace(/((?:xlink:)?href)=(["'])#(.*?)\2/g, (match, name, quote, id) =>
      ids.has(id) ? name + '=' + quote + '#' + prefix + id + quote : match);
}

function markdown() {
  const md = new MarkdownIt({
    html: false, linkify: true, typographer: false,
    highlight(code, language) {
      if (language === 'mermaid') throw new Error('Mermaid 须先导出 SVG，再以 Markdown 图片引用；不输出未渲染代码图。');
      if (language && hljs.getLanguage(language)) {
        return hljs.highlight(code, { language, ignoreIllegals: true }).value;
      }
      return escape(code);
    },
  }).use(mathPlugin, { delimiters: 'dollars' });
  // Isolate display TeX from Markdown headings, lists and emphasis, as on GitHub.
  // Normalize math fences to the existing token so counting and strict rendering agree.
  md.core.ruler.after('block', 'github_math_fence', state => {
    for (const token of state.tokens) {
      if (token.type === 'fence' && token.info.trim() === 'math') {
        token.type = 'math_block';
        token.tag = 'math';
      }
    }
  });
  // GitHub's protected inline form is $ followed by a backtick, TeX, backtick, $.
  // The dollar tokenizer has already kept the expression opaque to Markdown.
  md.core.ruler.after('inline', 'github_inline_math', state => {
    for (const token of state.tokens) {
      for (const child of token.children || []) {
        if (child.type === 'math_inline' && child.content.length >= 2 &&
            child.content.startsWith('\x60') && child.content.endsWith('\x60')) {
          child.content = child.content.slice(1, -1);
        }
      }
    }
  });
  // Keep the plugin's tokenizer, but enforce errors in our own renderer.
  const renderMath = (content, displayMode) => katexEngine.renderToString(content, {
    displayMode, throwOnError: true, strict: 'error', trust: false, output: 'htmlAndMathml',
  });
  md.renderer.rules.math_inline = (tokens, index) => renderMath(tokens[index].content, false);
  md.renderer.rules.math_block = (tokens, index) => '<div class="math-block">' + renderMath(tokens[index].content, true) + '</div>\n';
  return md;
}

function headings(md, text, prefix) {
  const tokens = md.parse(text, {});
  const list = [];
  const duplicates = new Map();
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type !== 'heading_open') continue;
    const label = tokens[i + 1].content;
    const base = slug(label) || 'section';
    const count = duplicates.get(base) || 0;
    duplicates.set(base, count + 1);
    const local = base + (count ? '-' + count : '');
    const id = prefix + '-' + local;
    tokens[i].attrSet('id', id);
    list.push({ local, id, level: Number(tokens[i].tag.slice(1)), title: label });
  }
  if (list.filter(h => h.level === 1).length !== 1) {
    throw new Error('每份章节必须恰有一个一级标题：' + prefix);
  }
  return { tokens, headings: list };
}

export function compileBook(root, config, entries, { sample = false } = {}) {
  const md = markdown();
  const attachments = new Map();
  const docs = entries.map(entry => {
    const absolute = sourcePath(root, root, entry.source);
    return { ...entry, absolute, ...headings(md, fs.readFileSync(absolute, 'utf8'), 'chapter-' + entry.id) };
  });
  const byPath = new Map(docs.map(doc => [doc.absolute, doc]));
  let imageCount = 0;
  let mathCount = 0;
  let current;
  const linkRenderer = md.renderer.rules.link_open ||
    ((tokens, index, options, env, self) => self.renderToken(tokens, index, options));
  md.renderer.rules.link_open = (tokens, index, options, env, self) => {
    const token = tokens[index];
    const href = token.attrGet('href') || '';
    if (!/^(https?:|mailto:)/i.test(href)) {
      if (/^[a-z][a-z0-9+.-]*:/i.test(href)) throw new Error('不支持的链接协议：' + href);
      const split = href.indexOf('#');
      const local = split < 0 ? href : href.slice(0, split);
      const fragment = split < 0 ? '' : decodeURIComponent(href.slice(split + 1));
      const target = local ? sourcePath(root, path.dirname(current.absolute), local) : current.absolute;
      let targetDoc = byPath.get(target);
      if (!targetDoc && path.extname(target).toLowerCase() === '.md') {
        targetDoc = headings(md, fs.readFileSync(target, 'utf8'), 'attachment');
      }
      let anchor;
      if (fragment && targetDoc) {
        anchor = targetDoc.headings.find(heading => heading.local === fragment);
        if (!anchor) throw new Error('缺少本地锚点：' + href + '（' + current.source + '）');
      }
      if (byPath.has(target)) {
        token.attrSet('href', '#' + (anchor ? anchor.id : byPath.get(target).headings[0].id));
      } else {
        const relative = path.relative(root, target).split(path.sep).join('/');
        attachments.set(relative, target);
        token.attrSet('href', './sources/' + relative.split('/').map(encodeURIComponent).join('/') +
          (fragment ? '#' + encodeURIComponent(fragment) : ''));
      }
    }
    return linkRenderer(tokens, index, options, env, self);
  };
  md.renderer.rules.image = (tokens, index) => {
    const token = tokens[index];
    const source = token.attrGet('src') || '';
    if (/^(?:[a-z]+:|\/\/)/i.test(source) || path.extname(source).toLowerCase() !== '.svg') {
      throw new Error('教程图片必须是仓库内的 SVG 矢量图：' + source);
    }
    const absolute = sourcePath(root, path.dirname(current.absolute), source);
    const svg = svgWithIds(validateSvg(fs.readFileSync(absolute, 'utf8'), source), 'figure-' + imageCount + '-');
    imageCount++;
    return '<span class="vector-figure" role="img" aria-label="' + escape(token.content) + '">' +
      svg + '<span class="figure-caption">' + escape(token.content) + '</span></span>';
  };
  let chapters = '';
  for (let i = 0; i < docs.length; i++) {
    current = docs[i];
    for (const token of current.tokens) {
      if (/math/.test(token.type)) mathCount++;
      if (token.children) mathCount += token.children.filter(child => /math/.test(child.type)).length;
    }
    const previous = docs[i - 1];
    const next = docs[i + 1];
    const navigation = '<nav class="chapter-nav" aria-label="章节导航">' +
      (previous ? '<a href="#' + previous.headings[0].id + '">← ' + escape(previous.title) + '</a>' : '<span></span>') +
      '<a href="#contents">目录</a>' +
      (next ? '<a href="#' + next.headings[0].id + '">' + escape(next.title) + ' →</a>' : '<span></span>') + '</nav>';
    chapters += '<article class="chapter">' +
      (current.draft ? '<p class="notice">草稿预览：本章尚未通过完成验收，不纳入默认全书。</p>' : '') +
      md.renderer.render(current.tokens, md.options, {}) + navigation + '</article>';
  }
  const toc = docs.map(doc => '<li><a href="#' + doc.headings[0].id + '">' + escape(doc.title) +
    '</a><ol>' + doc.headings.filter(h => h.level === 2).map(h =>
      '<li><a href="#' + h.id + '">' + escape(h.title) + '</a></li>').join('') + '</ol></li>').join('');
  const katexPath = require.resolve('katex/dist/katex.min.css');
  const katexCss = fs.readFileSync(katexPath, 'utf8').replace(/url\(([^)]+)\)/g, (_, value) => {
    const filename = value.replace(/^["']|["']$/g, '');
    const full = path.resolve(path.dirname(katexPath), filename);
    const mime = full.endsWith('.woff2') ? 'font/woff2' : full.endsWith('.woff') ? 'font/woff' : 'font/ttf';
    return 'url(data:' + mime + ';base64,' + fs.readFileSync(full).toString('base64') + ')';
  });
  const css = fs.readFileSync(path.join(root, 'docs/styles/book.css'), 'utf8');
  const codeCss = fs.readFileSync(require.resolve('highlight.js/styles/github.css'), 'utf8');
  const title = sample ? '图文与数学排版样张' : config.title;
  const html = '<!doctype html><html lang="' + escape(config.language || 'zh-CN') + '"><head>' +
    '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +
    '<title>' + escape(title) + '</title><style>' + katexCss + '\n' + codeCss + '\n' + css + '</style></head><body>' +
    '<div class="book-shell"><aside class="sidebar"><p class="brand">DEEP LEARNING<br>JOURNEY</p><nav aria-label="全书目录"><ol>' + toc +
    '</ol></nav></aside><main class="book-main"><section class="frontmatter">' +
    '<p class="eyebrow">' + (sample ? 'TYPESETTING / 排版验证' : 'DEEP LEARNING JOURNEY') + '</p>' +
    '<h1 class="book-title">' + escape(title) + '</h1><p class="subtitle">' + escape(config.subtitle) + '</p>' +
    (sample ? '<p class="notice">本文件用于检验中文、数学、矢量图和分页，不是已完成教程章节。</p>' : '') +
    '<div class="meta-row"><span>Markdown 同源构建</span><span>SVG 矢量配图</span><span>离线数学排版</span></div>' +
    '<nav class="contents" id="contents" aria-label="阅读目录"><h2>阅读目录</h2><ol>' + toc +
    '</ol></nav></section>' + chapters + '</main></div></body></html>';
  return {
    html, attachments,
    metadata: { title, sample, chapters: docs.map(d => ({ id: d.id, source: d.source, draft: d.draft })),
      vector_figures: imageCount, math_expressions: mathCount },
  };
}

export async function assertVectorPdf(bytes) {
  const pdf = await PDFDocument.load(bytes);
  const rasterImages = pdf.context.enumerateIndirectObjects().filter(([, object]) =>
    object instanceof PDFRawStream && object.dict.get(PDFName.of('Subtype')) === PDFName.of('Image'));
  if (rasterImages.length) throw new Error('PDF 含 ' + rasterImages.length + ' 个位图对象，未满足矢量图要求。');
  return { pages: pdf.getPageCount(), raster_images: 0,
    has_outline: pdf.catalog.has(PDFName.of('Outlines')),
    tagged: pdf.catalog.has(PDFName.of('StructTreeRoot')) };
}

export async function build({ root = projectRoot, format = 'html', sample = false, chapter, browserExecutable } = {}) {
  if (!['html', 'pdf', 'all'].includes(format)) throw new Error('Format 必须为 html、pdf 或 all');
  const config = JSON.parse(fs.readFileSync(path.join(root, 'docs/book.json'), 'utf8'));
  const plan = fs.readFileSync(path.join(root, 'docs/chapter-plan.md'), 'utf8');
  const entries = selectDocuments(config, plan, { sample, chapter });
  const compiled = compileBook(root, config, entries, { sample });
  const name = chapter ? (sample ? 'sample-' : 'chapter-') + chapter :
    sample ? 'typesetting-sample' : 'deep-learning-journey';
  const htmlDirectory = path.join(root, 'build/html');
  const pdfDirectory = path.join(root, 'build/pdf');
  const work = path.join(root, 'build/.work', crypto.randomUUID());
  fs.mkdirSync(work, { recursive: true });
  const stagedHtml = path.join(work, name + '.html');
  const attachmentBase = pathToFileURL(htmlDirectory + path.sep).href + 'sources/';
  fs.writeFileSync(stagedHtml, compiled.html.replaceAll('href="./sources/', 'href="' + attachmentBase));
  let browser;
  try {
    if (format !== 'html') {
      browser = await launchBrowser(browserExecutable);
      compiled.metadata.browser = browser.version();
      const page = await browser.newPage({ viewport: { width: 658, height: 1000 } });
      await page.route('**/*', route =>
        /^(file:|data:|about:)/.test(route.request().url()) ? route.continue() : route.abort());
      await page.goto(pathToFileURL(stagedHtml).href, { waitUntil: 'load' });
      await page.emulateMedia({ media: 'print' });
      await page.evaluate(() => document.fonts.ready);
      const overflow = await page.evaluate(() => Array.from(
        document.querySelectorAll('table,pre,.katex-display,.vector-figure')
      ).filter(el => el.scrollWidth > el.clientWidth + 3).map(el => el.tagName + ':' + el.textContent.slice(0, 80)));
      if (overflow.length) throw new Error('打印布局存在水平溢出，请拆分公式/表格：' + overflow.join('; '));
      await page.pdf({
        path: path.join(work, name + '.pdf'), format: 'A4',
        preferCSSPageSize: true, printBackground: true, tagged: true, outline: true,
        displayHeaderFooter: true, headerTemplate: '<span></span>',
        footerTemplate: '<div style="width:100%;font-size:9px;color:#667085;text-align:center;">' +
          '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
        margin: { top: '18mm', bottom: '18mm', left: '18mm', right: '18mm' },
      });
      compiled.metadata.pdf_check = await assertVectorPdf(fs.readFileSync(path.join(work, name + '.pdf')));
      if (!compiled.metadata.pdf_check.has_outline || !compiled.metadata.pdf_check.tagged) {
        throw new Error('浏览器未输出书签或可访问文档结构，请更新浏览器后重试。');
      }
      await browser.close();
      browser = null;
    }
    fs.mkdirSync(htmlDirectory, { recursive: true });
    fs.writeFileSync(path.join(htmlDirectory, name + '.html'), compiled.html);
    for (const [relative, source] of compiled.attachments) {
      const destination = path.join(htmlDirectory, 'sources', relative);
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      fs.copyFileSync(source, destination);
    }
    if (format !== 'html') {
      fs.mkdirSync(pdfDirectory, { recursive: true });
      fs.copyFileSync(path.join(work, name + '.pdf'), path.join(pdfDirectory, name + '.pdf'));
    }
    const report = { ...compiled.metadata, requested_format: format, built_at: new Date().toISOString(),
      html: path.join(htmlDirectory, name + '.html'),
      pdf: format === 'html' ? null : path.join(pdfDirectory, name + '.pdf') };
    fs.writeFileSync(path.join(format === 'html' ? htmlDirectory : pdfDirectory, name + '.build.json'),
      JSON.stringify(report, null, 2) + '\n');
    return report;
  } finally {
    if (browser) await browser.close();
    // Only remove this invocation's generated UUID directory under build/.work.
    const relative = path.relative(fs.realpathSync(path.join(root, 'build/.work')), fs.realpathSync(work));
    if (!relative.startsWith('..') && !path.isAbsolute(relative)) fs.rmSync(work, { recursive: true, force: true });
  }
}

async function main() {
  const args = process.argv.slice(2);
  const options = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--sample') options.sample = true;
    else if (['--format', '--chapter', '--browser'].includes(args[i]) && args[i + 1]) {
      const key = args[i].slice(2);
      options[key === 'browser' ? 'browserExecutable' : key] = args[++i];
    } else throw new Error('无效构建参数：' + args[i]);
  }
  console.log(JSON.stringify(await build(options), null, 2));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error('[构建失败] ' + error.message); process.exitCode = 1; });
}
