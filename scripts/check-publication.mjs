import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { assertVectorPdf } from './build.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
let name = 'deep-learning-journey';
let browserExecutable;
let capture = false;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--name' && args[i + 1]) name = args[++i];
  else if (args[i] === '--browser' && args[i + 1]) browserExecutable = args[++i];
  else if (args[i] === '--capture') capture = true;
  else throw new Error('无效验收参数：' + args[i]);
}
if (!/^(chapter-\d{2}|sample-(layout|vectors)|typesetting-sample|deep-learning-journey)$/.test(name)) {
  throw new Error('Name 必须是构建产物名称，不接受文件路径。');
}
const html = path.join(root, 'build/html', name + '.html');
const pdf = path.join(root, 'build/pdf', name + '.pdf');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'build/pdf', name + '.build.json'), 'utf8'));
const output = path.join(root, 'build/qa', name);
fs.mkdirSync(output, { recursive: true });
const sha256 = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const report = {
  checked_at: new Date().toISOString(),
  html, html_sha256: sha256(html), pdf, pdf_sha256: sha256(pdf),
  chapters: manifest.chapters,
  pdf_structure: await assertVectorPdf(fs.readFileSync(pdf)),
  viewports: [],
  errors: [],
};
if (!report.pdf_structure.has_outline || !report.pdf_structure.tagged) {
  report.errors.push('PDF 缺少书签或文档结构');
}
let browser;
try {
  if (browserExecutable) browser = await chromium.launch({ headless: true, executablePath: browserExecutable });
  else {
    for (const channel of ['msedge', 'chrome']) {
      try { browser = await chromium.launch({ headless: true, channel }); break; }
      catch {}
    }
  }
  if (!browser) throw new Error('未找到已有的 Edge/Chrome');
  report.browser = browser.version();
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } });
    const requests = [];
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.route('**/*', route => {
      if (/^(file:|data:|about:)/.test(route.request().url())) return route.continue();
      requests.push(route.request().url());
      return route.abort();
    });
    await page.goto(pathToFileURL(html).href, { waitUntil: 'load' });
    await page.evaluate(() => document.fonts.ready);
    const result = await page.evaluate(() => {
      const nodes = [...document.querySelectorAll('[id]')];
      const ids = nodes.map(node => node.id);
      const duplicates = [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))];
      const missingAnchors = [...document.querySelectorAll('a[href^="#"]')].map(link => {
        try { return decodeURIComponent(link.hash.slice(1)); }
        catch { return link.hash.slice(1); }
      }).filter(id => id && !document.getElementById(id));
      const figures = [...document.querySelectorAll('.vector-figure > svg')];
      const clippedFigures = figures.map((svg, index) => {
        const b = svg.getBBox();
        const v = svg.viewBox.baseVal;
        return { index: index + 1, bbox: { x: b.x, y: b.y, width: b.width, height: b.height },
          viewBox: { x: v.x, y: v.y, width: v.width, height: v.height },
          clipped: b.x < v.x - 2 || b.y < v.y - 2 ||
            b.x + b.width > v.x + v.width + 2 || b.y + b.height > v.y + v.height + 2 };
      }).filter(figure => figure.clipped);
      return {
        viewport_width: innerWidth,
        document_width: document.documentElement.scrollWidth,
        horizontal_page_overflow: document.documentElement.scrollWidth > innerWidth + 2,
        headings: document.querySelectorAll('.chapter h1').length,
        vector_figures: figures.length,
        math_expressions: document.querySelectorAll('.katex').length,
        math_errors: document.querySelectorAll('.katex-error').length,
        duplicates, missing_anchors: [...new Set(missingAnchors)], clipped_figures: clippedFigures,
        unrendered_markdown: [...document.querySelectorAll(".chapter p, .chapter li")].map(el => { const copy = el.cloneNode(true); copy.querySelectorAll("code, pre").forEach(node => node.remove()); return copy.textContent; }).filter(text => /\*\*[^*\n]+\*\*/.test(text)).slice(0, 10),
        raster_elements: document.querySelectorAll('img, canvas, video, svg image, svg foreignObject').length,
      };
    });
    result.external_requests = requests;
    result.page_errors = pageErrors;
    report.viewports.push(result);
    for (const key of ['horizontal_page_overflow', 'math_errors', 'raster_elements']) {
      if (result[key]) report.errors.push(width + 'px: ' + key);
    }
    for (const key of ['duplicates', 'missing_anchors', 'clipped_figures', 'external_requests', 'page_errors', 'unrendered_markdown']) {
      if (result[key].length) report.errors.push(width + 'px: ' + key);
    }
    if (result.vector_figures !== manifest.vector_figures || result.math_expressions !== manifest.math_expressions) {
      report.errors.push(width + 'px: 构建清单与实际图/公式数量不同');
    }
    if (capture) {
      await page.screenshot({ path: path.join(output, 'html-' + width + '.png') });
      if (width === 1440) {
        const figures = page.locator('.vector-figure');
        for (let i = 0; i < await figures.count(); i++) {
          await figures.nth(i).screenshot({ path: path.join(output, 'figure-' + (i + 1) + '.png') });
        }
      }
    }
    await page.close();
  }
} finally {
  if (browser) await browser.close();
}
report.status = report.errors.length ? 'failed' : 'passed';
fs.writeFileSync(path.join(output, 'publication-check.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
if (report.errors.length) process.exitCode = 1;
