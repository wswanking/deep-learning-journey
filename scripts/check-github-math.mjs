import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import MarkdownIt from 'markdown-it';
import { katex as mathPlugin } from '@mdit/plugin-katex';
import { launchBrowser } from './browser.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repository = 'wswanking/deep-learning-journey';
const articleSelector = 'article.markdown-body';
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const sha256 = value => crypto.createHash('sha256').update(value).digest('hex');

// Whitespace is normalized, but TeX commands, operators and row separators are retained.
export function normalizeTeX(tex) {
  return String(tex).replace(/\r\n?/g, '\n').replace(/\s+/g, ' ').trim();
}

// Recognize only the observed GitHub data-latex row-break encoding.
// Source and actual line prefixes must agree byte-for-byte after LF normalization.
export function canonicalizeGitHubTeX(sourceTex, actualTex) {
  const source = String(sourceTex).replace(/\r\n?/g, '\n');
  const actual = String(actualTex).replace(/\r\n?/g, '\n');
  const sourceLines = source.split('\n');
  const actualLines = actual.split('\n');
  const canonicalLines = [...actualLines];
  const events = [];
  // Both lines must end in a physical LF: never rewrite a final unterminated line.
  for (let i = 0; i < Math.min(sourceLines.length, actualLines.length) - 1; i++) {
    const left = sourceLines[i];
    const right = actualLines[i];
    const sourceRun = left.match(/(\\+)$/)?.[1] || '';
    const actualRun = right.match(/(\\+)$/)?.[1] || '';
    if (sourceRun.length !== 2 || actualRun.length !== 3 ||
        left.slice(0, -2) !== right.slice(0, -3)) continue;
    canonicalLines[i] = right.slice(0, -1);
    events.push({
      tex_line: i + 1, source_column: left.length - 1,
      actual_column: right.length - 2, removed_actual_column: right.length,
      source_backslashes: 2, actual_backslashes: 3,
    });
  }
  const canonicalActual = canonicalLines.join('\n');
  return {
    source_tex: source, actual_tex: actual, canonical_actual_tex: canonicalActual,
    normalized_source: normalizeTeX(source), normalized_actual: normalizeTeX(canonicalActual),
    equal: normalizeTeX(source) === normalizeTeX(canonicalActual), events,
  };
}

// Ordinary fenced code and inline code remain code; only math fences become formulas.
export function expectedMath(source, sourcePath = '') {
  const md = new MarkdownIt({ html: false }).use(mathPlugin, { delimiters: 'dollars' });
  const tokens = md.parse(source, {});
  const formulas = [];
  let heading = '';
  let h1 = 0;
  let ordinaryCodeBlocks = 0;
  const append = (token, display, line) => {
    // GitHub's inline delimiter wraps the original TeX in one pair of backticks.
    const tex = !display && token.content.startsWith('`') && token.content.endsWith('`')
      ? token.content.slice(1, -1) : token.content;
    formulas.push({ index: formulas.length, display, tex,
      normalized_tex: normalizeTeX(tex), line, heading });
  };
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.type === 'heading_open') {
      heading = tokens[i + 1]?.content || '';
      if (token.tag === 'h1') h1++;
    }
    if (token.type === 'math_block' ||
        (token.type === 'fence' && token.info.trim() === 'math')) {
      append(token, true, (token.map?.[0] ?? 0) + 1);
    } else if (token.type === 'fence' || token.type === 'code_block') {
      ordinaryCodeBlocks++;
    }
    for (const child of token.children || []) {
      if (child.type === 'math_inline') append(child, false, (token.map?.[0] ?? 0) + 1);
    }
  }
  return {
    source: sourcePath, source_sha256: sha256(source), h1,
    total: formulas.length, display: formulas.filter(formula => formula.display).length,
    inline: formulas.filter(formula => !formula.display).length,
    ordinary_code_blocks: ordinaryCodeBlocks, formulas,
  };
}

// This function is serialized into the page; keep all DOM helpers inside it.
export function collectArticleMath() {
  const article = document.querySelector('article.markdown-body');
  if (!article) return { article_found: false };
  const renderers = [...article.querySelectorAll('math-renderer')];
  const latexNodes = [...article.querySelectorAll('math[data-latex]')];
  const formulas = [];
  let incompleteRenderers = 0;
  let duplicateLatexNodes = 0;
  const nodes = [...article.querySelectorAll('math-renderer, math[data-latex]')];
  for (const node of nodes) {
    if (node.localName === 'math' && node.closest('math-renderer')) continue;
    const renderer = node.localName === 'math-renderer' ? node : null;
    const childNodes = renderer ? [...renderer.querySelectorAll('math[data-latex]')] : [node];
    const child = childNodes[0];
    if (childNodes.length > 1) duplicateLatexNodes += childNodes.length - 1;
    const tex = child?.getAttribute('data-latex') ??
      renderer?.getAttribute('data-latex') ??
      renderer?.getAttribute('data-math') ??
      renderer?.getAttribute('data-tex');
    if (tex === null || tex === undefined || !tex.trim()) {
      if (renderer) incompleteRenderers++;
      continue;
    }
    const display = Boolean(
      renderer?.classList.contains('js-display-math') ||
      renderer?.getAttribute('data-display') === 'true' ||
      child?.getAttribute('display') === 'block' ||
      (!renderer?.classList.contains('js-inline-math') && node.closest('.math-display'))
    );
    formulas.push({
      index: formulas.length, display, tex,
      renderer_index: renderer ? renderers.indexOf(renderer) : null,
      latex_node_index: child ? latexNodes.indexOf(child) : null,
    });
  }
  const withoutMathOrCode = element => {
    const copy = element.cloneNode(true);
    copy.querySelectorAll('math-renderer, math, code, pre').forEach(node => node.remove());
    return copy.textContent || '';
  };
  const headings = [...article.querySelectorAll('h1,h2,h3,h4,h5,h6')];
  const rawHeadings = headings.map(element => ({
    tag: element.localName, text: withoutMathOrCode(element).trim(),
  })).filter(item => /\$\$|\\(?:begin|end)\{/.test(item.text));
  const rawMarkers = [...article.querySelectorAll('p,li')].map(withoutMathOrCode)
    .map(text => text.trim()).filter(text => text.includes('$$')).slice(0, 20);
  const mathCodeBlocks = [...article.querySelectorAll('pre')].filter(element => {
    if (element.querySelector('math-renderer, math[data-latex]')) return false;
    const code = element.querySelector('code');
    const labels = [
      element.className, element.getAttribute('lang'), element.getAttribute('data-language'),
      code?.className, code?.getAttribute('lang'), code?.getAttribute('data-language'),
      element.parentElement?.className,
    ].filter(value => typeof value === 'string').join(' ');
    return /(?:language-|source-)math\b|(?:^|\s)math(?:\s|$)/.test(labels);
  }).map(element => element.textContent.slice(0, 500));
  const mathErrors = [...article.querySelectorAll(
    'merror, mjx-merror, .MathJax_Error, .katex-error, [data-math-error], math-renderer .flash-error, .flash-error'
  )].map(element => ({ tag: element.localName,
    class_name: typeof element.className === 'string' ? element.className : '',
    text: element.textContent.trim().slice(0, 500) }));
  const unrenderedCommands = [];
  renderers.forEach((renderer, rendererIndex) => {
    [...renderer.querySelectorAll('math')].forEach((math, mathIndex) => {
      const copy = math.cloneNode(true);
      copy.querySelectorAll('annotation, annotation-xml').forEach(node => node.remove());
      const visibleText = copy.textContent || '';
      const commands = [...new Set(visibleText.match(/\\[A-Za-z]+/g) || [])];
      if (commands.length) unrenderedCommands.push({ renderer_index: rendererIndex,
        math_index: mathIndex, commands, visible_text: visibleText.trim().slice(0, 500) });
    });
  });
  return {
    article_found: true, article_text_length: article.textContent.length,
    h1: [...article.querySelectorAll('h1')].map(element => element.textContent.trim()),
    math_renderer_count: renderers.length, data_latex_math_count: latexNodes.length,
    incomplete_renderers: incompleteRenderers, duplicate_latex_nodes: duplicateLatexNodes,
    total: formulas.length, display: formulas.filter(formula => formula.display).length,
    inline: formulas.filter(formula => !formula.display).length,
    raw_tex_headings: rawHeadings, raw_math_markers: rawMarkers,
    unrendered_math_code_blocks: mathCodeBlocks, unrendered_math_commands: unrenderedCommands,
    math_errors: mathErrors, formulas,
  };
}

export function compareMath(expected, actual) {
  const errors = [];
  const mismatches = [];
  const normalizations = [];
  const add = (condition, message) => { if (condition) errors.push(message); };
  add(expected.h1 !== 1, 'local_source_h1_count_' + expected.h1);
  add(!actual.article_found, 'missing_article_markdown_body');
  if (!actual.article_found) return { status: 'failed', errors, mismatches };
  add(actual.h1.length !== 1, 'github_h1_count_' + actual.h1.length);
  add(actual.total !== expected.total, 'formula_count_expected_' + expected.total + '_actual_' + actual.total);
  add(actual.display !== expected.display, 'display_count_expected_' + expected.display + '_actual_' + actual.display);
  add(actual.inline !== expected.inline, 'inline_count_expected_' + expected.inline + '_actual_' + actual.inline);
  add(actual.incomplete_renderers > 0, 'incomplete_math_renderers_' + actual.incomplete_renderers);
  add(actual.duplicate_latex_nodes > 0, 'duplicate_latex_nodes_' + actual.duplicate_latex_nodes);
  add(actual.raw_tex_headings.length > 0, 'raw_tex_in_heading');
  add(actual.raw_math_markers.length > 0, 'unrendered_double_dollar_markers');
  add(actual.unrendered_math_code_blocks.length > 0, 'unrendered_math_code_blocks');
  add(actual.math_errors.length > 0, 'math_rendering_errors');
  add((actual.unrendered_math_commands || []).length > 0, 'unrendered_math_commands');
  for (let i = 0; i < Math.max(expected.formulas.length, actual.formulas.length); i++) {
    const left = expected.formulas[i];
    const right = actual.formulas[i];
    const pair = left && right ? canonicalizeGitHubTeX(left.tex, right.tex) : null;
    if (pair?.events.length) normalizations.push({
      formula_index: i, local_token_line: left.line,
      events: pair.events.map(event => ({
        ...event, source_line: left.line + event.tex_line - (left.display ? 0 : 1),
      })),
    });
    if (!left || !right || left.display !== right.display || !pair.equal) {
      mismatches.push({
        index: i, local_line: left?.line ?? null, expected_display: left?.display ?? null,
        actual_display: right?.display ?? null, expected_tex: left?.tex ?? null,
        actual_tex: right?.tex ?? null,
      });
    }
  }
  add(mismatches.length > 0, 'ordered_tex_or_display_mismatch_' + mismatches.length);

  // Match a multiset using the same source-guided rule. A maximum matching keeps
  // repeated formulas and a reordered formula from creating false inventory gaps;
  // the ordered comparison above still rejects any reordering.
  const assigned = new Array(actual.formulas.length).fill(-1);
  const equivalent = new Map();
  const same = (e, a) => {
    const key = e + ':' + a;
    if (!equivalent.has(key)) equivalent.set(key,
      expected.formulas[e].display === actual.formulas[a].display &&
      canonicalizeGitHubTeX(expected.formulas[e].tex, actual.formulas[a].tex).equal);
    return equivalent.get(key);
  };
  const assign = (e, seen) => {
    const candidates = Array.from({ length: actual.formulas.length }, (_, a) => a);
    if (e < candidates.length) { candidates.splice(e, 1); candidates.unshift(e); }
    for (const a of candidates) {
      if (seen.has(a) || !same(e, a)) continue;
      seen.add(a);
      if (assigned[a] < 0 || assign(assigned[a], seen)) {
        assigned[a] = e;
        return true;
      }
    }
    return false;
  };
  for (let e = 0; e < expected.formulas.length; e++) assign(e, new Set());
  const matchedExpected = new Set(assigned.filter(index => index >= 0));
  const inventory = formulas => {
    const values = new Map();
    for (const formula of formulas) {
      const key = (formula.display ? 'display:' : 'inline:') +
        canonicalizeGitHubTeX(formula.tex, formula.tex).normalized_source;
      const entry = values.get(key) || { display: formula.display, tex: formula.tex, count: 0 };
      entry.count++;
      values.set(key, entry);
    }
    return [...values.values()];
  };
  return {
    status: errors.length ? 'failed' : 'passed', errors, mismatches,
    tex_normalization: {
      rule: 'source line ends in exactly two backslashes and LF; actual line has the identical prefix and exactly three backslashes and LF; remove only the extra actual backslash',
      transformed_formulas: normalizations.length,
      removed_control_spaces: normalizations.reduce((sum, item) => sum + item.events.length, 0),
      formulas: normalizations,
    },
    missing_tex: inventory(expected.formulas.filter((_, e) => !matchedExpected.has(e))),
    unexpected_tex: inventory(actual.formulas.filter((_, a) => assigned[a] < 0)),
  };
}

function optionsFrom(args) {
  const options = { ref: 'main', capture: false };
  for (let i = 0; i < args.length; i++) {
    if (['--chapter', '--ref', '--proxy', '--browser'].includes(args[i]) && args[i + 1]) {
      options[args[i].slice(2)] = args[++i];
    } else if (args[i] === '--capture') options.capture = true;
    else if (args[i] === '--help') options.help = true;
    else throw new Error('Invalid argument: ' + args[i]);
  }
  if (options.chapter && !/^\d{2}$/.test(options.chapter)) throw new Error('Chapter must be 00 through 30.');
  if (!/^(?:main|[0-9a-f]{7,40})$/i.test(options.ref)) throw new Error('Ref must be main or a 7-40 character commit SHA.');
  if (options.proxy) {
    const url = new URL(options.proxy);
    if (!['http:', 'https:', 'socks5:'].includes(url.protocol) || !url.hostname ||
        (url.pathname !== '/' && url.pathname !== '') || url.search || url.hash) {
      throw new Error('Proxy must be an HTTP(S) or SOCKS5 server URL without a path/query.');
    }
    options.browserProxy = { server: url.protocol + '//' + url.host };
    if (url.username) options.browserProxy.username = decodeURIComponent(url.username);
    if (url.password) options.browserProxy.password = decodeURIComponent(url.password);
  }
  return options;
}

async function stableArticle(page, expected) {
  const started = Date.now();
  const deadline = started + 90000;
  let signature = '';
  let stableSince = started;
  let snapshot = { article_found: false };
  while (Date.now() < deadline) {
    snapshot = await page.evaluate(collectArticleMath);
    if (snapshot.article_found) {
      const current = JSON.stringify(snapshot);
      if (current !== signature) { signature = current; stableSince = Date.now(); }
      // A concrete GitHub renderer error is terminal; do not wait for missing MathML.
      if (snapshot.math_errors.length > 0 && Date.now() - stableSince >= 1500) {
        return { ...snapshot, stability: { elapsed_ms: Date.now() - started,
          stable_ms: Date.now() - stableSince, timed_out: false, terminal_math_error: true } };
      }
      const ready = snapshot.incomplete_renderers === 0;
      const correctCount = snapshot.total === expected.total;
      // A stable wrong document is evidence of a rendering failure, not a pass.
      const minimumStable = correctCount ? 2250 : 8000;
      if (ready && Date.now() - stableSince >= minimumStable &&
          (correctCount || Date.now() - started >= 10000)) {
        return { ...snapshot, stability: { elapsed_ms: Date.now() - started,
          stable_ms: Date.now() - stableSince, timed_out: false } };
      }
    }
    await delay(750);
  }
  return { ...snapshot, stability: {
    elapsed_ms: Date.now() - started, stable_ms: Date.now() - stableSince, timed_out: true,
  } };
}

async function captureEquations(page, expected, actual, outputFolder) {
  // Bound late SVG/font decoding before the first scroll; capture only this article.
  await page.evaluate(() => Promise.race([
    Promise.all([
      document.fonts.ready,
      ...[...document.querySelectorAll('article.markdown-body img')].map(image =>
        typeof image.decode === 'function' ? image.decode().catch(() => {}) : Promise.resolve()),
    ]),
    new Promise(resolve => setTimeout(resolve, 8000)),
  ]));
  const captures = [];
  const targets = expected.formulas.filter(formula => formula.display && /^\s*=\s*$/m.test(formula.tex));
  for (let i = 0; i < targets.length; i++) {
    const target = targets[i];
    const matching = actual.formulas.find(formula =>
      formula.display && canonicalizeGitHubTeX(target.tex, formula.tex).equal);
    let locator;
    if (matching?.renderer_index !== null && matching?.renderer_index !== undefined) {
      locator = page.locator(articleSelector + ' math-renderer').nth(matching.renderer_index);
    } else if (matching?.latex_node_index !== null && matching?.latex_node_index !== undefined) {
      locator = page.locator(articleSelector + ' math[data-latex]').nth(matching.latex_node_index);
    } else {
      locator = page.locator(articleSelector + ' h1,' + articleSelector + ' h2,' +
        articleSelector + ' h3').filter({ hasText: target.heading }).first();
    }
    const file = path.join(outputFolder, 'github-math-01-equation-' + String(i + 1).padStart(2, '0') + '.png');
    try {
      await locator.waitFor({ state: 'visible', timeout: 5000 });
      await locator.evaluate(element => element.scrollIntoView({ block: 'center' }));
      await delay(200);
      await locator.evaluate(element => element.scrollIntoView({ block: 'center' }));
      await delay(200);
      await page.screenshot({ path: file });
      captures.push({ file: path.relative(root, file).split(path.sep).join('/'),
        local_line: target.line, heading: target.heading, matched_tex: Boolean(matching) });
    } catch (error) {
      captures.push({ local_line: target.line, error: error.message });
    }
  }
  return captures;
}

async function checkChapter(context, chapter, options, outputFolder) {
  const sourcePath = path.resolve(root, chapter.source);
  if (path.relative(root, sourcePath).startsWith('..')) throw new Error('Source escapes repository.');
  const expected = expectedMath(fs.readFileSync(sourcePath, 'utf8'), chapter.source);
  const url = 'https://github.com/' + repository + '/blob/' + encodeURIComponent(options.ref) + '/' +
    chapter.source.split('/').map(encodeURIComponent).join('/');
  const attempts = [];
  for (let attempt = 1; attempt <= 2; attempt++) {
    const page = await context.newPage();
    const pageErrors = [];
    page.on('pageerror', error => { if (pageErrors.length < 10) pageErrors.push(error.message); });
    try {
      const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
      const status = response?.status() ?? null;
      if (!response || status < 200 || status >= 400) {
        const error = new Error('HTTP ' + status + ' while opening GitHub blob.');
        error.http_status = status;
        throw error;
      }
      await page.locator(articleSelector).waitFor({ state: 'visible', timeout: 45000 });
      const actual = await stableArticle(page, expected);
      const comparison = compareMath(expected, actual);
      if (actual.stability.timed_out) {
        comparison.errors.push('article_math_did_not_stabilize_within_90_seconds');
        comparison.status = 'failed';
      }
      const result = {
        chapter: chapter.id, title: chapter.title, url, final_url: page.url(),
        http_status: status, expected, actual, ...comparison, page_errors: pageErrors,
        attempts: [...attempts, { attempt, http_status: status, status: comparison.status }],
      };
      if (options.capture && chapter.id === '01') {
        result.captures = await captureEquations(page, expected, actual, outputFolder);
        if (result.captures.some(capture => capture.error)) {
          result.status = 'failed';
          result.errors.push('requested_capture_failed');
        }
      }
      return result;
    } catch (error) {
      attempts.push({ attempt, status: 'network_or_page_failure',
        http_status: error.http_status ?? null, error: error.message, final_url: page.url() });
      const retryable = error.http_status === undefined || error.http_status === 429 || error.http_status >= 500;
      if (!retryable || attempt === 2) return {
        chapter: chapter.id, title: chapter.title, url, status: 'failed', expected,
        errors: ['github_page_unavailable_or_article_missing'], attempts, page_errors: pageErrors,
      };
    } finally {
      await page.close().catch(() => {});
    }
    await delay(1500 * attempt);
  }
}

export async function main(args = process.argv.slice(2)) {
  const options = optionsFrom(args);
  if (options.help) {
    console.log('Usage: node scripts/check-github-math.mjs [--chapter 01] [--ref main|SHA] [--proxy http://127.0.0.1:7890] [--capture] [--browser /path/to/chromium]');
    return;
  }
  const config = JSON.parse(fs.readFileSync(path.join(root, 'docs/book.json'), 'utf8'));
  const chapters = options.chapter ? config.chapters.filter(chapter => chapter.id === options.chapter) : config.chapters;
  if (!chapters.length) throw new Error('Chapter is not registered in docs/book.json.');
  const outputFolder = path.join(root, 'build/qa');
  fs.mkdirSync(outputFolder, { recursive: true });
  const output = path.join(outputFolder, 'github-math' + (options.chapter ? '-' + options.chapter : '') + '.json');
  const report = {
    checked_at: new Date().toISOString(), repository, ref: options.ref,
    expected_source: 'local working-tree Markdown, not a fetched remote source',
    tex_comparison: 'ordered formulas and display/inline type; whitespace normalized; source-guided LF row-break encoding only: exact source run 2 versus actual run 3 with identical line prefixes; each removed actual backslash is reported',
    proxy: options.browserProxy?.server ?? null, concurrency: 2, selected_chapters: chapters.length,
    errors: [], chapters: [],
  };
  let browser;
  try {
    browser = await launchBrowser(options.browser, { timeout: 30000,
      ...(options.browserProxy ? { proxy: options.browserProxy } : {}) });
    report.browser = browser.version();
    const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
    const results = new Array(chapters.length);
    let next = 0;
    await Promise.all(Array.from({ length: Math.min(2, chapters.length) }, async () => {
      while (next < chapters.length) {
        const index = next++;
        const chapter = chapters[index];
        try { results[index] = await checkChapter(context, chapter, options, outputFolder); }
        catch (error) {
          results[index] = { chapter: chapter.id, status: 'failed', errors: [error.message] };
        }
        console.log(chapter.id + ': ' + results[index].status);
      }
    }));
    report.chapters = results;
    await context.close();
  } catch (error) {
    report.errors.push(error.message);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  report.status = !report.errors.length && report.chapters.length === chapters.length &&
    report.chapters.every(chapter => chapter.status === 'passed') ? 'passed' : 'failed';
  report.summary = {
    checked_chapters: report.chapters.length,
    passed_chapters: report.chapters.filter(chapter => chapter.status === 'passed').length,
    failed_chapters: report.chapters.filter(chapter => chapter.status !== 'passed').length,
    expected_formulas: report.chapters.reduce((sum, chapter) => sum + (chapter.expected?.total || 0), 0),
    actual_formulas: report.chapters.reduce((sum, chapter) => sum + (chapter.actual?.total || 0), 0),
  };
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ status: report.status, output, ...report.summary }, null, 2));
  if (report.status !== 'passed') process.exitCode = 1;
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
