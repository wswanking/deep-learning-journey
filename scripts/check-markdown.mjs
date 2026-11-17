/** GitHub-compatible Markdown source checks. No rendering, writes, or network access.
 * Math source uses GitHub's `math` fence; ordinary code is intentionally opaque.
 * Syntax references: GitHub Docs /writing-mathematical-expressions and CommonMark 0.31.2.
 */
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import MarkdownIt from 'markdown-it';

export const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const parser = new MarkdownIt({ html: true, linkify: false, typographer: false });
const isMarkdown = name => /\.(?:md|markdown)$/i.test(name);
const isChapter = name => /(?:^|\/)docs\/chapters\/[^/]+\.md$/i.test(name.replaceAll('\\', '/'));
const contentLines = text => text ? text.split('\n').length - Number(text.endsWith('\n')) : 0;

function escaped(text, index) {
  let slashes = 0;
  while (index > 0 && text[--index] === '\\') slashes++;
  return slashes % 2 === 1;
}

/** Replace matched code spans with spaces, preserving source offsets and line numbers. */
function maskCodeSpans(text) {
  // Use UTF-16 indices like MarkdownIt/source maps, including Chinese/astral text.
  const chars = text.split('');
  for (let start = 0; start < text.length;) {
    if (text[start] !== '`' || escaped(text, start)) { start++; continue; }
    let end = start;
    while (text[end] === '`') end++;
    const length = end - start;
    let close = end;
    while (close < text.length) {
      close = text.indexOf('`', close);
      if (close < 0) break;
      let after = close;
      while (text[after] === '`') after++;
      if (after - close === length) {
        // GitHub's $`...`$ form is mathematics, not a literal code example.
        const githubMath = start > 0 && text[start - 1] === '$' && !escaped(text, start - 1) && text[after] === '$';
        if (!githubMath) for (let i = start; i < after; i++) if (chars[i] !== '\n') chars[i] = ' ';
        start = after;
        break;
      }
      close = after;
    }
    if (close < 0 || close >= text.length) start = end;
  }
  return chars.join('');
}

function stripContainers(line) {
  let value = line;
  while (/^ {0,3}>[ \t]?/.test(value)) value = value.replace(/^ {0,3}>[ \t]?/, '');
  value = value.replace(/^[ \t]*(?:[-+*]|\d+[.)])[ \t]+/, '');
  return value.trimStart();
}

function pipePositions(line) {
  const positions = [];
  for (let i = 0; i < line.length; i++) if (line[i] === '|' && !escaped(line, i)) positions.push(i);
  return positions;
}

/** GFM table separators apply even inside code/math; only an escaped pipe is safe. */
function tableCells(line) {
  const pipes = pipePositions(line);
  const cells = [];
  let start = 0;
  for (const i of pipes) { cells.push(line.slice(start, i)); start = i + 1; }
  cells.push(line.slice(start));
  if (pipes.length && !line.slice(0, pipes[0]).trim()) cells.shift();
  if (pipes.length && !line.slice(pipes.at(-1) + 1).trim()) cells.pop();
  return cells.map(cell => cell.trim());
}

function mathPipe(line) {
  const visible = maskCodeSpans(line);
  let opening = -1;
  for (let i = 0; i < visible.length; i++) {
    if (visible[i] !== '$' || escaped(visible, i)) continue;
    if (opening < 0) opening = i;
    else {
      for (let j = opening + 1; j < i; j++) if (visible[j] === '|' && !escaped(visible, j)) return j;
      opening = -1;
    }
  }
  return -1;
}

/**
 * Pure reusable audit. `chapter` defaults to true only for docs/chapters/*.md.
 * Errors use 1-based source line/column; stats are observations, not render QA.
 */
export function auditMarkdown(source, { filePath = '<markdown>', chapter = isChapter(filePath) } = {}) {
  const text = String(source).replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n');
  const lines = text.split('\n');
  const tokens = parser.parse(text, {});
  const errors = [];
  const stats = { headings: 0, h1Count: 0, mathFences: 0, ordinaryFences: 0, legacyMathBlocks: 0, tables: 0 };
  const add = (rule, line, message, column = 1) => errors.push({ rule, line, column, message });
  const opaque = new Set();
  const h1 = [];
  let previousLevel = 0;

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    if ((token.type === 'fence' || token.type === 'code_block' ||
        token.type === 'html_block' && token.content.trimStart().startsWith('<!--')) && token.map) {
      for (let line = token.map[0]; line < token.map[1]; line++) opaque.add(line);
    }
    if (token.type === 'fence') {
      const line = token.map[0] + 1;
      // A closed fence consumes its opener, its content lines, and its closer.
      // This also handles quoted/list fences and indented would-be closing markers.
      if (token.map[1] - token.map[0] !== contentLines(token.content) + 2) {
        add('unclosed-fence', line, '代码/数学围栏未关闭；关闭符号须与开头同类，长度不少于开头。');
      }
      const info = token.info.trim();
      if (/^math(?:\s|$)/i.test(info)) {
        stats.mathFences++;
        if (info !== 'math') add('math-fence-info', line, '数学围栏的语言标记须为小写 math，不附加其他参数。');
        if (!token.content.trim()) add('empty-math-fence', line, '数学围栏内容为空。');
        for (let i = 0; i < token.content.length - 1; i++) {
          if (token.content.slice(i, i + 2) === '$$' && !escaped(token.content, i)) {
            const before = token.content.slice(0, i);
            add('nested-math-delimiter', line + 1 + (before.match(/\n/g) || []).length,
              'math 围栏内直接写 TeX，不再嵌套 $$ 分隔符。', i - before.lastIndexOf('\n'));
            break;
          }
        }
      } else stats.ordinaryFences++;
    }
    if (token.type !== 'heading_open') continue;
    const level = Number(token.tag.slice(1));
    const line = (token.map?.[0] ?? 0) + 1;
    const label = maskCodeSpans(tokens[index + 1]?.content || '');
    stats.headings++;
    if (level === 1) h1.push(line);
    if (previousLevel && level > previousLevel + 1) {
      add('heading-level-jump', line, `标题从 H${previousLevel} 跳至 H${level}；请补全中间层级。`);
    }
    previousLevel = level;
    const setext = token.markup === '=' || token.markup === '-';
    if (/\$\$|\\(?:begin|end)\s*\{/.test(label) || setext && /\$|\\[A-Za-z]+|[_^]\{/.test(label)) {
      add('math-in-heading', setext ? token.map[1] : line,
        '数学内容被解析为标题；独立 = 行会成为 Setext 标题下划线，请将整段公式放入 math 围栏。');
    }
  }
  stats.h1Count = h1.length;
  if (chapter && h1.length !== 1) {
    add('chapter-h1-count', h1[1] || 1, `每章须恰有一个 H1；当前检测到 ${h1.length} 个。`);
  }

  const visible = lines.map((line, index) => opaque.has(index) ? ' '.repeat(line.length) : line).join('\n');
  const prose = maskCodeSpans(visible).split('\n');
  let legacyOpen = false;
  for (let i = 0; i < prose.length; i++) {
    const value = stripContainers(prose[i]);
    if (!value.startsWith('$$')) continue;
    if (legacyOpen) { legacyOpen = false; continue; }
    stats.legacyMathBlocks++;
    add('legacy-display-math', i + 1, '教材独立公式须使用 math 围栏；不继续使用独立 $$ 块。', prose[i].indexOf('$$') + 1);
    const remainder = value.slice(2);
    legacyOpen = ![...remainder.matchAll(/\$\$/g)].some(match => !escaped(remainder, match.index));
  }

  const parsedTableSeparators = new Set(tokens.filter(token => token.type === 'table_open' && token.map).map(token => token.map[0] + 1));
  const checkedMathRows = new Set();
  const checkMathRow = i => {
    if (checkedMathRows.has(i)) return;
    checkedMathRows.add(i);
    const value = stripContainers(lines[i]);
    const position = mathPipe(value);
    if (position >= 0) add('math-table-pipe', i + 1,
      '表格数学中的未转义 | 会被当成列分隔符；使用 \\mid、\\lvert/\\rvert、\\lVert/\\rVert，或按 GFM 转义竖线。',
      lines[i].length - value.length + position + 1);
  };
  for (let i = 1; i < lines.length; i++) {
    if (opaque.has(i) || opaque.has(i - 1)) continue;
    const separator = stripContainers(lines[i]);
    // A pipe in the delimiter distinguishes an explicit table from a normal Setext heading.
    if (!pipePositions(separator).length && !parsedTableSeparators.has(i)) continue;
    const delimiters = tableCells(separator);
    if (!delimiters.length || !delimiters.every(cell => /^:?-+:?$/.test(cell))) continue;
    const header = stripContainers(lines[i - 1]);
    if (!pipePositions(header).length) continue;
    const expected = delimiters.length;
    stats.tables++;
    if (tableCells(header).length !== expected) {
      add('table-column-count', i, `表头与分隔行列数不同：${tableCells(header).length} / ${expected}。`);
    }
    checkMathRow(i - 1);
    for (let row = i + 1; row < lines.length && !opaque.has(row); row++) {
      const value = stripContainers(lines[row]);
      if (!value.trim() || !pipePositions(value).length) break;
      const count = tableCells(value).length;
      // GFM intentionally permits omitted trailing empty cells, but discards extra cells.
      if (count > expected) add('table-column-count', row + 1,
        `表格行有 ${count} 列，表头只有 ${expected} 列；多出的单元格会被截去。`);
      checkMathRow(row);
    }
  }
  errors.sort((a, b) => a.line - b.line || a.column - b.column || a.rule.localeCompare(b.rule));
  return { filePath, status: errors.length ? 'failed' : 'passed', errors, stats };
}

function inRoot(root, file) {
  const relative = path.relative(root, file);
  return relative !== '..' && !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative);
}

/** Discover Markdown without following symlinks or walking dependency/build directories. */
export function collectMarkdownFiles({ root = projectRoot, mode = 'docs', paths = [] } = {}) {
  root = fs.realpathSync(root);
  const found = new Set();
  const add = file => {
    const real = fs.realpathSync(file);
    if (!inRoot(root, real)) throw new Error('Markdown 路径超出仓库：' + file);
    if (isMarkdown(real)) found.add(real);
  };
  const visit = target => {
    const info = fs.lstatSync(target);
    if (info.isSymbolicLink()) return;
    if (info.isFile()) { add(target); return; }
    if (!info.isDirectory()) return;
    if (!inRoot(root, fs.realpathSync(target))) throw new Error('扫描目录超出仓库：' + target);
    for (const entry of fs.readdirSync(target, { withFileTypes: true })) {
      if (['.git', '.venv', 'node_modules', 'build'].includes(entry.name)) continue;
      visit(path.join(target, entry.name));
    }
  };
  if (paths.length) for (const target of paths) visit(path.resolve(root, target));
  else if (mode === 'tracked') {
    let output;
    try { output = execFileSync('git', ['ls-files', '-z'], { cwd: root, encoding: 'utf8', windowsHide: true }); }
    catch (error) { throw new Error('无法读取已跟踪文件，请从Git工作区运行或使用 --docs。', { cause: error }); }
    for (const name of output.split('\0').filter(isMarkdown)) {
      const file = path.resolve(root, name);
      if (fs.existsSync(file) && !fs.lstatSync(file).isSymbolicLink()) add(file);
    }
  } else if (mode === 'docs') visit(path.join(root, 'docs'));
  else throw new Error('未知Markdown扫描模式：' + mode);
  return [...found].sort();
}

export function checkMarkdownFiles(files, { root = projectRoot } = {}) {
  const results = files.map(file => auditMarkdown(fs.readFileSync(file, 'utf8'), {
    filePath: path.relative(root, file).split(path.sep).join('/'),
  }));
  const errors = results.flatMap(result => result.errors.map(error => ({ filePath: result.filePath, ...error })));
  return { status: errors.length ? 'failed' : 'passed', files: results.length, errorCount: errors.length, errors, results };
}

export function runCli(argv = process.argv.slice(2)) {
  let root = projectRoot, mode, json = false;
  const paths = [];
  for (let i = 0; i < argv.length; i++) {
    const argument = argv[i];
    if (argument === '--help' || argument === '-h') {
      console.log('用法：node scripts/check-markdown.mjs [--docs | --tracked | 路径...] [--root 目录] [--json]\n默认扫描docs；只读检查，发现源格式错误退出1，参数/读取失败退出2。');
      return 0;
    }
    if (argument === '--json') json = true;
    else if (argument === '--root') {
      if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error('--root 缺少目录');
      root = path.resolve(argv[++i]);
    } else if (argument === '--tracked' || argument === '--docs') {
      if (mode) throw new Error('--docs 与 --tracked 只能选择一个');
      mode = argument.slice(2);
    } else if (argument.startsWith('-')) throw new Error('未知参数：' + argument);
    else paths.push(argument);
  }
  if (mode && paths.length) throw new Error('显式路径与 --docs/--tracked 不能同时使用');
  const files = collectMarkdownFiles({ root, mode: mode || 'docs', paths });
  if (!files.length) throw new Error('未找到可检查的Markdown文件');
  const report = checkMarkdownFiles(files, { root });
  if (json) console.log(JSON.stringify(report, null, 2));
  else {
    for (const error of report.errors) console.log(`${error.filePath}:${error.line}:${error.column} [${error.rule}] ${error.message}`);
    console.log(`Markdown源格式检查：${report.files}文件，${report.errorCount}错误，${report.status}。`);
  }
  return report.errorCount ? 1 : 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { process.exitCode = runCli(); }
  catch (error) { console.error('Markdown检查失败：' + error.message); process.exitCode = 2; }
}
