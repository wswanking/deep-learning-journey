import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { auditMarkdown, collectMarkdownFiles, checkMarkdownFiles, projectRoot } from '../scripts/check-markdown.mjs';
import { compileBook, build } from '../scripts/build.mjs';

const chapter = 'docs/chapters/00-test.md';
const audit = text => auditMarkdown(text, { filePath: chapter });
const rules = result => result.errors.map(error => error.rule);
const matrix = String.raw`\mathbf z=\mathbf W\mathbf x+\mathbf b
=
\begin{bmatrix}4+0.5\\-1-1\end{bmatrix}
=
\begin{bmatrix}4.5\\-2\end{bmatrix}`;
const fence = (body, language = 'math') => '```' + language + '\n' + body + '\n```';

function fixture(t, text = '# 测试章\n\n## 定义\n\n正文。') {
  const parent = path.join(projectRoot, 'build');
  fs.mkdirSync(parent, { recursive: true });
  const root = fs.mkdtempSync(path.join(parent, 'test-markdown-'));
  fs.mkdirSync(path.join(root, 'docs/chapters'), { recursive: true });
  fs.mkdirSync(path.join(root, 'docs/styles'), { recursive: true });
  fs.writeFileSync(path.join(root, chapter), text);
  fs.writeFileSync(path.join(root, 'docs/styles/book.css'), 'body { color: #172c44; }\n');
  t.after(() => {
    const absoluteParent = fs.realpathSync(parent), absoluteRoot = fs.realpathSync(root);
    const relative = path.relative(absoluteParent, absoluteRoot);
    assert.ok(!path.isAbsolute(relative) && relative.startsWith('test-markdown-') && !relative.includes(path.sep));
    fs.rmSync(absoluteRoot, { recursive: true, force: true });
  });
  return root;
}
const config = { title: '测试书', subtitle: '数学源回归', language: 'zh-CN' };
const entries = [{ id: '00', title: '测试章', source: chapter, draft: false }];

// Screenshot regression: a bare '=' in a legacy display becomes a Setext H1.
test('旧$$矩阵内独立等号误成标题：给出源行与数学标题诊断', () => {
  const source = '# 测试章\n\n## 手算\n\n$$\n' + matrix + '\n$$\n';
  const result = audit(source);
  assert.ok(rules(result).includes('legacy-display-math'));
  assert.ok(rules(result).includes('math-in-heading'));
  assert.ok(rules(result).includes('chapter-h1-count'));
  assert.equal(result.stats.legacyMathBlocks, 1);
  assert.ok(result.errors.filter(error => error.rule === 'math-in-heading').every(error => source.split('\n')[error.line - 1] === '='));
});

test('没有Setext触发的旧$$块仍违反教材迁移规范', () => {
  const result = audit('# 测试章\n\n$$\nx^2+y^2\n$$\n');
  assert.deepEqual(rules(result), ['legacy-display-math']);
  assert.equal(result.stats.legacyMathBlocks, 1);
  assert.ok(rules(audit('# 测试章\n\n$$x^2$$')).includes('legacy-display-math'));
});

test('合法math矩阵保留TeX独立等号，且只有一个章标题', () => {
  const result = audit('# 测试章\n\n## 手算\n\n' + fence(matrix));
  assert.deepEqual(result.errors, []);
  assert.equal(result.stats.mathFences, 1);
  assert.equal(result.stats.h1Count, 1);
});

test('普通代码、缩进代码与外四反引号Markdown演示中的旧语法免查', () => {
  const source = '# 测试章\n\n' + fence('# 假标题\n$$\n=\n\\thisCommandDoesNotExist{}', 'python') +
    '\n\n    $$\n    # 也是代码\n\n````markdown\n# 假章节\n```math\n$$\nx=y\n$$\n```\n````\n';
  const result = audit(source);
  assert.deepEqual(result.errors, []);
  assert.equal(result.stats.mathFences, 0);
  assert.equal(result.stats.ordinaryFences, 2);
  assert.equal(result.stats.h1Count, 1);
});

test('行内代码示例与HTML注释不误报，GitHub行内math仍允许', () => {
  assert.deepEqual(audit('# 测试章\n\n## `$L_2$`语法与$L_2$范数\n\n`$$`是旧分隔符。\n\n<!--\n$$\n# 注释\n-->').errors, []);
  assert.deepEqual(audit('# 测试章\n\n`多行代码\n$$\n结束`').errors, []);
});

for (const [name, block] of [
  ['只有开头', '```python'], ['末尾未关闭', '```text\nx'],
  ['错误符号关闭', '```text\nx\n~~~'], ['过短关闭', '````text\nx\n```'],
  ['四空格不是关闭', '```text\nx\n    ```'],
  ['引用容器中未关闭', '> ```text\n> x\n\n后文。'],
]) test('未关闭围栏：' + name, () => {
  assert.ok(rules(audit('# 测试章\n\n' + block)).includes('unclosed-fence'));
});

test('同类更长关闭、tilde围栏、列表与引用中的关闭合法', () => {
  const text = '# 测试章\n\n```text\nx\n````\n\n~~~text\ny\n~~~~\n\n- 示例\n\n  ```math\n  x=1\n  ```\n\n> ```math\n> y=2\n> ```';
  assert.deepEqual(audit(text).errors, []);
  assert.equal(audit(text).stats.mathFences, 2);
});

test('数学围栏不能为空、嵌套$$或附加未支持语言参数', () => {
  assert.ok(rules(audit('# 测试章\n\n' + fence(' \n '))).includes('empty-math-fence'));
  const nested = audit('# 测试章\n\n' + fence('x=1\n$$\ny=2'));
  assert.ok(rules(nested).includes('nested-math-delimiter'));
  assert.equal(nested.errors.find(error => error.rule === 'nested-math-delimiter').line, 5);
  assert.ok(rules(audit('# 测试章\n\n' + fence('x=1', 'Math'))).includes('math-fence-info'));
  assert.ok(rules(audit('# 测试章\n\n' + fence('x=1', 'math title'))).includes('math-fence-info'));
  assert.deepEqual(audit('# 测试章\n\n' + fence(String.raw`\text{\$\$}`)).errors, []);
});

test('每章一个H1和标题层级检查，非章节文档不强制H1', () => {
  assert.ok(rules(audit('## 缺章标题')).includes('chapter-h1-count'));
  assert.ok(rules(audit('# 一\n\n# 二')).includes('chapter-h1-count'));
  assert.ok(rules(audit('# 一\n\n### 跳级')).includes('heading-level-jump'));
  assert.deepEqual(audit('# 一\n\n## 二\n\n### 三\n\n## 四').errors, []);
  assert.deepEqual(auditMarkdown('## 片段\n\n正文。', { filePath: 'notes.md' }).errors, []);
  assert.deepEqual(audit('# 标题\n\n普通Setext小节\n---\n\n## `\\begin`命令').errors, []);
});

test('没有美元分隔符的裸TeX也不能被Setext误成标题', () => {
  const result = audit('# 测试章\n\n\\mathbf{x}=a\n=\nb');
  assert.ok(rules(result).includes('math-in-heading'));
});

const table = row => '# 测试章\n\n| 名称 | 数学 |\n| --- | --- |\n' + row + '\n';
test('数学绝对值/条件概率竖线会拆列，定位具体源行', () => {
  const result = audit(table('| 范数 | $|x|$ |'));
  assert.ok(rules(result).includes('math-table-pipe'));
  assert.ok(rules(result).includes('table-column-count'));
  assert.equal(result.errors.find(error => error.rule === 'math-table-pipe').line, 5);
  const cancelled = '# 测试章\n\n| A | B | C |\n| --- | --- | --- |\n| $p(x|y)$ | 条件概率 |';
  assert.ok(rules(audit(cancelled)).includes('math-table-pipe'));
  assert.ok(!rules(audit(cancelled)).includes('table-column-count'));
  assert.ok(rules(audit('# 测试章\n\n| 范数 |\n---\n| $|x|$ |')).includes('math-table-pipe'));
});

test('数学竖线命令或GFM正确转义保留表格列', () => {
  for (const expression of [String.raw`$p(x\mid y)$`, String.raw`$\lvert x\rvert$`, String.raw`$\lVert x\rVert$`, String.raw`$\|x\|$`]) {
    assert.deepEqual(audit(table('| 正常 | ' + expression + ' |')).errors, [], expression);
  }
  assert.deepEqual(audit(table('| 源代码 | `a\\|b` |')).errors, []);
  assert.ok(rules(audit(table(String.raw`| 错误转义 | $\\|x\\|$ |`))).includes('math-table-pipe'));
});

test('普通GFM短行可省略尾部空列，代码演示表不参与检查', () => {
  assert.deepEqual(audit(table('| 只有一列 |')).errors, []);
  assert.deepEqual(audit('# 测试章\n\n' + fence('| A | B |\n| --- | --- |\n| $|x|$ | x |', 'markdown')).errors, []);
});

test('发现文件与可复用多文件审计，只读扫描docs和显式路径', t => {
  const root = fixture(t);
  fs.writeFileSync(path.join(root, 'docs', 'notes.MD'), '# 备注\n');
  fs.mkdirSync(path.join(root, 'docs', 'node_modules'), { recursive: true });
  fs.writeFileSync(path.join(root, 'docs', 'node_modules', 'ignored.md'), '```');
  assert.equal(collectMarkdownFiles({ root }).length, 2);
  assert.equal(collectMarkdownFiles({ root, paths: [chapter] }).length, 1);
  const result = checkMarkdownFiles(collectMarkdownFiles({ root }), { root });
  assert.equal(result.status, 'passed');
  assert.equal(result.files, 2);
});

test('CLI机器可读输出、错误退出码与参数失败', t => {
  const root = fixture(t);
  const script = path.join(projectRoot, 'scripts/check-markdown.mjs');
  const run = extra => spawnSync(process.execPath, [script, '--root', root, ...extra], { encoding: 'utf8', windowsHide: true });
  let child = run(['--docs', '--json']);
  assert.equal(child.status, 0, child.stderr);
  assert.equal(JSON.parse(child.stdout).status, 'passed');
  fs.writeFileSync(path.join(root, chapter), '# 测试章\n\n$$\nx=1\n$$\n');
  child = run([chapter, '--json']);
  assert.equal(child.status, 1);
  assert.equal(JSON.parse(child.stdout).errors[0].rule, 'legacy-display-math');
  child = run(['--docs', '--tracked']);
  assert.equal(child.status, 2);
  assert.match(child.stderr, /只能选择一个/);
});

test('compileBook数学围栏保留矩阵、等号及严格公式计数', t => {
  const root = fixture(t, '# 测试章\n\n## 手算\n\n$x^2$\n\n' + fence(matrix));
  const result = compileBook(root, config, entries);
  assert.equal(result.metadata.math_expressions, 2);
  assert.match(result.html, /class="math-block"/);
  assert.match(result.html, /class="katex/);
  assert.doesNotMatch(result.html, /language-math/);
  assert.equal((result.html.match(/<h1\b/g) || []).length, 2); // book title + chapter title
});

test('compileBook对math围栏中的非法TeX仍严格抛错', t => {
  const root = fixture(t, '# 测试章\n\n' + fence(String.raw`\thisCommandDoesNotExist{x}`));
  assert.throws(() => compileBook(root, config, entries), /KaTeX|Undefined|ParseError/);
});

test('compileBook支持列表/引用数学，外四反引号示例保持代码', t => {
  const text = '# 测试章\n\n- 列表\n\n  ```math\n  x=1\n  ```\n\n> ```math\n> y=2\n> ```\n\n````markdown\n```math\n\\thisCommandDoesNotExist{}\n```\n````';
  const root = fixture(t, text);
  const result = compileBook(root, config, entries);
  assert.equal(result.metadata.math_expressions, 2);
  assert.equal((result.html.match(/class="math-block"/g) || []).length, 2);
  assert.match(result.html, /thisCommandDoesNotExist/);
  assert.deepEqual(audit(text).errors, []);
});

test('compileBook普通代码中的美元符号和非法TeX保持代码', t => {
  const root = fixture(t, '# 测试章\n\n' + fence('$$\n\\thisCommandDoesNotExist{}\n# 假标题', 'text'));
  const result = compileBook(root, config, entries);
  assert.equal(result.metadata.math_expressions, 0);
  assert.match(result.html, /\$\$/);
});

test('math围栏含非法TeX导致构建失败时不覆盖上一版HTML', async t => {
  const root = fixture(t, '# 测试章\n\n' + fence(String.raw`\thisCommandDoesNotExist{x}`));
  fs.writeFileSync(path.join(root, 'docs/book.json'), JSON.stringify({ ...config, chapters: entries }));
  fs.writeFileSync(path.join(root, 'docs/chapter-plan.md'), '### [x] 00：测试章\n');
  const output = path.join(root, 'build/html/deep-learning-journey.html');
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, 'previous-good-math-output');
  await assert.rejects(build({ root, format: 'html' }), /KaTeX|Undefined|ParseError/);
  assert.equal(fs.readFileSync(output, 'utf8'), 'previous-good-math-output');
});

for (const [name, tex] of [
  ['转置与多字母下标', String.raw`(\mathbf A^\top)_{ji}=a_{ij}`],
  ['含双反斜线换行的行内矩阵', String.raw`\begin{bmatrix}1&2\\3&4\end{bmatrix}`],
  ['没有TeX命令的普通坐标元组', '(2,1,0)'],
]) test('GitHub行内公式与本地普通美元写法逐字节一致：' + name, t => {
  const plain = '# 测试章\n\n公式： $' + tex + '$。\n';
  const github = '# 测试章\n\n公式： $`' + tex + '`$。\n';
  const legacy = compileBook(fixture(t, plain), config, entries);
  const result = compileBook(fixture(t, github), config, entries);
  assert.ok(Buffer.from(result.html, 'utf8').equals(Buffer.from(legacy.html, 'utf8')), '官方行内写法与普通美元写法的HTML字节不同');
  assert.deepEqual(result.metadata, legacy.metadata);
  assert.equal(result.metadata.math_expressions, 1);
  const article = result.html.match(/<article class="chapter">([\s\S]*?)<\/article>/)?.[1];
  assert.ok(article);
  assert.match(article, /class="katex/);
  assert.doesNotMatch(article, /`|&#96;|&#x60;|<em>/i);
  assert.deepEqual(audit(plain).errors, []);
  assert.deepEqual(audit(github).errors, []);
  if (name.includes('矩阵')) {
    const mathml = article.match(/<math[\s\S]*?<\/math>/)?.[0];
    assert.ok(mathml);
    assert.equal((mathml.match(/<mtr>/g) || []).length, 2);
    assert.equal((mathml.match(/<mtd>/g) || []).length, 4);
  }
});

test('普通围栏内的GitHub行内公式演示仍是代码', t => {
  const example = '$`' + String.raw`\thisCommandDoesNotExist{x}` + '`$';
  const root = fixture(t, '# 测试章\n\n' + fence(example, 'text'));
  const result = compileBook(root, config, entries);
  assert.equal(result.metadata.math_expressions, 0);
  assert.ok(result.html.includes(example));
  assert.deepEqual(audit('# 测试章\n\n' + fence(example, 'text')).errors, []);
});

test('GitHub数学围栏诊断operatorname及裸小于号，保留精确源位置', () => {
  const source = '# 测试章\n\n' + fence(String.raw`\operatorname{rank}(A)
\operatorname*{argmin}_{r+1<L_i} f(r)`);
  const result = audit(source);
  assert.deepEqual(rules(result), [
    'github-unsupported-operatorname', 'github-unsupported-operatorname', 'github-math-html-angle',
  ]);
  assert.deepEqual(result.errors.map(({ line, column }) => ({ line, column })), [
    { line: 4, column: 1 }, { line: 5, column: 1 }, { line: 5, column: 28 },
  ]);
  assert.match(result.errors[0].message, /mathop/);
  assert.match(result.errors[2].message, /\\lt/);
});

test('GitHub保护行内公式诊断operatorname和HTML截断，支持同行多个公式', () => {
  const source = '# 测试章\n\n秩 $`' + String.raw`\operatorname{rank}(A)` + '`$；条件 $`r+1<L_i`$。\n' +
    '极值 $`' + String.raw`\operatorname*{argmin}_x f(x)` + '`$。';
  const result = audit(source);
  assert.deepEqual(rules(result), [
    'github-unsupported-operatorname', 'github-math-html-angle', 'github-unsupported-operatorname',
  ]);
  for (const error of result.errors) {
    const tail = source.split('\n')[error.line - 1].slice(error.column - 1);
    assert.ok(error.rule === 'github-math-html-angle' ? tail.startsWith('<') : tail.startsWith('\\operatorname'));
  }
});

test('引用与列表中的数学兼容诊断保留容器前缀后的源列', () => {
  const source = '# 测试章\n\n> ```math\n> x<y\n> ```\n\n- 条件\n\n  ```math\n  ' + String.raw`\operatorname{rank}(A)` + '\n  ```';
  const result = audit(source);
  assert.deepEqual(result.errors.map(({ rule, line, column }) => ({ rule, line, column })), [
    { rule: 'github-math-html-angle', line: 4, column: 4 },
    { rule: 'github-unsupported-operatorname', line: 10, column: 3 },
  ]);
});

test('GitHub安全替代mathop、limits及lt在块和行内均合法且能严格渲染', t => {
  const block = String.raw`\mathop{\mathrm{argmin}}\limits_{r+1\lt L_i} f(r)`;
  const inline = String.raw`\mathop{\mathrm{rank}}\nolimits(A),\quad a\lt b,\quad\langle x,y\rangle`;
  const source = '# 测试章\n\n' + fence(block) + '\n\n秩与内积 $`' + inline + '`$。';
  assert.deepEqual(audit(source).errors, []);
  const result = compileBook(fixture(t, source), config, entries);
  assert.equal(result.metadata.math_expressions, 2);
  assert.match(result.html, /class="math-block"/);
});

test('数学兼容护栏不检查普通代码、行内语法演示、注释或旧行内兼容写法', () => {
  const bad = String.raw`\operatorname*{argmin}_{r+1<L_i}`;
  const source = '# 测试章\n\n' + fence(bad + '\n$`' + bad + '`$', 'text') +
    '\n\n````markdown\n```math\n' + bad + '\n```\n````\n\n' +
    '    ' + bad + '\n\n' +
    '普通代码 `' + bad + '`，官方语法演示 ``$`' + bad + '`$``。\n\n' +
    '<!-- $`' + bad + '`$ -->\n\n' +
    '兼容旧行内 $' + bad + '$，不在本护栏迁移范围。';
  assert.deepEqual(audit(source).errors, []);
  // A row-break command followed by letters is not an operatorname control sequence.
  assert.deepEqual(audit('# 测试章\n\n' + fence(String.raw`a\\operatorname`)).errors, []);
});
