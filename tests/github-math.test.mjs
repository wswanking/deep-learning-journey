import test from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright-core';
import {
  canonicalizeGitHubTeX, expectedMath, compareMath, normalizeTeX, collectArticleMath,
} from '../scripts/check-github-math.mjs';

const bs = '\\';
const two = bs.repeat(2);
const three = bs.repeat(3);
const four = bs.repeat(4);
const matrix = '\\begin{bmatrix}\n1&2' + two + '\n3&4\n\\end{bmatrix}\n';
const escaped = matrix.replace(two + '\n', three + '\n').trimEnd();
const fence = String.fromCharCode(96).repeat(3);

function expected(formulas) {
  return expectedMath('# Fixture\n\n' + formulas.map(tex =>
    fence + 'math\n' + tex.trimEnd() + '\n' + fence + '\n').join('\n'));
}
function actual(formulas) {
  return {
    article_found: true, h1: ['Fixture'], total: formulas.length,
    display: formulas.length, inline: 0, incomplete_renderers: 0,
    duplicate_latex_nodes: 0, raw_tex_headings: [], raw_math_markers: [],
    unrendered_math_code_blocks: [], math_errors: [],
    formulas: formulas.map((tex, index) => ({ index, display: true, tex })),
  };
}

test('recognizes only source-guided two-to-three backslashes before LF and preserves LF', () => {
  const pair = canonicalizeGitHubTeX(matrix, escaped);
  assert.equal(pair.equal, true);
  assert.equal(pair.canonical_actual_tex, matrix.trimEnd());
  assert.equal(pair.canonical_actual_tex.split('\n').length, escaped.split('\n').length);
  assert.deepEqual(pair.events, [{
    tex_line: 2, source_column: 4, actual_column: 4, removed_actual_column: 6,
    source_backslashes: 2, actual_backslashes: 3,
  }]);
});

test('comparison and inventory share the rule and record the physical source position', () => {
  const source = expected([matrix]);
  const result = compareMath(source, actual([escaped]));
  assert.equal(result.status, 'passed');
  assert.deepEqual(result.missing_tex, []);
  assert.deepEqual(result.unexpected_tex, []);
  assert.equal(result.tex_normalization.transformed_formulas, 1);
  assert.equal(result.tex_normalization.removed_control_spaces, 1);
  assert.equal(result.tex_normalization.formulas[0].events[0].source_line,
    source.formulas[0].line + 2);
});

test('genuine source triple backslashes are not compressed or accepted as two/four', () => {
  const source = matrix.replace(two + '\n', three + '\n');
  assert.equal(canonicalizeGitHubTeX(source, matrix).equal, false);
  assert.equal(canonicalizeGitHubTeX(source, source.replace(three + '\n', four + '\n')).equal, false);
  const unchanged = canonicalizeGitHubTeX(source, source);
  assert.equal(unchanged.equal, true);
  assert.deepEqual(unchanged.events, []);
});

test('four actual backslashes stay a mismatch in comparison and inventory', () => {
  const remote = matrix.replace(two + '\n', four + '\n');
  const pair = canonicalizeGitHubTeX(matrix, remote);
  assert.equal(pair.equal, false);
  assert.deepEqual(pair.events, []);
  const result = compareMath(expected([matrix]), actual([remote]));
  assert.equal(result.status, 'failed');
  assert.equal(result.missing_tex.length, 1);
  assert.equal(result.unexpected_tex.length, 1);
});

test('non-line-terminal and final unterminated runs are never rewritten', () => {
  for (const [source, remote] of [
    ['a' + two + ' b', 'a' + three + ' b'],
    ['a' + two, 'a' + three],
    ['a' + two + ' \nb', 'a' + three + ' \nb'],
  ]) {
    const pair = canonicalizeGitHubTeX(source, remote);
    assert.equal(pair.equal, false);
    assert.deepEqual(pair.events, []);
  }
});

test('changed line prefix blocks row-break normalization', () => {
  const remote = escaped.replace('1&2', '1&9');
  const pair = canonicalizeGitHubTeX(matrix, remote);
  assert.equal(pair.equal, false);
  assert.deepEqual(pair.events, []);
});

test('changed command or later numeric value remains a failure after an eligible row break', () => {
  for (const remote of [
    escaped.replace('\\end{bmatrix}', '\\end{pmatrix}'),
    escaped.replace('3&4', '3&9'),
  ]) {
    const pair = canonicalizeGitHubTeX(matrix, remote);
    assert.equal(pair.equal, false);
    assert.equal(pair.events.length, 1);
    assert.equal(compareMath(expected([matrix]), actual([remote])).status, 'failed');
  }
});

test('ordinary whitespace normalization remains unchanged without slash rewriting', () => {
  const pair = canonicalizeGitHubTeX('x + y\n', '  x  +\n  y  ');
  assert.equal(pair.equal, true);
  assert.deepEqual(pair.events, []);
  assert.equal(pair.normalized_source, normalizeTeX('x + y\n'));
  assert.equal(canonicalizeGitHubTeX('x+1', 'x+2').equal, false);
});

test('line prefixes must match exactly even when ordinary whitespace would be normalized', () => {
  const pair = canonicalizeGitHubTeX(matrix, escaped.replace('1&2', ' 1&2'));
  assert.equal(pair.equal, false);
  assert.deepEqual(pair.events, []);
});

test('inventory handles ambiguous repeated source patterns but ordered comparison still fails', () => {
  const sourceTriple = matrix.replace(two + '\n', three + '\n');
  const result = compareMath(expected([matrix, sourceTriple]), actual([sourceTriple, matrix]));
  assert.equal(result.status, 'failed');
  assert.deepEqual(result.missing_tex, []);
  assert.deepEqual(result.unexpected_tex, []);
});

test('display type mismatch is still rejected', () => {
  const remote = actual([escaped]);
  remote.display = 0;
  remote.inline = 1;
  remote.formulas[0].display = false;
  assert.equal(compareMath(expected([matrix]), remote).status, 'failed');
});

test('GitHub inline backticks preserve inner TeX and ordinary code stays code', () => {
  const tick = String.fromCharCode(96);
  const tex = '(\\mathbf A^\\top)_{ji}=a_{ij}';
  const source = '# Fixture\n\n$' + tick + tex + tick + '$ and ' + tick + '$ignored$' + tick + '.\n';
  const parsed = expectedMath(source);
  assert.equal(parsed.total, 1);
  assert.equal(parsed.formulas[0].tex, tex);
});

test('real renderer flash-error DOM is collected with its actual text and fails validation', async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ offline: true });
    await context.route('**/*', route => route.abort());
    const page = await context.newPage();
    await page.setContent(
      '<article class="markdown-body"><h1>Fixture</h1>' +
      '<math-renderer class="js-display-math">' +
      '<math xmlns="http://www.w3.org/1998/Math/MathML" data-latex="x=1"><mi>x</mi></math>' +
      '<div class="flash flash-error"><strong>Unable to render rich display</strong>' +
      '<p>The following macros are not allowed: operatorname</p></div></math-renderer>' +
      '<div class="flash-error">Extra open brace or missing close brace</div></article>'
    );
    const snapshot = await page.evaluate(collectArticleMath);
    assert.equal(snapshot.total, 1);
    assert.equal(snapshot.math_errors.length, 2);
    assert.match(snapshot.math_errors[0].text, /The following macros are not allowed: operatorname/);
    assert.match(snapshot.math_errors[1].text, /Extra open brace or missing close brace/);
    const result = compareMath(expected(['x=1']), snapshot);
    assert.equal(result.status, 'failed');
    assert.ok(result.errors.includes('math_rendering_errors'));
    await context.close();
  } finally {
    await browser.close();
  }
});

test('visible math control words fail, while annotations, ordinary code and a bare slash do not', async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ offline: true });
    await context.route('**/*', route => route.abort());
    const page = await context.newPage();
    const safe = '<article class="markdown-body"><h1>Fixture</h1>' +
      '<pre><code>\\displaylimits \\operatorname</code></pre>' +
      '<math-renderer class="js-display-math">' +
      '<math xmlns="http://www.w3.org/1998/Math/MathML" data-latex="x=1">' +
      '<semantics><mrow><mi>x</mi><mo>=</mo><mn>1</mn><mtext>\\</mtext></mrow>' +
      '<annotation encoding="application/x-tex">\\displaylimits \\operatorname</annotation>' +
      '<annotation-xml encoding="application/xhtml+xml"><span>\\ignored</span></annotation-xml>' +
      '</semantics></math></math-renderer></article>';
    await page.setContent(safe);
    const clean = await page.evaluate(collectArticleMath);
    assert.deepEqual(clean.unrendered_math_commands, []);
    assert.equal(compareMath(expected(['x=1']), clean).status, 'passed');
    await page.evaluate(() => {
      const math = document.querySelector('math-renderer math');
      const text = document.createElementNS('http://www.w3.org/1998/Math/MathML', 'mtext');
      text.textContent = '\\displaylimits';
      math.append(text);
    });
    const broken = await page.evaluate(collectArticleMath);
    assert.equal(broken.math_errors.length, 0);
    assert.equal(broken.unrendered_math_commands.length, 1);
    assert.deepEqual(broken.unrendered_math_commands[0].commands, ['\\displaylimits']);
    assert.match(broken.unrendered_math_commands[0].visible_text, /\\displaylimits/);
    assert.equal(broken.unrendered_math_commands[0].renderer_index, 0);
    const result = compareMath(expected(['x=1']), broken);
    assert.equal(result.status, 'failed');
    assert.ok(result.errors.includes('unrendered_math_commands'));
    await context.close();
  } finally {
    await browser.close();
  }
});
