import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { browserCandidates, launchBrowser } from '../scripts/browser.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const linux = process.platform === 'linux';

test('Windows retains Edge then Chrome discovery', () => {
  assert.deepEqual(browserCandidates({ platform: 'win32', env: {} }), [{ channel: 'msedge' }, { channel: 'chrome' }]);
});
test('explicit browser selection does not silently fall back', async () => {
  const calls = [];
  await assert.rejects(launchBrowser('/missing/chrome', {}, {env: {}, launch: async o => { calls.push(o); throw new Error('not found'); }}), /not found/);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].executablePath, '/missing/chrome');
});
test('environment selection and launch options are preserved', async () => {
  const result = await launchBrowser(undefined, { proxy: { server: 'http://localhost:9999' } }, {
    env: { BROWSER_EXECUTABLE: '/chosen/chromium' }, launch: async o => o,
  });
  assert.equal(result.executablePath, '/chosen/chromium');
  assert.equal(result.proxy.server, 'http://localhost:9999');
  assert.equal(result.headless, true);
});
test('Linux PATH detection handles spaces and duplicate entries', {skip: !linux}, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'linux browser '));
  try {
    const executable = path.join(dir, 'chromium');
    fs.writeFileSync(executable, '#!/bin/sh\nexit 0\n', { mode: 0o755 });
    const candidates = browserCandidates({ platform: 'linux', env: { PATH: `${dir}:${dir}` } });
    assert.equal(candidates.filter(c => c.executablePath === executable).length, 1);
    const result = await launchBrowser(undefined, {}, { platform: 'linux', env: { PATH: dir }, launch: async o => {
      if (o.channel) throw new Error('not installed');
      return o;
    }});
    assert.equal(result.executablePath, executable);
  } finally { fs.rmSync(dir, { recursive: true }); }
});
test('all unavailable browsers fail with actionable guidance', async () => {
  await assert.rejects(launchBrowser(undefined, {}, {platform: 'win32', env: {}, launch: async () => { throw new Error('missing'); }}), /BROWSER_EXECUTABLE/);
});
for (const script of ['build', 'setup', 'check-environment', 'verify']) {
  test(`${script}.sh has LF endings and valid Bash syntax`, {skip: !linux}, () => {
    const file = path.join(root, 'scripts', script + '.sh');
    assert.equal(fs.readFileSync(file, 'utf8').includes('\r'), false);
    assert.equal(spawnSync('bash', ['-n', file]).status, 0);
    assert.equal(spawnSync('bash', [file, '--help'], { cwd: os.tmpdir(), encoding: 'utf8' }).status, 0);
  });
}
test('shell wrappers reject invalid arguments', {skip: !linux}, () => {
  for (const script of ['build', 'setup', 'verify']) {
    assert.notEqual(spawnSync('bash', [path.join(root, 'scripts', script + '.sh'), '--typo'], {cwd: os.tmpdir()}).status, 0);
  }
});
test('environment report is parseable and never overwrites evidence', {skip: !linux}, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'env-report-'));
  const file = path.join(dir, 'report.json');
  try {
    const script = path.join(root, 'scripts/check-environment.py');
    const result = spawnSync('python3', ['-B', script, '--output', file], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    const report = JSON.parse(result.stdout);
    assert.equal(report.platform, 'Linux');
    assert.equal(report.network_probed, false);
    assert.equal(report.cpu_smoke.status, 'not_run');
    const before = fs.readFileSync(file, 'utf8');
    assert.notEqual(spawnSync('python3', ['-B', script, '--output', file]).status, 0);
    assert.equal(fs.readFileSync(file, 'utf8'), before);
  } finally { fs.rmSync(dir, { recursive: true }); }
});

test('setup --cpu checks its venv even when caller exports PYTHON', {skip: !linux}, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'linux setup '));
  try {
    fs.mkdirSync(path.join(dir, 'scripts'));
    fs.mkdirSync(path.join(dir, 'bin'));
    fs.mkdirSync(path.join(dir, '.venv-linux/bin'), {recursive: true});
    fs.copyFileSync(path.join(root, 'scripts/setup.sh'), path.join(dir, 'scripts/setup.sh'));
    const fake = '#!/usr/bin/env bash\nexit 0\n';
    for (const name of ['bin/npm', 'bin/system-python', '.venv-linux/bin/python']) {
      fs.writeFileSync(path.join(dir, name), fake, {mode: 0o755});
    }
    fs.writeFileSync(path.join(dir, 'scripts/check-environment.sh'),
      '#!/usr/bin/env bash\n[[ "$PYTHON" == "$PWD/.venv-linux/bin/python" ]] || exit 42\n', {mode: 0o755});
    const result = spawnSync('bash', [path.join(dir, 'scripts/setup.sh'), '--cpu'], {
      cwd: os.tmpdir(), encoding: 'utf8', env: {...process.env, PYTHON: path.join(dir, 'bin/system-python'), PATH: path.join(dir, 'bin') + path.delimiter + process.env.PATH},
    });
    assert.equal(result.status, 0, result.stderr);
    assert.ok(fs.existsSync(path.join(dir, 'build/environment/linux-freeze.txt')));
  } finally { fs.rmSync(dir, {recursive: true}); }
});

test('repository-wide inventory includes examples and rejects direct browser launches', {skip: !linux}, () => {
  const code = `import importlib.util,tempfile,pathlib
s=importlib.util.spec_from_file_location('audit',${JSON.stringify(path.join(root, 'scripts/audit-scripts.py'))});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
files=m.inventory(pathlib.Path(${JSON.stringify(root)}))
assert len([p for p in files if p.name=='run.py'])==30
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d)/'bad.mjs';p.write_text("chromium.launch({channel:'msedge'});")
 assert m.audit(pathlib.Path(d))['status']=='failed'
`;
  const result = spawnSync('python3', ['-B', '-c', code], {encoding:'utf8'});
  assert.equal(result.status, 0, result.stderr);
});
test('Linux chapter memory probes return bytes rather than an unlabelled null', {skip: !linux}, () => {
  for (const chapter of ['02-calculus', '04-numerical-optimization']) {
    const file = path.join(root, 'examples', chapter, 'run.py');
    const code = `import ast,sys,pathlib
m=ast.parse(pathlib.Path(${JSON.stringify(file)}).read_text())
f=next(n for n in m.body if isinstance(n,ast.FunctionDef) and n.name=='peak_working_set_bytes')
ns={'sys':sys};exec(compile(ast.Module(body=[f],type_ignores=[]),'probe','exec'),ns)
assert ns['peak_working_set_bytes']()>0
`;
    const result=spawnSync('python3',['-B','-c',code],{encoding:'utf8'});
    assert.equal(result.status,0,result.stderr);
  }
});

test('DPO generated figures use current result metrics, not historical Windows captions', () => {
  const code = `import runpy,tempfile
from pathlib import Path
m=runpy.run_path('examples/24-preference-alignment/figures.py')
hand=dict(two_response_optimum=[.7,.3],two_response_kl=.1,two_response_objective=.6,preference_probability=.58,loss=.53)
experiment=dict(history=[dict(step=0,train=dict(optimization_loss=.69)),dict(step=300,train=dict(optimization_loss=.1))],test=dict(raw_candidate_pair_accuracy=.417,relative_pair_accuracy=.583),generation=dict(exact_match=.25))
with tempfile.TemporaryDirectory() as d:
 m['write_figures'](dict(hand=hand,experiments=[experiment]*4),Path(d))
 svg=(Path(d)/'ranking-generation.svg').read_text()
 assert '41.7%' in svg and '58.3%' in svg and '25.0%' in svg
 assert '83.3%' not in svg and '2/12' not in svg and '3/12' not in svg
`;
  const result = spawnSync('python3', ['-B', '-c', code], {cwd: root, encoding: 'utf8'});
  assert.equal(result.status, 0, result.stderr);
});
