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
