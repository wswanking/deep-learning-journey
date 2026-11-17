import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';

// Never download a browser or reuse a personal browser profile.
export function browserCandidates({ platform = process.platform, env = process.env } = {}) {
  const candidates = [{ channel: 'msedge' }, { channel: 'chrome' }];
  if (platform === 'linux') {
    for (const directory of (env.PATH || '').split(path.delimiter).filter(Boolean)) {
      for (const name of ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable', 'microsoft-edge']) {
        const executablePath = path.join(directory, name);
        try {
          fs.accessSync(executablePath, fs.constants.X_OK);
          if (fs.statSync(executablePath).isFile()) candidates.push({ executablePath });
        } catch { /* Not installed here. */ }
      }
    }
  }
  return candidates.filter((candidate, index, all) =>
    all.findIndex(item => JSON.stringify(item) === JSON.stringify(candidate)) === index);
}

export async function launchBrowser(executablePath, options = {}, dependencies = {}) {
  const env = dependencies.env || process.env;
  const launch = dependencies.launch || (settings => chromium.launch(settings));
  const explicit = executablePath || env.BROWSER_EXECUTABLE;
  // An explicit selection must fail visibly rather than silently use another browser.
  if (explicit) return launch({ ...options, headless: true, executablePath: explicit });
  const errors = [];
  for (const candidate of browserCandidates({ platform: dependencies.platform, env })) {
    try { return await launch({ ...options, headless: true, ...candidate }); }
    catch (error) { errors.push((candidate.channel || candidate.executablePath) + ': ' + error.message.split('\n')[0]); }
  }
  throw new Error('未找到可用的 Edge/Chrome/Chromium。请安装浏览器，或使用 --browser / BROWSER_EXECUTABLE 指定路径。' + errors.join('; '));
}
