import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const mobile = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const cli = join(mobile, 'node_modules/expo/bin/cli');
const browsers = process.platform === 'win32' ? [
  join(process.env.ProgramFiles || 'C:/Program Files', 'Google/Chrome/Application/chrome.exe'),
  join(process.env['ProgramFiles(x86)'] || 'C:/Program Files (x86)', 'Microsoft/Edge/Application/msedge.exe'),
] : [];
const browser = browsers.find(existsSync);
let opened = false, buffer = '';
const child = spawn(process.execPath, [cli, 'login', '--browser'], {
  cwd: mobile, env: { ...process.env, ...(browser ? { BROWSER: 'none' } : {}) },
  stdio: ['inherit', 'pipe', 'inherit'], windowsHide: true,
});
child.stdout.on('data', data => {
  buffer += data.toString();
  let end;
  while ((end = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
    const url = line.match(/https:\/\/expo\.dev\/login\?\S+/)?.[0];
    if (url && browser && !opened) {
      opened = true;
      const tab = spawn(browser, [url.trim()], { detached: true, stdio: 'ignore', windowsHide: true });
      tab.on('error', () => console.log('Open this official Expo login URL in your browser:\n' + url));
      tab.unref();
      console.log('Expo browser login opened. Sign in with the same account as Expo Go on your iPhone.');
    } else if (!url || !browser) console.log(line.trimEnd());
  }
});
child.on('error', error => { console.error(error.message); process.exitCode = 1; });
child.on('exit', code => { if (buffer.trim()) console.log(buffer.trim()); process.exitCode = code || 0; });
process.once('SIGINT', () => child.kill());
process.once('SIGTERM', () => child.kill());
