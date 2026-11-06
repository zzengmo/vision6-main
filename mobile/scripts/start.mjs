import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { networkInterfaces } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const mobile = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const installedSDK = JSON.parse(readFileSync(join(mobile, 'node_modules/expo/package.json'), 'utf8')).version;
console.log(`SteelVision mobile - Expo SDK ${installedSDK.split('.')[0]} (${installedSDK})`);
const app = resolve(mobile, '../vision6-main/SteelVision_ML/test1_web');
const apiPort = Number(process.env.STEELVISION_API_PORT || '8504');
const metroPort = Number(process.env.STEELVISION_EXPO_PORT || '8081');
if (![apiPort, metroPort].every(p => Number.isInteger(p) && p > 0 && p < 65536)) throw Error('Invalid API or Expo port.');
const addresses = [...new Set(Object.values(networkInterfaces()).flat().filter(n => n && n.family === 'IPv4' && !n.internal && /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(n.address)).map(n => n.address))];
const configured = JSON.parse(readFileSync(join(app, 'settings.json'), 'utf8')).ui_python;
const python = [resolve(app, configured), join(app, '.venv/Scripts/python.exe'), join(app, '../.venv/bin/python')].find(existsSync);
if (!python) throw Error('Python environment missing. Run Setup_Test1_Web.cmd first.');
let api, metro, stopping = false;
function stop() {
  if (stopping) return;
  stopping = true;
  metro?.kill(); api?.kill();
}
process.once('SIGINT', stop); process.once('SIGTERM', stop); process.once('exit', stop);
async function health() {
  try { return await (await fetch(`http://127.0.0.1:${apiPort}/api/health`, { signal: AbortSignal.timeout(1500) })).json(); }
  catch { return null; }
}
try {
  const origins = ['localhost', '127.0.0.1', ...addresses].map(host => `http://${host}:${metroPort}`);
  let state = await health();
  if (state && (state.app !== 'steelvision-react' || !state.network_access)) throw Error(`Port ${apiPort} is occupied or only allows localhost. Close that API process or choose STEELVISION_API_PORT.`);
  if (!state) {
    api = spawn(python, [join(app, 'server.py'), '--host', '0.0.0.0', '--port', String(apiPort), ...origins.flatMap(o => ['--allow-origin', o])], { cwd: app, stdio: 'inherit', windowsHide: true });
    api.on('error', error => { console.error(error.message); stop(); process.exitCode = 1; });
    for (let i = 0; i < 60 && !state; i++) {
      if (api.exitCode !== null) throw Error('Python API did not start. Check the message above.');
      await new Promise(r => setTimeout(r, 500)); state = await health();
    }
    if (!state) throw Error('Python API startup timed out.');
  }
  console.log('\nSteelVision mobile API (same Wi-Fi):');
  for (const host of addresses) console.log(`  http://${host}:${apiPort}`);
  if (!addresses.length) console.log(`  No Wi-Fi address detected. Enter the PC IP manually with port ${apiPort}.`);
  console.log('\niPhone: open Expo Go with the QR below. Keep this window open. Ctrl+C stops both servers.\n');
  console.log('iPhone requires the same Expo account in Expo Go and Expo CLI. First login: npm.cmd run login\n');
  const env = { ...process.env, EXPO_PUBLIC_API_PORT: String(apiPort) };
  if (!env.EXPO_PUBLIC_API_URL && addresses[0]) env.EXPO_PUBLIC_API_URL = `http://${addresses[0]}:${apiPort}`;
  const extra = process.argv.slice(2);
  const hostFlags = extra.some(a => ['--offline', '--host', '--tunnel', '--lan', '--localhost'].includes(a)) ? [] : ['--lan'];
  metro = spawn(process.execPath, [join(mobile, 'node_modules/expo/bin/cli'), 'start', ...hostFlags, '--port', String(metroPort), ...extra], { cwd: mobile, env, stdio: 'inherit', windowsHide: true });
  metro.on('error', error => { console.error(error.message); stop(); process.exitCode = 1; });
  metro.on('exit', code => { stop(); process.exitCode = code || 0; });
} catch (error) {
  console.error(error.message); stop(); process.exitCode = 1;
}
