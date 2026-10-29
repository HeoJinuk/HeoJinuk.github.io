import {spawn, execFile as execFileCallback} from 'node:child_process';
import {promisify} from 'node:util';
import {existsSync, readFileSync, mkdirSync, realpathSync} from 'node:fs';
import {createRequire} from 'node:module';
import {createConnection} from 'node:net';
import {resolve, dirname, basename} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const execFile = promisify(execFileCallback);
const port = 4174;
const url = `http://127.0.0.1:${port}/admin/`;
const children = [];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) child.kill('SIGTERM');
  process.exitCode = code;
}
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
function start(command, args, env) {
  const child = spawn(command, args, {cwd: root, env, stdio: 'inherit'});
  children.push(child);
  child.on('error', error => {console.error(error.message); stop(1);});
  child.on('exit', code => {if (!stopping) {console.error(`서버 종료 (${code}).`); stop(code || 1);}});
  return child;
}
async function inUse(targetPort) {
  return new Promise(resolve => {
    const socket = createConnection({host: '127.0.0.1', port: targetPort});
    socket.once('connect', () => {socket.destroy(); resolve(true);});
    socket.once('error', () => resolve(false));
  });
}

// A reused server must belong to this checkout, not just occupy the same port.
async function belongsToThisCheckout(targetPort) {
  try {
    const {stdout: pidText} = await execFile('lsof', ['-nP', '-t', `-iTCP:${targetPort}`, '-sTCP:LISTEN'], {timeout: 3000});
    const pids = pidText.trim().split(/\s+/);
    if (pids.length !== 1 || !/^\d+$/.test(pids[0])) return false;
    const {stdout} = await execFile('lsof', ['-a', '-p', pids[0], '-d', 'cwd', '-Fn'], {timeout: 3000});
    const cwdLine = stdout.split('\n').find(line => line.startsWith('n'));
    return !!cwdLine && realpathSync(cwdLine.slice(1)) === realpathSync(root);
  } catch { return false; }
}
async function existingEditorIsReady() {
  try {
    const ownership = await Promise.all([port, 8081, 4176].map(belongsToThisCheckout));
    if (!ownership.every(Boolean)) return false;
    const signal = AbortSignal.timeout(3000);
    const [page, proxy, publisher] = await Promise.all([
      fetch(url, {signal}),
      fetch('http://127.0.0.1:8081/api/v1', {method: 'POST', headers: {'Content-Type': 'application/json', Origin: `http://127.0.0.1:${port}`}, body: JSON.stringify({action: 'info'}), signal}),
      fetch('http://127.0.0.1:4176/', {signal}),
    ]);
    if (!page.ok || !proxy.ok || !publisher.ok) return false;
    const [pageHtml, proxyInfo, publisherHtml] = await Promise.all([page.text(), proxy.json(), publisher.text()]);
    return pageHtml.includes('cms-config-url') && proxyInfo.type === 'local_fs' && proxyInfo.repo === basename(root) && publisherHtml.includes('publish') && publisherHtml.includes('/review.js');
  } catch { return false; }
}
function openEditor() {
  if (process.platform === 'darwin' && !process.env.EDITOR_NO_OPEN) spawn('open', [url]);
}

async function waitForPage() {
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline && !stopping) {
    try {
      const response = await fetch(url);
      if (response.ok) return true;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 300));
  }
  return false;
}

const occupied = await Promise.all([port, 8081, 4176].map(inUse));
if (occupied.some(Boolean)) {
  if (occupied.every(Boolean) && await existingEditorIsReady()) {
    console.log(`이미 실행 중인 글쓰기 화면을 엽니다: ${url}`);
    openEditor();
  } else {
    console.error(`포트 ${port}, 8081 또는 4176이 사용 중이지만 이 저장소의 정상적인 편집기인지 확인하지 못했습니다. 기존 프로그램을 확인한 뒤 다시 실행해 주세요.`);
    process.exitCode = 1;
  }
} else {
  let proxy;
  try {
    const packagePath = require.resolve('decap-server/package.json');
    const pkg = JSON.parse(readFileSync(packagePath, 'utf8'));
    proxy = resolve(dirname(packagePath), typeof pkg.bin === 'string' ? pkg.bin : pkg.bin['decap-server']);
  } catch {
    console.error('Decap 서버가 없습니다. 먼저 npm install 또는 pnpm install을 실행해 주세요.');
    process.exitCode = 1;
  }
  if (proxy) {
    mkdirSync(resolve(root, '.local-editor'), {recursive: true});
    const env = {...process.env};
    start(process.execPath, [resolve(root, 'scripts/publisher.mjs')], env);
    start(process.execPath, [proxy], {...env, BIND_HOST: '127.0.0.1', ORIGIN: `http://127.0.0.1:${port}`, MODE: 'fs', PORT: '8081', GIT_REPO_DIRECTORY: root});
    const localGems = resolve(root, '.local-editor/gems');
    const hasLocalJekyll = existsSync(resolve(localGems, 'bin/jekyll'));
    const args = ['serve', '--watch', '--host', '127.0.0.1', '--port', String(port), '--destination', resolve(root, '.local-editor/site')];
    if (hasLocalJekyll) {
      start('ruby', [resolve(localGems, 'bin/jekyll'), ...args], {...env, GEM_HOME: localGems, GEM_PATH: localGems, JEKYLL_NO_BUNDLER_REQUIRE: 'true'});
    } else {
      start('bundle', ['exec', 'jekyll', ...args], env);
    }
    if (await waitForPage()) {
      console.log(`\n글쓰기: ${url}\n미리보기: http://127.0.0.1:${port}/blog/\n저장하면 홈페이지가 다시 빌드됩니다. 미리보기 탭을 새로고침하세요.\n종료: 이 창에서 Ctrl+C\n글 읽기 화면에서 공개하기 버튼으로 GitHub에 올릴 수 있습니다.\n`);
      openEditor();
    } else if (!stopping) {
      console.error('편집기 서버 시작에 실패했습니다. 위 오류를 확인해 주세요.');
      stop(1);
    }
  }
}
