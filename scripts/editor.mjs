import {spawn} from 'node:child_process';
import {existsSync, readFileSync, mkdirSync} from 'node:fs';
import {createRequire} from 'node:module';
import {createConnection} from 'node:net';
import {resolve, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
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

if (await inUse(port) || await inUse(8081) || await inUse(4176)) {
  console.error(`포트 ${port}, 8081 또는 4176이 사용 중입니다. 실행 중인 편집기 창을 확인하거나 기존 편집기를 종료해 주세요.`);
  process.exitCode = 1;
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
      console.log(`\n글쓰기: ${url}\n미리보기: http://127.0.0.1:${port}/blog/\n저장하면 홈페이지가 다시 빌드됩니다. 미리보기 탭을 새로고침하세요.\n종료: 이 창에서 Ctrl+C\nGitHub 공개 반영은 별도 커밋·push가 필요합니다.\n`);
      if (process.platform === 'darwin' && !process.env.EDITOR_NO_OPEN) spawn('open', [url]);
    } else if (!stopping) {
      console.error('편집기 서버 시작에 실패했습니다. 위 오류를 확인해 주세요.');
      stop(1);
    }
  }
}
