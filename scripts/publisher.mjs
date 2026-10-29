import {createServer} from 'node:http';
import {execFile as execFileCb, spawn} from 'node:child_process';
import {promisify} from 'node:util';
import {createHash, randomBytes} from 'node:crypto';
import {readFile, realpath, mkdtemp, rm, stat} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, resolve, relative, sep} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const execFile = promisify(execFileCb);
const port = 4176;
const allowedOrigin = `http://127.0.0.1:${port}`;
let proposal = null;
let busy = false;

async function git(args, extraEnv = {}) {
  try {
    const result = await execFile('git', args, {cwd: root, env: {...process.env, GIT_TERMINAL_PROMPT: '0', ...extraEnv}, timeout: 60000, maxBuffer: 1024 * 1024});
    return result.stdout.trim();
  } catch (error) {
    throw new Error(`Git 처리에 실패했습니다: ${(error.stderr || error.message).trim().slice(0, 500)}`);
  }
}
async function fileDigest(path) { return createHash('sha256').update(await readFile(resolve(root, path))).digest('hex'); }
async function validatePath(path, folder) {
  const normalized = path.replaceAll('\\', '/');
  if (!normalized.startsWith(folder + '/') || normalized.includes('\0')) throw new Error('공개할 파일 경로가 올바르지 않습니다.');
  let realRoot, actual;
  try { realRoot = await realpath(resolve(root, folder)); actual = await realpath(resolve(root, normalized)); }
  catch { throw new Error(`참조된 파일을 찾을 수 없습니다: ${normalized}`); }
  if (!actual.startsWith(realRoot + sep) || relative(root, actual).startsWith('..')) throw new Error('파일의 저장 위치를 확인할 수 없습니다.');
  return normalized;
}
async function prepare(path) {
  if (!/^_posts\/[^/]+\.(md|markdown)$/.test(path)) throw new Error('블로그 글을 선택해 주세요.');
  path = await validatePath(path, '_posts');
  const raw = await readFile(resolve(root, path), 'utf8');
  const output = resolve(root, '.local-editor/site/posts', path.split('/').pop().replace(/\.(md|markdown)$/, '.html'));
  let built;
  try { built = await stat(output); } catch { throw new Error('미리보기 페이지가 아직 생성되지 않았습니다. 글을 저장하고 미리보기를 확인해 주세요.'); }
  const source = await stat(resolve(root, path));
  if (built.mtimeMs < source.mtimeMs) throw new Error('미리보기가 아직 갱신되지 않았습니다. 잠시 후 새로고침해 주세요.');
  const match = raw.match(/^---\s*\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
  if (!match) throw new Error('글 정보를 읽을 수 없습니다.');
  const ruby = 'require "json"; require "psych"; require "date"; x=Psych.safe_load(STDIN.read,[Date,Time],[],false); abort "invalid" unless x.is_a?(Hash); print JSON.generate({published:x.fetch("published",true),title:x["title"].to_s})';
  const metaProcess = await new Promise((resolvePromise, reject) => {
    const child = spawn('ruby', ['-e', ruby], {cwd: root, stdio: ['pipe','pipe','pipe']});
    let out = '', err = '';
    child.stdout.on('data', data => out += data); child.stderr.on('data', data => err += data);
    child.on('error', reject); child.on('close', code => code === 0 ? resolvePromise(out) : reject(new Error('글 정보를 해석할 수 없습니다.')));
    child.stdin.end(match[1]);
  });
  const meta = JSON.parse(metaProcess);
  if (meta.published !== true) throw new Error('미공개 글입니다. 글 정보에서 공개를 켜 주세요.');
  const branch = await git(['branch','--show-current']);
  if (branch !== 'main') throw new Error('main 브랜치에서만 공개할 수 있습니다.');
  const head = await git(['rev-parse','HEAD']);
  for (const state of ['MERGE_HEAD','CHERRY_PICK_HEAD','REVERT_HEAD','rebase-merge','rebase-apply','sequencer']) {
    const gitPath = await git(['rev-parse','--git-path',state]);
    if (existsSync(resolve(root, gitPath))) throw new Error('진행 중인 Git 작업을 먼저 완료해 주세요.');
  }
  const staged = await git(['diff','--cached','--name-only']);
  if (staged) throw new Error('스테이징된 변경사항이 있습니다. 먼저 확인해 주세요.');
  const remote = await git(['config','--get','remote.origin.url']);
  if (!['https://github.com/HeoJinuk/HeoJinuk.github.io.git','https://github.com/HeoJinuk/HeoJinuk.github.io','git@github.com:HeoJinuk/HeoJinuk.github.io.git'].includes(remote)) throw new Error('GitHub 공개 대상이 예상과 다릅니다.');
  const remoteHead = (await git(['ls-remote','--refs','origin','refs/heads/main'])).split(/\s+/)[0];
  if (remoteHead !== head) throw new Error('GitHub와 로컬 이력이 일치하지 않습니다. 동기화 상태를 확인해 주세요.');
  const assets = new Set();
  for (const ref of raw.matchAll(/(?:\/?assets\/img\/posts\/[^\s)"'<>?#]+)/g)) {
    let asset;
    try { asset = decodeURIComponent(ref[0]).replace(/^\//,''); } catch { throw new Error('이미지 경로를 읽을 수 없습니다.'); }
    assets.add(await validatePath(asset, 'assets/img/posts'));
  }
  const files = [path, ...assets];
  const hashes = {};
  for (const file of files) hashes[file] = await fileDigest(file);
  if (hashes[path] !== createHash('sha256').update(raw, 'utf8').digest('hex')) throw new Error('확인 중에 글이 변경되었습니다. 다시 확인해 주세요.');
  return {title: meta.title, files, hashes, head, branch};
}
async function publish(prepared) {
  const checked = await prepare(prepared.files[0]);
  if (checked.head !== prepared.head || JSON.stringify(checked.files) !== JSON.stringify(prepared.files) || JSON.stringify(checked.hashes) !== JSON.stringify(prepared.hashes)) throw new Error('확인 후 글이나 이미지가 바뀌었습니다. 다시 확인해 주세요.');
  const tmp = await mkdtemp(resolve(tmpdir(), 'blog-publish-'));
  const env = {GIT_INDEX_FILE: resolve(tmp, 'index')};
  try {
    await git(['read-tree', checked.head], env);
    await git(['add','--',...checked.files], env);
    const tree = await git(['write-tree'], env);
    if (tree === await git(['rev-parse', `${checked.head}^{tree}`])) throw new Error('공개할 변경사항이 없습니다.');
    const commit = await git(['commit-tree', tree, '-p', checked.head, '-m', `Publish blog post: ${checked.files[0].split('/').pop()}`], env);
    if (await git(['rev-parse','HEAD']) !== checked.head || await git(['branch','--show-current']) !== 'main' || await git(['diff','--cached','--name-only'])) throw new Error('공개 준비 중 Git 상태가 변경되었습니다. 다시 확인해 주세요.');
    await git(['push','origin',`${commit}:refs/heads/main`]);
    try {
      await git(['update-ref','refs/heads/main',commit,checked.head]);
      await git(['reset','-q','HEAD','--',...checked.files]);
    } catch {
      throw new Error(`GitHub 업로드는 성공했습니다 (${commit.slice(0,7)}) 하지만 로컬 이력 갱신에 실패했습니다. 다시 실행하지 말고 상태를 확인해 주세요.`);
    }
    return {commit: commit.slice(0,7)};
  } finally { await rm(tmp,{recursive:true,force:true}); }
}
const server = createServer(async (req,res) => {
  if (req.headers.host !== `127.0.0.1:${port}`) {res.writeHead(403);res.end();return;}
  const pathname = new URL(req.url, allowedOrigin).pathname;
  if (req.method === 'GET' && ['/', '/review.js', '/review.css'].includes(pathname)) {
    const name = pathname === '/' ? 'publish-review.html' : pathname === '/review.js' ? 'publish-review.js' : 'publish-review.css';
    const type = pathname === '/' ? 'text/html; charset=utf-8' : pathname === '/review.js' ? 'text/javascript; charset=utf-8' : 'text/css; charset=utf-8';
    try {
      const content = await readFile(resolve(root, 'scripts', name));
      res.writeHead(200, {'Content-Type':type, 'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff', 'Content-Security-Policy':"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-src http://127.0.0.1:4174; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"});
      res.end(content);
    } catch {res.writeHead(500);res.end();}
    return;
  }
  if (req.headers.origin !== allowedOrigin) {res.writeHead(403);res.end();return;}
  res.setHeader('Cache-Control','no-store');
  if (req.method !== 'POST' || !['/prepare','/publish'].includes(pathname) || !(req.headers['content-type'] || '').startsWith('application/json')) {res.writeHead(404);res.end();return;}
  let body = '';
  let locked = false;
  try {
    for await (const chunk of req) {body += chunk; if (body.length > 8192) throw new Error('요청이 너무 큽니다.');}
    if (busy) throw new Error('다른 공개 작업이 진행 중입니다.');
    busy = true;
    locked = true;
    const input = JSON.parse(body);
    let result;
    if (req.url === '/prepare') {
      const data = await prepare(input.path);
      const token = randomBytes(24).toString('hex');
      proposal = {token, data, expires: Date.now()+5*60*1000};
      result = {token, title:data.title, files:data.files};
    } else {
      if (!proposal || input.token !== proposal.token || Date.now()>proposal.expires) throw new Error('확인 유효시간이 지났습니다. 다시 확인해 주세요.');
      const current = proposal; proposal = null;
      result = await publish(current.data);
    }
    res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify(result));
  } catch(error) {res.writeHead(400,{'Content-Type':'application/json'});res.end(JSON.stringify({error:error.message}));}
  finally {if (locked) busy=false;}
});
server.listen(port,'127.0.0.1', () => console.log('공개 기능: http://127.0.0.1:4176'));
