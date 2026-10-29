(() => {
  const path = new URLSearchParams(location.search).get('path');
  const status = document.querySelector('#status');
  const files = document.querySelector('#files');
  const button = document.querySelector('#publish');
  const preview = document.querySelector('#preview');
  let token = null;
  async function request(endpoint, body) {
    const response = await fetch(endpoint, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '공개 처리에 실패했습니다.');
    return data;
  }
  async function prepare() {
    if (!path || !/^_posts\/[^/]+\.(md|markdown)$/.test(path)) {status.textContent = '글 읽기 화면에서 공개하기를 눌러 주세요.';return;}
    const slug = path.slice('_posts/'.length).replace(/\.(md|markdown)$/, '');
    preview.src = `http://127.0.0.1:4174/posts/${encodeURIComponent(slug)}?reader-preview=1`;
    try {
      const result = await request('/prepare',{path});
      token=result.token;
      for (const file of result.files) {const li=document.createElement('li');li.textContent=file;files.append(li);}
      status.textContent=`글 제목: ${result.title}`;
      button.hidden=false;
    } catch(error) {status.textContent=error instanceof TypeError ? '연결할 수 없습니다. 글쓰기 프로그램을 다시 실행해 주세요.' : error.message;}
  }
  button.addEventListener('click',async()=>{
    if (!token) return;
    button.disabled=true;status.textContent='GitHub에 업로드하고 있습니다…';
    try {
      const result=await request('/publish',{token});token=null;button.hidden=true;
      status.textContent=`GitHub 업로드 완료 (${result.commit}). 홈페이지 반영까지 잠시 걸릴 수 있습니다.`;
    } catch(error) {token=null;button.hidden=true;status.textContent=error instanceof TypeError ? '서버 응답을 받지 못했습니다. 다시 시도하기 전에 GitHub 상태를 확인해 주세요.' : error.message;}
    finally {button.disabled=false;}
  });
  prepare();
})();
