/* A local editor must never fall through to the remote GitHub backend. */
(async () => {
  const status = document.getElementById('editor-status');
  const message = document.getElementById('editor-message');
  if (!['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) {
    message.textContent = '이 편집기는 개인 컴퓨터에서 사용합니다. 저장소 폴더의 “블로그 글쓰기.command”를 실행해 주세요.';
    return;
  }
  try {
    const response = await fetch('http://127.0.0.1:8081/api/v1', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'info'}),
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok) throw new Error('Local proxy unavailable');
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/decap-cms@3.16.3/dist/decap-cms.js';
      script.integrity = 'sha384-A/Gdn928CNLufmnGWGs9RM4Q9o8bSNLeJERqBFrBjwzL7FRpoLT3MPF8Tl0+Wd2p';
      script.crossOrigin = 'anonymous';
      script.onload = resolve;
      script.onerror = () => reject(new Error('Could not load Decap CMS'));
      document.body.append(script);
    });
    CMS.registerPreviewStyle('/assets/css/academic.css');
    CMS.registerPreviewStyle('/admin/preview.css');
    CMS.registerPreviewTemplate('posts', ({entry, widgetFor}) => {
      const h = window.h;
      return h('article', {className: 'article'},
        h('header', {className: 'page-heading'}, h('p', {className: 'eyebrow'}, entry.getIn(['data', 'category'])),
          h('h1', {}, entry.getIn(['data', 'title']))),
        h('div', {className: 'prose'}, widgetFor('body')));
    });
    const notice = document.createElement('aside');
    notice.id = 'local-notice';
    notice.textContent = '로컬 편집 · “게시”는 이 컴퓨터에 저장합니다. ';
    const preview = document.createElement('a');
    preview.href = '/blog/';
    preview.target = '_blank';
    preview.rel = 'noopener';
    preview.textContent = '홈페이지 미리보기 ↗';
    notice.append(preview);
    document.body.append(notice);
    CMS.init();
    status.hidden = true;
  } catch (error) {
    message.textContent = '편집기를 열 수 없습니다. “블로그 글쓰기.command” 실행 상태와 인터넷 연결을 확인한 뒤 새로고침해 주세요.';
    console.error('Local editor startup failed:', error.message);
  }
})();
