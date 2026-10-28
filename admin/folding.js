/* Add a collapsible metadata section without moving Decap's controlled fields. */
(() => {
  function setExpanded(pane, expanded) {
    pane.dataset.metadataCollapsed = String(!expanded);
    const button = pane.querySelector(':scope > .metadata-toggle');
    if (!button) return;
    button.setAttribute('aria-expanded', String(expanded));
    button.querySelector('.metadata-toggle-action').textContent = expanded ? '접기' : '펼치기';
  }

  function attachToggle() {
    const bodyField = document.querySelector('#nc-root [class*="ControlPaneContainer"] > [aria-label="markdown field"]');
    if (!bodyField) return;
    const pane = bodyField.parentElement;
    if (pane.querySelector(':scope > .metadata-toggle')) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'metadata-toggle';
    button.setAttribute('aria-label', '글 정보 접기 또는 펼치기');
    button.innerHTML = '<span><strong>글 정보</strong><small>제목 · 작성일 · 분류 · 공개 설정</small></span><span class="metadata-toggle-action"></span>';
    button.addEventListener('click', () => setExpanded(pane, pane.dataset.metadataCollapsed === 'true'));
    pane.prepend(button);
    setExpanded(pane, false);
  }

  // Decap is a single-page application, so the editor mounts after navigation.
  new MutationObserver(attachToggle).observe(document.body, {childList: true, subtree: true});
  document.addEventListener('click', event => {
    const menuItem = event.target.closest('[role="menuitem"], button, a, [role="button"]');
    if (!menuItem || !/^(지금 게시|게시하고 새로 만들기|게시하고 복제)$/.test(menuItem.textContent.trim())) return;
    const pane = document.querySelector('#nc-root [class*="ControlPaneContainer"][data-metadata-collapsed="true"]');
    if (!pane) return;
    const title = pane.querySelector('input[id^="title-field-"]');
    const date = pane.querySelector('input[id^="date-field-"]');
    if (!title?.value.trim() || !date?.value) setExpanded(pane, true);
  }, true);
})();
