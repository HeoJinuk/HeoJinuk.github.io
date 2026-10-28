document.querySelectorAll('[data-print]').forEach(button => {
  button.addEventListener('click', () => window.print());
});

if (['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) {
  document.querySelectorAll('[data-local-editor]').forEach(link => { link.hidden = false; });
  document.querySelectorAll('[data-edit-post]').forEach(link => {
    const filename = link.dataset.editPost;
    const slug = filename.replace(/\.[^.]+$/, '');
    if (!slug || slug === filename) return;
    link.href = `${link.href}#/collections/posts/entries/${encodeURIComponent(slug)}`;
    link.hidden = false;
  });
}
