(() => {
  if (location.hostname !== '127.0.0.1' || location.port !== '4174' || new URLSearchParams(location.search).has('reader-preview')) return;
  const link = document.querySelector('[data-publish-post]');
  if (!link) return;
  const path = `_posts/${link.dataset.publishPost}`;
  link.href = `http://127.0.0.1:4176/?path=${encodeURIComponent(path)}`;
  link.target = '_blank';
  link.rel = 'noopener';
  link.hidden = false;
})();
