document.querySelectorAll('[data-print]').forEach(button => {
  button.addEventListener('click', () => window.print());
});

if (['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) {
  document.querySelectorAll('[data-local-editor]').forEach(link => { link.hidden = false; });
}
