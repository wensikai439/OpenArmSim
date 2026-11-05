const response = await fetch('/api/model');
if (response.ok) {
  const config = await response.json();
  const dialog = document.getElementById('model-dialog');
  dialog.addEventListener('cancel', event => {
    if (document.getElementById('model-close').disabled) event.preventDefault();
  });
  document.getElementById('profile-name').value = config.display_name || '';
  document.getElementById('profile-results').value = config.results_path || '';
  document.getElementById('model-base').value = config.base_url || '';
  document.getElementById('model-name').value = config.model || '';
  document.getElementById('model-vision').checked = config.vision;
  if (!config.configured && !dialog.open) {
    document.getElementById('model-title').textContent = '首次使用';
    document.getElementById('model-close').disabled = true;
    dialog.showModal();
  }
}
