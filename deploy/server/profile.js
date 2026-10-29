const response = await fetch('/api/model');
if (response.ok) {
  const config = await response.json();
  const dialog = document.getElementById('model-dialog');
  document.getElementById('profile-name').value = config.display_name || '';
  document.getElementById('model-base').value = config.base_url || '';
  document.getElementById('model-name').value = config.model || '';
  document.getElementById('model-vision').checked = config.vision;
  if (!config.confirmed && !dialog.open) {
    document.getElementById('model-title').textContent = '首次使用';
    dialog.showModal();
  }
}
