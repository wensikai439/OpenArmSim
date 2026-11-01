// Presentation only: the existing app owns API calls, commands, and evidence.
export function createWorkspaceUI() {
  const byId = id => document.getElementById(id);
  const workspace = byId('workspace');
  const focusButton = byId('focus-scene');
  const tabs = [...document.querySelectorAll('.evidence-tab')];

  function selectTab(tab, focus = false) {
    for (const item of tabs) {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
      byId(item.getAttribute('aria-controls')).hidden = !selected;
    }
    if (focus) tab.focus();
  }
  for (const [index, tab] of tabs.entries()) {
    tab.addEventListener('click', () => selectTab(tab));
    tab.addEventListener('keydown', event => {
      const next = {ArrowRight: (index + 1) % tabs.length,
        ArrowLeft: (index + tabs.length - 1) % tabs.length,
        Home: 0, End: tabs.length - 1}[event.key];
      if (next === undefined) return;
      event.preventDefault();
      selectTab(tabs[next], true);
    });
  }

  function focusScene(focused) {
    workspace.classList.toggle('scene-focused', focused);
    focusButton.setAttribute('aria-pressed', String(focused));
    focusButton.setAttribute('aria-label', focused ? '还原工作台' : '展开场景');
    focusButton.title = focused ? '还原工作台 · Esc' : '展开场景';
  }
  focusButton.addEventListener('click', () =>
    focusScene(!workspace.classList.contains('scene-focused')));
  byId('workspace-home').addEventListener('click', () => {
    focusScene(false);
    byId('chat-input').focus();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !document.querySelector('dialog[open]') &&
        workspace.classList.contains('scene-focused')) {
      focusScene(false);
      focusButton.focus();
    }
  });
  document.querySelectorAll('[data-open]').forEach(button => {
    button.addEventListener('click', () => byId(button.dataset.open).click());
  });

  const welcome = byId('messages').querySelector('.message');
  if (welcome) {
    welcome.classList.add('welcome-message');
    welcome.querySelector('.message-label').textContent = '从一个任务开始';
    welcome.querySelector('.message-body').textContent =
      '说出目标，我会观察场景、调用技能，并根据执行反馈复查结果。';
  }

  return {
    render(data, frames) {
      const connection = byId('connection');
      const stale = data.source_current === false;
      const busy = data.state === 'running' || data.agent?.busy;
      connection.textContent = stale ? '需要重启服务' : busy ? '执行中' : '已连接';
      connection.dataset.state = stale ? 'warning' : busy ? 'running' : 'ready';
      byId('robot-identity').textContent = `OpenArm ${data.robot_model || ''}`.trim();
      byId('frame-count').textContent = String(frames.length);
      document.querySelectorAll('[data-camera]').forEach(card => {
        card.setAttribute('aria-pressed', String(card.classList.contains('selected')));
      });
    },
    disconnected() {
      byId('connection').textContent = '连接中断';
      byId('connection').dataset.state = 'offline';
    },
  };
}
