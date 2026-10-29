"""Directory-local web variant; original simulation and control code is reused."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def simple_page(content):
    content = content.replace('和智能体对话', '对话')
    content = content.replace('描述任务，或询问本次实验…', '描述任务…')
    content = content.replace('开始一轮实验，留下可核验的过程。', '暂无关键帧')
    content = content.replace('阶段由接触与稳定性证据确认，播放到某个时间点不会自动判定完成。', '')
    content = content.replace('<span>固定相机</span>', '')
    content = content.replace('<div class="header-meta">', '<div class="header-meta"><form method="post" action="/logout"><button class="quiet">退出</button></form>')
    content = content.replace('<form id="model-form" class="model-form">', '''<form id="model-form" class="model-form">
        <label for="profile-name">姓名或昵称<input id="profile-name" required maxlength="40" autocomplete="nickname"></label>''')
    content = content.replace('<div class="actions"><button id="model-save"', '''<label class="checkbox-row"><input id="profile-confirm" type="checkbox" required>确认使用本人的地址、模型和 Key 发送任务与相机图像</label>
        <a href="https://zzq.si/keys" target="_blank" rel="noopener">前往零词元获取 Key ↗</a>
        <div class="actions"><button id="model-save"''')
    content = content.replace('>保存配置</button>', '>确认并保存</button>')
    content = content.replace('placeholder="https://api.deepseek.com/v1"', 'placeholder="填写本人的请求地址"')
    content = content.replace('placeholder="deepseek-chat"', 'placeholder="填写模型名称"')
    content = content.replace('</body>', '<script type="module" src="/ui/profile.js"></script></body>')
    style = '''<style>.task-context,.agent-mode,.rail-label,.camera-card em,
    .view-footer>.muted,.frames-heading>.muted,.title-row>#view-kind,
    .messages>.message.note:first-child{display:none}
    .conversation>.panel-heading{border-bottom:1px solid #e7e3da}
    .agent-toolbar{font-size:12px}.result-line:empty{display:none}
    .composer-bottom{font-size:11px}.camera-card>span{font-size:12px}
    .frames-heading h3{font-size:13px}.view-footer{font-size:12px}
    .stage-list li,.stage-source{font-size:12px}
    </style>'''
    return content.replace('</head>', style + '</head>')


def main():
    import server as web
    from deploy.server.profiles import PROFILE, profile_id, model_settings, bind_harness
    web.SecretStore = lambda: None
    web.ModelSettings = model_settings(web.ModelSettings)
    original_file = web.Handler._file
    def serve_file(self, path, root):
        relative = path.resolve().relative_to(ROOT)
        if str(relative) == 'ui/profile.js':
            return self._send(200, (ROOT / 'deploy/server/profile.js').read_bytes(), 'text/javascript; charset=utf-8')
        if str(relative) == 'ui/index.html':
            content = simple_page(path.read_text()).replace('<option value="research">实验与训练</option>', '')
            return self._send(200, content.encode(), 'text/html; charset=utf-8')
        if str(relative) in ('ui/app.js', 'ui/agent.js'):
            content = path.read_text().replace('开始一轮实验，留下可核验的过程。', '暂无关键帧')
            content = content.replace('阶段由接触与稳定性证据确认，播放到某个时间点不会自动判定完成。', '')
            content = content.replace('发送一个任务，模型会在这里制定运行阶段。', '等待任务')
            content = content.replace("message('描述任务后，模型会制定阶段、调用工具，并在运行后复查。过程与记忆都会保留。');", '')
            content = content.replace("config.storage==='system_keyring'?'密钥保存在本机系统密钥环，重启后自动恢复。':'当前密钥只在服务内存，保存配置后会使用本机存储。'", "'配置仅属于当前浏览器。'")
            content = content.replace('function renderModel(config, editing=false){', '''function renderModel(config, editing=false){
              document.getElementById('model-title').textContent=config.confirmed?'模型配置':'首次使用';
              document.getElementById('profile-name').value=config.display_name||'';
              document.getElementById('profile-confirm').checked=false;''')
            content = content.replace("const values={base_url:", "const values={confirmed:document.getElementById('profile-confirm').checked,display_name:document.getElementById('profile-name').value,base_url:")
            return self._send(200, content.encode(), 'text/javascript; charset=utf-8')
        return original_file(self, path, root)
    web.Handler._file = serve_file
    from tools import object_workbench
    original_builder = object_workbench.build_backend
    def build_backend(**kwargs):
        factory, harness = original_builder(**kwargs)
        return factory, bind_harness(harness)
    object_workbench.build_backend = build_backend
    # This subclass applies the variant before the upstream page shortcut.
    class PageHandler(web.Handler):
        def do_GET(self):
            token = PROFILE.set(profile_id(self.headers.get('X-OpenArm-Profile')))
            try: return super().do_GET()
            finally: PROFILE.reset(token)
        def do_POST(self):
            token = PROFILE.set(profile_id(self.headers.get('X-OpenArm-Profile')))
            try: return super().do_POST()
            finally: PROFILE.reset(token)
        def _send(self, status, data, content_type):
            if content_type.startswith('text/html') and b'<title>OpenArm Lab</title>' in data:
                data = simple_page(data.decode()).encode()
            return super()._send(status, data, content_type)
    web.Handler = PageHandler
    if os.environ.get('OPENARM_GPU_START') == '0':
        object_workbench.start_gpu = lambda _: None
    else:
        def start_gpu(root):
            from sim.cache_curobo import metadata
            with (root / '.runtime/logs/object-gpu.log').open('ab') as log:
                process = subprocess.Popen([str(root / '.runtime/curobo/bin/python'), '-B',
                    str(root / 'deploy/server/worker.py')], cwd=root, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 20
            while process.poll() is None and time.monotonic() < deadline:
                if metadata().get('ready'): break
                time.sleep(.2)
            return process
        object_workbench.start_gpu = start_gpu
    object_workbench.main()


if __name__ == '__main__': main()
