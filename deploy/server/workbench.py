"""Directory-local web variant; original simulation and control code is reused."""
from __future__ import annotations
import os
import json
import io
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
        <label for="profile-name">姓名或昵称<input id="profile-name" required maxlength="40" autocomplete="nickname"></label>
        <label for="profile-results">服务器个人实验目录<input id="profile-results" required maxlength="1024" placeholder="/data_ssdb/你的目录/实验目录"></label>''')
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
    from deploy.server.profiles import PROFILE, profile_id, model_settings, bind_harness, results_dir
    from deploy.server.records import install_service, workspace_factory
    from urllib.parse import urlsplit
    install_service()
    web.SecretStore = lambda: None
    web.ModelSettings = model_settings(web.ModelSettings)
    original_file = web.Handler._file
    def serve_file(self, path, root):
        relative = path.resolve().relative_to(ROOT)
        if relative.parts and relative.parts[0] == 'runs':
            try:
                own = results_dir()
                path = own.joinpath(*relative.parts[1:])
                return original_file(self, path, own)
            except ValueError:
                return self._json(404, {'error': 'file not found'})
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
              document.getElementById('model-title').textContent=config.configured?'模型配置':'首次使用';
              document.getElementById('model-close').disabled=!config.configured;
              document.getElementById('profile-name').value=config.display_name||'';
              document.getElementById('profile-results').value=config.results_path||'';
              document.getElementById('profile-confirm').checked=false;''')
            content = content.replace("const values={base_url:", "const values={confirmed:document.getElementById('profile-confirm').checked,display_name:document.getElementById('profile-name').value,results_path:document.getElementById('profile-results').value,base_url:")
            return self._send(200, content.encode(), 'text/javascript; charset=utf-8')
        return original_file(self, path, root)
    web.Handler._file = serve_file
    from tools import object_workbench
    original_builder = object_workbench.build_backend
    def build_backend(**kwargs):
        factory, harness = original_builder(**kwargs)
        import harness.bottle as bottle
        bottle.BottleWorkspace = workspace_factory(bottle.BottleWorkspace)
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
            original_stream = self.rfile
            try:
                route = urlsplit(self.path).path
                if route not in ('/api/model', '/api/model/clear', '/api/model/test'):
                    try: results_dir()
                    except (ValueError, OSError): return self._json(400, {'error': '请先确认已存在的个人实验目录'})
                # Research tools use the legacy shared runs directory. They are not
                # available on the shared web platform until separately isolated.
                if route in ('/api/tools', '/api/chat'):
                    try:
                        size = int(self.headers.get('Content-Length', '0'))
                        if not 0 < size <= web.MAX_BODY: raise ValueError('invalid request size')
                        raw = self.rfile.read(size)
                        body = json.loads(raw)
                        if not isinstance(body, dict): raise ValueError('JSON object required')
                        if body.get('role', 'execute') != 'execute':
                            return self._json(403, {'error': '共享平台暂不开放实验与训练工具'})
                        self.rfile = io.BytesIO(raw)
                    except (ValueError, TypeError, UnicodeError):
                        return self._json(400, {'error': '请求格式不正确'})
                return super().do_POST()
            finally:
                self.rfile = original_stream
                PROFILE.reset(token)
        def _send(self, status, data, content_type):
            if content_type.startswith('text/html') and b'<title>OpenArm Lab</title>' in data:
                data = simple_page(data.decode()).encode()
            if status == 200 and content_type.startswith('application/json') and urlsplit(self.path).path == '/api/status':
                value = json.loads(data)
                try:
                    own = results_dir()
                    value['runs'] = sorted([p.name for p in own.glob('*') if p.is_dir() and (p / 'summary.json').is_file()], reverse=True)[:30]
                except ValueError: value['runs'] = []
                data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            if status == 200 and content_type.startswith('application/json') and urlsplit(self.path).path == '/api/experiments':
                rows = []
                try:
                    for path in sorted(results_dir().glob('*'), reverse=True):
                        if path.is_symlink() or not (path / 'manifest.json').is_file(): continue
                        manifest = json.loads((path / 'manifest.json').read_text())
                        summary = json.loads((path / 'summary.json').read_text()) if (path / 'summary.json').exists() else {}
                        rows.append({'run_id': path.name, 'kind': manifest.get('kind'), 'status': summary.get('status', 'incomplete'), 'scope': 'simulation'})
                        if len(rows) >= 60: break
                except (ValueError, OSError): pass
                data = json.dumps({'experiments': rows}, ensure_ascii=False).encode()
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
