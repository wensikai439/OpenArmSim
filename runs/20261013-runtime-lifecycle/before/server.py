#!/usr/bin/env python3
"""Loopback-only HTTP UI and tool API for the native simulation service."""
from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

os.environ.setdefault('MUJOCO_GL', 'egl')
from sim.environment import ROOT
from sim.service import SimulationService
from harness.agent import Harness, tools_for
from harness.llm import ModelSettings
from harness.vault import SecretStore
from harness.workspace import Workspace
from sim.learning import LearningJobs

PORT = int(os.environ.get('OPENARM_PORT', '8765'))
MAX_BODY = 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = 'OpenArmSim/1.0'

    @property
    def service(self):
        return self.server.simulation

    def _local(self):
        allowed = {f'127.0.0.1:{PORT}', f'localhost:{PORT}'}
        if self.headers.get('Host', '') not in allowed:
            self._json(403, {'error': 'loopback host required'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://{h}' for h in allowed}:
            self._json(403, {'error': 'cross-origin requests are disabled'})
            return False
        return True

    def do_GET(self):
        if not self._local():
            return
        url = urlsplit(self.path)
        route = unquote(url.path)
        try:
            if route == '/health':
                self._json(200, {'state': self.service.snapshot()['state'], 'backend': 'native-mujoco'})
            elif route == '/api/status':
                status = self.service.snapshot()
                model=self.server.harness.settings.public()
                status.update({'llm': model['configured'],
                               'learning_job': self.server.learning.status(),
                               'agent_events': self.server.harness.notifications(),
                               'agent': self.server.harness.workspace.summary(status),
                               'llm_vision': model['vision'],
                               'llm_model': model['model'],
                               'runs': sorted([p.name for p in (ROOT / 'runs').glob('*') if p.is_dir()], reverse=True)[:30]})
                self._json(200, status)
            elif route == '/api/model':
                self._json(200,self.server.harness.settings.public())
            elif route == '/api/agent':
                role = parse_qs(url.query).get('role', ['execute'])[0]
                snapshot = self.service.snapshot()
                epoch = snapshot['observation_id'].split(':')[0]
                self._json(200, self.server.harness.workspace.public(role, epoch, snapshot.get('memory_context')))
            elif route == '/api/tools':
                role = parse_qs(url.query).get('role', ['execute'])[0]
                self._json(200, {'tools': tools_for(role), 'role': role, 'execution_domain': 'simulation only'})
            elif route == '/api/skills':
                self._json(200, self.server.harness.invoke('list_skills', {}))
            elif route == '/api/memory':
                query = parse_qs(url.query)
                snapshot = self.service.snapshot()
                self._json(200, self.server.harness.workspace.memory(query.get('role', ['execute'])[0],
                    snapshot['observation_id'].split(':')[0], query=query.get('q', [''])[0],
                    context=snapshot.get('memory_context'), include_oracle=query.get('oracle', ['0'])[0] == '1',
                    skill=query.get('skill', [None])[0]))
            elif route == '/api/experiments':
                self._json(200, {'experiments': self.server.learning.catalogue()})
            elif route == '/api/calibration':
                from sim.calibration import report
                self._json(200, report())
            elif route == '/api/alignment':
                from sim.alignment import latest
                self._json(200, latest())
            elif route == '/api/frame':
                query = parse_qs(url.query)
                camera = query.get('camera', ['overview'])[0]
                observation_id = query.get('v', [None])[0]
                self._send(200, self.service.frame(camera, observation_id), 'image/jpeg')
            elif route == '/api/manifest':
                self._json(200, self.service.call('manifest'))
            elif route == '/':
                self._file(ROOT / 'ui/index.html', ROOT / 'ui')
            elif route == '/calibrate.json':
                self._file(ROOT / 'calibrate.json', ROOT)
            elif any(route.startswith(prefix) for prefix in ('/ui/', '/reference/', '/runs/')):
                folder, relative = route[1:].split('/', 1)
                self._file(ROOT / folder / relative, ROOT / folder)
            else:
                self._json(404, {'error': 'not found'})
        except ValueError as exc:
            self._json(400, {'error': str(exc)})
        except Exception:
            self._json(500, {'error': 'server request failed; see local terminal'})
            raise

    def do_POST(self):
        if not self._local():
            return
        try:
            if not self.headers.get('Content-Type', '').startswith('application/json'):
                raise ValueError('application/json required')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                raise ValueError('invalid request size')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('JSON object required')
            route = urlsplit(self.path).path
            if route == '/api/command':
                command = body.get('command')
                args = body.get('arguments', {})
                if command not in {'reset', 'capture', 'replay', 'stop'} or not isinstance(args, dict):
                    raise ValueError('unknown command; use the harness for policy execution')
                allowed = {'episode', 'seed', 'record_images', 'realtime', 'label'}
                if command == 'capture':
                    allowed.add('camera')
                if set(args) - allowed:
                    raise ValueError('unknown command parameter')
                if 'camera' in args and (type(args['camera']) is not str
                                          or args['camera'] not in {'operator', 'overview', 'head', 'left', 'right'}):
                    raise ValueError('unknown capture camera')
                if 'episode' in args and (type(args['episode']) is not int or args['episode'] not in (0, 1, 2)):
                    raise ValueError('episode must be 0, 1 or 2')
                if 'seed' in args and (type(args['seed']) is not int or not 0 <= args['seed'] <= 2**32 - 1):
                    raise ValueError('invalid seed')
                for key in ('record_images', 'realtime'):
                    if key in args and type(args[key]) is not bool:
                        raise ValueError('boolean parameter required')
                with self.server.harness.tool_lock:
                    if command in {'reset','replay'} and self.server.harness.chat_lock.locked():
                        raise ValueError('模型正在处理请求，请等待完成后重置或回放')
                    if command == 'replay' and self.server.learning.busy:
                        raise ValueError('训练任务正在运行，请先等待或停止训练')
                    result = self.service.call(command, **args)
                self._json(200, result)
            elif route == '/api/tools':
                self._json(200, self.server.harness.invoke(body.get('name'), body.get('arguments', {}), role=body.get('role', 'execute')))
            elif route == '/api/chat':
                self._json(200, self.server.harness.chat(body.get('message', ''), role=body.get('role', 'execute')))
            elif route == '/api/model':
                self._json(200,self.server.harness.configure_model(body))
            elif route == '/api/memory':
                role = body.get('role', 'execute')
                self._json(200, self.server.harness.save_preference(role, body.get('text'), body.get('id')))
            elif route == '/api/memory/delete':
                self._json(200, self.server.harness.delete_memory(body.get('role', 'execute'), body.get('id')))
            elif route == '/api/view':
                self._json(200,self.service.set_view(body))
            elif route == '/api/model/clear':
                self._json(200,self.server.harness.configure_model())
            elif route == '/api/model/test':
                self._json(200,self.server.harness.settings.test())
            else:
                self._json(404, {'error': 'not found'})
        except (ValueError, KeyError) as exc:
            self._json(400, {'error': str(exc)})
        except RuntimeError as exc:
            self._json(503, {'error': str(exc)})
        except Exception:
            self._json(500, {'error': 'unexpected request failure'})
            raise

    def _file(self, path, root):
        path, root = path.resolve(), root.resolve()
        # Export authorization is for a local source snapshot, not a web route.
        if path.is_relative_to((ROOT / 'reference/source').resolve()):
            self._json(404, {'error': 'file not found'})
            return
        if not path.is_relative_to(root) or not path.is_file():
            self._json(404, {'error': 'file not found'})
            return
        self._send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream')

    def _json(self, status, data):
        self._send(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')

    def _send(self, status, data, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        if args and (str(args[0]).startswith('GET /api/frame') or str(args[0]).startswith('GET /api/status')):
            return
        super().log_message(fmt, *args)


def main():
    try:
        server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    except OSError as exc:
        raise SystemExit(f'端口 {PORT} 无法使用：{exc}。可用 OPENARM_PORT=8766 ./run serve 指定其他端口。') from exc
    try:
        server.simulation = SimulationService()
        server.learning = LearningJobs()
        server.harness = Harness(server.simulation, server.learning,settings=ModelSettings(store=SecretStore()),
                                 workspace=Workspace(ROOT/'harness/state/agent.sqlite3'))
        print(f'OpenArm Lab: http://127.0.0.1:{PORT}/', flush=True)
        print('原生 MuJoCo · 本机仿真 · 不连接真机', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print('正在停止仿真。', flush=True)
    finally:
        if hasattr(server, 'harness'):
            server.harness.close()
        if hasattr(server, 'learning'):
            server.learning.close()
        if hasattr(server, 'simulation'):
            server.simulation.close()
        server.server_close()


if __name__ == '__main__':
    main()
