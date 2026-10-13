"""On-demand multi-object development UI using the standard page structure."""
import os
import signal
import subprocess
import time
from http.server import ThreadingHTTPServer

os.environ.setdefault('MUJOCO_GL', 'egl')


def start_gpu(root):
    """Start only this on-demand worker; never replace an existing listener."""
    from sim.object_curobo import metadata
    import hashlib
    def compatible():
        value = metadata()
        return (value.get('ready') and value.get('scene_protocol') == 'native-scene-inventory/1'
            and value.get('gpu_context_healthy') is True and value.get('worker_port')==8005
            and value.get('scene_worker_source_sha256') == hashlib.sha256((root/'tools/scene_gpu.py').read_bytes()).hexdigest()
            and value.get('scene_cache_worker_source_sha256') == hashlib.sha256((root/'tools/scene_cache.py').read_bytes()).hexdigest()
            and value.get('exact_world_cache_source_sha256') == hashlib.sha256((root/'sim/scene_cache.py').read_bytes()).hexdigest())
    if compatible(): return None
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8005)) == 0: return None
    runtime = root / '.runtime/curobo/bin/python'
    if not runtime.is_file(): return None
    logs = root / '.runtime/logs'; logs.mkdir(exist_ok=True)
    with (logs / 'object-gpu.log').open('ab') as log:
        process = subprocess.Popen([str(runtime), '-B', '-m', 'tools.scene_cache'], cwd=root,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 20
    while process.poll() is None and time.monotonic() < deadline:
        if compatible(): break
        time.sleep(.2)
    return process


def stop_gpu(process):
    if process is None or process.poll() is not None: return
    process.send_signal(signal.SIGINT)
    try: process.wait(timeout=15)
    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)


def build_backend(*, harness_context=True, stable_hold=True, transit_hints=True):
    from harness.composite import install_isolated as composite
    from sim.grasp import install_isolated as grasp
    from harness.candidate import install_isolated as candidate
    from harness.scene import install_isolated as scene
    from harness.object_gpu import install_isolated as objects
    from harness.object_handover import install_isolated as relay
    from harness.object_transport import install_isolated as transport
    from harness.task_queue import install_isolated as tasks
    from harness.poses import PoseWorkspace, install_isolated as poses
    from harness.bottle import BottleHarness, install_isolated as bottle, service_factory
    from sim.environment import ROOT
    os.environ.setdefault('OPENARM_POLICY_CONFIG', str(ROOT/'harness/policy.json'))
    composite(); grasp(); candidate(); scene(); objects(); relay(); transport(); tasks(); poses(); bottle(); factory = service_factory()
    from harness.connected_placement import install_isolated as connected
    factory = connected(factory)
    from harness.fast_grasp import install_isolated as efficient
    factory = efficient(factory)
    from harness.margin_grasp import install_isolated as margin
    factory = margin(factory)
    from harness.free_transport import install_isolated as free
    factory = free(factory)
    from harness.task_placement import install_isolated as task_placement
    factory = task_placement(factory, deployed=True)
    from sim.scene_library import service_factory as library
    factory = library(factory)
    from sim.scene_curobo import install_isolated as scene_gpu
    scene_gpu()
    from sim.cache_curobo import install_isolated as cache_gpu
    cache_gpu()
    # Explicit startup selection preserves the user's optional objects across
    # a controlled development restart. Scene changes still use the physics
    # owner's existing atomic cold-compile command and fresh-observation gate.
    if 'OPENARM_SCENE_OBJECTS' in os.environ:
        import json
        import sim.service as native
        from sim.scene_library import validate_selection
        selected = validate_selection(json.loads(os.environ['OPENARM_SCENE_OBJECTS']))
        original_env = native.OpenArmEnv
        class InitialEnv(original_env):
            def __init__(self, *, selection=None, **kwargs):
                super().__init__(selection=selected if selection is None else selection, **kwargs)
        native.OpenArmEnv = InitialEnv
    from harness.act_adapter import install_isolated as act
    base = BottleHarness
    if harness_context:
        from harness.policy_trace import install_isolated as policy_trace
        policy_trace()
        from harness.stages import install_isolated as stages
        base = stages(reveal_by_stage=True, live_context=True, focused_context=True, bind_atomic=True)
    ActHarness = act(base)
    from harness.intent import install_isolated as intent
    Harness = intent(ActHarness)
    if harness_context:
        from harness.policy_route import install_isolated as policy_route
        Harness = policy_route(Harness)
        from harness.compact import install_isolated as compact
        Harness = compact(Harness)
    if stable_hold:
        from harness.settled_hold import install_isolated as settling
        factory = settling(factory)
    if transit_hints:
        from harness.transit_hint import install_isolated as hints
        Harness = hints(Harness)
    return factory, Harness


def main():
    factory, ActHarness = build_backend()
    from sim.environment import ROOT
    from harness.poses import PoseWorkspace
    from tools.source_identity import SourceIdentity
    source_identity = SourceIdentity(ROOT)
    import server as web
    from tools.contact_workbench import ExperimentArchive
    try:
        port = int(os.environ.get('OPENARM_PORT', '8767'))
    except ValueError as error:
        raise RuntimeError('OPENARM_PORT must be an integer from 1 to 65535') from error
    if not 1 <= port <= 65535:
        raise RuntimeError('OPENARM_PORT must be an integer from 1 to 65535')
    web.PORT = port
    class Handler(web.Handler):
        def do_POST(self):
            from urllib.parse import urlsplit
            if urlsplit(self.path).path != '/api/scene/library': return super().do_POST()
            if not self._local(): return
            if not self._assert_current_source(): return
            try:
                import json
                if not self.headers.get('Content-Type', '').startswith('application/json'):
                    raise ValueError('application/json required')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096: raise ValueError('invalid request size')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) != {'object_ids', 'observation_id'}:
                    raise ValueError('invalid scene request')
                with self.server.harness.tool_lock:
                    if self.server.harness.chat_lock.locked(): raise ValueError('智能体正在处理任务，请等待结束')
                    result = self.service.call('configure_scene', **body)
                self._json(200, result)
            except (ValueError, KeyError) as exc: self._json(400, {'error': str(exc)})
            except RuntimeError as exc: self._json(503, {'error': str(exc)})
        def _file(self, path, root):
            if path.resolve() == (ROOT/'ui/index.html').resolve():
                content = path.read_text().replace('<title>OpenArm Lab</title>', '<title>OpenArm Lab · 多物体开发</title>')
                content = content.replace('仅仿真 · 真机未连接', '独立开发场景 · 真机未连接')
                content = content.replace('<option value="research">实验与训练</option>', '')
                return self._send(200, content.encode(), 'text/html; charset=utf-8')
            return super()._file(path, root)
    class Harness(ActHarness):
        def chat(self, prompt, *, role='execute', **kwargs):
            if role != 'execute': raise ValueError('multi-object cell is execution-only')
            return super().chat(prompt, role=role, **kwargs)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    owned_gpu = None
    try:
        owned_gpu = start_gpu(ROOT)
        server.simulation = factory(); server.learning = ExperimentArchive()
        def workbench_status():
            return {
                **source_identity.status(),
                'factory_profile': 'objects-default',
                'harness_profile': 'live-context+policy-route+compact-context+ACT-adapter',
                'enabled_features': ['stable-hold', 'transit-hints', 'registered-object-scene', 'native-MuJoCo-verification'],
                'policy_config': 'harness/policy.json',
            }
        server.workbench_status = workbench_status
        server.assert_current_source = source_identity.assert_current
        server.harness = Harness(server.simulation, server.learning,
            settings=web.ModelSettings(store=web.SecretStore()), workspace=PoseWorkspace(ROOT/'harness/state/objects.sqlite3'))
        from harness.provider import install_isolated as provider
        provider(server.harness)
        print(f'OpenArm · http://127.0.0.1:{port}/ · 按需启动 · 真机未连接', flush=True)
        server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        if hasattr(server, 'harness'): server.harness.close(); server.harness.workspace.db.close()
        if hasattr(server, 'learning'): server.learning.close()
        if hasattr(server, 'simulation'): server.simulation.close()
        server.server_close()
        stop_gpu(owned_gpu)


if __name__ == '__main__': main()
