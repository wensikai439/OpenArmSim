"""On-demand multi-object development UI using the standard page structure."""
import os
import signal
import subprocess
import time
from http.server import ThreadingHTTPServer

os.environ.setdefault('MUJOCO_GL', 'egl')

_RUNTIME_TRACE = {'attempted': False, 'complete': False, 'profile': None,
                  'global_layers': [], 'factory_layers': [], 'planner_layers': [],
                  'backend_layers': [], 'harness_layers': [], 'failed_layer': None}


def _claim_backend_build():
    if _RUNTIME_TRACE['attempted']:
        raise RuntimeError('工作台安装器修改进程级模块；同一进程只允许build_backend一次，请用全新进程重建。')
    _RUNTIME_TRACE.update(attempted=True, complete=False, profile=None,
                          global_layers=[], factory_layers=[], planner_layers=[],
                          backend_layers=[], harness_layers=[], failed_layer=None)


def _run_layer(kind, name, function, *args, **kwargs):
    try:
        result = function(*args, **kwargs)
    except Exception as exc:
        _RUNTIME_TRACE['failed_layer'] = name
        raise RuntimeError(f'运行时组合层 {name} 安装失败；进程可能已部分修改，请退出并用新进程启动。') from exc
    _RUNTIME_TRACE[kind].append(name)
    return result


def runtime_composition(harness, simulation):
    """Internal API diagnostic for the concrete process-local wrapper stack."""
    import sim.service as service

    def lineage(value, suffix):
        cls = value if isinstance(value, type) else type(value)
        return [f'{row.__module__}.{row.__qualname__}' for row in cls.__mro__
                if row.__name__.endswith(suffix)]

    request_stack=[];layer=harness.request_transport;seen=set()
    while layer is not None and id(layer) not in seen and len(request_stack)<8:
        seen.add(id(layer))
        request_stack.append(f'{type(layer).__module__}.{type(layer).__qualname__}')
        layer=getattr(layer,'base',None)
    resolve=getattr(simulation,'_tamp_planner_factory',None)
    planner=(resolve() if callable(resolve) else
             getattr(simulation,'tamp_planner_factory',None) or service.TampPlanner)
    resolve_relocation=getattr(simulation,'_relocation_planner_factory',None)
    relocation=(resolve_relocation() if callable(resolve_relocation) else
                getattr(simulation,'relocation_planner_factory',None) or service.RelocationPlanner)
    resolve_curobo=getattr(simulation,'_curobo_planner_factory',None)
    curobo=(resolve_curobo() if callable(resolve_curobo) else
            getattr(simulation,'curobo_planner_factory',None) or service.CuRoboPlanner)
    metadata_provider=getattr(simulation,'curobo_metadata_provider',None)
    return {'schema':'openarm-runtime-composition/1','profile':_RUNTIME_TRACE.get('profile'),
        'builder_attempted':_RUNTIME_TRACE['attempted'],'builder_complete':_RUNTIME_TRACE['complete'],
        'failed_layer':_RUNTIME_TRACE['failed_layer'],
        'global_install_order':list(_RUNTIME_TRACE['global_layers']),
        'factory_install_order':list(_RUNTIME_TRACE['factory_layers']),
        'planner_install_order':list(_RUNTIME_TRACE['planner_layers']),
        'backend_composition_order':list(_RUNTIME_TRACE['backend_layers']),
        'harness_composition_order':list(_RUNTIME_TRACE['harness_layers']),
        'harness_mro':lineage(harness,'Harness'),'service_mro':lineage(simulation,'Service'),
        'active_planner_factory':f'{planner.__module__}.{planner.__qualname__}',
        'active_relocation_factory':f'{relocation.__module__}.{relocation.__qualname__}',
        'active_curobo_factory':f'{curobo.__module__}.{curobo.__qualname__}',
        'active_curobo_metadata_provider':(
            f'{metadata_provider.__module__}.{metadata_provider.__qualname__}'
            if metadata_provider is not None else
            f'{service.curobo_metadata.__module__}.{service.curobo_metadata.__qualname__}'),
        'current_request_middleware':request_stack,
        'diagnostic_scope':'internal composition/dispatch provenance; not model capability or task performance'}


def bind_tamp_planner(base, planner, relocation_planner=None, curobo_planner=None,
                      curobo_metadata_provider=None):
    """Bind composed planners to one service class, not the module defaults."""
    if (base.__dict__.get('_bound_tamp_planner_factory') is planner and
        base.__dict__.get('_bound_relocation_planner_factory') is relocation_planner and
        base.__dict__.get('_bound_curobo_planner_factory') is curobo_planner and
        base.__dict__.get('_bound_curobo_metadata_provider') is curobo_metadata_provider):return base
    class BoundTampService(base):
        _bound_tamp_planner_factory=planner
        _bound_relocation_planner_factory=relocation_planner
        _bound_curobo_planner_factory=curobo_planner
        _bound_curobo_metadata_provider=curobo_metadata_provider
        def __init__(self,*args,**kwargs):
            configured=kwargs.get('tamp_planner_factory')
            if configured is not None and configured is not planner:
                raise ValueError('TAMP planner already belongs to this service class')
            configured_relocation=kwargs.get('relocation_planner_factory')
            if (configured_relocation is not None and
                configured_relocation is not relocation_planner):
                raise ValueError('relocation planner already belongs to this service class')
            configured_curobo=kwargs.get('curobo_planner_factory')
            if configured_curobo is not None and configured_curobo is not curobo_planner:
                raise ValueError('cuRobo planner already belongs to this service class')
            configured_metadata=kwargs.get('curobo_metadata_provider')
            if (configured_metadata is not None and
                configured_metadata is not curobo_metadata_provider):
                raise ValueError('cuRobo metadata provider already belongs to this service class')
            kwargs['tamp_planner_factory']=planner
            kwargs['relocation_planner_factory']=relocation_planner
            kwargs['curobo_planner_factory']=curobo_planner
            kwargs['curobo_metadata_provider']=curobo_metadata_provider
            super().__init__(*args,**kwargs)
    return BoundTampService


def start_gpu(root):
    """Start only this on-demand worker; never replace an existing listener."""
    from sim.cache_curobo import metadata
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
    _claim_backend_build()
    import sim.service as native_service
    default_tamp_planner=native_service.TampPlanner
    default_relocation_planner=native_service.RelocationPlanner
    default_curobo_planner=native_service.CuRoboPlanner
    composed_tamp_planner=None
    _RUNTIME_TRACE['profile']={'harness_context':bool(harness_context),
        'stable_hold':bool(stable_hold),'transit_hints':bool(transit_hints)}
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
    for name, installer in (
        ('harness.composite',composite),('sim.grasp',grasp),('harness.candidate',candidate),
        ('harness.scene',scene),('harness.objects',objects),('harness.object_handover',relay),
        ('harness.object_transport',transport),('harness.task_queue',tasks),
        ('harness.poses',poses),('harness.bottle',bottle)):
        _run_layer('global_layers',name,installer)
    factory = _run_layer('factory_layers','harness.bottle.service_factory',service_factory)
    composed_tamp_planner=native_service.TampPlanner
    from harness.connected_placement import compose_planner as connected
    composed_tamp_planner=_run_layer('planner_layers','harness.connected_placement',connected,composed_tamp_planner)
    from harness.margin_grasp import compose_planner as margin
    composed_tamp_planner=_run_layer('planner_layers','harness.margin_grasp',margin,composed_tamp_planner)
    from harness.free_transport import compose_planner as free
    composed_tamp_planner=_run_layer('planner_layers','harness.free_transport',free,composed_tamp_planner)
    from harness.task_placement import install_catalogue as task_placement_catalogue
    from harness.connected_placement import install_catalogue as connected_catalogue
    _run_layer('global_layers','harness.connected_placement.catalogue',connected_catalogue)
    _run_layer('global_layers','harness.task_placement.catalogue',task_placement_catalogue,deployed=True)
    from sim.scene_library import service_factory as library
    factory = _run_layer('factory_layers','sim.scene_library.service_factory',library,factory)
    from sim.cache_curobo import compose_planner as cache_gpu, metadata as cache_metadata
    composed_curobo_planner=_run_layer('backend_layers','sim.cache_curobo',cache_gpu)
    # Explicit startup selection preserves the user's optional objects across
    # a controlled development restart. Scene changes still use the physics
    # owner's existing atomic cold-compile command and fresh-observation gate.
    if 'OPENARM_SCENE_OBJECTS' in os.environ:
        import json
        from sim.scene_library import validate_selection, with_initial_selection
        selected = validate_selection(json.loads(os.environ['OPENARM_SCENE_OBJECTS']))
        factory = _run_layer('factory_layers','OPENARM_SCENE_OBJECTS',with_initial_selection,factory,selected)
    from harness.act_adapter import install_isolated as act
    base = BottleHarness
    if harness_context:
        from harness.policy_trace import install_isolated as policy_trace
        factory = _run_layer('factory_layers','harness.policy_trace',policy_trace,factory)
        from harness.stages import install_isolated as stages
        base = _run_layer('global_layers','harness.stages',stages,
            reveal_by_stage=True, live_context=True, focused_context=True, bind_atomic=True)
    ActHarness = _run_layer('global_layers','harness.act_adapter',act,base)
    from harness.intent import install_isolated as intent
    Harness = _run_layer('harness_layers','harness.intent',intent,ActHarness)
    if harness_context:
        from harness.policy_route import install_isolated as policy_route
        Harness = _run_layer('global_layers','harness.policy_route',policy_route,Harness)
        from harness.compact import install_isolated as compact
        Harness = _run_layer('global_layers','harness.compact',compact,Harness)
    if transit_hints:
        from harness.transit_hint import install_isolated as hints
        Harness = _run_layer('harness_layers','harness.transit_hint',hints,Harness)
    from harness.task_placement import compose_planner as task_placement
    composed_tamp_planner=_run_layer('planner_layers','harness.task_placement',task_placement,
        composed_tamp_planner,deployed=True)
    if stable_hold:
        from harness.settled_hold import compose_planner as settling
        composed_tamp_planner=_run_layer('planner_layers','harness.settled_hold',settling,composed_tamp_planner)
    composed_relocation_planner=native_service.RelocationPlanner
    factory=_run_layer('factory_layers','sim.service.TampPlanner+RelocationPlanner+CuRoboPlanner',bind_tamp_planner,
        factory,composed_tamp_planner,composed_relocation_planner,composed_curobo_planner,cache_metadata)
    native_service.TampPlanner=default_tamp_planner
    native_service.RelocationPlanner=default_relocation_planner
    native_service.CuRoboPlanner=default_curobo_planner
    _RUNTIME_TRACE['complete']=True
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
                'runtime_composition': runtime_composition(server.harness, server.simulation),
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
