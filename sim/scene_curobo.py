"""Native-owner scene inventory binding around unchanged GPU/native motion."""
import copy
import hashlib
from contextvars import ContextVar

from .environment import ROOT
from .tamp import PlanningFailure
from . import object_curobo as original

CURRENT_ENV = ContextVar('openarm_scene_gpu_env', default=None)
BASE_CALL = original.call


def inventory(env):
    return {ident: [env.model.geom(g).name for g in sorted(row['geoms'])]
            for ident, row in env.registry.items()}


def call(path, data=None, timeout=90):
    return call_with_scene(BASE_CALL, path, data, timeout, CURRENT_ENV.get())


def call_with_scene(base_call, path, data=None, timeout=90, env=None):
    bound = inventory(env) if env is not None and path == '/plan' else None
    if bound is not None:
        data = copy.deepcopy(data); data['native_scene_inventory'] = bound
    result = base_call(path, data, timeout)
    if bound is not None and result.get('native_scene_inventory') != bound:
        raise PlanningFailure('scene_inventory_unacknowledged', 'GPU did not acknowledge the complete current native object registry')
    return result


class SceneCuRoboPlanner(original.ObjectCuRoboPlanner):
    _call_provider = staticmethod(original.call)

    def _gpu_call(self, path, data=None, timeout=90):
        return call_with_scene(self._call_provider,path,data,timeout,CURRENT_ENV.get())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        meta = self.result['backend_metadata']
        expected = hashlib.sha256((ROOT / 'tools/scene_gpu.py').read_bytes()).hexdigest()
        if meta.get('scene_protocol') != 'native-scene-inventory/1' or meta.get('scene_worker_source_sha256') != expected:
            raise PlanningFailure('scene_gpu_unavailable', 'Scene-aware worker must match the saved source before optional-scene motion')
        self.result['native_scene_inventory'] = inventory(self.env)

    def _move(self, *args, **kwargs):
        token = CURRENT_ENV.set(self.env)
        try:
            yield from super()._move(*args, **kwargs)
        finally:
            CURRENT_ENV.reset(token)


def install_isolated():
    from . import service
    service.CuRoboPlanner = SceneCuRoboPlanner
