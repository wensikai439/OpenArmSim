"""Isolated native client for explicitly identified8005cache candidate."""
import hashlib
import json
import urllib.request
import urllib.error

from . import scene_curobo,object_curobo,service
from .environment import ROOT
from .tamp import PlanningFailure


def call(path,data=None,timeout=90):
    request=urllib.request.Request('http://127.0.0.1:8005'+path,
        None if data is None else json.dumps(data,allow_nan=False).encode(),
        {} if data is None else {'Content-Type':'application/json'})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=timeout) as response:
            raw=response.read(1024*1024+1)
    except urllib.error.HTTPError as exc:
        raise PlanningFailure('object_gpu_rejected',exc.read(4096).decode()) from None
    except (OSError,ValueError) as exc:
        raise PlanningFailure('object_gpu_unavailable',str(exc)) from None
    if len(raw)>1024*1024:raise ValueError('GPU response exceeds original limit')
    return json.loads(raw)


class CacheCuRoboPlanner(scene_curobo.SceneCuRoboPlanner):
    CONTRACT={**scene_curobo.SceneCuRoboPlanner.CONTRACT,'worker_port':8005,
        'world_cache':'same topology; exact official GPU scene update with independent readback/pointer guard'}
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        meta=self.result['backend_metadata']
        expected={'scene_cache_worker_source_sha256':'tools/scene_cache.py',
            'exact_world_cache_source_sha256':'sim/scene_cache.py'}
        if meta.get('world_cache_candidate') is not True or meta.get('worker_port')!=8005:
            raise PlanningFailure('world_cache_worker_mismatch','Explicit8005candidate required')
        for field,name in expected.items():
            if meta.get(field)!=hashlib.sha256((ROOT/name).read_bytes()).hexdigest():
                raise PlanningFailure('world_cache_source_mismatch','Candidate must match the frozen scene-cache source')
        self.result['worker_port']=8005


def install_isolated():
    scene_curobo.BASE_CALL=call
    object_curobo.call=scene_curobo.call
    service.CuRoboPlanner=CacheCuRoboPlanner
