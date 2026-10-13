"""Isolated candidate: exact cuboid updates without rebuilding the GPU solver.

Only equal obstacle names/types/counts and equal robot topology can reuse.
Incoming poses/dimensions are copied through official update_world, verified
against independent fresh GPU storage, and captured buffer pointers must stay.
No rounding, stale obstacles, native physics change or solver setting change.
"""
import copy
import hashlib
import json
import time


def topology_key(side,locked,scene,carrying):
    from .curobo_server import validate_scene
    validate_scene(scene)
    # The currently reviewed worker accepts cuboids only. Unknown fields or
    # obstacle types must fail, not silently vanish from a reduced cache key.
    for row in scene['cuboid'].values():
        if set(row)!={'dims','pose'}:raise ValueError('unreviewed world fields for scene reuse')
    return json.dumps([side,sorted(locked),sorted(scene['cuboid']),bool(carrying)],sort_keys=True)


def update_exact_world(planner,scene,torch):
    from curobo._src.geom.types import SceneCfg
    from curobo._src.geom.data.data_cuboid import CuboidData
    checker=planner.scene_collision_checker
    data=checker.data.cuboids
    if data is None or data.num_envs!=1:raise ValueError('reviewed single cuboid environment required')
    fields=('dims','inv_pose','enable','count')
    pointers={field:getattr(data,field).data_ptr() for field in fields}
    cfg=SceneCfg.create(copy.deepcopy(scene))
    fresh=CuboidData.from_scene_cfg(cfg,data.device_cfg,env_idx=0,num_envs=1,max_n=data.max_n)
    started=time.monotonic()
    planner.update_world(cfg)
    torch.cuda.synchronize()
    updated=checker.data.cuboids
    if any(getattr(updated,field).data_ptr()!=ptr for field,ptr in pointers.items()):
        raise ValueError('world update replaced captured GPU buffers')
    if updated.names!=fresh.names or any(not torch.equal(getattr(updated,field),getattr(fresh,field)) for field in fields):
        raise ValueError('world update differs from fresh official GPU scene storage')
    return {'same_gpu_buffers':True,'same_as_fresh_official_scene_storage':True,
        'obstacle_count':len(scene['cuboid']),'seconds':time.monotonic()-started,
        'incoming_world_sha256':hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest(),
        'scope':'exact float32 scene encoding, not physical calibration'}


class SceneCacheMixin:
    """Used only by a single-owner HTTP candidate worker, never live physics."""
    def _solver(self,side,locked,scene,carrying):
        from . import curobo_server as base
        key_function=base.cache_key
        # Base._solver keeps its reviewed numerical lock refresh and complete
        # rebuild path. This private single-owner worker has no concurrent
        # requests; the global is restored even when base construction fails.
        base.cache_key=topology_key
        try:planner,info=super()._solver(side,locked,scene,carrying)
        finally:base.cache_key=key_function
        try:
            info['exact_world_update']=update_exact_world(planner,scene,self.torch)
        except Exception:
            # A partially updated scene can never dispatch a path or be reused.
            planner.destroy();self.cached=None;self.cached_locks=None
            self.refresh_fk=None;self.warmed_generation=0
            raise
        return planner,info
