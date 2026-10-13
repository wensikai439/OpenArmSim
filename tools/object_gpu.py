"""Isolated official cuRobo worker for registered primitive objects, port 8004.

Reuses the reviewed GPU solver/cache/robot geometry. The legacy 8002 process
is unchanged. Collision attachment is a prediction, never a physics weld.
"""
import hashlib
import logging
import time
from http.server import HTTPServer
from pathlib import Path

import numpy as np

from sim.curobo_server import Handler, Planner as BasePlanner, ROOT, validate_scene
from sim.curobo_cache import kinematic_models

GEOMS = {'bottle': ['bottle_body', 'bottle_neck', 'bottle_cap'],
    'block': ['block_shape'], 'can': ['can_shape']}


def validate_attachment(attached, scene):
    validate_scene(scene)
    if attached is None: return None
    if type(attached) is not dict or set(attached) != {'object_id', 'spheres_tcp', 'removed_world_geoms', 'representation'}:
        raise ValueError('bounded registered-object attachment fields required')
    ident = attached['object_id']
    if ident not in GEOMS or attached['removed_world_geoms'] != GEOMS[ident]:
        raise ValueError('object identity and native collision geometry must agree')
    spheres = np.asarray(attached['spheres_tcp'], dtype=np.float32)
    if (spheres.ndim != 2 or spheres.shape[1] != 4 or not 1 <= len(spheres) <= 32
        or not np.isfinite(spheres).all() or np.max(abs(spheres[:, :3])) > .5
        or np.any(spheres[:, 3] <= 0) or np.any(spheres[:, 3] > .1)):
        raise ValueError('invalid bounded attachment spheres')
    if any(name in scene['cuboid'] for name in GEOMS[ident]):
        raise ValueError('selected carried object cannot also be a static world obstacle')
    for other, names in GEOMS.items():
        if other != ident and not all(name in scene['cuboid'] for name in names):
            raise ValueError('unselected registered objects must remain world obstacles')
    return spheres


class Planner(BasePlanner):
    def __init__(self):
        super().__init__()
        self.metadata.update(protocol='openarm-object-curobo/1',
            scope='single-arm empty/carrying registered bottle, upright box or cylinder; native recheck required',
            attachment_contract='conservative primitive volume sphere cover; no payload dynamics or physics weld',
            worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            base_worker_source_sha256=hashlib.sha256((ROOT / 'sim/curobo_server.py').read_bytes()).hexdigest(),
            registered_object_geoms=GEOMS, isolation='separate on-demand worker; legacy 8002 unchanged')

    def plan(self, data):
        from curobo.types import JointState, GoalToolPose
        self.state(data); side = data['arm']
        if side not in ('left', 'right'): raise ValueError('invalid arm')
        target = np.asarray(data['position_m'], dtype=float); quat = np.asarray(data['quaternion_wxyz'], dtype=float)
        if (target.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[target, quat]).all()
            or abs(np.linalg.norm(quat) - 1) > .02 or np.any(abs(target) > 2)):
            raise ValueError('invalid bounded goal pose')
        scene = data['scene']; attached = data.get('attachment'); spheres = validate_attachment(attached, scene)
        active = [f'openarm_{side}_joint{i}' for i in range(1, 8)]
        locked = {n: float(data['joints'][n]) for n in self.names if n not in active}
        started = time.monotonic(); planner, cache = self._solver(side, locked, scene, attached is not None)
        js = JointState.from_position(self.torch.tensor([[data['joints'][n] for n in planner.joint_names]],
            device='cuda'), joint_names=planner.joint_names)
        cache['geometry_guard'] = self._geometry_guard(planner, js, data)
        debug = None
        if attached is not None:
            from curobo._src.collision.attachment_manager import AttachmentManager
            seen = set()
            for model in kinematic_models(planner):
                pointer = model.config.kinematics_config.link_spheres.data_ptr()
                if pointer in seen: continue
                seen.add(pointer)
                # This inherited slot name is internal topology, not object identity.
                AttachmentManager(model).update(self.torch.as_tensor(spheres, device='cuda'), js, link_name='carried_bottle')
            cache['attachment_buffers_updated'] = len(seen)
            params = planner.attachment_manager.kinematics_params
            world = planner.kinematics.compute_kinematics(js).get_link_spheres().detach().cpu().numpy().reshape(-1, 4)
            ids = params.get_sphere_index_from_link_name('carried_bottle').detach().cpu().numpy()
            carried = world[ids]; carried = carried[carried[:, 3] > 0]
            pairs = planner.kinematics.get_self_collision_config().collision_pairs.detach().cpu().numpy()
            gaps = np.linalg.norm(world[pairs[:, 0], :3] - world[pairs[:, 1], :3], axis=-1) - world[pairs[:, 0], 3] - world[pairs[:, 1], 3]
            labels = {}
            for name in [*self.config['robot_cfg']['kinematics']['collision_link_names'], 'carried_bottle']:
                for index in params.get_sphere_index_from_link_name(name).detach().cpu().tolist(): labels[index] = name
            debug = {'world_spheres': carried.tolist(), 'checked_self_overlap_m': float(max(0, -gaps.min())),
                'checked_overlapping_pairs': [{'links': [labels.get(int(pairs[i, 0])), labels.get(int(pairs[i, 1]))],
                    'depth_m': float(-gaps[i])} for i in np.argsort(gaps)[:10] if gaps[i] < 0]}
        goal = GoalToolPose(tool_frames=planner.tool_frames,
            position=self.torch.tensor(target.reshape(1, 1, 1, 1, 3), device='cuda', dtype=self.torch.float32),
            quaternion=self.torch.tensor(quat.reshape(1, 1, 1, 1, 4), device='cuda', dtype=self.torch.float32))
        result = planner.plan_pose(goal, js, max_attempts=3)
        success = bool(result is not None and result.success.any().item())
        response = {'success': success, 'seconds': time.monotonic() - started, 'metadata': self.metadata, 'cache': cache,
            'status': 'completed' if success else str(getattr(result, 'status', 'IK found no feasible seed')),
            'joint_names': planner.joint_names}
        if attached is not None:
            response['attachment'] = {'object_id': attached['object_id'], 'sphere_count': len(spheres),
                'removed_world_geoms': attached['removed_world_geoms'], 'diagnostics': debug,
                'scope': 'GPU collision prediction only; native path and actual object contact verification required'}
        if result is None:
            diagnostic = planner.ik_solver.solve_pose(goal, return_seeds=4, current_state=js)
            response['ik_diagnostics'] = {name: getattr(diagnostic, name).detach().cpu().tolist()
                for name in ('success', 'position_error', 'rotation_error', 'feasible') if getattr(diagnostic, name, None) is not None}
        if success:
            interpolated = result.get_interpolated_plan(); values = interpolated.position.detach().cpu().numpy()
            names = interpolated.joint_names
            if names is None or len(names) != values.shape[-1] or not set(active) <= set(names):
                raise ValueError('interpolated joint inventory mismatch')
            path = values.reshape(-1, len(names))[:, [names.index(n) for n in planner.joint_names]]
            if not 2 <= len(path) <= 1200 or not np.isfinite(path).all(): raise ValueError('invalid bounded path')
            response.update(path=path.tolist(), dt=1 / 30)
        return response


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    server = HTTPServer(('127.0.0.1', 8004), Handler)
    try:
        server.planner = Planner(); print('Registered-object cuRobo · 127.0.0.1:8004 · no hardware', flush=True)
        server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        if getattr(server, 'planner', None) and server.planner.cached: server.planner.cached[1].destroy()
        server.server_close()


if __name__ == '__main__': main()
