"""Registered-object GPU proposals followed by native path/contact execution."""
import hashlib
import json
import time
import urllib.error
import urllib.request

import mujoco
import numpy as np

from .environment import ROOT
from .curobo import joint_state, scene
from .object_planner import ObjectPlanner
from .tamp import PlanningFailure


def call(path, data=None, timeout=90):
    request = urllib.request.Request('http://127.0.0.1:8004' + path,
        None if data is None else json.dumps(data, allow_nan=False).encode(),
        {} if data is None else {'Content-Type': 'application/json'})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024: raise ValueError('GPU response exceeds limit')
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        raise PlanningFailure('object_gpu_rejected', exc.read(4096).decode()) from None
    except (OSError, ValueError) as exc:
        raise PlanningFailure('object_gpu_unavailable', str(exc)) from None


def metadata():
    try:
        result = call('/metadata', timeout=.5)
        expected = hashlib.sha256(b''.join((ROOT / 'task/planners' / name).read_bytes()
            for name in ('openarm.json', 'openarm.urdf'))).hexdigest()
        if result.get('protocol') != 'openarm-object-curobo/1' or result.get('geometry_sha256') != expected:
            raise ValueError('registered-object GPU geometry/protocol mismatch')
        for field, name in [('worker_source_sha256', 'tools/object_gpu.py'),
            ('base_worker_source_sha256', 'sim/curobo_server.py'), ('cache_source_sha256', 'sim/curobo_cache.py')]:
            if result.get(field) != hashlib.sha256((ROOT / name).read_bytes()).hexdigest():
                raise ValueError('GPU source changed; restart idle object worker before new experiments')
        return result
    except (PlanningFailure, ValueError, OSError) as exc:
        return {'backend': 'curobo', 'ready': False, 'reason': str(exc)}


def attachment(env, side):
    """Enclose every primitive volume; never fit just its visible surface."""
    hand = env.data.xmat[env.ee_ids[side]].reshape(3, 3)
    tcp = env.data.xpos[env.ee_ids[side]] + hand @ np.array([0, .006, .075])
    spheres = []; names = []
    for geom in sorted(env.bottle_geoms):
        kind = env.model.geom_type[geom]; size = env.model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half = size[1]; radial_square = size[0] ** 2
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            half = size[2]; radial_square = size[0] ** 2 + size[1] ** 2
        else: raise PlanningFailure('object_attachment_geometry', 'Only registered box/cylinder volumes have a verified cover')
        count = max(1, int(np.ceil(2 * half / .012))); slab = 2 * half / count
        rotation = env.data.geom_xmat[geom].reshape(3, 3)
        for index in range(count):
            world = env.data.geom_xpos[geom] + rotation @ np.array([0, 0, -half + (index + .5) * slab])
            spheres.append([*(hand.T @ (world - tcp)).tolist(), float(np.sqrt(radial_square + (slab / 2) ** 2))])
        names.append(env.model.geom(geom).name)
    if not 1 <= len(spheres) <= 32 or any(s[3] > .1 for s in spheres):
        raise PlanningFailure('object_attachment_geometry', 'Primitive cover exceeds the existing 32-sphere/radius contract')
    return {'object_id': env.object_id, 'spheres_tcp': spheres, 'removed_world_geoms': names,
        'representation': 'each complete primitive axial slab enclosed by a sphere; native selected-freejoint path recheck mandatory'}


class ObjectCuRoboPlanner(ObjectPlanner):
    CONTRACT = {**ObjectPlanner.CONTRACT, 'planner': 'official cuRobo V2 GPU IK/trajectory optimization',
        'scope': 'single-arm pose target with registered bottle/box/cylinder; native path/contact checks, no physical calibration',
        'native_path_revalidation': True, 'fallback': False, 'worker_port': 8004}

    def __init__(self, env, task, **kwargs):
        if task not in ('move_to_pose', 'move_relative'): raise PlanningFailure('unsupported_curobo_skill', 'GPU supports bounded TCP motion only')
        side = (kwargs.get('parameters') or {}).get('arm')
        if side in ('left', 'right') and env.evaluation['held']['right' if side == 'left' else 'left']:
            raise PlanningFailure('curobo_other_arm_hold_not_supported', 'Other-arm carrying has no verified locked-arm payload geometry')
        super().__init__(env, task, **kwargs)
        self.result['payload_state'] = 'single_hold' if env.evaluation['held'][side] else 'empty'
        self.deadline = time.monotonic() + 180; self.result['backend_metadata'] = metadata()
        if not self.result['backend_metadata'].get('ready'):
            raise PlanningFailure('object_gpu_unavailable', self.result['backend_metadata'].get('reason', 'GPU unavailable'))
        self.result['plans'] = []

    def _move(self, target, allow_fingers=False):
        command = np.array(target, copy=True); command_rotation = self.rotation.copy()
        for correction in range(4):
            self._sync(); self._check(); carrying = self.carried is not None
            if carrying:
                if not self.env.evaluation['held'][self.side]: raise PlanningFailure('grasp_lost', 'Actual object hold lost before GPU replan')
                hand = self.env.data.xmat[self.body].reshape(3, 3)
                self.carried = (hand.T @ (self.env.data.xpos[self.env.bottle_id] - self.env.data.xpos[self.body]),
                    hand.T @ self.env.data.xmat[self.env.bottle_id].reshape(3, 3))
            quat = np.empty(4); mujoco.mju_mat2Quat(quat, command_rotation.ravel())
            payload = {'arm': self.side, 'joints': joint_state(self.env), 'scene': scene(self.env, carrying=carrying),
                'position_m': command.tolist(), 'quaternion_wxyz': quat.tolist()}
            if carrying: payload['attachment'] = attachment(self.env, self.side)
            fk = call('/fk', {'joints': payload['joints']}); index = fk['tool_frames'].index(self.side + '_candidate_tcp')
            point, rotation = self._pose()
            fk_error = float(np.linalg.norm(np.asarray(fk['position']).reshape(2, 3)[index] - point))
            self.result['fk_consistency_error_m'] = fk_error
            if fk_error > 1e-5: raise PlanningFailure('curobo_fk_mismatch', 'GPU/native software frames disagree')
            response = call('/plan', payload); self._check()
            if carrying and (response.get('attachment', {}).get('object_id') != self.env.object_id
                or response['attachment'].get('removed_world_geoms') != payload['attachment']['removed_world_geoms']):
                raise PlanningFailure('curobo_attachment_missing', 'GPU did not acknowledge the actual selected object geometry')
            self.result['plans'].append({k: v for k, v in response.items() if k != 'path'})
            if not response['success']: raise PlanningFailure('curobo_no_path', 'GPU found no feasible path: ' + response['status'])
            names = [f'openarm_{self.side}_joint{i}' for i in range(1, 8)]
            if set(response['joint_names']) != set(names): raise PlanningFailure('curobo_joint_mismatch', 'Returned joints disagree')
            path = np.asarray(response['path'], dtype=float)[:, [response['joint_names'].index(n) for n in names]]
            bounds = np.array([self.env.model.joint(n).range for n in names])
            if (path.ndim != 2 or path.shape[1] != 7 or not 2 <= len(path) <= 1200 or not np.isfinite(path).all()
                or np.any(path < bounds[:, 0] - 1e-5) or np.any(path > bounds[:, 1] + 1e-5)):
                raise PlanningFailure('curobo_invalid_path', 'Returned path violates original bounds')
            measured = self.env.data.qpos[self.qids].copy()
            if np.max(abs(path[0] - measured)) > .02: raise PlanningFailure('curobo_start_mismatch', 'Returned start differs from measured joints')
            previous = measured
            for q in path:
                if not self._edge(previous, q, carrying):
                    self.result['native_blocked_pair'] = self.last_collision
                    raise PlanningFailure('native_collision_rejected', 'Native selected-object path recheck refused; no dispatch')
                previous = q
            self._sync()
            for start, end in zip(path[:-1], path[1:]):
                count = max(1, int(np.ceil(1 / self.speed_scale)), int(np.ceil(np.max(abs(end - start)) / .018)))
                for i in range(1, count + 1):
                    self._check(); action = self.base_action.copy()
                    action[self.start:self.start + 7] = start + (end - start) * i / count
                    yield action, {'phase': 'move', 'planner': 'curobo', 'object_id': self.env.object_id,
                        'path_points': len(path), 'feedback_correction': correction, 'scene_source': 'simulator_ground_truth'}
                    yield from self._check_hold(action)
                self.base_action[self.start:self.start + 7] = end
            for _ in range(15):
                yield self.base_action.copy(), {'phase': 'move', 'settling': True, 'planner': 'curobo', 'object_id': self.env.object_id}
                yield from self._check_hold(self.base_action)
            self._sync(); point, rotation = self._pose(); error = target - point
            quat = np.empty(4); mujoco.mju_mat2Quat(quat, (self.rotation @ rotation.T).ravel())
            angular = np.empty(3); mujoco.mju_quat2Vel(angular, quat, 1.)
            self.result['feedback_position_error_m'] = float(np.linalg.norm(error))
            self.result['feedback_rotation_error_rad'] = float(np.linalg.norm(angular))
            if np.linalg.norm(error) <= .012 and np.linalg.norm(angular) <= .06: return
            command += np.clip(error, -.04, .04); angle = float(np.linalg.norm(angular))
            if angle > 1e-8:
                quat = np.empty(4); matrix = np.empty(9)
                mujoco.mju_axisAngle2Quat(quat, angular / angle, min(angle, .15)); mujoco.mju_quat2Mat(matrix, quat)
                command_rotation = matrix.reshape(3, 3) @ command_rotation
        raise PlanningFailure('tracking_error', 'GPU path exceeded original native TCP tracking bounds')
