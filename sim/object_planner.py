"""Primitive-aware native grasp proposals for registered object identities.

Uses the existing connected IK/path/contact/lift/place implementation.
Only collision prediction attaches a selected free joint in scratch MjData.
"""
import copy

import mujoco
import numpy as np

from .grasp import GeometricPlanner, frames
from .tamp import PlanningFailure

CONTRACT = {'schema': 'primitive-object-grasp/1', 'source': 'simulator_ground_truth',
    'shapes': 'upright box and cylinder collision primitives; bottle uses existing geometric sampler',
    'box': 'face and diagonal jaw directions; three axial bands and four pitches',
    'cylinder': '24 azimuths; three axial bands and four pitches',
    'gates': 'existing connected grasp/pregrasp IK, native path, actual two-finger contact and lift',
    'handover': 'not offered for newly registered shapes; receiver geometry not yet generalized',
    'hardware_access': False, 'transfer_ready': False}


def candidates(env, data):
    for geom in sorted(env.bottle_geoms, key=lambda g: -data.geom_xpos[g, 2]):
        kind = int(env.model.geom_type[geom]); rotation = data.geom_xmat[geom].reshape(3, 3)
        axis = rotation[:, 2]; size = env.model.geom_size[geom]
        if axis[2] < .98: raise PlanningFailure('unsupported_object_orientation', '当前几何候选只覆盖直立的局部Z轴')
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half = size[1]; shifts = [.8, .3, -.3]
            angles = np.deg2rad([-45,-60,-30,-75,-15,-90,0,-105,15,-120,30,-135,45,-150,60,-165,75,180,90,165,105,150,120,135])
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            half = size[2]; shifts = [.65, 0., -.65]
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            angles = yaw + np.deg2rad([-45, -90, 0, -135, 45, 90, 180, 135])
        else: raise PlanningFailure('unsupported_grasp_geometry', '未声明该形状的抓取候选；不会按圆柱假装处理')
        for fraction in shifts:
            point = data.geom_xpos[geom] + axis * half * fraction
            for pitch in np.deg2rad([0, -15, -30, 15]):
                for theta in angles:
                    hand = frames(axis, theta, pitch)
                    width = 2 * size[0] if kind == mujoco.mjtGeom.mjGEOM_CYLINDER else 2 * float(np.abs(rotation.T @ hand[:, 1]) @ size)
                    opening = sum(max(f.range) - min(f.range) for f in env.fingers['left'])
                    if width + .004 > opening: continue
                    yield {'geom': env.model.geom(geom).name, 'shape': mujoco.mjtGeom(kind).name,
                        'grasp_m': point.copy(), 'pregrasp_m': point - .10 * hand[:, 2], 'rotation': hand,
                        'yaw_rad': float(theta), 'pitch_rad': float(pitch), 'axial_fraction': fraction,
                        'required_opening_m': width + .004}


class ObjectPlanner(GeometricPlanner):
    def __init__(self, env, task, **kwargs):
        parameters = copy.deepcopy(kwargs.get('parameters') or {})
        ident = parameters.get('object_id', env.object_id)
        if ident not in env.registry: raise PlanningFailure('unknown_object', '对象未注册为可操作的自由刚体')
        if task in ('handover_bottle', 'handover_to_basket') and ident != 'bottle':
            raise PlanningFailure('unsupported_object_handover', '新形状的接收抓取尚未验证，不能调用瓶子交接模板')
        env.select_object(ident)
        mapped = {'pick_object': 'pick_bottle', 'reach_object': 'reach_bottle', 'place_object': 'place_bottle'}.get(task, task)
        kwargs['parameters'] = parameters
        super().__init__(env, mapped, **kwargs)
        self.result['object_id'] = ident
        if ident != 'bottle':
            self.result['object_grasp_contract'] = CONTRACT
            self.result['task'] = task
            self.result['selected_object'] = env.selection()

    def _at(self, q):
        if self.env.object_id == 'bottle': return super()._at(q)
        self.data.qpos[self.qids] = q
        mujoco.mj_forward(self.env.model, self.data)
        if self.carried is not None:
            hand = self.data.xmat[self.body].reshape(3, 3)
            offset, relative = self.carried
            address = self.env.registry[self.env.object_id]['qposadr']
            self.data.qpos[address:address + 3] = self.data.xpos[self.body] + hand @ offset
            mujoco.mju_mat2Quat(self.data.qpos[address + 3:address + 7], (hand @ relative).ravel())
            mujoco.mj_forward(self.env.model, self.data)

    def _choose_pick_pose(self, ignored):
        if self.env.object_id == 'bottle': return super()._choose_pick_pose(ignored)
        rejected = []; seeds = copy.deepcopy(self.ik_seeds[self.side])
        live_q = self.env.data.qpos.copy(); live_v = self.env.data.qvel.copy()
        try:
            self.planning_qpos = live_q.copy()
            for finger in self.env.fingers[self.side]: self.planning_qpos[finger.qposadr[0]] = max(finger.range)
            self._sync()
            for index, candidate in enumerate(candidates(self.env, self.data)):
                self._check(); self.rotation = candidate['rotation']; self.ik_seeds[self.side] = copy.deepcopy(seeds)
                try:
                    grasp = self._ik(candidate['grasp_m'], True); self.ik_seeds[self.side].insert(0, grasp)
                    pre = self._ik(candidate['pregrasp_m']); self._path(pre)
                    if not self._edge(pre, grasp, True): raise PlanningFailure('approach_collision', '候选接近段发生碰撞')
                    self.result['grasp_search'] = {'contract': CONTRACT, 'object_id': self.env.object_id, 'arm': self.side,
                        'candidate_index': index, 'selected': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in candidate.items()},
                        'rejected': rejected, 'pregrasp_q': pre.tolist(), 'grasp_q': grasp.tolist()}
                    self.result.pop('planning_diagnostics', None); self.ik_seeds[self.side] = [grasp, pre, *seeds]
                    return candidate['grasp_m'].copy(), candidate['pregrasp_m'].copy()
                except PlanningFailure as exc:
                    if exc.code in ('cancelled', 'planning_timeout'): raise
                    rejected.append({'index': index, 'shape': candidate['shape'], 'code': exc.code,
                        'candidate': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in candidate.items()},
                        'diagnostics': copy.deepcopy(self.result.get('planning_diagnostics'))})
            self.result['grasp_search'] = {'contract': CONTRACT, 'object_id': self.env.object_id, 'rejected': rejected}
            raise PlanningFailure('grasp_search_exhausted', '已声明形状候选未找到完整抓取路线；不是物理不可达证明')
        finally:
            self.planning_qpos = None
            np.testing.assert_array_equal(live_q, self.env.data.qpos); np.testing.assert_array_equal(live_v, self.env.data.qvel)

    def actions(self):
        yield from super().actions()
        if self.env.object_id != 'bottle':
            if 'actual_bottle_position_m' in self.result:
                self.result['actual_object_position_m'] = self.result.pop('actual_bottle_position_m')
            self.result['object_id'] = self.env.object_id
