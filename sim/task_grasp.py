"""Original grasps first; continuous upright placement proposals on exhaustion.

Only scratch proposal search is changed. The original sampled provider still
runs first, and a failure is retained before trying free-yaw task constraints.
Native pose tolerances, joint bounds, collision/contact/hold/placement and the
original task budget remain mandatory. No package or physical model change.
"""
import copy
import time
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.optimize import minimize

from . import bottle_routing
from .margin_grasp import MarginBottlePick
from .bottle_routing import CONTRACT
from .tamp import PlanningFailure, TampPlanner


def pose_key(point, rotation):
    return (np.asarray(point, dtype=np.float64).tobytes(),
        np.asarray(rotation, dtype=np.float64).tobytes())


def upright_solution(planner, object_goal, position_tolerance):
    """Position + upright axis, with free yaw; accepts only native rechecking."""
    env = planner.env
    object_goal = np.asarray(object_goal).copy()
    initial = planner.data.qpos[planner.qids].copy()
    limits = np.asarray([env.model.joint(f'openarm_{planner.side}_joint{i}').range for i in range(1, 8)])
    margin = CONTRACT['grasp_joint_slack_rad'] + 1e-5
    low, high = limits[:, 0] + margin, limits[:, 1] - margin
    jp = np.zeros((3, env.model.nv)); jr = jp.copy()
    previous = getattr(planner, '_kinematics_only', False)
    planner._kinematics_only = True
    cached = None
    evaluations = 0
    started = time.monotonic()

    def evaluate(q):
        nonlocal cached, evaluations
        planner._check()
        if cached is not None and np.array_equal(q, cached[0]):
            return cached[1:]
        planner._at(q)
        origin = planner.data.xpos[env.bottle_id].copy()
        rotation = planner.data.xmat[env.bottle_id].reshape(3, 3).copy()
        axis = rotation[:, 2]
        mujoco.mj_jac(env.model, planner.data, jp, jr, origin, planner.body)
        # d(axis)/dq = angular_velocity x axis, at the carried body's origin.
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        axis_jac = -skew @ jr[:, planner.dids]
        cached = (q.copy(), np.r_[origin-object_goal, axis[:2]],
            np.vstack([jp[:, planner.dids], axis_jac[:2]]), float(axis[2]), axis_jac[2].copy(), rotation)
        evaluations += 1
        return cached[1:]

    def feasible(q):
        if q.shape != (7,) or not np.isfinite(q).all():
            return False
        error, _, z, _, _ = evaluate(q)
        return (np.linalg.norm(error[:3]) < position_tolerance*.5
            and np.linalg.norm(error[3:]) < .001 and z >= .999
            and np.all(q >= low) and np.all(q <= high))

    class FeasiblePose(Exception):
        pass

    def callback(q):
        if feasible(q):
            raise FeasiblePose(q.copy())

    audit = []
    try:
        seeds = [initial, *planner.ik_seeds[planner.side]]
        unique = []
        for seed in seeds:
            seed = np.clip(seed, low, high)
            if not any(np.array_equal(seed, other) for other in unique):
                unique.append(seed)
            if len(unique) == 4:
                break
        for index, seed in enumerate(unique):
            cached = None
            try:
                result = minimize(lambda q: .5*float((q-seed)@(q-seed)), seed,
                    jac=lambda q: q-seed, method='SLSQP', bounds=list(zip(low, high)),
                    constraints=[{'type': 'eq', 'fun': lambda q: evaluate(q)[0],
                        'jac': lambda q: evaluate(q)[1]},
                        {'type': 'ineq', 'fun': lambda q: evaluate(q)[2]-.999,
                         'jac': lambda q: evaluate(q)[3]}],
                    callback=callback, options={'maxiter': 100, 'ftol': 1e-10})
            except FeasiblePose as found:
                result = SimpleNamespace(x=found.args[0], success=False, nit=None,
                    message='Native task constraints passed before minimum-motion convergence')
            q = np.asarray(result.x)
            ok = feasible(q)
            row = {'seed_index': index, 'optimizer_success': bool(result.success),
                'optimizer_message': str(result.message), 'accepted': False, 'native_collision_checked': False}
            if ok:
                error, _, axis_z, _, object_rotation = evaluate(q)
                yaw = float(np.arctan2(object_rotation[1, 0], object_rotation[0, 0]))
                c, s = np.cos(yaw), np.sin(yaw)
                desired_object_rotation = np.array([[c,-s,0], [s,c,0], [0,0,1.]])
                offset, relative = planner.carried
                desired_hand = desired_object_rotation @ relative.T
                desired_tcp = object_goal + desired_hand @ (planner.local_point-offset)
                actual_tcp, actual_hand = planner._pose()
                quat = np.empty(4); angle = np.empty(3)
                mujoco.mju_mat2Quat(quat, (desired_hand @ actual_hand.T).ravel())
                mujoco.mju_quat2Vel(angle, quat, 1.)
                tcp_error = float(np.linalg.norm(actual_tcp-desired_tcp))
                angle_error = float(np.linalg.norm(angle))
                posed = tcp_error < position_tolerance and angle_error < .018
                collision = planner._collision(q, True) if posed else None
                row.update(object_error_m=float(np.linalg.norm(error[:3])), upright_cosine=axis_z,
                    TCP_error_m=tcp_error, rotation_error_rad=angle_error, native_collision_checked=posed,
                    collision=collision, collision_pair=copy.deepcopy(planner.last_collision) if collision else None,
                    yaw_rad=yaw, q_rad=q.tolist(),
                    minimum_joint_slack_rad=float(np.minimum(q-limits[:,0],limits[:,1]-q).min()))
                if posed and collision is False:
                    row['accepted'] = True
                    audit.append(row)
                    planner.result['last_task_placement_pose'] = row.copy()
                    return q.copy()
            audit.append(row)
        raise PlanningFailure('task_pose_unreachable', 'No independently checked upright free-yaw placement proposal')
    finally:
        planner._kinematics_only = previous
        planner.result.setdefault('private_task_pose_queries', []).append({
            'source': 'scratch position/upright task; continuous yaw selected by existing SciPy SLSQP',
            'object_goal_m': object_goal.tolist(), 'position_tolerance_m': position_tolerance,
            'rotation_tolerance_rad': .018, 'required_joint_margin_rad': CONTRACT['grasp_joint_slack_rad'],
            'native_evaluations': evaluations, 'seconds': time.monotonic()-started,
            'grasp_relation': {'offset_m': planner.carried[0].tolist(), 'relative_rotation': planner.carried[1].tolist()},
            'seeds': audit, 'proposal_only': True, 'hardware_access': False})


class TaskBottlePick(MarginBottlePick):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_fallback = False
        self._original_grasps = {}

    def _collision(self, q, allow_fingers=False):
        if not (self._task_fallback and self.phase == 'place' and self.carried is not None):
            return super()._collision(q, allow_fingers)
        # Actual ConnectedBottlePlacement inherits this exact native collision
        # predicate. The pick-only extra12mm non-target clearance made its
        # lookahead reject poses that the real placement provider admits.
        # Use the EXECUTION provider's original predicate only in this scratch
        # placement phase; grasp/path clearances outside it stay untouched.
        previous = getattr(self, '_kinematics_only', False)
        self._kinematics_only = False
        try:
            return TampPlanner._collision(self, q, allow_fingers)
        finally:
            self._kinematics_only = previous

    def _ik(self, target, allow_fingers=False, *, position_tolerance=.0015):
        if self._task_fallback and self.phase == 'place' and self.carried is not None:
            self._sync()
            object_goal = np.asarray(target) - self.rotation @ (self.local_point-self.carried[0])
            return upright_solution(self, object_goal, position_tolerance)
        q = super()._ik(target, allow_fingers, position_tolerance=position_tolerance)
        if not self._task_fallback and self.phase == 'approach' and allow_fingers and self.carried is None:
            self._original_grasps[pose_key(target, self.rotation)] = q.copy()
        return q

    def _choose_pick_pose(self, ignored):
        try:
            return super()._choose_pick_pose(ignored)
        except PlanningFailure as exc:
            if (exc.code != 'grasp_search_exhausted' or not self.parameters.get('next_placement')
                or self.parameters.get('next_handover')):
                raise
            original = copy.deepcopy(self.result['grasp_search'])
            values = []
            for row in original['rejected']:
                if row['code'] != 'next_place_unreachable':
                    continue
                value = copy.deepcopy(row['candidate'])
                for key in ('grasp_m', 'pregrasp_m', 'rotation'):
                    value[key] = np.asarray(value[key])
                seed = self._original_grasps.get(pose_key(value['grasp_m'], value['rotation']))
                if seed is not None:
                    values.append(value)
            self.result['original_grasp_search_failure'] = original
            self.result['task_pose_fallback'] = {'trigger': exc.code, 'eligible_original_candidates': len(values),
                'original_sampler_first': True, 'original_goal_and_native_gates_preserved': True,
                'placement_collision_provider': 'same native predicate as ConnectedBottlePlacement',
                'changed_proposal_condition': 'pick-only extra12mm clearance does not apply to placement lookahead',
                'source': 'continuous upright-placement proposal after discrete yaw pool exhaustion'}
            if not values:
                raise
        self._task_fallback = True
        # Single physics owner, scoped to this candidate only. Nothing in the
        # live scene/model is patched, and the global function is restored.
        try:
            with patch.object(bottle_routing, 'candidates', lambda bands: iter(copy.deepcopy(values))):
                points = super()._choose_pick_pose(ignored)
            pose = self.result['last_task_placement_pose']
            self.result['future_placement'].update(yaw_rad=pose['yaw_rad'],
                yaw_source='accepted native free-yaw low-placement pose', task_pose_fallback=True)
            return points
        finally:
            self._task_fallback = False
