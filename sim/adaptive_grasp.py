"""Efficient native IK and optional roll alternatives for registered bottles.

Only the IK iteration skips dynamics; every candidate acceptance and path edge
uses the original full native collision computation. No live state is edited.
The complete original pool is attempted before new roll hypotheses. Neither a
scratch grasp nor pool exhaustion proves a physical grasp or unreachability.
"""
import copy
import time
from collections import Counter

import mujoco
import numpy as np

from .bottle_routing import BottlePickPlanner, CONTRACT as BOTTLE_CONTRACT
from .grasp import candidates, cylinder_bands, frames
from .object_handover import ObjectHandoverPlanner
from .oriented_grasp import OrientedPlanner
from .tamp import PlanningFailure


class KinematicIK:
    """MuJoCo's documented minimal position/Jacobian pipeline, scratch only."""
    def _search_feedback(self, force=False):
        callback = getattr(self, 'search_feedback', None)
        now = time.monotonic()
        if callback is not None and (force or now - getattr(self, '_last_search_feedback', 0.) >= .5):
            self._last_search_feedback = now
            callback(copy.deepcopy(self.result.get('private_search_progress', {})))

    def _at(self, q):
        if not getattr(self, '_kinematics_only', False):
            return super()._at(q)
        self.data.qpos[self.qids] = q
        mujoco.mj_kinematics(self.env.model, self.data)
        if self.carried is not None:
            hand = self.data.xmat[self.body].reshape(3, 3)
            offset, relative = self.carried
            address = self.env.registry[self.env.object_id]['qposadr']
            self.data.qpos[address:address + 3] = self.data.xpos[self.body] + hand @ offset
            mujoco.mju_mat2Quat(self.data.qpos[address + 3:address + 7], (hand @ relative).ravel())
            mujoco.mj_kinematics(self.env.model, self.data)
        mujoco.mj_comPos(self.env.model, self.data)
        self.stats['kinematic_iterations'] = self.stats.get('kinematic_iterations', 0) + 1

    def _collision(self, q, allow_fingers=False):
        previous = getattr(self, '_kinematics_only', False)
        self._kinematics_only = False
        try:
            # Do not inspect contacts left over from an earlier kinematic query.
            return super()._collision(q, allow_fingers)
        finally:
            self._kinematics_only = previous

    def _ik(self, *args, **kwargs):
        previous = getattr(self, '_kinematics_only', False)
        self._kinematics_only = True
        started = time.monotonic()
        progress = self.result.setdefault('private_search_progress', {
            'source': 'scratch native search only; never task success evidence',
            'IK_calls': 0, 'IK_wall_seconds': 0., 'IK_failures': {}})
        progress['IK_calls'] += 1
        try:
            solved = super()._ik(*args, **kwargs)
            # A prior rejected IK query cannot explain a later accepted grasp
            # rejected for another reason (e.g. joint slack). Keep actual joint
            # limits in private feedback; never change the solver or gates.
            self.result.pop('planning_diagnostics', None)
            limits = np.asarray([self.env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1, 8)])
            slack = np.minimum(solved-limits[:, 0], limits[:, 1]-solved)
            index = int(np.argmin(slack))
            progress['last_accepted_ik'] = {'arm': self.side, 'q_rad': solved.tolist(),
                'minimum_joint_slack_rad': float(slack[index]),
                'limiting_joint': f'openarm_{self.side}_joint{index+1}',
                'source': 'accepted scratch IK and native joint limits; not executed motion or physical reachability'}
            return solved
        except PlanningFailure as exc:
            counts = progress['IK_failures']; counts[exc.code] = counts.get(exc.code, 0) + 1
            raise
        finally:
            self._kinematics_only = previous
            progress['IK_wall_seconds'] += time.monotonic() - started
            self._search_feedback()


class FastBottlePick(KinematicIK, BottlePickPlanner):
    """Original proposals and solver, with only scratch pipeline efficiency."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result['ik_pipeline_contract'] = {
            'iterations': 'mj_kinematics then mj_comPos, scratch MjData only',
            'acceptance_and_path_edges': 'original full native forward/collision/clearance',
            'proposal_solver_tolerances_joint_bounds_unchanged': True,
            'live_state_written_by_solver': False, 'hardware_access': False, 'transfer_ready': False}

    def _future_transport(self, grasp):
        if self._obstacles:
            return super()._future_transport(grasp)
        # Optional items may all be removed, but the native table remains.
        # Include it in this scratch proposal query instead of max([]). Its
        # original collision geometry and the existing EXTRA 12mm clearance
        # still apply; no support height, target or live geometry is invented.
        table = self.env.model.geom('cell_table_col').id
        original = self._obstacles
        self._obstacles = [table]
        self.result['empty_scene_transport_reference'] = {
            'geom': 'cell_table_col', 'source': 'native fixed support geometry; optional object list empty',
            'additional_clearance_m': .012, 'scope': 'scratch lookahead only; actual handover still required'}
        try:
            return super()._future_transport(grasp)
        finally:
            self._obstacles = original


def roll_alternatives(values):
    # Keep originals first and unchanged. A pi roll is a proper rotation about
    # local approach Z: TCP/approach stay fixed, pad X and jaw Y reverse. The
    # complete real hand geometry is checked; hand symmetry is not assumed.
    originals = [{**v, 'roll_rad': 0., 'pad_axis_sign': 1} for v in values]
    rolled = [{**v, 'rotation': v['rotation'] @ np.diag([-1., -1., 1.]),
               'roll_rad': float(np.pi), 'pad_axis_sign': -1} for v in values]
    return originals + rolled


class AdaptiveBottlePick(FastBottlePick):
    def _refinements(self, value, code, grasp):
        return []

    def _choose_pick_pose(self, ignored):
        env = self.env; seeds = copy.deepcopy(self.ik_seeds[self.side]); rejected = []
        before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time)
        started = time.monotonic()
        progress = self.result.setdefault('private_search_progress', {
            'source': 'scratch native search only; never task success evidence',
            'IK_calls': 0, 'IK_wall_seconds': 0., 'IK_failures': {}})
        search = self.result['grasp_search'] = {'rejected': rejected}
        def stage(name, index):
            progress.update(stage=name, candidate_index=index, rejected_count=len(rejected),
                rejection_codes=dict(Counter(row['code'] for row in rejected)),
                wall_seconds=time.monotonic() - started)
            self._search_feedback(name in ('selected_scratch_proposal', 'pool_exhausted'))
        try:
            self.planning_qpos = env.data.qpos.copy()
            for finger in env.fingers[self.side]: self.planning_qpos[finger.qposadr[0]] = max(finger.range)
            self._sync(); values = list(candidates(cylinder_bands(env.model, self.data, env.bottle_id)))
            if self.parameters.get('next_handover'):
                shoulders = [env.data.xanchor[env.model.joint(f'openarm_{side}_joint1').id, 1]
                             for side in ('left', 'right')]
                mid = float(np.mean(shoulders)); own = float(shoulders[0 if self.side == 'left' else 1])
                cross_body = (float(env.data.xpos[env.bottle_id, 1]) - mid) * (own - mid) < 0
                values.sort(key=lambda v: ((-1 if cross_body else 1) * v['grasp_m'][2], abs(v['pitch_rad'])))
                ranking = {'source': 'actual shoulder anchors and current bottle side', 'cross_body': cross_body,
                    'ranking': 'high bands first across torso; low bands first on same side'}
            elif self.parameters.get('next_placement'):
                center = env.data.xipos[env.bottle_id].copy()
                values.sort(key=lambda v: (float(np.linalg.norm(v['grasp_m'] - center)) / max(v['radius_m'], 1e-6), abs(v['pitch_rad'])))
                ranking = {'source': 'current native center of mass and cylinder radii',
                    'ranking': 'original lever-arm/contact-width heuristic'}
            else: ranking = {'source': 'original cylinder band/azimuth/pitch order'}
            original_count = len(values); values = roll_alternatives(values)
            self.result['grasp_order'] = {**ranking, 'all_original_candidates_retained': True,
                'originals_before_roll_alternatives': True, 'original_count': original_count,
                'total_count': len(values), 'scope': 'proposal order only; native execution mandatory'}
            limits = np.asarray([env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1, 8)])
            progress.update(total_candidates=len(values), original_candidates=original_count)
            for index, value in enumerate(values):
                stage('grasp_ik', index); self._check()
                self.rotation = value['rotation']; self.ik_seeds[self.side] = copy.deepcopy(seeds)
                if '_seed_q' in value: self.ik_seeds[self.side].insert(0, value['_seed_q'].copy())
                grasp = None
                try:
                    grasp = self._ik(value['grasp_m'], True)
                    if float(np.minimum(grasp - limits[:, 0], limits[:, 1] - grasp).min()) < BOTTLE_CONTRACT['grasp_joint_slack_rad']:
                        raise PlanningFailure('grasp_near_joint_stop', 'Candidate leaves no declared joint slack for loaded correction')
                    self.ik_seeds[self.side].insert(0, grasp); stage('pregrasp_ik', index)
                    pre = self._ik(value['pregrasp_m']); stage('connecting_path', index); self._path(pre)
                    stage('approach_edge', index)
                    if not self._edge(pre, grasp, True):
                        raise PlanningFailure('approach_collision', 'Native bottle approach edge collides')
                    future = transport = None; stage('goal_lookahead', index)
                    if self.parameters.get('next_handover'): transport = self._future_transport(grasp)
                    else: future = OrientedPlanner._future_place(self, grasp)
                    search.update(selected=ObjectHandoverPlanner._public(value), candidate_index=index,
                        pregrasp_q=pre.tolist(), grasp_q=grasp.tolist())
                    if future: self.result['future_placement'] = future
                    if transport: self.result['future_transport'] = transport
                    self.result.pop('planning_diagnostics', None); self.ik_seeds[self.side] = [grasp, pre, *seeds]
                    stage('selected_scratch_proposal', index)
                    return value['grasp_m'].copy(), value['pregrasp_m'].copy()
                except PlanningFailure as exc:
                    if exc.code in ('cancelled', 'planning_timeout'):
                        progress.update(interrupted_code=exc.code, wall_seconds=time.monotonic() - started)
                        raise
                    rejected.append({'index': index, 'candidate': ObjectHandoverPlanner._public(value),
                        'code': exc.code, 'diagnostics': copy.deepcopy(self.result.get('planning_diagnostics'))})
                    extra = self._refinements(value, exc.code, grasp)
                    if extra:
                        values[index + 1:index + 1] = extra
                        progress['total_candidates'] = len(values)
            stage('pool_exhausted', len(values))
            raise PlanningFailure('grasp_search_exhausted', 'Declared native bottle pool exhausted; not proof of physical unreachability')
        finally:
            self.planning_qpos = None
            for old, new in zip(before[:3], (env.data.qpos, env.data.qvel, env.data.ctrl)):
                np.testing.assert_array_equal(old, new)
            assert before[3] == env.data.time


class RefinedBottlePick(AdaptiveBottlePick):
    """Refine actual solvable grasps locally before abandoning their band.

Neighbors derive from current cylinder dimensions and the sampler's angular
spacing, never from a failed case's desired world coordinates. A rejected
joint vector is only an IK seed. Original slack/contact/goal gates still apply.
"""
    def _refinements(self, value, code, grasp):
        count = getattr(self, '_refinement_count', 0)
        if (value.get('refined') or count >= 96 or grasp is None
                or code not in ('grasp_near_joint_stop', 'next_place_unreachable', 'transport_clearance_unreachable')):
            return []
        band = next(row for row in cylinder_bands(self.env.model, self.env.data, self.env.bottle_id)
                    if row['geom'] == value['geom'])
        neighbors = []
        for shift, yaw, pitch, roll in ((.4 * band['half_height_m'], 0., 0., 0.),
                (-.4 * band['half_height_m'], 0., 0., 0.),
                (0., np.pi / 24, 0., 0.), (0., -np.pi / 24, 0., 0.),
                (0., 0., np.pi / 24, 0.), (0., 0., -np.pi / 24, 0.),
                (0., 0., 0., np.pi / 24), (0., 0., 0., -np.pi / 24),
                (0., 0., 0., np.pi / 12), (0., 0., 0., -np.pi / 12),
                (0., 0., 0., np.pi / 6), (0., 0., 0., -np.pi / 6)):
            rotation = frames(band['axis'], value['yaw_rad'] + yaw, value['pitch_rad'] + pitch)
            angle = value['roll_rad'] + roll; c, s = np.cos(angle), np.sin(angle)
            rotation = rotation @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            point = value['grasp_m'] + band['axis'] * shift
            neighbors.append({**value, 'grasp_m': point, 'pregrasp_m': point - .10 * rotation[:, 2],
                'rotation': rotation, 'yaw_rad': value['yaw_rad'] + yaw, 'pitch_rad': value['pitch_rad'] + pitch,
                'refined': True, 'axial_shift_m': shift, 'roll_rad': angle, '_seed_q': grasp.copy(),
                'proposal_source': 'native-solvable parent; dimension-derived axial and half-angle neighbors'})
        self._refinement_count = count + len(neighbors)
        self.result['private_search_progress']['local_refinement_candidates'] = self._refinement_count
        return neighbors
