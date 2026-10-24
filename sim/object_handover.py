"""Geometry-derived receiver grasps, connected dual-arm stages and real contacts.

Both arm hypotheses use scratch MjData. The selected live object stays free.
Original bottle demo routes and physical/contact/placement parameters are intact.
"""
import copy
import time

import mujoco
import numpy as np

from .grasp import GeometricPlanner, cylinder_bands, body_bands, candidates as cylinder_candidates
from .object_planner import ObjectPlanner, candidates as primitive_candidates
from .tamp import PlanningFailure

CONTRACT = {'schema': 'registered-object-handover/1', 'source': 'simulator_ground_truth',
    'proposal': 'current primitive geometry, donor world yaw and TCP select receiver grasps; proper hand frames, jaw width, both arm paths and optional original placement lookahead',
    'execution': 'donor motion -> fresh receiver planning -> native dual hold -> donor opening -> withdrawal -> receiver-only dwell',
    'verification': 'original >=100ms two-finger contact, actual donor release/no touch, >=1s independent receiver hold',
    'receiver_pose_tolerance_m': .004,
    'unselected_object_path_clearance_m': .012,
    'scope': 'declared upright native bottle/box/cylinder in candidate robot overlap corridor; not arbitrary meshes or physical calibration',
    'hardware_access': False, 'transfer_ready': False}


def receiver_candidates(env, data, donor, receiver, region=None):
    hand = data.xmat[env.ee_ids[donor]].reshape(3, 3)
    donor_tcp = data.xpos[env.ee_ids[donor]] + hand @ np.array([0, .006, .075])
    opening = sum(max(j.range) - min(j.range) for j in env.fingers[receiver])
    proposals = (cylinder_candidates(body_bands(env.model,data,env.bottle_id,region)
        if region else cylinder_bands(env.model, data, env.bottle_id))
        if env.object_id == 'bottle' else primitive_candidates(env, data))
    for value in proposals:
        geom = env.model.geom(value['geom']).id; axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        # Candidate filtering only; actual two-hand meshes still decide collision.
        separation = abs(float(np.dot(value['grasp_m'] - donor_tcp, axis)))
        width = value.get('required_opening_m', 2 * value.get('radius_m', 0) + .004)
        if width > opening: continue
        yield {**value, 'donor_axial_separation_m': separation, 'required_opening_m': width}


class ObjectHandoverPlanner(ObjectPlanner):
    CONTRACT = {**ObjectPlanner.CONTRACT, 'planner': 'registered-object two-arm connected grasp planner',
        'handover': CONTRACT}
    LABELS = {**ObjectPlanner.LABELS, 'transfer': '供物手进入交接区域', 'receive': '接收手接近物体',
        'dual_hold': '确认双手持续夹持', 'release_right': '供物手松开', 'retreat': '供物手撤离', 'hold': '接收手独立持物'}

    def __init__(self, env, task, **kwargs):
        if task not in ('handover_object', 'handover_bottle'): raise ValueError('directed handover task required')
        params = copy.deepcopy(kwargs.get('parameters') or {})
        ident = params.get('object_id', env.object_id)
        if ident not in env.registry: raise PlanningFailure('unknown_object', 'Object is not registered')
        env.select_object(ident)
        # The common base provides actual donor-hold/height/distinct-arm guards;
        # only the older ObjectPlanner's capability gate is replaced here.
        kwargs['parameters'] = params
        GeometricPlanner.__init__(self, env, 'handover_bottle', **kwargs)
        if env.evaluation['held'][self.receiver]:
            raise PlanningFailure('single_donor_required', 'Receiver must initially be empty; complete an existing dual hold separately')
        self.strict_orientation = True; self.deadline = time.monotonic() + 90
        self._clearance_obstacles = [g for object_id, row in env.registry.items() if object_id != ident for g in row['geoms']]
        self._clearance_geoms = {side: [g for g in range(env.model.ngeom) if env.model.geom_contype[g]
            and (f'openarm_{side}_' in env.model.body(int(env.model.geom_bodyid[g])).name
                or int(env.model.geom_bodyid[g]) == env.bottle_id)] for side in ('left', 'right')}
        self.next_place = params.get('next_place')
        if self.next_place is not None:
            goal = self.next_place
            if (set(goal) != {'surface', 'target_xy_m'} or goal['surface'] not in ('table','basket')
                or np.asarray(goal['target_xy_m']).shape != (2,) or not np.isfinite(goal['target_xy_m']).all()):
                raise ValueError('handover lookahead needs an original finite table/basket target')
        self.result.update(object_id=ident, task='handover_object', selected_object=env.selection(), handover_contract=CONTRACT)

    @staticmethod
    def _public(value): return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}

    def _unselected_clearance(self, q, limit=.005):
        self._at(q); minimum = limit; pair = None
        for obstacle in self._clearance_obstacles:
            for geom in self._clearance_geoms[self.side]:
                distance = float(mujoco.mj_geomDistance(self.env.model, self.data, obstacle, geom, limit, None))
                if distance < minimum: minimum, pair = distance, (obstacle, geom)
        self.last_clearance_pair = pair
        return minimum

    def _collision(self, q, allow_fingers=False):
        if super()._collision(q, allow_fingers): return True
        margin = CONTRACT['unselected_object_path_clearance_m']
        separation = self._unselected_clearance(q, margin)
        if separation < margin:
            self.last_collision = {'geoms': [self.env.model.geom(g).name for g in self.last_clearance_pair],
                'separation_m': separation, 'required_clearance_m': margin,
                'source': 'additional scratch planning clearance; native geometry/contact threshold unchanged'}
            return True
        return False

    def _future_place(self, snapshot, grasp_q):
        if self.next_place is None: return None
        saved = (self.planning_qpos, self.rotation.copy(), self.carried, copy.deepcopy(self.ik_seeds))
        predicted = snapshot.copy(); predicted[self.env.qids[self.receiver]] = grasp_q
        for finger in self.env.fingers[self.donor]: predicted[finger.qposadr[0]] = max(finger.range)
        failures = []
        try:
            # Placement follows donor withdrawal, not a donor frozen at the
            # transfer pose. Prove a scratch withdrawal edge first; never
            # remove a robot link/obstacle or substitute live joint state.
            self.planning_qpos = predicted; self._arm(self.donor); self.rotation = None; self._sync()
            donor_point, donor_rotation = self._pose(); donor_q = predicted[self.qids].copy()
            side_sign = -1 if self.donor == 'right' else 1; retreat = None
            for delta in (-.08 * donor_rotation[:, 2] + [0, .03 * side_sign, 0],
                np.array([.04, .08 * side_sign, -.04]), -.12 * donor_rotation[:, 2] + [0, .04 * side_sign, .02]):
                try:
                    q = self._ik(donor_point + delta)
                    if not self._edge(donor_q, q): continue
                    predicted[self.qids] = q; retreat = {'target_m': (donor_point + delta).tolist(), 'q': q.tolist()}; break
                except PlanningFailure as exc:
                    if exc.code in ('cancelled', 'planning_timeout'): raise
            if retreat is None: raise PlanningFailure('next_donor_retreat_unreachable', 'No scratch donor withdrawal before original placement')
            self._arm(self.receiver)
            self.planning_qpos = predicted; self._sync()
            hand = self.data.xmat[self.body].reshape(3, 3)
            object_rotation = self.data.xmat[self.env.bottle_id].reshape(3, 3).copy()
            self.carried = (hand.T @ (self.data.xpos[self.env.bottle_id] - self.data.xpos[self.body]), hand.T @ object_rotation)
            yaw = np.arctan2(object_rotation[1, 0], object_rotation[0, 0])
            upright = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])
            surface = self.next_place['surface']
            base = np.r_[self.next_place['target_xy_m'], self.env.placement_surface(surface)['top_z']]
            for angle in (0., -.25, .25, -.50, .50, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2, np.pi):
                turn = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
                self.rotation = turn @ upright @ self.carried[1].T
                offset = self.rotation @ (self.local_point - self.carried[0])
                try:
                    lower = self._ik(base + [0, 0, .020 if surface=='table' else .019] + offset, True)
                    lower_clearance = self._unselected_clearance(lower)
                    if lower_clearance < .005:
                        raise PlanningFailure('next_place_clearance', 'Predicted lower pose needs5mm clearance from unselected objects')
                    for height in ((.12, .08, .05) if surface=='table' else (.114,)):
                        try:
                            upper = self._ik(base + [0, 0, height] + offset, True)
                            if not self._edge(upper, lower, True): raise PlanningFailure('next_place_edge_collision', 'Predicted lower segment collides')
                            return {'target_xy_m': list(self.next_place['target_xy_m']), 'surface': surface,
                                'donor_retreat': retreat,
                                'unselected_lower_clearance_m': lower_clearance,
                                'yaw_offset_rad': angle, 'clearance_m': height, 'upper_q': upper.tolist(), 'lower_q': lower.tolist(),
                                'source': 'scratch-only upright transit/lower endpoint and connecting-edge constraints',
                                'scope': 'geometric lookahead only; actual contact, donor withdrawal and full live placement remain mandatory'}
                        except PlanningFailure as exc:
                            if exc.code in ('cancelled', 'planning_timeout'): raise
                            failures.append({'yaw_rad': angle, 'height_m': height, 'code': exc.code})
                except PlanningFailure as exc:
                    if exc.code in ('cancelled', 'planning_timeout'): raise
                    failures.append({'yaw_rad': angle, 'code': exc.code})
            self.result['future_place_search'] = {'target': self.next_place, 'rejected': failures}
            raise PlanningFailure('next_place_unreachable', 'Receiver grasp has no declared upright placement endpoint/edge')
        finally:
            self._arm(self.receiver)
            self.planning_qpos, self.rotation, self.carried, self.ik_seeds = saved

    def _receiver(self, snapshot, preferred=None):
        self._arm(self.receiver); self.planning_qpos = snapshot.copy()
        for finger in self.env.fingers[self.receiver]: self.planning_qpos[finger.qposadr[0]] = max(finger.range)
        self._sync(); seed = copy.deepcopy(self.ik_seeds[self.receiver]); rejected = []
        values = list(receiver_candidates(self.env, self.data, self.donor, self.receiver,
            self.parameters.get('receiver_grasp_region')))
        if self.next_place:
            # A low grasp may transfer but place its palm below the table.
            # Prioritize high bands; actual full meshes decide feasibility.
            values.sort(key=lambda v: (-v['grasp_m'][2], abs(v['pitch_rad']), -v['donor_axial_separation_m']))
        if preferred is not None:
            values.sort(key=lambda v: (v['geom'] != preferred['geom'], abs(v.get('axial_fraction', 0) - preferred.get('axial_fraction', 0)),
                abs(v['yaw_rad'] - preferred['yaw_rad']), abs(v['pitch_rad'] - preferred['pitch_rad'])))
        for index, value in enumerate(values):
            self._check(); self.rotation = value['rotation']; self.ik_seeds[self.receiver] = copy.deepcopy(seed)
            try:
                grasp = self._ik(value['grasp_m'], True)
                self.ik_seeds[self.receiver].insert(0, grasp)
                pre = self._ik(value['pregrasp_m']); self._path(pre)
                if not self._edge(pre, grasp, True): raise PlanningFailure('approach_collision', 'Receiver approach collides')
                future = self._future_place(self.planning_qpos, grasp)
                if future: self.result['future_place'] = future
                return value, pre, grasp, rejected
            except PlanningFailure as exc:
                if exc.code in ('cancelled', 'planning_timeout'): raise
                rejected.append({'index': index, 'geom': value['geom'], 'yaw_rad': value['yaw_rad'],
                    'pitch_rad': value['pitch_rad'], 'axial_fraction': value.get('axial_fraction'), 'code': exc.code,
                    'diagnostics': copy.deepcopy(self.result.get('planning_diagnostics'))})
        self.result['receiver_search'] = {'rejected': rejected, 'candidates': len(values)}
        raise PlanningFailure('handover_unreachable', 'No connected receiver pregrasp/grasp hypothesis in current geometry')

    def _pair(self):
        self._arm(self.donor, True); donor_rotation = self.env.data.xmat[self.body].reshape(3, 3).copy()
        live_q = self.env.data.qpos.copy(); live_v = self.env.data.qvel.copy()
        failures = []
        table_z = float(self.env.data.geom_xpos[self.env.table_id, 2] + self.env.model.geom_size[self.env.table_id, 2])
        try:
            # Declared candidate overlap corridor; every point needs both arm
            # IK/paths. This is not a calibrated/guaranteed workspace boundary.
            approach = float(np.arctan2(donor_rotation[1, 2], donor_rotation[0, 2]))
            yaw_choices = ([ -approach, 0., np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]
                if self.donor == 'right' else [0., np.pi/4, -np.pi/4, np.pi/2, -np.pi/2, -approach])
            for yaw_offset in dict.fromkeys(yaw_choices):
                turn = np.array([[np.cos(yaw_offset), -np.sin(yaw_offset), 0], [np.sin(yaw_offset), np.cos(yaw_offset), 0], [0, 0, 1.]])
                proposed_rotation = turn @ donor_rotation
                for height in (.210, .230, .190, .170, .250):
                    for y in (-.02, .02, -.06, .06):
                        for x in (.48, .50, .46, .52, .44):
                            result = self._try_pair(x, y, table_z + height, proposed_rotation, failures)
                            if result is not None: return result
            self.result['pair_search'] = {'rejected_pairs': failures}
            raise PlanningFailure('handover_unreachable', 'No connected donor/receiver pair in the declared corridor')
        finally:
            self.planning_qpos = None
            np.testing.assert_array_equal(live_q, self.env.data.qpos); np.testing.assert_array_equal(live_v, self.env.data.qvel)

    def _try_pair(self, x, y, z, donor_rotation, failures):
        goal = np.array([x, y, z])
        self._check(); self.planning_qpos = None; self._arm(self.donor, True); self.rotation = donor_rotation
        self._sync()
        try:
            target = goal + donor_rotation @ (self.local_point - self.carried[0])
            donor_q = self._ik(target, True); self._path(donor_q, True)
            self._at(donor_q); predicted = self.data.qpos.copy()
            receiver, pre, grasp, rejected = self._receiver(predicted)
            self.result['pair_search'] = {'selected_object_world_m': goal.tolist(),
                'donor_rotation': donor_rotation.tolist(), 'receiver': self._public(receiver), 'donor_q': donor_q.tolist(),
                'receiver_pre_q': pre.tolist(), 'receiver_grasp_q': grasp.tolist(),
                'receiver_rejected': rejected, 'rejected_pairs': failures}
            self.result.pop('planning_diagnostics', None)
            return goal, donor_rotation, receiver
        except PlanningFailure as exc:
            if exc.code in ('cancelled', 'planning_timeout'): raise
            failures.append({'object_world_m': goal.tolist(), 'donor_rotation': donor_rotation.tolist(), 'code': exc.code,
                'diagnostics': copy.deepcopy(self.result.get('planning_diagnostics')),
                'receiver_search': copy.deepcopy(self.result.get('receiver_search')),
                'future_place_search': copy.deepcopy(self.result.get('future_place_search'))})
            self.result['pair_search'] = {'rejected_pairs': failures}
            return None

    def _handover(self):
        self._phase('transfer'); goal, donor_rotation, preferred = self._pair()
        self._arm(self.donor, True); self.rotation = donor_rotation; self._sync()
        yield from self._move(goal + donor_rotation @ (self.local_point - self.carried[0]), True)
        if not self.env.evaluation['held'][self.donor]: raise PlanningFailure('grasp_lost', 'Donor lost actual hold before receiving')
        self._arm(self.receiver); self._phase('receive')
        yield from self._grip(self.receiver, self.env.grippers[self.receiver].open_motor)
        actual_q = self.env.data.qpos.copy(); actual_v = self.env.data.qvel.copy()
        try:
            value, pre_q, grasp_q, rejected = self._receiver(actual_q, preferred)
            self.result['receiver_replan'] = {'actual_object_world_m': self.env.data.xpos[self.env.bottle_id].tolist(),
                'selected': self._public(value), 'rejected': rejected, 'source': 'fresh post-donor-motion native geometry'}
            self.ik_seeds[self.receiver] = [grasp_q, pre_q, *self.ik_seeds[self.receiver]]
        finally:
            self.planning_qpos = None
            np.testing.assert_array_equal(actual_q, self.env.data.qpos); np.testing.assert_array_equal(actual_v, self.env.data.qvel)
        self.rotation = value['rotation']; self.result.pop('planning_diagnostics', None)
        yield from self._move(value['pregrasp_m'])
        yield from self._move(value['grasp_m'], True, position_tolerance=.004)
        self._phase('dual_hold'); yield from self._grip(self.receiver, .05)
        if not all(self.env.evaluation['held'].values()):
            raise PlanningFailure('dual_hold_not_confirmed', 'Native sustained two-finger receiver hold missing; donor remains closed')
        self.result['dual_hold'] = {'held': dict(self.env.evaluation['held']), 'hold_seconds': dict(self.env.contacts.hold_seconds),
            'object_id': self.env.object_id, 'tick': self.env.tick, 'sim_time': float(self.env.data.time)}
        self._phase('release_right'); yield from self._grip(self.donor, self.env.grippers[self.donor].open_motor)
        if not self.env.evaluation['held'][self.receiver]:
            raise PlanningFailure('receiver_grasp_lost', 'Receiver lost contact after donor opened')
        self._phase('retreat'); self._arm(self.donor); self.rotation = None; self._sync(); point, _ = self._pose()
        side_sign = -1 if self.donor == 'right' else 1
        attempts = []; chosen = None
        for delta in (-.08 * donor_rotation[:, 2] + [0, .03 * side_sign, 0],
            np.array([.04, .08 * side_sign, -.04]), -.12 * donor_rotation[:, 2] + [0, .04 * side_sign, .02]):
            try:
                target_q = self._ik(point + delta); self._path(target_q); chosen = point + delta; break
            except PlanningFailure as exc:
                if exc.code in ('cancelled', 'planning_timeout'): raise
                attempts.append({'delta_m': delta.tolist(), 'code': exc.code, 'diagnostics': copy.deepcopy(self.result.get('planning_diagnostics'))})
        self.result['donor_retreat'] = {'target_m': None if chosen is None else chosen.tolist(), 'rejected': attempts}
        if chosen is None: raise PlanningFailure('handover_retreat_unreachable', 'Receiver holds, but donor cannot safely withdraw')
        yield from self._move(chosen)
        self._phase('hold'); independent_start = float(self.env.data.time)
        for _ in range(30):
            yield self.base_action.copy(), {'phase': 'hold', 'object_id': self.env.object_id, 'receiver': self.receiver}
            if (not self.env.evaluation['held'][self.receiver] or self.env.evaluation['held'][self.donor]
                or self.env.evaluation['touching'][self.donor]):
                raise PlanningFailure('handover_not_confirmed', 'Actual independent receiver hold did not persist after donor withdrawal')
        self.result.update(from_arm=self.donor, to_arm=self.receiver, held_by=self.receiver,
            verification_source='native_receiver_contact_then_donor_release',
            independent_hold_seconds=float(self.env.data.time) - independent_start,
            final_held=dict(self.env.evaluation['held']), final_touching=dict(self.env.evaluation['touching']))
        if self.next_place:
            hand = self.env.data.xmat[self.env.ee_ids[self.receiver]].reshape(3, 3)
            obj = self.env.data.xmat[self.env.bottle_id].reshape(3, 3)
            yaw = np.arctan2(obj[1, 0], obj[0, 0]) + self.result['future_place']['yaw_offset_rad']
            upright = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])
            rotation = upright @ (hand.T @ obj).T
            offset = hand.T @ (self.env.data.xpos[self.env.bottle_id] - self.env.data.xpos[self.env.ee_ids[self.receiver]])
            base = np.r_[self.next_place['target_xy_m'], self.env.placement_surface('table')['top_z'] + .20]
            quat = np.empty(4); mujoco.mju_mat2Quat(quat, rotation.ravel())
            self.result['next_placement_transit'] = {'arm': self.receiver, 'object_id': self.env.object_id, 'frame': 'world',
                'position_m': (base + rotation @ (self.local_point - offset)).tolist(),
                'quaternion_xyzw': quat[[1, 2, 3, 0]].tolist(), 'planner': 'curobo',
                'source': 'proposal from declared original goal and actual post-relay grasp; observe again and replan before execution',
                'source_tick': self.env.tick, 'scope': 'unexecuted transport hypothesis; not a placement success receipt'}


class ObjectRelayPickPlanner(ObjectPlanner):
    """Task-conditioned proposal ranking; all original pick gates still run.

    A declared next receiver needs another axial band. Lower donor candidates
    leave higher bands available for a receiver that must subsequently place.
    This is a search heuristic, not a guarantee of relay feasibility.
    """
    def __init__(self, env, task, **kwargs):
        params = kwargs.get('parameters') or {}
        future = params.get('next_handover')
        if (task not in ('pick_object', 'pick_bottle') or not isinstance(future, dict)
            or set(future) != {'to_arm'} or future['to_arm'] not in ('left', 'right')
            or future['to_arm'] == params.get('arm')):
            raise ValueError('relay-conditioned pick requires a distinct declared receiver')
        super().__init__(env, task, **kwargs)
        self.result['grasp_intent'] = {'next_handover': copy.deepcopy(future),
            'source': 'proposal ranking only; every actual grasp, relay and placement gate remains mandatory'}

    def _choose_pick_pose(self, ignored):
        rejected = []; seeds = copy.deepcopy(self.ik_seeds[self.side])
        live_q = self.env.data.qpos.copy(); live_v = self.env.data.qvel.copy()
        try:
            self.planning_qpos = live_q.copy()
            for finger in self.env.fingers[self.side]: self.planning_qpos[finger.qposadr[0]] = max(finger.range)
            self._sync()
            values = list(cylinder_candidates(cylinder_bands(self.env.model, self.data, self.env.bottle_id))
                if self.env.object_id == 'bottle' else primitive_candidates(self.env, self.data))
            values.sort(key=lambda v: (v['grasp_m'][2], abs(v['pitch_rad'])))
            for index, value in enumerate(values):
                self._check(); self.rotation = value['rotation']; self.ik_seeds[self.side] = copy.deepcopy(seeds)
                try:
                    grasp = self._ik(value['grasp_m'], True); self.ik_seeds[self.side].insert(0, grasp)
                    pre = self._ik(value['pregrasp_m']); self._path(pre)
                    if not self._edge(pre, grasp, True): raise PlanningFailure('approach_collision', 'Relay donor approach collides')
                    self.result['grasp_search'] = {'object_id': self.env.object_id, 'arm': self.side,
                        'candidate_index': index, 'selected': ObjectHandoverPlanner._public(value), 'rejected': rejected,
                        'pregrasp_q': pre.tolist(), 'grasp_q': grasp.tolist(), 'source': 'task-conditioned primitive proposal ranking'}
                    self.result.pop('planning_diagnostics', None); self.ik_seeds[self.side] = [grasp, pre, *seeds]
                    return value['grasp_m'].copy(), value['pregrasp_m'].copy()
                except PlanningFailure as exc:
                    if exc.code in ('cancelled', 'planning_timeout'): raise
                    rejected.append({'index': index, 'candidate': ObjectHandoverPlanner._public(value), 'code': exc.code,
                        'diagnostics': copy.deepcopy(self.result.get('planning_diagnostics'))})
            self.result['grasp_search'] = {'rejected': rejected}
            raise PlanningFailure('grasp_search_exhausted', 'Relay donor proposal pool has no connected grasp')
        finally:
            self.planning_qpos = None
            np.testing.assert_array_equal(live_q, self.env.data.qpos); np.testing.assert_array_equal(live_v, self.env.data.qvel)
