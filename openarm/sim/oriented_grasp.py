"""Isolated 6D primitive grasp proposals using the existing native executor.

The legacy demo and upright planner are unchanged. This optional planner only
replaces the unsupported-start assumption for explicitly selected rigid
box/cylinder picks. It does not change contact, lift, placement or physics.
"""
import copy

import mujoco
import numpy as np

from .object_planner import ObjectPlanner
from .tamp import PlanningFailure

CONTRACT = {'schema': 'oriented-primitive-grasp/1', 'source': 'simulator_ground_truth',
    'proposal': 'object-axis radial frames, actual primitive widths, top approach ranked first',
    'execution': 'existing connected native IK/path, actuator, two-finger contact and60mm lift',
    'scope': 'known registered box/cylinder geometry at non-upright orientations; not mesh/visual grasp prediction',
    'graspgenx_model_used': False, 'hardware_access': False, 'transfer_ready': False}


def initial_conflicts(env):
    """Native selected-object overlaps at the original .5mm planning margin."""
    pairs = {}
    for contact in env.data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        if a < 0 or b < 0 or contact.dist >= -.0005: continue
        if a not in env.bottle_geoms and b not in env.bottle_geoms: continue
        names = tuple(sorted((env.model.geom(a).name, env.model.geom(b).name)))
        pairs[names] = max(pairs.get(names, 0.), -float(contact.dist))
    return [{'geoms':list(names), 'penetration_m':depth} for names,depth in pairs.items()]


class HoldAudit:
    """Requested native hold validation after pick; no changed motor targets."""
    def __init__(self, planner, seconds):
        if type(seconds) not in (int, float) or not np.isfinite(seconds) or not 2 <= seconds <= 10:
            raise ValueError('pick hold_seconds must cover the original2s audit window and be at most10s')
        self.planner, self.seconds = planner, seconds

    def __getattr__(self, name): return getattr(self.planner, name)

    def actions(self):
        from .decision_timing import GraspStability
        yield from self.planner.actions()
        if not self.result.get('primitive_success'): raise PlanningFailure('grasp_not_confirmed', '先完成原生抓取，再检查持稳')
        side = self.result.get('held_by'); env = self.env
        if side not in ('left', 'right'): raise PlanningFailure('single_hold_required', '持稳核验需要本次独立持物回执')
        self.result.update(status='holding_audit', primitive_success=False)
        audit = GraspStability(env); command = env.last_action.copy(); start = float(env.data.time)
        count = int(np.ceil(self.seconds * env.profile['control_hz']))
        for _ in range(count):
            self.planner._check()
            yield command.copy(), {'phase': 'hold', 'hold_validation': 'native_contact_and_relative_pose',
                'scene_source': 'simulator_ground_truth'}
            audit.read()
            self.result['hold_audit'] = {**audit.summary(), 'requested_seconds': self.seconds,
                'actual_seconds': float(env.data.time) - start}
            if not env.evaluation['held'][side]: raise PlanningFailure('grasp_lost', '本次持稳核验中失去持续双指接触')
        value = self.result['hold_audit']
        if (value['actual_seconds'] + .002 < self.seconds
            or value['arms'][side]['status'] != 'stable_within_audit_window'):
            raise PlanningFailure('grasp_unstable', '已请求的持稳时长或原相对位姿稳定判据未通过')
        self.result.update(status='completed', primitive_success=True)


def radial_frame(axis, angle, pitch=0.):
    axis = np.asarray(axis, dtype=float).copy(); axis /= np.linalg.norm(axis)
    reference = np.array([0., 0., -1.])
    radial = reference - axis * np.dot(reference, axis)
    if np.linalg.norm(radial) < 1e-8:
        reference = np.array([1., 0., 0.]); radial = reference - axis * np.dot(reference, axis)
    radial /= np.linalg.norm(radial); lateral = np.cross(axis, radial)
    approach = np.cos(angle) * radial + np.sin(angle) * lateral
    frame = np.column_stack((axis, np.cross(approach, axis), approach))
    c, s = np.cos(pitch), np.sin(pitch)
    return frame @ np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def candidates(env, data, side):
    opening = sum(max(finger.range) - min(finger.range) for finger in env.fingers[side])
    rows = []
    for geom in env.bottle_geoms:
        kind = int(env.model.geom_type[geom]); size = env.model.geom_size[geom]
        if kind not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
            raise PlanningFailure('unsupported_grasp_geometry', '姿态候选尚不支持该碰撞形状；不按盒子或圆柱假装处理')
        rotation = data.geom_xmat[geom].reshape(3, 3); axis = rotation[:, 2]
        half = size[2] if kind == mujoco.mjtGeom.mjGEOM_BOX else size[1]
        for fraction in (0., .35, -.35, .65, -.65):
            point = data.geom_xpos[geom] + fraction * half * axis
            for angle in np.deg2rad([0, -15, 15, -30, 30, -45, 45, -60, 60, -90, 90, 180]):
                for pitch in np.deg2rad([0., -15., 15.]):
                    hand = radial_frame(axis, angle, pitch)
                    width = (2 * size[0] if kind == mujoco.mjtGeom.mjGEOM_CYLINDER else
                        2 * float(np.abs(rotation.T @ hand[:, 1]) @ size))
                    if width + .004 > opening: continue
                    rows.append({'geom': env.model.geom(geom).name, 'shape': mujoco.mjtGeom(kind).name,
                        'grasp_m': point.copy(), 'pregrasp_m': point - .10 * hand[:, 2],
                        'rotation': hand, 'required_opening_m': width + .004,
                        'radial_angle_rad': float(angle), 'pitch_rad': float(pitch), 'axial_fraction': fraction})
    # This is a proposal ranking, not an acceptance shortcut. Native geometry
    # still checks the full open gripper, arm, support and connecting path.
    rows.sort(key=lambda row: (row['rotation'][2, 2], abs(row['axial_fraction']), abs(row['pitch_rad'])))
    yield from rows


class OrientedPlanner(ObjectPlanner):
    def _at(self,q):
        super()._at(q)
        probe=getattr(self,'_pose_ik_probe',None)
        if probe:
            point,rotation=self._pose();position=float(np.linalg.norm(probe['target']-point));angular=0.
            if probe['rotation'] is not None:
                angular=float(np.arccos(np.clip((np.trace(probe['rotation']@rotation.T)-1)/2,-1,1)))
            cost=position+.15*angular
            if probe.get('best') is None or cost<probe['best']['cost']:
                probe['best']={'q':np.asarray(q).copy(),'cost':cost,'position_m':position,'rotation_rad':angular}

    def _ik(self,target,allow_fingers=False,*,position_tolerance=.0015):
        probe={'target':np.asarray(target).copy(),'rotation':None if self.rotation is None else self.rotation.copy()}
        self._pose_ik_probe=probe;seeds=copy.deepcopy(self.ik_seeds[self.side])
        try:
            try:return super()._ik(target,allow_fingers,position_tolerance=position_tolerance)
            except PlanningFailure as exc:
                best=probe.get('best')
                if (exc.code!='ik_unreachable' or not best or best['position_m']>.006 or best['rotation_rad']>.06):raise
                # Continue the closest numerical candidate through the SAME
                # original solver, collision and tolerance gates. Never turn
                # near-convergence into success or apply its joints directly.
                self.ik_seeds[self.side]=[best['q'],*seeds]
                self.result.setdefault('ik_refinements',[]).append({'position_error_m':best['position_m'],
                    'rotation_error_rad':best['rotation_rad'],
                    'source':'scratch near-converged seed; original tolerances unchanged'})
                return super()._ik(target,allow_fingers,position_tolerance=position_tolerance)
        finally:self.ik_seeds[self.side]=seeds;self._pose_ik_probe=None

    def _future_place(self,grasp_q):
        goal=self.parameters.get('next_placement')
        if goal is None:return None
        if (not {'surface','target_xy_m'} <= set(goal) or set(goal)-{'surface','target_xy_m','position_tolerance_m'} or goal['surface']!='table'
            or len(goal['target_xy_m'])!=2 or not np.isfinite(goal['target_xy_m']).all()):
            raise ValueError('next_placement must be the declared original table XY')
        tolerance = goal.get('position_tolerance_m', .01)
        if not np.isfinite(tolerance) or not .001 <= tolerance <= .05:
            raise ValueError('invalid declared placement tolerance')
        precision = max(.0015, min(.003, tolerance * .6))
        low_ik_tolerance = min(.0015, precision * .35)
        saved=(self.carried,self.rotation.copy(),self.phase,copy.deepcopy(self.ik_seeds))
        rejected=[]
        try:
            self._at(grasp_q);hand=self.data.xmat[self.body].reshape(3,3).copy()
            origin=self.data.xpos[self.env.bottle_id].copy();obj=self.data.xmat[self.env.bottle_id].reshape(3,3).copy()
            offset=hand.T@(origin-self.data.xpos[self.body]);relative=hand.T@obj
            self.carried=(offset,relative);self.phase='place'
            base=np.r_[goal['target_xy_m'],self.env.placement_surface('table')['top_z']]
            for height in (.12,.08,.05):
                for yaw in (0.,-np.pi/4,np.pi/4,-np.pi/2,np.pi/2,np.pi):
                    c,s=np.cos(yaw),np.sin(yaw);upright=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                    self.rotation=upright@relative.T;tcp_offset=self.rotation@(self.local_point-offset)
                    try:
                        high=self._ik(base+[0,0,height]+tcp_offset,True)
                        self.ik_seeds[self.side].insert(0,high)
                        low=self._ik(base+[0,0,.020]+tcp_offset,True,position_tolerance=low_ik_tolerance)
                        if not self._edge(high,low,True):raise PlanningFailure('next_place_edge_collision','原目标上方到释放位姿的连接段碰撞')
                        return {'target_xy_m':list(goal['target_xy_m']),'yaw_rad':float(yaw),'transit_height_m':height,
                            'lower_ik_tolerance_m':low_ik_tolerance,
                            'source':'scratch-only original-goal endpoint and lowering-edge lookahead at original precision; actual placement still mandatory','rejected':rejected}
                    except PlanningFailure as exc:
                        if exc.code in ('cancelled','planning_timeout'):raise
                        rejected.append({'height_m':height,'yaw_rad':float(yaw),'code':exc.code})
            raise PlanningFailure('next_place_unreachable','该抓取关系未找到原直立放置目标的端点与连接段')
        finally:self.carried,self.rotation,self.phase,self.ik_seeds=saved

    def _collision(self, q, allow_fingers=False):
        if super()._collision(q,allow_fingers):return True
        if 'oriented_grasp_contract' not in self.result or self.phase!='approach' or allow_fingers:return False
        # The target must remain contact-free during opening and pregrasp.
        # Account for native servo tracking rather than checking only negative
        # penetration of commanded joint configurations. Grasp/contact/lift
        # gates are unchanged; this is extra proposal clearance before contact.
        model=self.env.model
        robot=[g for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g])
            and (model.body(int(model.geom_bodyid[g])).name or '').startswith('openarm_')]
        for geom in self.env.bottle_geoms:
            for arm_geom in robot:
                distance=mujoco.mj_geomDistance(model,self.data,int(geom),arm_geom,.012,None)
                if distance<.012:
                    self.last_collision={'geoms':[model.geom(int(geom)).name,model.geom(arm_geom).name],
                        'distance_m':float(distance),'required_pregrasp_clearance_m':.012,'scope':'non-contact pregrasp only'}
                    return True
        return False

    def __init__(self, env, task, **kwargs):
        parameters = copy.deepcopy(kwargs.get('parameters') or {})
        ident = parameters.get('object_id', env.object_id)
        if ident not in env.registry: raise PlanningFailure('unknown_object', '对象未注册为可操作自由刚体')
        env.select_object(ident)
        tilt = float(env.data.xmat[env.bottle_id].reshape(3, 3)[2, 2])
        mapped = {'pick_object': 'pick_bottle', 'reach_object': 'reach_bottle'}.get(task, task)
        if mapped not in ('pick_bottle', 'reach_bottle') or tilt >= .98 or not parameters:
            super().__init__(env, task, **kwargs); return
        if env.robot_version != 'v1': raise ValueError('oriented pick requires the official v1 model')
        if env._done or env._clock_mode not in (None, 'periodic'):
            raise PlanningFailure('episode_finished', '原任务时钟已终止；不会重置或重新开放已失败的任务')
        side = parameters.get('arm', 'right')
        if side not in ('left', 'right'): raise ValueError('unknown grasp arm')
        if any(env.evaluation['held'].values()) or any(env.evaluation['touching'].values()):
            raise PlanningFailure('already_holding', '接触或持物中的夹爪不允许倒卧重抓')
        if env.evaluation.get('invalid') or env._bottle_bounds()[0][2] < .99:
            raise PlanningFailure('unsupported_start', '物体不在候选工作面上或原生场景已无效')
        if initial_conflicts(env):
            raise PlanningFailure('initial_object_overlap', '目标物体起点与其它碰撞体相交；保留原场景证据，不自动移动或重置来通过')
        supported = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER)
        if any(int(env.model.geom_type[g]) not in supported for g in env.bottle_geoms):
            raise PlanningFailure('unsupported_grasp_geometry', '未声明该对象外形的姿态抓取候选')
        before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), float(env.data.time))
        # Reuse the existing Cartesian planner's checked kinematics/context
        # construction at the CURRENT measured TCP. Its generator is never
        # started. The oriented-pick guards above replace only the legacy
        # upright start requirement; all motion/contact code remains inherited.
        body = env.ee_ids[side]; rotation = env.data.xmat[body].reshape(3, 3)
        point = env.data.xpos[body] + rotation @ np.array([0., .006, .075])
        quat = np.empty(4); mujoco.mju_mat2Quat(quat, rotation.ravel())
        context = ObjectPlanner(env, 'move_to_pose', parameters={'object_id': ident, 'arm': side,
            'frame': 'world', 'position_m': point.tolist(), 'quaternion_xyzw': np.r_[quat[1:], quat[0]].tolist(),
            'speed_scale': parameters.get('speed_scale', 1.)},
            cancel=kwargs.get('cancel', lambda: False), feedback=kwargs.get('feedback', lambda value: None))
        self.__dict__.update(context.__dict__)
        self.task = mapped; self.parameters = parameters; self.pick_arm = side
        self.parameterized_pick = True; self.strict_orientation = True
        self.result = {'task': mapped, 'status': 'planning', **self.CONTRACT, 'object_id': ident,
            'selected_object': env.selection(), 'oriented_grasp_contract': CONTRACT,
            'grasp_start': {'orientation_class': 'non_upright', 'body_axis_z_component': tilt,
                'source': 'current native body/geometry pose; not memory or image inference'}, 'task_termination': 'skill'}
        env.task_termination = 'skill'
        for old, current in zip(before[:3], (env.data.qpos, env.data.qvel, env.data.ctrl)):
            np.testing.assert_array_equal(old, current)
        assert before[3] == float(env.data.time)

    def _choose_pick_pose(self, ignored):
        if 'oriented_grasp_contract' not in self.result: return super()._choose_pick_pose(ignored)
        rejected = []; seeds = copy.deepcopy(self.ik_seeds[self.side])
        q, v = self.env.data.qpos.copy(), self.env.data.qvel.copy()
        try:
            self.planning_qpos = q.copy()
            for finger in self.env.fingers[self.side]: self.planning_qpos[finger.qposadr[0]] = max(finger.range)
            self._sync()
            for index, candidate in enumerate(candidates(self.env, self.data, self.side)):
                self._check(); self.rotation = candidate['rotation']; self.ik_seeds[self.side] = copy.deepcopy(seeds)
                try:
                    grasp = self._ik(candidate['grasp_m'], True); self.ik_seeds[self.side].insert(0, grasp)
                    pre = self._ik(candidate['pregrasp_m']); self._path(pre)
                    if not self._edge(pre, grasp, True): raise PlanningFailure('approach_collision', '完整预抓取连接段碰撞')
                    future=self._future_place(grasp)
                    self.result['grasp_search'] = {'contract': CONTRACT, 'object_id': self.env.object_id,
                        'arm': self.side, 'candidate_index': index,
                        'selected': {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in candidate.items()},
                        'rejected': rejected, 'pregrasp_q': pre.tolist(), 'grasp_q': grasp.tolist()}
                    if future:self.result['future_placement']=future
                    self.result.pop('planning_diagnostics', None); self.ik_seeds[self.side] = [grasp, pre, *seeds]
                    return candidate['grasp_m'].copy(), candidate['pregrasp_m'].copy()
                except PlanningFailure as exc:
                    if exc.code in ('cancelled', 'planning_timeout'): raise
                    rejected.append({'index': index, 'code': exc.code,
                        'candidate': {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in candidate.items()},
                        'diagnostics': copy.deepcopy(self.result.get('planning_diagnostics'))})
            self.result['grasp_search'] = {'contract': CONTRACT, 'object_id': self.env.object_id, 'rejected': rejected}
            raise PlanningFailure('grasp_search_exhausted', '姿态候选没有找到完整可执行路径；不能当作物理不可达证明')
        finally:
            self.planning_qpos = None
            np.testing.assert_array_equal(q, self.env.data.qpos); np.testing.assert_array_equal(v, self.env.data.qvel)

    def _pick(self):
        if 'oriented_grasp_contract' not in self.result:
            yield from super()._pick(); return
        side = self.pick_arm; self._arm(side); initial_bottom = self.env._bottle_bounds()[0][2]
        grasp, pre = self._choose_pick_pose(self.env.data.xpos[self.env.bottle_id].copy())
        self._phase('approach'); yield from self._grip(side,self.env.grippers[side].open_motor)
        yield from self._move(pre)
        if self.task == 'reach_bottle': return
        # Rolling or contact-displaced objects cannot reuse the world point
        # selected before opening/pregrasp. Re-ground the SAME local primitive
        # band from the current native observation before its approach.
        selected = self.result['grasp_search']['selected']; geom = self.env.model.geom(selected['geom']).id
        rotation = self.env.data.geom_xmat[geom].reshape(3,3); size = self.env.model.geom_size[geom]
        half = size[2] if int(self.env.model.geom_type[geom]) == mujoco.mjtGeom.mjGEOM_BOX else size[1]
        current = self.env.data.geom_xpos[geom] + selected['axial_fraction'] * half * rotation[:,2]
        self.rotation = radial_frame(rotation[:,2], selected['radial_angle_rad'], selected['pitch_rad'])
        self.result['approach_regrounding'] = {'original_point_m':grasp.tolist(),'current_point_m':current.tolist(),
            'displacement_m':float(np.linalg.norm(current-grasp)), 'source':'fresh native geometry after pregrasp; original task target unchanged'}
        self._phase('grasp'); yield from self._move(current)
        self._phase('close'); yield from self._grip(side,.05)
        if not self.env.evaluation['held'][side]: raise PlanningFailure('grasp_not_confirmed','闭合后未形成原100ms持续双指接触；不抬升')
        hand = self.env.data.xmat[self.body].reshape(3,3)
        self.carried = (hand.T@(self.env.data.xpos[self.env.bottle_id]-self.env.data.xpos[self.body]),
            hand.T@self.env.data.xmat[self.env.bottle_id].reshape(3,3))
        self._phase('lift'); self._sync();point,_=self._pose()
        yield from self._move(point+np.array([0.,0.,.12]),True)
        self._phase('hold')
        for _ in range(30):
            yield self.base_action.copy(),{'phase':'hold','scene_source':'simulator_ground_truth'}
            if not self.env.evaluation['held'][side]: raise PlanningFailure('grasp_lost','抬升后未保持原持续双指夹持')
        bottom=self.env._bottle_bounds()[0][2]
        if bottom < initial_bottom+.06: raise PlanningFailure('lift_not_confirmed','实际物体没有抬离初始支撑面达到60mm')
        self.result.update(arm=side,verification_source='native_two_finger_contact_and_lift',held_by=side,
            lift_height_m=float(bottom-initial_bottom))
