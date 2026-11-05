"""Bounded task/motion primitives for the candidate OpenArm v1 simulation.

Scene poses come from simulator truth, explicitly recorded as an oracle
baseline. IK and collision search use a separate MjData. Only actuator targets
reach the live environment; objects are never welded or teleported in physics.
"""
from __future__ import annotations

import time
import mujoco
import numpy as np


class PlanningFailure(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class TampPlanner:
    TASKS = ('reach_bottle', 'pick_bottle', 'handover_bottle', 'place_bottle', 'handover_to_basket', 'move_to_pose', 'move_relative','set_gripper')
    LABELS = {'approach':'规划接近瓶子', 'grasp':'对准夹持位置', 'close':'闭合并验证接触',
              'lift':'抬起瓶子', 'hold':'验证稳定夹持', 'transfer':'移动到交接区域',
              'receive':'接收手接近瓶身', 'dual_hold':'验证双手共同夹持', 'release_right':'供物手松开',
              'retreat':'夹爪撤离', 'place':'移动到放置区域', 'release_left':'释放物体', 'settle':'等待支撑与稳定', 'move':'规划末端到位','gripper':'调整夹爪开合'}
    CONTRACT = {'planner':'MuJoCo Jacobian IK + joint-space collision search',
        'scene_source':'simulator_ground_truth', 'scope':'oracle manipulation baseline, not visual perception or calibrated transfer',
        'physics':'native actuator targets and contact; no weld, teleport or qpos overwrite',
        'native_finger_range':True,
        'action_bounds':'official simulated finger slide range through candidate motor map; broader than recorded/VLA command range',
        'hardware_access':False, 'transfer_ready':False}

    def __init__(self, env, task, *, parameters=None, cancel=lambda:False, feedback=lambda value:None):
        if task not in self.TASKS:
            raise ValueError('unsupported TAMP task')
        if env.robot_version != 'v1':
            raise ValueError('TAMP requires the current official v1 model')
        if env._done or env._clock_mode not in (None, 'periodic'):
            raise PlanningFailure('episode_finished','仿真回合已结束或使用记录时钟；调用 prepare_simulation 开启新任务，再重新观察和规划')
        parameters = parameters or {}
        self.pick_arm = parameters.get('arm','right')
        self.donor = parameters.get('from_arm','right')
        self.receiver = parameters.get('to_arm','left')
        if self.pick_arm not in ('left','right') or self.donor not in ('left','right') or self.receiver not in ('left','right'):
            raise ValueError('unknown arm')
        if task=='handover_bottle' and self.donor==self.receiver:
            raise PlanningFailure('distinct_arms_required','交接必须选择两只不同机械臂')
        if task in ('reach_bottle','pick_bottle','handover_to_basket') and any(env.evaluation['held'].values()):
            raise PlanningFailure('already_holding','已经有机械臂夹持瓶子；当前抓取原语不会松手重抓，请先重置或使用后续放置能力')
        if task=='handover_bottle' and not (env.evaluation['held'][self.donor] and env._bottle_bounds()[0][2]>1.035):
            raise PlanningFailure('donor_hold_required','交接要求供物手已稳定夹持，并将瓶子抬离工作台')
        self.place_arm=parameters.get('arm','left')
        if task=='place_bottle' and not (env.evaluation['held'][self.place_arm] and not env.evaluation['held']['right' if self.place_arm=='left' else 'left']):
            raise PlanningFailure('single_hold_required','放置要求所选手独立稳定夹持瓶子；未持瓶时先调用 pick，双手夹持时先完成交接')
        upright=env.data.xmat[env.bottle_id].reshape(3,3)[2,2]
        if task in ('reach_bottle','pick_bottle','handover_to_basket') and (upright < .98 or env._bottle_bounds()[0][2]<.99):
            raise PlanningFailure('unsupported_start','抓取要求瓶子直立并在工作台或篮底上；倒卧抓取尚未支持')
        self.env, self.task, self.cancel, self.feedback = env, task, cancel, feedback
        self.parameters = parameters
        self.speed_scale = float(self.parameters.get('speed_scale', 1.0))
        if not np.isfinite(self.speed_scale) or not .05 <= self.speed_scale <= 1:
            raise ValueError('speed_scale must be 0.05..1')
        self.data = mujoco.MjData(env.model)
        self.side = 'right'
        self.start = 8
        self.qids, self.dids = env.qids[self.side], env.dids[self.side]
        self.body = env.ee_ids[self.side]
        self.local_point = np.array([0., .006, .075])
        self.rotation = np.array([[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]])
        self.rng = np.random.default_rng(42)
        self.result = {'task':task, 'status':'planning', **self.CONTRACT}
        self.stats = {'ik_attempts':0,'collision_checks':0,'rrt_nodes':0}
        self.deadline = time.monotonic()+90
        self.phase = 'approach'
        self.carried = None
        self.strict_orientation = task in ('move_to_pose','move_relative')
        self.base_action = env.last_action.copy()
        self.last_collision=None
        self.planning_qpos=None
        from .reference_phases import landmarks
        self.ik_seeds={side:[] for side in ('left','right')}
        for episode in env.reference.values():
            frame=landmarks(episode)['events']['left_close']
            for side in self.ik_seeds:
                self.ik_seeds[side].append(np.array(episode[f'{side}_joints'][frame]))
        proposal=mujoco.MjData(env.model);proposal.qpos[:]=env.data.qpos
        proposal.qpos[env.qids['left']]=self.ik_seeds['left'][0]
        mujoco.mj_kinematics(env.model,proposal)
        self.receiver_rotation=proposal.xmat[env.ee_ids['left']].reshape(3,3).copy()
        self.result['ik_proposals']='audited real left-close poses as IK seeds; receiver orientation from ep0 FK'
        if task in ('move_to_pose', 'move_relative'):
            self._prepare_motion()
        if task=='place_bottle':
            self._prepare_placement()
        if task=='set_gripper':
            if self.parameters.get('arm') not in ('left','right'):raise ValueError('gripper requires selected arm')
            fraction=self.parameters.get('opening_fraction')
            if type(fraction) not in (int,float) or not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('opening_fraction must be 0..1')
        env.task_termination='legacy_benchmark' if task=='handover_to_basket' or not parameters else 'skill'
        self.result['task_termination']=env.task_termination

    def _prepare_placement(self):
        surface=self.parameters.get('receptacle_id','basket')
        absolute=self.parameters.get('target_xy_m');relative=self.parameters.get('offset_xy_m')
        if absolute is not None and relative is not None:raise ValueError('choose target_xy_m OR offset_xy_m')
        if surface=='table' and absolute is None and relative is None:raise ValueError('table placement needs target_xy_m or offset_xy_m')
        if surface=='basket' and relative is not None:raise ValueError('basket placement requires its cavity center or an absolute target')
        xy=np.asarray(absolute if absolute is not None else self.env.data.xpos[self.env.bottle_id,:2]+relative
                      if relative is not None else self.env.data.xpos[self.env.box_id,:2],dtype=float)
        self.env.set_placement_target(surface,xy,self.parameters.get('position_tolerance_m'))
        self.placement_surface=surface;self.placement_xy=xy
        self.result.update(arm=self.place_arm,surface=surface,target_xy_m=xy.tolist(),parameters=self.parameters)

    def _prepare_motion(self):
        side = self.parameters.get('arm')
        if side not in ('left', 'right'):
            raise ValueError('motion requires left/right arm')
        if all(self.env.evaluation['held'].values()):
            raise PlanningFailure('dual_hold_requires_handover', '双手夹持时请使用交接技能，不允许独立移动一只手')
        self._arm(side, bool(self.env.evaluation['held'][side]))
        self._sync()
        current, rotation = self._pose()
        self.rotation = rotation.copy()
        if self.task == 'move_relative':
            delta = np.asarray(self.parameters.get('delta_m'), dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > .15:
                raise ValueError('relative delta must be a finite 3-vector of at most 0.15 m')
            if self.parameters.get('frame') not in ('world', 'tool'):
                raise ValueError('relative frame must be world or tool')
            self.motion_target = current + (rotation @ delta if self.parameters['frame'] == 'tool' else delta)
        else:
            if self.parameters.get('frame') != 'world':
                raise ValueError('pose frame must be world')
            self.motion_target = np.asarray(self.parameters.get('position_m'), dtype=float)
            quat = np.asarray(self.parameters.get('quaternion_xyzw'), dtype=float)
            if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1) > .02:
                raise ValueError('orientation must be a unit quaternion in xyzw order')
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, np.r_[quat[3], quat[:3]] / np.linalg.norm(quat))
            self.rotation = matrix.reshape(3, 3)
        if self.motion_target.shape != (3,) or not np.isfinite(self.motion_target).all():
            raise ValueError('target must be a finite world-frame 3-vector')
        if np.any(abs(self.motion_target[:2]) > 1.5) or not .2 <= self.motion_target[2] <= 1.8:
            raise PlanningFailure('workspace_bounds', '目标超出候选仿真工作区：x/y ±1.5 m、z 0.2–1.8 m')
        self.result.update(arm=side, target_position_m=self.motion_target.tolist(), frame='world',
                           parameters=self.parameters, tcp_local_offset_m=self.local_point.tolist())

    def _arm(self, side, carrying=False):
        self.side=side
        self.start=0 if side=='left' else 8
        self.qids,self.dids=self.env.qids[side],self.env.dids[side]
        self.body=self.env.ee_ids[side]
        # Both official hand frames have local x along the vertical pad span.
        # The receiver approaches from the robot's left (+y), as supported by
        # the recorded hand orientation; the donor approaches along +x.
        self.rotation=self.receiver_rotation.copy() if side=='left' else np.array([[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]])
        self.carried=None
        if carrying:
            hand=self.env.data.xmat[self.body].reshape(3,3)
            self.carried=(hand.T@(self.env.data.xpos[self.env.bottle_id]-self.env.data.xpos[self.body]),
                          hand.T@self.env.data.xmat[self.env.bottle_id].reshape(3,3))

    def _check(self):
        if self.cancel():
            raise PlanningFailure('cancelled', '规划或执行已停止')
        if time.monotonic() > self.deadline:
            raise PlanningFailure('planning_timeout', 'TAMP 达到时间预算')

    def _phase(self, name):
        self._check(); self.phase = name
        self.feedback({'phase':self.LABELS[name], 'task':self.task, 'planner':dict(self.stats)})

    def _sync(self):
        self.data.qpos[:] = self.env.data.qpos if self.planning_qpos is None else self.planning_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.env.model, self.data)

    def _pose(self):
        rot = self.data.xmat[self.body].reshape(3,3)
        return self.data.xpos[self.body] + rot@self.local_point, rot

    def _at(self, q):
        self.data.qpos[self.qids] = q
        mujoco.mj_forward(self.env.model, self.data)
        if self.carried is not None:
            # Rigid attachment is for collision prediction in this scratch
            # MjData only. The live bottle remains a free body under contact.
            hand = self.data.xmat[self.body].reshape(3,3)
            offset, relative = self.carried
            addr = self.env.model.joint('bottle_free').qposadr[0]
            self.data.qpos[addr:addr+3] = self.data.xpos[self.body]+hand@offset
            mujoco.mju_mat2Quat(self.data.qpos[addr+3:addr+7], (hand@relative).ravel())
            mujoco.mj_forward(self.env.model, self.data)

    def _collision(self, q, allow_fingers=False):
        self._check(); self._at(q); self.stats['collision_checks'] += 1
        self.last_collision=None
        model = self.env.model
        for contact in self.data.contact:
            if contact.dist >= -.0005:
                continue
            a,b = int(contact.geom1),int(contact.geom2)
            if min(a,b) < 0:
                continue
            ba,bb = int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
            names = (model.body(ba).name or '', model.body(bb).name or '')
            if not any('openarm_' in n for n in names):
                # A carried object must clear all obstacles too.
                if self.carried is None or self.env.bottle_id not in (ba,bb):
                    continue
                self.last_collision={'geoms':[model.geom(a).name,model.geom(b).name],'depth_m':-float(contact.dist)}
                return True
            if self.env.bottle_id in (ba,bb):
                other = names[1] if ba == self.env.bottle_id else names[0]
                allowed_sides={s for s,held in self.env.evaluation['held'].items() if held}
                if allow_fingers:allowed_sides.add(self.side)
                if 'finger' in other and any(f'openarm_{s}_' in other for s in allowed_sides):
                    continue
            self.last_collision={'geoms':[model.geom(a).name,model.geom(b).name],'depth_m':-float(contact.dist)}
            return True
        return False

    def _ik(self, target, allow_fingers=False, *, position_tolerance=.0015):
        self._sync()
        initial = self.data.qpos[self.qids].copy()
        bounds = np.array([self.env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1,8)])
        jp,jr = np.zeros((3,self.env.model.nv)),np.zeros((3,self.env.model.nv))
        best=None;collisions=[]
        # Audited real joint poses are IK proposals, never direct execution.
        # This avoids repeatedly searching from an unrelated arm configuration.
        seeds=[initial]+[np.clip(q,bounds[:,0],bounds[:,1]) for q in self.ik_seeds[self.side]]
        for trial in range(12):
            self.stats['ik_attempts'] += 1
            q = seeds[trial].copy() if trial<len(seeds) else np.clip(initial+self.rng.normal(0,.65,7),bounds[:,0],bounds[:,1])
            for _ in range(130):
                self._check();self._at(q);point,rot = self._pose()
                # Rotation log in world coordinates, well-behaved at pi.
                angular=np.zeros(3)
                if self.rotation is not None:
                    quat=np.empty(4);mujoco.mju_mat2Quat(quat,(self.rotation@rot.T).ravel())
                    mujoco.mju_quat2Vel(angular,quat,1.)
                delta=target-point
                cost=float(np.linalg.norm(delta)+.15*np.linalg.norm(angular))
                if best is None or cost<best['cost']:
                    best={'cost':cost,'position_error_m':float(np.linalg.norm(delta)),'rotation_error_rad':float(np.linalg.norm(angular))}
                if np.linalg.norm(delta)<position_tolerance and np.linalg.norm(angular)<.018:
                    if not self._collision(q,allow_fingers):
                        return q
                    if self.last_collision not in collisions:collisions.append(self.last_collision)
                    break
                mujoco.mj_jac(self.env.model,self.data,jp,jr,point,self.body)
                jac=np.vstack((jp[:,self.dids],(.15 if self.rotation is not None else 0.)*jr[:,self.dids]))
                error=np.r_[delta,.15*angular]
                step=jac.T@np.linalg.solve(jac@jac.T+.0001*np.eye(6),error)
                q=np.clip(q+step*min(1.,.16/max(np.linalg.norm(step),1e-9)),bounds[:,0],bounds[:,1])
        self.result['planning_diagnostics']={'arm':self.side,'phase':self.phase,'best_ik':best,'blocked_pairs':collisions}
        if collisions:
            raise PlanningFailure('goal_collision','已找到可达逆解，但目标处存在碰撞；请查看阻挡的几何体')
        raise PlanningFailure('ik_unreachable', '在关节范围内未找到满足目标位置和朝向的逆解')

    def _edge(self, start, end, allow_fingers=False):
        count=max(1,int(np.ceil(np.max(abs(end-start))/.025)))
        return not any(self._collision(start+(end-start)*i/count,allow_fingers) for i in range(1,count+1))

    def _path(self, goal, allow_fingers=False):
        self._sync();start=self.env.data.qpos[self.qids].copy()
        if self._collision(start,allow_fingers):
            self.result['planning_diagnostics']={'arm':self.side,'phase':self.phase,
                'blocked_pairs':[self.last_collision], 'location':'current_state'}
            raise PlanningFailure('start_collision','当前状态存在碰撞，无法直接规划；请检查场景或重置')
        if self._edge(start,goal,allow_fingers):
            return [start,goal]
        nodes,parents=[start],[-1]
        bounds=np.array([self.env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1,8)])
        for _ in range(400):
            self._check()
            sample=goal if self.rng.random()<.25 else self.rng.uniform(bounds[:,0],bounds[:,1])
            index=int(np.argmin([np.linalg.norm(q-sample) for q in nodes]));delta=sample-nodes[index]
            next_q=nodes[index]+delta*min(1.,.25/max(np.linalg.norm(delta),1e-9))
            if not self._edge(nodes[index],next_q,allow_fingers):
                continue
            nodes.append(next_q);parents.append(index);self.stats['rrt_nodes']+=1
            if self._edge(next_q,goal,allow_fingers):
                result=[goal];i=len(nodes)-1
                while i>=0:
                    result.append(nodes[i]);i=parents[i]
                return result[::-1]
        raise PlanningFailure('path_not_found','有可达逆解，但未在预算内找到无碰撞路径')

    def _move(self, target, allow_fingers=False, *, position_tolerance=.012):
        command_target=np.array(target,copy=True)
        desired_rotation=None if self.rotation is None else self.rotation.copy()
        command_rotation=None if self.rotation is None else self.rotation.copy()
        for correction in range(4):
            self.rotation=command_rotation
            goal=self._ik(command_target,allow_fingers,position_tolerance=min(.0015,position_tolerance*.35));path=self._path(goal,allow_fingers)
            for start,end in zip(path[:-1],path[1:]):
                bounds=np.array([self.env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1,8)])
                # MuJoCo limits are soft constraints: a measured encoder can
                # be just outside a stop. Keep target commands inside the
                # declared range, while checking the actual start above.
                target_start=np.clip(start,bounds[:,0],bounds[:,1])
                if np.max(abs(target_start-start))>.01:
                    raise PlanningFailure('joint_limit_violation','实测关节超过限位范围 0.01 rad，拒绝继续规划')
                count=max(24,int(np.ceil(np.max(abs(end-start))/(.018*self.speed_scale))))
                for i in range(1,count+1):
                    self._check();u=i/count;blend=u*u*(3-2*u)
                    action=self.base_action.copy();action[self.start:self.start+7]=target_start+(end-target_start)*blend
                    yield action,{'phase':self.phase,'scene_source':'simulator_ground_truth',
                                  'path_nodes':len(path),'feedback_correction':correction}
                    yield from self._check_hold(action)
                self.base_action[self.start:self.start+7]=end
            for _ in range(15):
                yield self.base_action.copy(),{'phase':self.phase,'settling':True}
                yield from self._check_hold(self.base_action)
            self._sync();point,actual_rotation=self._pose()
            error=target-point
            angular=np.zeros(3)
            if desired_rotation is not None:
                quat=np.empty(4);mujoco.mju_mat2Quat(quat,(desired_rotation@actual_rotation.T).ravel())
                mujoco.mju_quat2Vel(angular,quat,1.)
            self.rotation=desired_rotation
            self.result['tracking_feedback']={'position_error_m':float(np.linalg.norm(error)),
                'rotation_error_rad':float(np.linalg.norm(angular)),'corrections':correction}
            if np.linalg.norm(error)<=position_tolerance and (not self.strict_orientation or np.linalg.norm(angular)<=.06):
                return
            # Closed-loop Cartesian correction for load-induced servo error.
            # Every correction goes through IK and collision checking again.
            command_target += np.clip(error,-.04,.04)
            if self.strict_orientation and command_rotation is not None:
                angle=float(np.linalg.norm(angular))
                if angle>1e-8:
                    # Compensation is a target adjustment, never a live pose
                    # edit. Re-run IK and full collision checks each time.
                    quat=np.empty(4);matrix=np.empty(9)
                    mujoco.mju_axisAngle2Quat(quat,angular/angle,min(angle,.15))
                    mujoco.mju_quat2Mat(matrix,quat)
                    command_rotation=matrix.reshape(3,3)@command_rotation
        raise PlanningFailure('tracking_error','规划已执行，但反馈修正后末端仍未到达目标范围')

    def _check_hold(self, action):
        if self.carried is None or self.env.evaluation['held'][self.side]:
            return
        # Stop path advancement during a contact transient. Preserve the
        # current targets for at most 300 ms to re-establish the evaluator's
        # unbroken 100 ms two-finger dwell; never count intermittent touches as
        # stable grasp, and never continue carrying on an unconfirmed hold.
        self.result['contact_waits']=self.result.get('contact_waits',0)+1
        for _ in range(9):
            self._check()
            yield action.copy(),{'phase':self.phase,'contact_wait':True,'scene_source':'simulator_ground_truth'}
            if self.env.evaluation['held'][self.side]:
                return
        raise PlanningFailure('grasp_lost','暂停路径 300 ms 后仍未恢复持续双指夹持；停止后续操作')

    def _pick(self):
        side=self.pick_arm
        self._arm(side)
        bottle=self.env.data.xpos[self.env.bottle_id].copy()
        # Candidate neck/cap grasp fits the currently declared motor opening.
        # This is explicit model geometry, not inferred camera perception.
        grasp=bottle+np.array([0.,0.,.195])
        pre=grasp-np.array([.13,0.,0.])
        if side=='left':
            grasp,pre=self._choose_pick_pose(bottle)
            yield from self._grip(side,self.env.grippers[side].open_motor)
        else:
            self.base_action[15]=self.env.profile['action_gripper_open']
        self._phase('approach');yield from self._move(pre)
        if self.task=='reach_bottle':
            return
        self._phase('grasp');yield from self._move(grasp)
        self._phase('close')
        index=7 if side=='left' else 15
        start=float(self.base_action[index])
        for i in range(45):
            action=self.base_action.copy();action[index]=start+(.05-start)*min(1.,(i+1)/24)
            yield action,{'phase':'close','scene_source':'simulator_ground_truth'}
        self.base_action[index]=.05
        if not self.env.evaluation['held'][side]:
            raise PlanningFailure('grasp_not_confirmed','夹爪已闭合，但没有形成持续双指接触；未开始抬升')
        hand=self.env.data.xmat[self.body].reshape(3,3)
        self.carried=(hand.T@(self.env.data.xpos[self.env.bottle_id]-self.env.data.xpos[self.body]),
                      hand.T@self.env.data.xmat[self.env.bottle_id].reshape(3,3))
        self._phase('lift');yield from self._move(grasp+np.array([0.,0.,.12]),True)
        self._phase('hold')
        for _ in range(30):
            yield self.base_action.copy(),{'phase':'hold','scene_source':'simulator_ground_truth'}
            if not self.env.evaluation['held'][side]:
                raise PlanningFailure('grasp_lost','抬升过程中失去稳定双指夹持')
        lo,_=self.env._bottle_bounds()
        if lo[2] < bottle[2]+.06:
            raise PlanningFailure('lift_not_confirmed','机器人已抬手，但瓶子没有离开桌面达到目标高度')
        self.result.update(arm=side,verification_source='native_two_finger_contact_and_lift',
                           held_by=side,lift_height_m=float(lo[2]-bottle[2]))

    def _choose_pick_pose(self,bottle):
        """Search upright bottle grasps in scratch state, never move a live object."""
        failures=[]
        rotations=[self.receiver_rotation.copy()]
        for theta in (-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2):
            rotations.append(np.array([[0,np.sin(theta),np.cos(theta)],
                                       [0,-np.cos(theta),np.sin(theta)],[1,0,0.]]))
        # Retain the proven proposals first, then fill their coarse yaw gaps.
        # IK/collision checks and native contact verification stay unchanged.
        for theta in np.deg2rad([-75,-60,-30,-15,15,30,60,75]):
            rotations.append(np.array([[0,np.sin(theta),np.cos(theta)],
                                       [0,-np.cos(theta),np.sin(theta)],[1,0,0.]]))
        try:
            self.planning_qpos=self.env.data.qpos.copy()
            for finger in self.env.fingers[self.side]:
                self.planning_qpos[finger.qposadr[0]]=max(finger.range)
            for height in (.195,.095,.145):
                for rotation in rotations:
                    self.rotation=rotation
                    grasp=bottle+np.array([0,0,height]);pre=grasp-.10*rotation[:,2]
                    try:
                        self._ik(pre);self._ik(grasp,True)
                        self.result['grasp_search']={'arm':self.side,'height_m':height,
                            'rotation':rotation.tolist(),'rejected':failures}
                        self.result.pop('planning_diagnostics',None)
                        return grasp,pre
                    except PlanningFailure as exc:
                        if exc.code in ('cancelled','planning_timeout'):raise
                        failures.append({'height_m':height,'code':exc.code,
                                         'diagnostics':self.result.get('planning_diagnostics')})
            self.result['grasp_search']={'arm':self.side,'rejected':failures}
            raise PlanningFailure('grasp_pose_unreachable','当前瓶子位置未找到选定机械臂可达且无碰撞的抓取位姿；需调整接近方向或先由另一只手递送')
        finally:self.planning_qpos=None
    def _grip(self, side, value, *, samples=45, ramp_samples=24):
        index=7 if side=='left' else 15
        start=float(self.base_action[index])
        for i in range(samples):
            self._check()
            action=self.base_action.copy();action[index]=start+(value-start)*min(1.,(i+1)/ramp_samples)
            yield action,{'phase':self.phase,'scene_source':'simulator_ground_truth','native_finger_range':True}
        self.base_action[index]=value

    def _handover(self):
        if self.donor=='left':
            yield from self._reverse_handover()
            return
        target_bottle=self._choose_handover()
        self._arm('right',True);self._sync();point,_=self._pose()
        bottle=self.env.data.xpos[self.env.bottle_id].copy()
        self._phase('transfer')
        yield from self._move(point+target_bottle-bottle,True)
        if not self.env.evaluation['held']['right']:
            raise PlanningFailure('grasp_lost','到达交接区域前右手失去夹持')
        self._arm('left');self._phase('receive')
        yield from self._grip('left',self.env.grippers['left'].open_motor)
        bottle=self.env.data.xpos[self.env.bottle_id].copy()
        grasp=bottle+np.array([0.,0.,.095])
        yield from self._move(grasp-.06*self.rotation[:,2])
        yield from self._move(grasp,True)
        self._phase('dual_hold');yield from self._grip('left',.05)
        if not all(self.env.evaluation['held'].values()):
            raise PlanningFailure('dual_hold_not_confirmed','左手未形成稳定双指夹持，右手保持闭合，不释放瓶子')
        self._phase('release_right');yield from self._grip('right',self.env.grippers['right'].open_motor)
        if not self.env.evaluation['held']['left']:
            raise PlanningFailure('left_grasp_lost','释放右手后左手夹持未保持')
        self._phase('retreat');self._arm('right');self._sync();point,_=self._pose()
        # Retreat toward the donor's original workspace. Pulling straight back
        # at this height traps the elbow against the torso.
        yield from self._retreat_after_forward_handover(point)
        if not self.env._left_held or not self.env.evaluation['held']['left']:
            raise PlanningFailure('handover_not_confirmed','右手撤离后未确认左手单独持瓶')
        self.result.update(from_arm='right',to_arm='left',held_by='left',
                           verification_source='native_receiver_contact_then_donor_release')

    def _retreat_after_forward_handover(self, point):
        """Original fixed-demo donor withdrawal, preserved by default."""
        yield from self._move(point+np.array([.04,-.10,-.08]))

    def _reverse_handover(self):
        donor,receiver=self.donor,self.receiver
        failures=[];chosen=None
        self._arm(donor,True)
        donor_rotation=self.env.data.xmat[self.body].reshape(3,3).copy()
        receiver_rotation=np.array([[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]])
        # Current donor grasp determines the receiver's free grasp band.
        bottle_axis=self.env.data.xmat[self.env.bottle_id].reshape(3,3)[:,2]
        donor_height=float(np.dot(self.env.data.xpos[self.body]+donor_rotation@self.local_point-self.env.data.xpos[self.env.bottle_id],bottle_axis))
        receiver_height=.195 if donor_height<.145 else .095
        try:
            for z in (1.195,1.215,1.235,1.175):
                for x in (.46,.48,.50,.44,.52):
                    candidate=np.array([x,.02,z])
                    self.planning_qpos=None;self._arm(donor,True);self.rotation=donor_rotation
                    self._sync();point,_=self._pose();bottle=self.env.data.xpos[self.env.bottle_id]
                    try:
                        q=self._ik(point+candidate-bottle,True);self._at(q)
                        predicted=self.data.qpos.copy();predicted_bottle=self.data.xpos[self.env.bottle_id].copy()
                        predicted_axis=self.data.xmat[self.env.bottle_id].reshape(3,3)[:,2].copy()
                        self._arm(receiver);self.planning_qpos=predicted
                        for finger in self.env.fingers[receiver]:self.planning_qpos[finger.qposadr[0]]=max(finger.range)
                        grasp=predicted_bottle+predicted_axis*receiver_height
                        orientation_failures=[]
                        for theta in (np.pi/4,np.pi/2,0,-np.pi/4):
                            self.rotation=self._grasp_frame(predicted_axis,theta)
                            try:
                                self._ik(grasp-.10*self.rotation[:,2]);self._ik(grasp,True)
                                receiver_rotation=self.rotation.copy();chosen=candidate;break
                            except PlanningFailure as exc:
                                if exc.code in ('cancelled','planning_timeout'):raise
                                orientation_failures.append({'theta':theta,'code':exc.code,'diagnostics':self.result.get('planning_diagnostics')})
                        if chosen is not None:break
                        failures.append({'candidate':candidate.tolist(),'receiver_orientations':orientation_failures})
                    except PlanningFailure as exc:
                        if exc.code in ('cancelled','planning_timeout'):raise
                        failures.append({'candidate':candidate.tolist(),'code':exc.code,'diagnostics':self.result.get('planning_diagnostics')})
                if chosen is not None:break
        finally:self.planning_qpos=None
        self.result['handover_search']={'from_arm':donor,'to_arm':receiver,'selected_bottle_world':None if chosen is None else chosen.tolist(),'receiver_height_m':receiver_height,'rejected':failures}
        if chosen is None:raise PlanningFailure('handover_unreachable','当前夹持姿态未找到双方可达且无碰撞的交接区域')
        self._arm(donor,True);self.rotation=donor_rotation;self._sync();point,_=self._pose()
        self._phase('transfer')
        try:
            yield from self._move(point+chosen-self.env.data.xpos[self.env.bottle_id],True)
        except PlanningFailure as exc:
            if exc.code not in ('ik_unreachable','path_not_found') or not self.env.evaluation['held'][donor]:raise
            # Native tracking can shift the object within a body grasp.
            # Re-ground at the reached physical pose instead of pretending
            # the donor achieved the predicted bottle position.
            self.result['recovery']={'code':exc.code,'strategy':'replan_receiver_from_actual_bottle',
                                     'actual_bottle_position_m':self.env.data.xpos[self.env.bottle_id].tolist()}
        if not self.env.evaluation['held'][donor]:raise PlanningFailure('grasp_lost','供物手在接收前失去夹持')
        self._arm(receiver);self._phase('receive')
        yield from self._grip(receiver,self.env.grippers[receiver].open_motor)
        bottle_axis=self.env.data.xmat[self.env.bottle_id].reshape(3,3)[:,2].copy()
        grasp=self.env.data.xpos[self.env.bottle_id]+bottle_axis*receiver_height
        # Always replan from the post-transit object observation, not a stale
        # geometric prediction made before the donor moved.
        solved=False;attempts=[]
        for theta in (float(np.arctan2(receiver_rotation[1,2],receiver_rotation[0,2])),0,np.pi/4,np.pi/2,-np.pi/4):
            self.rotation=self._grasp_frame(bottle_axis,theta)
            try:
                self._ik(grasp-.10*self.rotation[:,2]);self._ik(grasp,True);solved=True;break
            except PlanningFailure as exc:
                if exc.code in ('cancelled','planning_timeout'):raise
                attempts.append({'theta':theta,'code':exc.code,'diagnostics':self.result.get('planning_diagnostics')})
        self.result['receiver_replan']={'bottle_world':self.env.data.xpos[self.env.bottle_id].tolist(),'bottle_axis':bottle_axis.tolist(),'rotation':self.rotation.tolist(),'rejected':attempts}
        if not solved:raise PlanningFailure('handover_unreachable','供物手到位后，接收手未在实际瓶子位姿处找到安全逆解')
        self.result.pop('planning_diagnostics',None)
        yield from self._move(grasp-.10*self.rotation[:,2]);yield from self._move(grasp,True)
        self._phase('dual_hold');yield from self._grip(receiver,.05)
        if not all(self.env.evaluation['held'].values()):raise PlanningFailure('dual_hold_not_confirmed','接收手尚无持续双指接触，供物手保持闭合')
        self._phase('release_right');yield from self._grip(donor,self.env.grippers[donor].open_motor)
        if not self.env.evaluation['held'][receiver]:raise PlanningFailure('receiver_grasp_lost','供物手释放后，接收手未保持夹持')
        self._phase('retreat');self._arm(donor);self.rotation=donor_rotation;self._sync();point,_=self._pose()
        # Donor is empty: relax its grasp orientation while withdrawing.
        # Receiver targets stay fixed and all geometry is still collision checked.
        self.rotation=None
        yield from self._move(point-.06*donor_rotation[:,2]+np.array([0,.02,0]))
        if not self.env.evaluation['held'][receiver] or self.env.evaluation['touching'][donor]:
            raise PlanningFailure('handover_not_confirmed','撤离后未确认接收手独立持瓶')
        self.result.update(from_arm=donor,to_arm=receiver,held_by=receiver,
                           verification_source='native_receiver_contact_then_donor_release')

    @staticmethod
    def _grasp_frame(axis,theta):
        """Align pad span to the observed bottle axis, approach perpendicular."""
        axis=np.asarray(axis,dtype=float);axis=axis/np.linalg.norm(axis)
        approach=np.array([np.cos(theta),np.sin(theta),0.])
        approach-=axis*np.dot(approach,axis);approach/=np.linalg.norm(approach)
        return np.column_stack((axis,np.cross(approach,axis),approach))

    def _choose_handover(self):
        """Geometric refinement of a handover operator in scratch state only.

        Require donor IK plus receiver pregrasp and grasp IK with the donor
        already at its candidate pose. Each live segment is replanned later.
        """
        self._phase('transfer')
        attempts=[]
        try:
            for z in (1.215,1.235,1.195,1.255):
                for x in (.48,.50,.46,.54):
                    candidate=np.array([x,-.02,z])
                    self.planning_qpos=None;self._arm('right',True);self._sync()
                    point,_=self._pose();bottle=self.env.data.xpos[self.env.bottle_id]
                    try:
                        donor=self._ik(point+candidate-bottle,True)
                        self._at(donor)
                        predicted=self.data.qpos.copy()
                        predicted_bottle=self.data.xpos[self.env.bottle_id].copy()
                        self._arm('left');self.planning_qpos=predicted
                        # Receiver must have the full declared model opening.
                        for finger in self.env.fingers['left']:
                            self.planning_qpos[finger.qposadr[0]]=max(finger.range)
                        grasp=predicted_bottle+np.array([0.,0.,.095])
                        self._ik(grasp-.06*self.rotation[:,2]);self._ik(grasp,True)
                        self.result['handover_search']={'selected_bottle_world':candidate.tolist(),'rejected':attempts}
                        self.result.pop('planning_diagnostics',None)
                        return candidate
                    except PlanningFailure as exc:
                        if exc.code in ('cancelled','planning_timeout'):raise
                        attempts.append({'candidate':candidate.tolist(),'diagnostics':self.result.get('planning_diagnostics')})
            self.result['handover_search']={'selected_bottle_world':None,'rejected':attempts}
            raise PlanningFailure('handover_unreachable','候选交接区域未找到同时满足双臂可达与碰撞条件的位置')
        finally:
            self.planning_qpos=None

    def _legacy_place(self):
        self._arm('left',True);self._sync();point,_=self._pose()
        bottle=self.env.data.xpos[self.env.bottle_id].copy()
        box=self.env.data.xpos[self.env.box_id]
        self._phase('place')
        # Turn the grasp about world vertical to reach the left workspace.
        # The carried bottle retains its tilt; the target accounts for the
        # rotated grasp offset instead of treating the hand as the object.
        current=self.env.data.xmat[self.body].reshape(3,3).copy()
        self.rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])@current
        target=box+np.array([0.,0.,.13])+self.rotation@(self.local_point-self.carried[0])
        yield from self._move(target,True)
        if not self.env.evaluation['held']['left']:
            raise PlanningFailure('left_grasp_lost','到达篮子上方前左手失去夹持')
        # Lower before release to reduce drop impact. Re-estimate the measured
        # grasp transform after the transit; both hand and object paths must
        # still clear the basket rim.
        self._arm('left',True)
        self.rotation=self.env.data.xmat[self.body].reshape(3,3).copy()
        lower=box+np.array([0.,0.,.035])+self.rotation@(self.local_point-self.carried[0])
        yield from self._move(lower,True)
        if not self.env.evaluation['held']['left']:
            raise PlanningFailure('left_grasp_lost','降低到篮内时失去夹持；停止后续释放指令')
        self._phase('release_left');yield from self._grip('left',self.env.grippers['left'].open_motor)
        self._phase('settle')
        for _ in range(60):
            yield self.base_action.copy(),{'phase':'settle','scene_source':'simulator_ground_truth'}
        if not self.env.evaluation['success']:
            raise PlanningFailure('placement_not_confirmed','释放后未满足完整外形入篮、篮底支撑和持续稳定条件')

    def _placement_upright(self):
        return self.placement_surface == 'table'

    def _placement_yaws(self):
        return (0.,-.25,.25,-.50,.50,np.pi/4,-np.pi/4,np.pi/2,-np.pi/2,np.pi)

    def _place(self):
        """Ground target in support geometry, keep a measured grasp transform.

        Planning attachments remain scratch-only. Release is permitted only
        after the carried body reaches the declared support region.
        """
        side=self.place_arm;self._arm(side,True);self.strict_orientation=self._placement_upright()
        surface=self.env.placement_surface(self.placement_surface)
        base=np.r_[self.placement_xy,surface['top_z']]
        self._phase('place')
        measured=self.env.data.xmat[self.env.bottle_id].reshape(3,3)
        yaw=np.arctan2(measured[1,0],measured[0,0])
        rotation=(np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
                  if self._placement_upright() else measured.copy())
        self.rotation=rotation@self.carried[1].T
        offset=self.rotation@(self.local_point-self.carried[0])
        # Search alternative upright yaw about the object's vertical axis.
        # The target position is always the user's, never silently changed.
        choices=[]
        selected=None
        # Release above the table instead of pressing a loaded gripper onto it.
        # Native free fall/contact still has to satisfy the final support gate.
        lower_height=.020 if self.placement_surface=='table' else .019
        heights=(.12,.08,.05) if self.placement_surface=='table' else (.114,)
        for height in heights:
            for angle in self._placement_yaws():
                turn=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
                self.rotation=turn@rotation@self.carried[1].T
                offset=self.rotation@(self.local_point-self.carried[0])
                try:
                    self._ik(base+[0,0,height]+offset,True)
                    self._ik(base+[0,0,lower_height]+offset,True)
                    selected=(height,angle)
                    self.result['placement_search']={'yaw_offset_rad':angle,'transit_clearance_m':height,'rejected':choices}
                    self.result.pop('planning_diagnostics',None);break
                except PlanningFailure as exc:
                    if exc.code in ('cancelled','planning_timeout'):raise
                    choices.append({'yaw_offset_rad':angle,'height_m':height,'failure_code':exc.code,'diagnostics':self.result.get('planning_diagnostics')})
            if selected:break
        if not selected:
            self.result['placement_search']={'rejected':choices}
            raise PlanningFailure('placement_unreachable','指定放置点未找到兼顾携物外形与夹爪姿态的可达方案')
        yield from self._move(base+[0,0,selected[0]]+offset,True)
        # Refresh the grasp offset after transport so servo slip is not ignored.
        desired_object_rotation=turn@rotation
        self._arm(side,True);self.rotation=desired_object_rotation@self.carried[1].T
        offset=self.rotation@(self.local_point-self.carried[0])
        precision=min(.003,self.env.placement_target['tolerance_m']*.6)
        yield from self._move(base+[0,0,lower_height]+offset,True,position_tolerance=max(.0015,precision))
        # Use measured OBJECT error, including slip and loaded servo error.
        # Corrections change actuator goals only; every path is checked again.
        desired_bottom=base[2]+lower_height
        attempts=[];alignment_rotation=self.env.data.xmat[self.body].reshape(3,3).copy()
        for correction in range(91):
            self._check()
            actual=self.env.data.xpos[self.env.bottle_id].copy()
            xy_error=base[:2]-actual[:2]
            bottom=float(self.env._bottle_bounds()[0][2])
            if correction%3==0 or np.linalg.norm(xy_error)<=precision:
                attempts.append({'iteration':correction,'object_xy_m':actual[:2].tolist(),
                    'position_error_m':float(np.linalg.norm(xy_error)),'bottom_z_m':bottom,
                    'held':bool(self.env.evaluation['held'][side]),'tolerance_m':precision})
            self.result['object_tracking_feedback']=attempts
            if np.linalg.norm(xy_error)<=precision and abs(bottom-desired_bottom)<=.006:
                break
            if correction==90:break
            # At the native command rate, integrate small measured-object
            # corrections into the loaded joint targets. Preserve the current
            # grasp attitude. Whole IK trajectories per correction can leave
            # a tall, slipping bottle behind while the TCP catches up.
            self._arm(side,True);self._sync();point,actual_rotation=self._pose()
            self.rotation=alignment_rotation.copy()
            adjustment=np.r_[xy_error,desired_bottom-bottom]
            goal=self._ik(point+np.clip(adjustment*.6,-.004,.004),True,position_tolerance=.0002)
            measured=self.env.data.qpos[self.qids].copy()
            bounds=np.array([self.env.model.joint(f'openarm_{side}_joint{i}').range for i in range(1,8)])
            if np.max(np.maximum(bounds[:,0]-measured,measured-bounds[:,1]))>.01:
                raise PlanningFailure('joint_limit_violation','物体修正期间实测关节超出限位 0.01 rad，拒绝继续')
            next_q=np.clip(self.base_action[self.start:self.start+7]+np.clip((goal-measured)*.5,-.008,.008),bounds[:,0],bounds[:,1])
            if not self._edge(measured,next_q,True):
                self.result['planning_diagnostics']={'arm':side,'phase':'object_alignment','blocked_pairs':[self.last_collision]}
                raise PlanningFailure('goal_collision','物体反馈修正路径发生碰撞，拒绝继续')
            self.base_action[self.start:self.start+7]=next_q
            yield self.base_action.copy(),{'phase':self.phase,'object_alignment':correction,'scene_source':'simulator_ground_truth'}
            yield from self._check_hold(self.base_action)
        actual=self.env.data.xpos[self.env.bottle_id]
        if np.linalg.norm(actual[:2]-base[:2])>precision or abs(self.env._bottle_bounds()[0][2]-desired_bottom)>.006:
            raise PlanningFailure('release_pose_not_reached','瓶子本体未到达放置区域，拒绝松手；末端到位不能代替物体到位')
        self._phase('release_left');yield from self._grip(side,self.env.grippers[side].open_motor,samples=30,ramp_samples=6)
        self.carried=None
        self._sync();point,rot=self._pose();self.rotation=rot.copy()
        self._phase('retreat')
        # An open jaw can be contact-free while still surrounding the bottle.
        # Search a withdrawal goal that also clears the CLOSED jaw in scratch
        # geometry; execute it open, through unchanged native collision checks.
        cleared=False;retreat_trials=[]
        withdrawals=[-distance*rot[:,2] for distance in (.10,.14,.18)]
        withdrawals += [np.array([0.,0.,height]) for height in (.08,.12,.18)]
        withdrawals += [-distance*rot[:,2]+[0,0,.08] for distance in (.06,.10)]
        for delta in withdrawals:
            candidate=point+delta
            try:q=self._ik(candidate)
            except PlanningFailure as exc:
                if exc.code in ('cancelled','planning_timeout'):raise
                retreat_trials.append({'delta_m':delta.tolist(),'code':exc.code,'diagnostics':self.result.get('planning_diagnostics')});continue
            if self._probe_closed_jaw(q):
                retreat_trials.append({'delta_m':delta.tolist(),'code':'closed_jaw_object_contact'});continue
            # This path uses actual open-jaw geometry and all native collisions.
            yield from self._move(candidate)
            if self._probe_closed_jaw(self.env.data.qpos[self.qids]):
                retreat_trials.append({'delta_m':delta.tolist(),'code':'actual_closed_jaw_object_contact'});continue
            cleared=True;self.result['retreat_delta_m']=delta.tolist()
            self.result.pop('planning_diagnostics',None);break
        self.result['retreat_search']=retreat_trials
        if not cleared:raise PlanningFailure('retreat_not_clear','释放后找不到实测夹爪闭合时也避开瓶子的撤离位姿')
        self._phase('settle')
        for _ in range(90):
            yield self.base_action.copy(),{'phase':'settle','scene_source':'simulator_ground_truth'}
            report=self.env.evaluation.get('placement',{})
            if report.get('confirmed'):
                self.result.update(verification_source='native_support_release_and_stability',
                    held_by=None,actual_bottle_position_m=self.env.data.xpos[self.env.bottle_id].tolist(),
                    placement=dict(report),position_error_m=float(np.linalg.norm(self.env.data.xpos[self.env.bottle_id,:2]-base[:2])))
                return
        self.result['placement']=dict(self.env.evaluation.get('placement',{}))
        raise PlanningFailure('placement_not_confirmed','松手后未满足目标区域、直立、支撑、脱离夹爪及连续 1 秒稳定条件')

    def _probe_closed_jaw(self,q):
        # Only additional object clearance is probed at the commanded closure.
        # The actual jaw settles against its own pad contacts in native physics;
        # this hypothetical closing geometry never replaces executed state.
        saved=self.planning_qpos
        try:
            self.planning_qpos=self.env.data.qpos.copy()
            for finger in self.env.fingers[self.side]:
                self.planning_qpos[finger.qposadr[0]]=np.clip(self.env.grippers[self.side].to_joint(self.env.grippers[self.side].closed_motor),*finger.range)
            self._sync()
            return self._closed_jaw_touches_bottle(q)
        finally:self.planning_qpos=saved

    def _closed_jaw_touches_bottle(self,q):
        self._at(q)
        model=self.env.model
        for contact in self.data.contact:
            a,b=int(contact.geom1),int(contact.geom2)
            if min(a,b)<0:continue
            ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
            if self.env.bottle_id not in (ba,bb):continue
            other=bb if ba==self.env.bottle_id else ba
            name=model.body(other).name or ''
            if f'openarm_{self.side}_' in name and 'finger' in name:return True
        return False

    def actions(self):
        if self.task=='set_gripper':
            side=self.parameters['arm'];fraction=self.parameters['opening_fraction'];mapping=self.env.grippers[side]
            motor=mapping.open_motor*fraction+mapping.closed_motor*(1-fraction)
            self._phase('gripper');yield from self._grip(side,motor)
            history=[];index=7 if side=='left' else 15
            for _ in range(30):
                yield self.base_action.copy(),{'phase':'gripper','source':'robot_gripper_encoder'}
                history.append(float(self.env.state()[index]))
            if max(history[-6:])-min(history[-6:])>.01:
                raise PlanningFailure('gripper_not_settled','夹爪编码器仍在变化；开合未确认，不代表夹住物体')
            self.result.update(status='completed',primitive_success=True,arm=side,
                requested_opening_fraction=fraction,measured_motor_angle_rad=history[-1],
                held_after=dict(self.env.evaluation['held']),touching_after=dict(self.env.evaluation['touching']),
                verification_source='robot_gripper_encoder_settling',
                scope='gripper command and settling only; no object grasp or placement claim',stats=self.stats)
            return
        if self.task in ('move_to_pose', 'move_relative'):
            self._phase('move')
            yield from self._move(self.motion_target)
            self._sync()
            position, rotation = self._pose()
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (self.rotation @ rotation.T).ravel())
            angular = np.empty(3)
            mujoco.mju_quat2Vel(angular, quat, 1.)
            rotation_error = float(np.linalg.norm(angular))
            self.result.update(actual_position_m=position.tolist(),
                position_error_m=float(np.linalg.norm(position-self.motion_target)),
                rotation_error_rad=rotation_error, verification_source='robot_forward_kinematics')
            if rotation_error > .08:
                raise PlanningFailure('orientation_tracking_error', '位置已接近目标，但末端朝向误差超过 0.08 rad')
            self.result.update(status='completed', primitive_success=True, stats=self.stats)
            return
        self.start=8
        if self.task in ('reach_bottle','pick_bottle','handover_to_basket'):
            yield from self._pick()
        if self.task in ('handover_bottle','handover_to_basket'):
            yield from self._handover()
        if self.task in ('place_bottle','handover_to_basket'):
            if self.task=='handover_to_basket' or not self.parameters:yield from self._legacy_place()
            else:yield from self._place()
        self.result.update(status='completed',primitive_success=True,stats=self.stats)
