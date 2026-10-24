"""Optional goal-conditioned bottle grasp/relay search, native execution.

The original bottle demo and actuator/contact/placement gates remain unchanged.
Proposal clearance and task lookahead are additional planning constraints only.
"""
import copy

import mujoco
import numpy as np

from .grasp import candidates, cylinder_bands
from .object_planner import ObjectPlanner
from .object_handover import ObjectHandoverPlanner, receiver_candidates
from .oriented_grasp import OrientedPlanner, HoldAudit
from .tamp import PlanningFailure

CONTRACT = {'schema':'bottle-goal-routing/1', 'source':'simulator_ground_truth',
    'proposal':'current bottle geometry and grasp relation; goal-conditioned ranking and native connected paths',
    'planning_clearance_m':.012, 'carried_object_clearance_m':.040,
    'transport_vertical_clearance_m':.080,
    'transport_proposal_clearance_m':.080,
    'grasp_joint_slack_rad':.020,
    'hardware_access':False, 'transfer_ready':False,
    'scope':'experimental registered upright bottle; original demo retained; not open-object perception'}


def bypass_origins(env,data):
    """Obstacle-derived side corridors; every hypothesis still needs native IK.

    X is the candidate robot's fore/aft axis. Bounds cover the current tilted
    bottle, rather than its nominal radius. No calibrated workspace is implied.
    """
    from .scene import bounds
    origin=data.xpos[env.bottle_id].copy()
    lo=np.min([bounds(env.model,data,g)[0] for g in env.bottle_geoms],axis=0)
    hi=np.max([bounds(env.model,data,g)[1] for g in env.bottle_geoms],axis=0)
    proposals=[]
    for ident,row in env.registry.items():
        if ident=='bottle':continue
        low=np.min([bounds(env.model,data,g)[0] for g in row['geoms']],axis=0)
        high=np.max([bounds(env.model,data,g)[1] for g in row['geoms']],axis=0)
        padding=CONTRACT['transport_proposal_clearance_m']+.003
        for axis,x in [('base_side',low[0]-(hi[0]-origin[0])-padding),
            ('front_side',high[0]+(origin[0]-lo[0])+padding)]:
            goal=origin.copy();goal[0]=x
            if .1<x<.9:proposals.append({'origin_m':goal,'around_object':ident,'side':axis})
    return sorted(proposals,key=lambda p:(p['side']!='base_side',abs(p['origin_m'][0]-origin[0])))


class ContinuousHoldAudit(HoldAudit):
    """Same motor commands, with every complete native audit window checked."""
    def actions(self):
        from .decision_timing import GraspStability
        audit=None;start=None;stable=True;completed=False;windows=0
        try:
            for action,diagnostic in super().actions():
                checking=diagnostic.get('hold_validation')=='native_contact_and_relative_pose'
                if checking and audit is None:
                    audit=GraspStability(self.env);start=float(self.env.data.time)
                yield action,diagnostic
                if checking:
                    audit.read();side=self.result['held_by']
                    if self.env.data.time-start+1e-9>=2:
                        windows+=1
                        stable &= audit.summary()['arms'][side]['status']=='stable_within_audit_window'
            completed=True;stable &= windows>0
        finally:
            # An early contact-loss/timeout remains a failed continuous audit,
            # not a legacy final-window receipt or an invented passed window.
            if 'hold_audit' in self.result:
                self.result['hold_audit'].update(validation_method='all_full_windows',
                    full_window_checks=windows,audit_completed=completed,
                    continuous_full_windows_stable=completed and stable and windows>0)
        if not stable:
            self.result.update(primitive_success=False,status='failed')
            raise PlanningFailure('grasp_unstable','At least one complete native hold window failed')


class BottlePickPlanner(ObjectPlanner):
    def __init__(self, env, task, **kwargs):
        super().__init__(env, task, **kwargs)
        if env.object_id != 'bottle' or task not in ('pick_bottle','pick_object','reach_bottle','reach_object'):
            raise ValueError('optional upright bottle pick provider only')
        future = self.parameters.get('next_handover')
        if future and (set(future) != {'to_arm'} or future['to_arm'] not in ('left','right')
            or future['to_arm'] == self.pick_arm): raise ValueError('distinct declared next receiver required')
        self._obstacles = [g for ident,row in env.registry.items() if ident!='bottle' for g in row['geoms']]
        self._robot = {side:[g for g in range(env.model.ngeom) if env.model.geom_contype[g]
            and f'openarm_{side}_' in (env.model.body(int(env.model.geom_bodyid[g])).name or '')]
            for side in ('left','right')}
        self.result['bottle_goal_contract'] = CONTRACT

    def _collision(self, q, allow_fingers=False):
        if super()._collision(q, allow_fingers): return True
        if not hasattr(self,'_obstacles'): return False
        obstacles = self._obstacles
        geoms = self._robot[self.side] + (list(self.env.bottle_geoms) if self.carried is not None else [])
        for obstacle in obstacles:
            for geom in geoms:
                gap=float(mujoco.mj_geomDistance(self.env.model,self.data,geom,obstacle,.012,None))
                if gap < .012:
                    self.last_collision={'geoms':[self.env.model.geom(g).name for g in (geom,obstacle)],
                        'separation_m':gap,'required_clearance_m':.012,'source':'additional native planning clearance'}
                    return True
        return False

    def _choose_pick_pose(self, ignored):
        env=self.env; seeds=copy.deepcopy(self.ik_seeds[self.side]); rejected=[]
        before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),env.data.time)
        try:
            self.planning_qpos=env.data.qpos.copy()
            for finger in env.fingers[self.side]:self.planning_qpos[finger.qposadr[0]]=max(finger.range)
            self._sync();values=list(candidates(cylinder_bands(env.model,self.data,env.bottle_id)))
            if self.parameters.get('next_handover'):
                shoulders=[env.data.xanchor[env.model.joint(f'openarm_{side}_joint1').id,1] for side in ('left','right')]
                mid=float(np.mean(shoulders));own=float(shoulders[0 if self.side=='left' else 1])
                cross_body=(float(env.data.xpos[env.bottle_id,1])-mid)*(own-mid)<0
                values.sort(key=lambda v:((-1 if cross_body else 1)*v['grasp_m'][2],abs(v['pitch_rad'])))
                self.result['grasp_order']={'source':'actual shoulder anchors and current bottle side',
                    'cross_body':cross_body,'ranking':'high bands first across the torso; low donor bands first on the same side',
                    'all_original_candidates_retained':True}
            elif self.parameters.get('next_placement'):
                center=env.data.xipos[env.bottle_id].copy()
                shoulder=env.data.xanchor[env.model.joint(f'openarm_{self.side}_joint1').id,:2].copy()
                bearing=float(np.arctan2(center[1]-shoulder[1],center[0]-shoulder[0]))
                def angular_distance(value):
                    delta=float(value['yaw_rad'])-bearing
                    return abs(float(np.arctan2(np.sin(delta),np.cos(delta))))
                def placement_rank(value):
                    lever=float(np.linalg.norm(value['grasp_m']-center))/max(value['radius_m'],1e-6)
                    return (-float(value['grasp_m'][2]),angular_distance(value),abs(float(value['pitch_rad'])),lever)
                values.sort(key=placement_rank)
                self.result['grasp_order']={'source':'native shoulder anchor to current bottle center; placement goal enables high-band hypothesis',
                    'ranking':'higher grasp band first, approach yaw toward bottle from selected arm, level pitch first, normalized lever-arm tie break',
                    'desired_approach_yaw_rad':bearing,'shoulder_xy_m':shoulder.tolist(),
                    'all_original_candidates_retained':True,
                    'scope':'proposal order for registered single-arm bottle table placement; native grasp/hold/transport/place still required'}
            for index,value in enumerate(values):
                self._check();self.rotation=value['rotation'];self.ik_seeds[self.side]=copy.deepcopy(seeds)
                try:
                    grasp=self._ik(value['grasp_m'],True)
                    limits=np.asarray([env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1,8)])
                    slack=np.minimum(grasp-limits[:,0],limits[:,1]-grasp)
                    if float(slack.min())<CONTRACT['grasp_joint_slack_rad']:
                        raise PlanningFailure('grasp_near_joint_stop','Candidate leaves no declared joint slack for loaded feedback correction')
                    self.ik_seeds[self.side].insert(0,grasp)
                    pre=self._ik(value['pregrasp_m']);self._path(pre)
                    if not self._edge(pre,grasp,True):raise PlanningFailure('approach_collision','Native bottle approach edge collides')
                    future=None;transport=None
                    if self.parameters.get('next_handover'):
                        transport=self._future_transport(grasp)
                    else:future=OrientedPlanner._future_place(self,grasp)
                    self.result['grasp_search']={'selected':ObjectHandoverPlanner._public(value),'candidate_index':index,
                        'rejected':rejected,'pregrasp_q':pre.tolist(),'grasp_q':grasp.tolist()}
                    if future:self.result['future_placement']=future
                    if transport:self.result['future_transport']=transport
                    self.result.pop('planning_diagnostics',None);self.ik_seeds[self.side]=[grasp,pre,*seeds]
                    return value['grasp_m'].copy(),value['pregrasp_m'].copy()
                except PlanningFailure as exc:
                    if exc.code in ('cancelled','planning_timeout'):raise
                    rejected.append({'index':index,'candidate':ObjectHandoverPlanner._public(value),'code':exc.code,
                        'diagnostics':copy.deepcopy(self.result.get('planning_diagnostics'))})
            self.result['grasp_search']={'rejected':rejected}
            raise PlanningFailure('grasp_search_exhausted','Native goal-conditioned bottle candidate pool exhausted')
        finally:
            self.planning_qpos=None
            for old,new in zip(before[:3],(env.data.qpos,env.data.qvel,env.data.ctrl)):np.testing.assert_array_equal(old,new)
            assert before[3]==env.data.time

    def _future_transport(self,grasp):
        """Reject grasps that cannot reach this scene's elevated transport.

        The prediction is scratch-only and leaves joint slack for the observed
        candidate controller's loaded correction. It is not a hold guarantee.
        """
        from .scene import bounds
        saved=(self.carried,self.rotation.copy(),self.phase,copy.deepcopy(self.ik_seeds))
        rejected=[]
        try:
            self._at(grasp);hand=self.data.xmat[self.body].reshape(3,3).copy()
            origin=self.data.xpos[self.env.bottle_id].copy();obj=self.data.xmat[self.env.bottle_id].reshape(3,3).copy()
            self.carried=(hand.T@(origin-self.data.xpos[self.body]),hand.T@obj);self.phase='transfer'
            top=max(bounds(self.env.model,self.data,g)[1][2] for g in self._obstacles)
            lower=min(bounds(self.env.model,self.data,g)[0][2] for g in self.env.bottle_geoms)
            goal=origin+[0,0,top+CONTRACT['transport_vertical_clearance_m']-lower]
            goals=[{**p,'origin_m':p['origin_m']+[0,0,.12]} for p in bypass_origins(self.env,self.data)]
            goals.append({'origin_m':goal,'side':'elevated','around_object':None})
            approach=float(np.arctan2(hand[1,2],hand[0,2]))
            limits=np.asarray([self.env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1,8)])
            for proposed in goals:
                for yaw in dict.fromkeys([0.,-approach,np.pi/4,-np.pi/4,np.pi/2,-np.pi/2]):
                    c,s=np.cos(yaw),np.sin(yaw);self.rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@hand
                    try:
                        target=proposed['origin_m']+self.rotation@(self.local_point-self.carried[0])
                        q=self._ik(target,True)
                        margin=float(np.minimum(q-limits[:,0],limits[:,1]-q).min())
                        if margin<.04:raise PlanningFailure('transport_near_joint_stop','Transport hypothesis leaves less than0.04rad joint slack')
                        if not self._edge(grasp,q,True):raise PlanningFailure('transport_lift_edge_collision','Connected native transport hypothesis collides')
                        return {'goal_object_origin_m':proposed['origin_m'].tolist(),'yaw_offset_rad':yaw,'joint_slack_rad':margin,
                            'route':proposed['side'],'around_object':proposed['around_object'],
                            'source':'scratch-only current grasp to obstacle-derived transport; actual native lift/hold/relay mandatory',
                            'rejected':rejected}
                    except PlanningFailure as exc:
                        if exc.code in ('cancelled','planning_timeout'):raise
                        rejected.append({'route':proposed['side'],'yaw_offset_rad':yaw,'code':exc.code,
                            'diagnostics':copy.deepcopy(self.result.get('planning_diagnostics'))})
            self.result['future_transport_search']={'rejected':rejected}
            raise PlanningFailure('transport_clearance_unreachable','This donor grasp has no connected elevated transport with joint slack')
        finally:self.carried,self.rotation,self.phase,self.ik_seeds=saved


class BottleHandoverPlanner(ObjectHandoverPlanner):
    def __init__(self, env, task, **kwargs):
        super().__init__(env,task,**kwargs)
        if env.object_id!='bottle':raise ValueError('optional bottle relay provider only')
        self.result['bottle_goal_contract']=CONTRACT
        self._lookahead_audit=None

    def _ik(self,target,allow_fingers=False,**kwargs):
        try:return super()._ik(target,allow_fingers,**kwargs)
        except PlanningFailure as exc:
            if self._lookahead_audit is not None:
                self._lookahead_audit.append({'arm':self.side,'target_m':np.asarray(target).tolist(),'code':exc.code,
                    'diagnostics':copy.deepcopy(self.result.get('planning_diagnostics'))})
            raise

    def _future_place(self,snapshot,grasp_q):
        self._lookahead_audit=[]
        try:return super()._future_place(snapshot,grasp_q)
        finally:
            self.result['future_native_search_audit']=self._lookahead_audit
            self._lookahead_audit=None

    def _collision(self,q,allow_fingers=False):
        if super()._collision(q,allow_fingers):return True
        if self.carried is not None:
            for obstacle in self._clearance_obstacles:
                for geom in self.env.bottle_geoms:
                    gap=float(mujoco.mj_geomDistance(self.env.model,self.data,geom,obstacle,.040,None))
                    if gap < .040:
                        self.last_collision={'geoms':[self.env.model.geom(g).name for g in (geom,obstacle)],
                            'separation_m':gap,'required_clearance_m':.040,
                            'source':'additional carried-object planning clearance; native acceptance unchanged'}
                        return True
        return False

    def _transport_height(self):
        from .scene import bounds
        obstacles=[{'object_id':ident,'top_z_m':max(bounds(self.env.model,self.env.data,g)[1][2] for g in row['geoms'])}
            for ident,row in self.env.registry.items() if ident!='bottle']
        top=max([self.env.placement_surface('table')['top_z'],*[o['top_z_m'] for o in obstacles]])
        return float(top+CONTRACT['transport_vertical_clearance_m']),obstacles

    def _handover(self):
        # Lift the actual free body above the neighbouring objects before
        # crossing their footprint. The height is derived from this scene;
        # it is not a fixture-specific waypoint or a direct pose assignment.
        self._phase('transfer');self._transport_mode='elevated';self._bypass_x=None
        self._arm(self.donor,True);self._sync();point,rotation=self._pose();self.rotation=rotation.copy()
        bypass=None;rejected=[]
        for proposed in bypass_origins(self.env,self.env.data):
            target=proposed['origin_m']+self.rotation@(self.local_point-self.carried[0])
            try:q=self._ik(target,True);self._path(q,True);bypass=(proposed,target,q);break
            except PlanningFailure as exc:
                if exc.code in ('cancelled','planning_timeout'):raise
                rejected.append({'around_object':proposed['around_object'],'side':proposed['side'],'code':exc.code,
                    'diagnostics':copy.deepcopy(self.result.get('planning_diagnostics'))})
        self.result['transport_bypass_search']={'rejected':rejected}
        if bypass is not None:
            proposed,target,q=bypass;self.ik_seeds[self.donor].insert(0,q)
            self.result.pop('planning_diagnostics',None)
            yield from self._move(target,True,position_tolerance=.004)
            self._transport_mode='lateral_bypass';self._bypass_x=float(self.env.data.xpos[self.env.bottle_id,0])
            self.result['native_transport_bypass']={'proposed_object_origin_m':proposed['origin_m'].tolist(),
                'actual_object_origin_m':self.env.data.xpos[self.env.bottle_id].tolist(),
                'around_object':proposed['around_object'],'side':proposed['side'],
                'source':'native contact-held object moved through checked actuator path; no object state overwrite'}
            yield from super()._handover();yield from self._park_donor();return
        safe_bottom,obstacles=self._transport_height()
        before=float(self.env._bottle_bounds()[0][2]);moves=[]
        for _ in range(3):
            bottom=float(self.env._bottle_bounds()[0][2]);delta=safe_bottom-bottom
            if delta<=.003:break
            self._arm(self.donor,True);self._sync();point,rotation=self._pose();self.rotation=rotation.copy()
            origin=self.env.data.xpos[self.env.bottle_id].copy();goal_origin=origin+[0,0,delta]
            approach=float(np.arctan2(rotation[1,2],rotation[0,2]));trials=[];chosen=None
            for yaw in dict.fromkeys([0.,-approach,np.pi/4,-np.pi/4,np.pi/2,-np.pi/2]):
                c,s=np.cos(yaw),np.sin(yaw);turn=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                self.rotation=turn@rotation
                target=goal_origin+self.rotation@(self.local_point-self.carried[0])
                try:
                    q=self._ik(target,True);self._path(q,True);chosen=(target,self.rotation.copy(),q,yaw);break
                except PlanningFailure as exc:
                    if exc.code in ('cancelled','planning_timeout'):raise
                    trials.append({'yaw_offset_rad':yaw,'code':exc.code,'diagnostics':copy.deepcopy(self.result.get('planning_diagnostics'))})
            self.result['transport_lift_search']={'rejected':trials,'goal_object_origin_m':goal_origin.tolist()}
            if chosen is None:raise PlanningFailure('transport_clearance_unreachable','Scene-derived actual bottle clearance has no checked donor pose/path')
            target,self.rotation,q,yaw=chosen;self.ik_seeds[self.donor].insert(0,q)
            self.result.pop('planning_diagnostics',None)
            yield from self._move(target,True,position_tolerance=.004)
            moves.append({'requested_lift_m':delta,'actual_bottom_z_m':float(self.env._bottle_bounds()[0][2]),
                'yaw_offset_rad':yaw,'rejected':trials})
        after=float(self.env._bottle_bounds()[0][2])
        self.result['native_transport_clearance']={'initial_bottom_z_m':before,'actual_bottom_z_m':after,
            'required_bottom_z_m':safe_bottom,'obstacles':obstacles,'moves':moves,
            'source':'actual contact-held object and registered obstacle bounds; no live object pose edit'}
        if after<safe_bottom-.003:raise PlanningFailure('transport_clearance_not_reached','Actual held bottle did not reach scene-derived clearance')
        yield from super()._handover();yield from self._park_donor()

    def _park_donor(self):
        """Withdraw the empty donor to the recorded initial joint goal.

        The reference is a kinematic goal, not a live-state assignment. Every
        path edge and actual command still uses the native controller/gates.
        """
        self._phase('retreat');self._arm(self.donor);self._sync()
        goal=np.asarray(self.env.reference[str(self.env.episode)][self.donor+'_joints'][0],dtype=float)
        limits=np.asarray([self.env.model.joint(f'openarm_{self.donor}_joint{i}').range for i in range(1,8)])
        if np.max(np.maximum(limits[:,0]-goal,goal-limits[:,1]))>.01:
            raise PlanningFailure('donor_parking_joint_limits','Recorded initial pose lies outside this candidate joint contract')
        goal=np.clip(goal,limits[:,0],limits[:,1]);path=self._path(goal)
        before=self.env.tick
        for start,end in zip(path[:-1],path[1:]):
            start=np.clip(start,limits[:,0],limits[:,1])
            count=max(24,int(np.ceil(np.max(abs(end-start))/(.018*self.speed_scale))))
            for i in range(1,count+1):
                self._check();u=i/count;blend=u*u*(3-2*u)
                action=self.base_action.copy();action[self.start:self.start+7]=start+(end-start)*blend
                yield action,{'phase':'retreat','donor_parking':True,'source':'checked native joint-space path to recorded initial goal'}
                if not self.env.evaluation['held'][self.receiver]:
                    raise PlanningFailure('receiver_grasp_lost','Receiver lost native hold while empty donor withdrew')
            self.base_action[self.start:self.start+7]=end
        for _ in range(30):
            yield self.base_action.copy(),{'phase':'retreat','donor_parking_settle':True}
            if (not self.env.evaluation['held'][self.receiver] or self.env.evaluation['touching'][self.donor]):
                raise PlanningFailure('handover_not_confirmed','Actual independent receiver hold failed during donor parking')
        error=float(np.max(abs(self.env.data.qpos[self.qids]-goal)))
        self.result['donor_parking']={'arm':self.donor,'source':'recorded initial joints as checked motion goal only',
            'actual_joint_error_rad':error,'native_ticks':self.env.tick-before,
            'receiver_held':dict(self.env.evaluation['held']),'no_live_joint_or_object_pose_assignment':True}
        if error>.04:raise PlanningFailure('donor_parking_tracking','Empty donor did not reach its checked parking joint goal')

    def _receiver(self,snapshot,preferred=None):
        if self.phase=='transfer':
            # Endpoint hypotheses include extra margin for the measured
            # candidate's object rotation under load. Execution still uses
            # the original contacts and actual carried-body feedback.
            for obstacle in self._clearance_obstacles:
                for geom in self.env.bottle_geoms:
                    gap=float(mujoco.mj_geomDistance(self.env.model,self.data,geom,obstacle,.080,None))
                    if gap<.080:raise PlanningFailure('handover_object_clearance','Proposed held object lacks80mm obstacle clearance')
        self._arm(self.receiver);self.planning_qpos=snapshot.copy()
        for finger in self.env.fingers[self.receiver]:self.planning_qpos[finger.qposadr[0]]=max(finger.range)
        self._sync();seed=copy.deepcopy(self.ik_seeds[self.receiver]);rejected=[]
        values=list(receiver_candidates(self.env,self.data,self.donor,self.receiver,
            self.parameters.get('receiver_grasp_region')))
        # Search order changes only; no candidate, mesh, path or acceptance gate
        # is removed. Prefer axial room away from the actual donor grasp before
        # band height: a high donor cannot share the neck/cap with the receiver.
        # Low donor grasps still prefer the high bands. Full meshes and native
        # placement lookahead decide feasibility; fresh preferred grasp leads.
        desired=0. if self.receiver=='right' else np.pi/2
        def rank(v):
            distance=abs(float(np.arctan2(np.sin(v['yaw_rad']-desired),np.cos(v['yaw_rad']-desired))))
            return (-v['donor_axial_separation_m'],-v['grasp_m'][2],abs(v['pitch_rad']),distance)
        values.sort(key=rank)
        if preferred is not None:
            values.sort(key=lambda v:(v['geom']!=preferred['geom'],abs(v['yaw_rad']-preferred['yaw_rad']),
                abs(v['pitch_rad']-preferred['pitch_rad'])))
        for index,value in enumerate(values):
            self._check();self.rotation=value['rotation'];self.ik_seeds[self.receiver]=copy.deepcopy(seed)
            try:
                grasp=self._ik(value['grasp_m'],True);self.ik_seeds[self.receiver].insert(0,grasp)
                pre=self._ik(value['pregrasp_m']);self._path(pre)
                if not self._edge(pre,grasp,True):raise PlanningFailure('approach_collision','Receiver approach collides')
                future=self._future_place(self.planning_qpos,grasp)
                if future:self.result['future_place']=future
                return value,pre,grasp,rejected
            except PlanningFailure as exc:
                if exc.code in ('cancelled','planning_timeout'):raise
                rejected.append({'index':index,'geom':value['geom'],'yaw_rad':value['yaw_rad'],'pitch_rad':value['pitch_rad'],
                    'code':exc.code,'diagnostics':copy.deepcopy(self.result.get('planning_diagnostics')),
                    'lookahead_native_search':copy.deepcopy(self.result.get('future_native_search_audit'))})
        self.result['receiver_search']={'rejected':rejected,'candidates':len(values)}
        raise PlanningFailure('handover_unreachable','No connected native receiver hypothesis')

    def _pair(self):
        self._arm(self.donor,True);rotation=self.env.data.xmat[self.body].reshape(3,3).copy()
        before=(self.env.data.qpos.copy(),self.env.data.qvel.copy(),self.env.data.ctrl.copy(),self.env.data.time)
        failures=[];table=self.env.placement_surface('table')['top_z']
        safe_bottom,_=self._transport_height()
        origin_offset=float(self.env.data.xpos[self.env.bottle_id,2]-self.env._bottle_bounds()[0][2])
        needed=safe_bottom+origin_offset-table
        heights=sorted((.210,.230,.190,.170,.250),key=lambda h:(h<needed,abs(h-max(.210,needed))))
        xs=[p['origin_m'][0] for p in bypass_origins(self.env,self.env.data)]+[.48,.50,.46,.52,.44]
        if getattr(self,'_transport_mode',None)=='lateral_bypass':
            heights=[.170,.190,.210,.230,.250]
            xs=[self._bypass_x,*xs]
        approach=float(np.arctan2(rotation[1,2],rotation[0,2]))
        y_choices=(.06,.02,-.02,-.06) if self.donor=='left' else (-.06,-.02,.02,.06)
        try:
            for offset in dict.fromkeys([-approach,np.pi/4,-np.pi/4,0.,np.pi/2,-np.pi/2]):
                c,s=np.cos(offset),np.sin(offset);turn=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                for height in heights:
                    for y in y_choices:
                        for x in dict.fromkeys(float(x) for x in xs):
                            value=self._try_pair(x,y,table+height,turn@rotation,failures)
                            if value is not None:return value
            raise PlanningFailure('handover_unreachable','Native bottle donor/receiver corridor search exhausted')
        finally:
            self.planning_qpos=None
            for old,new in zip(before[:3],(self.env.data.qpos,self.env.data.qvel,self.env.data.ctrl)):
                np.testing.assert_array_equal(old,new)
            assert before[3]==self.env.data.time


class BottlePlacementPlanner(ObjectPlanner):
    """Native object alignment after a completed loaded lowering segment.

    A TCP correction can run out of IK reach although the actual held bottle
    is close enough for the existing small-step object servo. This only lets
    that servo attempt the original target; release/support/stability gates
    and every collision check remain the original ones.
    """
    def _ik(self,target,allow_fingers=False,*,position_tolerance=.0015):
        if (not (self.result.get('lowering_capture') or getattr(self,'_pre_release_servo',False))
            or self.phase not in ('place','release_left') or position_tolerance>.0002):
            return super()._ik(target,allow_fingers,position_tolerance=position_tolerance)
        # The final task constrains the bottle's upright axis, not its yaw or
        # the full hand attitude. Each small alignment goal uses position IK,
        # with the original upright axis independently required in scratch.
        # The existing object servo limits each actuator increment to0.008rad
        # and rechecks that actual increment for collision before dispatch.
        saved=self.rotation
        try:
            self.rotation=None
            q=super()._ik(target,allow_fingers,position_tolerance=position_tolerance)
            self._at(q)
            upright=float(self.data.xmat[self.env.bottle_id].reshape(3,3)[2,2])
            if upright<.98:raise PlanningFailure('alignment_attitude_rejected','Position-only candidate tilts the bottle beyond the original upright threshold')
            self.result.setdefault('object_alignment_position_IK',[]).append({
                'target_m':np.asarray(target).tolist(),'predicted_upright_cosine':upright,
                'source':'native position/collision IK; actual object release and stability still required'})
            self.result.pop('planning_diagnostics',None)
            return q
        finally:self.rotation=saved

    def _grip(self,side,value,*,samples=45,ramp_samples=24):
        if self.phase=='release_left' and self.placement_surface=='table':
            env=self.env;precision=min(.003,env.placement_target['tolerance_m']*.6)
            # Reduce free-fall before opening, using native motor commands.
            # The target XY and support/release/upright/stability score stay
            # unchanged; this is an extra servo stage, not a state assignment.
            desired=np.r_[self.placement_xy,env.placement_surface('table')['top_z']+.002]
            did=env.registry['bottle']['dofadr'];stable=0;history=[]
            for index in range(121):
                self._check()
                actual=env.data.xpos[env.bottle_id].copy();bottom=float(env._bottle_bounds()[0][2])
                velocity=env.data.qvel[did:did+6].copy();error=desired-np.r_[actual[:2],bottom]
                pose_ok=np.linalg.norm(error[:2])<=precision and abs(error[2])<=.006
                speed_ok=np.linalg.norm(velocity[:3])<=.01 and np.linalg.norm(velocity[3:])<=.15
                stable=stable+1 if pose_ok and speed_ok else 0
                history.append({'iteration':index,'object_xy_error_m':float(np.linalg.norm(error[:2])),
                    'bottom_error_m':float(error[2]),'linear_speed_m_s':float(np.linalg.norm(velocity[:3])),
                    'angular_speed_rad_s':float(np.linalg.norm(velocity[3:])),'stable_endpoints':stable})
                self.result['pre_release_stability']={'required_endpoints':6,'linear_speed_limit_m_s':.01,
                    'angular_speed_limit_rad_s':.15,'samples':history,'original_target_and_final_gates_unchanged':True}
                if stable>=6:break
                if index==120:raise PlanningFailure('pre_release_not_stable','Original release pose did not settle at low speed; gripper stays closed')
                if not pose_ok:
                    self._arm(side,True);self._sync();point,_=self._pose()
                    adjustment=np.clip((error-.10*velocity[:3])*.6,-.004,.004)
                    self._pre_release_servo=True
                    try:goal=self._ik(point+adjustment,True,position_tolerance=.0002)
                    finally:self._pre_release_servo=False
                    measured=env.data.qpos[self.qids].copy()
                    limits=np.asarray([env.model.joint(f'openarm_{side}_joint{i}').range for i in range(1,8)])
                    if np.max(np.maximum(limits[:,0]-measured,measured-limits[:,1]))>.01:
                        raise PlanningFailure('joint_limit_violation','Pre-release native joint state exceeds the original joint guard')
                    next_q=np.clip(self.base_action[self.start:self.start+7]+np.clip((goal-measured)*.5,-.008,.008),limits[:,0],limits[:,1])
                    if not self._edge(measured,next_q,True):raise PlanningFailure('goal_collision','Pre-release feedback increment collides')
                    self.base_action[self.start:self.start+7]=next_q
                yield self.base_action.copy(),{'phase':self.phase,'pre_release_alignment':index,'scene_source':'simulator_ground_truth'}
                yield from self._check_hold(self.base_action)
        yield from super()._grip(side,value,samples=samples,ramp_samples=ramp_samples)

    def _move(self,target,allow_fingers=False,*,position_tolerance=.012):
        lower=(self.task=='place_bottle' and self.env.object_id=='bottle'
            and self.placement_surface=='table' and self.phase=='place'
            and position_tolerance<=.003 and self.carried is not None)
        if not lower:
            yield from super()._move(target,allow_fingers,position_tolerance=position_tolerance)
            return
        self.result.pop('tracking_feedback',None)
        try:
            yield from super()._move(target,allow_fingers,position_tolerance=position_tolerance)
        except PlanningFailure as exc:
            # Tracking feedback only appears after a full checked path and its
            # native settling period. Never turn initial IK/path/collision
            # failure or a lost grasp into permission to continue.
            feedback=self.result.get('tracking_feedback')
            if exc.code not in ('ik_unreachable','tracking_error') or feedback is None:raise
            env=self.env;surface=env.placement_surface('table')
            xy=np.asarray(self.placement_xy)-env.data.xpos[env.bottle_id,:2]
            bottom=float(env._bottle_bounds()[0][2]);desired=float(surface['top_z']+.020)
            safe=(env.evaluation['held'][self.side] and not env.evaluation.get('invalid')
                and env.evaluation.get('non_target_object_guard',{}).get('passed',False)
                and np.linalg.norm(xy)<=.040 and abs(bottom-desired)<=.040)
            if not safe or self._collision(env.data.qpos[self.qids].copy(),True):raise
            self.result['lowering_capture']={'source':'actual native bottle pose and sustained contact',
                'prior_TCP_failure':exc.code,'TCP_feedback':copy.deepcopy(feedback),
                'object_xy_error_m':float(np.linalg.norm(xy)),'object_bottom_error_m':bottom-desired,
                'capture_radius_m':.040,'next':'original collision-checked small-step object alignment',
                'original_release_and_placement_gates_unchanged':True,'transfer_ready':False}
            self.result.pop('planning_diagnostics',None)
