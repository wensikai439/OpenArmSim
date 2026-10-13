"""Optional native object-feedback descent for an anchored table goal.

Never writes live qpos or model parameters. The old placement/demo routes stay
available; scratch IK proposals still require native edges and actual contact.
"""
import mujoco
import numpy as np

from .bottle_routing import BottlePlacementPlanner
from .tamp import PlanningFailure


class CartesianBottlePlacement(BottlePlacementPlanner):
    def _ik(self,target,allow_fingers=False,*,position_tolerance=.0015):
        if not (getattr(self,'_pre_release_servo',False) and self.carried is not None
            and self.phase in ('place','release_left') and position_tolerance<=.0002):
            return super()._ik(target,allow_fingers,position_tolerance=position_tolerance)
        # Five task constraints: TCP position and carried-object upright axis.
        # Yaw is free, but tilt is optimized rather than accepted unchecked.
        self._sync();initial=self.data.qpos[self.qids].copy()
        bounds=np.asarray([self.env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1,8)])
        q=np.clip(initial,bounds[:,0],bounds[:,1]);jp=np.zeros((3,self.env.model.nv));jr=jp.copy();best=None
        self.stats['ik_attempts']+=1
        for _ in range(130):
            self._check();self._at(q);point,_=self._pose()
            axis=self.data.xmat[self.env.bottle_id].reshape(3,3)[:,2];delta=np.asarray(target)-point
            angle=float(np.arccos(np.clip(axis[2],-1,1)));cost=float(np.linalg.norm(delta)+.15*angle)
            if best is None or cost<best['cost']:best={'cost':cost,'position_error_m':float(np.linalg.norm(delta)),'axis_angle_rad':angle}
            if np.linalg.norm(delta)<position_tolerance and angle<.018:
                if self._collision(q,allow_fingers):
                    self.result['planning_diagnostics']={'arm':self.side,'phase':self.phase,'best_ik':best,'blocked_pairs':[self.last_collision]}
                    raise PlanningFailure('goal_collision','Position/upright-axis solution collides under original native geometry')
                self.result['object_constraint_IK']={'source':'native Jacobian position plus carried-body upright axis',
                    'position_tolerance_m':position_tolerance,'axis_tolerance_rad':.018,'yaw_free':True,
                    'joint_limits_and_original_collision_checks':True,'best':best}
                self.result.pop('planning_diagnostics',None);return q
            mujoco.mj_jac(self.env.model,self.data,jp,jr,point,self.body)
            axis_jac=np.array([[0.,axis[2],-axis[1]],[-axis[2],0.,axis[0]]])@jr[:,self.dids]
            jac=np.vstack((jp[:,self.dids],.15*axis_jac));error=np.r_[delta,-.15*axis[:2]]
            # Re-solve with saturated joints removed. Solving unconstrained
            # then clipping wastes the residual on a joint that cannot move,
            # even when the remaining degrees of freedom admit a solution.
            active=np.ones(7,dtype=bool);step=np.zeros(7)
            for _ in range(8):
                step[:]=0.;free=jac[:,active]
                step[active]=free.T@np.linalg.solve(free@free.T+.0001*np.eye(5),error)
                blocked=active & (((q<=bounds[:,0]+1e-7)&(step<0))
                    |((q>=bounds[:,1]-1e-7)&(step>0)))
                if not np.any(blocked):break
                active[blocked]=False
            q=np.clip(q+step*min(1.,.16/max(np.linalg.norm(step),1e-9)),bounds[:,0],bounds[:,1])
        self.result['planning_diagnostics']={'arm':self.side,'phase':self.phase,'best_ik':best,'blocked_pairs':[]}
        raise PlanningFailure('ik_unreachable','No native position/upright-axis solution within original joint limits')

    def _move(self,target,allow_fingers=False,*,position_tolerance=.012):
        lower=(self.env.object_id=='bottle' and self.placement_surface=='table'
            and self.phase=='place' and allow_fingers and self.carried is not None
            and position_tolerance<=.004)
        if not lower:
            yield from super()._move(target,allow_fingers,position_tolerance=position_tolerance)
            return
        env=self.env;side=self.side;precision=min(.003,env.placement_target['tolerance_m']*.6)
        desired=np.r_[self.placement_xy,env.placement_surface('table')['top_z']+.020]
        did=env.registry['bottle']['dofadr'];stable=0;samples=[]
        self.result.pop('tracking_feedback',None)
        for index in range(361):
            self._check();actual=env.data.xpos[env.bottle_id].copy()
            bottom=float(env._bottle_bounds()[0][2]);velocity=env.data.qvel[did:did+6].copy()
            error=desired-np.r_[actual[:2],bottom]
            upright=float(env.data.xmat[env.bottle_id].reshape(3,3)[2,2])
            pose_ok=np.linalg.norm(error[:2])<=precision and abs(error[2])<=.006 and upright>=.98
            speed_ok=np.linalg.norm(velocity[:3])<=.01 and np.linalg.norm(velocity[3:])<=.15
            stable=stable+1 if pose_ok and speed_ok else 0
            samples.append({'iteration':index,'sim_time':float(env.data.time),'object_xy_m':actual[:2].tolist(),
                'xy_error_m':float(np.linalg.norm(error[:2])),'bottom_error_m':float(error[2]),
                'linear_speed_m_s':float(np.linalg.norm(velocity[:3])),
                'upright_cosine':upright,'stable_endpoints':stable})
            self.result['lowering_object_servo']={'source':'current native object/encoder feedback; no state assignment',
                'target_xy_m':list(self.placement_xy),'target_bottom_z_m':float(desired[2]),
                'maximum_joint_increment_rad':.008,'required_stable_endpoints':6,'samples':samples,
                'original_collision_hold_goal_release_support_stability_gates_unchanged':True}
            if not env.evaluation['held'][side]:raise PlanningFailure('grasp_lost','Object-feedback descent requires actual independent hold')
            if stable>=6:return
            if index==360:raise PlanningFailure('release_pose_not_reached','Original object lowering goal did not settle; gripper stays closed')
            if not pose_ok:
                self._arm(side,True);self._sync();point,_=self._pose()
                adjustment=np.clip((error-.10*velocity[:3])*.6,-.004,.004)
                # The task constrains the object's upright axis and original
                # position. Reuse the guarded position-IK servo, not a guessed
                # wrist attitude or a long compensating joint-space segment.
                self._pre_release_servo=True
                try:goal=self._ik(point+adjustment,True,position_tolerance=.0002)
                finally:self._pre_release_servo=False
                measured=env.data.qpos[self.qids].copy()
                bounds=np.asarray([env.model.joint(f'openarm_{side}_joint{i}').range for i in range(1,8)])
                if np.max(np.maximum(bounds[:,0]-measured,measured-bounds[:,1]))>.01:
                    raise PlanningFailure('joint_limit_violation','Native joint state exceeds the original joint guard')
                next_q=np.clip(self.base_action[self.start:self.start+7]+np.clip((goal-measured)*.5,-.008,.008),bounds[:,0],bounds[:,1])
                if not self._edge(measured,next_q,True):raise PlanningFailure('goal_collision','Object-feedback descent increment collides')
                self.base_action[self.start:self.start+7]=next_q
            yield self.base_action.copy(),{'phase':self.phase,'cartesian_object_lowering':index,'scene_source':'simulator_ground_truth'}
            yield from self._check_hold(self.base_action)
