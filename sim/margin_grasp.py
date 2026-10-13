"""Experimental bounded pose refinement; existing physical gates unchanged.

Refine only a native-solvable empty-hand grasp at insufficient joint slack.
Pose acceptance uses the ORIGINAL1.5mm/18mrad limits and full native collision.
An optimizer exit flag is never a grasp/hold/path or physical reachability proof.
"""
import copy
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

from .adaptive_grasp import RefinedBottlePick,FastBottlePick
from .bottle_routing import CONTRACT
from .tamp import PlanningFailure


def refine(planner, target, rotation, seed, allow_fingers, position_tolerance):
    """Search strictly inside the original mandatory joint margin, scratch only."""
    env=planner.env
    limits=np.asarray([env.model.joint(f'openarm_{planner.side}_joint{i}').range for i in range(1,8)])
    margin=CONTRACT['grasp_joint_slack_rad']+1e-5
    low,high=limits[:,0]+margin,limits[:,1]-margin
    original=getattr(planner,'_kinematics_only',False);planner._kinematics_only=True
    target=np.asarray(target).copy();rotation=np.asarray(rotation).copy();calls=0;started=time.monotonic()
    def residual(q):
        nonlocal calls
        calls+=1;planner._check();planner._at(q);point,current=planner._pose()
        quat=np.empty(4);angular=np.empty(3)
        mujoco.mju_mat2Quat(quat,(rotation@current.T).ravel());mujoco.mju_quat2Vel(angular,quat,1.)
        return np.r_[point-target,angular]
    def objective(q):return float(np.sum((residual(q)[:3]/position_tolerance)**2))
    def orientation(q):return 1-float(np.sum((residual(q)[3:]/(.018-1e-6))**2))
    record={'source':'scratch bounded pose proposal, not executed grasp or physical reachability',
        'method':'SciPy SLSQP; position objective and original angular ball constraint',
        'arm':planner.side,'target_position_m':target.tolist(),'target_rotation':rotation.tolist(),
        'required_joint_margin_rad':CONTRACT['grasp_joint_slack_rad'],
        'solver_margin_rad':margin,'position_tolerance_m':position_tolerance,'rotation_tolerance_rad':.018,
        'accepted':False}
    try:
        solve=minimize(objective,np.clip(seed,low+1e-8,high-1e-8),method='SLSQP',
            bounds=list(zip(low,high)),constraints=[{'type':'ineq','fun':orientation}],
            options={'maxiter':160,'ftol':1e-10})
        q=np.asarray(solve.x)
        valid_q=q.shape==(7,) and np.isfinite(q).all()
        error=residual(q) if valid_q else np.full(6,np.nan)
        slack=np.minimum(q-limits[:,0],limits[:,1]-q) if valid_q else np.full(7,np.nan)
        finite=valid_q and np.isfinite(error).all()
        bounded=finite and np.all(q>=low) and np.all(q<=high)
        posed=finite and np.linalg.norm(error[:3])<position_tolerance and np.linalg.norm(error[3:])<.018
        # The full original collision/12mm clearance call, never a solver's
        # collision flag or contacts left over from a previous kinematic query.
        collision=planner._collision(q,allow_fingers) if bounded and posed else None
        accepted=bool(bounded and posed and collision is False)
        record.update(accepted=accepted,solver_status=int(solve.status),solver_message=str(solve.message),
            solver_reported_success=bool(solve.success),position_error_m=float(np.linalg.norm(error[:3])) if finite else None,
            rotation_error_rad=float(np.linalg.norm(error[3:])) if finite else None,minimum_joint_slack_rad=float(slack.min()) if finite else None,
            limiting_joint=int(np.argmin(slack))+1 if finite else None,q_rad=q.tolist() if finite else None,native_collision_checked=collision is not None,
            collision=collision,proposal_only=True)
        return (q.copy() if accepted else None),record
    except PlanningFailure as exc:
        record.update(interrupted_code=exc.code,status='interrupted',accepted=False)
        raise
    finally:
        planner._kinematics_only=original
        record.update(kinematic_queries=calls,wall_seconds=time.monotonic()-started)
        planner.result.setdefault('private_margin_refinements',[]).append(copy.deepcopy(record))


class MarginBottlePick(RefinedBottlePick):
    def _choose_pick_pose(self,ignored):
        if not self.parameters.get('next_handover'):
            # Preserve the successful single-arm sampler and seed ordering.
            self.result['margin_proposal_scope']='existing single-arm sampler retained; no relay refinements'
            return FastBottlePick._choose_pick_pose(self,ignored)
        self.result['margin_proposal_scope']='directed relay goal; dimension-derived refinement portfolio'
        return super()._choose_pick_pose(ignored)

    """Original+dimension-derived proposals; constrained IK only at joint stops."""
    def _ik(self,target,allow_fingers=False,*,position_tolerance=.0015):
        q=super()._ik(target,allow_fingers,position_tolerance=position_tolerance)
        if not self.parameters.get('next_handover') or not allow_fingers or self.carried is not None:return q
        limits=np.asarray([self.env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1,8)])
        if float(np.minimum(q-limits[:,0],limits[:,1]-q).min())>=CONTRACT['grasp_joint_slack_rad']:return q
        proposal=None;record=None
        try:
            proposal,record=refine(self,target,self.rotation,q,allow_fingers,position_tolerance)
        finally:
            self._at(q if proposal is None else proposal)
        if proposal is None:return q
        self.result['private_search_progress']['last_accepted_ik']={'arm':self.side,
            'q_rad':proposal.tolist(),'minimum_joint_slack_rad':record['minimum_joint_slack_rad'],
            'limiting_joint':f'openarm_{self.side}_joint{record["limiting_joint"]}',
            'source':'bounded scratch pose refinement; original full collision and pose gates; not executed motion'}
        return proposal
