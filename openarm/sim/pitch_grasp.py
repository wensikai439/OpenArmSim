"""Opt-in pitch ranking for original-goal cylindrical grasps.

No candidate pose, endpoint tolerance, path/collision check or task score is
changed. All original candidates remain. Realized grasp geometry still needs
native verification; this preference is a development hypothesis.
"""
import copy
import mujoco
import numpy as np
from .grasp import candidates,cylinder_bands
from .object_handover import ObjectHandoverPlanner
from .oriented_grasp import OrientedPlanner
from .margin_grasp import MarginBottlePick
from .task_grasp import TaskBottlePick
from .bottle_routing import CONTRACT
from .tamp import PlanningFailure


class PitchSampler(MarginBottlePick):
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
                values.sort(key=lambda v:(float(np.linalg.norm(v['grasp_m']-center))/max(v['radius_m'],1e-6),abs(v['pitch_rad'])))
                self.result['grasp_order']={'source':'current native center of mass and declared cylinder radii',
                    'ranking':'geometric lever-arm/contact-width heuristic for a single-arm placement goal',
                    'all_original_candidates_retained':True,'scope':'proposal order only; native grasp/hold/transport/place still required'}
            if self.parameters.get('next_placement') and not self.parameters.get('next_handover'):
                values.sort(key=lambda v: abs(v['pitch_rad']-self.preferred_pitch_rad))
                self.result['grasp_order']['ranking'] = 'declared pitch first; original order within equal pitch'
                self.result['grasp_order']['preferred_pitch_rad'] = self.preferred_pitch_rad
                self.result['grasp_order']['all_original_candidates_retained'] = True
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


class PitchTaskPick(TaskBottlePick,PitchSampler):
    def __init__(self,*args,preferred_pitch_rad=-np.pi/12,**kwargs):
        if not np.isfinite(preferred_pitch_rad) or not -np.pi/6<=preferred_pitch_rad<=np.pi/12:
            raise ValueError('finite existing-sampler pitch preference required')
        self.preferred_pitch_rad=float(preferred_pitch_rad)
        super().__init__(*args,**kwargs)
        self.result['pitch_proposal_preference']={'rad':self.preferred_pitch_rad,
            'scope':'ranking only, all original candidates and goals retained',
            'hardware_access':False,'transfer_ready':False}
