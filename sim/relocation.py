"""Parameterized native object relocation, with no LLM wait between subskills.

Every subskill re-grounds from the current physical state. A grasp/lift, directed
handover or TCP receipt is never promoted to an object-placement receipt. Native
physics, individual tools, collision checks and stability thresholds are intact.
"""
import copy
import time

import numpy as np

from .tamp import TampPlanner,PlanningFailure
from .curobo import CuRoboPlanner,metadata,call,joint_state,scene


class RelocationPlanner:
    CONTRACT={**TampPlanner.CONTRACT,'planner':'parameterized native skill coordinator',
        'scope':'declared upright bottle relocation on known table/basket; oracle geometry, not open-object perception',
        'composition':'re-ground and verify each native subskill; no model round trip while carrying',
        'timing':'native subskill computation follows the caller planning_timing contract; measured_delay integrates latency and rejects stale proposals; paused is an explicit diagnostic baseline'}
    RECOVERABLE={'placement_unreachable','grasp_pose_unreachable','ik_unreachable'}

    def __init__(self,env,task,*,parameters=None,cancel=lambda:False,feedback=lambda value:None):
        p=parameters or {}
        allowed={'object_id','arm','via_arm','receptacle_id','target_xy_m','offset_xy_m',
                 'transport_planner','transport_delta_m','speed_scale','allow_handover_recovery',
                 'goal_anchor_observation_id','position_tolerance_m'}
        if set(p)-allowed:raise ValueError('unknown relocation parameters')
        if task!='relocate_object' or p.get('object_id')!='bottle':raise ValueError('relocation supports only the declared bottle')
        if env.robot_version!='v1' or env._done or env.evaluation['invalid'] or env._clock_mode not in (None,'periodic'):
            raise PlanningFailure('episode_finished','搬运要求有效的当前 v1 场景，不能恢复已失效物理回合')
        arm=p.get('arm');via=p.get('via_arm')
        if arm not in ('left','right') or via not in (None,'left','right'):raise ValueError('select valid relocation arms')
        if via==arm:raise PlanningFailure('distinct_arms_required','交接需要不同机械臂')
        held=[s for s,v in env.evaluation['held'].items() if v]
        if len(held)>1:raise PlanningFailure('dual_hold_requires_handover','双手持物需先完成单独交接')
        if held and held!=[arm]:raise PlanningFailure('single_hold_required','所选手与当前独立持物手不一致')
        transport=p.get('transport_planner','tamp')
        if transport not in ('tamp','curobo'):raise ValueError('unknown transport planner')
        delta=p.get('transport_delta_m')
        if delta is not None:
            delta=np.asarray(delta,dtype=float)
            if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>.15:raise ValueError('transport delta must be at most 15 cm')
        if transport=='curobo' and delta is None:raise ValueError('cuRobo transport needs an explicit transport_delta_m')
        if transport=='curobo' and not metadata().get('ready'):
            raise PlanningFailure('curobo_unavailable','cuRobo 未就绪；搬运未开始，不替换后端')
        speed=p.get('speed_scale',1.)
        if type(speed) not in (int,float) or not np.isfinite(speed) or not .05<=speed<=1:raise ValueError('invalid relocation speed')
        if type(p.get('allow_handover_recovery',False)) is not bool:raise ValueError('recovery must be an explicit boolean')
        surface=p.get('receptacle_id');absolute=p.get('target_xy_m');offset=p.get('offset_xy_m')
        if absolute is not None and offset is not None:raise ValueError('choose one object target')
        if surface=='basket' and offset is not None:raise ValueError('basket goal cannot use bottle-relative offset')
        if surface=='table' and absolute is None and offset is None:raise ValueError('table relocation requires a target')
        xy=np.asarray(absolute if absolute is not None else env.data.xpos[env.bottle_id,:2]+offset
                      if offset is not None else env.data.xpos[env.box_id,:2],dtype=float)
        # Validate the target BEFORE opening/grasping. The target is frozen;
        # handover or recovery may change the route, never the user's goal.
        env.set_placement_target(surface,xy,p.get('position_tolerance_m'))
        self.env,self.task,self.parameters=env,task,p
        self.cancel,self.feedback=cancel,feedback
        self.arm,self.via,self.delta,self.transport=arm,via,delta,transport
        self.xy,self.surface,self.speed=xy.copy(),surface,float(speed)
        self.initial_held=bool(held);self.deadline=time.monotonic()+90
        self.phase='preflight';self.stats={'ik_attempts':0,'collision_checks':0,'rrt_nodes':0}
        self.result={**self.CONTRACT,'task':task,'status':'planning','primitive_success':False,
            'surface':surface,'target_xy_m':xy.tolist(),'substeps':[],'recovery_attempted':False,
            'requested_transport_planner':transport,'parameters':copy.deepcopy(p)}
        env.task_termination='skill'

    def _check(self):
        if self.cancel():raise PlanningFailure('cancelled','搬运已停止，后续子动作禁止')
        if time.monotonic()>self.deadline:raise PlanningFailure('planning_timeout','原子搬运达到规划时间预算')
        if self.env._done or self.env.evaluation['invalid']:raise PlanningFailure('episode_finished','物理回合失效，后续子动作禁止')

    def _skill(self,name,parameters,planner_type=TampPlanner):
        self._check();index=len(self.result['substeps']);start=self.env.tick
        row={'skill':name,'parameters':copy.deepcopy(parameters),'status':'planning','start_tick':start,
             'backend':'curobo' if planner_type is CuRoboPlanner else 'tamp'}
        self.result['substeps'].append(row)
        def feedback(value):
            self.phase=value.get('phase',name)
            self.feedback({**value,'subskill':name,'subskill_index':index+1,
                           'composite':'relocate','target_xy_m':self.xy.tolist()})
        child=None
        try:
            tasks={'pick':'pick_bottle','handover':'handover_bottle','place':'place_bottle'}
            child=planner_type(self.env,tasks.get(name,name),parameters=parameters,cancel=self.cancel,feedback=feedback)
            child.deadline=min(child.deadline,self.deadline)
            for action,diagnostic in child.actions():
                self._check()
                yield action,{**diagnostic,'composite':'relocate','subskill':name,'subskill_index':index+1}
            if not child.result.get('primitive_success'):raise PlanningFailure('subskill_not_confirmed','子动作缺少原生成功回执')
            row.update(status='completed',end_tick=self.env.tick,feedback=copy.deepcopy(child.result))
            self.feedback({'composite_step_complete':name,'subskill_index':index+1,'subskill':name,
                           'phase':name+' 已验证','target_xy_m':self.xy.tolist()})
        except GeneratorExit:
            row.update(status='aborted',end_tick=self.env.tick,failure_code='coordinator_interrupted',
                       feedback=copy.deepcopy(child.result) if child else None)
            raise
        except Exception as exc:
            row.update(status='failed',end_tick=self.env.tick,
                       failure_code=exc.code if isinstance(exc,PlanningFailure) else 'execution_error',
                       message=str(exc),feedback=copy.deepcopy(child.result) if child else None)
            raise
        finally:
            if child:
                for key,value in child.stats.items():self.stats[key]=self.stats.get(key,0)+value

    def _place(self):
        return self._skill('place',{'arm':self.arm,'object_id':'bottle','receptacle_id':self.surface,
            'target_xy_m':self.xy.tolist(),'position_tolerance_m':self.parameters.get('position_tolerance_m'),
            'speed_scale':self.speed})

    def actions(self):
        if self.transport=='curobo':
            self._check()
            self.phase='准备GPU规划器'
            self.feedback({'phase':self.phase,'composite':'relocate','target_xy_m':self.xy.tolist()})
            self.result['backend_preparation']=call('/prepare',{'arm':self.via or self.arm,
                'joints':joint_state(self.env),'scene':scene(self.env,carrying=True),'carrying':True})
            if not self.result['backend_preparation'].get('prepared'):
                raise PlanningFailure('curobo_preparation_failed','GPU规划器准备未确认，未开始搬运')
            self._check()
        if not self.initial_held:yield from self._skill('pick',{'arm':self.arm,'object_id':'bottle'})
        if self.via:
            yield from self._skill('handover',{'object_id':'bottle','from_arm':self.arm,'to_arm':self.via})
            self.arm=self.via
        if self.delta is not None:
            yield from self._skill('move_relative',{'arm':self.arm,'delta_m':self.delta.tolist(),
                'frame':'world','speed_scale':self.speed},CuRoboPlanner if self.transport=='curobo' else TampPlanner)
        try:yield from self._place()
        except PlanningFailure as exc:
            if (not self.parameters.get('allow_handover_recovery') or exc.code not in self.RECOVERABLE
                    or self.env._done or self.env.evaluation['invalid']
                    or not self.env.evaluation['held'][self.arm]
                    or self.env.evaluation['held']['left' if self.arm=='right' else 'right']):raise
            self.result['recovery_attempted']=True
            receiver='left' if self.arm=='right' else 'right'
            yield from self._skill('handover',{'object_id':'bottle','from_arm':self.arm,'to_arm':receiver})
            self.arm=receiver
            yield from self._place()
        receipt=self.result['substeps'][-1]['feedback']
        self.result.update({key:copy.deepcopy(receipt[key]) for key in ('verification_source','placement',
            'position_error_m','actual_bottle_position_m','held_by')})
        self.result.update(status='completed',primitive_success=True,arm=self.arm,
                           continued_without_model_round_trip=True)
