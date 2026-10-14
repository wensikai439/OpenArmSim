"""Verified child-stage bindings for native composite skills.

Installed in the isolated candidate harness first. Existing main-process
validators and workspace remain unchanged until explicitly integrated.
"""
import copy
import json

from .workspace import Workspace as BaseWorkspace,validate_plan as base_validate
from .agent import Harness as BaseHarness
from sim.relocation import RelocationPlanner

CHILDREN={'grasp_confirmed':'pick','handover_confirmed':'handover','motion_reached':'move_relative'}
SOURCES={'pick':'native_two_finger_contact_and_lift','handover':'native_receiver_contact_then_donor_release',
         'move_relative':'robot_forward_kinematics','move_to_pose':'robot_forward_kinematics'}


def preserved_anchor(initial,before,after,result):
    """Move only the task identity, never the original object target."""
    if result.get('mode')!='preserve' or not result.get('physical_state_preserved'):return initial
    if initial is None:return None
    if (initial['observation_id'].split(':')[0]!=before['observation_id'].split(':')[0]
            or result.get('observation_id')!=after['observation_id']):
        raise ValueError('准备任务的身份回执不匹配，不能重绑定目标')
    for key in ('state','tcp'):
        if before.get('observation',{}).get(key)!=after.get('observation',{}).get(key):
            raise ValueError('保留任务改变了机械臂状态，不能重绑定目标')
    old=before.get('diagnostics',{}).get('manipulation_scene',{})
    new=after.get('diagnostics',{}).get('manipulation_scene',{})
    for key in ('object_id','position_m','held_by','upright'):
        if key not in old or key not in new or old[key]!=new[key]:
            raise ValueError('保留任务缺少相同物体状态的依据，不能重绑定目标')
    rebased=copy.deepcopy(initial)
    rebased['observation_id']=after['observation_id'].split(':')[0]+':'+initial['observation_id'].split(':')[1]
    rebased['goal_anchor_preservation']={'original_observation_id':initial['observation_id'],
        'prepared_observation_id':after['observation_id'],'physical_state_preserved':True,
        'original_object_position_retained':True,'source':'native_prepare_and_snapshot_equality'}
    return rebased


class CompositeHarness(BaseHarness):
    def _invoke(self,name,args,role,context=None):
        before=self.service.snapshot() if name=='prepare_simulation' else None
        result=super()._invoke(name,args,role,context)
        if name=='prepare_simulation' and context and context.get('type')=='llm':
            root=context.get('plan_turn_id',context.get('turn_id'))
            session=self.sessions.get(root,{})
            session['initial_snapshot']=preserved_anchor(session.get('initial_snapshot'),before,
                                                         self.service.snapshot(),result)
        return result


def validate_composite_plan(values):
    if not isinstance(values,dict):raise ValueError('计划参数需要对象')
    converted=copy.deepcopy(values);bindings={}
    steps=converted.get('steps')
    if not isinstance(steps,list):return base_validate(converted)
    for index,step in enumerate(steps):
        if not isinstance(step,dict):continue  # the normal validator rejects it
        if step.get('skill')=='relocate' and step.get('checkpoint') in CHILDREN:
            child=step.pop('subskill',CHILDREN[step['checkpoint']]);occurrence=step.pop('sub_occurrence',None)
            allowed=({'move_relative','move_to_pose'} if step['checkpoint']=='motion_reached' else
                     {CHILDREN[step['checkpoint']]})
            if not isinstance(child,str) or child not in allowed:raise ValueError('子动作与阶段检查条件不一致')
            if occurrence is not None and (type(occurrence) is not int or not 1<=occurrence<=8):
                raise ValueError('子动作次数必须为1到8的整数')
            bindings[index]=(child,occurrence);step['skill']=child
        elif 'subskill' in step or 'sub_occurrence' in step:
            raise ValueError('只有连续搬运的抓取、交接和运动阶段可绑定子动作')
    normalized=base_validate(converted)
    for index,(child,occurrence) in bindings.items():
        normalized[index].update(skill='relocate',subskill=child)
        if occurrence is not None:normalized[index]['sub_occurrence']=occurrence
    return normalized


def child_receipt(step,feedback):
    rows=[r for r in feedback.get('substeps',[]) if r.get('skill')==step['subskill']]
    selected=step.get('sub_occurrence')
    if selected is None:
        if len(rows)!=1:return None  # repeated/recovery actions require an explicit occurrence
        row=rows[0]
    else:
        if len(rows)<selected:return None
        row=rows[selected-1]
    f=row.get('feedback') or {};parameters=row.get('parameters') or {}
    if row.get('status')!='completed' or not f.get('primitive_success') or f.get('verification_source')!=SOURCES[step['subskill']]:return None
    if step['subskill']=='pick' and f.get('held_by')!=parameters.get('arm'):return None
    if step['subskill']=='handover' and f.get('held_by')!=parameters.get('to_arm'):return None
    return f


class CompositeWorkspace(BaseWorkspace):
    def bind(self,turn,execution_id,role,goal,tool,task,model,context=None):
        super().bind(turn,execution_id,role,goal,tool,task,model,context)
        if tool!='relocate':return
        with self.lock:
            plan=self.active_plan
            if not plan or plan['turn_id']!=turn:return
            records=[r for r in plan.get('execution_tools',[]) if r['skill']=='relocate']
            bindings=plan.setdefault('stage_bindings',{})
            # All stages of one composite share its outer execution, not the
            # next sibling action. Retries must declare their occurrence.
            for step in plan['steps']:
                if step.get('skill')!='relocate':continue
                index=step.get('occurrence',1)-1
                if len(records)>index:bindings[step['id']]=records[index]['id']
                else:bindings.pop(step['id'],None)
            self._write('UPDATE plans SET body=? WHERE turn_id=?',
                        (json.dumps(plan),turn))

    @staticmethod
    def _step_done(step,feedback,plan):
        if step.get('skill')=='relocate' and step.get('checkpoint') in CHILDREN:
            f=child_receipt(step,feedback)
            if f is None:return False
            child={**step,'skill':step['subskill']}
            return BaseWorkspace._step_done(child,f,plan)
        return BaseWorkspace._step_done(step,feedback,plan)

    def plan_status(self,snapshot):
        plan=super().plan_status(snapshot)
        job=snapshot.get('job') or {}
        if not plan or job.get('execution_id')!=plan.get('execution_id'):return plan
        feedback={'substeps':job.get('completed_substeps',[])}
        active=False
        for step in plan['steps']:
            if step.get('skill')=='relocate' and step.get('checkpoint') in CHILDREN:
                # The base workspace has already checked epoch and the outer
                # execution identity. A sibling/older composite is not evidence.
                if plan.get('stage_bindings',{}).get(step['id'])==job['execution_id']:
                    step['status']='done' if self._step_done(step,feedback,plan) else 'pending'
            if step['status'] in ('pending','running'):
                step['status']='running' if not active else 'pending';active=True
        return plan


class ReportingRelocation(RelocationPlanner):
    def _skill(self,*args,**kwargs):
        try:yield from super()._skill(*args,**kwargs)
        finally:
            keys=('primitive_success','verification_source','held_by','position_error_m','rotation_error_rad',
                  'placement','from_arm','to_arm','status')
            rows=[{k:copy.deepcopy(row.get(k)) for k in ('skill','parameters','status','backend')}|
                  {'feedback':{k:copy.deepcopy((row.get('feedback') or {}).get(k)) for k in keys}}
                  for row in self.result['substeps'] if row.get('status') in ('completed','failed')]
            self.feedback({'completed_substeps':rows,'composite':'relocate'})


def install_isolated():
    """Process-local integration; never changes the running main server."""
    import harness.workspace as workspace
    import harness.agent as agent
    workspace.validate_plan=agent.validate_plan=validate_composite_plan
    workspace.Workspace=CompositeWorkspace
    tools=copy.deepcopy(agent.EXECUTION_TOOLS)
    plan=next(t['function'] for t in tools if t['function']['name']=='set_plan')
    plan['description']+=' A relocate stage can bind verified pick/handover/move child receipts; sub_occurrence disambiguates repeated children. All stages with the same outer occurrence bind the same relocate invocation (default first); retries need an explicit occurrence. Placement uses the whole relocate result.'
    properties=plan['parameters']['properties']['steps']['items']['properties']
    properties['title'].update(minLength=1,maxLength=40,description='Short stage label, 1–40 characters; put route details in tool parameters.')
    properties['subskill']={'type':'string','enum':list(SOURCES),'description':'Optional relocate child; defaults from checkpoint. Current transport child is move_relative.'}
    properties['sub_occurrence']={'type':'integer','minimum':1,'maximum':8,'description':'Occurrence of this child inside the bound relocate, required if repeated. Outer occurrence still selects relocate invocation.'}
    agent.EXECUTION_TOOLS=tools
