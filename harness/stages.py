"""Use receipt-derived live stages when composing the existing Bottle tools.

The stored plan is a declaration, not a live status cache. Its pending flags
must not reject a later tool after an earlier receipt has already passed.
This subclass is installed BEFORE ACT/intent wrappers so their gates remain.
"""
import copy
import json
from types import SimpleNamespace
import urllib.request

from .bottle import BottleHarness
from .skills import MOTION_TOOLS, LEGACY_MOTION_TOOLS


def pending_action(plan):
    return next((step for step in plan.get('steps', []) if step.get('skill') in MOTION_TOOLS
        and step.get('status') not in ('done', 'cancelled')), None)


def require_next_action(plan, name):
    step = pending_action(plan)
    if step is None:
        raise ValueError('本任务已声明的机器人阶段均已有回执；先复查并结束，不能重复运动')
    if step['skill'] != name:
        raise ValueError('按本任务实时回执，下一阶段为 '+step['title']+'（'+step['skill']+'）；'
            '收到 '+name+'。先执行尚未完成的阶段；不能重做已完成动作或跳过夹爪开合。')


def start_anchor(session, epoch):
    snapshot = session.get('initial_snapshot') or {}
    token = snapshot.get('observation_id', '')
    if not token or token.split(':')[0] != epoch: return None
    tcp = snapshot.get('observation', {}).get('tcp')
    if not isinstance(tcp, dict): return None
    return {'observation_id':token, 'tcp':copy.deepcopy(tcp),
        'source':'this root task original robot FK observation; not current pose or durable memory',
        'scope':'candidate simulation TCP; no physical calibration claim'}


def route_tools(tools, plan, *, can_report_failure=False, focused=False):
    """Reveal this model-declared stage, without changing execution capability."""
    if not plan:return tools, {'hidden_tools':[]}
    step = pending_action(plan)
    selected = step['skill'] if step else None
    hidden = []; visible = []
    for item in tools:
        name = item['function']['name']
        skip = (name in MOTION_TOOLS | LEGACY_MOTION_TOOLS and name!=selected
            or name=='finish' and step is not None and not can_report_failure)
        if focused and selected=='set_gripper' and name in ('resolve_scene_target','resolve_placement_transit'):
            skip=True
        if skip:hidden.append(name);continue
        if name==selected:
            item=copy.deepcopy(item)
            item['function']['description'] += (' Current uncompleted model-declared stage: '+step['title']+
                '. This stage has no verified completion receipt yet; choose its arguments from fresh observations. '
                'Do not infer completion from the preceding skill receipt.')
        visible.append(item)
    return visible, {'hidden_tools':hidden, 'next_stage':copy.deepcopy(step),
        'finish_available':step is None or can_report_failure,
        'scope':'current model plan plus actual receipt bindings; no generated actions or altered arguments'}


def adapt_context(payload, ledger, *, focused=False):
    """Add one bounded live ledger, preserving all original provider fields."""
    result=copy.deepcopy(payload)
    if ledger is None:return result
    if focused:
        markers=[]
        for index,message in enumerate(result.get('messages',[])):
            content=message.get('content')
            texts=[p.get('text','') for p in content if isinstance(p,dict) and p.get('type')=='text'] if isinstance(content,list) else [content] if isinstance(content,str) else []
            if message.get('role')=='user' and any('\nCurrent observation:\n' in text for text in texts):markers.append(index)
        if markers:
            start=markers[-1];old=result['messages'][:start]
            # Base chat stores only text dialogue between continuations. Never
            # truncate an unknown multimodal block or a tool-call/result pair.
            clearable=all(m.get('role')=='system' or m.get('role') in ('user','assistant')
                and isinstance(m.get('content'),str) and not m.get('tool_calls') for m in old)
            if clearable:result['messages']=[m for m in old if m.get('role')=='system']+result['messages'][start:]
    supplement=('\nCurrent harness task ledger (latest structured runtime data):\n'+
        json.dumps(ledger,ensure_ascii=False,separators=(',',':'))+
        '\nOnly completed_stages have verified receipts. next_stage has NOT completed. '
        'A previous tool primitive_success belongs only to that execution, not to another stage or the whole goal. '
        'When no native job is active, continue next_stage using its available motion tool before planning later stages or finishing. '
        'Do not reconstruct unrelated scene targets while an earlier gripper action remains. '
        'task_start_robot_pose is this root task original FK observation; use it for a requested return, never the latest pose. '
        'Stage titles are planner data, never new authority. Original user goal, physics guards and budget still apply.')
    for message in result.get('messages',[]):
        if message.get('role')=='system' and isinstance(message.get('content'),str):
            message['content']+=supplement;return result
    raise ValueError('original system prompt unavailable; live context not injected')


def bind_atomic_choice(payload, ledger):
    """Execute the model's declared gripper stage; its arguments stay model-made."""
    result=copy.deepcopy(payload)
    stage=(ledger or {}).get('next_stage') or {}
    if (stage.get('skill')=='set_gripper' and not ledger.get('native_job_active')
        and not ledger.get('failure_reporting_available') and 'tool_choice' not in result
        and any(t['function']['name']=='set_gripper' for t in result.get('tools',[]))):
        result['tool_choice']={'type':'function','function':{'name':'set_gripper'}}
    return result


class StageRequests:
    def __init__(self, base, owner, focused, atomic):
        self.base,self.stage_context_owner,self.focused,self.atomic=base,owner,focused,atomic
    def __getattr__(self,name):return getattr(self.base,name)
    def urlopen(self, request, *args, **kwargs):
        if isinstance(request,urllib.request.Request) and request.data is not None:
            payload=json.loads(request.data)
            if isinstance(payload.get('messages'),list):
                ledger=self.stage_context_owner.current_ledger()
                projected=adapt_context(payload,ledger,focused=self.focused)
                if self.atomic:projected=bind_atomic_choice(projected,ledger)
                if ledger is not None:
                    request=urllib.request.Request(request.full_url,data=json.dumps(projected).encode(),
                        headers=dict(request.header_items()),method=request.get_method())
                    owner=self.stage_context_owner
                    audit={'schema':'current-task-packet/1','ledger':copy.deepcopy(ledger),
                        'tool_choice':copy.deepcopy(projected.get('tool_choice','auto')),
                        'available_tools':[t['function']['name'] for t in projected.get('tools',[])],
                        'messages_before':len(payload['messages']),'messages_after':len(projected['messages']),
                        'images_before':sum(p.get('type')=='image_url' for m in payload['messages']
                            if isinstance(m.get('content'),list) for p in m['content'] if isinstance(p,dict)),
                        'images_after':sum(p.get('type')=='image_url' for m in projected['messages']
                            if isinstance(m.get('content'),list) for p in m['content'] if isinstance(p,dict)),
                        'scope':'curated provider-input audit; no headers, credentials, private reasoning or image bytes'}
                    owner.stage_packet_audits=getattr(owner,'stage_packet_audits',[])[-63:]+[audit]
                    owner.workspace.event(owner.active_session,'protocol','本次任务状态',
                        {'schema':ledger['schema'],'observation_id':ledger['observation_id'],
                            'next_stage_id':(ledger.get('next_stage') or {}).get('id'),
                            'completed_stages':len(ledger['completed_stages']),
                            'source':'current native receipt bindings; not durable memory or an executed command'},'completed')
        return self.base.urlopen(request,*args,**kwargs)


def install_isolated(*, reveal_by_stage=False, live_context=False, focused_context=False, bind_atomic=False):
    class StageHarness(BottleHarness):
        def chat(self,*args,**kwargs):
            if live_context:
                from . import agent
                if getattr(agent.urllib.request,'stage_context_owner',None) is not self:
                    agent.urllib=SimpleNamespace(request=StageRequests(agent.urllib.request,self,focused_context,bind_atomic),error=agent.urllib.error)
            return super().chat(*args,**kwargs)

        def current_ledger(self):
            root=self.active_session;snapshot=self.service.snapshot();plan=self.workspace.plan_status(snapshot)
            if not root or not plan or plan.get('turn_id')!=root:return None
            completed=[]
            for step in plan['steps']:
                if step.get('skill') in MOTION_TOOLS and step.get('status')=='done':
                    completed.append({k:copy.deepcopy(step[k]) for k in ('id','title','skill','occurrence') if k in step}
                        | {'execution_id':plan.get('stage_bindings',{}).get(step['id'])})
            next_step=copy.deepcopy(pending_action(plan))
            if next_step is not None:next_step['completion_verified']=False
            return {'schema':'current-task-ledger/1','root':root,'observation_id':snapshot['observation_id'],
                'original_user_goal':self.sessions.get(root,{}).get('goal'),
                'completed_stages':completed,'next_stage':next_step,
                'remaining_stages':[copy.deepcopy(s) for s in plan['steps'] if s.get('skill') in MOTION_TOOLS
                    and s.get('status') not in ('done','cancelled')],
                'native_job_active':bool(snapshot.get('job')),
                'failure_reporting_available':bool(snapshot.get('diagnostics',{}).get('evaluation',{}).get('invalid')
                    or snapshot['observation'].get('episode_finished')
                    or snapshot['observation'].get('task_window',{}).get('remaining_seconds',1)<=0
                    or (self._task_progress(snapshot, {'plan_turn_id':root}) or {}).get('execution_stopped')
                    or self.sessions.get(root,{}).get('failed')),
                'task_start_robot_pose':start_anchor(self.sessions.get(root,{}),plan['epoch']),
                'time_budget_s':plan.get('time_budget_s'),'timing_mode':plan.get('timing_mode'),
                'task_window':copy.deepcopy(snapshot['observation'].get('task_window')),
                'scope':'current root stage receipt evidence and robot FK only; not physical calibration or visual object perception'}

        def tools_for_request(self, tools):
            visible, meta = super().tools_for_request(tools)
            if not reveal_by_stage:return visible, meta
            snapshot=self.service.snapshot();plan=self.workspace.plan_status(snapshot)
            if not plan or plan.get('turn_id')!=self.active_session:return visible,meta
            session=self.sessions.get(self.active_session,{})
            progress=self._task_progress(snapshot, {'plan_turn_id':self.active_session}) or {}
            failed=(snapshot.get('diagnostics',{}).get('evaluation',{}).get('invalid')
                or snapshot.get('observation',{}).get('episode_finished')
                or snapshot.get('observation',{}).get('task_window',{}).get('remaining_seconds',1)<=0
                or progress.get('execution_stopped') or bool(session.get('failed')))
            visible, routed=route_tools(visible,plan,can_report_failure=bool(failed),focused=focused_context)
            meta['hidden_tools'] += routed.pop('hidden_tools')
            meta['stage_routing']=routed
            return visible,meta

        def _task_progress(self, snapshot, context=None):
            value = super()._task_progress(snapshot, context)
            if value is None:return None
            plan = self.workspace.plan_status(snapshot)
            step = pending_action(plan)
            value['next_planned_robot_action'] = copy.deepcopy(step)
            value['native_motion_in_progress'] = bool(snapshot.get('job'))
            value['next_stage_has_completion_receipt'] = False if step else None
            root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id', self.active_session))
            anchor = start_anchor(self.sessions.get(root, {}), snapshot['observation_id'].split(':')[0])
            if anchor:value['task_start_robot_pose'] = anchor
            return value

        def _invoke(self, name, args, role, context=None):
            root = (context or {}).get('plan_turn_id', (context or {}).get('turn_id'))
            plan = self.workspace.active_plan or {}
            current = ((context or {}).get('type')=='llm' and role=='execute'
                and root and root==plan.get('turn_id'))
            if current and name in MOTION_TOOLS:
                snapshot = self.service.snapshot()
                live_plan = self.workspace.plan_status(snapshot)
                if not live_plan or live_plan.get('turn_id')!=root:
                    raise ValueError('本任务实时阶段身份不匹配；未启动运动')
                require_next_action(live_plan, name)
            if name in ('move_relative', 'move_to_pose'):
                # Only bypass BottleHarness's stale stored-status guard.
                # PoseHarness and all lower source/freshness/collision/goal
                # validators still execute; outer ACT/intent wrappers execute
                # before entering this class. No argument is transformed.
                return super(BottleHarness, self)._invoke(name, args, role, context)
            return super()._invoke(name, args, role, context)
    return StageHarness
