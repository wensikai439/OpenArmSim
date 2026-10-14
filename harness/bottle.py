"""Optional directed bottle tools; old demo and tested upright route retained."""
import copy
import json
import urllib.request

from .poses import PoseHarness, PoseWorkspace, consolidate as pose_consolidate


class VisibleRequests:
    """Process-local tool filtering around the unchanged provider adapter."""
    def __init__(self,base,owner):self.base,self.tool_visibility_owner=base,owner
    def __getattr__(self,name):return getattr(self.base,name)
    def urlopen(self,request,*args,**kwargs):
        if isinstance(request,urllib.request.Request) and request.data is not None:
            payload=json.loads(request.data)
            if isinstance(payload.get('tools'),list):
                owner=self.tool_visibility_owner
                payload['tools'],meta=owner.tools_for_request(payload['tools'])
                if meta['hidden_tools']:
                    request=urllib.request.Request(request.full_url,data=json.dumps(payload).encode(),
                        headers=dict(request.header_items()),method=request.get_method())
                    if owner.active_session:
                        owner.workspace.event(owner.active_session,'protocol','当前可用工具',meta,'completed')
        return self.base.urlopen(request,*args,**kwargs)


def consolidate_hold(receipts,forgotten=()):
    """Requested duration and audit strength are reusable method conditions."""
    scoped=[];contexts={};conditions={}
    for receipt in receipts:
        item=copy.deepcopy(receipt);seconds=item.get('arguments',{}).get('hold_seconds')
        tags=[];applies={}
        if (item['tool']=='pick' and item['evidence_kind']=='simulator_oracle'
            and item['feedback'].get('future_placement')
            and not item['arguments'].get('next_handover')):
            tags.append('grasp_intent=table_placement_lookahead')
            applies['grasp_intent']='table_placement_lookahead'
        if (item['tool'] in ('pick','handover') and item['evidence_kind']=='simulator_oracle'
            and type(seconds) in (int,float) and 2<=seconds<=10):
            audit=item['feedback'].get('hold_audit',{})
            mode='all_full_windows' if (audit.get('validation_method')=='all_full_windows'
                or 'continuous_full_windows_stable' in audit) else 'final_window'
            tags.extend([f'requested_hold={seconds:g}',f'audit={mode}'])
            applies.update(requested_hold_seconds=seconds,hold_validation=mode)
        if tags:
            # Keep old forget IDs effective when this version splits a group.
            if any(x['id'] in forgotten for x in pose_consolidate([item])):continue
            key=str(item.get('context'))+'|'+'|'.join(tags)
            contexts[key]=item.get('context');conditions[key]=applies;item['context']=key
        scoped.append(item)
    lessons=pose_consolidate(scoped,forgotten)
    for lesson in lessons:
        if lesson['context'] not in contexts:continue
        applies=conditions[lesson['context']];lesson['context']=contexts[lesson['context']]
        lesson.setdefault('conditions',{}).update(applies)
        if applies.get('grasp_intent'):
            lesson['title']+=' · 面向桌面放置'
            lesson['evidence_scope']+=' 抓取候选经过原目标放置前瞻；不代表通用抓取或已完成放置。'
        if 'requested_hold_seconds' in applies:
            lesson['title']+=f' · 持稳请求 {applies["requested_hold_seconds"]:g} 秒'
            lesson['evidence_scope']+=(' 成功回执须覆盖请求时长内每个完整稳定窗口；失败回执不能证明持稳。'
                if applies['hold_validation']=='all_full_windows' else
                ' 此类记录只带末段窗口审核，不能当作全时段稳定证据。')
    return lessons


class BottleWorkspace(PoseWorkspace):
    """Keep the final bottle-method reducer on the workspace instance."""
    def consolidate_memory(self, receipts, forgotten=()):
        return consolidate_hold(receipts, forgotten)


class BottleHarness(PoseHarness):
    def chat(self,*args,**kwargs):
        if getattr(self.request_transport,'tool_visibility_owner',None) is not self:
            self.request_transport=VisibleRequests(self.request_transport,self)
        return super().chat(*args,**kwargs)

    def tools_for_request(self,tools):
        """Expose preparation only when a new task actually needs a window."""
        snapshot=self.service.snapshot();observation=snapshot.get('observation',{})
        session=self.sessions.get(self.active_session,{})
        plan=self.workspace.active_plan or {}
        pristine=(observation.get('tick')==0 and observation.get('sim_time')==0
            and not observation.get('episode_finished')
            and not snapshot.get('diagnostics',{}).get('evaluation',{}).get('invalid'))
        active=(plan.get('turn_id')==self.active_session and self.active_session is not None)
        hide=pristine or active or session.get('actions',0)>0 or session.get('prepared',False)
        visible=[t for t in tools if not(hide and t['function']['name']=='prepare_simulation')]
        return visible,{'schema':'observation-tool-visibility/1','hidden_tools':
            ['prepare_simulation'] if len(visible)!=len(tools) else [],
            'reason':'already pristine or same active goal; no new task window needed' if hide else 'new user goal in an advanced scene',
            'source':'current native task clock and this root session; no physical state mutation'}

    def _invoke(self,name,args,role,context=None):
        plan=self.workspace.active_plan or {}
        root=(context or {}).get('plan_turn_id',(context or {}).get('turn_id'))
        current_plan=bool(root and root==plan.get('turn_id') and role=='execute')
        default_fields=[]
        # New geometric tasks include computation by default. Explicit paused
        # diagnostics and the independent VLA contract remain unchanged. Never
        # change a declared mode after actions or during a continuation.
        if (name=='set_plan' and role=='execute' and root and not current_plan
            and not any(s.get('skill')=='run_policy' for s in args.get('steps',[]))):
            args=copy.deepcopy(args)
            if 'timing_mode' not in args:
                args['timing_mode']='measured_delay';default_fields.append('timing_mode')
            if args['timing_mode']=='measured_delay' and 'time_budget_s' not in args:
                args['time_budget_s']=120;default_fields.append('time_budget_s')
        if current_plan and name in ('move_relative','move_to_pose'):
            pending=[s for s in plan.get('steps',[]) if s.get('skill') in ('move_relative','move_to_pose')
                and s.get('status') not in ('done','cancelled','failed')]
            if pending and pending[0]['skill']!=name:
                raise ValueError('运动工具与本任务阶段不一致：当前声明 '+pending[0]['skill']+
                    '，收到 '+name+'。先按原目标调用声明工具；如需改用其它运动工具，先更新计划并保留已完成阶段、原目标、计时与预算。')
        if current_plan and name=='pick' and args.get('object_id')=='bottle':
            goal=plan.get('placement_goal') or {}
            relay=any(s.get('skill')=='handover' and s.get('status') not in ('done','failed','cancelled')
                and s.get('object_id') in (None,'bottle') for s in plan.get('steps',[]))
            if goal.get('surface')=='table' and goal.get('object_id')=='bottle' and not relay:
                context=copy.deepcopy(context or {})
                context['bottle_single_arm_placement']=True
        if current_plan and name=='pick' and args.get('object_id')=='bottle' and args.get('arm')=='left' and not args.get('next_handover'):
            future=[s for s in plan.get('steps',[]) if s.get('skill')=='handover'
                and s.get('status') not in ('done','failed','cancelled') and s.get('object_id') in (None,'bottle')]
            goal=plan.get('placement_goal') or {}
            if future and (not goal or goal.get('object_id')=='bottle'):
                args=copy.deepcopy(args);args['next_handover']={'to_arm':'right'}
                context=copy.deepcopy(context or {})
                context['bottle_grasp_intent_source']={'plan_step_id':future[0]['id'],
                    'source':'this model-declared pending directed relay; proposal hint only, no new action'}
        if current_plan and name=='handover' and args.get('object_id')=='bottle' and args.get('from_arm')=='left':
            goal=plan.get('placement_goal')
            if goal and goal['surface']=='table' and goal['object_id']=='bottle':
                context=copy.deepcopy(context or {})
                context['relay_lookahead']={'surface':'table','target_xy_m':copy.deepcopy(goal['target_xy_m'])}
        result=super()._invoke(name,args,role,context)
        if default_fields:
            result['timing_policy']={'mode':'measured_delay','time_budget_s':args['time_budget_s'],
                'defaulted_fields':default_fields,'source':'new geometric task defaults; omitted fields only',
                'locked_from_first_plan':True,'initial_model_planning_excluded':True,
                'explicit_paused_diagnostics_and_VLA_contract_unchanged':True}
        if name=='observe':
            result['active_scene_contract']['bottle_directed_tools']=(
                'Explicit left→right bottle relay uses current native grasp geometry, obstacle-derived bypass, '
                'connected receiver search and checked empty-donor parking. A locked table goal supplies placement lookahead. '
                'For a planned left→right relay, pick.next_handover.to_arm=right ranks donor grasps before pick. '
                'handover.hold_seconds requests2–10s actual independent receiver hold after donor parking; every complete native audit window must pass. '
                'Right→left and the old full bottle demo remain the original routes. '
                'New geometric plans default to measured_delay and a bounded120s budget when omitted; the first declared mode/budget remains locked. '
                'Only fixed native development cases have evidence; no arbitrary actions or physical transfer claim.')
        return result


def install_isolated():
    import harness.agent as agent
    previous=agent.catalogue;tools=copy.deepcopy(agent.SKILL_TOOLS)
    for item in tools:
        f=item['function']
        if f['name']=='pick':
            f['description']+=(' A planned left→right bottle relay can declare next_handover.to_arm=right '
                'to choose a donor grasp with transport feasibility. This changes candidate ranking only; '
                'actual contact/lift/relay and original goal validation remain required.')
        elif f['name']=='handover':
            f['parameters']['properties']['hold_seconds']={'type':'number','minimum':2,'maximum':10,
                'description':'Optional actual independent receiver hold; all complete native audit windows checked before the skill returns.'}
            f['description']=('Transfer the currently SELECTED registered object between distinct arms. '
                'Left→right bottle, box/cylinder use connected geometry search; the right→left bottle demo remains available. '
                'Requires actual lifted donor hold and empty receiver. Bottle left→right includes obstacle-derived bypass '
                'and checked empty-donor parking after native dual contact/release/independent receiver hold. '
                'A locked table goal adds scratch-only placement lookahead; predictions are not execution or success. '
                'Optional hold_seconds2–10 requests actual independent receiver hold, not wall-clock waiting. '
                'Full contact, unselected-object and original placement gates remain required; not visual localization or real calibration.')
    agent.SKILL_TOOLS=tools
    def catalogue(snapshot,policy_type='pi05'):
        value=previous(snapshot,policy_type)
        value['bottle_directed_scope']={'provider':'optional native bottle left→right search',
            'planning':'current bottle bounds, obstacle bypass, grasp relation, receiver search, donor parking',
            'evidence':'fixed development cases; inspect reports for complete task status',
            'original_demo_preserved':True,'physical_calibration':False,'transfer_ready':False}
        for skill in value['skills']:
            f=next((t['function'] for t in agent.SKILL_TOOLS if t['function']['name']==skill['id']),None)
            if f:skill.update(description=f['description'],parameters=f['parameters'])
        return value
    agent.catalogue=catalogue


def service_factory():
    from .poses import service_factory as poses
    from sim.bottle_routing import BottlePickPlanner,BottleHandoverPlanner,BottlePlacementPlanner,ContinuousHoldAudit
    from sim.oriented_grasp import HoldAudit
    base=poses();previous=getattr(base,'_tamp_planner_factory_default',None)
    if previous is None:
        raise RuntimeError('bottle service requires an explicit base TAMP planner factory')
    class Router:
        CONTRACT={**previous.CONTRACT,'bottle_left_right':'optional current-geometry relay and native donor parking'}
        def __new__(cls,env,task,**kwargs):
            params=kwargs.get('parameters') or {}
            if env.object_id=='bottle' and task=='place_bottle' and params.get('receptacle_id')=='table':
                return BottlePlacementPlanner(env,task,**kwargs)
            if task=='handover_bottle':
                planner=(BottleHandoverPlanner(env,task,**kwargs) if env.object_id=='bottle' and params.get('from_arm')=='left'
                    else previous(env,task,**kwargs))
                return ContinuousHoldAudit(planner,params['hold_seconds']) if 'hold_seconds' in params else planner
            if (env.object_id=='bottle' and task=='pick_bottle'
                and (params.get('bottle_single_arm_placement') is True
                    or params.get('arm')=='left' and params.get('next_handover',{}).get('to_arm')=='right')
                and env.data.xmat[env.bottle_id].reshape(3,3)[2,2]>=.98):
                planner=BottlePickPlanner(env,task,**kwargs)
                return HoldAudit(planner,params['hold_seconds']) if 'hold_seconds' in params else planner
            return previous(env,task,**kwargs)
    class BottleService(base):
        _tamp_planner_factory_default=Router
        def __init__(self,*args,**kwargs):
            kwargs.setdefault('tamp_planner_factory',Router)
            super().__init__(*args,**kwargs)

        def call(self,command,**arguments):
            if (command=='tamp' and arguments.get('task')=='pick_bottle'
                and (arguments.get('initiator') or {}).get('bottle_single_arm_placement') is True):
                arguments=copy.deepcopy(arguments)
                arguments['parameters']={**arguments.get('parameters',{}),'bottle_single_arm_placement':True}
            return super().call(command,**arguments)
    return BottleService
