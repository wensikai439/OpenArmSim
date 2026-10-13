"""Compact model-only tool results; raw evidence and commands are untouched.

Projection occurs at the provider boundary, after the original native receipts
and UI events have been saved. Known search/trajectory history is summarized,
while tool identities, arguments, current state, failure/hold/goal gates and
sensor blocks retain their original meaning. Not a physical verification.
"""
import copy
import json
from collections import Counter
from types import SimpleNamespace
import urllib.request


TRAJECTORIES = {'trajectory','joint_trajectory','joint_positions','position_trajectory',
    'trajectory_positions','waypoints','path','path_positions','path_q_rad'}
OPAQUE = {'parameters','arguments','dispatch_arguments','placement_goal','motion_goal',
    'original_absolute_motion','revalidation','current_observation','user_requirements','task_progress'}


def encoded(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def project(value):
    """Copy a tool result, without modifying it or changing its scalar gates."""
    omitted=Counter()
    def visit(item,key='',native=False):
        if isinstance(item,dict):
            native=native or key in ('tool_feedback','feedback') and ('planner' in item or 'scene_source' in item)
            out={}
            for name,child in item.items():
                if name in OPAQUE:
                    out[name]=copy.deepcopy(child);continue
                if native and name.startswith('private_'):
                    omitted[name]+=1;continue
                if native and name in TRAJECTORIES and isinstance(child,list) and len(child)>3:
                    omitted[name]+=len(child)
                    out[name+'_summary']={'points':len(child),'first':visit(child[0],native=native),
                        'last':visit(child[-1],native=native),'scope':'archived trajectory summary; not a new command'}
                    continue
                if native and name in ('rejected','attempts') and isinstance(child,list) and len(child)>3:
                    omitted[name]+=len(child)-3
                    codes=Counter(str(row.get('code',row.get('failure_code','unknown'))) for row in child if isinstance(row,dict))
                    out[name+'_summary']={'count':len(child),'failure_codes':dict(codes),
                        'last':visit(child[-3:],native=native),'scope':'historical rejected scratch proposals, not executed actions'}
                    continue
                if native and name in ('samples','object_tracking_feedback') and isinstance(child,list) and len(child)>2:
                    omitted[name]+=len(child)-2
                    out[name+'_summary']={'count':len(child),'first':visit(child[0],native=native),'last':visit(child[-1],native=native)}
                    continue
                out[name]=visit(child,name,native)
            return out
        if isinstance(item,list):return [visit(child,key,native) for child in item]
        return item
    result=visit(value)
    if omitted and isinstance(result,dict):
        result['model_context_projection']={'schema':'model-feedback/1',
            'scope':'model-input view only; raw native evidence remains in execution archive and logs',
            'omitted_diagnostic_items':dict(omitted),'scalar_gates_and_current_state_retained':True,
            'original_run_id':value.get('run_id'), 'transfer_ready':False}
    return result,dict(omitted)


def adapt(payload):
    """Only JSON tool-result strings change; all other payload fields stay exact."""
    result=copy.deepcopy(payload);changed=0;totals=Counter();before=after=0
    for message in result.get('messages',[]):
        content=message.get('content')
        if message.get('role')!='tool' or not isinstance(content,str):continue
        before+=len(content.encode())
        try:value=json.loads(content)
        except (TypeError,ValueError):
            after+=len(content.encode());continue
        compact,omitted=project(value)
        if omitted:
            message['content']=encoded(compact);changed+=1;totals.update(omitted)
        after+=len(message['content'].encode())
    return result,{'schema':'model-context/1','tool_result_bytes_before':before,'tool_result_bytes_after':after,
        'changed_tool_results':changed,'omitted_diagnostic_items':dict(totals),
        'tool_calls_arguments_system_prompt_and_images_unchanged':True,
        'source':'deterministic provider-boundary projection; not an LLM-generated memory or task verdict'}


class CompactRequests:
    def __init__(self,base,owner):self.base,self.context_projection_owner=base,owner
    def __getattr__(self,name):return getattr(self.base,name)
    def urlopen(self,request,*args,**kwargs):
        if isinstance(request,urllib.request.Request) and request.data is not None:
            payload=json.loads(request.data)
            if isinstance(payload.get('messages'),list):
                projected,detail=adapt(payload)
                if detail['changed_tool_results']:
                    request=urllib.request.Request(request.full_url,data=json.dumps(projected).encode(),
                        headers=dict(request.header_items()),method=request.get_method())
                root=self.context_projection_owner.active_session
                if root:
                    self.context_projection_owner.workspace.event(root,'protocol','模型上下文',detail,'completed')
        return self.base.urlopen(request,*args,**kwargs)


def install_isolated(Harness):
    class CompactHarness(Harness):
        def chat(self,*args,**kwargs):
            from . import agent
            if getattr(agent.urllib.request,'context_projection_owner',None) is not self:
                agent.urllib=SimpleNamespace(request=CompactRequests(agent.urllib.request,self),error=agent.urllib.error)
            return super().chat(*args,**kwargs)
    return CompactHarness
