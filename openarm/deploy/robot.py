"""Real-only API harness. No simulator, policy imports, camera or CAN startup.

The admitted program runs through the official ASPIRE executor only after a
separate real-robot service supplies calibrated measurements and tool adapters.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
TOOLS=('observe','propose_grasps','pregrasp','approach','close_gripper','open_gripper',
       'verify_hold','lift','retreat','start_transfer','advance','finish_transfer',
       'start_place','release_on_support')
RELATIVE_TOOLS=TOOLS+('align_relative','withdraw_donor')


def tools_for(config):
    return RELATIVE_TOOLS if config.get('robot_contract')=='openarm-relative-v1/1' else TOOLS


def read(path):return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value,str):return re.sub(r'\bsk-[A-Za-z0-9_-]{12,}','[REDACTED]',value)
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()
        if k.lower() not in ('api_key','authorization','thinking','reasoning_content','images','session_token','access_token','secret')}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def request(url,body=None,*,key=None,timeout=15):
    headers={'Content-Type':'application/json','User-Agent':'OpenArm-RealHarness/1'}
    if key:headers['Authorization']='Bearer '+key
    req=urllib.request.Request(url,data=None if body is None else json.dumps(body,allow_nan=False).encode(),headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            data=response.read(2*1024*1024+1)
            if len(data)>2*1024*1024:raise RuntimeError('response exceeds limit')
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        # Keep a short structured service error; never include headers/tokens.
        detail=''
        try:
            body=json.loads(exc.read(16384))
            detail=clean(str(body.get('error') or body.get('reason') or ''))[:1000]
        except Exception:pass
        raise RuntimeError('HTTP '+str(exc.code)+(': '+detail if detail else '')) from None
    except Exception:raise RuntimeError('service unavailable or invalid JSON response') from None


class Robot:
    def __init__(self,config,*,execute=False,transport=request,journal=None):
        self.config=config;self.execute=execute;self.transport=transport;self.calls=0
        self.tools=tools_for(config)
        self.motion_requests_attempted=0
        self.journal=journal
        self.last_observation=None
        self.session_id=None
        self.base=config.get('robot_service')
        if self.base:
            p=urllib.parse.urlsplit(self.base)
            if (p.scheme!='http' or p.hostname not in ('127.0.0.1','localhost','::1')
                or p.username or p.password or p.query or p.fragment or p.path not in ('','/')):
                raise ValueError('robot service must be an explicit loopback HTTP origin')
            self.base=self.base.rstrip('/')

    def status(self):
        if not self.base:return {'ready':False,'missing':['real perception and motion service not connected']}
        try:info=self.transport(self.base+'/status',timeout=5)
        except RuntimeError:return {'ready':False,'missing':['real robot service unreachable']}
        missing=[]
        for key,value in [('contract',self.config.get('robot_contract','openarm-body-v4/1')),('source','real_robot'),
                          ('calibration_verified',True),('hardware_execution_available',True)]:
            if info.get(key)!=value:missing.append(key+' not verified')
        if not set(self.tools)<=set(info.get('tools',[])):
            missing.append(str(len(self.tools))+' real tools installed but not active/admitted' if info.get('tool_implementations_included')
                else str(len(self.tools))+' real tool implementations required')
        missing.extend(x for x in info.get('missing',[]) if x not in missing)
        return {'ready':not missing,'missing':missing,'service':info}

    def observe(self):
        if not self.base:raise RuntimeError('real perception service not connected')
        state=validate_observation(self.transport(self.base+'/observation',timeout=5))
        self.last_observation=copy.deepcopy(state)
        if self.journal:self.journal.record('observation',state)
        return state

    def session_request(self,path,body):
        token_path=ROOT/'.runtime/session-token'
        if not token_path.is_file() or token_path.stat().st_mode&0o077:
            raise RuntimeError('private service session token unavailable')
        return self.transport(self.base+path,body,key=token_path.read_text().strip(),timeout=3)

    def begin(self):
        if self.session_id is not None:raise RuntimeError('session already active')
        self.session_id=self.session_request('/session',{})['session_id']

    def abort(self):
        if self.session_id is not None:
            try:self.session_request('/abort',{'session_id':self.session_id})
            except Exception:
                if self.journal:self.journal.record('error',{'reason':'abort not acknowledged; hardware outcome unknown'})

    def end(self):
        if self.session_id is not None:
            self.session_request('/session/end',{'session_id':self.session_id});self.session_id=None

    def call(self,name,*args,**kwargs):
        if name not in self.tools:raise ValueError('unknown real tool')
        if name=='observe':return self.observe()
        if not self.execute:raise RuntimeError('motion requires the explicit --execute session option')
        self.calls+=1
        if self.calls>(192 if self.tools==RELATIVE_TOOLS else 64):raise RuntimeError('real tool call budget exceeded')
        # Revalidate a fresh real measurement before EVERY tool, not just start.
        state=self.observe()
        if state['terminal'] is not False or state['invalid'] is not False:
            raise RuntimeError('terminal or invalid measured state; motion refused')
        required={'lift':{'right'},'start_transfer':{'right'},
                  'finish_transfer':{'right','left'},'start_place':{'left'},'withdraw_donor':{'left'}}
        if name in required and set(state['held_by'])!=required[name]:
            raise RuntimeError('measured hold precondition not satisfied for '+name)
        if name in ('lift','start_transfer'):
            error=right_region_error(state)
            if error:return {'ok':False,'done':True,'error_code':error}
        if name=='release_on_support':
            support=state.get('support_evidence',{}).get('table',{})
            if support.get('verified') is not True or support.get('stationary') is not True or 'left' in state['held_by']:
                return {'ok':False,'done':True,'error_code':'supported_release_required'}
        if (name=='advance' and (state.get('operation') or {}).get('phase')=='release_right'
            and 'left' not in state['held_by']):
            raise RuntimeError('donor release requires measured receiver hold')
        body={'args':list(args),'kwargs':kwargs,'request_id':uuid.uuid4().hex,'session_id':self.session_id}
        if self.journal:self.journal.record('tool_requested',{'name':name,'request':body})
        token_path=ROOT/'.runtime/session-token'
        transport_kwargs={'timeout':120}
        if token_path.is_file():
            if token_path.stat().st_mode&0o077:raise RuntimeError('service session token must be private')
            transport_kwargs['key']=token_path.read_text().strip()
        if name not in ('propose_grasps','verify_hold'):self.motion_requests_attempted+=1
        try:result=self.transport(self.base+'/tools/'+name,body,**transport_kwargs)
        except Exception:
            self.abort()
            if self.journal:self.journal.record('error',{'name':name,'outcome':'unknown; no automatic resend'})
            raise
        if self.journal:self.journal.record('tool_returned',{'name':name,'result':result})
        if name=='close_gripper' and kwargs.get('arm',args[0] if args else 'right')=='right' and result.get('ok') is True:
            after=self.observe()
            if 'right' not in after['held_by']:return {'ok':False,'done':True,'error_code':'grasp_not_confirmed'}
            error=right_region_error(after)
            if error:return {'ok':False,'done':True,'error_code':error}
        return result


def right_region_error(state):
    """Measured region interval includes uncertainty; a predicted grasp is insufficient."""
    region=state.get('grasp_region_evidence',{}).get('right',{})
    if (region.get('verified') is not True or region.get('source')!='measured_geometry'
        or region.get('part')!='body'):
        return 'grasp_region_unverified'
    low,high=region.get('fraction_min'),region.get('fraction_max')
    if (type(low) not in (int,float) or type(high) not in (int,float)
        or not math.isfinite(low) or not math.isfinite(high) or low>high):
        return 'grasp_region_unverified'
    return None if 0.<=low<=high<=.5 else 'grasp_region_mismatch'


def validate_observation(o,*,now=None):
    now=time.time() if now is None else now
    stamp=o.get('timestamp_unix_s')
    if (o.get('source')!='measured_perception' or o.get('frame')!='robot_base'
        or type(stamp) not in (int,float) or not -.001<=now-stamp<=.2):
        raise ValueError('fresh measured robot-base observation required; simulation/state replay rejected')
    required=('held_by','touching','upright','bottle_center_m','pose_revision',
              'gripper_motor_feedback_rad','gripper_open_rad','checkpoint_id',
              'current_grasp_id','last_failure','completed_steps','operation',
              'remaining_seconds','terminal','invalid')
    if any(k not in o for k in required):raise ValueError('incomplete real observation contract')
    if o.get('hold_evidence_verified') is not True or o.get('object_pose_verified') is not True:
        raise ValueError('measured object poses and independent hold evidence required')
    if (not isinstance(o['held_by'],list) or any(v not in ('left','right') for v in o['held_by'])
        or len(set(o['held_by']))!=len(o['held_by']) or type(o['terminal']) is not bool or type(o['invalid']) is not bool):
        raise ValueError('invalid holder or terminal-state evidence')
    def number(v):return type(v) in (int,float) and math.isfinite(v)
    if (not isinstance(o['bottle_center_m'],list) or len(o['bottle_center_m'])!=3 or
        not all(number(v) for v in o['bottle_center_m']) or type(o['pose_revision']) is not int or o['pose_revision']<1 or
        not number(o['remaining_seconds']) or not 0<=o['remaining_seconds']<=120 or type(o['upright']) is not bool):
        raise ValueError('invalid measured geometry/time values')
    for side in ['left','right']:
        if (type(o['touching'].get(side)) is not bool or
            not number(o['gripper_motor_feedback_rad'].get(side)) or not number(o['gripper_open_rad'].get(side))):
            raise ValueError('invalid gripper feedback')
    def finite_tree(v):
        if isinstance(v,float) and not math.isfinite(v):raise ValueError('nonfinite observation')
        if isinstance(v,dict):
            for value in v.values():finite_tree(value)
        if isinstance(v,list):
            for value in v:finite_tree(value)
    finite_tree(o)
    return o


def run_skill(robot,folder):
    status=robot.status()
    if not robot.execute or not status['ready']:
        return {'status':'blocked','missing':status['missing']+([] if robot.execute else ['--execute not supplied']),
                'motion_commands':0,'task_success':None}
    initial=robot.observe()
    program=ROOT/'app/skill/program.py'
    cfg=read(ROOT/'config/robot.json')
    if hashlib.sha256(program.read_bytes()).hexdigest()!=cfg['program_sha256']:
        raise ValueError('admitted skill hash mismatch')
    sys.path.insert(0,str(ROOT/'app/aspire/src'))
    from cap_harness.registry import ToolRegistry
    from cap_harness.runtime import ProgramExecutor
    registry=ToolRegistry(shared_public_names=robot.tools,public_extension_allowlist=())
    for name in robot.tools:
        def tool(*args,_name=name,**kwargs):return robot.call(_name,*args,**kwargs)
        registry.register(tool,name=name,public=True,layer='shared_high_level',
                          capability='measured real robot only; no reset or simulation')
    result=None;original_error=None;cleanup_errors=[];final=None;started=False
    try:
        robot.begin();started=True
        result=ProgramExecutor(registry).execute_program(program.read_text())
        if not result.ok:original_error=str(result.error)
        else:final=robot.observe()
    except Exception as exc:original_error=type(exc).__name__+': '+str(exc)
    finally:
        if started and original_error:
            try:robot.abort()
            except Exception as exc:cleanup_errors.append('abort: '+str(exc))
        if started:
            try:robot.end()
            except Exception as exc:cleanup_errors.append('session end: '+str(exc))
    # A stopped provider intentionally refuses fresh observations. Preserve the
    # ORIGINAL error and last valid evidence instead of overwriting it with 409.
    program_ok=bool(result and result.ok)
    last=copy.deepcopy(getattr(robot,'last_observation',None) or initial)
    support=(final or {}).get('support_evidence',{}).get('basket',{})
    verified=bool(program_ok and not original_error and not cleanup_errors and final and
        not final['held_by'] and support.get('verified') is True and support.get('stationary') is True)
    # Completion of the program is not independent task success evidence.
    data={'status':'program_finished' if program_ok and not original_error and not cleanup_errors else 'program_failed',
          'program_ok':program_ok,'error':original_error,'cleanup_errors':cleanup_errors,
          'final_observation':final,'last_valid_observation':last,
          'tool_calls':len(result.calls) if result else 0,'task_success':True if verified else None,
          'motion_requests_attempted':robot.motion_requests_attempted,
          'note':'real task verdict requires independent hold, release and basket support evidence'}
    (folder/'skill-result.json').write_text(json.dumps(clean(data),ensure_ascii=False,indent=2)+'\n')
    return data


def settings():
    p=ROOT/'.private/api.json'
    if not p.is_file():raise RuntimeError('API not configured in this deployment')
    if p.stat().st_mode&0o077:raise RuntimeError('API file must have permission 600')
    s=read(p);url=urllib.parse.urlsplit(s['base_url'])
    if url.scheme!='https' or url.username or url.password or url.query or url.fragment:
        raise ValueError('external model API must use HTTPS without URL credentials')
    return s


def model(messages,*,tools=None):
    s=settings();payload={'model':s['model'],'messages':messages,'max_tokens':800}
    if tools is not None:payload.update(tools=tools,tool_choice='auto')
    return request(s['base_url'].rstrip('/')+'/chat/completions',payload,key=s['api_key'],timeout=60)


def chat(task,robot,folder):
    names={'get_status':'Read real deployment readiness. No motion.',
           'get_observation':'Read fresh measured robot observation. No simulated state.',
           'run_bottle_skill':'Run right lower-body grasp, handover left, basket placement. Requires real tools and --execute; stop on missing inputs.'}
    tools=[{'type':'function','function':{'name':name,'description':desc,
        'parameters':{'type':'object','properties':{},'additionalProperties':False}}} for name,desc in names.items()]
    messages=[{'role':'system','content':'You operate an OpenArm real robot harness. First get_status. Distinguish installed tool implementations from active and physically admitted device sessions. Never claim real execution or success from simulation, replay, configured paths, jaw closure or program completion. If perception/calibration/tools are missing, explain the blocker. Do not invent measurements. Never request raw joints, shell, CAN or reset. Maximum one skill execution per session. Reply in Chinese.'},
              {'role':'user','content':task}]
    attempted=False;trace=[]
    for turn in range(6):
        response=model(messages,tools=tools);message=response['choices'][0]['message']
        if robot.journal:robot.journal.record('api_response',{'turn':turn,'model':response.get('model'),
            'usage':response.get('usage'),'content':message.get('content'),'tool_calls':message.get('tool_calls',[])})
        calls=message.get('tool_calls',[])
        messages.append({'role':'assistant','content':message.get('content'),**({'tool_calls':calls} if calls else {})})
        if not calls:
            text=clean(message.get('content') or '')
            (folder/'agent.json').write_text(json.dumps({'reply':text,'tools':trace,'model':response.get('model')},ensure_ascii=False,indent=2)+'\n')
            return {'reply':text,'tools':trace}
        for c in calls:
            name=c['function']['name']
            try:
                if json.loads(c['function'].get('arguments','{}'))!={}:raise ValueError('tool arguments must be empty')
                if name=='get_status':value=robot.status()
                elif name=='get_observation':value=robot.observe()
                elif name=='run_bottle_skill' and not attempted:
                    attempted=True;value=run_skill(robot,folder)
                else:raise ValueError('unavailable or repeated tool')
            except (RuntimeError,ValueError) as exc:value={'status':'blocked','reason':str(exc),
                'motion_requests_attempted':robot.motion_requests_attempted,
                'execution_outcome_unknown':robot.motion_requests_attempted>0}
            value=clean(value);trace.append({'tool':name,'result':value})
            messages.append({'role':'tool','tool_call_id':c['id'],'content':json.dumps(value,ensure_ascii=False)})
    raise RuntimeError('agent turn budget exhausted')


def verify():
    m=read(ROOT/'records/manifest.json')
    for name,digest in m['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('deployment hash mismatch: '+name)
    for name in m.get('mutable_configuration',[]):
        if name not in ('config/calibration.json','config/collision.json'):raise ValueError('unexpected mutable path')
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT):raise ValueError('configuration path escape')
        if p.is_file():read(p)
    return {'verified_files':len(m['files']),'simulator_included':False,
        'configuration_note':'calibration and collision parameters require separately bound physical acceptance'}


def main():
    p=argparse.ArgumentParser(description='OpenArm 纯真机 harness；不包含仿真平台')
    p.add_argument('command',choices=['status','verify','api-test','chat','skill','relative'])
    p.add_argument('task',nargs='?',default='右手抓瓶子下半瓶身，交接给左手，然后放进篮子')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();verify();cfg=read(ROOT/'config/robot.json');robot=Robot(cfg,execute=a.execute)
    if a.command=='verify':result=verify()
    elif a.command=='relative':
        from relative import description
        from service import preflight
        result={**description(),'readiness':preflight(),
            'configuration':'config/calibration.json → motion.relative',
            'entrypoint':'./run chat "右手抓瓶子下半身，交接左手再放入篮子" --execute'}
    elif a.command=='status':
        try:s=settings();api={'configured':True,'model':s['model'],'base_url':s['base_url']}
        except RuntimeError:api={'configured':False}
        result={'deployment':'real_only','simulator_included':False,'api':api,'robot':robot.status()}
    else:
        from service import Journal
        folder=ROOT/'records'/(time.strftime('%Y%m%d-%H%M%S')+'-'+a.command+'-'+uuid.uuid4().hex[:6])
        journal=Journal(folder,{'command':a.command,'task':a.task,'execute_requested':a.execute,
            'skill_sha256':cfg['program_sha256'],'hardware_access_by_harness':False,
            'motion_commands_unknown_until_service_receipts':True})
        robot.journal=journal
        try:
            if a.command=='api-test':
                r=model([{'role':'user','content':'Reply only: connected'}]);result={'connected':True,'model':r.get('model'),'scope':'text connectivity only; no robot motion'}
            elif a.command=='chat':result=chat(a.task,robot,folder)
            else:result=run_skill(robot,folder)
            journal.record('result',result)
        finally:journal.close()
    print(json.dumps(clean(result),ensure_ascii=False,indent=2))
    return 2 if result.get('status')=='blocked' else 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError) as exc:
        print(json.dumps({'status':'blocked','reason':clean(str(exc))},ensure_ascii=False));raise SystemExit(2)
