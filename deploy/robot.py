"""Real-only API harness. No simulator, policy imports, camera or CAN startup.

The admitted program runs through the official ASPIRE executor only after a
separate real-robot service supplies calibrated measurements and tool adapters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
TOOLS=('observe','propose_grasps','pregrasp','approach','close_gripper','open_gripper',
       'verify_hold','lift','retreat','start_transfer','advance','finish_transfer',
       'start_place','release_on_support')


def read(path):return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value,str):return re.sub(r'\bsk-[A-Za-z0-9_-]{12,}','[REDACTED]',value)
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()
        if k.lower() not in ('api_key','authorization','thinking','reasoning_content','images')}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def request(url,body=None,*,key=None,timeout=15):
    headers={'Content-Type':'application/json','User-Agent':'OpenArm-RealHarness/1'}
    if key:headers['Authorization']='Bearer '+key
    req=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode(),headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            data=response.read(2*1024*1024+1)
            if len(data)>2*1024*1024:raise RuntimeError('response exceeds limit')
            return json.loads(data)
    except urllib.error.HTTPError as exc:raise RuntimeError('HTTP '+str(exc.code)) from None
    except Exception:raise RuntimeError('service unavailable or invalid JSON response') from None


class Robot:
    def __init__(self,config,*,execute=False,transport=request):
        self.config=config;self.execute=execute;self.transport=transport;self.calls=0
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
        for key,value in [('contract','openarm-body-v3/1'),('source','real_robot'),
                          ('calibration_verified',True),('hardware_execution_available',True)]:
            if info.get(key)!=value:missing.append(key+' not verified')
        if not set(TOOLS)<=set(info.get('tools',[])):missing.append('14 real tool implementations required')
        return {'ready':not missing,'missing':missing,'service':info}

    def observe(self):
        if not self.base:raise RuntimeError('real perception service not connected')
        return validate_observation(self.transport(self.base+'/observation',timeout=5))

    def call(self,name,*args,**kwargs):
        if name not in TOOLS:raise ValueError('unknown real tool')
        if name=='observe':return self.observe()
        if not self.execute:raise RuntimeError('motion requires the explicit --execute session option')
        self.calls+=1
        if self.calls>64:raise RuntimeError('real tool call budget exceeded')
        # Revalidate a fresh real measurement before EVERY tool, not just start.
        self.observe()
        return self.transport(self.base+'/tools/'+name,{'args':args,'kwargs':kwargs},timeout=60)


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
    if not o.get('hold_evidence_verified') or not o.get('object_pose_verified'):
        raise ValueError('measured object poses and independent hold evidence required')
    return o


def run_skill(robot,folder):
    status=robot.status()
    if not robot.execute or not status['ready']:
        return {'status':'blocked','missing':status['missing']+([] if robot.execute else ['--execute not supplied']),
                'motion_commands':0,'task_success':None}
    robot.observe()
    program=ROOT/'skills/bottle_recovery/program.py'
    cfg=read(ROOT/'config.json')
    if hashlib.sha256(program.read_bytes()).hexdigest()!=cfg['program_sha256']:
        raise ValueError('admitted skill hash mismatch')
    sys.path.insert(0,str(ROOT/'third_party/aspire/src'))
    from cap_harness.registry import ToolRegistry
    from cap_harness.runtime import ProgramExecutor
    registry=ToolRegistry(shared_public_names=TOOLS,public_extension_allowlist=())
    for name in TOOLS:
        def tool(*args,_name=name,**kwargs):return robot.call(_name,*args,**kwargs)
        registry.register(tool,name=name,public=True,layer='shared_high_level',
                          capability='measured real robot only; no reset or simulation')
    result=ProgramExecutor(registry).execute_program(program.read_text())
    # Completion of the program is not independent task success evidence.
    data={'status':'program_finished' if result.ok else 'program_failed',
          'program_ok':result.ok,'error':str(result.error) if result.error else None,
          'tool_calls':len(result.calls),'task_success':None,
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
    messages=[{'role':'system','content':'You operate an OpenArm real robot harness. First get_status. Never claim real execution or success from simulation, replay, configured paths, jaw closure or program completion. If perception/calibration/tools are missing, explain the blocker. Do not invent measurements. Never request raw joints, shell, CAN or reset. Maximum one skill execution per session. Reply in Chinese.'},
              {'role':'user','content':task}]
    attempted=False;trace=[]
    for turn in range(6):
        response=model(messages,tools=tools);message=response['choices'][0]['message']
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
            except (RuntimeError,ValueError) as exc:value={'status':'blocked','reason':str(exc),'motion_commands':0}
            value=clean(value);trace.append({'tool':name,'result':value})
            messages.append({'role':'tool','tool_call_id':c['id'],'content':json.dumps(value,ensure_ascii=False)})
    raise RuntimeError('agent turn budget exhausted')


def verify():
    m=read(ROOT/'MANIFEST.json')
    for name,digest in m['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('deployment hash mismatch: '+name)
    return {'verified_files':len(m['files']),'simulator_included':False}


def main():
    p=argparse.ArgumentParser(description='OpenArm 纯真机 harness；不包含仿真平台')
    p.add_argument('command',choices=['status','verify','api-test','chat','skill'])
    p.add_argument('task',nargs='?',default='右手抓瓶子下半瓶身，交接给左手，然后放进篮子')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();verify();cfg=read(ROOT/'config.json');robot=Robot(cfg,execute=a.execute)
    if a.command=='verify':result=verify()
    elif a.command=='status':
        try:s=settings();api={'configured':True,'model':s['model'],'base_url':s['base_url']}
        except RuntimeError:api={'configured':False}
        result={'deployment':'real_only','simulator_included':False,'api':api,'robot':robot.status()}
    else:
        folder=ROOT/'records'/(time.strftime('%Y%m%d-%H%M%S')+'-'+a.command)
        folder.mkdir(parents=True,exist_ok=False)
        if a.command=='api-test':
            r=model([{'role':'user','content':'Reply only: connected'}]);result={'connected':True,'model':r.get('model'),'scope':'text connectivity only; no robot motion'}
        elif a.command=='chat':result=chat(a.task,robot,folder)
        else:result=run_skill(robot,folder)
    print(json.dumps(clean(result),ensure_ascii=False,indent=2))
    return 2 if result.get('status')=='blocked' else 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError) as exc:
        print(json.dumps({'status':'blocked','reason':clean(str(exc))},ensure_ascii=False));raise SystemExit(2)
