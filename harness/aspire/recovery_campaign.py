"""Practice recovery in the SAME native episode, distill, then freeze/test.

AI decisions are restricted programs over public sensor/segment tools. A
classical recovery baseline gets identical tools and observations. No restore,
reset after start, hidden task-score query, or neural-weight training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .runtime import ROOT, dump
from .client import ModelClient, image_part

API='''All observations/results are recursively read-only; lists become TUPLES. Use list(x) before list equality.
Allowed injected Python functions (no imports except math, no files/network/private attrs):
observe() -> {held_by:tuple of arm names (empty tuple means no hold; ('right',) means right-only), touching:{left:bool,right:bool}, upright:bool, bottle_center_m:xyz, pose_revision:int, gripper_motor_feedback_rad:{left,right}, gripper_open_rad:{left,right}, checkpoint_id:str or None, current_grasp_id:str or None, completed_steps:tuple of receipt names, operation:None or {kind,phase}, last_failure:None or {code,tool,tick}, remaining_seconds:float, terminal:bool, invalid:bool}.
Sensor data are simulated geometry, optionally diagnostic noise; NOT real visual localization. No task score.
last_failure is a HISTORICAL receipt, not a reason to stop after successful recovery. Use fresh holder/contact/operation evidence to decide what remains.
propose_grasps(exclude_ids=()) -> {candidates:[{id,geom,yaw_rad,pitch_rad,grasp_m,pregrasp_m}],pose_revision}; scratch IK/path checked, NOT a successful grasp. Exclusions apply only within the same pose revision. Empty hands required.
pregrasp(grasp_id) -> opens empty right jaw, moves to pregrasp, records verified empty-hand checkpoint.
approach(grasp_id) -> checked approach only. Does not close or lift.
close_gripper(arm='right') -> closes and requires actual >=100ms two-finger hold. ok=False grasp_not_confirmed is recoverable; no lift happened. left closure requires donor right hold.
open_gripper(arm='right') -> only if that arm is NOT holding an object; held-object release forbidden.
verify_hold(arm='right') -> {ok,contact_seconds}; do not infer hold from jaw closure alone.
lift() -> requires right hold, native upward motion and60mm actual object lift,2s contact/relative-pose audit.
retreat(checkpoint_id) -> ACTUAL collision-checked motion to last pregrasp TCP. Empty hands and no right touch required. No state restoration.
start_transfer(placement_aware=True) -> begins COMPLETE right-to-left relay including donor release and withdrawal. Must then call advance() repeatedly until done=True. After SUCCESS DO NOT call finish_transfer(): the donor is already released. Start does not execute motion. Native receiver contact must hold before donor release.
advance() -> {ok,done,error_code,phase_completed,next_phase}; executes ONE native phase. Repeat at most16times until done=True; if ok=False the operation is discarded and state preserved. Reobserve before recovery. Native invalid/timeout states cannot be resumed/reset.
finish_transfer() -> only after BOTH actual holds. Opens donor, collision-checked withdrawal,2s left independent hold. Suitable after fixing an unsuccessful receiver closure; not a shortcut around hold checks.
start_place() -> begins left basket placement; call advance() until done. Requires left-only hold.
All motion result dicts have ok,done,error_code; pregrasp/approach/close/open/lift/retreat/finish_transfer are complete when returned.
No direct XYZ/joints, no reset, no task-success assertion. Max64calls /120sim seconds total. Successful text is not a score.
Normal goal: right pick -> right-to-left transfer -> left basket. Recovery uses CURRENT observed state, never repeats already completed pick/lift. Terminal collision/timeout stops. No automatic budget extensions.'''


def require(row):
    if not row['ok']:raise RuntimeError(str(row.get('error_code')))


def drain(w):
    for _ in range(16):
        row=w.advance()
        if not row['ok'] or row['done']:return row
    raise ValueError('operation phase budget exhausted')


def fixed_policy(w):
    """Strong explicit rules; same public state/tools as generated programs."""
    o=w.observe()
    if 'lift' not in o['completed_steps']:
        for attempt in range(3):
            prior=o['current_grasp_id'];g=w.propose_grasps(exclude_ids=[prior] if prior else [])['candidates']
            if not g:raise ValueError('no grasp')
            ident=g[0]['id'];require(w.pregrasp(ident));require(w.approach(ident))
            close=w.close_gripper('right')
            if not close['ok']:
                fresh=w.observe()
                # Closed jaw/no object indicates pose/grasp error. Still-open
                # jaw permits one closure retry at the same checked approach.
                if abs(fresh['gripper_motor_feedback_rad']['right']-fresh['gripper_open_rad']['right'])<.12:
                    close=w.close_gripper('right')
                if not close['ok']:
                    require(w.open_gripper('right'));require(w.retreat(fresh['checkpoint_id']))
                    o=w.observe();continue
            require(w.verify_hold('right'));require(w.lift());break
        else:raise ValueError('grasp recovery exhausted')
    o=w.observe()
    if list(o['held_by'])==['right']:
        require(w.start_transfer());row=drain(w)
        if not row['ok']:
            o=w.observe()
            if list(o['held_by'])==['right'] and row.get('error_code') in ('dual_hold_not_confirmed','receiver_hold_not_confirmed'):
                require(w.close_gripper('left'));require(w.verify_hold('left'));require(w.finish_transfer())
            else:raise ValueError('no fixed recovery for '+str(row.get('error_code')))
    o=w.observe()
    if list(o['held_by'])!=['left']:raise ValueError('left hold required')
    require(w.start_place());require(drain(w))


def run_trial(case,output,program,mode,skill=None):
    from .recovery import RecoveryWorld
    world=None;started=time.monotonic();executions=[];error=None
    try:
        world=RecoveryWorld(case,output)
        if mode=='fixed':fixed_policy(world)
        else:
            code=Path(program).read_text()
            result=world.execute(code,'program-0');executions.append(result)
            if mode=='online':
                client=ModelClient(output/'api',request_limit=2,token_limit=10000)
                for attempt in range(2):
                    if result['program_ok']:break
                    state=result['state']
                    if world.env._done or world.env.evaluation['invalid']:break
                    label=f'repair-observation-{attempt+1}';world.capture(label)
                    content=[{'type':'text','text':API+'\nThe previous program stopped. Continue from CURRENT state in the SAME episode. Diagnose the failure before choosing recovery. Never reset, restart completed grasp/lift, or release an unverified hold. Return short recovery code that also completes the remaining relay/place.\n'+json.dumps({'previous_program':code,'execution_error':result['error'],
                        'current_state':state,'recent_native_stages':world.rows[-4:]},ensure_ascii=False)}]
                    content += [image_part(output/'media'/f'{label}-{camera}.jpg') for camera in ('head','right')]
                    submission=client.ask([{'role':'system','content':'You repair robot behavior from native evidence. Do not assume a held object from a close command. Use tool result checks and bounded loops.'},{'role':'user','content':content}],
                        tool_name='submit_recovery',description='Submit a bounded continuation from the current live simulator state',
                        schema={'type':'object','properties':{'code':{'type':'string'},'diagnosis':{'type':'string'}},'required':['code','diagnosis'],'additionalProperties':False},max_tokens=2400)
                    code=submission['code'];dump(output/f'repair-{attempt+1}.json',submission)
                    result=world.execute(code,f'program-{attempt+1}');executions.append(result)
        result=world.result()
    except Exception as exc:
        error=str(exc);result=world.result() if world else {'task_success':False}
    finally:
        if world:
            world.capture('final');world.close()
    execution_ok=(error is None and (mode=='fixed' or bool(executions and executions[-1]['program_ok'])))
    result['physical_task_success']=result['task_success']
    result['program_ok']=execution_ok
    result['task_success']=bool(result['physical_task_success'] and execution_ok)
    result.update(mode=mode,error=error,executions=executions,wall_seconds=time.monotonic()-started,
        program_sha256=hashlib.sha256(Path(program).read_bytes()).hexdigest() if program else None)
    dump(output/'result.json',result)
    print(json.dumps({'mode':mode,'success':result['task_success'],'error':error,
        'failure':result.get('native_failures')}),flush=True)
    return result


def subprocess_trial(case,output,program,mode):
    output.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-B','-m','harness.aspire.recovery_campaign','trial','--case',str(case),
        '--output',str(output),'--program',str(program),'--mode',mode]
    with output.with_suffix('.log').open('w') as stream:
        try:subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
                MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),timeout=240,check=False)
        except subprocess.TimeoutExpired:pass
    path=output/'result.json'
    return json.loads(path.read_text()) if path.exists() else {'task_success':False,'error':'process_or_wall_budget','mode':mode}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['trial'])
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--program',type=Path,required=True);p.add_argument('--mode',choices=['frozen','fixed','online'],required=True)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs'):p.error('output must stay inside runs/')
    run_trial(json.loads(a.case.read_text()),out,a.program,a.mode)
