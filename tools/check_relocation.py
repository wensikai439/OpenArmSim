"""Declared native/API relocation cases; all outcomes kept, no goal tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from tools.check_api import http
from sim.environment import ROOT,OpenArmEnv
from sim.relocation import RelocationPlanner
from sim.recording import Recorder,dump


CASES={
 'forward':{'object_id':'bottle','arm':'right','receptacle_id':'table','offset_xy_m':[.08,0]},
 'transfer':{'object_id':'bottle','arm':'left','via_arm':'right','receptacle_id':'table','offset_xy_m':[.06,0]},
 'curobo':{'object_id':'bottle','arm':'right','receptacle_id':'table','offset_xy_m':[.08,0],
           'transport_planner':'curobo','transport_delta_m':[.03,0,0]},
 'recovery':{'object_id':'bottle','arm':'left','receptacle_id':'table','offset_xy_m':[.06,0],
             'allow_handover_recovery':True},
}


def native(case,out):
    out.mkdir(parents=True,exist_ok=False);env=OpenArmEnv(images=True)
    env.reset(seed=0);env.set_task_budget(120);initial=env.data.xpos[env.bottle_id].copy()
    rec=None;planner=None;status='error';error=None
    frames=[]
    def capture(label):
        from PIL import Image
        name=f'frame-{len(frames)}.jpg';Image.fromarray(env.render('overview')).save(out/name,quality=92)
        frames.append({'label':label,'tick':env.tick,'sim_time':float(env.data.time),'url':name})
    try:
        rec=Recorder(env,'tamp',root=out,record_images=False,seed=0,
                     details={'study':'declared continuous relocation','case':case,'parameters':CASES[case],
                              'model_round_trips_inside':0,'actual_LLM_API':False})
        capture('开始')
        def feedback(value):
            if value.get('composite_step_complete'):capture(value['composite_step_complete'])
        planner=RelocationPlanner(env,'relocate_object',parameters=CASES[case],feedback=feedback)
        iterator=planner.actions();before=rec.observation(env)
        while True:
            q=env.data.qpos.copy()
            try:action,diag=next(iterator)
            except StopIteration:status='primitive_completed';break
            assert np.array_equal(q,env.data.qpos),'planner edited live state'
            _,_,term,trunc,info=env.step_tamp(action);after=rec.observation(env)
            rec.append(before,action,after,info,terminated=term,truncated=trunc,diagnostics=diag);before=after
            if term or trunc:status='invalid' if info['evaluation']['invalid'] else 'truncated';break
    except Exception as exc:error=str(exc)
    finally:
        if rec:rec.finish(status,env.info(),error=error,metrics={'tamp':planner.result if planner else {}})
        report={'case':case,'mode':'native','status':status,'error':error,'parameters':CASES[case],
            'initial_bottle_position_m':initial.tolist(),'manifest':env.manifest(),
            'feedback':planner.result if planner else {},'sim_seconds':float(env.data.time),
            'frames':frames,'run':rec.path.name if rec else None,'actual_API':False,
            'hardware_access':False,'transfer_ready':False}
        dump(out/'result.json',report);env.close()
        print(json.dumps({k:report[k] for k in ('case','status','error','sim_seconds')},ensure_ascii=False),flush=True)


def api(case,out,*,natural=False):
    if out.exists():raise ValueError('preserve previous API result')
    s=http('/api/status')
    if s['job'] or s['agent']['busy']:raise ValueError('platform busy, leave live task alone')
    if not s['llm']:raise ValueError('saved API unavailable')
    p=CASES[case]
    task=('右手抓起并放到任务起始瓶位向前8厘米的桌面位置' if case in ('forward','curobo') else
          '左手抓起瓶子，交给右手，再放到任务起始瓶位向前6厘米的桌面位置' if case=='transfer' else
          '先尝试左手直接放到起始瓶位向前6厘米；仅在真实不可达时允许一次换手恢复，保持原目标')
    prompt=('只在仿真做一次连续搬运：第一个set_plan锁定timing_mode=measured_delay和time_budget_s=120。'
        +task+'。调用一次relocate，由底层逐子动作确认，完成松手、撤离和稳定支撑，再用新观测复查。'
        '阶段由你拟定，搬运条件绑定placement_confirmed/relocate。目标不能改变，任务中禁止重置、加时、切回paused。'
        '首个计划前和底层规划耗时仍不代表真机实时控制。使用如下明确路线参数，目标坐标由锁定placement_goal解析：'
        +json.dumps({k:v for k,v in p.items() if k!='offset_xy_m'},ensure_ascii=False))
    if natural:
        if case!='transfer':raise ValueError('plain-language protocol currently declares transfer only')
        prompt='用左手拿起桌上的瓶子，交给右手，再用右手把瓶子放到起始瓶位向前6厘米的桌面上。放好后松开并退开，完成后复查。不要在动作中重置场景。'
    protocol={'case':case,'mode':'real_API','prompt':prompt,'parameters':p,'single_scene_reset_before_goal':True,
        'natural_prompt':natural,'timing':'model_declared' if natural else 'measured_delay',
        'budget_s':'model_declared' if natural else 120,
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'hardware_access':False,'transfer_ready':False}
    dump(out.with_suffix('.protocol.json'),protocol)
    http('/api/command',{'command':'reset','arguments':{'episode':0,'seed':0}})
    initial=http('/api/status');started=time.monotonic();response=http('/api/chat',{'message':prompt})
    root=response['turn_id'];print(json.dumps({'case':case,'started':True,'turn_id':root}),flush=True)
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        s=http('/api/status');a=http('/api/agent');turns=[t for t in a['turns'] if t['id']==root or t.get('parent')==root]
        if not s['job'] and not s['agent']['busy'] and turns and all(t['status'] not in ('running','waiting') for t in turns):break
        time.sleep(2)
    else:raise TimeoutError('inspect same live task; do not reset or restart')
    a=http('/api/agent');plan=s['agent'].get('plan') or {};feedbacks=[]
    with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as db:
        for execution in plan.get('execution_tools',[]):
            row=db.execute('SELECT feedback,evidence FROM experiences WHERE id=?',(execution['id'],)).fetchone()
            feedbacks.append({'execution':execution,**({'feedback':json.loads(row[0]),'evidence':json.loads(row[1])} if row else {})})
    turns=[t for t in a['turns'] if t['id']==root or t.get('parent')==root]
    current=[f for f in feedbacks if f['execution']['skill']=='relocate'];goal=plan.get('goal_assessment') or {}
    good=bool(len(current)==1 and current[0].get('feedback',{}).get('primitive_success') and goal.get('confirmed'))
    expected=np.asarray(initial['diagnostics']['manipulation_scene']['position_m'][:2])+np.asarray(p['offset_xy_m'])
    declared=np.asarray((plan.get('placement_goal') or {}).get('target_xy_m',[]))
    target_unchanged=bool(declared.shape==(2,) and np.linalg.norm(declared-expected)<1e-6)
    good=good and target_unchanged
    if case=='transfer':good=good and any(t['skill']=='handover' and t['status']=='completed' and t['parameters'].get('from_arm')=='left' and t['parameters'].get('to_arm')=='right' for t in current[0]['feedback']['substeps']) if current else False
    if case=='curobo':good=good and any(t['backend']=='curobo' and t['status']=='completed' and t['feedback'].get('native_path_revalidation') and not t['feedback'].get('fallback_used',False) for t in current[0]['feedback']['substeps']) if current else False
    good=good and (natural or plan.get('timing_mode')=='measured_delay') and plan.get('epoch')==initial['observation_id'].split(':')[0]
    result={**protocol,'status':'passed' if good else 'failed','turn_id':root,'turns':turns,
        'actual_api_calls':sum(1 for t in turns for event in t.get('trace',[]) if event['kind']=='api'),
        'wall_seconds':time.monotonic()-started,'response':response,'initial_observation_id':initial['observation_id'],
        'feedbacks':feedbacks,'terminal':s,'model_verdict':plan.get('verdict'),
        'expected_target_xy_m':expected.tolist(),'target_unchanged':target_unchanged,
        'scope':'one declared real-API development case; native goal and model verdict separate; no statistical generalization or transfer claim'}
    dump(out,result);print(json.dumps({k:result[k] for k in ('case','status','actual_api_calls','model_verdict')},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['native','api'],required=True)
    parser.add_argument('--case',choices=CASES,required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--natural',action='store_true',help='plain-language transfer; model chooses tools and timing')
    args=parser.parse_args()
    if args.natural and args.mode!='api':raise ValueError('natural language requires actual API')
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs'):raise ValueError('keep experiment in local runs')
    out.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=='native':native(args.case,out)
    else:api(args.case,out,natural=args.natural)


if __name__=='__main__':main()
