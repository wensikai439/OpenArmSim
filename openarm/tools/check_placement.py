"""Native object placement probes; keep all trials, including failures.

MUJOCO_GL=egl .venv/bin/python -B -m tools.check_placement --case right-forward
No live scene edits. Initial-only variation is declared in each run manifest.
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from sim.environment import OpenArmEnv,ROOT
from sim.tamp import TampPlanner
from sim.recording import Recorder

CASES={
    'right-forward':('right',[.08,0],None),
    'right-side':('right',[.02,-.06],None),
    'left-forward':('left',[.06,0],None),
    'left-side':('left',[0,.06],None),
    'shifted-forward':('right',[.06,0],[-.03,.03,0]),
    'handover-basket':('left',None,None),
    'left-forward-via-handover':('left',[.06,0],None),
}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=list(CASES)+['all'],default='all')
    parser.add_argument('--out',default='runs/20261008-placement')
    args=parser.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,(side,offset,delta) in CASES.items():
        if args.case not in (name,'all'):continue
        env=OpenArmEnv(images=False);env.reset(seed=0)
        if delta:
            address=env.model.joint('bottle_free').qposadr[0]
            env.data.qpos[address:address+3]+=delta
            mujoco.mj_forward(env.model,env.data);env.evaluation=env._evaluate(0.)
        initial=env.data.xpos[env.bottle_id].copy()
        rec=Recorder(env,'tamp',record_images=False,seed=0,details={'case':name,
            'initial_condition_delta_m':delta,'scope':'declared native-physics engineering probe, not a randomized benchmark'})
        wall=time.monotonic();steps=[];error=None;planner=None;count=0
        try:
            operations=([('pick_bottle',{'arm':'right'}),('handover_bottle',{}),
                         ('place_bottle',{'arm':'left','receptacle_id':'basket'})] if offset is None else
                        [('pick_bottle',{'arm':side}),('place_bottle',{'arm':side,'receptacle_id':'table','target_xy_m':(initial[:2]+offset).tolist()})])
            if name=='left-forward-via-handover':
                operations=[('pick_bottle',{'arm':'left'}),('handover_bottle',{'from_arm':'left','to_arm':'right'}),
                            ('place_bottle',{'arm':'right','receptacle_id':'table','target_xy_m':(initial[:2]+offset).tolist()})]
            for task,params in operations:
                planner=TampPlanner(env,task,parameters=params);actions=planner.actions();n=0
                while True:
                    q=env.data.qpos.copy();v=env.data.qvel.copy()
                    try:action,diagnostic=next(actions)
                    except StopIteration:break
                    assert np.array_equal(q,env.data.qpos) and np.array_equal(v,env.data.qvel)
                    before=rec.observation(env);_,_,terminal,truncated,info=env.step_tamp(action)
                    n+=1;count+=1
                    rec.append(before,action,rec.observation(env),info,terminated=terminal,truncated=truncated,diagnostics=diagnostic)
                    if terminal and info['evaluation']['success'] and task=='place_bottle' and info['evaluation']['placement']['confirmed']:
                        planner.result.update(primitive_success=True,verification_source='native_support_release_and_stability',placement=info['evaluation']['placement'])
                        break
                    if terminal or truncated:raise RuntimeError('native terminal: '+str(info['evaluation']))
                    if n>=1200:raise RuntimeError('skill step budget')
                steps.append({'task':task,'parameters':params,'steps':n,'feedback':planner.result})
            assert env.evaluation['placement']['confirmed']
            assert not any(env.evaluation['touching'].values()) and not env.evaluation['invalid']
            status='passed'
        except Exception as exc:
            status='error';error=str(exc)
            if planner:planner.result.update(primitive_success=False,failure_code=getattr(exc,'code','execution_error'),message=error)
        row={'case':name,'status':status,'error':error,'steps':steps,'control_steps':count,
             'sim_seconds':float(env.data.time),'wall_seconds':time.monotonic()-wall,'run_id':rec.path.name,
             'initial_bottle_position_m':initial.tolist(),'final_bottle_position_m':env.data.xpos[env.bottle_id].tolist(),
             'final_evaluation':env.evaluation,'failed_feedback':planner.result if error and planner else None}
        rec.finish('diagnostic_complete' if status=='passed' else 'error',env.info(),error=error,metrics=row)
        trials=out/'trials';trials.mkdir(exist_ok=True)
        (trials/(rec.path.name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2,default=str)+'\n')
        Image.fromarray(env.render('operator')).save(trials/(rec.path.name+'.jpg'))
        results.append(row);print(json.dumps({'case':name,'status':status,'error':error,'run_id':rec.path.name,
            'failure_code':(row['failed_feedback'] or {}).get('failure_code'),'final':row['final_bottle_position_m']},ensure_ascii=False),flush=True)
        env.close()
    (out/'latest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str)+'\n')
    if any(row['status']!='passed' for row in results):raise SystemExit(1)

if __name__=='__main__':main()
