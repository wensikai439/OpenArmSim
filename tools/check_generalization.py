"""Reproduce directed bottle transfers; preserve failed trials as diagnostics.

Run from the project: MUJOCO_GL=egl .venv/bin/python -B -m tools.check_generalization
Initial scene variation is declared before recording, never a live control tool.
"""
import json
import time
import numpy as np
import mujoco
from PIL import Image
from sim.environment import OpenArmEnv,ROOT
from sim.tamp import TampPlanner
from sim.recording import Recorder

OUT=ROOT/'runs/20261008-generalization'
CASES=[
 ('left-to-right',None,[('pick_bottle',{'arm':'left'}),('handover_bottle',{'from_arm':'left','to_arm':'right'})]),
 ('held-left-to-right',None,[('pick_bottle',{}),('handover_bottle',{}),('prepare',{}),('handover_bottle',{'from_arm':'left','to_arm':'right'})]),
 ('shifted-left-to-right',[-.03,.03,0],[('pick_bottle',{'arm':'left'}),('handover_bottle',{'from_arm':'left','to_arm':'right'})]),
 ('after-basket',None,[('handover_to_basket',{}),('prepare',{}),('pick_bottle',{'arm':'left'}),('handover_bottle',{'from_arm':'left','to_arm':'right'})]),
]

def main():
    OUT.mkdir(exist_ok=True);rows=[]
    for name,delta,operations in CASES:
        env=OpenArmEnv(images=False);env.reset(seed=0)
        if delta:
            address=env.model.joint('bottle_free').qposadr[0]
            env.data.qpos[address:address+3]+=delta
            mujoco.mj_forward(env.model,env.data);env.evaluation=env._evaluate(0.)
        rec=Recorder(env,'tamp',record_images=False,seed=0,details={
            'case':name,'initial_condition_delta_m':delta,'scope':'native-physics engineering check, not general success rate'})
        wall=time.monotonic();results=[];error=None;planner=None;count=0
        try:
            for task,params in operations:
                if task=='prepare':
                    q=env.data.qpos.copy();v=env.data.qvel.copy();a=env.last_action.copy();t=env.data.time
                    env.prepare_task()
                    assert np.array_equal(q,env.data.qpos) and np.array_equal(v,env.data.qvel)
                    assert np.array_equal(a,env.last_action) and env.data.time==t
                    continue
                planner=TampPlanner(env,task,parameters=params);actions=planner.actions();steps=0
                before=rec.observation(env)
                while True:
                    q=env.data.qpos.copy();v=env.data.qvel.copy()
                    try:action,detail=next(actions)
                    except StopIteration:break
                    assert np.array_equal(q,env.data.qpos) and np.array_equal(v,env.data.qvel)
                    _,_,terminal,truncated,info=env.step_tamp(action);after=rec.observation(env)
                    rec.append(before,action,after,info,terminated=terminal,truncated=truncated,diagnostics=detail)
                    before=after;count+=1;steps+=1
                    if terminal or truncated:
                        if task=='handover_to_basket' and env.evaluation['success']:break
                        raise RuntimeError('native terminal: '+str(env.evaluation['failure_reasons']))
                    if steps>=1200:raise RuntimeError('skill step budget exhausted')
                results.append({'task':task,'parameters':params,'steps':steps,'feedback':planner.result,
                                'held':dict(env.evaluation['held']),'evaluation_success':env.evaluation['success']})
            assert env.evaluation['held']['right'] and not env.evaluation['touching']['left']
            assert not env.evaluation['invalid']
            status='passed'
        except Exception as exc:status='error';error=str(exc)
        row={'case':name,'status':status,'error':error,'steps':results,'control_steps':count,
             'sim_seconds':float(env.data.time),'wall_seconds':time.monotonic()-wall,
             'final_evaluation':env.evaluation,'failed_feedback':planner.result if error and planner else None,
             'run_id':rec.path.name,'initial_condition_delta_m':delta}
        rec.finish('diagnostic_complete' if status=='passed' else 'error',env.info(),error=error,metrics=row)
        Image.fromarray(env.render('operator')).save(OUT/(name+'.jpg'))
        rows.append(row);(OUT/'cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,default=str)+'\n')
        print(name,status,error,flush=True);env.close()
    if any(row['status']!='passed' for row in rows):raise SystemExit(1)

if __name__=='__main__':main()
