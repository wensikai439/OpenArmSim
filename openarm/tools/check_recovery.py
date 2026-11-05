"""Replay an archived native failure, then test recovery without resetting it."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from sim.environment import ROOT,OpenArmEnv
from sim.tamp import TampPlanner
from sim.recording import Recorder
from sim.calibration import response_dependencies

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--time-budget',type=int,choices=(40,60,90,120),default=40)
    args=parser.parse_args();source=ROOT/args.source;out=(ROOT/args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('use a new output folder under runs')
    prior=json.loads(source.read_text());manifest=json.loads((ROOT/prior['run_path']/'manifest.json').read_text())
    params=prior['failure']['feedback']['parameters'];side=params['arm'];other='right' if side=='left' else 'left'
    operations=[('move_relative',{'arm':side,'frame':'world','delta_m':[0,0,.06]}),
                ('handover_bottle',{'from_arm':side,'to_arm':other}),('place_bottle',{**params,'arm':other})]
    out.mkdir(parents=True);protocol={'source':args.source,'operations':operations,
        'goal_xy_m':prior['goal_xy_m'],'time_budget_s':args.time_budget,'budget_declared_before_prefix_replay':True,
        'reset_within_task':False,'scope':'development recovery probe, distinct from the 40 s benchmark',
        'hardware_access':False,'transfer_ready':False}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    env=OpenArmEnv(images=False);env.reset(seed=manifest['seed'],options={'episode':prior['case']['episode']})
    env.set_task_budget(args.time_budget)
    assert response_dependencies(env.manifest())==response_dependencies(manifest),'native response contract changed'
    adr=env.model.joint('bottle_free').qposadr[0];env.data.qpos[adr:adr+3]+=prior['case']['initial_delta_m']
    mujoco.mj_forward(env.model,env.data);env.evaluation=env._evaluate(0.);env.task_termination='skill'
    for line in (ROOT/prior['run_path']/'transitions.jsonl').read_text().splitlines():
        row=json.loads(line);env.step_tamp(np.asarray(row['action']))
    residual=float(np.linalg.norm(env.data.xpos[env.bottle_id]-prior['bottle_position_m']))
    assert residual<1e-8 and not env._done and not env.evaluation['invalid']
    recorder=Recorder(env,'recovery',root=out/'episodes',record_images=False,seed=manifest['seed'],details=protocol)
    steps=[];error=None;planner=None
    try:
        for task,parameters in operations:
            planner=TampPlanner(env,task,parameters=parameters);commands=0;iterator=planner.actions()
            while True:
                before_q=env.data.qpos.copy()
                try:action,diagnostic=next(iterator)
                except StopIteration:break
                assert np.array_equal(before_q,env.data.qpos)
                before=recorder.observation(env);_,_,terminal,truncated,info=env.step_tamp(action)
                recorder.append(before,action,recorder.observation(env),info,terminated=terminal,truncated=truncated,diagnostics=diagnostic);commands+=1
                if terminal or truncated:raise RuntimeError('native terminal or predeclared time budget')
            steps.append({'task':task,'parameters':parameters,'commands':commands,'feedback':planner.result})
        assert env.evaluation['placement']['confirmed']
        status='passed'
    except Exception as exc:
        status='failed';error={'code':getattr(exc,'code','execution_error'),'message':str(exc),'feedback':planner.result if planner else None}
    report={**protocol,'reproduction_residual_m':residual,'status':status,'steps':steps,'failure':error,'run_path':str(recorder.path.relative_to(ROOT)),
        'sim_seconds':float(env.data.time),'bottle_position_m':env.data.xpos[env.bottle_id].tolist(),'evaluation':env.evaluation}
    recorder.finish('diagnostic_complete' if status=='passed' else 'error',env.info(),error=error,metrics=report)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
    Image.fromarray(env.render('operator')).save(out/'result.jpg');env.close()
    print(json.dumps({'status':status,'sim_seconds':report['sim_seconds'],'failure':(error or {}).get('code'),'completed':[s['task'] for s in steps]}))
    if status!='passed':raise SystemExit(2)

if __name__=='__main__':main()
