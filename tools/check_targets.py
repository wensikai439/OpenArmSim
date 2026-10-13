"""Predeclared seeded object targets and real cuRobo motion probes.

Native engineering evidence only. All failures are kept; no parameter fitting.
"""
import argparse
import json
import time
import mujoco
import numpy as np
from PIL import Image
from sim.environment import OpenArmEnv,ROOT
from sim.tamp import TampPlanner
from sim.curobo import CuRoboPlanner
from sim.recording import Recorder
from sim.placement import placement_tolerance

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/20261008-control/targets')
    parser.add_argument('--kind',choices=('placement','precision','curobo'),default='placement')
    parser.add_argument('--seed',type=int,default=82)
    parser.add_argument('--recover',action='store_true',help='declare handover recovery for an unreachable held-bottle placement')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('use a new evidence folder under runs')
    out.mkdir(parents=True);rng=np.random.default_rng(args.seed);cases=[]
    if args.kind=='placement':
        for i in range(6):
            cases.append({'id':f'target-{i}','episode':i%3,'arm':'right' if i<4 else 'left',
                'initial_delta_m':[*rng.uniform(-.025,.025,2),0.],
                'offset_xy_m':[float(rng.uniform(.04,.09)),float(rng.uniform(-.06,.06))]})
    elif args.kind=='precision':
        cases=[{'id':name,'episode':0,'arm':arm,'initial_delta_m':[0,0,0],
                'offset_xy_m':offset,'handover_first':transfer}
            for name,arm,offset,transfer in [('right-1cm','right',[.01,0],False),
                ('right-8cm','right',[.08,0],False),('left-transfer-6cm','left',[.06,0],True),
                ('right-left-4cm','right',[0,.04],False),('right-back-4cm','right',[-.04,0],False)]]
    else:
        for i,(side,delta) in enumerate([('right',[0,0,.02]),('left',[0,0,.02]),('right',[0,-.03,.01])]):
            cases.append({'id':f'curobo-{i}','episode':0,'arm':side,'delta_m':delta})
    (out/'protocol.json').write_text(json.dumps({'seed':args.seed,'cases':cases,'selection':'fixed before running all cases',
        'handover_recovery':args.recover,'development_retest':args.recover,
        'scope':'engineering probes; not visual or real transfer benchmark','transfer_ready':False},indent=2)+'\n')
    rows=[]
    for case in cases:
        env=OpenArmEnv(images=False);env.reset(seed=args.seed,options={'episode':case['episode']})
        if args.kind in ('placement','precision'):
            address=env.model.joint('bottle_free').qposadr[0]
            env.data.qpos[address:address+3]+=case['initial_delta_m']
            mujoco.mj_forward(env.model,env.data);env.evaluation=env._evaluate(0.)
            goal=(env.data.xpos[env.bottle_id,:2]+case['offset_xy_m']).tolist()
            tolerance=placement_tolerance('table',goal,env.data.xpos[env.bottle_id,:2])
            operations=[('pick_bottle',{'arm':case['arm']})]
            side=case['arm']
            if case.get('handover_first'):
                other='left' if side=='right' else 'right'
                operations.append(('handover_bottle',{'from_arm':side,'to_arm':other}));side=other
            operations.append(('place_bottle',{'arm':side,'receptacle_id':'table','target_xy_m':goal,'position_tolerance_m':tolerance}))
        else:
            goal=None;operations=[('move_relative',{'arm':case['arm'],'delta_m':case['delta_m'],'frame':'world','planner':'curobo','speed_scale':.5})]
        recorder=Recorder(env,'tamp',record_images=False,seed=args.seed,root=out/'episodes',details={'probe':case,'goal_xy_m':goal,'hardware_access':False})
        start=time.monotonic();steps=[];failure=None;planner=None;count=0
        try:
            for task,params in operations:
                planner=(CuRoboPlanner if args.kind=='curobo' else TampPlanner)(env,task,parameters=params)
                iterator=planner.actions();n=0
                while True:
                    q=env.data.qpos.copy();v=env.data.qvel.copy()
                    try:action,feedback=next(iterator)
                    except StopIteration:break
                    except Exception as exc:
                        if args.recover and task=='place_bottle' and getattr(exc,'code',None)=='placement_unreachable' and n==0:
                            steps.append({'task':task,'parameters':params,'commands':n,'rejected':True,'failure_code':exc.code,'feedback':planner.result})
                            other='left' if params['arm']=='right' else 'right'
                            operations.extend([('handover_bottle',{'from_arm':params['arm'],'to_arm':other}),
                                ('place_bottle',{**params,'arm':other})])
                            planner=None;break
                        raise
                    assert np.array_equal(q,env.data.qpos) and np.array_equal(v,env.data.qvel)
                    before=recorder.observation(env);_,_,terminal,truncated,info=env.step_tamp(action)
                    recorder.append(before,action,recorder.observation(env),info,terminated=terminal,truncated=truncated,diagnostics=feedback)
                    n+=1;count+=1
                    if terminal or truncated:raise RuntimeError('native terminal or 40 s task budget')
                    if n>=1200:raise RuntimeError('skill command budget')
                if planner is None:continue
                assert planner.result['primitive_success']
                steps.append({'task':task,'parameters':params,'commands':n,'feedback':planner.result})
            if args.kind in ('placement','precision'):assert env.evaluation['placement']['confirmed']
            status='passed'
        except Exception as exc:
            status='error';failure={'code':getattr(exc,'code','execution_error'),'message':str(exc),'feedback':planner.result if planner else None}
        row={'case':case,'status':status,'failure':failure,'steps':steps,'commands':count,
             'run_id':recorder.path.name,'run_path':str(recorder.path.relative_to(ROOT)),'wall_seconds':time.monotonic()-start,'sim_seconds':float(env.data.time),
             'goal_xy_m':goal,'bottle_position_m':env.data.xpos[env.bottle_id].tolist(),'evaluation':env.evaluation,
             'hardware_access':False,'transfer_ready':False}
        recorder.finish('diagnostic_complete' if status=='passed' else 'error',env.info(),error=failure,metrics=row)
        (out/(case['id']+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2,default=str)+'\n')
        Image.fromarray(env.render('operator')).save(out/(case['id']+'.jpg'))
        rows.append(row);(out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,default=str)+'\n')
        print(json.dumps({'case':case['id'],'status':status,'failure_code':(failure or {}).get('code'),'run_id':row['run_id']},ensure_ascii=False),flush=True)
        env.close()

if __name__=='__main__':main()
