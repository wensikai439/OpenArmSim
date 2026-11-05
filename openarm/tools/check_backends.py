"""Bounded local backend acceptance; raw failures are kept alongside passes."""
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from PIL import Image

from sim.environment import OpenArmEnv,ROOT
from sim.curobo import call,joint_state,CuRoboPlanner
from sim.tamp import TampPlanner
from sim.policy import load_config,make_client
from sim.policy_wire import TASK
from sim.recording import Recorder

OUT=ROOT/'runs/20261007-platform-upgrade'


def save(name,value):
    (OUT/(name+'.json')).write_text(json.dumps(value,indent=2,default=str)+'\n')


def fk():
    env=OpenArmEnv(images=False);env.reset(seed=0);rows=[]
    try:
        for k in range(5):
            data=mujoco.MjData(env.model);data.qpos[:]=env.data.qpos
            for side in ('left','right'):data.qpos[env.qids[side]]+=.025*k
            mujoco.mj_forward(env.model,data)
            result=call('/fk',{'joints':joint_state(env,data)})
            pos=np.asarray(result['position']).reshape(2,3)
            quat=np.asarray(result['quaternion_wxyz']).reshape(2,4)
            for i,side in enumerate(('left','right')):
                b=env.ee_ids[side];rot=data.xmat[b].reshape(3,3);want=data.xpos[b]+rot@np.array([0,.006,.075])
                q=np.empty(4);mujoco.mju_mat2Quat(q,rot.ravel())
                rows.append({'sample':k,'arm':side,'position_error_m':float(np.linalg.norm(pos[i]-want)),
                    'quaternion_error':float(min(np.linalg.norm(quat[i]-q),np.linalg.norm(quat[i]+q)))})
        save('curobo-fk',{'scope':'software kinematics only, not physical calibration','passed':max(r['position_error_m'] for r in rows)<1e-5,'samples':rows})
    finally:env.close()


def moves():
    rows=[]
    for backend,arm,delta in [('curobo','right',[0,0,.02]),('curobo','left',[0,0,.02]),
                              ('curobo','right',[.025,0,0]),('tamp','left',[0,0,.03]),('tamp','right',[0,-.02,0])]:
        env=OpenArmEnv(images=False);env.reset(seed=0);name=f'{backend}-{arm}-{len(rows)}';rec=Recorder(env,backend,record_images=False,seed=0)
        start=time.monotonic();error=None;status='error';count=0;planner=None
        try:
            Image.fromarray(env.render('operator')).save(OUT/(name+'-before.jpg'))
            planner=(CuRoboPlanner if backend=='curobo' else TampPlanner)(env,'move_relative',parameters={'arm':arm,'delta_m':delta,'frame':'world','speed_scale':1.})
            before=rec.observation(env)
            for action,diagnostic in planner.actions():
                _,_,term,trunc,info=env.step_tamp(action);after=rec.observation(env)
                rec.append(before,action,after,info,terminated=term,truncated=trunc,diagnostics=diagnostic);before=after;count+=1
                if term or trunc:raise RuntimeError('native terminal: '+str(info['evaluation']))
                if count>=1200:raise RuntimeError('bounded step budget')
            status='passed'
        except Exception as exc:error=str(exc)
        finally:
            metrics={'backend':backend,'arm':arm,'delta_m':delta,'status':status,'error':error,'steps':count,
                     'wall_seconds':time.monotonic()-start,'sim_seconds':env.data.time,'feedback':planner.result if planner else {},
                     'evaluation':env.evaluation,'run_id':rec.path.name}
            rec.finish(status,env.info(),error=error,metrics=metrics)
            Image.fromarray(env.render('operator')).save(OUT/(name+'-after.jpg'))
            rows.append(metrics);save('motion-tests',rows);env.close()
            print(name,status,error,flush=True)


def act(clip_gripper=False):
    rows=[];config=load_config(ROOT/'harness/policy.json')
    for seed in (0,1):
        env=OpenArmEnv(images=False);env.reset(seed=seed);rec=Recorder(env,'act',record_images=False,seed=seed,details={'policy':config})
        start=time.monotonic();count=0;latencies=[];error=None;status='incomplete'
        try:
            before=rec.observation(env)
            class ObservationAdapter:
                def state(self):return env.state()
                def render(self,name):return env.render(name)
                def prepare_policy_action(self,action):
                    proposed=action.copy()
                    if clip_gripper:proposed[[7,15]]=np.clip(proposed[[7,15]],env.action_space.low[[7,15]],env.action_space.high[[7,15]])
                    return env.prepare_policy_action(proposed)
            with make_client(config,seed=seed) as client:
                while count<config['max_steps']:
                    t=time.monotonic();chunk=client.infer(ObservationAdapter(),TASK);latencies.append(time.monotonic()-t)
                    for action in chunk:
                        requested=action.copy()
                        if clip_gripper:action[[7,15]]=np.clip(action[[7,15]],env.action_space.low[[7,15]],env.action_space.high[[7,15]])
                        action,controller=env.prepare_policy_action(action)
                        _,_,term,trunc,info=env.step(action);after=rec.observation(env)
                        rec.append(before,action,after,info,terminated=term,truncated=trunc,diagnostics={'controller':controller,'requested_action':requested.tolist(),
                            'gripper_clip_diagnostic':clip_gripper,'clipped':bool(np.any(action[[7,15]]!=requested[[7,15]]))});before=after;count+=1
                        if count%150==0:Image.fromarray(env.render('operator')).save(OUT/f'act-{"clipped-" if clip_gripper else ""}{seed}-{count:04}.jpg')
                        if term or trunc:break
                        if count>=config['max_steps']:break
                    if term or trunc:
                        status='success' if env.evaluation['success'] else 'invalid' if env.evaluation['invalid'] else 'truncated';break
                else:status='budget_exhausted'
        except Exception as exc:status='error';error=str(exc)
        finally:
            result={'seed':seed,'status':status,'error':error,'steps':count,'wall_seconds':time.monotonic()-start,
                'sim_seconds':env.data.time,'inference_seconds':latencies,'evaluation':env.evaluation,
                'run_id':rec.path.name,'gripper_clip_diagnostic':clip_gripper,
                'scope':'copied real ACT weights on candidate simulation; two repeats of same candidate reset, not hardware validation',
                'adapter_note':'clipped to observed action range for diagnosis only; differs from reviewed real deployment; production remains strict' if clip_gripper else 'strict production guard'}
            rec.finish(status,env.info(),error=error,metrics=result)
            Image.fromarray(env.render('operator')).save(OUT/f'act-{"clipped-" if clip_gripper else ""}{seed}-final.jpg')
            rows.append(result);save('act-rollouts-clipped' if clip_gripper else 'act-rollouts',rows);env.close();print('ACT',seed,status,error,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('fk','moves','act'));p.add_argument('--clip-gripper',action='store_true');args=p.parse_args()
    if args.kind=='act':act(args.clip_gripper)
    else:{'fk':fk,'moves':moves}[args.kind]()
