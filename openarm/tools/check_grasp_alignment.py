"""Fixed geometric/controller-target candidates; native physics is unchanged."""
import argparse
import hashlib
import json
import re
import time

import numpy as np
from PIL import Image

from sim.environment import ROOT,OpenArmEnv
from sim.tamp import TampPlanner,PlanningFailure
from sim.recording import Recorder
from sim.decision_timing import GraspStability
from tools.probe_grasp import sample


class AlignedGrasp(TampPlanner):
    def __init__(self,*args,height,yaw_deg,roll_deg=0,pitch_deg=0,position_tolerance=.012,**kwargs):
        super().__init__(*args,**kwargs)
        self.height,self.yaw,self.position_tolerance=height,np.deg2rad(yaw_deg),position_tolerance
        self.roll,self.pitch=np.deg2rad([roll_deg,pitch_deg])
        self.strict_orientation=True

    def _pick(self):
        side=self.pick_arm;self._arm(side)
        bottle=self.env.data.xpos[self.env.bottle_id].copy()
        axis=self.env.data.xmat[self.env.bottle_id].reshape(3,3)[:,2]
        cr,sr=np.cos(self.roll),np.sin(self.roll);cp,sp=np.cos(self.pitch),np.sin(self.pitch)
        tool_roll=np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])
        tool_pitch=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        self.rotation=self._grasp_frame(axis,self.yaw)@tool_roll@tool_pitch
        grasp=bottle+axis*self.height;pre=grasp-.13*self.rotation[:,2]
        yield from self._grip(side,self.env.grippers[side].open_motor)
        self._phase('approach');yield from self._move(pre,position_tolerance=self.position_tolerance)
        self._phase('grasp');yield from self._move(grasp,position_tolerance=self.position_tolerance)
        self._sync();_,rotation=self._pose()
        self.result['before_close_rotation_error_rad']=float(np.arccos(np.clip((np.trace(self.rotation.T@rotation)-1)/2,-1,1)))
        self._phase('close');yield from self._grip(side,.05)
        if not self.env.evaluation['held'][side]:raise PlanningFailure('grasp_not_confirmed','no sustained bilateral contact')
        hand=self.env.data.xmat[self.body].reshape(3,3)
        self.carried=(hand.T@(self.env.data.xpos[self.env.bottle_id]-self.env.data.xpos[self.body]),
                      hand.T@self.env.data.xmat[self.env.bottle_id].reshape(3,3))
        self._phase('lift');yield from self._move(grasp+np.array([0,0,.12]),True,position_tolerance=self.position_tolerance)
        self._phase('hold')
        for _ in range(30):
            yield self.base_action.copy(),{'phase':'hold'}
            if not self.env.evaluation['held'][side]:raise PlanningFailure('grasp_lost','bilateral contact lost during hold')
        lo,_=self.env._bottle_bounds()
        if lo[2]<bottle[2]+.06:raise PlanningFailure('lift_not_confirmed','original lift gate failed')
        self.result.update(arm=side,verification_source='native_two_finger_contact_and_lift',held_by=side,
                           lift_height_m=float(lo[2]-bottle[2]))


CASES=[{'id':'production','baseline':True,'height_m':.195,'yaw_deg':0,'position_tolerance_m':.012},
       {'id':'cap-aligned','height_m':.195,'yaw_deg':0,'position_tolerance_m':.012},
       {'id':'neck-aligned','height_m':.185,'yaw_deg':0,'position_tolerance_m':.012},
       {'id':'body-aligned','height_m':.145,'yaw_deg':0,'position_tolerance_m':.012},
       {'id':'cap-tight','height_m':.195,'yaw_deg':0,'position_tolerance_m':.003},
       {'id':'neck-tight','height_m':.185,'yaw_deg':0,'position_tolerance_m':.003}]+[
       {'id':f'{band}-yaw-{int(yaw):+d}'.replace('+','p').replace('-','n'),
        'height_m':height,'yaw_deg':yaw,'position_tolerance_m':.012}
       for band,height in [('cap',.195),('neck',.185)] for yaw in (-15,15,-30,30)]

ROLL_CASES=[CASES[0]]+[
    {'id':f'{band}-roll-{"neg" if angle<0 else "pos"}-{abs(angle)}','height_m':height,
     'yaw_deg':0,'roll_deg':angle,'position_tolerance_m':.012}
    for band,height in [('cap',.195),('neck',.185)] for angle in (-30,30,-60,60,-90,90)]+[
    {'id':f'{band}-pitch-{"neg" if angle<0 else "pos"}-{abs(angle)}','height_m':height,
     'yaw_deg':0,'pitch_deg':angle,'position_tolerance_m':.012}
    for band,height in [('cap',.195),('neck',.185)] for angle in (-15,15)]


def execute(env,planner,folder):
    rec=Recorder(env,'tamp',root=folder,record_images=False,seed=0,details={'study':'grasp alignment','case':planner.parameters})
    status='error';error=None
    try:
        iterator=planner.actions()
        while True:
            q=env.data.qpos.copy()
            try:action,diag=next(iterator)
            except StopIteration:break
            assert np.array_equal(q,env.data.qpos),'planner edited live physics'
            before=rec.observation(env);_,_,term,trunc,info=env.step_tamp(action)
            rec.append(before,action,rec.observation(env),info,terminated=term,truncated=trunc,diagnostics=diag)
            if term or trunc:raise RuntimeError('native terminal')
        assert planner.result['primitive_success'];status='primitive_completed'
    except Exception as exc:error=str(exc);raise
    finally:rec.finish(status,env.info(),error=error,metrics={'tamp':planner.result})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',default='proposals');parser.add_argument('--case');parser.add_argument('--set',choices=('alignment','roll'),default='alignment');args=parser.parse_args()
    if not re.fullmatch('[a-z0-9-]{1,40}',args.group):raise ValueError('simple study name required')
    cases=[c for c in (CASES if args.set=='alignment' else ROLL_CASES) if args.case is None or c['id']==args.case]
    if not cases:raise ValueError('unknown predeclared case')
    out=ROOT/'runs/20261008-alignment'/args.group
    if out.exists():raise ValueError('preserve prior study')
    out.mkdir(parents=True)
    protocol={'cases':cases,'episode':0,'seed':0,'task_budget_s':120,'hold_seconds':10,
        'audit':{'window_seconds':2,'position_limit_m':.001,'rotation_limit_rad':.02},
        'selection':'evaluate every predeclared candidate; no deployment or holdout tuning',
        'physics':'unchanged native mass, friction, solver, actuator gains/limits; no weld or live pose edits',
        'scope':'engineering grasp-target study, not an independent benchmark or real calibration',
        'checker_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        'hardware_access':False,'transfer_ready':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results=[]
    for case in cases:
        folder=out/case['id'];folder.mkdir();env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
        rows=[];failure=None;started=time.monotonic();planner=None
        try:
            params={'arm':'right'}
            planner=(TampPlanner(env,'pick_bottle',parameters=params) if case.get('baseline') else
                AlignedGrasp(env,'pick_bottle',parameters=params,height=case['height_m'],yaw_deg=case['yaw_deg'],
                             roll_deg=case.get('roll_deg',0),pitch_deg=case.get('pitch_deg',0),
                             position_tolerance=case['position_tolerance_m']))
            execute(env,planner,folder/'episodes')
            audit=GraspStability(env);target=env.last_action.copy();rows.append(sample(env))
            rec=Recorder(env,'wait',root=folder/'episodes',record_images=False,seed=0,details={'study':'fixed 10-second grasp hold','waiting_action':'unchanged last actuator target'})
            wait_status,wait_error='incomplete',None
            try:
                for _ in range(300):
                    before=rec.observation(env);_,_,term,trunc,info=env.step_tamp(target);audit.read();rows.append(sample(env))
                    rec.append(before,target,rec.observation(env),info,terminated=term,truncated=trunc,diagnostics={'grasp_stability':audit.summary()})
                    if term or trunc:raise RuntimeError('native terminal')
                wait_status='completed'
            except Exception as exc:wait_status='error';wait_error=str(exc);raise
            finally:rec.finish(wait_status,env.info(),error=wait_error,metrics={'grasp_stability':audit.summary()})
            grasp=audit.summary();status='stable' if grasp['arms']['right']['status']=='stable_within_audit_window' else 'unstable'
        except Exception as exc:
            failure={'code':getattr(exc,'code','execution_error'),'message':str(exc)};status='failed';grasp=None
        result={'case':case,'status':status,'failure':failure,'grasp_stability':grasp,
            'samples':rows,'feedback':planner.result if planner else {},'evaluation':env.evaluation,
            'sim_seconds':float(env.data.time),'wall_seconds':time.monotonic()-started,'manifest':env.manifest(),
            'hardware_access':False,'transfer_ready':False}
        (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        Image.fromarray(env.render('operator')).save(folder/'result.jpg')
        env.close();results.append({k:v for k,v in result.items() if k not in {'samples','manifest','feedback'}})
        print(json.dumps({'id':case['id'],'status':status,'failure':failure,'grasp_stability':grasp},ensure_ascii=False),flush=True)
        (out/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
