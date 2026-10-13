"""Native/GPU carried-motion development cases with fixed conditions."""
import json
import argparse
import hashlib
import re
import time

import numpy as np
from PIL import Image

from sim.environment import ROOT, OpenArmEnv
from sim.curobo import CuRoboPlanner, attachment, metadata
from sim.tamp import TampPlanner
from sim.recording import Recorder
from tools.replay_decision import measurement

OUT=ROOT/'runs/20261008-carry/native'


def execute(env,planner,folder):
    recorder=Recorder(env,'tamp',root=folder,record_images=False,seed=0,
                      details={'task':planner.task,'parameters':planner.parameters,'study':'native/GPU carried Cartesian development'})
    status='error';commands=0
    try:
        actions=planner.actions()
        while True:
            q=env.data.qpos.copy()
            try:action,diag=next(actions)
            except StopIteration:break
            assert np.array_equal(q,env.data.qpos),'planner edited live physics'
            before=recorder.observation(env)
            _,_,terminal,truncated,info=env.step_tamp(action)
            recorder.append(before,action,recorder.observation(env),info,terminated=terminal,truncated=truncated,diagnostics=diag)
            commands+=1
            if terminal or truncated:raise RuntimeError('native terminal')
        assert planner.result['primitive_success'];status='primitive_completed'
        return commands
    finally:
        planner.result['executed_commands']=commands
        recorder.finish(status,env.info(),metrics={'tamp':planner.result})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',default='native');parser.add_argument('--case');args=parser.parse_args()
    if not re.fullmatch('[a-z0-9-]{1,40}',args.group):raise ValueError('simple study name required')
    ready_until=time.monotonic()+30
    while not metadata().get('ready'):
        if time.monotonic()>ready_until:raise RuntimeError('worker did not become ready; no scene test started')
        time.sleep(.2)
    output=ROOT/'runs/20261008-carry'/args.group
    if output.exists():raise ValueError('preserve earlier development receipts')
    output.mkdir()
    cases=[{'id':'empty-right','arm':'right','hold':False,'delta_m':[0,0,.03]},
           {'id':'carry-right-up','arm':'right','hold':True,'delta_m':[0,0,.03]},
           {'id':'carry-right-forward','arm':'right','hold':True,'delta_m':[.03,0,0]},
           {'id':'carry-left-up','arm':'left','hold':True,'delta_m':[0,0,.03]},
           {'id':'carry-right-yaw','arm':'right','hold':True,'delta_m':[0,0,0],'yaw_rad':.17453292519943295},
           {'id':'carry-right-table-collision','arm':'right','hold':True,'delta_m':[0,0,-.15],'expect_rejection':True}]
    if args.case:
        cases=[c for c in cases if c['id']==args.case]
        if not cases:raise ValueError('unknown declared case')
    protocol={'cases':cases,'episode':0,'seed':0,'budget_seconds':120,
              'robot_physics':'unchanged native model; free bottle, no weld, no pose edits after reset',
              'planning':'actual official cuRobo V2 worker, CUDA graph enabled; no fallback',
              'reset':'once before each independent case, never within case','scope':'fixed engineering development cases',
              'hardware_access':False,'transfer_ready':False}
    protocol['checker_sha256']=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=[]
    for case in cases:
        env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
        folder=output/case['id'];folder.mkdir();planner=None;before=None;failure=None;commands=0;started=time.monotonic()
        try:
            if case['hold']:
                execute(env,TampPlanner(env,'pick_bottle',parameters={'arm':case['arm']}),folder/'episodes')
            before=measurement(env);other='left' if case['arm']=='right' else 'right'
            before_other=env.last_action[:8] if other=='left' else env.last_action[8:]
            before_other=before_other.copy()
            before_gripper=env.last_action[7 if case['arm']=='left' else 15]
            before_tick=env.tick
            parameters={'arm':case['arm'],'frame':'world','delta_m':case['delta_m'],'planner':'curobo'}
            task='move_relative'
            if 'yaw_rad' in case:
                import mujoco
                angle=case['yaw_rad'];c,s=np.cos(angle),np.sin(angle)
                rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])@np.array(before['tcp'][case['arm']]['rotation'])
                q=np.empty(4);mujoco.mju_mat2Quat(q,rotation.ravel())
                parameters={'arm':case['arm'],'frame':'world','position_m':before['tcp'][case['arm']]['position_m'],
                            'quaternion_xyzw':np.r_[q[1:],q[0]].tolist(),'planner':'curobo'}
                task='move_to_pose'
            planner=CuRoboPlanner(env,task,parameters=parameters)
            commands=execute(env,planner,folder/'episodes')
            if case.get('expect_rejection'):raise AssertionError('collision goal was accepted')
            other_action=env.last_action[:8] if other=='left' else env.last_action[8:]
            np.testing.assert_array_equal(other_action,before_other)
            assert env.last_action[7 if case['arm']=='left' else 15]==before_gripper
            assert not case['hold'] or env.evaluation['held'][case['arm']]
            status='passed'
        except Exception as exc:
            failure={'code':getattr(exc,'code','execution_error'),'message':str(exc)}
            commands=(planner.result.get('executed_commands',0) if planner else 0)
            expected=case.get('expect_rejection') and failure['code'] in {'curobo_no_path','native_collision_rejected'} and commands==0 and env.tick==before_tick
            status='rejected_as_expected' if expected else 'failed'
        result={'case':case,'status':status,'failure':failure,'commands':commands,'before':before,'after':measurement(env),
                'sim_seconds':float(env.data.time),'wall_seconds':time.monotonic()-started,
                'feedback':planner.result if planner else {},'evaluation':env.evaluation,
                'hardware_access':False,'transfer_ready':False}
        (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
        Image.fromarray(env.render('operator')).save(folder/'result.jpg')
        env.close();results.append(result)
        print(json.dumps({'id':case['id'],'status':status,'commands':commands,'failure':failure},ensure_ascii=False),flush=True)
    (output/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str)+'\n')


if __name__=='__main__':main()
