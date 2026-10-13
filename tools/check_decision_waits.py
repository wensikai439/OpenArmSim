"""Predeclared paired native tasks with injected waits, never a learned policy."""
import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from sim.environment import ROOT, OpenArmEnv
from sim.decision_timing import DecisionWait
from sim.recording import Recorder
from sim.tamp import TampPlanner

OUT=ROOT/'runs/20261008-decision/native'


def main():
    if OUT.exists():raise ValueError('keep prior receipts; choose a new study directory')
    OUT.mkdir()
    conditions=[{'id':'paused','delay_seconds':0.}, {'id':'wait-2s','delay_seconds':2.}, {'id':'wait-5s','delay_seconds':5.}]
    protocol={'conditions':conditions,'episode':0,'seed':0,'budget_seconds':120,'offset_xy_m':[.08,0],
              'waits':'before pick, between pick/place and after place; exact previous target',
              'scope':'controlled injected latency, native engineering comparison; no API or real robot',
              'reset':'once before each independent condition only','transfer_ready':False}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    results=[]
    for case in conditions:
        env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
        folder=OUT/case['id'];folder.mkdir();root=uuid.uuid4().hex
        target=(env.data.xpos[env.bottle_id,:2]+[.08,0]).tolist();records=[];samples=[];error=None
        try:
            for i,phase in enumerate(('pick_bottle','place_bottle','final')):
                clock=[0.]
                wait=DecisionWait(env,root,i+1,'native:'+str(env.tick),clock=lambda:clock[0],output=folder/'episodes')
                clock[0]=case['delay_seconds'];receipt=wait.finish();records.append(receipt)
                if receipt['task_failed']:raise RuntimeError(receipt['status'])
                if phase=='final':break
                parameters={'arm':'right'} if phase=='pick_bottle' else {'arm':'right','receptacle_id':'table','target_xy_m':target}
                recorder=Recorder(env,'tamp',root=folder/'episodes',record_images=False,seed=0,
                    details={'study':protocol,'condition':case,'task':phase,'parameters':parameters})
                planner=None;status='error'
                try:
                    planner=TampPlanner(env,phase,parameters=parameters);iterator=planner.actions()
                    while True:
                        q=env.data.qpos.copy()
                        try:action,diagnostic=next(iterator)
                        except StopIteration:break
                        assert np.array_equal(q,env.data.qpos),'planner edited live state'
                        before=recorder.observation(env);_,_,terminal,truncated,info=env.step_tamp(action)
                        recorder.append(before,action,recorder.observation(env),info,terminated=terminal,truncated=truncated,diagnostics=diagnostic)
                        samples.append({'sim_time':float(env.data.time),'bottle_x_m':float(env.data.xpos[env.bottle_id,0])})
                        if terminal or truncated:raise RuntimeError('native terminal')
                    assert planner.result['primitive_success'];status='primitive_completed'
                finally:
                    recorder.finish(status,env.info(),metrics={'tamp':planner.result if planner else {}})
                    records.append({'run_path':str(recorder.path.relative_to(ROOT)),'task':phase,'status':status})
            assert env.evaluation['placement']['confirmed'] and not env.evaluation['invalid']
            status='passed'
        except Exception as exc:status='failed';error={'code':getattr(exc,'code','execution_error'),'message':str(exc)}
        result={'condition':case,'status':status,'failure':error,'target_xy_m':target,
                'position_error_m':float(np.linalg.norm(env.data.xpos[env.bottle_id,:2]-target)),
                'sim_seconds':float(env.data.time),'evaluation':env.evaluation,'records':records,'samples':samples,
                'hardware_access':False,'transfer_ready':False}
        (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
        Image.fromarray(env.render('operator')).save(folder/'result.jpg')
        env.close();results.append(result)
        print(json.dumps({'case':case['id'],'status':status,'sim_seconds':result['sim_seconds'],'error':error}),flush=True)
    (OUT/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str)+'\n')


if __name__=='__main__':main()
