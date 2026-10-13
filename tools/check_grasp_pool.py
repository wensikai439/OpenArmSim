"""Declared native grasp-pool cases; all failures/holds are retained."""
import hashlib
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from sim.environment import ROOT
from sim.grasp import GeometricPlanner,CONTRACT
from sim.recording import Recorder,dump
from sim.decision_timing import GraspStability
from tools.contact_candidate import ContactCandidate

OUT=ROOT/'runs/20261008-ik/native'
POSITIONS={'original':[0,0,0],'reported-failure':[.015,-.02,0],
           'inward':[-.02,.02,0],'outward':[.025,-.03,0],'sideways':[0,.04,0]}


def main():
    OUT.mkdir(exist_ok=False)
    cases=[{'name':side+'-'+name,'arm':side,'delta_m':delta} for name,delta in POSITIONS.items() for side in ('left','right')]
    protocol={'schema':'native-grasp-pool/1','cases':cases,'seed':0,'episode':0,'hold_seconds':10,
        'task_budget_s':120,'planning_timing':'paused for geometry diagnostic; all actuator/hold physics integrated',
        'contact_model':'same cold finite-patch candidate,dim4/noslip3','grasp_sampler':CONTRACT,
        'initial_scene':'only declared bottle translation before any native action; no pose edits/reset within task',
        'original_stability_limits':{'window_seconds':2,'relative_position_m':.001,'relative_rotation_rad':.02},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sampler_sha256':hashlib.sha256((ROOT/'sim/grasp.py').read_bytes()).hexdigest(),
        'scope':'predefined development cases, not a statistical generalization or real transfer test',
        'main_page_commands':0,'hardware_access':False,'transfer_ready':False}
    dump(OUT/'protocol.json',protocol);results=[]
    for case in cases:
        folder=OUT/case['name'];folder.mkdir();env=ContactCandidate(images=False);rec=None;planner=None;audit=None
        env.reset(seed=0);env.set_task_budget(120)
        addr=env.model.joint('bottle_free').qposadr[0];env.data.qpos[addr:addr+3]+=case['delta_m']
        mujoco.mj_forward(env.model,env.data);env.evaluation=env._evaluate(0.)
        dump(folder/'initial.json',{'case':case,'manifest':env.manifest(),'qpos':env.data.qpos.tolist(),
            'qvel':env.data.qvel.tolist(),'bottle_position_m':env.data.xpos[env.bottle_id].tolist()})
        Image.fromarray(env.render('overview')).save(folder/'start.jpg',quality=90)
        failure=None;status='error';started=time.monotonic();held_s=0.;quality=[];grasp_run=None
        try:
            planner=GeometricPlanner(env,'pick_bottle',parameters={'arm':case['arm'],'object_id':'bottle'})
            rec=Recorder(env,'tamp',root=folder/'episodes',record_images=False,seed=0,
                details={'purpose':'native geometric grasp diagnostic','case':case,'planning_timing':'paused','grasp_sampler':CONTRACT})
            generator=planner.actions()
            while True:
                previous=rec.observation(env);q=env.data.qpos.copy();v=env.data.qvel.copy()
                try:action,diagnostic=next(generator)
                except StopIteration:break
                np.testing.assert_array_equal(q,env.data.qpos);np.testing.assert_array_equal(v,env.data.qvel)
                _,_,term,trunc,info=env.step_tamp(action)
                rec.append(previous,action,rec.observation(env),info,terminated=term,truncated=trunc,diagnostics=diagnostic)
                if term or trunc:raise ValueError('native terminal during grasp')
            if not planner.result.get('primitive_success'):raise ValueError('missing native grasp/lift receipt')
            rec.finish('primitive_completed',env.info(),metrics={'tamp':planner.result});grasp_run=str(rec.path.relative_to(ROOT))
            target=env.last_action.copy();audit=GraspStability(env);start_time=float(env.data.time)
            rec=Recorder(env,'wait',root=folder/'episodes',record_images=False,seed=0,
                details={'purpose':'10s exact-prior-target hold after geometric pick','case':case})
            quality.append({'elapsed_s':0.,'feedback':env.grasp_feedback.summary()['arms'][case['arm']]})
            for _ in range(300):
                previous=rec.observation(env);_,_,term,trunc,info=env.step_tamp(target);audit.read()
                held_s=float(env.data.time)-start_time
                quality.append({'elapsed_s':held_s,'feedback':env.grasp_feedback.summary()['arms'][case['arm']]})
                rec.append(previous,target,rec.observation(env),info,terminated=term,truncated=trunc)
                if term or trunc:raise ValueError('native terminal during hold')
            status='completed';rec.finish(status,env.info(),metrics={'grasp_stability':audit.summary()})
        except Exception as exc:
            failure={'code':getattr(exc,'code','execution_error'),'message':str(exc)}
            if rec and not rec._closed:rec.finish('error',env.info(),error=str(exc),metrics={'tamp':planner.result if planner else None})
        finally:
            stability=audit.summary()['arms'][case['arm']] if audit else None
            passed=bool(status=='completed' and stability['status']=='stable_within_audit_window')
            result={'case':case,'status':status,'passed':passed,'failure':failure,
                'native_feedback':planner.result if planner else None,'grasp_run':grasp_run,
                'stability':stability,'actual_hold_s':held_s,'wall_s':time.monotonic()-started,
                'sim_time':float(env.data.time),'evaluation':env.evaluation,
                'runtime_parameter_changes':env.manifest()['model_evidence']['runtime_parameters']['changed_fields'],
                'hardware_access':False,'transfer_ready':False}
            dump(folder/'result.json',result);dump(folder/'quality.json',quality)
            Image.fromarray(env.render('overview')).save(folder/'final.jpg',quality=90)
            results.append(result);dump(OUT/'summary.json',{'complete':len(results)==len(cases),'cases':results})
            print(json.dumps({'case':case['name'],'passed':passed,'failure':failure,'hold_s':held_s,
                'candidate_index':(planner.result.get('grasp_search')or{}).get('candidate_index') if planner else None},ensure_ascii=False),flush=True)
            env.close()


if __name__=='__main__':main()
