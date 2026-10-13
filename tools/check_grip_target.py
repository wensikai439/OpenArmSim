"""Compare existing-range motor targets after the same native grasp.

Command targets may differ; mass, friction, gains, solver and acceptance do not.
The 0.07 rad case is in the existing oracle TAMP range, outside the recorded
policy command maximum. It must never be promoted to a learned-policy limit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sim.environment import ROOT, OpenArmEnv
from sim.tamp import TampPlanner
from sim.recording import Recorder, dump
from sim.decision_timing import GraspStability
from tools.check_contact_mechanics import contact_mechanics


OUT=ROOT/'runs/20261008-hold'
CASES=[('recorded-close',.05),('native-close',.07),('less-close',.03)]


def main():
    OUT.mkdir(exist_ok=False)
    fixture=OUT/'grasp';fixture.mkdir()
    env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
    planner=TampPlanner(env,'pick_bottle',parameters={'arm':'right'})
    rec=Recorder(env,'tamp',root=fixture,record_images=False,seed=0,
        details={'study':'same-grasp motor-target contrast','parameters':{'arm':'right'}})
    actions=[];states=[];state=None;status='incomplete';error=None
    try:
        iterator=planner.actions()
        while True:
            q=env.data.qpos.copy()
            try:action,diag=next(iterator)
            except StopIteration:break
            assert np.array_equal(q,env.data.qpos)
            before=rec.observation(env)
            _,_,term,trunc,info=env.step_tamp(action)
            rec.append(before,action,rec.observation(env),info,terminated=term,truncated=trunc,diagnostics=diag)
            actions.append(action.copy());states.append(env.state().copy())
            if term or trunc:raise RuntimeError('fixture terminated')
        if not planner.result.get('primitive_success'):raise RuntimeError('fixture not confirmed')
        state=(env.data.qpos.copy(),env.data.qvel.copy());status='primitive_completed'
        manifest=env.manifest();hold_target=env.last_action.copy()
    except Exception as exc:error=str(exc);raise
    finally:
        rec.finish(status,env.info(),error=error,metrics={'tamp':planner.result});env.close()
    protocol={'schema':'grip-command-study/1','cases':[{'name':n,'motor_target_rad':v} for n,v in CASES],
        'grasp_run':str(rec.path.relative_to(ROOT)),'episode':0,'seed':0,'fixture_commands':len(actions),
        'hold_seconds':10,'ramp_seconds':.3,'audit_start':'before ramp, no omitted settling samples',
        'limits':{'rolling_window_seconds':2,'relative_position_m':.001,'relative_rotation_rad':.02},
        'unchanged':'mass, inertia, friction, solver, actuator gains/force limits, arm targets and native acceptance',
        'native_maximum':'0.07 rad already belongs to full simulated joint range; recorded/VLA max remains 0.05',
        'model_evidence':manifest['model_evidence'],'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selection':'all three predeclared targets; no deployment, threshold tuning or held-out success-rate claim',
        'hardware_access':False,'transfer_ready':False}
    dump(OUT/'protocol.json',protocol);results=[]
    for name,motor in CASES:
        folder=OUT/name;folder.mkdir()
        env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
        wait=None;status='incomplete';error=None;rows=[]
        try:
            for action,expected in zip(actions,states):
                env.step_tamp(action)
                if np.max(np.abs(env.state()-expected))>1e-9:raise ValueError('fixture reproduction differs')
            if not np.array_equal(env.data.qpos,state[0]) or not np.array_equal(env.data.qvel,state[1]):
                raise ValueError('full initial state differs')
            if env.manifest()['model_evidence']!=protocol['model_evidence']:raise ValueError('model differs')
            audit=GraspStability(env);rows.append(contact_mechanics(env))
            wait=Recorder(env,'wait',root=folder,record_images=False,seed=0,
                details={'study':'grip-command contrast','case':name,'motor_target_rad':motor,
                    'scope':'native oracle-only command study, not a physical force calibration'})
            for i in range(300):
                action=hold_target.copy()
                action[15]=hold_target[15]+(motor-hold_target[15])*min(1,(i+1)/9)
                before=wait.observation(env)
                _,_,term,trunc,info=env.step_tamp(action)
                audit.read();rows.append(contact_mechanics(env))
                wait.append(before,action,wait.observation(env),info,terminated=term,truncated=trunc,
                    diagnostics={'command_case':name,'grasp_stability':audit.summary()})
                if term or trunc:
                    status='physics_invalid' if info['evaluation']['invalid'] else 'truncated';break
            else:status='completed'
            summary=audit.summary();finger=[c for r in rows for c in r['contacts'] if 'finger' in c['other_body']]
            result={'case':name,'motor_target_rad':motor,'status':status,'error':error,
                'same_initial_qpos_qvel':True,'model_evidence':env.manifest()['model_evidence'],
                'grasp_stability':summary,'stable':summary['arms']['right']['status']=='stable_within_audit_window',
                'actual_seconds':rows[-1]['sim_seconds']-rows[0]['sim_seconds'],
                'finger_normal_N_p5_p50_p95':np.percentile([c['normal_N'] for c in finger],[5,50,95]).tolist(),
                'finger_actuator_abs_N_p5_p50_p95':np.percentile([abs(r['finger_actuator_force_N']) for r in rows],[5,50,95]).tolist(),
                'target_changed_parameters':False,'native_hold_run':wait.path.name}
            image='final.jpg';Image.fromarray(env.render('overview')).save(folder/image,quality=92)
            dump(folder/'series.json',rows);dump(folder/'result.json',result);results.append(result)
            print(json.dumps({k:result[k] for k in ('case','motor_target_rad','status','stable','actual_seconds','grasp_stability')},ensure_ascii=False),flush=True)
        except Exception as exc:error=str(exc);raise
        finally:
            if wait:wait.finish(status,env.info(),error=error,metrics={'grasp_stability':audit.summary()})
            env.close()
    dump(OUT/'summary.json',results)


if __name__=='__main__':main()
