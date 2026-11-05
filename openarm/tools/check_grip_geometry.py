"""Declared grasp/receiver geometry comparison with identical physical model."""
import hashlib
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sim.environment import ROOT,OpenArmEnv
from sim.tamp import TampPlanner
from sim.recording import Recorder,dump
from sim.decision_timing import GraspStability
from sim.planning_timing import PlanningTiming
from sim.relocation import RelocationPlanner
from tools.check_contact_mechanics import contact_mechanics

OUT=ROOT/'runs/20261008-grip'
CASES={'direct-right':{'held_by':'right','skills':[('pick_bottle',{'arm':'right','object_id':'bottle'})]},
       'direct-left':{'held_by':'left','skills':[('pick_bottle',{'arm':'left','object_id':'bottle'})]},
       'receiver-right':{'held_by':'right','skills':[('pick_bottle',{'arm':'left','object_id':'bottle'}),
           ('handover_bottle',{'from_arm':'left','to_arm':'right','object_id':'bottle'})]}}


def main():
    OUT.mkdir(exist_ok=False)
    protocol={'schema':'grip-geometry/1','cases':CASES,'seed':0,'episode':0,'task_budget_s':120,
        'hold_seconds':10,'timing':'paused native planning for this geometry diagnostic; hold integrates ordinary 30Hz targets/1ms physics',
        'hold_target':'exact last applied native action; no extra closure, settling omitted or corrective motion',
        'limits':{'rolling_window_s':2,'relative_position_m':.001,'relative_rotation_rad':.02},
        'physics_changed':False,'controller_changed':False,'score_changed':False,
        'scope':'three declared development grasps/one seed, not a generalization or hardware test',
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'diagnostic_sha256':hashlib.sha256((ROOT/'tools/check_contact_mechanics.py').read_bytes()).hexdigest(),
        'main_page_commands':0,'hardware_access':False,'transfer_ready':False}
    dump(OUT/'protocol.json',protocol);results=[]
    for name,case in CASES.items():
        folder=OUT/name;folder.mkdir();env=OpenArmEnv(images=False)
        env.reset(seed=0);env.set_task_budget(120);rec=None;rows=[];audit=None;failure=None
        status='incomplete';plans=[];started=time.monotonic()
        try:
            for index,(task,parameters) in enumerate(case['skills']):
                planner=TampPlanner(env,task,parameters=parameters);iterator=planner.actions()
                rec=Recorder(env,'tamp',root=folder/'episodes',record_images=False,seed=0,
                    details={'study':'grip geometry','task':task,'parameters':parameters,'native_planning_timing':'paused'})
                try:
                    while True:
                        previous=rec.observation(env);q=env.data.qpos.copy()
                        try:action,diag=next(iterator)
                        except StopIteration:break
                        assert np.array_equal(q,env.data.qpos),'planner edited live qpos'
                        _,_,term,trunc,info=env.step_tamp(action)
                        rec.append(previous,action,rec.observation(env),info,terminated=term,truncated=trunc,diagnostics=diag)
                        if term or trunc:raise ValueError('native terminal during grasp preparation')
                    if not planner.result.get('primitive_success'):raise ValueError('missing native grasp/receiver evidence')
                    rec.finish('primitive_completed',env.info(),metrics={'tamp':planner.result})
                    plans.append({'run':str(rec.path.relative_to(ROOT)),'task':task,'feedback':planner.result})
                except Exception as exc:
                    rec.finish('error',env.info(),error=str(exc),metrics={'tamp':planner.result});raise
            side=case['held_by'];assert env.evaluation['held'][side]
            target=env.last_action.copy();audit=GraspStability(env);first=contact_mechanics(env,side)
            Image.fromarray(env.render('overview')).save(folder/'start.jpg',quality=92)
            rows.append({**first,'elapsed_s':0.,'quality':env.grasp_feedback.summary()['arms'][side]})
            rec=Recorder(env,'wait',root=folder/'episodes',record_images=False,seed=0,
                details={'study':'grip geometry hold','case':name,'hold_target':target.tolist(),'limits':protocol['limits']})
            for _ in range(300):
                before=rec.observation(env);_,_,term,trunc,info=env.step_tamp(target);audit.read()
                row={**contact_mechanics(env,side),'elapsed_s':float(env.data.time)-first['sim_seconds'],
                    'quality':env.grasp_feedback.summary()['arms'][side]}
                rows.append(row)
                rec.append(before,target,rec.observation(env),info,terminated=term,truncated=trunc,
                    diagnostics={'grasp_quality':row['quality'],'diagnostic_arm':side})
                if term or trunc:status='physics_invalid' if info['evaluation']['invalid'] else 'truncated';break
            else:status='completed'
            rec.finish(status,env.info(),metrics={'grasp_stability':audit.summary()})
        except Exception as exc:failure={'code':getattr(exc,'code','execution_error'),'message':str(exc)};status='error'
        finally:
            if rec and not rec._closed:rec.finish(status,env.info(),error=str(failure) if failure else None)
            Image.fromarray(env.render('overview')).save(folder/'final.jpg',quality=92)
            assessment=audit.summary()['arms'][case['held_by']] if audit else None
            result={'case':name,'status':status,'failure':failure,'arm':case['held_by'],'grasp_plans':plans,
                'stability':assessment,'passed':bool(status=='completed' and assessment['status']=='stable_within_audit_window'),
                'actual_hold_s':rows[-1]['elapsed_s'] if rows else 0,'last_quality':rows[-1]['quality'] if rows else None,
                'hold_run':str(rec.path.relative_to(ROOT)) if rec else None,
                'wall_s':time.monotonic()-started,'model_evidence':env.manifest()['model_evidence']}
            dump(folder/'series.json',rows);dump(folder/'result.json',result)
            results.append({k:v for k,v in result.items() if k!='model_evidence'})
            dump(OUT/'summary.json',{'protocol':'protocol.json','cases':results,'complete':len(results)==len(CASES)})
            print(json.dumps({k:result[k] for k in ('case','status','passed','actual_hold_s','stability','failure')},ensure_ascii=False),flush=True)
            env.close()


def numerics(torsion=False):
    """Numerical stiction hypothesis applied before acquisition, isolated only."""
    out=OUT/('torsion' if torsion else 'numerics');out.mkdir(exist_ok=False)
    values=[0,3,20] if torsion else [0,1,3,20]
    protocol={'schema':'grip-torsion/1' if torsion else 'grip-numerics/1','noslip_iterations':values,
        'change':('contact dimension4 on bottle and right fingers plus declared NoSlip cap; before acquisition' if torsion else
                  'only NoSlip solver iteration cap, before any grasp action'),
        'contact_dimension':4 if torsion else 'original3',
        'contact_model_calibrated':False,
        'initial_qpos_qvel':'identical reset state in all four cases; no pose edit',
        'grasp':'current native right pick replans under each explicitly changed numerical model',
        'hold_seconds':10,'seed':0,'episode':0,'task_budget_s':120,
        'native_planning_timing':'paused; diagnostic numerical comparison, not latency-aware API score',
        'limits':{'rolling_window_s':2,'relative_position_m':.001,'relative_rotation_rad':.02},
        'physical_materials_gains_force_limits_and_acceptance_changed':False,
        'production_model_changed':False,'automatically_applied':False,
        'source':'https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip',
        'recording':'explicit numerical counterfactual; ordinary Recorder contract is not bypassed or weakened',
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'main_page_commands':0,'hardware_access':False,'transfer_ready':False}
    dump(out/'protocol.json',protocol);results=[];initial=None
    for iterations in protocol['noslip_iterations']:
        folder=out/str(iterations);folder.mkdir();env=OpenArmEnv(images=False)
        env.reset(seed=0);env.set_task_budget(120);baseline=env.manifest()
        state=(env.data.qpos.copy(),env.data.qvel.copy())
        if initial is None:initial=state
        assert all(np.array_equal(a,b) for a,b in zip(initial,state))
        env.model.opt.noslip_iterations=iterations
        if torsion:
            for geom in range(env.model.ngeom):
                body=int(env.model.geom_bodyid[geom]);name=env.model.body(body).name
                if body==env.bottle_id or name.startswith('openarm_right_') and 'finger' in name:
                    env.model.geom_condim[geom]=4
        assert all(np.array_equal(a,b) for a,b in zip(state,(env.data.qpos,env.data.qvel)))
        dump(folder/'manifest.json',{'baseline_model':baseline,'numerical_model':env.manifest(),
            'kind':'research_numerical_contact_counterfactual','details':protocol,'noslip_iterations':iterations,
            'initial_robot_state':env.state().tolist(),'initial_qpos':state[0].tolist(),'initial_qvel':state[1].tolist()})
        rows=[];audit=None;planner=None;failure=None;status='error';started=time.monotonic()
        try:
            planner=TampPlanner(env,'pick_bottle',parameters={'arm':'right','object_id':'bottle'})
            with (folder/'transitions.jsonl').open('w') as stream:
                def step(action,diagnostic):
                    previous={'state':env.state().tolist(),'tick':env.tick,'sim_time':float(env.data.time)}
                    _,_,term,trunc,info=env.step_tamp(action)
                    stream.write(json.dumps({'observation':previous,'action':action.tolist(),
                        'next_observation':{'state':env.state().tolist(),'tick':env.tick,'sim_time':float(env.data.time)},
                        'terminated':bool(term),'truncated':bool(trunc),'evaluation':info['evaluation'],
                        'diagnostics':diagnostic})+'\n')
                    return term,trunc,info
                for action,diag in planner.actions():
                    term,trunc,info=step(action,diag)
                    if term or trunc:raise ValueError('native terminal during numerical grasp')
                assert planner.result['primitive_success'] and env.evaluation['held']['right']
                target=env.last_action.copy();audit=GraspStability(env)
                first=contact_mechanics(env);rows.append({**first,'elapsed_s':0.,'quality':env.grasp_feedback.summary()['arms']['right']})
                Image.fromarray(env.render('overview')).save(folder/'start.jpg',quality=92)
                for _ in range(300):
                    term,trunc,info=step(target,{'phase':'hold','exact_prior_target':True});audit.read()
                    row={**contact_mechanics(env),'elapsed_s':float(env.data.time)-first['sim_seconds'],
                        'quality':env.grasp_feedback.summary()['arms']['right']};rows.append(row)
                    if term or trunc:status='physics_invalid' if info['evaluation']['invalid'] else 'truncated';break
                else:status='completed'
        except Exception as exc:failure={'code':getattr(exc,'code','execution_error'),'message':str(exc)}
        finally:
            assessment=audit.summary()['arms']['right'] if audit else None
            result={'noslip_iterations':iterations,'status':status,'failure':failure,'same_initial_state':True,
                'stability':assessment,'passed':bool(status=='completed' and assessment['status']=='stable_within_audit_window'),
                'actual_hold_s':rows[-1]['elapsed_s'] if rows else 0.,'last_quality':rows[-1]['quality'] if rows else None,
                'grasp_feedback':planner.result if planner else None,
                'runtime_parameter_changes':env.manifest()['model_evidence']['runtime_parameters']['changed_fields'],
                'wall_s':time.monotonic()-started,'automatically_applied':False}
            Image.fromarray(env.render('overview')).save(folder/'final.jpg',quality=92)
            dump(folder/'series.json',rows);dump(folder/'result.json',result)
            results.append(result);dump(out/'summary.json',{'cases':results,'complete':len(results)==len(values)})
            print(json.dumps({k:result[k] for k in ('noslip_iterations','status','passed','stability','failure','actual_hold_s')},ensure_ascii=False),flush=True)
            env.close()


def validation():
    """Selected research contact candidate; new declared states/tasks only."""
    import mujoco
    out=OUT/'validation';out.mkdir(exist_ok=False)
    cases=[{'name':'right-long','episode':0,'initial_delta_m':[0,0,0],'arm':'right','hold_s':60},
           {'name':'left-episode1','episode':1,'initial_delta_m':[0,0,0],'arm':'left','hold_s':20},
           {'name':'right-episode2-move','episode':2,'initial_delta_m':[-.015,.02,0],
            'parameters':{'arm':'right','object_id':'bottle','receptacle_id':'table','offset_xy_m':[.08,0],
                          'transport_planner':'curobo','transport_delta_m':[.015,0,0]}},
           {'name':'left-to-right-move','episode':0,'initial_delta_m':[.015,-.02,0],
            'parameters':{'arm':'left','via_arm':'right','object_id':'bottle','receptacle_id':'table','offset_xy_m':[.06,0],
                          'transport_planner':'curobo','transport_delta_m':[.015,0,0]}}]
    protocol={'schema':'contact-candidate-validation/1','cases':cases,'seed':0,'task_budget_s':120,
        'candidate':{'noslip_iterations':3,'contact_dimension':4,'applies_to':'bottle and both finger bodies; contact coefficients remain original'},
        'candidate_selection_source':'torsion/summary.json','selection_source_sha256':hashlib.sha256((OUT/'torsion/summary.json').read_bytes()).hexdigest(),
        'hold_timing':'paused planning then original 30Hz hold','relocation_timing':'measured_delay including native planning waits',
        'initial_pose_edits':'only declared bottle translation before ANY task/physics action; no resets or pose edits within task',
        'limits':{'rolling_window_s':2,'relative_position_m':.001,'relative_rotation_rad':.02},
        'independent_physical_calibration':False,'candidate_deployed':False,'hardware_access':False,
        'main_page_commands':0,'transfer_ready':False,
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(out/'protocol.json',protocol);results=[]
    for case in cases:
        folder=out/case['name'];folder.mkdir();env=OpenArmEnv(images=False)
        env.reset(seed=0,options={'episode':case['episode']});env.set_task_budget(120);baseline=env.manifest()
        env.model.opt.noslip_iterations=3
        for g in range(env.model.ngeom):
            body=int(env.model.geom_bodyid[g]);name=env.model.body(body).name
            if body==env.bottle_id or name.startswith('openarm_') and 'finger' in name:env.model.geom_condim[g]=4
        adr=env.model.joint('bottle_free').qposadr[0]
        env.data.qpos[adr:adr+3]+=case['initial_delta_m']
        mujoco.mj_forward(env.model,env.data);env.evaluation=env._evaluate(0.)
        initial={'robot_state':env.state().tolist(),'bottle_position_m':env.data.xpos[env.bottle_id].tolist(),
                 'qpos':env.data.qpos.tolist(),'qvel':env.data.qvel.tolist()}
        dump(folder/'manifest.json',{'baseline_model':baseline,'candidate_model':env.manifest(),
            'case':case,'protocol':protocol,'initial_state':initial,'kind':'research_contact_candidate'})
        Image.fromarray(env.render('overview')).save(folder/'start.jpg',quality=92)
        rows=[];audit=None;planner=None;clock=None;failure=None;passed=False;status='error';started=time.monotonic()
        events=[]
        try:
            planner=(RelocationPlanner(env,'relocate_object',parameters=case['parameters']) if 'parameters' in case else
                     TampPlanner(env,'pick_bottle',parameters={'arm':case['arm'],'object_id':'bottle'}))
            with (folder/'transitions.jsonl').open('w') as stream:
                def step(action,diagnostic,is_hold=False):
                    before={'state':env.state().tolist(),'tick':env.tick,'sim_time':float(env.data.time)}
                    if is_hold:_,_,term,trunc,info=env.hold_tamp(action,diagnostic['planning_timing']['duration_s'])
                    else:_,_,term,trunc,info=env.step_tamp(action)
                    after={'state':env.state().tolist(),'tick':env.tick,'sim_time':float(env.data.time)}
                    stream.write(json.dumps({'observation':before,'action':action.tolist(),'next_observation':after,
                        'terminated':bool(term),'truncated':bool(trunc),'evaluation':info['evaluation'],'diagnostics':diagnostic})+'\n')
                    if clock:clock.committed(is_hold,before['sim_time'],after['sim_time'])
                    return term,trunc,info
                source=planner.actions()
                if 'parameters' in case:
                    clock=PlanningTiming('measured_delay');actions=clock.actions(env,source,events.append)
                else:actions=((a,d,False) for a,d in source)
                for action,diag,is_hold in actions:
                    term,trunc,info=step(action,diag,is_hold)
                    if term or trunc:raise ValueError('native terminal or declared task budget')
                assert planner.result['primitive_success']
                if 'parameters' in case:
                    passed=bool(env.evaluation['placement']['confirmed']);status='completed'
                else:
                    side=case['arm'];target=env.last_action.copy();audit=GraspStability(env);first=contact_mechanics(env,side)
                    rows.append({**first,'elapsed_s':0.,'quality':env.grasp_feedback.summary()['arms'][side]})
                    for _ in range(case['hold_s']*30):
                        term,trunc,info=step(target,{'phase':'hold','exact_prior_target':True});audit.read()
                        rows.append({**contact_mechanics(env,side),'elapsed_s':float(env.data.time)-first['sim_seconds'],
                                     'quality':env.grasp_feedback.summary()['arms'][side]})
                        if term or trunc:status='physics_invalid' if info['evaluation']['invalid'] else 'truncated';break
                    else:status='completed'
                    passed=status=='completed' and audit.summary()['arms'][side]['status']=='stable_within_audit_window'
        except Exception as exc:failure={'code':getattr(exc,'code','execution_error'),'message':str(exc)}
        finally:
            if 'source' in locals():
                try:source.close()
                except Exception:pass
            result={'case':case,'status':status,'passed':bool(passed),'failure':failure,
                'native_feedback':planner.result if planner else None,'planning_timing':clock.summary() if clock else None,
                'stability':audit.summary() if audit else None,'actual_hold_s':rows[-1]['elapsed_s'] if rows else None,
                'sim_time':float(env.data.time),'wall_s':time.monotonic()-started,
                'evaluation':env.evaluation,'runtime_parameter_changes':env.manifest()['model_evidence']['runtime_parameters']['changed_fields'],
                'candidate_deployed':False,'hardware_access':False,'transfer_ready':False}
            dump(folder/'result.json',result);dump(folder/'series.json',rows);dump(folder/'events.json',events)
            Image.fromarray(env.render('overview')).save(folder/'final.jpg',quality=92)
            results.append(result);dump(out/'summary.json',{'cases':results,'complete':len(results)==len(cases)})
            print(json.dumps({k:result[k] for k in ('case','status','passed','failure','actual_hold_s','sim_time')},ensure_ascii=False),flush=True)
            env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['geometry','numerics','torsion','validation'],default='geometry')
    args=parser.parse_args()
    if args.mode=='geometry':main()
    elif args.mode=='validation':validation()
    else:numerics(torsion=args.mode=='torsion')
