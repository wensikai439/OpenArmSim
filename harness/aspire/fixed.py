"""Honest nonadaptive control: replay one successful nominal command sequence.

Validate nominal reproduction before comparing new positions. No IK, pose
query or planner runs during execution; task scoring remains host-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .runtime import fixture_environment, dump, validate_case, ROOT
from sim.scene_library import LibraryEnv
from sim.recording import Recorder


def load_sequence(nominal):
    report=json.loads((nominal/'result.json').read_text())
    if report['task_success'] is not True:raise ValueError('nominal source must actually succeed')
    actions=[];stops=[];source=[];last_state=None;model=None
    for stage in report['stages']:
        path=nominal/stage['native_run']
        manifest=json.loads((path/'manifest.json').read_text())
        compiled=manifest['model_evidence']['compiled']['sha256']
        if model is None:model=compiled
        if model!=compiled:raise ValueError('nominal sequence mixed different models')
        data=(path/'transitions.jsonl').read_bytes()
        for line in data.splitlines():
            row=json.loads(line);actions.append(row['action']);last_state=row['next_observation']['state']
        stops.append({'tool':stage['tool'],'tick':len(actions)})
        source.append({'native_run':stage['native_run'],'transitions_sha256':hashlib.sha256(data).hexdigest()})
    return np.asarray(actions,np.float32),stops,source,last_state,model


def run(nominal, case, output):
    validate_case(case);output.mkdir(parents=True,exist_ok=False)
    actions,stops,sources,reference_end,model=load_sequence(nominal)
    env=fixture_environment(LibraryEnv,case)(images=False,selection=[])
    recorder=None
    result={'case':case,'task_success':False,'method':'fixed nominal actuator-command replay',
            'replanning':False,'uses_current_bottle_pose_for_control':False,
            'source_nominal':str(nominal),'sources':sources,'stages':[],
            'hardware_access':False,'transfer_ready':False}
    try:
        env.reset(seed=case['seed']);env.set_task_budget(120);env.task_termination='skill'
        if env.manifest()['model_evidence']['compiled']['sha256']!=model:
            raise ValueError('fixed and adaptive runs compiled different physical models')
        # Declare the same stationary basket goal once, never relocate it after failure.
        env.set_placement_target('basket',env.data.xpos[env.box_id,:2])
        recorder=Recorder(env,'fixed-replay',record_images=False,seed=case['seed'],root=output/'native',
            details={'nonadaptive':True,'source_nominal':str(nominal),'case':case})
        Image.fromarray(env.render('overview')).save(output/'initial.jpg')
        before=recorder.observation(env)
        stop_by_tick={s['tick']:s['tool'] for s in stops}
        status='completed'
        for action in actions:
            _,_,terminal,truncated,info=env.step_tamp(action)
            after=recorder.observation(env)
            recorder.append(before,action,after,info,terminated=terminal,truncated=truncated)
            before=after
            if env.tick in stop_by_tick:
                tool=stop_by_tick[env.tick];e=env.evaluation
                lifted=env._bottle_bounds()[0][2]>1.03
                ok=(e['held']['right'] and lifted if tool=='pick' else
                    e['held']['left'] and not e['held']['right'] and lifted if tool=='handover' else
                    e.get('inside_basket') and e.get('supported_by_basket')
                    and e.get('placement',{}).get('confirmed') and not any(e['held'].values()))
                result['stages'].append({'tool':tool,'ok':bool(ok),'tick':env.tick})
            if terminal or truncated:
                status='invalid' if info['evaluation']['invalid'] else 'truncated';break
        e=env.evaluation
        result.update(steps=env.tick,sim_seconds=float(env.data.time),evaluation=e,
            completed_all_commands=env.tick==len(actions),
            final_robot_state_error_to_nominal_rad=float(np.max(np.abs(env.state()-reference_end))))
        result['task_success']=bool(len(result['stages'])==3 and all(s['ok'] for s in result['stages']) and not e['invalid'])
        recorder.finish(status,env.info(),metrics={'fixed_replay':result})
        Image.fromarray(env.render('overview')).save(output/'final.jpg')
    except Exception as exc:
        result['error']=str(exc)
        if recorder:recorder.finish('error',env.info(),error=str(exc))
    finally:
        env.close();dump(output/'result.json',result)
    print(json.dumps({'fixed_success':result['task_success'],'steps':result.get('steps'),'error':result.get('error')}),flush=True)
    return 0 if result['task_success'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nominal',type=Path,required=True)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'runs'):parser.error('output must stay inside runs/')
    raise SystemExit(run(args.nominal,json.loads(args.case.read_text()),output))
