"""Explicit numerical contact hypotheses after one identical native grasp.

Production recording rightly refuses changed solver parameters. These research
artifacts identify the change, preserve every applied command/state and retain
the original stability thresholds. No candidate is automatically deployed.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from sim.environment import ROOT,OpenArmEnv
from sim.recording import dump
from sim.decision_timing import GraspStability
from tools.check_contact_mechanics import contact_mechanics


def main():
    out=ROOT/'runs/20261008-hold/solver';out.mkdir(exist_ok=False)
    source=json.loads((out.parent/'protocol.json').read_text())
    grasp=ROOT/source['grasp_run']
    commands=[json.loads(line) for line in (grasp/'transitions.jsonl').read_text().splitlines()]
    protocol={'schema':'hold-solver-counterfactual/1','noslip_iterations':[0,1,20],
        'source_grasp':source['grasp_run'],'source_commands_sha256':hashlib.sha256((grasp/'transitions.jsonl').read_bytes()).hexdigest(),
        'hold_seconds':10,'target':'exact last applied actuator target, all cases equal',
        'limits':source['limits'],'change':'only noslip_iterations, AFTER identical initial grasp',
        'unchanged':'state, target, mass, inertia, friction, gains, force limits, contact dimensions, success thresholds',
        'kind':'explicit research numerical-contact hypothesis; ordinary recording guard is not weakened',
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'automatically_applied':False,'hardware_access':False,'transfer_ready':False}
    dump(out/'protocol.json',protocol);results=[];baseline_state=None
    for iterations in protocol['noslip_iterations']:
        folder=out/str(iterations);folder.mkdir()
        env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
        rows=[];transitions=[];status='completed';started=time.monotonic()
        try:
            for transition in commands:
                env.step_tamp(np.asarray(transition['action']))
                if np.max(np.abs(env.state()-transition['next_observation']['state']))>1e-9:
                    raise ValueError('initial grasp reproduction differs')
            q,v=env.data.qpos.copy(),env.data.qvel.copy()
            if baseline_state is None:baseline_state=(q.copy(),v.copy())
            assert np.array_equal(q,baseline_state[0]) and np.array_equal(v,baseline_state[1])
            assert env.manifest()['model_evidence']==source['model_evidence']
            env.model.opt.noslip_iterations=iterations
            assert np.array_equal(q,env.data.qpos) and np.array_equal(v,env.data.qvel)
            target=env.last_action.copy();audit=GraspStability(env);rows.append(contact_mechanics(env))
            manifest={**env.manifest(),'kind':'research_numerical_contact_counterfactual',
                'details':{**protocol,'noslip_iterations':iterations},'start_tick':env.tick,
                'hold_applied_target':target.tolist(),'initial_robot_state':env.state().tolist()}
            dump(folder/'manifest.json',manifest)
            for _ in range(300):
                before={'tick':env.tick,'state':env.state().tolist(),'sim_time':float(env.data.time)}
                _,_,term,trunc,info=env.step_tamp(target);audit.read();rows.append(contact_mechanics(env))
                transitions.append({'observation':before,'action':target.tolist(),
                    'next_observation':{'tick':env.tick,'state':env.state().tolist(),'sim_time':float(env.data.time)},
                    'terminated':bool(term),'truncated':bool(trunc),'evaluation':info['evaluation']})
                if term or trunc:
                    status='physics_invalid' if info['evaluation']['invalid'] else 'truncated';break
            result={'noslip_iterations':iterations,'status':status,'same_initial_state':True,
                'grasp_stability':audit.summary(),'stable':audit.summary()['arms']['right']['status']=='stable_within_audit_window',
                'actual_hold_seconds':rows[-1]['sim_seconds']-rows[0]['sim_seconds'],
                'runtime_parameter_changes':env.manifest()['model_evidence']['runtime_parameters']['changed_fields'],
                'wall_seconds':time.monotonic()-started,'automatically_applied':False}
            dump(folder/'series.json',rows);dump(folder/'result.json',result)
            (folder/'transitions.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in transitions))
            Image.fromarray(env.render('overview')).save(folder/'final.jpg',quality=92)
            results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:env.close()
    dump(out/'summary.json',results)


if __name__=='__main__':main()
