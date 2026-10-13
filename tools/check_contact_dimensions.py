"""Fixed contact-model counterfactuals after one identical archived grasp.

Only effective contact dimension changes, after the grasp. These are explicit
physical-model hypotheses, never silently applied to the platform or hardware.
"""
import json
import sqlite3

import numpy as np

from sim.environment import ROOT,OpenArmEnv
from sim.decision_timing import GraspStability
from sim.recording import dump
from tools.check_contact_mechanics import contact_mechanics


def main():
    out=ROOT/'runs/20261008-model/dimensions-final';out.mkdir(exist_ok=False)
    report=json.loads((ROOT/'runs/20261008-alignment/api-terminal.json').read_text())
    first=sorted((ROOT/'runs'/('20261008-agent-'+report['turn_id'][:8])/'waits').glob('*/manifest.json'),
                 key=lambda p:json.loads(p.read_text())['start_tick'])[0].parent
    with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as db:
        execution=report['terminal']['agent']['plan']['execution_tools'][0]['id']
        pick=ROOT/'runs'/db.execute('SELECT run_id FROM episodes WHERE id=?',(execution,)).fetchone()[0]
    transitions=[json.loads(line) for p in [first,pick] for line in (p/'transitions.jsonl').read_text().splitlines()]
    protocol={'schema':'contact-dimensions/1','dimensions':[3,4,6],'hold_seconds':10,
        'source_grasp':[str(p.relative_to(ROOT)) for p in [first,pick]],
        'change':'only four finger collision geom condim values, AFTER identical native grasp',
        'unchanged':'mass, friction coefficients, gains, actuator limits, solver, pose, applied hold target, stability thresholds',
        'effective_contacts':'priority-1 finger geoms dominate priority-0 bottle geoms; verify actual contact dimensions',
        'scope':'physical-model counterfactual, not independently measured patch friction or a selected production model',
        'automatic_application':False,'hardware_access':False,'transfer_ready':False}
    dump(out/'protocol.json',protocol);results=[]
    for dimension in protocol['dimensions']:
        env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
        try:
            for transition in transitions:
                env.step_tamp(np.asarray(transition['action']))
                if np.max(np.abs(env.state()-np.asarray(transition['next_observation']['state'])))>1e-9:
                    raise ValueError('grasp reproduction mismatch')
            q,v,pose=env.data.qpos.copy(),env.data.qvel.copy(),env.data.xpos[env.bottle_id].copy()
            for i in range(env.model.ngeom):
                if 'finger' in env.model.geom(i).name and env.model.geom_contype[i]:
                    env.model.geom_condim[i]=dimension
            assert np.array_equal(q,env.data.qpos) and np.array_equal(v,env.data.qvel)
            audit=GraspStability(env);target=env.last_action.copy();samples=[contact_mechanics(env)]
            # The production Recorder correctly refuses changed parameters.
            # Keep this counterfactual in an explicit research-only artifact;
            # never weaken that production guard or relabel a normal receipt.
            dump(out/(str(dimension)+'-manifest.json'),{**env.manifest(),
                'kind':'research_contact_counterfactual','details':{**protocol,'dimension':dimension,
                'holding_state_before_change':pose.tolist()},'start_tick':env.tick,
                'hold_applied_target':target.tolist(),'initial_robot_state':env.state().tolist()})
            status='completed'
            for _ in range(300):
                _,_,term,trunc,info=env.step_tamp(target);audit.read()
                samples.append({**contact_mechanics(env),'robot_state':env.state().tolist()})
                if term or trunc:status='physics_invalid' if info['evaluation']['invalid'] else 'truncated';break
            summary={'dimension':dimension,'status':status,'grasp_stability':audit.summary(),
                'held':dict(env.evaluation['held']),
                'effective_dimensions':sorted({c['dimension'] for s in samples[1:] for c in s['contacts']}),
                'runtime_parameter_changes':env.manifest()['model_evidence']['runtime_parameters']['changed_fields'],
                'relative_displacement_m':float(np.linalg.norm(np.subtract(samples[-1]['relative_position_m'],samples[0]['relative_position_m']))),
                'manifest':str(dimension)+'-manifest.json'}
            dump(out/(str(dimension)+'.json'),{**summary,'samples':samples})
            results.append(summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
        finally:env.close()
    dump(out/'summary.json',results)


if __name__=='__main__':main()
