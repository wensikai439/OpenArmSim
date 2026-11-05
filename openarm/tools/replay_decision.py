"""Explain an archived LLM wait from its actual native control transitions."""
import json
import sqlite3
from pathlib import Path

import numpy as np

from sim.environment import ROOT,OpenArmEnv
from sim.calibration import response_dependencies

OUT=ROOT/'runs/20261008-decision'


def measurement(env):
    p=env.data.xpos[env.bottle_id].copy();result={'bottle_position_m':p.tolist(),'tcp':{}}
    for side,body in env.ee_ids.items():
        r=env.data.xmat[body].reshape(3,3);tcp=env.data.xpos[body]+r@np.array([0,.006,.075])
        result['tcp'][side]={'position_m':tcp.tolist(),'rotation':r.tolist(),'object_relative_m':(r.T@(p-tcp)).tolist()}
    return result


def main():
    report=json.loads((OUT/'api-first-failure.json').read_text());root=report['turn_id']
    paths=[ROOT/w['run_path'] for w in report['waits']]
    with sqlite3.connect('file:'+str(ROOT/'harness/state/agent.sqlite3')+'?mode=ro',uri=True) as db:
        for execution in report['terminal']['agent']['plan']['execution_tools']:
            row=db.execute('SELECT run_id FROM episodes WHERE id=?',(execution['id'],)).fetchone()
            paths.append(ROOT/'runs'/row[0])
    paths.sort(key=lambda p:json.loads((p/'manifest.json').read_text())['start_tick'])
    env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120);results=[]
    try:
        for path in paths:
            manifest=json.loads((path/'manifest.json').read_text())
            assert response_dependencies(env.manifest())==response_dependencies(manifest)
            assert env.tick==manifest['start_tick'],(env.tick,manifest['start_tick'])
            before=measurement(env)
            for line in (path/'transitions.jsonl').read_text().splitlines():
                transition=json.loads(line);env.step_tamp(np.asarray(transition['action']))
                assert env.tick==transition['next_observation']['tick']
            after=measurement(env)
            row={'run_path':str(path.relative_to(ROOT)),'kind':manifest['kind'],'before':before,'after':after,
                 'sim_seconds':float(env.data.time)-manifest['start_sim_seconds'], 'held':dict(env.evaluation['held'])}
            if manifest['kind']=='wait':
                row['object_drift_m']=float(np.linalg.norm(np.array(after['bottle_position_m'])-before['bottle_position_m']))
                row['tcp_drift_m']={side:float(np.linalg.norm(np.array(after['tcp'][side]['position_m'])-before['tcp'][side]['position_m'])) for side in ('left','right')}
                row['grasp_relative_drift_m']={side:float(np.linalg.norm(np.array(after['tcp'][side]['object_relative_m'])-before['tcp'][side]['object_relative_m'])) for side in ('left','right')}
                row['tcp_rotation_rad']={side:float(np.arccos(np.clip((np.trace(np.array(before['tcp'][side]['rotation']).T@np.array(after['tcp'][side]['rotation']))-1)/2,-1,1))) for side in ('left','right')}
            results.append(row)
        actual=report['terminal']['diagnostics']['manipulation_scene']['position_m']
        residual=float(np.linalg.norm(env.data.xpos[env.bottle_id]-actual));assert residual<1e-8
        output={'source_root':root,'method':'archived actuator commands; native step_tamp; no pose edits after reset',
                'reset_within_task':False,'reproduction_residual_m':residual,'records':results,'hardware_access':False,'transfer_ready':False}
        (OUT/'replay.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'reproduction_residual_m':residual,'waits':[{'seconds':r['sim_seconds'],'object_mm':r['object_drift_m']*1000,'right_tcp_mm':r['tcp_drift_m']['right']*1000,'right_relative_mm':r['grasp_relative_drift_m']['right']*1000} for r in results if r['kind']=='wait']},ensure_ascii=False))
    finally:env.close()


if __name__=='__main__':main()
