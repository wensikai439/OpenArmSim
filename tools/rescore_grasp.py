"""Replay archived controls for a new diagnostic, preserving original scores."""
import json
import numpy as np

from sim.environment import ROOT,OpenArmEnv
from sim.calibration import response_dependencies
from sim.decision_timing import GraspStability

OUT=ROOT/'runs/20261008-alignment'


def main():
    output=OUT/'rescored.json'
    if output.exists():raise ValueError('preserve previous rescore')
    source=json.loads((OUT/'proposals/summary.json').read_text());results=[]
    for candidate in source:
        folder=OUT/'proposals'/candidate['case']['id']
        original=json.loads((folder/'summary.json').read_text())
        if original['status']=='failed':
            results.append({'case':candidate['case'],'status':'failed','failure':original['failure'],'rescored':False});continue
        paths=sorted((folder/'episodes').iterdir(),key=lambda p:json.loads((p/'manifest.json').read_text())['start_tick'])
        env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
        maximum_state_error=0.;maximum_body_error=0.;audit=None
        try:
            for path in paths:
                manifest=json.loads((path/'manifest.json').read_text())
                assert response_dependencies(env.manifest())==response_dependencies(manifest)
                assert env.tick==manifest['start_tick']
                if manifest['kind']=='wait':
                    audit=GraspStability(env);samples=original['samples']
                    maximum_body_error=float(np.max(abs(env.data.xpos[env.bottle_id]-samples[0]['body_m'])))
                for index,line in enumerate((path/'transitions.jsonl').read_text().splitlines()):
                    row=json.loads(line);env.step_tamp(np.asarray(row['action']))
                    assert env.tick==row['next_observation']['tick']
                    maximum_state_error=max(maximum_state_error,float(np.max(abs(env.state()-row['next_observation']['state']))))
                    if audit is not None:
                        audit.read();maximum_body_error=max(maximum_body_error,float(np.max(abs(env.data.xpos[env.bottle_id]-samples[index+1]['body_m']))))
            assert audit is not None and maximum_state_error<1e-9 and maximum_body_error<1e-9
            summary=audit.summary();right=summary['arms']['right']
            result={'case':candidate['case'],'status':'stable' if right['status']=='stable_within_audit_window' else 'unstable',
                'original_diagnostic':original['grasp_stability'],'grasp_stability':summary,
                'maximum_native_state_reproduction_error':maximum_state_error,'maximum_body_reproduction_error_m':maximum_body_error,
                'method':'archived native controls from seed 0; physics/data hashes agree; per-tick robot and object state reproduction checked',
                'source_scores_overwritten':False,'hardware_access':False,'transfer_ready':False,'rescored':True}
            results.append(result)
            print(json.dumps({'id':candidate['case']['id'],'status':result['status'],'right':right,'reproduction_error':maximum_state_error},ensure_ascii=False),flush=True)
        finally:env.close()
    output.write_text(json.dumps({'schema':'grasp-diagnostic-rescore/1','cases':results,
        'scope':'software diagnostic comparison, not physical calibration','original_scores_preserved':True},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
