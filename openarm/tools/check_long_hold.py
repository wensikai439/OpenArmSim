"""Long unmodified hold: distinguish initial slip from a later stable suffix.

The entire original trajectory remains failed if any original rolling gate
fails. Later stable windows are diagnosis, never a rewritten whole-run score.
"""
import json

import numpy as np

from sim.environment import ROOT,OpenArmEnv
from sim.decision_timing import GraspStability
from sim.recording import Recorder,dump
from tools.check_contact_mechanics import contact_mechanics


def main():
    out=ROOT/'runs/20261008-hold/long';out.mkdir(exist_ok=False)
    source=json.loads((out.parent/'protocol.json').read_text());grasp=ROOT/source['grasp_run']
    commands=[json.loads(line) for line in (grasp/'transitions.jsonl').read_text().splitlines()]
    env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(120)
    rec=None;status='incomplete';error=None;rows=[];first_stable=None;run_start=None;best=0.
    try:
        for transition in commands:
            env.step_tamp(np.asarray(transition['action']))
            if np.max(np.abs(env.state()-transition['next_observation']['state']))>1e-9:raise ValueError('grasp differs')
        audit=GraspStability(env);target=env.last_action.copy();start=float(env.data.time)
        protocol={'schema':'long-native-hold/1','seconds':60,'source_grasp':source['grasp_run'],
            'hold_action':target.tolist(),'parameters_changed':False,'thresholds':source['limits'],
            'scope':'diagnose settling and later slip; full original hold score never replaced by best window',
            'model_evidence':env.manifest()['model_evidence'],'hardware_access':False,'transfer_ready':False}
        dump(out/'protocol.json',protocol)
        rec=Recorder(env,'wait',root=out,record_images=False,seed=0,details=protocol)
        for _ in range(1800):
            before=rec.observation(env);_,_,term,trunc,info=env.step_tamp(target);audit.read()
            current=audit.samples['right'];positions=np.asarray([r[1] for r in current])
            rotations=np.asarray([r[2] for r in current]).reshape(len(current),9)
            drift=float(np.max(np.linalg.norm(positions[:,None]-positions[None,:],axis=2)))
            angle=float(np.arccos(np.clip((np.min(rotations@rotations.T)-1)/2,-1,1)))
            stable=(current[-1][0]-current[0][0]>=2-1e-6 and drift<=.001 and angle<=.02
                    and env.contacts.hold_seconds['right']>=2-1e-6)
            now=float(env.data.time)-start
            if stable:
                if first_stable is None:first_stable=now
                if run_start is None:run_start=now
                best=max(best,now-run_start)
            else:run_start=None
            row={**contact_mechanics(env),'elapsed_s':now,'recent_position_range_m':drift,
                 'recent_rotation_range_rad':angle,'continuous_two_finger_seconds':env.contacts.hold_seconds['right'],
                 'recent_stable':bool(stable)}
            rows.append(row)
            rec.append(before,target,rec.observation(env),info,terminated=term,truncated=trunc,
                diagnostics={'recent_stable':bool(stable),'recent_position_range_m':drift,'recent_rotation_range_rad':angle})
            if term or trunc:
                status='physics_invalid' if info['evaluation']['invalid'] else 'truncated';break
        else:status='completed'
        result={'status':status,'actual_seconds':rows[-1]['elapsed_s'],'full_original_audit':audit.summary(),
            'first_stable_recent_window_s':first_stable,'longest_continuous_stable_windows_s':best,
            'stable_at_end':rows[-1]['recent_stable'],'model_evidence':env.manifest()['model_evidence'],
            'native_run':rec.path.name,'whole_run_score_not_rewritten':True}
        dump(out/'series.json',rows);dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:error=str(exc);raise
    finally:
        if rec:rec.finish(status,env.info(),error=error,metrics={'grasp_stability':audit.summary()})
        env.close()


if __name__=='__main__':main()
