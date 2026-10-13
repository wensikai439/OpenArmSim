"""Fixed-proposal source-MIT response audit; no fitting or automatic application."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sim.environment import OpenArmEnv,ROOT
from sim.reference_clock import load
from sim.calibration import metrics

MODES={'position_pd_bias':{},'source_mit_zero':{'controller_mode':'source_mit_candidate','gravity_compensation':'none'},
       'source_mit_gravity':{'controller_mode':'source_mit_candidate','gravity_compensation':'none','source_initial_integral':'static_gravity_proxy'}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/20261008-control/controller')
    parser.add_argument('--mode',choices=list(MODES)+['all'],default='all')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs') or out.exists():raise ValueError('use a new evidence directory under runs')
    out.mkdir(parents=True);clock=load();rows=[];start=time.monotonic()
    for name,profile in MODES.items():
        if args.mode not in (name,'all'):continue
        for episode in (0,1,2):
            env=OpenArmEnv(images=False,profile=profile);env.reset(seed=0,options={'episode':episode})
            times=np.asarray(clock['episodes'][str(episode)]['recorded_seconds']);n=int(np.searchsorted(times,4.,side='right'))
            target=np.array([env.reference[str(episode)][side+'_joints'][:n] for side in ('left','right')])
            prediction=[];commands=[];contacts=[];error=None
            try:
                for frame in range(n):
                    prediction.append([env.data.qpos[env.qids[side]].tolist() for side in ('left','right')])
                    if frame==n-1:break
                    env.step_until(env.reference_action(episode,frame),times[frame+1])
                    commands.append({'frame':frame,'time_s':float(env.data.time),'feedback':env.controller_feedback.copy()})
                    window=env.evaluation['contact_window']
                    if any(p['robot_external'] or p['robot_self'] for p in window['pairs']):contacts.append({'frame':frame,'window':window})
                    if env._done:raise RuntimeError('native terminal '+str(env.evaluation['failure_reasons']))
            except Exception as exc:error=str(exc)
            predicted=np.asarray(prediction).transpose(1,0,2);scored=len(prediction)
            row={'mode':name,'episode':episode,'status':'completed' if error is None else 'error','error':error,
                'metrics':{side:metrics(predicted[i],target[i,:scored],10) for i,side in enumerate(('left','right'))},
                'robot_contact_windows':len(contacts),'contact_evidence':contacts,
                'recorded_times_s':times[:scored].tolist(),'prediction':predicted.tolist(),'target':target[:,:scored].tolist(),
                'controller_commands':commands,'model':env.manifest(),'initial_state_assumption':profile.get('source_initial_integral','zero'),
                'transfer_ready':False}
            path=out/f'{name}-ep{episode}.json';path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
            rows.append({k:row[k] for k in ('mode','episode','status','error','metrics','robot_contact_windows')})
            print(json.dumps(rows[-1],ensure_ascii=False),flush=True);env.close()
    summary={'status':'studied','cases':rows,'wall_seconds':time.monotonic()-start,'applied_to_runtime':False,
        'selection':'none; fixed proposals reported on all three previously inspected development trajectories',
        'limits':['current on-disk source is not bound to collection-time revision','initial integral/error state and motor firmware behavior unknown',
                  'native torque limits/geometry/transmission not independently measured','first four seconds only; not contact/full-task/real-transfer validation'],
        'transfer_ready':False}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
