"""Paired native planner-clock development cases; preserve every outcome."""
import json
import time
import hashlib
from pathlib import Path

from sim.environment import ROOT
from sim.service import SimulationService
from sim.recording import dump


CASES = {
    'forward': {'object_id':'bottle','arm':'right','receptacle_id':'table','offset_xy_m':[.08,0.]},
    'transfer': {'object_id':'bottle','arm':'left','via_arm':'right','receptacle_id':'table','offset_xy_m':[.06,0.]},
    'curobo': {'object_id':'bottle','arm':'right','receptacle_id':'table','offset_xy_m':[.08,0.],
               'transport_planner':'curobo','transport_delta_m':[.015,0.,0.]},
}


def main():
    out=ROOT/'runs/20261008-planning'
    if (out/'native.json').exists():raise ValueError('preserve previous native study')
    dump(out/'protocol.json',{'cases':CASES,'modes':['paused','measured_delay'],'seed':0,'episode':0,
        'budget_s':120,'reset_before_each_condition':True,'physics_parameters_changed':False,
        'scoring_thresholds_changed':False,'actual_LLM_API':False,'hardware_access':False,'transfer_ready':False,
        'scope':'six declared development cases; computation timings depend on local load, not statistical generalization',
        'measured_scope':'native construction/generator computation and before-dispatch integration; no LLM request in these cases',
        'quantization':'existing 1ms physics grid; preserve ordinary 30Hz command cadence',
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    service=SimulationService();rows=[]
    try:
        for case,parameters in CASES.items():
            for mode in ('paused','measured_delay'):
                service.call('reset',episode=0,seed=0)
                service.call('task_budget',seconds=120)
                before=service.snapshot();manifest=service.call('manifest')
                started=time.monotonic()
                response=service.call('tamp',task='relocate_object',skill='relocate',parameters=parameters,
                    observation_id=before['observation_id'],timing_mode=mode,realtime=False)
                deadline=time.monotonic()+240
                while True:
                    s=service.snapshot()
                    if not s.get('job'):break
                    if time.monotonic()>deadline:raise TimeoutError('same native owner remains live; inspect before stopping')
                    time.sleep(.1)
                last=s['last_run'];summary=json.loads((ROOT/'runs'/last['run_id']/'summary.json').read_text())
                row={'case':case,'mode':mode,'parameters':parameters,'response':response,
                    'initial':before['observation'],'status':last['status'],'error':last['error'],
                    'run_id':last['run_id'],'wall_seconds':time.monotonic()-started,
                    'duration_sim_seconds':summary['duration_sim_seconds'],'metrics':last['metrics'],
                    'terminal_evaluation':s['diagnostics']['evaluation'],
                    'model_evidence_unchanged':manifest['model_evidence']==service.call('manifest')['model_evidence'],
                    'actual_LLM_API':False,'hardware_access':False,'transfer_ready':False}
                rows.append(row);dump(out/'native.json',{'protocol':'protocol.json','cases':rows,'complete':len(rows)==6})
                print(json.dumps({k:row[k] for k in ('case','mode','status','error','duration_sim_seconds')},ensure_ascii=False),flush=True)
    finally:service.close()


if __name__=='__main__':main()
