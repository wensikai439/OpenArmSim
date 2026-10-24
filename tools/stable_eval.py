"""Frozen paired donor-support regression; native transitions, not video scoring."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261024-ready'


def main():
    rng=random.Random(2026102421)
    cases=[]
    for i in range(4):
        xy=[round(rng.uniform(.46,.55),6),round(rng.uniform(-.25,-.16),6)]
        for stress in [False,True]:
            case={'seed':3200+2*i+int(stress),'bottle_xy_m':xy,'fault':'none',
                'motion_profile':'precision','grasp_region':'body'}
            if stress:case.update(bottle_mass_scale=1.2,bottle_friction_scale=.8,
                bottle_yaw_rad=round(rng.uniform(-3.14,3.14),6),
                basket_offset_m=[round(rng.uniform(-.02,.02),6) for _ in range(2)],
                fault=['right_close_loss','receiver_close_loss','stale_grasp','none'][i],pose_bias_m=.05)
            cases.append(case)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['sim','harness'] for p in sorted((ROOT/folder).rglob('*.py'))}
    protocol={'cases':cases,'sources':sources,'program_sha256':hashlib.sha256((OUT/'program.py').read_bytes()).hexdigest(),
        'partition':'fresh paired regression frozen after one development position; no heldout tuning',
        'acceptance':'original native/body/left checks plus right stationary TCP residual and decline<=5mm',
        'hardware_access':False}
    path=OUT/'paired/protocol.json'
    if path.exists():raise ValueError('never overwrite frozen tests')
    path.parent.mkdir(parents=True);path.write_text(json.dumps(protocol,indent=2))
    jobs=[(case,enabled) for case in cases for enabled in [False,True]]
    def run(job):
        case,enabled=job;name=f"{case['seed']}-{'support' if enabled else 'baseline'}"
        cp=OUT/'cases'/(name+'.json');cp.write_text(json.dumps({**case,'right_payload_compensation':enabled}))
        folder=OUT/'paired'/name
        with folder.with_suffix('.log').open('w') as log:
            p=subprocess.run([sys.executable,'-B','-m','harness.aspire.stable','--case',str(cp),
                '--output',str(folder),'--program',str(OUT/'program.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                timeout=240,env=dict(os.environ,MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        r=json.loads((folder/'result.json').read_text())
        sys.path.insert(0,str(ROOT))
        from harness.aspire.recovery_evaluation import audit_trial
        audit=audit_trial(folder,r)
        row={'seed':case['seed'],'support':enabled,'task_success':r['task_success'],
            'native_success':r.get('native_task_success'),'donor':r.get('donor_support'),
            'left_drop_m':r.get('body_relay',{}).get('max_left_handover_drop_m'),'error':r.get('error'),
            'trial':str(folder.relative_to(ROOT)),'audit':audit,'returncode':p.returncode}
        print(json.dumps(row),flush=True);return row
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,jobs))
    changed=[p for p,h in sources.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError('source changed during evaluation: '+str(changed))
    (OUT/'paired/results.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({'trials':len(rows),'sources_unchanged':True,'supported_success':sum(r['task_success'] for r in rows if r['support'])}),flush=True)


if __name__=='__main__':main()
