#!/usr/bin/env python3
"""Freeze and check body-only relay, payload stability and a matched control."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def save(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs') or out.exists():p.error('use a new directory under runs/')
    out.mkdir(parents=True);rng=random.Random(2026102404);cases=[]
    for i in range(6):
        xy=[round(rng.uniform(.46,.54),6),round(rng.uniform(-.27,-.16),6)]
        for stress in (False,True):
            seed=1400+i+6*stress
            case={'seed':seed,'bottle_xy_m':xy,'grasp_region':'body','motion_profile':'precision',
                'fault':('none','right_close_loss','stale_grasp','receiver_close_loss','none','stale_grasp')[i] if stress else 'none'}
            if stress:case.update(bottle_yaw_rad=round(rng.uniform(-.5,.5),6),
                basket_offset_m=[round(rng.uniform(-.012,.012),6) for _ in range(2)],
                bottle_mass_scale=round(rng.uniform(.9,1.1),6),bottle_friction_scale=round(rng.uniform(.9,1.1),6),
                pose_bias_m=round(rng.uniform(.043,.067),6))
            path=out/'cases'/f'{seed}.json';save(path,case)
            cases.append((path,out/'trials'/str(seed),'combined' if stress else 'nominal'))
    first=json.loads((out/'cases/1400.json').read_text());first['payload_compensation']=False
    control=out/'cases/control.json';save(control,first);cases.append((control,out/'trials/control','control'))
    for seed in (1302,1308,1311):
        path=ROOT/f'runs/20261024-body/cases/{seed}.json'
        cases.append((path,out/'regressions'/str(seed),'regression'))
    files=[*list((ROOT/'sim').glob('*.py')),*list((ROOT/'harness').glob('*.py')),
        *[ROOT/'harness/aspire'/name for name in ('body.py','precision.py','recovery.py','recovery_campaign.py','runtime.py','variants.py')],
        ROOT/'tools/object_workbench.py',ROOT/'skills/bottle_recovery/program.py',Path(__file__)]
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
    protocol={'schema':'body-relay/2','seed':2026102404,'source_hashes':hashes,
        'cases':[{'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'output':str(trial.relative_to(ROOT)),'group':group} for path,trial,group in cases],
        'partition':'six fresh positions crossed with nominal/combined conditions; three prior failure regressions separate',
        'acceptance':'native task gates AND body contact regions AND left stationary TCP drop<=5mm',
        'timing':'paused simulation planning; not realtime or hardware validation','hardware_access':False}
    save(out/'protocol.json',protocol)
    def verify_sources():
        for path,digest in hashes.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('source changed: '+path)
    def trial(job):
        path,folder,group=job;folder.parent.mkdir(parents=True,exist_ok=True)
        with folder.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable,'-B','-m','harness.aspire.body','--case',str(path),
                '--output',str(folder),'--program',str(ROOT/'skills/bottle_recovery/program.py')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=True)
        result=json.loads((folder/'result.json').read_text())
        from harness.aspire.recovery_evaluation import audit_trial
        audit=audit_trial(folder,result)
        return {'case':path.stem,'group':group,'success':result['task_success'],
            'native_task_success':result.get('native_task_success'),
            'body':result['body_relay'],'sim_seconds':result['sim_seconds'],
            'xy_error_m':result['pose_error_m'],'native_failures':result.get('native_failures'),
            'trial':str(folder.relative_to(ROOT)),'audit':audit}
    rows=[];verify_sources()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(trial,job) for job in cases]):
            row=future.result();rows.append(row);save(out/'results.json',rows)
            print(json.dumps({'case':row['case'],'group':row['group'],'success':row['success'],
                'left_drop_mm':row['body']['max_left_handover_drop_m']*1000,'failure':row['native_failures']}),flush=True)
    verify_sources()
    assessment={group:{'passed':sum(r['success'] for r in rows if r['group']==group),
        'total':sum(r['group']==group for r in rows)} for group in ('nominal','combined','regression','control')}
    assessment['max_accepted_left_drop_m']=max(r['body']['max_left_handover_drop_m'] for r in rows if r['group'] in ('nominal','combined'))
    save(out/'assessment.json',assessment);print(json.dumps(assessment),flush=True)


if __name__=='__main__':main()
