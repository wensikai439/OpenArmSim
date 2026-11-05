"""Local-only, frozen development/held-out tests for bottle skill robustness."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def freeze(out):
    if out.exists():raise ValueError('use a new experiment directory')
    out.mkdir(parents=True)
    program=ROOT/'skills/bottle_recovery/program.py'
    (out/'baseline.py').write_bytes(program.read_bytes())
    dev=[[.44,-.28],[.58,-.28],[.44,-.12],[.58,-.12],[.50,-.21],[.55,-.18]]
    rng=random.Random(2026102411)
    held=[[round(rng.uniform(.455,.555),6),round(rng.uniform(-.255,-.155),6)] for _ in range(8)]
    held += [[round(rng.uniform(.43,.595),6),round(rng.uniform(-.29,-.105),6)] for _ in range(8)]
    cases=[]
    for partition,points,start in [('development',dev,2000),('heldout',held,2200)]:
        for i,xy in enumerate(points):
            for stress in (False,True):
                seed=start+2*i+int(stress)
                case={'seed':seed,'bottle_xy_m':xy,'fault':'none','motion_profile':'precision','grasp_region':'body'}
                if stress:
                    case.update(bottle_yaw_rad=round(rng.uniform(-3.14,3.14),6),
                        basket_offset_m=[round(rng.uniform(-.023,.023),6) for _ in range(2)],
                        bottle_mass_scale=round(rng.uniform(.8,1.2),6),
                        bottle_friction_scale=round(rng.uniform(.8,1.2),6),
                        fault=('right_close_loss','stale_grasp','receiver_close_loss','none')[i%4],
                        pose_bias_m=round(rng.uniform(.04,.07),6))
                path=out/'cases'/partition/(str(seed)+'.json');save(path,case)
                cases.append({'seed':seed,'partition':partition,'group':'combined' if stress else 'nominal',
                    'region':'core' if partition=='heldout' and i<8 else 'challenge',
                    'path':str(path.relative_to(ROOT)),
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    save(out/'protocol.json',{'schema':'bottle-robustness/1','random_seed':2026102411,'cases':cases,
        'development_positions':6,'heldout_positions':16,'conditions_per_position':2,
        'partition':'development evidence only may be used to revise; freeze candidate before heldout; no heldout tuning',
        'faults':'same one-shot command loss/height bias as previous diagnostic, not learned visual perception',
        'domain':'same registered upright bottle; yaw full circle; basket +-23mm; mass/friction x0.8-1.2',
        'acceptance':'native success, bottle body zones, left stationary drop<=5mm; collision/tracking/120s gates unchanged',
        'geometry':'simulation oracle; paused planning; no hardware or first-real-success claim',
        'hardware_access':False,'transfer_ready':False})


def evaluate(out,partition,method,program,module):
    protocol=json.loads((out/'protocol.json').read_text())
    target=out/partition/method
    if target.exists():raise ValueError('refuse overwriting previous trials')
    target.mkdir(parents=True)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('sim','harness','tools') for p in sorted((ROOT/folder).rglob('*.py'))}
    save(target/'frozen.json',{'program':str(program.relative_to(ROOT)),
        'program_sha256':hashlib.sha256(program.read_bytes()).hexdigest(),'module':module,'sources':hashes})
    def one(case):
        path=ROOT/case['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case['sha256']
        folder=target/str(case['seed']);error=None
        with folder.with_suffix('.log').open('w') as log:
            try:
                r=subprocess.run([sys.executable,'-B','-m',module,'--case',str(path),'--output',str(folder),
                    '--program',str(program)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
                if r.returncode:error='process exit '+str(r.returncode)
            except subprocess.TimeoutExpired:error='wall budget exceeded'
        result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
        audit=None
        if result:
            from harness.aspire.recovery_evaluation import audit_trial
            try:audit=audit_trial(folder,result)
            except Exception as exc:error='audit: '+str(exc)
        body=result.get('body_relay',{})
        return {'seed':case['seed'],'group':case['group'],'region':case['region'],
            'success':bool(result.get('task_success') and not error),'failure':result.get('native_failures'),
            'error':error or result.get('error'),'executions':result.get('executions'),
            'left_drop_mm':body.get('max_left_handover_drop_m',0)*1000,
            'body_contact_verified':body.get('body_contact_verified'),
            'sim_seconds':result.get('sim_seconds'),'wall_seconds':result.get('wall_seconds'),
            'trial':str(folder.relative_to(ROOT)),'audit':audit}
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(one,c) for c in protocol['cases'] if c['partition']==partition]):
            row=future.result();rows.append(row);save(target/'results.json',rows)
            print(json.dumps({k:row[k] for k in ['seed','group','success','failure','error','left_drop_mm']},ensure_ascii=False),flush=True)
    changed=[p for p,d in hashes.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=d]
    if changed:raise RuntimeError('sources changed during evaluation: '+str(changed))
    summary={g:{'passed':sum(r['success'] for r in rows if r['group']==g),
                'total':sum(r['group']==g for r in rows)} for g in ['nominal','combined']}
    summary.update(total_passed=sum(r['success'] for r in rows),total=len(rows),sources_unchanged=True)
    save(target/'assessment.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','evaluate'])
    p.add_argument('output',type=Path);p.add_argument('--partition',choices=['development','heldout'],default='development')
    p.add_argument('--method',default='baseline');p.add_argument('--module',default='harness.aspire.body')
    p.add_argument('--program',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs'):p.error('local experiment output belongs in runs/')
    if a.command=='freeze':freeze(out)
    else:evaluate(out,a.partition,a.method,(a.program or out/'baseline.py').resolve(),a.module)
