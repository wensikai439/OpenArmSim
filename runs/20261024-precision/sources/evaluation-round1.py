"""Paired new-position evaluation after development on old counterexamples."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from .runtime import ROOT,dump
from .recovery_evaluation import audit_trial


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    files=[p for f in ('sim','harness') for p in (ROOT/f).glob('*.py')
        if 'check' not in p.stem and p.stem!='cli']
    files += [ROOT/'harness/aspire'/f for f in
        ('runtime.py','variants.py','recovery.py','recovery_campaign.py','precision.py')]
    files += [ROOT/'tools/object_workbench.py',Path(__file__)]
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}


def prepare(folder):
    if (folder/'protocol.json').exists():raise ValueError('already frozen')
    rng=random.Random(2026102403);xy=[]
    for i in range(8):
        xy.append([round(rng.uniform(.46,.50) if i<4 else rng.uniform(.44,.59),6),
            round(rng.uniform(-.235,-.19) if i<4 else rng.uniform(-.29,-.11),6)])
    rows=[]
    for g in ('position','combined'):
        for i,point in enumerate(xy):
            c={'seed':1100+len(rows),'bottle_xy_m':point,'fault':'none',
                'bottle_yaw_rad':0.,'basket_offset_m':[0.,0.],
                'bottle_mass_scale':1.,'bottle_friction_scale':1.}
            if g=='combined':
                c.update(bottle_yaw_rad=round(rng.uniform(-3.14,3.14),6),
                    basket_offset_m=[round(rng.uniform(-.012,.012),6) for _ in range(2)],
                    bottle_mass_scale=round(rng.uniform(.9,1.1),6),
                    bottle_friction_scale=round(rng.uniform(.9,1.1),6),
                    fault=('none','right_close_loss','stale_grasp','receiver_close_loss')[i%4])
                if c['fault']=='stale_grasp':c['pose_bias_m']=round(rng.uniform(.043,.067),6)
            path=folder/'cases/heldout'/f'{c["seed"]}.json';path.parent.mkdir(parents=True,exist_ok=True)
            dump(path,c);rows.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path),'group':g})
    program=folder/'program.py';program.write_bytes((ROOT/'skills/bottle_recovery/program.py').read_bytes())
    dump(folder/'protocol.json',{'schema':'precision-relay/1','cases':rows,'positions_m':xy,
        'source_hashes':sources(),'program_sha256':digest(program),
        'development':'old recovery heldout1000,1002,1011 are now development; none counts as new heldout',
        'partition':'8 new XY positions crossed with nominal and combined stress,16 cases,32 paired trials',
        'methods':['baseline','precision'],'sim_budget_s':120,'adapter_call_budget':64,
        'wall_budget_s':240,'workers':2,'geometry_source':'simulation_geometry_oracle',
        'stress_distribution':'engineering stress ranges, not measured real distributions',
        'timing':'paused planning diagnostic, not real-time validation','hardware_access':False})


def evaluate(folder):
    protocol=json.loads((folder/'protocol.json').read_text());program=folder/'program.py'
    if protocol['source_hashes']!=sources() or digest(program)!=protocol['program_sha256']:
        raise ValueError('frozen runtime/program changed')
    def trial(case,profile):
        original=ROOT/case['path']
        if digest(original)!=case['sha256']:raise ValueError('case changed')
        value=json.loads(original.read_text());value['motion_profile']=profile
        src=folder/'inputs'/f'{value["seed"]}-{profile}.json';src.parent.mkdir(exist_ok=True)
        dump(src,value);out=folder/'heldout'/profile/str(value['seed']);out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():raise ValueError('cannot overwrite trial')
        with out.with_suffix('.log').open('w') as s:
            try:subprocess.run([sys.executable,'-B','-m','harness.aspire.precision','--case',str(src),
                '--program',str(program),'--output',str(out)],stdout=s,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,timeout=protocol['wall_budget_s'],env={**os.environ,
                'PYTHONDONTWRITEBYTECODE':'1','MUJOCO_GL':'egl','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            except subprocess.TimeoutExpired:pass
        r=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {'task_success':False,'error':'process_or_wall_budget'}
        r.update(method=profile,case=value,group=case['group'],trial=str(out.relative_to(ROOT)))
        try:r['audit']=audit_trial(out,r)
        except Exception as exc:r.update(task_success=False,audit_error=str(exc))
        return r
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(trial,c,m) for c in protocol['cases'] for m in protocol['methods']]
        for job in as_completed(jobs):
            r=job.result();rows.append(r);dump(folder/'results.json',rows)
            print(json.dumps({'method':r['method'],'seed':r['case']['seed'],'success':r['task_success'],
                'failure':r.get('native_failures'),'error':r.get('error'),'audit_error':r.get('audit_error')}),flush=True)
    if protocol['source_hashes']!=sources():raise ValueError('runtime changed during evaluation')
    assessment={m:{g:{'passed':sum(r['task_success'] for r in rows if r['method']==m and r['group']==g),
        'total':sum(r['method']==m and r['group']==g for r in rows)} for g in ('position','combined')}
        for m in protocol['methods']}
    shared=[]
    for c in protocol['cases']:
        seed=json.loads((ROOT/c['path']).read_text())['seed'];pair={r['method']:r for r in rows if r['case']['seed']==seed}
        if all(r['task_success'] for r in pair.values()):
            shared.append({m:pair[m]['pose_error_m'] for m in protocol['methods']})
    dump(folder/'assessment.json',{'methods':assessment,'paired_success_xy_error_m':shared,
        'improvements':[r['case']['seed'] for r in rows if r['method']=='precision' and r['task_success']
            and not next(x for x in rows if x['method']=='baseline' and x['case']['seed']==r['case']['seed'])['task_success']],
        'regressions':[r['case']['seed'] for r in rows if r['method']=='baseline' and r['task_success']
            and not next(x for x in rows if x['method']=='precision' and x['case']['seed']==r['case']['seed'])['task_success']]})
    print(json.dumps(assessment),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','evaluate'))
    p.add_argument('folder',type=Path);a=p.parse_args();folder=a.folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):p.error('output must stay in runs/')
    (prepare if a.command=='prepare' else evaluate)(folder)
