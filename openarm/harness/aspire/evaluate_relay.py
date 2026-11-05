"""Paired fresh-position and dynamics stress evaluation; immutable artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .runtime import ROOT, dump


def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('sim','harness','tools') for p in sorted((ROOT/folder).rglob('*.py'))}


def prepare(folder):
    if (folder/'protocol.json').exists():raise ValueError('protocol already frozen')
    rng=random.Random(2026102401);cases=[]
    for group, count in (('position',12),('combined',12)):
        for index in range(count):
            # Half the positions cover the wider rectangle. Half stress the
            # previously failing front/right region using NEW XY values.
            x=rng.uniform(.41,.59) if index<6 else rng.uniform(.56,.59)
            y=rng.uniform(-.29,-.09) if index<6 else rng.uniform(-.29,-.25)
            case={'schema':'relay-fixture/2','seed':800+len(cases),
                'bottle_xy_m':[round(x,6),round(y,6)],
                'bottle_yaw_rad':0.,'basket_offset_m':[0.,0.],
                'bottle_mass_scale':1.,'bottle_friction_scale':1.}
            if group=='combined':
                case.update(bottle_yaw_rad=round(rng.uniform(-3.14,3.14),6),
                    basket_offset_m=[round(rng.uniform(-.02,.02),6) for _ in range(2)],
                    bottle_mass_scale=round(rng.uniform(.8,1.2),6),
                    bottle_friction_scale=round(rng.uniform(.8,1.2),6))
            path=folder/'cases'/f'{case["seed"]}.json';path.parent.mkdir(exist_ok=True)
            dump(path,case);cases.append({'group':group,'path':str(path.relative_to(ROOT)),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    frozen={'schema':'relay-evaluation/2','cases':cases,'random_seed':2026102401,
        'partition':'fresh heldout, development cases are excluded; all outcomes retained',
        'stress_distribution':'engineering ranges, NOT measured real-robot distributions',
        'source_hashes':source_hashes(),
        'programs':{name:{'path':str((folder/file).relative_to(ROOT)),
            'sha256':hashlib.sha256((folder/file).read_bytes()).hexdigest()}
            for name,file in (('v1','baseline.py'),('v2','candidate.py'))},
        'wall_budget_s':150,'motion_calls':12,'sim_budget_s':120,
        'timing':'paused action-only diagnostic; not real-time validation',
        'hardware_access':False,'transfer_ready':False}
    dump(folder/'protocol.json',frozen)


def trial(folder, protocol, method, case):
    path=ROOT/case['path'];value=json.loads(path.read_text())
    out=folder/'heldout'/method/str(value['seed']);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('refuse overwriting a trial')
    command=[sys.executable,'-B','-m','harness.aspire.variants','--program',
        str(ROOT/protocol['programs'][method]['path']),'--case',str(path),'--output',str(out)]
    start=time.monotonic();timed_out=False
    with out.with_suffix('.log').open('w') as stream:
        try:
            process=subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
                    MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                timeout=protocol['wall_budget_s'],check=False)
        except subprocess.TimeoutExpired:timed_out=True
    result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {
        'task_success':False,'program_ok':False,'error':'wall_budget' if timed_out else 'process_error'}
    result.update(method=method,group=case['group'],case=value,
        wall_seconds=time.monotonic()-start,trial=str(out.relative_to(ROOT)))
    return result


def evaluate(folder):
    protocol=json.loads((folder/'protocol.json').read_text())
    if protocol['source_hashes']!=source_hashes():raise ValueError('sources changed after freeze')
    for row in [*protocol['programs'].values(),*protocol['cases']]:
        if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('frozen input changed')
    rows=[]
    # Two isolated simulator processes; no additional AI agents or services.
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(trial,folder,protocol,method,case)
            for case in protocol['cases'] for method in ('v1','v2')]
        for job in as_completed(jobs):
            row=job.result();rows.append(row);dump(folder/'results.json',rows)
            print(json.dumps({k:row.get(k) for k in ('method','group','task_success','failure_code','error')}
                | {'seed':row['case']['seed']}),flush=True)
    if protocol['source_hashes']!=source_hashes():raise ValueError('sources changed during evaluation')
    result={'methods':{},'pairs':[],'hardware_access':False,'transfer_ready':False}
    for method in ('v1','v2'):
        result['methods'][method]={group:{'passed':sum(r['task_success'] is True for r in rows
            if r['method']==method and r['group']==group),'total':sum(r['method']==method and r['group']==group for r in rows)}
            for group in ('position','combined')}
    for case in protocol['cases']:
        seed=json.loads((ROOT/case['path']).read_text())['seed']
        pair={r['method']:r for r in rows if r['case']['seed']==seed}
        result['pairs'].append({'seed':seed,'group':case['group'],
            'v1':pair['v1']['task_success'],'v2':pair['v2']['task_success'],
            'v1_failure':pair['v1'].get('failure_code',pair['v1'].get('error')),
            'v2_failure':pair['v2'].get('failure_code',pair['v2'].get('error'))})
    dump(folder/'assessment.json',result)
    print(json.dumps(result['methods']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','evaluate'))
    p.add_argument('folder',type=Path);args=p.parse_args();folder=args.folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):p.error('folder must stay inside runs/')
    (prepare if args.command=='prepare' else evaluate)(folder)
