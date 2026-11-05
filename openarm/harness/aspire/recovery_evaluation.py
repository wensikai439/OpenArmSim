"""Freeze first; compare no recovery, strong rules and condensed skill."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random

from .runtime import ROOT, dump
from .recovery_campaign import subprocess_trial


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    # Exclude presentation, tests and CLI. All motion/provider sources remain
    # pinned; edits there invalidate the experiment instead of silently drifting.
    files=[]
    for folder in ('sim','harness','tools'):
        files.extend(p for p in (ROOT/folder).rglob('*.py') if 'check' not in p.stem
            and 'report' not in p.stem and p.name not in ('__main__.py','publish.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}


def prepare(folder):
    if (folder/'protocol.json').exists():raise ValueError('protocol already frozen')
    rng=random.Random(2026102402)
    xy=[[round(rng.uniform(.47,.59),6),round(rng.uniform(-.29,-.16),6)] for _ in range(3)]
    rows=[]
    for group,fault in enumerate(('none','right_close_loss','stale_grasp','receiver_close_loss')):
        for i,point in enumerate(xy):
            case={'seed':1000+3*group+i,'bottle_xy_m':point,'fault':fault}
            if fault=='stale_grasp':case['pose_bias_m']=[.045,.055,.065][i]
            path=folder/'cases/heldout'/f'{case["seed"]}.json';path.parent.mkdir(parents=True,exist_ok=True)
            dump(path,case);rows.append({'path':str(path.relative_to(ROOT)),
                'sha256':digest(path),'group':fault})
    dump(folder/'protocol.json',{'schema':'recovery-evaluation/1',
        'random_seed':2026102402,'positions_m':xy,'cases':rows,
        'partition':'3 fresh positions crossed with 4 conditions; not 12 independent positions',
        'source_hashes':sources(),'programs':{method:{'path':str((folder/file).relative_to(ROOT)),
            'sha256':digest(folder/file)} for method,file in (('no_recovery','initial.py'),('condensed','candidate.py'))},
        'methods':['no_recovery','fixed','condensed'],
        'fixed_policy':'harness.aspire.recovery_campaign.fixed_policy; same tools and observations',
        'sim_budget_s':120,'adapter_call_budget':64,'wall_budget_s':240,
        'timing':'paused diagnostic; planning/API waiting excluded from sim clock; not realtime validation',
        'fault_distribution':'synthetic commanded closure loss or one-shot perceived-height error; not real failure rates',
        'geometry_source':'simulation oracle','hardware_access':False,'transfer_ready':False})


def audit_trial(folder,result):
    """Check preserved control-time continuity and donor-hold release ordering."""
    last=None;count=0;loss_count=0;release_count=0
    for stage in result.get('stages',[]):
        path=folder/stage['native']/'transitions.jsonl'
        for line in path.read_text().splitlines():
            r=json.loads(line);before=r['observation'];after=r['next_observation']
            if last is not None and before!=last:raise ValueError('state/time discontinuity')
            if after['tick']!=before['tick']+1 or after['sim_time']<=before['sim_time']:
                raise ValueError('non-monotone native time')
            d=r.get('diagnostics') or {}
            if d.get('diagnostic_actuator_command_loss'):loss_count+=1
            if d.get('phase')=='release_right':
                release_count+=1
                if not r['evaluation']['held']['left']:raise ValueError('donor release without receiver hold')
            count+=1;last=after
    return {'joint_state_and_time_continuous':True,'native_transitions':count,
        'injected_command_frames':loss_count,'verified_donor_release_frames':release_count,
        'object_pose_edit_exclusion':'source audit plus private native feedback; joint continuity alone is not proof'}


def evaluate(folder):
    protocol=json.loads((folder/'protocol.json').read_text())
    if protocol['source_hashes']!=sources():raise ValueError('runtime changed after protocol freeze')
    for row in [*protocol['programs'].values(),*protocol['cases']]:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('frozen input changed')
    def trial(method,case):
        p=ROOT/case['path'];value=json.loads(p.read_text());out=folder/'heldout'/method/str(value['seed'])
        program=ROOT/protocol['programs']['condensed' if method=='condensed' else 'no_recovery']['path']
        r=subprocess_trial(p,out,program,'fixed' if method=='fixed' else 'frozen')
        r.update(method=method,case=value,group=case['group'],trial=str(out.relative_to(ROOT)))
        try:r['audit']=audit_trial(out,r)
        except Exception as exc:r.update(task_success=False,audit_error=str(exc))
        return r
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(trial,m,c) for c in protocol['cases'] for m in protocol['methods']]
        for job in as_completed(jobs):
            r=job.result();rows.append(r);dump(folder/'results.json',rows)
            print(json.dumps({'method':r['method'],'seed':r['case']['seed'],
                'success':r['task_success'],'failure':r.get('native_failures'),'audit_error':r.get('audit_error')}),flush=True)
    if protocol['source_hashes']!=sources():raise ValueError('runtime changed during evaluation')
    assessment={m:{g:{'passed':sum(r['task_success'] is True for r in rows if r['method']==m and r['group']==g),
        'total':sum(r['method']==m and r['group']==g for r in rows)}
        for g in ('none','right_close_loss','stale_grasp','receiver_close_loss')} for m in protocol['methods']}
    dump(folder/'assessment.json',assessment)
    print(json.dumps(assessment),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','evaluate'))
    p.add_argument('folder',type=Path);a=p.parse_args();f=a.folder.resolve()
    if not f.is_relative_to(ROOT/'runs'):p.error('outputs must stay inside runs/')
    (prepare if a.command=='prepare' else evaluate)(f)
