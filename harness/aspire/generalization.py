"""Three-way frozen comparison: fixed commands, geometry tools, model program."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .campaign import ROOT, write


def execute(folder, method, case, output):
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('never overwrite an existing trial')
    if method=='fixed':
        command=[sys.executable,'-B','-m','harness.aspire.fixed','--nominal',
                 str(folder/'development/v01/051'),'--case',str(case),'--output',str(output)]
    else:
        program=folder/('skill/program.py' if method=='skill' else 'baseline.py')
        command=[sys.executable,'-B','-m','harness.aspire.runtime','--program',str(program),
                 '--case',str(case),'--output',str(output)]
    started=time.monotonic();timed_out=False
    with output.with_suffix('.log').open('w') as log:
        try:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl'),
                           stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=60,check=False)
        except subprocess.TimeoutExpired:timed_out=True
    result_path=output/'result.json'
    row=json.loads(result_path.read_text()) if result_path.exists() else {
        'task_success':False,'failure_code':'wall_budget' if timed_out else 'process_error',
        'incomplete_native_files_preserved':True}
    row.update(case=json.loads(case.read_text()),wall_seconds=time.monotonic()-started,
               trial=str(output),method=method)
    return row


def trajectory_evidence(folder):
    rows=[]
    for aggregate in (folder/'heldout/results.json',folder/'wide/skill/results.json'):
        if not aggregate.exists():continue
        for row in json.loads(aggregate.read_text()):
            trial=Path(row['trial']);raw=json.loads((trial/'result.json').read_text()) if (trial/'result.json').exists() else {}
            picks=[s for s in raw.get('stages',[]) if s['tool']=='pick' and s.get('native_run')]
            if not picks:continue
            native=trial/picks[0]['native_run']
            summary=json.loads((native/'summary.json').read_text())
            selected=summary['metrics']['tamp'].get('grasp_search',{}).get('selected')
            actions=[json.loads(line)['action'] for line in (native/'transitions.jsonl').read_text().splitlines()]
            rows.append({'case':row['case'],'grasp':selected,
                'pick_commands_sha256':hashlib.sha256(np.asarray(actions,np.float32).tobytes()).hexdigest(),
                'pick_steps':len(actions),'task_success':row.get('task_success')})
    hashes={r['pick_commands_sha256'] for r in rows}
    result={'cases':rows,'distinct_pick_command_sequences':len(hashes),'checked_cases':len(rows),
            'source':'native requested actuator commands, not LLM self-report',
            'meaning':'different observed bottle positions actually produce different grasps/commands; this alone does not attribute the adaptation to skill learning'}
    write(folder/'trajectory-evidence.json',result)


def main(folder):
    nominal=json.loads((folder/'fixed-nominal/result.json').read_text())
    if not nominal.get('task_success') or nominal.get('final_robot_state_error_to_nominal_rad',1)>1e-4:
        raise ValueError('fixed replay must reproduce nominal success and robot state before comparison')
    protocol=json.loads((folder/'wide-protocol.json').read_text())
    all_rows={}
    for method in ('skill','geometry','fixed'):
        rows=[]
        for case_path in protocol['cases']:
            case=Path(case_path);seed=json.loads(case.read_text())['seed']
            output=folder/'wide'/method/f'{seed:03d}'
            row=execute(folder,method,case,output);rows.append(row)
            write(output.parent/'results.json',rows)
            print(json.dumps({'method':method,'seed':seed,'passed':row['task_success'],
                'failure':row.get('failure_code',row.get('error'))}),flush=True)
        all_rows[method]=rows
    # Fixed commands are also measured at the same20original held-out positions.
    fixed=[]
    for case in sorted((folder/'cases/heldout').glob('*.json')):
        seed=json.loads(case.read_text())['seed']
        row=execute(folder,'fixed',case,folder/'fixed-heldout'/f'{seed:03d}')
        fixed.append(row);write(folder/'fixed-heldout/results.json',fixed)
    summary={name:{'passes':sum(r['task_success'] is True for r in rows),'total':len(rows)}
             for name,rows in all_rows.items()}
    summary.update(fixed_original_heldout={'passes':sum(r['task_success'] is True for r in fixed),'total':len(fixed)},
                   original_program_frozen=True,original_skill_tests_unchanged=True,
                   no_trial_filtering=True,position_domain_m=protocol['region_m'],
                   per_trial_wall_budget_s=60,hardware_access=False,transfer_ready=False)
    write(folder/'wide-result.json',summary)
    trajectory_evidence(folder)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    args=parser.parse_args();folder=args.folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):parser.error('output must stay inside runs/')
    main(folder)
