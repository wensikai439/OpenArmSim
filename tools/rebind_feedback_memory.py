"""Bridge only four frozen receipts after proving this passive feedback change.

Never generalize the exception to an arbitrary environment-code change. The
original feedback, timestamp, run, user forgetting and source context persist.
"""
import copy
import hashlib
import json
import sqlite3
import time
import urllib.request

import numpy as np

from sim.environment import ROOT,OpenArmEnv
from sim.service import physical_memory_contract
from sim.recording import dump
from harness.workspace import Workspace
from harness.knowledge import consolidate


FILES=['api-forward-verified.json','api-transfer.json','api-curobo.json','api-natural-transfer-verified.json']
OUT=ROOT/'runs/20261008-hold/memory'


def fetch(route):
    with urllib.request.urlopen('http://127.0.0.1:8765/api/'+route,timeout=10) as r:return json.load(r)


def equivalent(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,bool):return type(b) is bool and a==b
    if type(a) in (float,int):return type(b) in (float,int) and abs(a-b)<1e-9
    return a==b


def main():
    if OUT.exists():raise ValueError('preserve previous proof/migration')
    status=fetch('status');assert not status['job'] and not status['agent']['busy']
    current=status['memory_context'];manifest=fetch('manifest')
    previous=json.loads((OUT.parent/'pre-deploy-status.json').read_text())['memory_context']
    assert current['policy_contract']==previous['policy_contract']
    OUT.mkdir();proofs=[];verified=[]
    for file in FILES:
        source=ROOT/'runs/20261008-relocate'/file;report=json.loads(source.read_text())
        assert report['status']=='passed';assert report['terminal']['memory_context']==previous
        entry=report['feedbacks'][0];ident=entry['execution']['id'];run=entry['evidence']['run_id']
        paths=list((ROOT/'runs'/('20261008-agent-'+report['turn_id'][:8])/'waits').glob('*/manifest.json'))
        paths.append(ROOT/'runs'/run/'manifest.json');paths.sort(key=lambda p:json.loads(p.read_text())['start_tick'])
        env=OpenArmEnv(images=False);env.reset(seed=0);env.set_task_budget(report['terminal']['agent']['plan']['time_budget_s'])
        goal=report['terminal']['agent']['plan']['placement_goal'];maximum=0.;ticks=0;bindings=[]
        try:
            for path in paths:
                archived=json.loads(path.read_text());assert env.tick==archived['start_tick']
                assert archived['model_evidence']==manifest['model_evidence']
                before=physical_memory_contract(archived);after=physical_memory_contract(manifest)
                assert {k:v for k,v in before.items() if k!='hashes'}=={k:v for k,v in after.items() if k!='hashes'}
                changed=[k for k in set(before['hashes'])|set(after['hashes']) if before['hashes'].get(k)!=after['hashes'].get(k)]
                assert changed==['sim/environment.py'],changed
                # This exact deployed diff is the narrowly reviewed passive
                # feedback hook. Bind it to the backup and current file bytes.
                assert archived['hashes']['sim/environment.py']==hashlib.sha256((ROOT/'.runtime/backups/20261008-hold/sim/environment.py').read_bytes()).hexdigest()
                if archived['kind']=='tamp':
                    env.set_placement_target(goal['surface'],np.asarray(goal['target_xy_m']),goal['tolerance_m'])
                transitions=path.parent/'transitions.jsonl'
                bindings.append({'run':archived['run_id'],'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'transitions_sha256':hashlib.sha256(transitions.read_bytes()).hexdigest()})
                for line in transitions.read_text().splitlines():
                    transition=json.loads(line);_,_,term,trunc,info=env.step_tamp(np.asarray(transition['action']))
                    expected=transition['next_observation'];error=float(np.max(np.abs(env.state()-expected['state'])))
                    if error>1e-9 or env.tick!=expected['tick']:raise ValueError('state/tick differs')
                    if not equivalent(info['evaluation'],transition['evaluation']):raise ValueError('native scoring differs')
                    assert bool(term)==transition['terminated'] and bool(trunc)==transition['truncated']
                    maximum=max(maximum,error);ticks+=1
            body_error=float(np.linalg.norm(env.data.xpos[env.bottle_id]-report['terminal']['diagnostics']['manipulation_scene']['position_m']))
            if body_error>1e-9:raise ValueError('terminal bottle position differs')
            proof={'receipt_id':ident,'run_id':run,'report':str(source.relative_to(ROOT)),
                'report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'commands':ticks,
                'max_robot_state_error':maximum,'terminal_bottle_error_m':body_error,
                'every_native_evaluation_and_terminal_flag_matches':True,'bound_archives':bindings,
                'same_compiled_inputs_and_runtime_parameters':True,'scope':'only this frozen declared trajectory; not physical calibration or arbitrary-code equivalence'}
            proof['current_feedback_code_sha256']=hashlib.sha256((ROOT/'sim/grasp_feedback.py').read_bytes()).hexdigest()
            proof['current_environment_sha256']=hashlib.sha256((ROOT/'sim/environment.py').read_bytes()).hexdigest()
            proofs.append(proof);verified.append((ident,entry['feedback'],run));print(json.dumps(proof,ensure_ascii=False),flush=True)
        finally:env.close()
    dump(OUT/'proofs.json',proofs)
    status=fetch('status');assert not status['job'] and not status['agent']['busy'] and status['memory_context']==current
    work=Workspace(ROOT/'harness/state/agent.sqlite3')
    try:
        backup_path=ROOT/'.runtime/backups/20261008-hold/memory-before.sqlite3'
        with sqlite3.connect(backup_path) as backup:work.db.backup(backup)
        backup_path.chmod(0o600)
        before=work._context(previous);after=work._context(current)
        with work.lock,work.db:
            old=[];new=[]
            for ident,feedback,run in verified:
                row=dict(work.db.execute('SELECT * FROM experiences WHERE id=? AND context=?',(ident,before)).fetchone())
                assert json.loads(row['feedback'])==feedback and row['tool']=='relocate' and row['evidence_kind']=='simulator_oracle'
                for field in ('arguments','feedback','evidence'):row[field]=json.loads(row[field])
                assert row['evidence']['run_id']==run
                old.append(row);proposed=copy.deepcopy(row);proposed['context']=after;new.append(proposed)
            forgotten={tuple(r) for r in work.db.execute('SELECT role,id FROM forgotten_knowledge')};forget_count=0
            for old_lesson in consolidate(old):
                if (old[0]['role'],old_lesson['id']) not in forgotten:continue
                receipts={s['receipt_id'] for s in old_lesson['sources']}
                for new_lesson in consolidate(new):
                    if receipts&{s['receipt_id'] for s in new_lesson['sources']}:
                        work.db.execute('INSERT OR REPLACE INTO forgotten_knowledge VALUES (?,?,?)',(old[0]['role'],new_lesson['id'],time.time()));forget_count+=1
            for row in new:
                row['evidence']['context_migration']={'original_context':previous,'current_context':current,
                    'proof':'/runs/20261008-hold/memory/proofs.json',
                    'basis':'same frozen model and parameters; exact archived command re-execution under passive feedback addition; sampled native evaluation and terminal object match',
                    'physical_calibration':False}
                key=work._lesson_key(row['tool'],row['arguments'],row['feedback'],after,row['evidence_kind'],row['outcome'])
                work.db.execute('UPDATE experiences SET context=?,evidence=?,lesson_key=? WHERE id=?',(after,json.dumps(row['evidence'],ensure_ascii=False),key,row['id']))
                work.db.execute('UPDATE episodes SET context=? WHERE id=? AND context=?',(after,row['id'],before))
        result={'receipts':[r['id'] for r in new],'previous_context':previous,'current_context':current,
                'proofs':'proofs.json','forgetting_markers_preserved':forget_count,'original_feedback_and_time_changed':False,
                'policy_receipts_migrated':False,'scene_notes_migrated':False,'other_archives_migrated':False,
                'hardware_access':False,'transfer_ready':False}
        dump(OUT/'migration.json',result);print(json.dumps({'migrated':len(new),'forgetting_preserved':forget_count}),flush=True)
    finally:work.db.close()


if __name__=='__main__':main()
