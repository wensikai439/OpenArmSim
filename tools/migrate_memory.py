"""Rebind proven native receipts; retain source contexts and forgetting rules."""
import copy
import json
import time
import urllib.request

from sim.environment import ROOT, OpenArmEnv
from sim.service import physical_memory_contract, memory_context
from sim.policy import load_config
from harness.workspace import Workspace
from harness.knowledge import consolidate


def eligible(source,current):
    required={'sim/environment.py','sim/contacts.py','sim/gripper.py','sim/profile.json',
              'calibrate.json','task/bottle_to_box.xml','sim/tamp.py'}
    return (source.get('kind')=='tamp' and required<=source.get('hashes',{}).keys()
            and physical_memory_contract(source)==physical_memory_contract(current)
            and source['hashes']['sim/tamp.py']==current['hashes']['sim/tamp.py'])


def migrate(work,previous,current,manifest,load_manifest):
    """Only the previously compatible bank, native TAMP, and unchanged verifier."""
    if previous['policy_contract']!=current['policy_contract']:
        raise ValueError('policy identity changed; do not bridge contexts')
    before_context,after_context=work._context(previous),work._context(current)
    with work.lock,work.db:
        if work.busy:raise ValueError('cannot migrate active work')
        rows=[dict(r) for r in work.db.execute('SELECT * FROM experiences WHERE context=?',(before_context,))]
        verified=[];skipped=[]
        for row in rows:
            evidence=json.loads(row['evidence']);run=evidence.get('run_id')
            args=json.loads(row['arguments']);feedback=json.loads(row['feedback'])
            if row['evidence_kind']!='simulator_oracle' or args.get('planner','tamp')!='tamp' or not run:
                skipped.append({'id':row['id'],'reason':'not a native TAMP receipt'});continue
            try:source,native_feedback=load_manifest(run)
            except (OSError,ValueError):
                skipped.append({'id':row['id'],'reason':'source manifest or native feedback missing'});continue
            if not eligible(source,manifest):
                skipped.append({'id':row['id'],'reason':'physics/data or native verifier changed'});continue
            if (not native_feedback or any(feedback.get(key)!=value for key,value in native_feedback.items())
                    or set(feedback)-set(native_feedback)-{'decision_timing_mode'}):
                skipped.append({'id':row['id'],'reason':'archived native feedback does not match receipt'});continue
            verified.append({**row,'arguments':args,'feedback':feedback,'evidence':evidence})
        proposed=[{**copy.deepcopy(row),'context':after_context} for row in verified]
        forgotten={tuple(r) for r in work.db.execute('SELECT role,id FROM forgotten_knowledge')}
        preserved_forgetting=0
        for role in {row['role'] for row in verified}:
            old_lessons=consolidate([r for r in verified if r['role']==role])
            new_lessons=consolidate([r for r in proposed if r['role']==role])
            for old in old_lessons:
                sources={s['receipt_id'] for s in old['sources']}
                if (role,old['id']) not in forgotten:continue
                for new in new_lessons:
                    if sources&{s['receipt_id'] for s in new['sources']}:
                        work.db.execute('INSERT OR REPLACE INTO forgotten_knowledge VALUES (?,?,?)',(role,new['id'],time.time()))
                        preserved_forgetting+=1
        for row in proposed:
            row['evidence']['context_migration']={'original_context':previous,'current_context':current,
                'basis':'native physics/data and TAMP verifier hashes match; archived native feedback matches receipt; original run/feedback/timestamp preserved'}
            key=work._lesson_key(row['tool'],row['arguments'],row['feedback'],after_context,row['evidence_kind'],row['outcome'])
            work.db.execute('UPDATE experiences SET context=?,evidence=?,lesson_key=? WHERE id=?',
                (after_context,json.dumps(row['evidence'],ensure_ascii=False),key,row['id']))
            work.db.execute('UPDATE episodes SET context=? WHERE id=? AND context=?',(after_context,row['id'],before_context))
        work.revision+=1
    return {'migrated':[row['id'] for row in proposed],'skipped':skipped,
            'forgetting_markers_preserved':preserved_forgetting,'raw_feedback_changed':False,
            'scene_notes_migrated':False,'policy_receipts_migrated':False}


def main():
    output=ROOT/'runs/20261008-carry/memory-migration.json'
    if output.exists():raise ValueError('preserve prior migration audit')
    status=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/status',timeout=5))
    if status['job'] or status['agent']['busy']:raise ValueError('platform task is active')
    previous=json.loads((ROOT/'runs/20261008-carry/baseline.json').read_text())['previous_memory_context']
    policy=load_config(ROOT/'harness/policy.json')
    env=OpenArmEnv(images=False)
    try:
        manifest=env.manifest();current=memory_context(manifest,policy,env.robot_version)
        work=Workspace(ROOT/'harness/state/agent.sqlite3')
        try:
            def load(run):
                if not __import__('re').fullmatch('[A-Za-z0-9-]+',run):raise ValueError('invalid archived run id')
                folder=ROOT/'runs'/run
                source=json.loads((folder/'manifest.json').read_text())
                summary=json.loads((folder/'summary.json').read_text())
                if source.get('run_id')!=run:raise ValueError('archived manifest identity mismatch')
                return source,summary.get('metrics',{}).get('tamp',{})
            result=migrate(work,previous,current,manifest,load)
            output.write_text(json.dumps({**result,'previous_context':previous,'current_context':current,
                'hardware_access':False,'transfer_ready':False},ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'migrated':len(result['migrated']),'skipped':len(result['skipped'])},ensure_ascii=False))
        finally:work.db.close()
    finally:env.close()


if __name__=='__main__':main()
