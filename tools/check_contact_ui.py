"""Read-only collection of one browser-submitted candidate task."""
import argparse
import hashlib
import json
import sqlite3
import time
import urllib.request
from pathlib import Path
from sim.recording import dump
from sim.environment import ROOT

OUT=ROOT/'runs/20261008-grip'


def collect(after_turn):
    target=OUT/'workbench-current.json'
    if target.exists():raise ValueError('preserve previous browser task')
    protocol=OUT/'workbench.protocol.json'
    declaration={'browser_protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'expected_initial_xy_m':[.503,-.173],'expected_target_xy_m':[.563,-.173],
        'recipient_arm':'right','donor_arm':'left','budget_s':120,'timing_mode':'measured_delay',
        'after_turn':after_turn,'collector_commands':0,'provider_mode':'official DeepSeek non-thinking; same1600 output cap',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'hardware_access':False,
        'collection_revision':'native completion and formal review; composite or separately verified primitives',
        'previous_partial_record':'workbench-early.json',
        'limitation':'resumed read-only collection after premature first collector exit; missing intermediate live milestones are not reconstructed'}
    dump(OUT/'workbench-collector-current.protocol.json',declaration)
    def get(path):
        with urllib.request.urlopen('http://127.0.0.1:8766'+path,timeout=10) as response:return json.load(response)
    started=time.monotonic();root=None;stage_key=None;revision=None;published_length=None;history=[]
    early=OUT/'workbench-early.json'
    if early.exists():history=json.loads(early.read_text()).get('stage_history',[])
    while time.monotonic()-started<600:
        state=get('/api/status');public=get('/api/agent')
        if root is None:
            roots=[t for t in public['turns'] if t['parent'] is None and t['id']!=after_turn]
            if roots:root=max(roots,key=lambda t:t['created'])['id']
            else:time.sleep(.25);continue
        turns=[t for t in public['turns'] if t['id']==root or t['parent']==root]
        plan=state['agent'].get('plan') or {};job=state.get('job') or {}
        milestones=[(s['id'],s['status']) for s in plan.get('steps',[])]
        key=(job.get('execution_id'),tuple(milestones))
        if key!=stage_key:
            history.append({'observation_id':state['observation_id'],'sim_time':state['observation']['sim_time'],
                'outer_execution_id':job.get('execution_id'),'outer_running':bool(job),
                'stage_bindings':plan.get('stage_bindings',{}),'steps':milestones,
                'completed_substeps':job.get('completed_substeps',[])})
            stage_key=key
        if revision!=state['agent']['revision'] or len(history)!=published_length:
            dump(target.with_suffix('.progress.json'),{'turn_id':root,'turns':turns,'state':state,'stage_history':history,
                'collector_commands':0,'hardware_access':False})
            revision=state['agent']['revision'];published_length=len(history)
        finished=(not job and not state['agent']['busy'] and turns
            and all(t['status'] not in ('running','waiting') for t in turns))
        # A native completion is queued for asynchronous continuation/review.
        # The brief gap before its child turn appears is not task completion.
        review=plan.get('verdict') in ('success','failure','uncertain')
        no_execution=not plan.get('executions')
        terminal_error=any(t['status'] in ('error','interrupted','budget_exhausted') for t in turns)
        if finished and (review or no_execution or terminal_error):break
        time.sleep(.25)
    else:raise TimeoutError('read-only collector timed out; partial evidence retained')
    last=state.get('last_run') or {};feedback=last.get('metrics',{}).get('tamp',{});receipts=[]
    with sqlite3.connect('file:'+str(ROOT/'harness/state/contact.sqlite3')+'?mode=ro',uri=True) as database:
        for execution in plan.get('execution_tools',[]):
            row=database.execute('SELECT arguments,feedback,evidence FROM experiences WHERE id=?',(execution['id'],)).fetchone()
            if row:receipts.append({'execution':execution,'parameters':json.loads(row[0]),'feedback':json.loads(row[1]),'evidence':json.loads(row[2])})
    children=[]
    for receipt in receipts:
        if receipt['execution']['skill']=='relocate':children.extend(receipt['feedback'].get('substeps',[]))
        else:
            f=receipt['feedback'];p=receipt['parameters']
            children.append({'skill':receipt['execution']['skill'],'parameters':p,'feedback':f,
                'status':'completed' if f.get('primitive_success') else 'failed','backend':p.get('planner','tamp')})
    steps=children
    target_unchanged=(plan.get('placement_goal') or {}).get('target_xy_m')==[.563,-.173]
    curobo=any(s.get('backend')=='curobo' and s['status']=='completed' and s['feedback'].get('fallback') is False
        and s['feedback'].get('native_path_revalidation') and s['feedback'].get('verification_source')=='robot_forward_kinematics'
        for s in steps)
    transfer=any(s['skill']=='handover' and s['status']=='completed' and s['parameters'].get('from_arm')=='left'
        and s['parameters'].get('to_arm')=='right' for s in steps)
    grasp=any(s['skill']=='pick' and s['status']=='completed' and s['parameters'].get('arm')=='left'
        and s['feedback'].get('verification_source')=='native_two_finger_contact_and_lift' for s in steps)
    native=bool(grasp and feedback.get('primitive_success') and (plan.get('goal_assessment') or {}).get('confirmed') and target_unchanged
        and curobo and transfer and plan.get('time_budget_s')==120 and plan.get('timing_mode')=='measured_delay')
    review=plan.get('verdict') in ('success','failure','uncertain')
    good=native and review and all(t['status']=='completed' for t in turns)
    result={'status':'passed' if good else 'failed','turn_id':root,'turns':turns,'terminal':state,'native_feedback':feedback,
        'native_goal_confirmed':native,'verified_curobo':bool(curobo),'verified_transfer':transfer,
        'target_unchanged':target_unchanged,'review_complete':review,'model_verdict':plan.get('verdict'),
        'actual_api_calls':sum(e['kind']=='api' for t in turns for e in t.get('trace',[])),
        'feedbacks':receipts,'stage_history':history,'collector_wall_seconds':time.monotonic()-started,'collector_commands':0,
        'collector_revision':declaration,'collection_only_resumed':True,
        'hardware_access':False,'transfer_ready':False,'scope':'one browser-submitted declared candidate case; not a statistical success rate'}
    dump(target,result);print(json.dumps({k:result[k] for k in ('status','actual_api_calls','native_goal_confirmed','model_verdict')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--after-turn',required=True);args=parser.parse_args();collect(args.after_turn)
