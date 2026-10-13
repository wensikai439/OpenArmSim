"""Read-only receipt collection for the predeclared multi-object browser task."""
import hashlib
import json
import sqlite3
import time
import urllib.request

from sim.environment import ROOT
from sim.recording import dump

OUT = ROOT / 'runs/20261009-objects'


def main():
    protocol = json.loads((OUT/'workbench.protocol.json').read_text())
    target = OUT/'workbench.json'
    if target.exists(): raise ValueError('preserve earlier browser task')
    dump(OUT/'workbench.collector.json', {'source_sha256': hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
        'protocol_sha256': hashlib.sha256((OUT/'workbench.protocol.json').read_bytes()).hexdigest(),
        'commands': 0, 'scope': 'read-only collection after UI submission; no unseen intermediate stages reconstructed'})
    def get(path):
        with urllib.request.urlopen('http://127.0.0.1:8767'+path, timeout=10) as response: return json.load(response)
    root = None; key = None; history = []; deadline = time.monotonic()+300
    while True:
        state = get('/api/status'); public = get('/api/agent')
        if root is None:
            roots = [t for t in public['turns'] if t.get('parent') is None and t['created']>protocol['after_created']]
            if roots: root = min(roots,key=lambda t:t['created'])['id']
        if root:
            turns = [t for t in public['turns'] if t['id']==root or t.get('parent')==root]
            plan = state['agent'].get('plan') or {}
            if plan and plan['turn_id'] != root: raise ValueError('another task owns the current plan')
            current = (state['agent']['revision'], (state.get('job') or {}).get('execution_id'))
            if current != key:
                history.append({'observation_id':state['observation_id'],'sim_time':state['observation']['sim_time'],
                    'steps':plan.get('steps',[]),'job':state.get('job')}); key = current
                dump(OUT/'workbench.progress.json', {'turn_id':root,'turns':turns,'terminal':state,'stage_history':history})
            if not state.get('job') and not state['agent']['busy'] and plan.get('verdict') and all(t['status'] not in ('running','waiting') for t in turns): break
        if time.monotonic()>deadline: raise TimeoutError('partial evidence retained; no reset or restart')
        time.sleep(.5)
    receipts = []
    with sqlite3.connect('file:'+str(ROOT/'harness/state/objects.sqlite3')+'?mode=ro',uri=True) as database:
        for execution in plan.get('execution_tools',[]):
            row = database.execute('SELECT arguments,feedback,evidence FROM experiences WHERE id=?',(execution['id'],)).fetchone()
            if row is None: raise ValueError('missing native receipt')
            receipts.append({'execution':execution,'arguments':json.loads(row[0]),'feedback':json.loads(row[1]),'evidence':json.loads(row[2])})
    events = [e for t in turns for e in t['trace'] if e['kind']=='tool']
    selection = [e for e in events if e['name']=='select_manipulation_object' and e['status']=='completed']
    actions = [e['name'] for e in events if e['name'] in {'pick','place','handover','relocate','move_relative','move_to_pose','run_policy','set_gripper'} and e['status']=='completed']
    goal = plan.get('placement_goal',{}); manifests = []
    for receipt in receipts:
        run = receipt['evidence']['run_id']; manifests.append(json.loads((ROOT/'runs'/run/'manifest.json').read_text()))
    checks = {'one_requested_object_selection':len(selection)==1 and selection[0]['detail']['arguments']['object_id']==protocol['object_id']
        and selection[0]['detail']['result']['physical_state_preserved'],
        'exact_requested_native_tools':actions==protocol['expected_native_tools'],
        'correct_native_object_receipts':len(receipts)==2 and all(r['feedback'].get('object_id')==protocol['object_id'] and r['feedback'].get('primitive_success') for r in receipts),
        'correct_selected_arm':all(r['arguments'].get('arm')==protocol['arm'] for r in receipts),
        'locked_original_object_goal':goal.get('object_id')==protocol['object_id'] and goal.get('offset_xy_m')==[protocol['forward_m'],0],
        'physical_placement_confirmed':plan.get('goal_assessment',{}).get('confirmed') is True,
        'declared_waiting_and_budget':plan.get('timing_mode')==protocol['timing_mode'] and plan.get('time_budget_s')==protocol['time_budget_s'],
        'all_stages_done':bool(plan.get('steps')) and all(s['status']=='done' for s in plan['steps']),
        'all_turns_completed':all(t['status']=='completed' for t in turns),
        'no_prepare_or_reset':not any(e['name']=='prepare_simulation' and e['status']=='completed' for e in events),
        'run_manifests_name_actual_object':all(m['details'].get('object_selection',{}).get('object_id')==protocol['object_id']
            and m['task']=='Manipulate selected registered object: '+protocol['object_id'] for m in manifests)}
    result = {'passed':all(checks.values()),'checks':checks,'turn_id':root,'turns':turns,'receipts':receipts,
        'terminal':state,'stage_history':history,'manifests':manifests,'memory':public['memory'],
        'actual_api_calls':sum(e['kind']=='api' for t in turns for e in t['trace']),
        'commands':0,'hardware_access':False,'transfer_ready':False}
    dump(target,result);print(json.dumps({'passed':result['passed'],'checks':checks,'API_calls':result['actual_api_calls']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
