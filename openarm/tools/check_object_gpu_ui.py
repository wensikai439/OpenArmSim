"""Read-only collection and field-level verification of the real UI GPU task."""
import hashlib
import json
import sqlite3
import time
import urllib.request

from sim.environment import ROOT
from sim.recording import dump

OUT = ROOT / 'runs/20261009-object-gpu'


def get(name):
    with urllib.request.urlopen('http://127.0.0.1:8767/api/' + name, timeout=10) as response:
        return json.load(response)


def score(record, protocol):
    plan = record['terminal']['agent']['plan']; goal = plan.get('placement_goal', {})
    ident = protocol['object_id']; arm = protocol['arm']
    initial = next(o for o in protocol['initial']['diagnostics']['scene_directory']['objects'] if o['object_id'] == ident)['origin_m'][:2]
    target = [initial[0] + protocol['forward_m'], initial[1]]
    def same(a, b): return len(a) == len(b) == 2 and all(abs(x - y) <= 1e-6 for x, y in zip(a, b))
    events = [e for t in record['turns'] for e in t['trace'] if e['kind'] == 'tool' and e['status'] == 'completed']
    selections = [e for e in events if e['name'] == 'select_manipulation_object']
    motion_names = {'pick', 'place', 'move_relative', 'move_to_pose', 'handover', 'relocate', 'set_gripper', 'run_policy'}
    actions = [e for e in events if e['name'] in motion_names]; receipts = record['receipts']
    gpu = next((r['feedback'] for r in receipts if r['execution']['skill'] == 'move_relative'), {})
    gpu_args = next((r['arguments'] for r in receipts if r['execution']['skill'] == 'move_relative'), {})
    proposals = gpu.get('plans', []); place = receipts[-1]['feedback'] if receipts else {}
    sources = ['native_two_finger_contact_and_lift', 'robot_forward_kinematics', 'native_support_release_and_stability']
    checks = {'one_correct_object_selection': len(selections) == 1 and selections[0]['detail']['arguments']['object_id'] == ident
            and selections[0]['detail']['result'].get('physical_state_preserved') is True,
        'exact_three_native_tools': [e['name'] for e in actions] == protocol['expected_native_tools'] and len(receipts) == 3,
        'same_original_object_and_target': goal.get('object_id') == ident and goal.get('surface') == 'table'
            and same(goal.get('target_xy_m', []), target) and same(goal.get('anchor', {}).get('object_xy_m', []), initial),
        'correct_native_objects_arms_sources': len(receipts) == 3 and all(r['feedback'].get('object_id') == ident
            and r['arguments'].get('arm') == arm and r['feedback'].get('primitive_success') is True
            and r['feedback'].get('verification_source') == source for r, source in zip(receipts, sources)),
        'actual_GPU_carry_and_requested_delta': bool(proposals) and gpu_args.get('planner') == 'curobo'
            and gpu_args.get('frame') == 'world' and gpu_args.get('delta_m') == protocol['GPU_TCP_delta_m']
            and gpu.get('payload_state') == 'single_hold' and all(p['success'] and p['attachment']['object_id'] == ident
                and p['metadata']['protocol'] == 'openarm-object-curobo/1' for p in proposals),
        'frozen_GPU_source_and_no_substitution': bool(proposals) and gpu.get('fallback') is False
            and gpu.get('native_path_revalidation') is True and all(p['metadata']['worker_source_sha256']
                == protocol['source_sha256']['tools/object_gpu.py'] for p in proposals),
        'original_TCP_motion_gate': gpu.get('position_error_m', 1) <= .012 and gpu.get('rotation_error_rad', 1) <= .08,
        'native_placement_target_release_support': place.get('placement', {}).get('confirmed') is True
            and place['placement'].get('object_id') == ident and same(place['placement'].get('xy_m', []), target)
            and place['placement'].get('stable_seconds', 0) >= 1 and place['placement'].get('tolerance_m', 1) <= .01
            and place.get('position_error_m', 1) <= .01,
        'correct_execution_bindings': len(actions) == len(receipts) == 3 and all(
            e['detail']['result'].get('execution_id') == r['execution']['id'] for e, r in zip(actions, receipts)),
        'all_stages_confirmed': len(plan.get('steps', [])) >= 3 and all(s['status'] == 'done' for s in plan['steps']),
        'original_waiting_and_budget': plan.get('timing_mode') == protocol['timing_mode'] and plan.get('time_budget_s') == protocol['time_budget_s'],
        'formal_review_and_completed_turns': plan.get('verdict') in ('success', 'uncertain', 'failure')
            and bool(record['turns']) and all(t['status'] == 'completed' for t in record['turns']),
        'no_reset_or_prepare': not any(e['name'] == 'prepare_simulation' for e in events),
        'actual_object_manifests': len(record['manifests']) == 3 and all(m['task'] == 'Manipulate selected registered object: ' + ident
            and m['details'].get('object_selection', {}).get('object_id') == ident for m in record['manifests'])}
    return {'passed': all(checks.values()), 'checks': checks, 'expected_target_xy_m': target,
        'scope': 'predeclared registered-object GPU carry task; no generalization/calibration claim'}


def main():
    if (OUT / 'workbench.json').exists(): raise ValueError('retain previous UI evidence')
    protocol = json.loads((OUT / 'workbench.protocol.json').read_text())
    dump(OUT / 'workbench.collector.json', {'source_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
        'protocol_sha256': hashlib.sha256((OUT / 'workbench.protocol.json').read_bytes()).hexdigest(), 'commands': 0})
    root = None; history = []; key = None; deadline = time.monotonic() + 600
    while True:
        state = get('status'); public = get('agent')
        if root is None:
            candidates = [t for t in public['turns'] if t.get('parent') is None and t['created'] > protocol['after_created']]
            if candidates: root = min(candidates, key=lambda t: t['created'])['id']
        if root:
            turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
            plan = state['agent'].get('plan') or {}
            if plan and plan['turn_id'] != root: raise ValueError('another task owns this scene')
            current = (state['agent']['revision'], (state.get('job') or {}).get('execution_id'))
            if current != key:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'steps': plan.get('steps', []), 'job': state.get('job')}); key = current
                dump(OUT / 'workbench.progress.json', {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if not state.get('job') and not state['agent']['busy'] and plan.get('verdict') and all(t['status'] not in ('running', 'waiting') for t in turns): break
        if time.monotonic() > deadline: raise TimeoutError('retain partial task and inspect same live owner')
        time.sleep(.5)
    receipts = []
    with sqlite3.connect('file:' + str(ROOT / 'harness/state/objects.sqlite3') + '?mode=ro', uri=True) as database:
        for execution in plan.get('execution_tools', []):
            row = database.execute('SELECT arguments,feedback,evidence FROM experiences WHERE id=?', (execution['id'],)).fetchone()
            if row is None: raise ValueError('missing actual native receipt')
            receipts.append({'execution': execution, 'arguments': json.loads(row[0]), 'feedback': json.loads(row[1]), 'evidence': json.loads(row[2])})
    manifests = [json.loads((ROOT / 'runs' / r['evidence']['run_id'] / 'manifest.json').read_text()) for r in receipts]
    record = {'turn_id': root, 'turns': turns, 'terminal': state, 'receipts': receipts, 'manifests': manifests,
        'stage_history': history, 'memory': public['memory'], 'actual_api_calls': sum(e['kind'] == 'api' for t in turns for e in t['trace']),
        'commands': 0, 'hardware_access': False, 'transfer_ready': False}
    result = {**record, **score(record, protocol)}; dump(OUT / 'workbench.json', result)
    print(json.dumps({k: result[k] for k in ['passed', 'checks', 'actual_api_calls']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
