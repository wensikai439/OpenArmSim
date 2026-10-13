"""Read-only collection of the browser-submitted API relay and real receipts."""
import json
import argparse
import sqlite3
import time

from tools.check_object_gpu_ui import get
from sim.environment import ROOT
from sim.recording import dump


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='workbench'); args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple trial name required')
    out = ROOT / 'runs/20261009-handover'; target = out / (args.name + '.json')
    if target.exists(): raise ValueError('retain previous browser task evidence')
    protocol = json.loads((out / (args.name + '.protocol.json')).read_text())
    root = None; history = []; key = None; deadline = time.monotonic() + 600
    while True:
        state = get('status'); public = get('agent')
        if root is None:
            candidates = [turn for turn in public['turns'] if turn.get('parent') is None and turn['created'] > protocol['after_created']]
            if candidates: root = min(candidates, key=lambda turn: turn['created'])['id']
        if root:
            turns = [turn for turn in public['turns'] if turn['id'] == root or turn.get('parent') == root]
            plan = state['agent'].get('plan') or {}
            if plan and plan['turn_id'] != root: raise ValueError('another task owns this scene')
            current = (state['agent']['revision'], (state.get('job') or {}).get('execution_id'))
            if current != key:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'steps': plan.get('steps', []), 'job': state.get('job')}); key = current
                dump(out / (args.name + '.progress.json'), {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if not state.get('job') and not state['agent']['busy'] and plan.get('verdict') and all(turn['status'] not in ('running', 'waiting') for turn in turns): break
        if time.monotonic() > deadline: raise TimeoutError('inspect the same live owner; never reset a partial task')
        time.sleep(.5)
    receipts = []
    with sqlite3.connect('file:' + str(ROOT / 'harness/state/objects.sqlite3') + '?mode=ro', uri=True) as db:
        for execution in plan.get('execution_tools', []):
            row = db.execute('SELECT arguments,feedback,evidence FROM experiences WHERE id=?', (execution['id'],)).fetchone()
            if row is None: raise ValueError('missing native execution receipt')
            receipts.append({'execution': execution, 'arguments': json.loads(row[0]), 'feedback': json.loads(row[1]), 'evidence': json.loads(row[2])})
    events = [event for turn in turns for event in turn['trace'] if event['kind'] == 'tool']
    actions = [event for event in events if event['name'] in
        {'pick', 'handover', 'place', 'relocate', 'move_relative', 'move_to_pose', 'run_policy', 'set_gripper'} and event['status'] == 'completed']
    manifests = [json.loads((ROOT / 'runs' / receipt['evidence']['run_id'] / 'manifest.json').read_text()) for receipt in receipts]
    feedback = next((receipt['feedback'] for receipt in receipts if receipt['execution']['skill'] == 'handover'), {})
    checks = {'exact_native_route': [event['name'] for event in actions] == protocol['expected_tools'] and len(receipts) == 2,
        'actual_can_and_requested_direction': feedback.get('object_id') == protocol['object_id']
            and feedback.get('from_arm') == protocol['from_arm'] and feedback.get('to_arm') == feedback.get('held_by') == protocol['to_arm']
            and feedback.get('execution_binding', {}).get('valid') is True,
        'native_relay_confirmed': feedback.get('primitive_success') is True
            and feedback.get('verification_source') == 'native_receiver_contact_then_donor_release'
            and feedback.get('independent_hold_seconds', 0) >= .999,
        'actual_left_only_hold': state['diagnostics']['evaluation']['held'] == {'left': True, 'right': False},
        'non_target_objects_undisturbed': state['diagnostics']['evaluation']['non_target_object_guard']['passed'],
        'correct_execution_bindings': len(actions) == len(receipts) == 2 and all(
            event['detail']['result']['execution_id'] == receipt['execution']['id'] for event, receipt in zip(actions, receipts)),
        'all_stages_done': bool(plan.get('steps')) and all(step['status'] == 'done' for step in plan['steps']),
        'formal_review_and_completed_turns': plan.get('verdict') in ('success', 'failure', 'uncertain')
            and bool(turns) and all(turn['status'] == 'completed' for turn in turns),
        'original_timing_contract': plan.get('timing_mode') == protocol['timing_mode'] and plan.get('time_budget_s') == protocol['time_budget_s'],
        'no_reset_or_prepare': not any(event['name'] == 'prepare_simulation' for event in events),
        'unchanged_compiled_parameters': len(manifests) == 2 and manifests[0]['model_evidence'] == manifests[1]['model_evidence']}
    result = {'passed': all(checks.values()), 'checks': checks, 'turn_id': root, 'turns': turns,
        'receipts': receipts, 'manifests': manifests, 'terminal': state, 'stage_history': history,
        'actual_api_calls': sum(event['kind'] == 'api' for turn in turns for event in turn['trace']),
        'memory': public['memory'], 'collector_commands': 0, 'hardware_access': False, 'transfer_ready': False}
    dump(target, result); print({'passed': result['passed'], 'checks': checks, 'api_calls': result['actual_api_calls']}, flush=True)


if __name__ == '__main__': main()
