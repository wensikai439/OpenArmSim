"""Read-only recovery after collector KeyError; never rerun physics or API."""
import json

from sim.environment import ROOT
from sim.recording import dump
from harness.workspace import redact


def main():
    out = ROOT / 'runs/20261009-handover/api-relay'; target = out / 'result.json'
    if target.exists(): raise ValueError('keep an existing result')
    raw = json.loads((out / 'progress.json').read_text()); turns = raw['turns']; state = raw['terminal']
    protocol = json.loads((out / 'protocol.json').read_text())
    events = [event for turn in turns for event in turn['trace'] if event['kind'] == 'tool']
    actions = [event for event in events if event['name'] in
        {'pick', 'handover', 'place', 'relocate', 'move_relative', 'move_to_pose', 'run_policy', 'set_gripper'} and event['status'] == 'completed']
    receipts = []
    for action in actions:
        execution = action['detail']['result']['execution_id']
        status = next(event['detail']['result'] for event in events if event['name'] == 'get_run_status'
            and any(r['execution_id'] == execution for r in event['detail']['result'].get('task_progress', {}).get('receipts', []))
            and event['detail']['result'].get('tool_feedback', {}).get('task') in
                {'pick': ('pick_bottle', 'pick_object'), 'handover': ('handover_bottle', 'handover_object')}[action['name']])
        folder = ROOT / 'runs' / status['run_id']; manifest = json.loads((folder / 'manifest.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        if redact(summary['metrics']['tamp']) != status['tool_feedback']: raise ValueError('trace/native feedback mismatch after journal redaction')
        if manifest['details']['execution_id'] != execution: raise ValueError('native execution identity mismatch')
        receipts.append({'execution': {'id': execution, 'skill': action['name']}, 'feedback': status['tool_feedback'],
            'arguments': action['detail']['arguments'], 'run_id': status['run_id'], 'manifest': manifest})
    feedback = next((r['feedback'] for r in receipts if r['execution']['skill'] == 'handover'), {})
    steps = raw['stage_history'][-1]['steps']; plans = [event['detail']['arguments'] for event in events if event['name'] == 'set_plan' and event['status'] == 'completed']
    checks = {'exact_native_route': [event['name'] for event in actions] == protocol['expected_tools'] and len(receipts) == 2,
        'correct_object_and_direction': feedback.get('object_id') == protocol['object_id'] and feedback.get('from_arm') == protocol['from_arm']
            and feedback.get('to_arm') == feedback.get('held_by') == protocol['to_arm']
            and receipts[-1]['arguments']['from_arm'] == protocol['from_arm'] and receipts[-1]['arguments']['to_arm'] == protocol['to_arm'],
        'native_relay_confirmed': feedback.get('primitive_success') is True
            and feedback.get('verification_source') == 'native_receiver_contact_then_donor_release',
        'native_receiver_still_holds': state['diagnostics']['evaluation']['held'] == {'left': False, 'right': True},
        'non_target_objects_undisturbed': state['diagnostics']['evaluation']['non_target_object_guard']['passed'],
        'all_stages_done': bool(steps) and all(step['status'] == 'done' for step in steps),
        'all_turns_completed': bool(turns) and all(turn['status'] == 'completed' for turn in turns),
        'original_timing_contract': bool(plans) and plans[0].get('timing_mode') == protocol['timing_mode']
            and plans[0].get('time_budget_s') == protocol['time_budget_s'],
        'no_reset_or_object_switch_after_motion': sum(event['name'] == 'select_manipulation_object' and event['status'] == 'completed' for event in events) == 1
            and not any(event['name'] == 'prepare_simulation' and event['status'] == 'completed' for event in events),
        'model_parameters_unchanged': len(receipts) == 2 and receipts[0]['manifest']['model_evidence'] == receipts[1]['manifest']['model_evidence']}
    result = {'passed': all(checks.values()), 'checks': checks, 'turn_id': raw['turn_id'], 'turns': turns,
        'receipts': receipts, 'terminal': state, 'stage_history': raw['stage_history'],
        'actual_api_calls': sum(event['kind'] == 'api' for turn in turns for event in turn['trace']),
        'collector_failure': 'original collector KeyError: manifest after all turns completed',
        'recovery': 'read-only trace/native summary/manifest rescore; original log and progress retained; no new API or motion',
        'production_memory_writes': False, 'hardware_access': False, 'transfer_ready': False}
    dump(target, result); print({'passed': result['passed'], 'checks': checks, 'api_calls': result['actual_api_calls']})


if __name__ == '__main__': main()
