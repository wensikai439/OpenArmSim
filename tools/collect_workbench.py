"""Read-only collection of one exact UI prompt and its bound native receipts.

Never submits, cancels, resets or restarts a task. A timeout is collector-only;
the job's authoritative state must be checked before any subsequent action.
"""
import argparse
import hashlib
import json
import sqlite3
import time
import urllib.request

from sim.environment import ROOT
from sim.recording import dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('protocol', type=lambda x: (ROOT / x).resolve())
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    folder = args.protocol.parent
    if (folder / 'result.json').exists(): raise ValueError('preserve previous UI result')
    def fetch(route):
        with urllib.request.urlopen(f'http://127.0.0.1:{protocol["port"]}/api/{route}', timeout=10) as response:
            return json.load(response)
    started = time.monotonic(); root = None; previous = None
    with sqlite3.connect('file:' + str(ROOT / protocol['database']) + '?mode=ro', uri=True) as database:
        while True:
            if root is None:
                row = database.execute('SELECT id FROM turns WHERE parent IS NULL AND prompt=? AND epoch=? AND created>? ORDER BY created LIMIT 1',
                    (protocol['prompt'], protocol['observation_id'].split(':')[0], protocol['after_created'])).fetchone()
                if row: root = row[0]
            state = fetch('status'); public = fetch('agent')
            if root:
                turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
                plan = state['agent'].get('plan') or {}
                if plan and plan.get('turn_id') != root:
                    raise ValueError('another task owns the plan; do not merge evidence')
                key = (state['agent']['revision'], state['observation_id'])
                if key != previous:
                    dump(folder / 'progress.json', {'turn_id': root, 'terminal': state, 'turns': turns})
                    previous = key
                if (not state.get('job') and not state['agent']['busy'] and turns
                        and all(t['status'] not in ('running', 'waiting') for t in turns)):
                    break
            if time.monotonic() - started > 600:
                dump(folder / 'collector-timeout.json', {'root': root, 'task_was_not_stopped': True,
                    'latest_observation_id': state['observation_id']})
                raise TimeoutError('collector timed out; do not restart a live task')
            time.sleep(.5)
        receipts = []
        for execution in plan.get('execution_tools', []):
            row = database.execute('SELECT arguments,feedback,evidence FROM experiences WHERE id=?', (execution['id'],)).fetchone()
            if row is None: raise ValueError('missing bound native receipt')
            receipts.append({'execution': execution, **{k: json.loads(v) for k, v in zip(('arguments', 'feedback', 'evidence'), row)}})
    manifest = fetch('manifest')
    places = [r for r in receipts if r['execution']['skill'] == 'place']
    tools = [e for turn in turns for e in turn['trace'] if e['kind'] == 'tool']
    origin = protocol['origin_xy_m']; expected = [origin[0] + .08, origin[1]]
    goal = plan.get('placement_goal') or {}
    checks = {
        'actual_API': any(e['kind'] == 'api' for turn in turns for e in turn['trace']),
        'original_forward_goal': goal.get('object_id') == 'bottle' and goal.get('surface') == 'table'
            and goal.get('target_xy_m') == expected,
        'connected_place_executed': len(places) == 1 and places[0]['feedback'].get('connected_placement_proposal', {}).get('connected_upright_lowering') is True,
        'native_support_release_stability': len(places) == 1 and places[0]['feedback'].get('primitive_success') is True
            and places[0]['feedback'].get('verification_source') == 'native_support_release_and_stability'
            and places[0]['feedback'].get('placement', {}).get('confirmed') is True
            and places[0]['feedback'].get('position_error_m', 1) <= .01,
        'original_default_budget': plan.get('time_budget_s') == 120 and plan.get('timing_mode') == 'measured_delay'
            and state['observation']['sim_time'] <= 120,
        'same_physics': manifest['model_evidence'] == protocol['model_evidence'],
        'no_task_reset': not any(e['name'] in ('prepare_simulation', 'reset') for e in tools),
        'all_stages_done': bool(plan.get('steps')) and all(s['status'] == 'done' for s in plan['steps']),
        'formal_review': plan.get('verdict') in ('success', 'failure', 'uncertain'),
        'unselected_objects_undisturbed': state['diagnostics']['evaluation']['non_target_object_guard']['passed'],
        'source_current': manifest['source_status']['matches_disk']}
    validation = protocol.get('validation')
    if validation:
        from tools.check_bottle_api import gpu_evidence, route_evidence
        hashes=protocol['source_sha256']
        checks['same_declared_frozen_sources']=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in hashes.items())
        record={'receipts':receipts}
        checks['actual_curobo_native_transport']=gpu_evidence(record,hashes['tools/object_gpu.py'],validation['gpu_worker_port'])
        checks['native_requested_arm_and_route']=route_evidence(receipts,False,validation['requested_arm'])
        pick=next((r['feedback'] for r in receipts if r['execution']['skill']=='pick'),{})
        hold=pick.get('hold_audit',{})
        checks['continuous_requested_pick_hold']=(hold.get('requested_seconds')==validation['hold_seconds']
            and hold.get('actual_seconds',0)+.002>=validation['hold_seconds']
            and hold.get('continuous_full_windows_stable') is True and hold.get('audit_completed') is True)
        checks['deployed_task_placement_provider']=(len(places)==1
            and places[0]['feedback'].get('preview_servo_contract',{}).get('not_deployed') is False)
        checks['single_model_plan_no_replacement']=sum(e['name']=='set_plan' and e['status']=='completed' for e in tools)==1
        moves=[r['feedback'] for r in receipts if r['execution']['skill'] in ('move_to_pose','move_relative')
            and r['feedback'].get('primitive_success') is True]
        checks['exact_world_cache_provenance']=bool(moves) and all(
            f.get('backend_metadata',{}).get('scene_cache_worker_source_sha256')==hashes['tools/scene_cache.py']
            and f.get('backend_metadata',{}).get('exact_world_cache_source_sha256')==hashes['sim/scene_cache.py']
            and all(p.get('cache',{}).get('exact_world_update',{}).get('same_gpu_buffers') is True
                and p.get('cache',{}).get('exact_world_update',{}).get('same_as_fresh_official_scene_storage') is True
                for p in f.get('plans',[])) for f in moves)
    result = {'passed': all(checks.values()), 'checks': checks, 'turn_id': root, 'terminal': state,
        'turns': turns, 'receipts': receipts, 'manifest': manifest,
        'actual_api_calls': sum(e['kind'] == 'api' for turn in turns for e in turn['trace']),
        'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'collector_commands': 0, 'hardware_access': False, 'transfer_ready': False}
    dump(folder / 'result.json', result)
    print({'passed': result['passed'], 'checks': checks, 'calls': result['actual_api_calls'], 'root': root}, flush=True)


if __name__ == '__main__': main()
