"""Read-only observer for one predeclared browser-submitted multi-object task."""
import argparse
import json
import sqlite3
import time
import urllib.request

from sim.environment import ROOT
from sim.recording import dump
from tools.check_task_api import OUT, freeze, protocol, score


def get(endpoint):
    with urllib.request.urlopen('http://127.0.0.1:8767/api/' + endpoint, timeout=15) as response:
        return json.load(response)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    initial, public, manifest = get('status'), get('agent'), get('manifest')
    if initial.get('job') or public['busy'] or initial['observation']['sim_time'] != 0.:
        raise ValueError('predeclare this task on the fresh idle scene; do not reset a running task')
    if not manifest['source_status']['matches_disk']: raise ValueError('live source differs before submission')
    extra = ['ui/index.html', 'ui/styles.css', 'tools/check_task_ui.py']
    hashes = freeze(out)
    import hashlib
    for name in extra:
        source = ROOT / name
        if not source.is_file(): continue
        target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes()); hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = {**protocol(), 'source_sha256': hashes, 'initial': initial, 'initial_manifest': manifest,
        'after_created': max((turn['created'] for turn in public['turns']), default=0),
        'mode': 'API', 'submission': 'browser UI', 'collector_commands': 0}
    dump(out / 'protocol.json', spec)
    print('Browser task predeclared; submit the recorded prompt through the page.', flush=True)


def observe(out):
    if (out / 'result.json').exists(): raise ValueError('retain the prior immutable experiment')
    spec = json.loads((out / 'protocol.json').read_text()); root = None; history = []; key = None
    while True:
        state, public = get('status'), get('agent')
        if root is None:
            candidates = [turn for turn in public['turns'] if turn.get('parent') is None and turn['created'] > spec['after_created']]
            if candidates: root = min(candidates, key=lambda turn: turn['created'])['id']
        if root:
            turns = [turn for turn in public['turns'] if turn['id'] == root or turn.get('parent') == root]
            plan = state['agent'].get('plan') or {}
            if plan and plan['turn_id'] != root: raise ValueError('another task owns the native scene; no command sent')
            current = (state['agent']['revision'], (state.get('job') or {}).get('execution_id'))
            if key != current:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'steps': plan.get('steps', []), 'object_goals': plan.get('object_goals', []),
                    'goal_assessment': plan.get('goal_assessment'), 'job': state.get('job')})
                dump(out / 'progress.json', {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history}); key = current
            if (not state.get('job') and not public['busy'] and not state['agent']['busy']
                and turns and all(turn['status'] not in ('running', 'waiting') for turn in turns)):
                break
        # Observation has no fixed expiry that could restart or mask the task.
        time.sleep(.5)
    receipts = []; manifests = []
    with sqlite3.connect('file:' + str(ROOT / 'harness/state/objects.sqlite3') + '?mode=ro', uri=True) as db:
        for execution in plan.get('execution_tools', []):
            row = db.execute('SELECT arguments,feedback,evidence,epoch FROM experiences WHERE id=?', (execution['id'],)).fetchone()
            if not row: raise ValueError('terminal native receipt missing; do not infer success')
            receipt = {'execution': execution, 'arguments': json.loads(row[0]), 'feedback': json.loads(row[1]),
                'evidence': json.loads(row[2]), 'epoch': row[3]}
            receipts.append(receipt)
            run_id = receipt['evidence'].get('run_id')
            if run_id: manifests.append(json.loads((ROOT / 'runs' / run_id / 'manifest.json').read_text()))
    record = {'mode': 'API', 'submission': 'browser UI', 'turn_id': root, 'turns': turns, 'receipts': receipts,
        'initial': spec['initial'], 'initial_manifest': spec['initial_manifest'], 'terminal_manifest': get('manifest'),
        'manifests': manifests, 'terminal': state, 'stage_history': history, 'memory': public['memory'],
        'actual_api_calls': sum(event['kind'] == 'api' for turn in turns for event in turn['trace']),
        'collector_commands': 0, 'hardware_access': False, 'transfer_ready': False}
    result = {**record, **score(record, spec)}; dump(out / 'result.json', result)
    print({'passed': result['passed'], 'checks': result['checks'], 'api_calls': record['actual_api_calls'],
        'sim_time': state['observation']['sim_time']}, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--name', default='workbench'); args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple immutable name required')
    out = OUT / args.name
    if args.prepare: prepare(out)
    else: observe(out)


if __name__ == '__main__': main()
