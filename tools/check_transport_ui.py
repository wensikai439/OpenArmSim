"""Read-only collector for a predeclared browser-submitted transport task."""
import argparse
import json
import sqlite3
import time

from tools.check_object_gpu_ui import get
from tools.check_transport_api import score
from sim.environment import ROOT
from sim.recording import dump


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='workbench-full'); args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple experiment name required')
    out = ROOT / 'runs/20261009-handover' / args.name
    if (out / 'result.json').exists(): raise ValueError('retain previous browser task evidence')
    spec = json.loads((out / 'protocol.json').read_text()); root = None; history = []; key = None; deadline = time.monotonic() + 600
    while True:
        state = get('status'); public = get('agent')
        if root is None:
            candidates = [t for t in public['turns'] if t.get('parent') is None and t['created'] > spec['after_created']]
            if candidates: root = min(candidates, key=lambda t: t['created'])['id']
        if root:
            turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
            plan = state['agent'].get('plan') or {}
            if plan and plan['turn_id'] != root: raise ValueError('another task owns this native scene')
            current = (state['agent']['revision'], (state.get('job') or {}).get('execution_id'))
            if current != key:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'steps': plan.get('steps', []), 'job': state.get('job')}); key = current
                dump(out / 'progress.json', {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if (not state.get('job') and not state['agent']['busy'] and plan.get('verdict')
                and turns and all(t['status'] not in ('running', 'waiting') for t in turns)): break
        if time.monotonic() > deadline: raise TimeoutError('inspect same owner; preserve partial evidence without a restart')
        time.sleep(.5)
    receipts = []; manifests = []
    with sqlite3.connect('file:' + str(ROOT / 'harness/state/objects.sqlite3') + '?mode=ro', uri=True) as db:
        for execution in plan.get('execution_tools', []):
            row = db.execute('SELECT arguments,feedback,evidence FROM experiences WHERE id=?', (execution['id'],)).fetchone()
            if not row: raise ValueError('missing terminal native receipt')
            receipt = {'execution': execution, 'arguments': json.loads(row[0]), 'feedback': json.loads(row[1]), 'evidence': json.loads(row[2])}
            receipts.append(receipt); run_id = receipt['evidence'].get('run_id')
            if run_id: manifests.append(json.loads((ROOT / 'runs' / run_id / 'manifest.json').read_text()))
    record = {'turn_id': root, 'turns': turns, 'receipts': receipts, 'initial': spec['initial'],
        'initial_manifest': spec['initial_manifest'], 'terminal_manifest': get('manifest'), 'manifests': manifests,
        'terminal': state, 'stage_history': history, 'memory': public['memory'],
        'actual_api_calls': sum(e['kind'] == 'api' for t in turns for e in t['trace']),
        'collector_commands': 0, 'hardware_access': False, 'transfer_ready': False}
    result = {**record, **score(record, spec)}; dump(out / 'result.json', result)
    print({'passed': result['passed'], 'checks': result['checks'], 'api_calls': result['actual_api_calls'],
        'sim_time': state['observation']['sim_time']}, flush=True)


if __name__ == '__main__': main()
