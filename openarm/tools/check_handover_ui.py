"""Read-only collection of a predeclared browser handover acceptance task."""
import hashlib
import json
import sqlite3
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261008-ik'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    protocol = json.loads((OUT / 'workbench-final.protocol.json').read_text())
    target = OUT / 'workbench-final.json'
    if target.exists(): raise ValueError('preserve previous browser acceptance')
    write(OUT / 'workbench-final.collector.json', {
        'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'browser_protocol_sha256': hashlib.sha256((OUT / 'workbench-final.protocol.json').read_bytes()).hexdigest(),
        'collector_commands': 0, 'scope': 'collection begins after UI submission; no unseen intermediate milestones reconstructed'})
    history = []; key = None; root = None
    def get(path):
        with urllib.request.urlopen('http://127.0.0.1:8766' + path, timeout=10) as response:
            return json.load(response)
    for _ in range(600):
        state = get('/api/status'); public = get('/api/agent')
        if root is None:
            roots = [turn for turn in public['turns'] if turn.get('parent') is None
                and turn['created'] > protocol['after_created']]
            if roots: root = max(roots, key=lambda turn: turn['created'])['id']
        if root:
            turns = [turn for turn in public['turns'] if turn['id'] == root or turn.get('parent') == root]
            plan = state['agent'].get('plan') or {}
            current = (state['agent']['revision'], (state.get('job') or {}).get('execution_id'),
                tuple((step['id'], step['status']) for step in plan.get('steps', [])))
            if current != key:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'running': bool(state['job']), 'steps': plan.get('steps', []), 'bindings': plan.get('stage_bindings', {})})
                key = current
                write(target.with_suffix('.progress.json'), {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if (not state['job'] and not state['agent']['busy'] and plan.get('verdict')
                    and all(turn['status'] not in ('running', 'waiting') for turn in turns)): break
        time.sleep(.5)
    else: raise TimeoutError('partial task retained; no reset or restart')
    receipts = []
    with sqlite3.connect('file:' + str(ROOT / 'harness/state/contact.sqlite3') + '?mode=ro', uri=True) as database:
        for execution in plan.get('execution_tools', []):
            row = database.execute('SELECT arguments,feedback,evidence FROM experiences WHERE id=?', (execution['id'],)).fetchone()
            if row: receipts.append({'execution': execution, 'arguments': json.loads(row[0]),
                'feedback': json.loads(row[1]), 'evidence': json.loads(row[2])})
    held = state['diagnostics']['manipulation_scene']['held_by']
    tools = [receipt['execution']['skill'] for receipt in receipts]
    passed = bool(held == ['right'] and tools == ['pick', 'handover']
        and all(receipt['feedback'].get('primitive_success') for receipt in receipts)
        and all(step['status'] == 'done' for step in plan['steps'])
        and plan.get('timing_mode') == 'measured_delay' and plan.get('time_budget_s') == 120
        and all(turn['status'] == 'completed' for turn in turns))
    result = {'passed': passed, 'turn_id': root, 'turns': turns, 'receipts': receipts, 'terminal': state,
        'stage_history': history, 'actual_api_calls': sum(event['kind'] == 'api' for turn in turns for event in turn['trace']),
        'default_memory': {'methods': len(public['memory']['knowledge']['lessons']), 'preferences': len(public['memory']['notes']),
            'raw_experiences': len(public['memory']['experiences']), 'raw_episodes': len(public['memory']['episodes'])},
        'collector_commands': 0, 'hardware_access': False, 'transfer_ready': False}
    write(target, result)
    print(json.dumps({'passed': passed, 'held_by': held, 'tools': tools, 'api_calls': result['actual_api_calls']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
