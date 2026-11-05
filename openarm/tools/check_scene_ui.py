"""Collect a predeclared browser pose exercise, without issuing any commands."""
import argparse
import hashlib
import json
import sqlite3
import time
import urllib.request

from .check_scene import ROOT, OUT, score_api


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main(name='workbench'):
    protocol_path = OUT / (name + '.protocol.json')
    protocol = json.loads(protocol_path.read_text())
    target = OUT / (name + '.json')
    if target.exists(): raise ValueError('preserve the previous browser exercise')
    write(OUT / (name + '.collector.json'), {
        'collector_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
        'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'commands': 0, 'scope': 'read-only collection after UI submission; no intermediate milestones reconstructed'})
    def get(path):
        with urllib.request.urlopen('http://127.0.0.1:8766' + path, timeout=10) as response:
            return json.load(response)
    history = []; key = None; root = None
    deadline = time.monotonic() + 300
    while True:
        state = get('/api/status'); public = get('/api/agent')
        if root is None:
            roots = [turn for turn in public['turns'] if turn.get('parent') is None
                and turn['created'] > protocol['after_created']]
            if roots: root = min(roots, key=lambda turn: turn['created'])['id']
        if root:
            turns = [turn for turn in public['turns'] if turn['id'] == root or turn.get('parent') == root]
            plan = state['agent'].get('plan') or {}
            if plan.get('turn_id') != root: raise ValueError('another task owns the current plan; do not merge evidence')
            current = (state['agent']['revision'], (state.get('job') or {}).get('execution_id'))
            if key != current:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'job': state.get('job'), 'steps': plan.get('steps', [])}); key = current
                write(OUT / (name + '.progress.json'), {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if (not state.get('job') and not state['agent']['busy'] and plan.get('verdict')
                    and all(turn['status'] not in ('running', 'waiting') for turn in turns)): break
        if time.monotonic() > deadline: raise TimeoutError('partial evidence retained; no reset or restart')
        time.sleep(.5)
    receipts = []
    with sqlite3.connect('file:' + str(ROOT / 'harness/state/contact.sqlite3') + '?mode=ro', uri=True) as database:
        for execution in plan.get('execution_tools', []):
            row = database.execute('SELECT arguments,feedback,evidence FROM experiences WHERE id=?', (execution['id'],)).fetchone()
            if row is None: raise ValueError('missing native execution receipt')
            receipts.append({'execution': execution, 'arguments': json.loads(row[0]),
                'feedback': json.loads(row[1]), 'evidence': json.loads(row[2])})
    tools = [event for turn in turns for event in turn['trace'] if event['kind'] == 'tool']
    resolved = [event['detail']['result'] for event in tools if event['name'] == 'resolve_scene_target' and event['status'] == 'completed']
    motions = [event for event in tools if event['name'] == 'move_to_pose' and event['status'] == 'completed']
    score = score_api(receipts, turns, plan, resolved, motions)
    result = {**score, 'turn_id': root, 'turns': turns, 'receipts': receipts, 'resolved_targets': resolved,
        'terminal': state, 'stage_history': history, 'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'actual_api_calls': sum(event['kind'] == 'api' for turn in turns for event in turn['trace']),
        'collector_commands': 0, 'hardware_access': False, 'transfer_ready': False}
    write(target, result)
    print(json.dumps({'passed': score['passed'], 'native_motions': len(receipts), 'api_calls': result['actual_api_calls'],
        'checks': score['checks']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='workbench')
    name = parser.parse_args().name
    if not name.replace('-', '').isalnum(): raise ValueError('simple experiment name required')
    main(name)
