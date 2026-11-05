"""Verify natural-language demo goals through the saved API and native receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8767'


def http(route, body=None):
    request = urllib.request.Request(URL + route,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def trace(root):
    with sqlite3.connect('file:' + str(ROOT / 'harness/state/objects.sqlite3') + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        turns = [dict(row) for row in db.execute(
            'SELECT * FROM turns WHERE id=? OR parent=? ORDER BY created', (root, root))]
        for turn in turns:
            turn['trace'] = [{**dict(row), 'detail': json.loads(row['detail'])} for row in db.execute(
                'SELECT * FROM trace WHERE turn_id=? ORDER BY id', (turn['id'],))]
        return turns


def completed_skills(native):
    sources = {'pick': 'native_two_finger_contact_and_lift',
               'place': 'native_support_release_and_stability'}
    skills = {row['tool'] for row in native if row['passed']}
    for row in native:
        if row['tool'] != 'relocate' or not row['passed']:
            continue
        for child in row['summary']['metrics']['tamp'].get('substeps', []):
            feedback = child.get('feedback') or {}
            name = child.get('skill')
            if (child.get('status') == 'completed' and feedback.get('primitive_success')
                    and name in sources and feedback.get('verification_source') == sources[name]):
                skills.add(name)
    return skills


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=('arms', 'ui-arms', 'front', 'point', 'basket'))
    parser.add_argument('--name')
    args = parser.parse_args()
    out = ROOT / 'runs/20261014-demo' / (args.name or args.case)
    if out.exists() or not out.resolve().is_relative_to(ROOT / 'runs/20261014-demo'):
        raise ValueError('Use a new case name under runs/20261014-demo')
    status = http('/api/status')
    if status.get('job') or status['agent']['busy']:
        raise RuntimeError('A task is live; leave it running and wait for its receipt')
    if status.get('source_current') is not True or status['scene_library']['selected']:
        raise RuntimeError('Demo requires current sources and the clean bottle/basket scene')
    if not http('/api/model')['configured']:
        raise RuntimeError('Saved API configuration is required')
    http('/api/command', {'command': 'reset', 'arguments': {'episode': 0, 'seed': 0}})
    initial = http('/api/status')
    bottle = initial['diagnostics']['manipulation_scene']['position_m']
    targets = {}
    if args.case in ('arms', 'ui-arms'):
        for arm in ('right', 'left'):
            position = initial['observation']['tcp'][arm]['position_m']
            targets[arm] = [round(position[0] + .02, 6), round(position[1], 6), round(position[2] + .03, 6)]
        prompt = (f'把右手末端移动到世界坐标{targets["right"]}米，再把左手末端移动到世界坐标'
                  f'{targets["left"]}米。保持各自当前朝向，桌面物品不要移动。')
        if args.case == 'ui-arms':
            for arm, delta in (('right', [0, 0, .03]), ('left', [.04, 0, 0])):
                targets[arm] = [x + d for x, d in zip(initial['observation']['tcp'][arm]['position_m'], delta)]
            prompt = '让右手末端向上移动3厘米，再让左手末端向前移动4厘米，保持各自朝向。'
    elif args.case == 'front':
        targets['bottle_xy'] = [bottle[0] + .08, bottle[1]]
        prompt = '用右手把黑瓶子向机器人正前方移动8厘米，放回桌面，松手并撤离。'
    elif args.case == 'point':
        targets['bottle_xy'] = [round(bottle[0] + .03, 6), round(bottle[1] - .08, 6)]
        prompt = f'把黑瓶子抓起来，放到桌面的世界坐标XY={targets["bottle_xy"]}米处，松手并撤离。你选择合适的手臂。'
    else:
        prompt = '把桌上的黑瓶子放进篮子，放好后松手并撤离。你自己决定动作阶段和使用的工具。'
    out.mkdir()
    protocol = {'case': args.case, 'prompt': prompt, 'targets': targets,
        'initial_observation_id': initial['observation_id'],
        'source_revision': initial['source_revision_sha256'],
        'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reset': 'one reset before the goal; no resets or pose edits during execution',
        'provider': 'saved real API; no scripted motion substitution', 'hardware_access': False}
    dump(out / 'protocol.json', protocol)
    dump(out / 'initial.json', initial)
    started = time.monotonic()
    submitted = time.time()
    try:
        if args.case == 'ui-arms':
            print(json.dumps({'state': 'awaiting_browser_form', 'prompt': prompt}, ensure_ascii=False), flush=True)
            root = None
            while time.monotonic() - started < 180:
                with sqlite3.connect('file:' + str(ROOT / 'harness/state/objects.sqlite3') + '?mode=ro', uri=True) as db:
                    row = db.execute('SELECT id FROM turns WHERE prompt=? AND epoch=? AND created>=? ORDER BY created DESC LIMIT 1',
                                     (prompt, initial['observation_id'].split(':')[0], submitted)).fetchone()
                if row:
                    root = row[0]
                    break
                time.sleep(.5)
            if root is None:
                raise TimeoutError('Browser form was not submitted; collector sent no chat request')
            response = {'turn_id': root, 'transport': 'browser form; collector did not POST /api/chat'}
        else:
            response = http('/api/chat', {'message': prompt})
    except urllib.error.HTTPError as error:
        with sqlite3.connect('file:' + str(ROOT / 'harness/state/objects.sqlite3') + '?mode=ro', uri=True) as db:
            row = db.execute('SELECT id FROM turns WHERE prompt=? AND epoch=? AND created>=? ORDER BY created DESC LIMIT 1',
                             (prompt, initial['observation_id'].split(':')[0], submitted)).fetchone()
        response = {'request_error': error.code, 'turn_id': row[0] if row else None,
                    'message': 'HTTP failure retained; any live native execution was not cancelled'}
    root = response.get('turn_id')
    dump(out / 'response.json', response)
    print(json.dumps({'case': args.case, 'turn_id': root, 'initial_reply': response.get('message')}, ensure_ascii=False), flush=True)
    if not root:
        raise RuntimeError('API did not create a task')
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        status = http('/api/status')
        if not status.get('job') and not status['agent']['busy']:
            turns = trace(root)
            if turns and turns[-1]['status'] not in ('running', 'waiting'):
                break
        time.sleep(1)
    else:
        dump(out / 'timeout.json', {'turn_id': root, 'task_not_cancelled': True})
        raise TimeoutError('Collector timeout; task was not stopped or restarted')
    turns = trace(root)
    events = [event for turn in turns for event in turn['trace']]
    motions = [event for event in events if event['kind'] == 'tool'
               and event['detail'].get('result', {}).get('execution_id')]
    native = []
    with sqlite3.connect('file:' + str(ROOT / 'harness/state/objects.sqlite3') + '?mode=ro', uri=True) as db:
        for event in motions:
            execution = event['detail']['result']['execution_id']
            row = db.execute('SELECT run_id FROM episodes WHERE id=?', (execution,)).fetchone()
            if not row or not row[0]:
                continue
            folder = (ROOT / 'runs' / row[0]).resolve()
            if not folder.is_relative_to(ROOT / 'runs'):
                raise ValueError('Receipt outside runs')
            summary = json.loads((folder / 'summary.json').read_text())
            feedback = summary.get('metrics', {}).get('tamp', {})
            native.append({'tool': event['name'], 'arguments': event['detail'].get('arguments'),
                'execution_id': execution, 'run_id': row[0], 'summary': summary,
                'passed': bool(summary.get('transitions', 0) > 0 and feedback.get('primitive_success'))})
    plan = status['agent'].get('plan') or {}
    checks = {'http_request_completed': 'request_error' not in response,
        'real_api': any(event['kind'] == 'api' and event['status'] == 'completed' for event in events),
        'model_generated_plan': plan.get('turn_id') == root and bool(plan.get('steps')),
        'same_scene': status['observation_id'].split(':')[0] == initial['observation_id'].split(':')[0],
        'no_task_reset': not any(event['name'] in ('prepare_simulation', 'reset') for event in events if event['kind'] == 'tool'),
        'native_motion': any(row['passed'] for row in native),
        'no_invalid_physics': not status['diagnostics']['evaluation']['invalid'],
        'task_window_not_exhausted': status['observation'].get('task_window', {}).get('remaining_seconds', 1) > 0,
        'model_finished': plan.get('verdict') in ('success', 'uncertain'),
        'physical_stages_done': all(step['status'] == 'done' for step in plan.get('steps', []) if step['checkpoint'] != 'visual_review')}
    final_bottle = status['diagnostics']['manipulation_scene']['position_m']
    if args.case in ('arms', 'ui-arms'):
        checks['both_arms_native'] = {'left', 'right'} <= {row['arguments'].get('arm') for row in native if row['passed']}
        checks['target_positions'] = all(math.dist(status['observation']['tcp'][arm]['position_m'], target) <= .003 for arm, target in targets.items())
        checks['bottle_unchanged'] = math.dist(final_bottle[:2], bottle[:2]) <= .003
        angles = []
        for arm in targets:
            before = initial['observation']['tcp'][arm]['quaternion_xyzw']
            after = status['observation']['tcp'][arm]['quaternion_xyzw']
            norm = math.sqrt(sum(x*x for x in before) * sum(x*x for x in after))
            angles.append(2 * math.acos(min(1., abs(sum(a*b for a,b in zip(before, after))) / norm)))
        checks['orientations_kept'] = max(angles) <= .01
        checks['bottle_upright'] = status['diagnostics']['manipulation_scene']['upright']
    else:
        evaluation = status['diagnostics']['evaluation']
        checks['released'] = not any(evaluation['held'].values()) and not any(evaluation['touching'].values())
        checks['native_pick_and_place'] = {'pick', 'place'} <= completed_skills(native)
        if args.case == 'basket':
            checks['basket_support'] = bool(evaluation['inside_basket'] and evaluation['supported_by_basket'] and evaluation['placement']['confirmed'])
        else:
            checks['target_position'] = math.dist(final_bottle[:2], targets['bottle_xy']) <= .003
            checks['native_goal_confirmation'] = bool(plan.get('goal_assessment', {}).get('confirmed'))
    passed = all(checks.values())
    report = {**protocol, 'turn_id': root, 'status': 'passed' if passed else 'failed', 'checks': checks,
        'wall_seconds': time.monotonic() - started, 'actual_api_calls': sum(event['kind'] == 'api' for event in events),
        'plan': plan, 'turns': turns, 'native_receipts': native, 'terminal': status}
    dump(out / 'result.json', report)
    print(json.dumps({key: report[key] for key in ('case', 'status', 'checks', 'wall_seconds', 'actual_api_calls')}, ensure_ascii=False), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
