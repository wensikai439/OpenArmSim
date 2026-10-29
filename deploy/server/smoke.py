"""One synthetic ACT inference and one native simulated GPU-planned move; no LLM/API use."""
import json
from pathlib import Path
import sys
import time
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.policy_wire import IMAGE_SHAPES, TASK, encode

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(port, path, body=None, content_type='application/json'):
    if body is not None and not isinstance(body, bytes): body = json.dumps(body).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}' + path, body,
        {'Content-Type': content_type} if body is not None else {})
    with OPENER.open(request, timeout=180) as response: return json.load(response)


def main():
    health = call(18767, '/health')
    if not all(health.get(key) for key in ('policy_ready', 'curobo_ready', 'source_current')):
        raise RuntimeError('Workers are not ready')
    if health.get('task_active') or health.get('agent_busy'): raise RuntimeError('Existing task left untouched')
    session = call(8003, '/session', {'seed': 20261029})['session']
    try:
        observation = {'state': np.zeros(16, dtype=np.float32), 'task': np.array(TASK),
            'session': np.array(session), 'sequence': np.array(0, dtype=np.int64),
            **{name: np.zeros(shape, dtype=np.uint8) for name, shape in IMAGE_SHAPES.items()}}
        result = call(8003, '/infer', encode(observation), 'application/x-npz')
        actions = np.asarray(result['actions'])
        act = {'shape': list(actions.shape), 'finite': bool(np.isfinite(actions).all()),
               'inference_seconds': result['inference_seconds'], 'input': 'synthetic zeros; no policy actions executed'}
        if actions.shape != (32, 16) or not act['finite']: raise RuntimeError('Unexpected ACT output')
    finally:
        call(8003, '/close', {'session': session})
    call(18767, '/api/command', {'command': 'reset', 'arguments': {}})
    observed = call(18767, '/api/tools', {'name': 'observe', 'arguments': {}})
    started = time.monotonic()
    movement = call(18767, '/api/tools', {'name': 'move_relative', 'arguments': {
        'observation_id': observed['observation_id'], 'arm': 'right', 'delta_m': [0, 0, .03],
        'frame': 'world', 'planner': 'curobo', 'speed_scale': .3}})
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = call(18767, '/api/status')
        if not status.get('job'): break
        time.sleep(.5)
    else: raise RuntimeError('Simulation did not complete within test budget')
    run = status.get('last_run') or {}
    report = {'act': act, 'movement_start': movement, 'movement_seconds': time.monotonic() - started,
              'run_id': run.get('run_id'), 'status': run.get('status'),
              'metrics': run.get('metrics'), 'hardware_access': False, 'llm_requests': 0}
    (ROOT / '.runtime/smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('metrics', 'movement_start')}, ensure_ascii=False))
    call(18767, '/api/command', {'command': 'reset', 'arguments': {}})
    if report['status'] != 'primitive_completed': raise RuntimeError('Native movement did not pass')


if __name__ == '__main__': main()
