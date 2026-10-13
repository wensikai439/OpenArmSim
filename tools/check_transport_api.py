"""Actual model composes object relay, fresh TCP resolution, GPU and placement."""
import argparse
import hashlib
import time

from sim.environment import ROOT
from sim.recording import dump
from harness.goals import same_xy

MOTIONS = {'pick', 'handover', 'place', 'relocate', 'move_relative', 'move_to_pose', 'run_policy', 'set_gripper'}
PROMPT = ('请左手抓起长方体block，交给右手，再用cuRobo把当前持物运输到桌面原目标上方，'
    '最终由右手放在桌面world XY [0.58,-0.05]米，松手并确认稳定后停止。不要移动瓶子或圆柱。'
    '自行制定简洁中文阶段，第一份set_plan锁定本次物体与原放置目标，120秒measured_delay计入API和规划等待。'
    '根据当前真实夹持关系计算末端目标，不能猜坐标或复用历史位姿。工具完成后重新观察。'
    '这是显式仿真几何检查，不是视觉检测或真机迁移成绩。禁止任务中重置、换物体、换方向、加时、改目标或跳过原生验收。')


def protocol():
    return {'prompt': PROMPT, 'object_id': 'block', 'from_arm': 'left', 'to_arm': 'right',
        'surface': 'table', 'target_xy_m': [.58, -.05], 'planner': 'curobo',
        'expected_tools': ['pick', 'handover', 'move_to_pose', 'place'],
        'time_budget_s': 120, 'timing_mode': 'measured_delay', 'hardware_access': False, 'transfer_ready': False}


def freeze(out):
    names = [str(p.relative_to(ROOT)) for folder in ['harness', 'sim'] for p in sorted((ROOT / folder).glob('*.py'))]
    names += ['server.py', 'sim/profile.json', 'tools/object_gpu.py', 'tools/object_workbench.py',
        'tools/check_transport_api.py', 'tools/check_object_transport.py', 'ui/agent.js']
    hashes = {}
    for name in names:
        source = ROOT / name
        if not source.is_file(): continue
        frozen = out / 'source' / name; frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(source.read_bytes()); hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def score(record, spec):
    state = record['terminal']; plan = state['agent'].get('plan') or {}; goal = plan.get('placement_goal') or {}
    events = [e for turn in record['turns'] for e in turn['trace'] if e['kind'] == 'tool']
    successful = [e for e in events if e['status'] == 'completed']; actions = [e for e in successful if e['name'] in MOTIONS]
    receipts = record['receipts']; relay = next((r['feedback'] for r in receipts if r['execution']['skill'] == 'handover'), {})
    move = next((r for r in receipts if r['execution']['skill'] == 'move_to_pose'), {})
    gpu = move.get('feedback', {}); proposals = gpu.get('plans', []); placement = receipts[-1]['feedback'] if receipts else {}
    resolutions = [e for e in successful if e['name'] == 'resolve_placement_transit']; request = move.get('arguments', {})
    checks = {'exact_native_route': [e['name'] for e in actions] == spec['expected_tools'] and len(receipts) == 4,
        'locked_original_object_goal': goal.get('object_id') == spec['object_id'] and goal.get('surface') == spec['surface']
            and same_xy(goal.get('target_xy_m', []), spec['target_xy_m']),
        'directed_native_relay': relay.get('object_id') == spec['object_id'] and relay.get('from_arm') == spec['from_arm']
            and relay.get('to_arm') == relay.get('held_by') == spec['to_arm'] and relay.get('primitive_success') is True
            and relay.get('execution_binding', {}).get('valid') is True
            and relay.get('verification_source') == 'native_receiver_contact_then_donor_release'
            and relay.get('independent_hold_seconds', 0) >= .999,
        'fresh_resolution_used_exactly': len(resolutions) == 1 and resolutions[0]['detail']['result'].get('motion_started') is False
            and request.get('scene_target_id') == resolutions[0]['detail']['result'].get('target_id')
            and all(request.get(k) == resolutions[0]['detail']['result']['motion_request'].get(k)
                for k in ('arm', 'position_m', 'quaternion_xyzw', 'frame', 'planner')),
        'actual_selected_object_GPU_carry': bool(proposals) and request.get('arm') == spec['to_arm'] and request.get('planner') == 'curobo'
            and gpu.get('payload_state') == 'single_hold' and gpu.get('primitive_success') is True
            and gpu.get('fallback') is False and gpu.get('native_path_revalidation') is True
            and all(p.get('success') and p['attachment']['object_id'] == spec['object_id']
                and p['metadata']['protocol'] == 'openarm-object-curobo/1'
                and p['metadata']['worker_source_sha256'] == spec['source_sha256']['tools/object_gpu.py'] for p in proposals),
        'native_original_placement': placement.get('object_id') == spec['object_id'] and placement.get('primitive_success') is True
            and placement.get('verification_source') == 'native_support_release_and_stability'
            and placement.get('position_error_m', 1) <= .01 and placement.get('placement', {}).get('confirmed') is True
            and same_xy(placement.get('placement', {}).get('xy_m', []), spec['target_xy_m'])
            and placement.get('placement', {}).get('stable_seconds', 0) >= 1.,
        'non_target_objects_undisturbed': state['diagnostics']['evaluation']['non_target_object_guard']['passed']
            and not state['diagnostics']['evaluation'].get('invalid'),
        'correct_execution_bindings': len(actions) == len(receipts) == 4 and all(
            e['detail']['result'].get('execution_id') == r['execution']['id'] for e, r in zip(actions, receipts)),
        'all_stages_done': bool(plan.get('steps')) and all(s['status'] == 'done' for s in plan['steps']),
        'formal_review_and_completed_turns': plan.get('verdict') in ('success', 'failure', 'uncertain')
            and bool(record['turns']) and all(t['status'] == 'completed' for t in record['turns']),
        'original_timing_contract': plan.get('timing_mode') == spec['timing_mode'] and plan.get('time_budget_s') == spec['time_budget_s']
            and state['observation']['sim_time'] <= spec['time_budget_s'],
        'one_selection_no_reset_or_prepare': sum(e['name'] == 'select_manipulation_object' for e in successful) == 1
            and not any(e['name'] == 'prepare_simulation' for e in events),
        'compiled_model_unchanged': record['initial_manifest']['model_evidence'] == record['terminal_manifest']['model_evidence']
            and all(m['model_evidence'] == record['initial_manifest']['model_evidence'] for m in record['manifests'])}
    return {'passed': all(checks.values()), 'checks': checks,
        'scope': 'one predeclared registered-object multi-turn API composition; not arbitrary-object or physical transfer evidence'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='api-full-1'); args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple immutable experiment name required')
    out = ROOT / 'runs/20261009-handover' / args.name; out.mkdir(exist_ok=False)
    from harness.composite import install_isolated as composite
    from sim.grasp import install_isolated as grasp
    from harness.candidate import install_isolated as candidate
    from harness.scene import install_isolated as scene
    from harness.object_gpu import install_isolated as gpu
    from harness.object_handover import install_isolated as relay, service_factory
    from harness.object_transport import ObjectTransportHarness, install_isolated as transport
    from harness.object_memory import ObjectMemoryWorkspace
    composite(); grasp(); candidate(); scene(); gpu(); relay(); transport(); factory = service_factory()
    from harness.llm import ModelSettings
    from harness.vault import SecretStore
    from harness.provider import install_isolated as provider
    spec = {**protocol(), 'source_sha256': freeze(out)}; dump(out / 'protocol.json', spec)
    service = factory(); workspace = ObjectMemoryWorkspace(); harness = None; root = None; failure = None
    try:
        settings = ModelSettings(store=SecretStore())
        if not settings.public()['configured']: raise ValueError('saved model API is unavailable')
        harness = ObjectTransportHarness(service, workspace=workspace, settings=settings); provider(harness)
        initial = service.snapshot(); manifest = service.call('manifest'); started = time.monotonic()
        root = harness.chat(spec['prompt'])['turn_id']; history = []; key = None; deadline = time.monotonic() + 600
        while True:
            state = service.snapshot(); summary = workspace.summary(state)
            public = workspace.public('execute', state['observation_id'].split(':')[0], state['memory_context'])
            plan = summary.get('plan') or {}; turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
            current = (summary['revision'], (state.get('job') or {}).get('execution_id'))
            if current != key:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'steps': plan.get('steps', []), 'job': state.get('job')}); key = current
                dump(out / 'progress.json', {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if (not state.get('job') and not summary['busy'] and not harness.pending_review and not harness.pending_operations
                and turns and all(t['status'] not in ('running', 'waiting') for t in turns)): break
            if time.monotonic() > deadline:
                failure = {'code': 'collector_timeout', 'message': 'same owner exceeded600s; partial record preserved'}; break
            time.sleep(.5)
        receipts = []; manifests = []
        for execution in plan.get('execution_tools', []):
            receipt = workspace.execution_receipt('execute', execution['id'], context=state['memory_context'], include_oracle=True)
            row = workspace.db.execute('SELECT arguments FROM experiences WHERE id=?', (execution['id'],)).fetchone()
            receipts.append({'execution': execution, 'arguments': __import__('json').loads(row[0]), **receipt})
            run_id = receipt['evidence'].get('run_id')
            if run_id: manifests.append(__import__('json').loads((ROOT / 'runs' / run_id / 'manifest.json').read_text()))
        record = {'turn_id': root, 'turns': turns, 'receipts': receipts, 'initial': initial,
            'initial_manifest': manifest, 'terminal_manifest': service.call('manifest'), 'manifests': manifests,
            'terminal': {**state, 'agent': summary}, 'stage_history': history,
            'actual_api_calls': sum(e['kind'] == 'api' for t in turns for e in t['trace']),
            'wall_seconds': time.monotonic() - started, 'memory': public['memory'], 'collector_error': failure,
            'production_memory_writes': False, 'hardware_access': False, 'transfer_ready': False}
        result = {**record, **score(record, spec)}; dump(out / 'result.json', result)
        print({'passed': result['passed'], 'checks': result['checks'], 'actual_api_calls': result['actual_api_calls'],
            'sim_time': state['observation']['sim_time']}, flush=True)
    finally:
        if harness: harness.close()
        service.close(); workspace.db.close()


if __name__ == '__main__': main()
