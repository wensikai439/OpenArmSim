"""Actual saved model chooses a directed registered-object relay."""
import hashlib
import time

from sim.environment import ROOT
from sim.recording import dump


def main():
    out = ROOT / 'runs/20261009-handover/api-relay'; out.mkdir(exist_ok=False)
    from harness.composite import install_isolated as composite
    from sim.grasp import install_isolated as grasp
    from harness.candidate import install_isolated as candidate
    from harness.scene import install_isolated as scene
    from harness.object_gpu import install_isolated as gpu
    from harness.object_handover import ObjectRelayHarness, install_isolated as relay, service_factory
    from harness.object_memory import ObjectMemoryWorkspace
    composite(); grasp(); candidate(); scene(); gpu(); relay(); factory = service_factory()
    from harness.llm import ModelSettings
    from harness.vault import SecretStore
    from harness.provider import install_isolated as provider
    prompt = ('请让左手抓起长方体block，交给右手后保持夹持，完成后停止。不要放下，不要移动瓶子或圆柱。'
        '自行制定简洁中文阶段。第一份set_plan声明120秒measured_delay，计入API与规划等待。'
        '这是显式仿真几何执行检查，不是视觉识别或真机迁移成绩。禁止任务中重置、换物体、换方向、加时或跳过原生验收。')
    hashes = {}
    for name in ['sim/object_handover.py', 'sim/objects.py', 'sim/object_planner.py', 'harness/object_handover.py',
        'harness/object_memory.py', 'harness/object_gpu.py', 'tools/check_object_relay_api.py']:
        source = ROOT / name; frozen = out / 'source' / name; frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(source.read_bytes()); hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    dump(out / 'protocol.json', {'prompt': prompt, 'source_sha256': hashes, 'object_id': 'block',
        'from_arm': 'left', 'to_arm': 'right', 'time_budget_s': 120, 'timing_mode': 'measured_delay',
        'expected_tools': ['pick', 'handover'], 'main_page_commands': 0, 'hardware_access': False, 'transfer_ready': False})
    service = factory(); workspace = ObjectMemoryWorkspace(); harness = None
    try:
        settings = ModelSettings(store=SecretStore())
        if not settings.public()['configured']: raise ValueError('saved model API is unavailable')
        harness = ObjectRelayHarness(service, workspace=workspace, settings=settings); provider(harness)
        initial = service.snapshot(); initial_manifest = service.call('manifest')
        started = time.monotonic(); root = harness.chat(prompt)['turn_id']
        deadline = time.monotonic() + 600; history = []; key = None
        while True:
            state = service.snapshot(); public = workspace.public('execute', state['observation_id'].split(':')[0], state['memory_context'])
            summary = workspace.summary(state); plan = summary.get('plan') or {}
            turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
            current = (summary['revision'], (state.get('job') or {}).get('execution_id'))
            if current != key:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'steps': plan.get('steps', []), 'job': state.get('job')}); key = current
                dump(out / 'progress.json', {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if (not state.get('job') and not summary['busy'] and not harness.pending_review and not harness.pending_operations
                and turns and all(t['status'] not in ('running', 'waiting') for t in turns)): break
            if time.monotonic() > deadline: raise TimeoutError('same owner exceeded600s; preserve the partial record')
            time.sleep(.5)
        receipts = [{'execution': execution, **workspace.execution_receipt('execute', execution['id'],
            context=state['memory_context'], include_oracle=True)} for execution in plan.get('execution_tools', [])]
        events = [event for turn in turns for event in turn['trace'] if event['kind'] == 'tool']
        actions = [event['name'] for event in events if event['name'] in
            {'pick', 'handover', 'place', 'relocate', 'move_relative', 'move_to_pose', 'run_policy', 'set_gripper'} and event['status'] == 'completed']
        feedback = next((r['feedback'] for r in receipts if r['execution']['skill'] == 'handover'), {})
        checks = {'exact_native_route': actions == ['pick', 'handover'] and len(receipts) == 2,
            'correct_object_and_direction': feedback.get('object_id') == 'block' and feedback.get('from_arm') == 'left'
                and feedback.get('to_arm') == feedback.get('held_by') == 'right' and feedback.get('execution_binding', {}).get('valid') is True,
            'native_relay_confirmed': feedback.get('primitive_success') is True
                and feedback.get('verification_source') == 'native_receiver_contact_then_donor_release',
            'native_receiver_still_holds': state['diagnostics']['evaluation']['held'] == {'left': False, 'right': True},
            'non_target_objects_undisturbed': state['diagnostics']['evaluation']['non_target_object_guard']['passed'],
            'all_stages_done': bool(plan.get('steps')) and all(s['status'] == 'done' for s in plan['steps']),
            'all_turns_completed': bool(turns) and all(t['status'] == 'completed' for t in turns),
            'original_timing_contract': plan.get('timing_mode') == 'measured_delay' and plan.get('time_budget_s') == 120,
            'no_reset_or_object_switch_after_motion': sum(e['name'] == 'select_manipulation_object' and e['status'] == 'completed' for e in events) == 1
                and not any(e['name'] == 'prepare_simulation' and e['status'] == 'completed' for e in events),
            'model_parameters_unchanged': initial_manifest['model_evidence'] == service.call('manifest')['model_evidence']}
        result = {'passed': all(checks.values()), 'checks': checks, 'turn_id': root, 'turns': turns, 'receipts': receipts,
            'initial': initial, 'terminal': {**state, 'agent': summary}, 'stage_history': history,
            'actual_api_calls': sum(e['kind'] == 'api' for turn in turns for e in turn['trace']),
            'wall_seconds': time.monotonic() - started, 'memory': public['memory'],
            'production_memory_writes': False, 'hardware_access': False, 'transfer_ready': False}
        dump(out / 'result.json', result)
        print({'passed': result['passed'], 'checks': checks, 'actual_api_calls': result['actual_api_calls']}, flush=True)
    finally:
        if harness: harness.close()
        service.close(); workspace.db.close()


if __name__ == '__main__': main()
