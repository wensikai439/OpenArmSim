"""Actual saved-model registered-object tool call; independent native owner."""
import hashlib
import json
import time
from pathlib import Path

from sim.environment import ROOT
from sim.recording import dump

OUT = ROOT / 'runs/20261009-objects'


def main():
    target = OUT / 'api.json'
    if target.exists(): raise ValueError('preserve previous API trial')
    from harness.composite import install_isolated as composite
    from sim.grasp import install_isolated as grasp
    from harness.candidate import install_isolated as candidate
    from harness.scene import install_isolated as scene
    from harness.objects import ObjectHarness, ObjectWorkspace, install_isolated as objects, service_factory
    composite(); grasp(); candidate(); scene(); objects(); factory = service_factory()
    from harness.llm import ModelSettings
    from harness.vault import SecretStore
    from harness.provider import install_isolated as provider
    prompt = ('将场景中的长方体 block 从它的任务起始位置向机器人前方放8厘米。不要抓取或移动瓶子或圆柱。'
        '先inspect_scene确认ID，再select_manipulation_object选择block；选择不做物理移动，但观测身份会更新，必须重新observe与inspect_manipulation。'
        '自己规划简洁中文阶段，placement_goal锁定block/table/offset_xy_m=[0.08,0]；第一份set_plan声明measured_delay与120秒。'
        '仅用左手pick和place这两个原生动作；新的形状尚不支持relocate、handover或cuRobo携物。'
        '根据原生回执继续执行并在放置后复查，不能用末端移动冒充物体放置。禁止任务中重置、加时、换对象或修改目标。'
        '这是显式仿真几何开发检查，不是真机或纯视觉抓取成绩。')
    files = ['sim/objects.py', 'sim/object_planner.py', 'harness/objects.py', 'tools/check_objects_api.py']
    hashes = {}
    for name in files:
        path = ROOT / name; frozen = OUT / 'source' / 'api' / name
        frozen.parent.mkdir(parents=True, exist_ok=True); frozen.write_bytes(path.read_bytes())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(OUT / 'api.protocol.json', {'prompt': prompt, 'source_sha256': hashes,
        'timing_mode': 'measured_delay', 'time_budget_s': 120, 'object_id': 'block', 'forward_m': .08,
        'native_actions': ['pick', 'place'], 'default_geometry_and_primitive_thresholds_unchanged': True,
        'main_page_commands': 0, 'hardware_access': False, 'transfer_ready': False})
    service = factory(); workspace = ObjectWorkspace(); harness = None
    try:
        settings = ModelSettings(store=SecretStore())
        if not settings.public()['configured']: raise ValueError('saved model API unavailable')
        harness = ObjectHarness(service, workspace=workspace, settings=settings); provider(harness)
        initial = service.snapshot(); started = time.monotonic(); root = harness.chat(prompt)['turn_id']
        deadline = time.monotonic() + 600; history = []; key = None
        while True:
            state = service.snapshot(); public = workspace.public('execute', state['observation_id'].split(':')[0], state['memory_context'])
            summary = workspace.summary(state); plan = summary.get('plan') or {}
            turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
            current = (summary['revision'], (state.get('job') or {}).get('execution_id'))
            if current != key:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'selected_object': state['diagnostics']['manipulation_scene']['object_id'],
                    'steps': plan.get('steps', []), 'job': state.get('job')}); key = current
                dump(OUT / 'api.progress.json', {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if not state.get('job') and not summary['busy'] and not harness.pending_review and not harness.pending_operations and turns and all(t['status'] not in ('running','waiting') for t in turns): break
            if time.monotonic() > deadline: raise TimeoutError('same native owner must be inspected; partial record preserved')
            time.sleep(.5)
        receipts = [{'execution': execution, **workspace.execution_receipt('execute', execution['id'],
            context=state['memory_context'], include_oracle=True)} for execution in plan.get('execution_tools', [])]
        events = [event for turn in turns for event in turn['trace'] if event['kind'] == 'tool']
        selections = [e for e in events if e['name'] == 'select_manipulation_object' and e['status'] == 'completed']
        native_motions = [e['name'] for e in events if e['name'] in {'pick','place','handover','relocate','move_relative','move_to_pose','run_policy','set_gripper'} and e['status'] == 'completed']
        checks = {'one_block_selection': len(selections) == 1 and selections[0]['detail']['arguments']['object_id'] == 'block'
            and selections[0]['detail']['result']['physical_state_preserved'],
            'exact_requested_native_tools': native_motions == ['pick','place'],
            'correct_object_native_receipts': len(receipts) == 2 and all(r['feedback'].get('object_id') == 'block'
                and r['feedback'].get('primitive_success') for r in receipts),
            'locked_block_forward_goal': plan.get('placement_goal', {}).get('object_id') == 'block'
                and plan.get('placement_goal', {}).get('offset_xy_m') == [.08,0],
            'physical_goal_confirmed': plan.get('goal_assessment', {}).get('confirmed') is True,
            'declared_waiting_and_budget': plan.get('timing_mode') == 'measured_delay' and plan.get('time_budget_s') == 120,
            'all_stages_done': bool(plan.get('steps')) and all(s['status'] == 'done' for s in plan['steps']),
            'all_turns_completed': all(t['status'] == 'completed' for t in turns),
            'formal_review': plan.get('verdict') in ('success','failure','uncertain'),
            'no_reset_or_object_switch': not any(e['name'] == 'prepare_simulation' and e['status'] == 'completed' for e in events)}
        result = {'passed': bool(all(checks.values())), 'checks': checks, 'turn_id': root, 'turns': turns,
            'receipts': receipts, 'initial': initial, 'terminal': {**state, 'agent': summary}, 'stage_history': history,
            'wall_seconds': time.monotonic() - started,
            'actual_api_calls': sum(e['kind'] == 'api' for t in turns for e in t['trace']),
            'memory': public['memory'], 'production_memory_writes': False, 'hardware_access': False, 'transfer_ready': False}
        dump(target, result); print(json.dumps({'passed': result['passed'], 'checks': checks,
            'api_calls': result['actual_api_calls'], 'sim_time': state['observation']['sim_time']}, ensure_ascii=False), flush=True)
    finally:
        if harness: harness.close()
        service.close(); workspace.db.close()


if __name__ == '__main__': main()
