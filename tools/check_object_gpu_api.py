"""Actual model, independently owned scene, exact generic GPU tool sequence."""
import hashlib
import json
import time

from sim.environment import ROOT
from sim.recording import dump

OUT = ROOT / 'runs/20261009-object-gpu'


def main():
    if (OUT / 'api.json').exists() or (OUT / 'api.protocol.json').exists(): raise ValueError('retain prior actual model run')
    from harness.composite import install_isolated as composite
    from sim.grasp import install_isolated as grasp
    from harness.candidate import install_isolated as candidate
    from harness.scene import install_isolated as scene
    from harness.object_gpu import ObjectGPUHarness, ObjectGPUWorkspace, install_isolated as objects, service_factory
    composite(); grasp(); candidate(); scene(); objects(); factory = service_factory()
    from harness.llm import ModelSettings
    from harness.vault import SecretStore
    from harness.provider import install_isolated as provider
    prompt = ('只用左手把长方体block从任务起始位置向机器人前方放8厘米，不操作瓶子或圆柱。'
        '先观察、读取场景目录，select_manipulation_object选择block，然后重新观察、inspect_manipulation。'
        '自己制定四个中文阶段：抓取、cuRobo携物运动、放置、复查；placement_goal锁定block/table/offset_xy_m=[0.08,0]，'
        '第一份set_plan声明120秒measured_delay，计入模型和规划计算等待。'
        '仅依次执行左手pick、左手move_relative(planner=curobo,frame=world,delta_m=[0.03,0,0])、左手place。'
        '每次读新观测与原生回执，再继续；放置必须使用原始锁定目标，不能按移动后的物体位置重新加8厘米。'
        '最后复查并finish。禁止重置、加时、换对象、交接、relocate或以TAMP替代GPU运动。'
        '这是已知仿真形状的开发验收，不能宣称视觉泛化或真机迁移。')
    hashes = {}
    for name in ['sim/object_curobo.py', 'tools/object_gpu.py', 'harness/object_gpu.py',
        'harness/objects.py', 'sim/objects.py', 'sim/object_planner.py', 'tools/check_object_gpu_api.py']:
        source = ROOT / name; target = OUT / 'source/api' / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes()); hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    dump(OUT / 'api.protocol.json', {'prompt': prompt, 'source_sha256': hashes, 'object_id': 'block', 'arm': 'left',
        'original_forward_m': .08, 'GPU_TCP_delta_m': [.03, 0., 0.], 'timing_mode': 'measured_delay',
        'time_budget_s': 120, 'expected_native_tools': ['pick', 'move_relative', 'place'],
        'production_memory_writes': False, 'hardware_access': False, 'transfer_ready': False})
    service = factory(); workspace = ObjectGPUWorkspace(); harness = None
    try:
        settings = ModelSettings(store=SecretStore())
        if not settings.public()['configured']: raise ValueError('saved model unavailable')
        harness = ObjectGPUHarness(service, workspace=workspace, settings=settings); provider(harness)
        initial = service.snapshot(); start = time.monotonic(); root = harness.chat(prompt)['turn_id']
        deadline = time.monotonic() + 600; history = []; key = None
        while True:
            state = service.snapshot(); public = workspace.public('execute', state['observation_id'].split(':')[0], state['memory_context'])
            summary = workspace.summary(state); plan = summary.get('plan') or {}
            turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
            current = (summary['revision'], (state.get('job') or {}).get('execution_id'))
            if current != key:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'steps': plan.get('steps', []), 'job': state.get('job')}); key = current
                dump(OUT / 'api.progress.json', {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if (not state.get('job') and not summary['busy'] and not harness.pending_review and not harness.pending_operations
                and turns and all(t['status'] not in ('running', 'waiting') for t in turns)): break
            if time.monotonic() > deadline: raise TimeoutError('retain partial run; inspect same owner rather than restarting')
            time.sleep(.5)
        receipts = [{'execution': execution, **workspace.execution_receipt('execute', execution['id'],
            context=state['memory_context'], include_oracle=True)} for execution in plan.get('execution_tools', [])]
        events = [e for turn in turns for e in turn['trace'] if e['kind'] == 'tool']
        actions = [e['name'] for e in events if e['name'] in {'pick', 'place', 'move_relative', 'move_to_pose', 'handover', 'relocate', 'set_gripper', 'run_policy'} and e['status'] == 'completed']
        selections = [e for e in events if e['name'] == 'select_manipulation_object' and e['status'] == 'completed']
        gpu = next((r['feedback'] for r in receipts if r['execution']['skill'] == 'move_relative'), {})
        plans = gpu.get('plans', []); goal = plan.get('placement_goal', {})
        checks = {'one_actual_block_selection': len(selections) == 1 and selections[0]['detail']['arguments']['object_id'] == 'block'
                and selections[0]['detail']['result'].get('physical_state_preserved') is True,
            'exact_three_native_skills': actions == ['pick', 'move_relative', 'place'] and len(receipts) == 3,
            'original_block_target_locked': goal.get('object_id') == 'block' and goal.get('surface') == 'table'
                and goal.get('offset_xy_m') == [.08, 0] and np_same(goal.get('target_xy_m', []), [.57, .075]),
            'actual_GPU_object_coverage': bool(plans) and all(p['success'] and p['attachment']['object_id'] == 'block'
                and p['metadata']['protocol'] == 'openarm-object-curobo/1' for p in plans),
            'correct_native_motion': gpu.get('verification_source') == 'robot_forward_kinematics'
                and gpu.get('position_error_m', 1) <= .012 and gpu.get('rotation_error_rad', 1) <= .08,
            'actual_selected_object_receipts': bool(receipts) and all(r['feedback'].get('object_id') == 'block'
                and r['feedback'].get('primitive_success') is True for r in receipts),
            'original_budget_and_waiting': plan.get('timing_mode') == 'measured_delay' and plan.get('time_budget_s') == 120,
            'all_stages_done': bool(plan.get('steps')) and all(s['status'] == 'done' for s in plan['steps']),
            'all_turns_completed': all(t['status'] == 'completed' for t in turns),
            'native_goal_confirmed': plan.get('goal_assessment', {}).get('confirmed') is True,
            'no_reset_or_backend_substitution': not any(e['name'] == 'prepare_simulation' and e['status'] == 'completed' for e in events)}
        result = {'passed': all(checks.values()), 'checks': checks, 'turn_id': root, 'turns': turns, 'receipts': receipts,
            'initial': initial, 'terminal': {**state, 'agent': summary}, 'stage_history': history,
            'actual_api_calls': sum(e['kind'] == 'api' for t in turns for e in t['trace']),
            'memory': public['memory'], 'wall_seconds': time.monotonic() - start,
            'production_memory_writes': False, 'hardware_access': False, 'transfer_ready': False}
        dump(OUT / 'api.json', result); print(json.dumps({k: result[k] for k in ['passed', 'checks', 'actual_api_calls']}, ensure_ascii=False), flush=True)
    finally:
        if harness: harness.close()
        service.close(); workspace.db.close()


def np_same(left, right): return len(left) == len(right) and all(abs(a - b) <= 1e-6 for a, b in zip(left, right))


if __name__ == '__main__': main()
