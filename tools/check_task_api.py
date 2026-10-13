"""Predeclared occupied-space task: one scene, two objects, one clock."""
import argparse
import copy
import hashlib
import json
import time

from sim.environment import ROOT
from sim.recording import dump
from tools.check_task_queue import setup

OUT = ROOT / 'runs/20261009-tasks'
PROMPT = ('先用右手把圆柱can从本任务初始位置向机器人前方放8厘米，腾出它原来的桌面位置；'
    '然后用左手夹起长方体block，交给右手，由右手放在圆柱最初的位置。瓶子不动。'
    '这是同一项仿真任务，自行制定中文阶段和有序物体目标，一开始锁定所有原目标，'
    '总预算120秒measured_delay，计入模型与规划等待。物体松手稳定后再选下一对象。'
    '不要重置、prepare、加时、替换任务目标或用当前圆柱位置代替初始位置。'
    '是显式仿真几何验收，不是视觉泛化或实物迁移。最后复查两个物体的结果。')
STEPS = [{'id': 'can_pick', 'title': '右手抓起圆柱', 'object_id': 'can', 'checkpoint': 'grasp_confirmed', 'skill': 'pick'},
    {'id': 'can_place', 'title': '圆柱向前放8厘米', 'object_id': 'can', 'checkpoint': 'placement_confirmed', 'skill': 'place'},
    {'id': 'block_pick', 'title': '左手抓起长方体', 'object_id': 'block', 'checkpoint': 'grasp_confirmed', 'skill': 'pick'},
    {'id': 'block_handover', 'title': '长方体交给右手', 'object_id': 'block', 'checkpoint': 'handover_confirmed', 'skill': 'handover'},
    {'id': 'block_place', 'title': '放在圆柱原位', 'object_id': 'block', 'checkpoint': 'placement_confirmed', 'skill': 'place'},
    {'id': 'review', 'title': '复查两个目标', 'checkpoint': 'visual_review'}]
GOALS = [{'object_id': 'can', 'surface': 'table', 'offset_xy_m': [.08, 0.]},
    {'object_id': 'block', 'surface': 'table', 'target_xy_m': [.503, -.345]}]


def freeze(out):
    names = [str(p.relative_to(ROOT)) for folder in ('harness', 'sim') for p in sorted((ROOT / folder).glob('*.py'))]
    names += ['server.py', 'tools/check_task_api.py', 'tools/check_task_queue.py', 'tools/object_gpu.py', 'tools/scene_gpu.py', 'tools/object_workbench.py', 'ui/app.js', 'ui/agent.js']
    hashes = {}
    for name in names:
        source = ROOT / name; target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes()); hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def protocol():
    return {'prompt': PROMPT, 'object_goals': GOALS, 'original_origins_xy_m': {'can': [.503, -.345], 'block': [.49, .075]},
        'original_targets_xy_m': {'can': [.583, -.345], 'block': [.503, -.345]},
        'expected_tools': ['pick', 'place', 'pick', 'handover', 'place'],
        'expected_objects': ['can', 'can', 'block', 'block', 'block'],
        'time_budget_s': 120, 'timing_mode': 'measured_delay', 'hardware_access': False, 'transfer_ready': False}


def collect(service, workspace, root):
    state = service.snapshot(); summary = workspace.summary(state)
    public = workspace.public('execute', state['observation_id'].split(':')[0], state['memory_context'])
    plan = summary.get('plan') or {}; turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
    receipts = []; manifests = []
    for execution in plan.get('execution_tools', []):
        receipt = workspace.execution_receipt('execute', execution['id'], context=state['memory_context'], include_oracle=True)
        row = workspace.db.execute('SELECT arguments,epoch FROM experiences WHERE id=?', (execution['id'],)).fetchone()
        receipts.append({'execution': execution, 'arguments': json.loads(row[0]), 'epoch': row[1], **receipt})
        run_id = receipt['evidence'].get('run_id')
        if run_id: manifests.append(json.loads((ROOT / 'runs' / run_id / 'manifest.json').read_text()))
    return {'terminal': {**state, 'agent': summary}, 'turns': turns, 'receipts': receipts, 'manifests': manifests,
        'memory': public['memory'], 'actual_api_calls': sum(e['kind'] == 'api' for t in turns for e in t['trace'])}


def score(record, spec):
    state = record['terminal']; plan = state['agent'].get('plan') or {}; goals = plan.get('object_goals', [])
    receipts = record['receipts']; events = [e for t in record['turns'] for e in t['trace'] if e['kind'] == 'tool']
    selections = []
    for event in events:
        if event['status'] != 'completed': continue
        if event['name'] == 'select_manipulation_object': selections.append(event)
        if event['name'] == 'set_plan' and event['detail']['result'].get('first_object_selection'):
            result = event['detail']['result']['first_object_selection']
            selections.append({'detail': {'arguments': {'object_id': result['object_id']}, 'result': result},
                'source': 'first-object bookkeeping requested by typed set_plan.object_goals; no motor command'})
    def close(a, b): return len(a) == len(b) == 2 and all(abs(x-y) <= 1e-6 for x,y in zip(a,b))
    assessment = plan.get('goal_assessment') or {}
    checks = {'exact_native_tools_objects': [r['execution']['skill'] for r in receipts] == spec['expected_tools']
            and [r['feedback'].get('object_id') for r in receipts] == spec['expected_objects']
            and all(r['feedback'].get('primitive_success') is True for r in receipts),
        'two_original_targets_not_current_can_pose': len(goals) == 2 and [g['object_id'] for g in goals] == ['can', 'block']
            and all(close(g['target_xy_m'], spec['original_targets_xy_m'][g['object_id']])
                and close(g['anchor']['object_xy_m'], spec['original_origins_xy_m'][g['object_id']]) for g in goals),
        'current_all_object_support_release_stability': assessment.get('confirmed') is True
            and assessment.get('native_original_ledger_matches') is True and len(assessment.get('objects', [])) == 2
            and all(g['confirmed'] and g['stable_seconds'] >= 1. for g in assessment['objects']),
        'bottle_undisturbed_and_scene_valid': assessment.get('unlisted_objects_undisturbed') is True
            and not state['diagnostics']['evaluation']['invalid'],
        'no_reset_prepare_or_extra_plan': not any(e['name'] == 'prepare_simulation' for e in events)
            and sum(e['name'] == 'set_plan' and e['status'] == 'completed' for e in events) == 1,
        'ordered_switch_after_placement_keeps_task_window': len(selections) == 2
            and [e['detail']['arguments']['object_id'] for e in selections] == ['can', 'block']
            and selections[-1]['detail']['result'].get('shared_task_window_preserved') is True
            and selections[-1]['detail']['result'].get('completed_placement_execution_id') == receipts[1]['execution']['id'] if len(receipts) > 1 else False,
        'original_shared_budget_and_timing': plan.get('time_budget_s') == 120 and plan.get('timing_mode') == 'measured_delay'
            and state['observation']['task_window']['start_sim_seconds'] == 0. and state['observation']['sim_time'] <= 120,
        'all_stages_confirmed': bool(plan.get('steps')) and all(step['status'] == 'done' for step in plan['steps']),
        'same_compiled_physical_model': record['initial_manifest']['model_evidence'] == record['terminal_manifest']['model_evidence']
            and all(m['model_evidence'] == record['initial_manifest']['model_evidence'] for m in record['manifests'])}
    if record.get('mode') == 'API':
        checks['actual_API_and_formal_review'] = record['actual_api_calls'] > 0 and plan.get('verdict') in ('success', 'failure', 'uncertain')
    return {'passed': all(checks.values()), 'checks': checks}


def native_motion(harness, name, arguments, context, out):
    service = harness.service; workspace = harness.workspace; root = context['turn_id']
    harness.invoke('observe', {}, context=context); before = service.snapshot()
    result = native_tool(harness, name, {**arguments, 'observation_id': before['observation_id']}, context)
    execution = result['execution_id']; workspace.bind(root, execution, 'execute', PROMPT, name, name, 'offline-native', before['memory_context'])
    harness.sessions[root]['actions'] = harness.sessions[root].get('actions', 0) + 1
    while service.snapshot().get('job'):
        dump(out / 'progress.json', {'mode': 'native', 'job': service.snapshot().get('job'), 'terminal': service.snapshot()})
        time.sleep(.2)
    state = service.snapshot(); last = state['last_run']; feedback = last.get('metrics', {}).get('tamp', {})
    workspace.record_feedback(execution, 'execute', state['observation_id'].split(':')[0], PROMPT, name,
        arguments, feedback, state['memory_context'], 'simulator_oracle',
        {'before': {k:before['observation'][k] for k in ('sim_time', 'state', 'tcp')},
            'after': {'observation_id': state['observation_id'], **{k:state['observation'][k] for k in ('sim_time', 'state', 'tcp')}},
            'run_id': last['run_id']})
    workspace.complete(execution, last['run_id'])
    if not feedback.get('primitive_success'): raise RuntimeError('native skill failed: ' + str(feedback.get('message', last.get('error'))))


def native_tool(harness, name, args, context):
    event = harness.workspace.event(context['turn_id'], 'tool', name, {'arguments': args})
    try:
        result = harness.invoke(name, args, context=context)
        harness.workspace.end_event(event, {'arguments': args, 'result': result}, 'completed')
        return result
    except Exception as exc:
        harness.workspace.end_event(event, {'arguments': args, 'result': {'error': str(exc)}}, 'error')
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native', action='store_true'); parser.add_argument('--name', default='api-1')
    parser.add_argument('--objects', action='store_true', help='use the actual isolated objects workbench and user-role contract')
    parser.add_argument('--output', type=lambda x:(ROOT/x).resolve(), help='new immutable directory inside runs')
    args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple immutable experiment name required')
    out = args.output or OUT / args.name
    if not out.is_relative_to((ROOT/'runs').resolve()) or out == (ROOT/'runs').resolve():raise ValueError('new output inside runs required')
    out.mkdir(parents=True, exist_ok=False)
    if not args.objects:setup()
    from harness.task_queue import TaskHarness, TaskWorkspace, service_factory
    from harness.llm import ModelSettings
    from harness.vault import SecretStore
    from harness.provider import install_isolated as provider
    if args.objects:
        from tools.object_workbench import build_backend
        from harness.poses import PoseWorkspace
        factory,Harness=build_backend();Workspace=PoseWorkspace
    else:
        factory=service_factory();Harness=TaskHarness;Workspace=TaskWorkspace
    spec = {**protocol(), 'source_sha256': freeze(out), 'mode': 'native' if args.native else 'API',
        'backend':'objects-workbench' if args.objects else 'task-queue'}; dump(out / 'protocol.json', spec)
    service = factory(); workspace = Workspace(); harness = None; root = None; error = None; started = time.monotonic()
    try:
        settings = ModelSettings(store=SecretStore()); harness = Harness(service, workspace=workspace, settings=settings)
        initial = service.snapshot(); manifest = service.call('manifest')
        if args.native:
            root = workspace.start('execute', initial['observation_id'].split(':')[0], PROMPT, {'model': 'offline-native', 'vision': False})
            context = {'type': 'llm', 'turn_id': root, 'plan_turn_id': root, 'model': 'offline-native'}
            harness.sessions[root] = {'goal':PROMPT, 'actions': 0, 'initial_snapshot': initial}
            observed = harness.invoke('observe', {}, context=context)
            native_tool(harness, 'select_manipulation_object', {'observation_id': observed['observation_id'], 'object_id': 'can'}, context)
            observed = harness.invoke('observe', {}, context=context)
            harness.invoke('inspect_manipulation', {'observation_id': observed['observation_id']}, context=context)
            native_tool(harness, 'set_plan', {'observation_id': observed['observation_id'], 'object_goals': GOALS,
                'steps': copy.deepcopy(STEPS[:-1]), 'time_budget_s': 120, 'timing_mode': 'measured_delay'}, context)
            for name, parameters in [('pick', {'object_id': 'can', 'arm': 'right'}),
                ('place', {'object_id': 'can', 'arm': 'right', 'receptacle_id': 'table'})]: native_motion(harness, name, parameters, context, out)
            observed = harness.invoke('observe', {}, context=context)
            result = native_tool(harness, 'select_manipulation_object', {'observation_id': observed['observation_id'], 'object_id': 'block'}, context)
            dump(out / 'switch.json', result)
            for name, parameters in [('pick', {'object_id': 'block', 'arm': 'left'}),
                ('handover', {'object_id': 'block', 'from_arm': 'left', 'to_arm': 'right'}),
                ('place', {'object_id': 'block', 'arm': 'right', 'receptacle_id': 'table'})]: native_motion(harness, name, parameters, context, out)
            workspace.finish(root, 'Native two-object route completed; no API used.')
            workspace.busy = False
        else:
            if not settings.public()['configured']: raise ValueError('saved model API unavailable')
            provider(harness); root = harness.chat(PROMPT)['turn_id']; key = None
            while True:
                state = service.snapshot(); summary = workspace.summary(state); public = workspace.public('execute', state['observation_id'].split(':')[0], state['memory_context'])
                turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
                current = (summary['revision'], (state.get('job') or {}).get('execution_id'))
                if current != key:
                    dump(out / 'progress.json', {'turn_id': root, 'terminal': state, 'turns': turns, 'agent': summary}); key = current
                if not state.get('job') and not summary['busy'] and not harness.pending_review and not harness.pending_operations and turns and all(t['status'] not in ('running', 'waiting') for t in turns): break
                time.sleep(.5)
    except Exception as exc:
        error = {'code': type(exc).__name__, 'message': str(exc)}
    finally:
        if root:
            record = collect(service, workspace, root)
            record.update(mode=spec['mode'], turn_id=root, initial_manifest=manifest, terminal_manifest=service.call('manifest'),
                error=error, wall_seconds=time.monotonic()-started, hardware_access=False, transfer_ready=False)
            result = {**record, **score(record, spec)}; dump(out / 'result.json', result)
            print({'passed': result['passed'], 'checks': result['checks'], 'error': error,
                'sim_time': record['terminal']['observation']['sim_time'], 'api_calls': record['actual_api_calls']}, flush=True)
        else:
            dump(out / 'collector-error.json', {'passed': False, 'error': error, 'root_created': False,
                'terminal': service.snapshot(), 'source': 'caught startup failure before the task root; no robot tool dispatched'})
            print({'passed': False, 'startup_error': error}, flush=True)
        if harness: harness.close()
        service.close(); workspace.db.close()


if __name__ == '__main__': main()
