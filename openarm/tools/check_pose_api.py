"""One real API attempt per declared pose; original hold/place goals unchanged."""
import argparse
import copy
import hashlib
import json
import time

from sim.environment import ROOT
from sim.recording import dump
from tools.check_pose_grasp import CASES, initialize
from tools.check_task_queue import setup
from tools.check_task_api import collect
from tools.score_pose_api import gpu_evidence

OUT = ROOT / 'runs/20261009-poses'


def fixture_service():
    from harness.poses import service_factory
    base = service_factory()
    class FixtureCommands:
        def __init__(self, original, owner): self.original, self.owner = original, owner
        def put(self, item): return self.original.put(item)
        def get(self, *args, **kwargs):
            while True:
                command, values, future = self.original.get(*args, **kwargs)
                if command != '_initial_pose_fixture': return command, values, future
                if not future.set_running_or_notify_cancel(): continue
                owner = self.owner
                try:
                    if owner.job or owner.decision_wait or owner.env.tick != 0 or owner.env.data.time != 0.:
                        raise ValueError('fixture is initialisation only; never reset or edit an active task')
                    owner.env.require_current_sources()
                    case = next(row for row in CASES if row['name'] == values['name'])
                    initialize(owner.env, case); owner._publish()
                    future.set_result({'initialised': True, 'at_time_zero': True})
                except Exception as exc: future.set_exception(exc)
    class FixtureService(base):
        def _worker(self):
            self.commands = FixtureCommands(self.commands, self)
            return super()._worker()
    return FixtureService


def freeze(out):
    names = [str(p.relative_to(ROOT)) for folder in ('sim', 'harness') for p in sorted((ROOT/folder).glob('*.py'))]
    names += ['server.py', 'tools/check_pose_api.py', 'tools/check_pose_grasp.py', 'tools/object_gpu.py',
        'tools/object_workbench.py', 'tools/score_pose_api.py', 'ui/agent.js', 'ui/styles.css']
    hashes = {}
    for name in names:
        path = ROOT/name; target = out/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes()); hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def prompt(case):
    arm = '右手' if case['arm'] == 'right' else '左手'
    return (f'仅仿真验收：用{arm}抓起当前倒卧或倾斜的{case["object_id"]}，保持实际夹持10秒，'
        f'再经cuRobo运输，最终放在桌面world XY {case["target_xy_m"]}米并直立、松手、撤离、稳定至少1秒。'
        '其它物体不动。这是本次原始任务，不得重置、prepare、加时、换物体、换手或修改目标。'
        '自行制定中文阶段，第一份set_plan锁定单物体原目标、120秒measured_delay，计入API和规划等待。'
        'pick用hold_seconds=10请求原生持稳核验，不能把抬升回执当成10秒持稳。'
        '抓取完成后重新观察并检查几何，resolve_placement_transit(planner=curobo,clearance_m=0.12)'
        '读取本次抓取的原目标朝向预检查，再把返回的完整motion_request原样交给move_to_pose。'
        '确认实际GPU反馈后调用place，不能靠猜坐标或历史位姿搬运。'
        '任何候选或预测都不能当作完成；实际失败请保留原证据并正式finish failure/uncertain，不能另做更容易的任务。'
        '最后使用新相机复查，这是已注册仿真几何的开发测试，不是真机迁移或视觉泛化成绩。')


def score(record, case):
    state = record['terminal']; plan = state['agent'].get('plan') or {}; goal = plan.get('placement_goal') or {}
    receipts = record['receipts']; motions = [r['execution']['skill'] for r in receipts]
    pick = next((r['feedback'] for r in receipts if r['execution']['skill'] == 'pick'), {})
    transit = next((r['feedback'] for r in receipts if r['execution']['skill'] == 'move_to_pose'), {})
    placement = next((r['feedback'] for r in receipts if r['execution']['skill'] == 'place'), {})
    events = [e for turn in record['turns'] for e in turn['trace'] if e['kind'] == 'tool']
    target = goal.get('target_xy_m', [])
    same_target = len(target) == 2 and max(abs(x-y) for x,y in zip(target, case['target_xy_m'])) < 1e-8
    hold = pick.get('hold_audit', {})
    checks = {'actual_API': record['actual_api_calls'] > 0,
        'original_object_arm_and_three_native_skills': motions == ['pick', 'move_to_pose', 'place']
            and all(r['feedback'].get('object_id') == case['object_id'] and r['feedback'].get('primitive_success') for r in receipts)
            and pick.get('held_by') == case['arm'],
        'actual_requested10s_hold': hold.get('actual_seconds', 0.) >= 9.998
            and hold.get('requested_seconds') == 10 and hold.get('arms', {}).get(case['arm'], {}).get('status') == 'stable_within_audit_window',
        'fresh_pick_hint_and_real_GPU': gpu_evidence(record, record['expected_worker_sha256']),
        'original_target_release_support_stability': same_target and goal.get('object_id') == case['object_id']
            and placement.get('verification_source') == 'native_support_release_and_stability'
            and placement.get('placement', {}).get('confirmed') is True,
        'no_reset_prepare_or_plan_replacement': not any(e['name'] == 'prepare_simulation' for e in events)
            and sum(e['name'] == 'set_plan' and e['status'] == 'completed' for e in events) == 1,
        'original120s_clock_and_valid_scene': plan.get('time_budget_s') == 120 and plan.get('timing_mode') == 'measured_delay'
            and state['observation']['task_window']['start_sim_seconds'] == 0. and state['observation']['sim_time'] <= 120
            and not state['diagnostics']['evaluation']['invalid'],
        'all_requested_stages_confirmed': bool(plan.get('steps')) and all(s['status'] == 'done' for s in plan['steps']),
        'compiled_model_unchanged': record['initial_manifest']['model_evidence'] == record['terminal_manifest']['model_evidence'],
        'formal_model_review': plan.get('verdict') in ('success', 'failure', 'uncertain')}
    return {'passed': all(checks.values()), 'checks': checks}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='api-1'); parser.add_argument('--admission-only', action='store_true'); args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple immutable name required')
    out = OUT/args.name; out.mkdir(exist_ok=False); setup()
    from harness.poses import PoseHarness, PoseWorkspace, install_isolated
    from harness.llm import ModelSettings
    from harness.vault import SecretStore
    from harness.provider import install_isolated as provider
    install_isolated(); factory = fixture_service(); settings = ModelSettings(store=SecretStore())
    selected_cases = [CASES[1]] if args.admission_only else CASES
    spec = {'mode': 'actual_API', 'cases': selected_cases, 'source_sha256': freeze(out), 'hold_seconds': 10,
        'purpose': 'expected rejection of the unchanged invalid initial fixture; not a successful manipulation' if args.admission_only else 'full unchanged pose tasks',
        'time_budget_s': 120, 'timing_mode': 'measured_delay', 'table_transit_clearance_m': .12,
        'one_declared_attempt_per_case': True, 'no_reset_or_pose_edits_after_task_start': True,
        'hardware_access': False, 'transfer_ready': False}
    dump(out/'protocol.json', spec); rows = []
    if not settings.public()['configured']: raise ValueError('saved model API unavailable')
    for case in selected_cases:
        folder = out/case['name']; folder.mkdir(); service = factory(); workspace = PoseWorkspace(); harness = None
        root = None; error = None; started = time.monotonic(); manifest = None
        try:
            service.call('_initial_pose_fixture', name=case['name']); manifest = service.call('manifest')
            for camera in ('operator', 'head', 'left', 'right'): (folder/(camera+'-start.jpg')).write_bytes(service.frame(camera))
            harness = PoseHarness(service, workspace=workspace, settings=settings); provider(harness)
            requested = prompt(case); dump(folder/'request.json', {'prompt': requested, 'case':case})
            root = harness.chat(requested)['turn_id']; last = None
            while True:
                state = service.snapshot(); summary = workspace.summary(state)
                public = workspace.public('execute', state['observation_id'].split(':')[0], state['memory_context'])
                turns = [t for t in public['turns'] if t['id'] == root or t.get('parent') == root]
                key = (summary['revision'], state['observation_id'])
                if key != last:
                    dump(folder/'progress.json', {'turn_id':root, 'terminal':state, 'agent':summary, 'turns':turns}); last = key
                if not state.get('job') and not summary['busy'] and not harness.pending_review and not harness.pending_operations and turns and all(t['status'] not in ('running','waiting') for t in turns): break
                time.sleep(.5)
        except Exception as exc: error = {'code':type(exc).__name__, 'message':str(exc)}
        finally:
            if root:
                record = collect(service, workspace, root)
                record.update(mode='API', case=case, turn_id=root, initial_manifest=manifest, terminal_manifest=service.call('manifest'),
                    expected_worker_sha256=spec['source_sha256']['tools/object_gpu.py'],
                    error=error, wall_seconds=time.monotonic()-started, hardware_access=False, transfer_ready=False)
                result = {**record, **score(record, case)}
                if args.admission_only:
                    plan = record['terminal']['agent'].get('plan') or {}; admission = plan.get('start_grasp_admission', {})
                    result['admission_gate_verified'] = (admission.get('checked') is True and admission.get('valid') is False
                        and not record['receipts'] and plan.get('verdict') in ('failure', 'uncertain')
                        and result['checks']['no_reset_prepare_or_plan_replacement'])
            else: result = {'passed':False, 'case':case, 'error':error, 'root_created':False, 'terminal':service.snapshot()}
            for camera in ('operator', 'head', 'left', 'right'): (folder/(camera+'-final.jpg')).write_bytes(service.frame(camera))
            dump(folder/'result.json', result); rows.append(result); dump(out/'results.json', {'complete':len(rows)==len(selected_cases),'cases':rows})
            print(case['name'], {'passed':result['passed'], 'error':error, 'api_calls':result.get('actual_api_calls',0),
                'sim_time':result['terminal']['observation']['sim_time'], 'checks':result.get('checks')}, flush=True)
            if harness: harness.close()
            service.close(); workspace.db.close()


if __name__ == '__main__': main()
