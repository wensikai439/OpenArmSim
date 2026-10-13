"""Declared native/API scene-reference pose tasks; no main-page commands."""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from sim.environment import ROOT
from sim.recording import dump
from sim.scene import SceneDirectory, CONTRACT

OUT = ROOT / 'runs/20261009-scene'
CASES = [
    {'name': 'bottle-right-tamp', 'arm': 'right', 'object_id': 'bottle', 'reference_point': 'origin',
        'offset_frame': 'object', 'offset_m': [-.13, -.06, .32], 'planner': 'tamp'},
    {'name': 'basket-left-curobo', 'arm': 'left', 'object_id': 'basket', 'reference_point': 'top_center',
        'offset_frame': 'world', 'offset_m': [-.12, 0., .20], 'planner': 'curobo'},
]


def install():
    from tools.contact_candidate import ContactCandidate
    from harness.composite import install_isolated as composite
    from sim.grasp import install_isolated as grasp
    from harness.candidate import install_isolated as candidate
    from harness.scene import install_isolated as scene
    composite(); grasp(); candidate(); scene()
    import sim.service as service
    service.OpenArmEnv = ContactCandidate
    class DirectoryService(service.SimulationService):
        def _publish(self):
            super()._publish()
            if not hasattr(self, 'directory'): self.directory = SceneDirectory(self.env)
            with self.lock:
                self.cache['diagnostics']['scene_directory'] = self.directory.snapshot(self.cache['observation_id'])
    return DirectoryService


def provenance():
    return {'contract': CONTRACT, 'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / 'sim/scene.py', ROOT / 'harness/scene.py', Path(__file__))},
        'scope': 'explicit simulator-geometry target resolution plus existing native pose tools; no new grasp or perception claim',
        'physics_parameters_changed': False, 'scoring_thresholds_changed': False,
        'main_page_commands': 0, 'hardware_access': False, 'transfer_ready': False}


def score_api(receipts, turns, plan, resolved, motions):
    """Score actual bound motions; extra unused read-only proposals are allowed."""
    import numpy as np
    targets = {row['target_id']: row for row in resolved}
    used = [event['detail']['arguments'].get('scene_target_id') for event in motions]
    checks = {'exactly_two_native_motions': len(receipts) == len(motions) == len(CASES),
        'two_distinct_used_targets': len(set(used)) == len(CASES),
        'used_targets_resolved': all(ident in targets for ident in used),
        'native_fk_receipts': all(receipt['feedback'].get('primitive_success')
            and receipt['feedback'].get('verification_source') == 'robot_forward_kinematics'
            and receipt['feedback'].get('position_error_m', 1) <= .012
            and receipt['feedback'].get('rotation_error_rad', 1) <= .08 for receipt in receipts),
        'all_declared_stages_done': bool(plan.get('steps')) and all(step['status'] == 'done' for step in plan.get('steps', [])),
        'timing_contract': plan.get('timing_mode') == 'measured_delay' and plan.get('time_budget_s') == 120,
        'formal_review': plan.get('verdict') in ('success', 'failure', 'uncertain'),
        'all_turns_completed': bool(turns) and all(turn['status'] == 'completed' for turn in turns),
        'no_extra_manipulation': not any(event['name'] in {'pick', 'handover', 'place', 'relocate', 'set_gripper', 'run_policy', 'run_tamp', 'run_pi05_vla'}
            and event['status'] == 'completed' for turn in turns for event in turn['trace'] if event['kind'] == 'tool')}
    matching = len(receipts) == len(motions) == len(CASES)
    for case, event, receipt in zip(CASES, motions, receipts):
        arguments = event['detail']['arguments']; result = event['detail']['result']
        row = targets.get(arguments.get('scene_target_id'), {})
        request = row.get('motion_request', {})
        matching &= bool(row.get('reference', {}).get('object_id') == case['object_id']
            and row.get('reference_point') == case['reference_point'] and row.get('offset_frame') == case['offset_frame']
            and row.get('offset_m') == case['offset_m'] and request.get('arm') == arguments.get('arm') == case['arm']
            and request.get('planner', 'tamp') == arguments.get('planner', 'tamp') == case['planner']
            and result.get('execution_id') == receipt['execution']['id'])
        first_position = np.asarray(arguments.get('position_m', [])); second_position = np.asarray(request.get('position_m', []))
        matching &= (first_position.shape == second_position.shape == (3,)
            and np.isfinite(first_position).all() and np.isfinite(second_position).all()
            and np.allclose(first_position, second_position, atol=1e-9, rtol=0))
        if arguments.get('quaternion_xyzw') is None or request.get('quaternion_xyzw') is None: matching = False
        else:
            first = np.asarray(arguments['quaternion_xyzw']); second = np.asarray(request['quaternion_xyzw'])
            matching &= min(np.linalg.norm(first - second), np.linalg.norm(first + second)) < 1e-6
        if case['planner'] == 'curobo':
            feedback = receipt['feedback']
            matching &= (feedback.get('fallback') is False and bool(feedback.get('native_path_revalidation'))
                and feedback.get('planner') == 'official cuRobo V2 GPU IK/trajectory optimization'
                and any(proposal.get('success') for proposal in feedback.get('plans', [])))
    checks['declared_targets_backends_and_execution_bindings_match'] = bool(matching)
    return {'passed': all(checks.values()), 'checks': checks, 'resolved_proposals': len(resolved),
        'used_target_ids': used, 'unused_proposals_are_not_actions': True,
        'scope': 'exact declared two-pose exercise; no generalization or grasp/placement claim'}


def rescore(output='api-verified.json'):
    source = OUT / 'api.json'; destination = OUT / output
    if destination.exists(): raise ValueError('preserve earlier scoring revision')
    result = json.loads(source.read_text())
    events = [event for turn in result['turns'] for event in turn['trace'] if event['kind'] == 'tool']
    motions = [event for event in events if event['name'] == 'move_to_pose' and event['status'] == 'completed']
    score = score_api(result['receipts'], result['turns'], result['terminal']['agent']['plan'], result['resolved_targets'], motions)
    dump(destination, {**score, 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'original_input': 'api.json', 'original_collector_verdict': result['passed'],
        'correction': 'original checker wrongly required exactly two read-only target resolutions; three proposals but only two used motions',
        'new_API_calls': 0, 'new_physics_actions': 0, 'hardware_access': False, 'transfer_ready': False,
        'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(score, ensure_ascii=False), flush=True)


def native(output='native-timed.json'):
    OUT.mkdir(exist_ok=True)
    target = OUT / output
    if target.exists(): raise ValueError('preserve previous declared task')
    dump(target.with_suffix('.protocol.json'), {**provenance(), 'cases': CASES, 'seed': 0, 'episode': 0,
        'timing_mode': 'measured_delay', 'budget_s': 120, 'initial_condition': 'unchanged cold contact candidate'})
    factory = install()
    from harness.scene import SceneHarness
    service = factory(); harness = None; rows = []
    try:
        harness = SceneHarness(service)
        harness.sessions['manual'] = {'timing_mode': 'measured_delay'}
        for case in CASES:
            service.call('reset', episode=0, seed=0); service.call('task_budget', seconds=120)
            initial = service.snapshot(); manifest = service.call('manifest')
            observation = harness.invoke('observe', {})
            directory = harness.invoke('inspect_scene', {'observation_id': observation['observation_id']})
            arguments = {k: v for k, v in case.items() if k != 'name'} | {'observation_id': observation['observation_id']}
            resolved = harness.invoke('resolve_scene_target', arguments)
            started = time.monotonic()
            accepted = harness.invoke('move_to_pose', resolved['motion_request'],
                context={'type': 'direct-tool', 'plan_turn_id': 'manual'})
            deadline = time.monotonic() + 300
            while service.snapshot().get('job'):
                if time.monotonic() > deadline: raise TimeoutError('native owner still active; no restart')
                time.sleep(.1)
            terminal = service.snapshot(); last = terminal['last_run']; feedback = last['metrics']['tamp']
            model_unchanged = manifest['model_evidence'] == service.call('manifest')['model_evidence']
            native_goal = bool(feedback.get('primitive_success') and feedback.get('verification_source') == 'robot_forward_kinematics'
                and feedback.get('position_error_m', 1) <= .012 and feedback.get('rotation_error_rad', 1) <= .08)
            correct_backend = (case['planner'] != 'curobo' or feedback.get('fallback') is False
                and feedback.get('native_path_revalidation') and feedback.get('planner') == 'official cuRobo V2 GPU IK/trajectory optimization')
            timing_matches = feedback.get('planning_timing', {}).get('mode') == 'measured_delay'
            row = {'case': case, 'passed': bool(native_goal and correct_backend and model_unchanged and timing_matches),
                'directory': directory, 'resolved': resolved, 'accepted': accepted, 'native_feedback': feedback,
                'run_id': last['run_id'], 'status': last['status'], 'error': last['error'],
                'wall_seconds': time.monotonic() - started, 'terminal': terminal, 'manifest': manifest,
                'model_evidence_unchanged': model_unchanged, 'timing_matches_protocol': timing_matches,
                'physics_commands_bypassing_harness': 0}
            rows.append(row); dump(target, {'cases': rows, 'complete': len(rows) == len(CASES), **provenance()})
            print(json.dumps({'case': case['name'], 'passed': row['passed'], 'error': row['error'],
                'position_error_m': feedback.get('position_error_m')}, ensure_ascii=False), flush=True)
    finally:
        if harness: harness.close(); harness.workspace.db.close()
        service.close()


def api():
    OUT.mkdir(exist_ok=True); target = OUT / 'api.json'
    if target.exists(): raise ValueError('preserve earlier API task')
    os.environ.setdefault('OPENARM_POLICY_CONFIG', str(ROOT / 'harness/policy.json'))
    factory = install()
    from harness.scene import SceneHarness
    from harness.candidate import CandidateWorkspace
    from harness.llm import ModelSettings
    from harness.vault import SecretStore
    from harness.provider import install_isolated as provider
    prompt = ('这是空手位姿练习，不抓取、不交接、不放置瓶子。先inspect_scene，使用resolve_scene_target生成以下目标，再把原样motion_request（含scene_target_id）传给move_to_pose。'
        '右手先用TAMP到瓶子body原点，偏移object坐标[-0.13,-0.06,0.32]米；之后左手用cuRobo到篮子top_center，偏移world坐标[-0.12,0,0.20]米。'
        '两次都保持各手当时的TCP朝向。不要自行抄坐标或直接move_relative；阶段由你制定，绑定move_to_pose的第1、2次调用，末端到位用motion_reached。'
        '第一份计划声明measured_delay及120秒预算；禁止任务中重置、加时、换后端或改变目标。两个原生位姿回执完成后用新观测复查，明确这不是抓取验收。')
    dump(OUT / 'api.protocol.json', {**provenance(), 'prompt': prompt, 'cases': CASES, 'timing_mode': 'measured_delay',
        'budget_s': 120, 'seed': 0, 'episode': 0, 'provider_mode': 'documented non-thinking; same output token cap',
        'independent_native_owner': True, 'production_memory_writes': False})
    service = factory(); workspace = CandidateWorkspace(); harness = None
    try:
        settings = ModelSettings(store=SecretStore())
        if not settings.public()['configured']: raise ValueError('saved API unavailable')
        harness = SceneHarness(service, workspace=workspace, settings=settings); provider(harness)
        initial = service.snapshot(); started = time.monotonic(); response = harness.chat(prompt); root = response['turn_id']
        deadline = time.monotonic() + 600; history = []; previous = None
        while True:
            state = service.snapshot(); public = workspace.public('execute', state['observation_id'].split(':')[0], state['memory_context'])
            summary = workspace.summary(state); plan = summary.get('plan') or {}
            turns = [turn for turn in public['turns'] if turn['id'] == root or turn.get('parent') == root]
            key = (summary['revision'], (state.get('job') or {}).get('execution_id'))
            if key != previous:
                history.append({'observation_id': state['observation_id'], 'sim_time': state['observation']['sim_time'],
                    'steps': plan.get('steps', []), 'job': state.get('job')}); previous = key
                dump(OUT / 'api.progress.json', {'turn_id': root, 'turns': turns, 'terminal': state, 'stage_history': history})
            if (not state.get('job') and not summary['busy'] and turns and not harness.pending_review
                    and all(turn['status'] not in ('running', 'waiting') for turn in turns)): break
            if time.monotonic() > deadline: raise TimeoutError('same API owner requires inspection; partial evidence retained')
            time.sleep(.5)
        receipts = []
        for execution in plan.get('execution_tools', []):
            receipt = workspace.execution_receipt('execute', execution['id'], context=state['memory_context'], include_oracle=True)
            receipts.append({'execution': execution, **receipt})
        tools = [event for turn in turns for event in turn['trace'] if event['kind'] == 'tool']
        resolved = [event['detail']['result'] for event in tools if event['name'] == 'resolve_scene_target' and event['status'] == 'completed']
        motions = [event for event in tools if event['name'] == 'move_to_pose' and event['status'] == 'completed']
        score = score_api(receipts, turns, plan, resolved, motions)
        passed = score['passed']
        result = {**provenance(), 'passed': passed, 'turn_id': root, 'turns': turns, 'receipts': receipts,
            'resolved_targets': resolved, 'initial': initial, 'terminal': {**state, 'agent': summary},
            'stage_history': history, 'wall_seconds': time.monotonic() - started,
            'actual_api_calls': sum(event['kind'] == 'api' for turn in turns for event in turn['trace']),
            'independent_native_owner': True, 'production_memory_writes': False, 'score': score}
        dump(target, result)
        print(json.dumps({'passed': passed, 'actual_api_calls': result['actual_api_calls'],
            'tools': [execution['skill'] for execution in plan.get('execution_tools', [])]}, ensure_ascii=False), flush=True)
    finally:
        if harness: harness.close()
        service.close(); workspace.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['native', 'api', 'rescore'], required=True)
    parser.add_argument('--output')
    arguments = parser.parse_args()
    if arguments.output and ('/' in arguments.output or not arguments.output.endswith('.json')): raise ValueError('simple JSON filename required')
    native(arguments.output or 'native-timed.json') if arguments.mode == 'native' else api() if arguments.mode == 'api' else rescore(arguments.output or 'api-verified.json')
