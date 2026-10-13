"""Predeclared native grasp -> real GPU carry -> native place, with latency."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sim.environment import ROOT
from sim.objects import ObjectEnv
from sim.object_planner import ObjectPlanner
from sim.object_curobo import ObjectCuRoboPlanner, call, metadata, attachment, scene, joint_state
from sim.planning_timing import PlanningTiming
from sim.recording import Recorder, dump

OUT = ROOT / 'runs/20261009-object-gpu'
CASES = [{'name': 'block-left', 'object_id': 'block', 'arm': 'left', 'forward_m': .08,
    'gpu_delta_m': [.03, 0., 0.]}, {'name': 'can-right', 'object_id': 'can', 'arm': 'right',
    'forward_m': .06, 'gpu_delta_m': [.03, 0., 0.]}]


def main():
    if (OUT / 'native.protocol.json').exists(): raise ValueError('retain prior trials, including failures')
    hashes = {}
    for name in ['sim/object_curobo.py', 'tools/object_gpu.py', 'tools/check_object_gpu.py',
        'sim/objects.py', 'sim/object_planner.py', 'sim/curobo.py', 'sim/curobo_server.py', 'sim/curobo_cache.py']:
        path = ROOT / name; frozen = OUT / 'source/native' / name
        frozen.parent.mkdir(parents=True, exist_ok=True); frozen.write_bytes(path.read_bytes())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata()['ready']
    dump(OUT / 'native.protocol.json', {'cases': CASES, 'source_sha256': hashes, 'timing_mode': 'measured_delay',
        'time_budget_s': 120, 'robot_and_object_parameters_changed': False, 'placement_tolerance_m': .01,
        'tcp_gate': {'position_m': .012, 'rotation_rad': .06}, 'no_reset_or_pose_edits_after_task_start': True,
        'scope': 'declared primitive development cases; not statistical generalization or physical calibration',
        'hardware_access': False, 'transfer_ready': False})
    rows = []
    for case in CASES:
        folder = OUT / case['name']; folder.mkdir(); env = ObjectEnv(images=False); rec = None
        selected = case['object_id']; steps = []; error = None; planner = None; phase = 'initial'
        started = time.monotonic()
        try:
            env.reset(seed=0); env.select_object(selected); env.set_task_budget(120)
            initial = env.data.xpos[env.bottle_id].copy(); target = initial[:2] + [case['forward_m'], 0.]
            manifest = env.manifest()
            dump(folder / 'initial.json', {'case': case, 'manifest': manifest, 'qpos': env.data.qpos,
                'qvel': env.data.qvel, 'target_xy_m': target, 'selection': env.selection()})
            Image.fromarray(env.render('operator')).save(folder / 'start.jpg', quality=90)
            rec = Recorder(env, 'tamp', root=folder / 'episode', record_images=False, seed=0,
                details={'case': case, 'selection': env.selection(), 'timing_mode': 'measured_delay',
                    'goal_target_xy_m': target, 'source_sha256': hashes})
            rec.manifest['task'] = 'Registered-object real GPU transport: ' + selected
            dump(rec.path / 'manifest.json', rec.manifest)
            before = rec.observation(env)
            def execute(planner_type, task, params):
                nonlocal planner, before
                timing = PlanningTiming('measured_delay'); planner = None
                def native():
                    nonlocal planner
                    planner = planner_type(env, task, parameters=params)
                    yield from planner.actions()
                for action, diagnostic, is_hold in timing.actions(env, native(), rec.event):
                    _, _, term, trunc, info = (env.hold_tamp(action, diagnostic['planning_timing']['duration_s'])
                        if is_hold else env.step_tamp(action))
                    after = rec.observation(env)
                    diagnostic['registered_objects'] = {ident: {'position_m': env.data.xpos[row['body']].copy(),
                        'free_velocity': env.data.qvel[row['dofadr']:row['dofadr'] + 6].copy()}
                        for ident, row in env.registry.items()}
                    rec.append(before, action, after, info, terminated=term, truncated=trunc, diagnostics=diagnostic)
                    timing.committed(is_hold, before['sim_time'], after['sim_time']); before = after
                    if term or trunc: raise RuntimeError('native physics/declared budget stopped task')
                result = {**planner.result, 'planning_timing': timing.summary()}
                if not result.get('primitive_success'): raise RuntimeError('missing actual native skill receipt')
                steps.append({'task': task, 'parameters': params, 'feedback': result})
                dump(folder / 'progress.json', {'steps': steps, 'phase': task, 'sim_time': float(env.data.time)})
            phase = 'pick'
            execute(ObjectPlanner, 'pick_object', {'object_id': selected, 'arm': case['arm']})
            Image.fromarray(env.render('operator')).save(folder / 'picked.jpg', quality=90)
            phase = 'gpu_move'
            execute(ObjectCuRoboPlanner, 'move_relative', {'object_id': selected, 'arm': case['arm'],
                'frame': 'world', 'delta_m': case['gpu_delta_m'], 'planner': 'curobo'})
            Image.fromarray(env.render('operator')).save(folder / 'carried.jpg', quality=90)
            phase = 'place'
            execute(ObjectPlanner, 'place_object', {'object_id': selected, 'arm': case['arm'],
                'receptacle_id': 'table', 'target_xy_m': target.tolist(), 'position_tolerance_m': .01})
            status = 'completed'
        except Exception as exc:
            status = 'error'; error = {'code': getattr(exc, 'code', 'execution_error'), 'message': str(exc)}
        finally:
            gpu = next((r['feedback'] for r in steps if r['task'] == 'move_relative'), {})
            plans = gpu.get('plans', []); placement = steps[-1]['feedback'] if steps else {}
            checks = {'all_three_original_skills': len(steps) == 3 and status == 'completed',
                'official_GPU_success_with_actual_object': bool(plans) and all(p['success']
                    and p['attachment']['object_id'] == selected and p['metadata']['protocol'] == 'openarm-object-curobo/1'
                    for p in plans), 'native_GPU_motion_source': gpu.get('verification_source') == 'robot_forward_kinematics',
                'actual_object_placement': placement.get('object_id') == selected and placement.get('primitive_success') is True
                    and placement.get('verification_source') == 'native_support_release_and_stability'
                    and placement.get('position_error_m', 1) <= .01 and env.evaluation['placement'].get('confirmed') is True,
                'model_evidence_unchanged': env.manifest()['model_evidence'] == manifest['model_evidence'],
                'all_compute_wait_integrated': all(r['feedback']['planning_timing']['mode'] == 'measured_delay'
                    and r['feedback']['planning_timing']['unintegrated_seconds'] <= 1e-8 for r in steps)}
            result = {'case': case, 'status': status, 'passed': all(checks.values()), 'checks': checks, 'error': error,
                'phase': phase, 'steps': steps, 'failed_feedback': planner.result if error and planner else None,
                'target_xy_m': target, 'actual_object_position_m': env.data.xpos[env.bottle_id],
                'final_evaluation': env.evaluation, 'sim_time': float(env.data.time),
                'wall_seconds': time.monotonic() - started, 'hardware_access': False, 'transfer_ready': False}
            dump(folder / 'result.json', result)
            if rec and not rec._closed: rec.finish(status, env.info(), error=error, metrics={'object_GPU': result})
            Image.fromarray(env.render('operator')).save(folder / 'final.jpg', quality=90)
            env.close(); rows.append(result); dump(OUT / 'native.json', {'complete': len(rows) == len(CASES), 'cases': rows})
            print(json.dumps({'case': case['name'], 'passed': result['passed'], 'checks': checks, 'error': error,
                'sim_time': result['sim_time']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
