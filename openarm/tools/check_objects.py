"""Declared native multi-object tasks; every trial/failure is retained."""
import hashlib
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sim.environment import ROOT
from sim.objects import ObjectEnv, ASSETS, CONTRACT
from sim.object_planner import ObjectPlanner, CONTRACT as GRASP
from sim.decision_timing import GraspStability
from sim.recording import Recorder, dump

OUT = ROOT / 'runs/20261009-objects'
CASES = [
    {'name': 'block-left', 'object_id': 'block', 'arm': 'left', 'delta_m': [0., 0., 0.], 'yaw_rad': 0., 'forward_m': .08},
    {'name': 'can-right', 'object_id': 'can', 'arm': 'right', 'delta_m': [0., 0., 0.], 'yaw_rad': 0., 'forward_m': .06},
    {'name': 'block-left-shifted', 'object_id': 'block', 'arm': 'left', 'delta_m': [.015, -.020, 0.], 'yaw_rad': .25, 'forward_m': .08},
    {'name': 'can-right-shifted', 'object_id': 'can', 'arm': 'right', 'delta_m': [-.015, .020, 0.], 'yaw_rad': -.25, 'forward_m': .06},
]


def main():
    OUT.mkdir(exist_ok=False)
    files = ['sim/objects.py', 'sim/object_planner.py', 'tools/check_objects.py']
    sources = {}
    for name in files:
        path = ROOT / name; target = OUT / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes()); sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = {'schema': 'native-object-tasks/1', 'cases': CASES, 'declared_assets': ASSETS,
        'object_contract': CONTRACT, 'grasp_contract': GRASP, 'source_sha256': sources,
        'seed': 0, 'episode': 0, 'time_budget_s': 120, 'timing_mode': 'paused', 'hold_seconds': 10,
        'hold_gate': {'window_seconds': 2., 'position_m': .001, 'rotation_rad': .02},
        'placement_tolerance_m': .010, 'placement_stability_s': 1.,
        'initial_condition': 'three real native free bodies; only predeclared selected-object pose edits before recording',
        'no_reset_or_pose_edits_after_task_start': True, 'robot_and_bottle_parameters_changed': False,
        'scope': 'predefined development cases, no statistical generalization/visual grounding/physical calibration claim',
        'main_page_commands': 0, 'hardware_access': False, 'transfer_ready': False}
    dump(OUT / 'protocol.json', protocol); rows = []
    for case in CASES:
        folder = OUT / case['name']; folder.mkdir(); env = ObjectEnv(images=False)
        rec = None; planner = None; quality = None; steps = []; error = None; phase = 'initial'; held_s = 0.
        started = time.monotonic()
        try:
            env.reset(seed=0); row = env.registry[case['object_id']]; address = row['qposadr']
            env.data.qpos[address:address + 3] += case['delta_m']
            angle = case['yaw_rad']; env.data.qpos[address + 3:address + 7] = [np.cos(angle / 2), 0., 0., np.sin(angle / 2)]
            mujoco.mj_forward(env.model, env.data)
            env.select_object(case['object_id']); env.set_task_budget(120)
            initial = {ident: env.data.xpos[value['body']].copy() for ident, value in env.registry.items()}
            target_xy = initial[case['object_id']][:2] + [case['forward_m'], 0.]
            manifest = env.manifest()
            dump(folder / 'initial.json', {'case': case, 'selection': env.selection(), 'manifest': manifest,
                'qpos': env.data.qpos, 'qvel': env.data.qvel, 'initial_objects': initial, 'target_xy_m': target_xy})
            Image.fromarray(env.render('operator')).save(folder / 'start.jpg', quality=90)
            rec = Recorder(env, 'tamp', root=folder / 'episode', record_images=False, seed=0,
                details={**protocol, 'case': case, 'selection': env.selection(), 'locked_target_xy_m': target_xy.tolist()})
            rec.manifest['task'] = 'Pick, hold10s, then place registered object ' + case['object_id'] + ' at the declared forward target.'
            dump(rec.path / 'manifest.json', rec.manifest)
            def apply(action, diagnostic):
                before = rec.observation(env); _, _, terminal, truncated, info = env.step_tamp(action)
                object_state = {ident: {'position_m': env.data.xpos[value['body']].copy(),
                    'free_velocity': env.data.qvel[value['dofadr']:value['dofadr'] + 6].copy()}
                    for ident, value in env.registry.items()}
                rec.append(before, action, rec.observation(env), info, terminated=terminal, truncated=truncated,
                    diagnostics={**diagnostic, 'objects': object_state})
                if terminal or truncated: raise RuntimeError('native terminal: ' + str(env.evaluation['failure_reasons']))
            for task in ('pick_object', 'hold', 'place_object'):
                phase = task
                if task == 'hold':
                    quality = GraspStability(env); hold_start = float(env.data.time); command = env.last_action.copy()
                    for _ in range(300):
                        apply(command, {'phase': '10s-hold'}); quality.read()
                    held_s = float(env.data.time) - hold_start
                    steps.append({'task': task, 'actual_seconds': held_s, 'feedback': quality.summary()})
                    Image.fromarray(env.render('operator')).save(folder / 'held.jpg', quality=90)
                    if quality.summary()['arms'][case['arm']]['status'] != 'stable_within_audit_window':
                        raise RuntimeError('original full-hold audit failed; placement not attempted')
                    continue
                params = {'object_id': case['object_id'], 'arm': case['arm']}
                if task == 'place_object': params.update(receptacle_id='table', target_xy_m=target_xy.tolist(), position_tolerance_m=.010)
                planner = ObjectPlanner(env, task, parameters=params)
                generator = planner.actions(); count = 0
                while True:
                    q = env.data.qpos.copy(); v = env.data.qvel.copy()
                    try: action, diagnostic = next(generator)
                    except StopIteration: break
                    np.testing.assert_array_equal(q, env.data.qpos); np.testing.assert_array_equal(v, env.data.qvel)
                    apply(action, diagnostic); count += 1
                if not planner.result.get('primitive_success'): raise RuntimeError('missing native primitive receipt')
                steps.append({'task': task, 'commands': count, 'feedback': planner.result.copy()})
            status = 'completed'
        except Exception as exc:
            status = 'error'; error = {'code': getattr(exc, 'code', 'native_task_error'), 'message': str(exc)}
        finally:
            final = {ident: env.data.xpos[value['body']].copy() for ident, value in env.registry.items()}
            feedback = steps[-1]['feedback'] if steps else {}
            passed = bool(status == 'completed' and feedback.get('primitive_success')
                and feedback.get('verification_source') == 'native_support_release_and_stability'
                and feedback.get('object_id') == case['object_id']
                and np.linalg.norm(final[case['object_id']][:2] - target_xy) <= .010
                and env.evaluation['placement'].get('confirmed'))
            unchanged = env.manifest()['model_evidence'] == manifest['model_evidence']
            result = {'case': case, 'status': status, 'passed': bool(passed and unchanged), 'error': error,
                'phase': phase, 'steps': steps, 'failed_feedback': planner.result if status == 'error' and planner else None,
                'locked_target_xy_m': target_xy, 'initial_objects': initial, 'final_objects': final,
                'final_evaluation': env.evaluation, 'hold': quality.summary() if quality else None,
                'actual_hold_seconds': held_s, 'wall_seconds': time.monotonic() - started,
                'model_evidence_unchanged': unchanged, 'sim_time': float(env.data.time),
                'native_episode': str(rec.path.relative_to(ROOT)) if rec else None,
                'hardware_access': False, 'transfer_ready': False}
            if rec: rec.finish(status, env.info(), error=str(error) if error else None, metrics={'object_task': result})
            Image.fromarray(env.render('operator')).save(folder / 'final.jpg', quality=90)
            dump(folder / 'result.json', result); rows.append(result)
            dump(OUT / 'results.json', {'complete': len(rows) == len(CASES), 'cases': rows})
            print(case['name'], status, error, 'hold', held_s, flush=True)
            env.close()


if __name__ == '__main__': main()
