"""Declared bidirectional primitive relay, hold audit and downstream placement."""
import argparse
import hashlib
import json
import time

import mujoco
import numpy as np
from PIL import Image

from sim.environment import ROOT
from sim.objects import ObjectEnv
from sim.object_planner import ObjectPlanner
from sim.object_handover import ObjectHandoverPlanner, ObjectRelayPickPlanner
from sim.object_curobo import ObjectCuRoboPlanner
from sim.planning_timing import PlanningTiming
from sim.decision_timing import GraspStability
from sim.recording import Recorder, dump

CASES = [{'name': 'block-left-right', 'object_id': 'block', 'donor': 'left', 'receiver': 'right', 'target_xy_m': [.58, -.05]},
    {'name': 'can-right-left', 'object_id': 'can', 'donor': 'right', 'receiver': 'left', 'target_xy_m': [.57, .02]}]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--revision', default='try-1')
    parser.add_argument('--gpu-transit', action='store_true')
    parser.add_argument('--case', choices=[case['name'] for case in CASES])
    parser.add_argument('--relay-only', action='store_true')
    args = parser.parse_args()
    if not args.revision.replace('-', '').isalnum(): raise ValueError('simple revision name required')
    cases = [case for case in CASES if not args.case or case['name'] == args.case]
    out = ROOT / 'runs/20261009-handover' / args.revision; out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ['sim/object_handover.py', 'sim/object_planner.py', 'sim/objects.py', 'sim/tamp.py', 'sim/grasp.py', 'sim/object_curobo.py', 'tools/check_object_handover.py']:
        source = ROOT / name; frozen = out / 'source' / name; frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(source.read_bytes()); hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    dump(out / 'protocol.json', {'cases': cases, 'source_sha256': hashes, 'timing_mode': 'measured_delay',
        'time_budget_s': 120, 'hold_seconds': 10, 'hold_gate': {'window_s': 2, 'position_m': .001, 'rotation_rad': .02},
        'placement_tolerance_m': .01, 'no_reset_or_pose_edits_after_task_start': True,
        'handover_lookahead': 'original final table target supplied before searching donor yaw and receiver grasp; geometric filtering only',
        'donor_grasp_intent': 'declared distinct receiver; lower primitive bands ranked first, original grasp gates unchanged',
        'gpu_transit': args.gpu_transit, 'non_target_displacement_limit_m': .003,
        'relay_only': args.relay_only,
        'robot_object_contact_parameters_changed': False, 'scope': 'declared primitive development cases; not arbitrary-object transfer or physical calibration',
        'hardware_access': False, 'transfer_ready': False})
    rows = []
    for case in cases:
        folder = out / case['name']; folder.mkdir(); env = ObjectEnv(images=False); rec = None
        steps = []; planner = None; phase = 'initial'; error = None; quality = None; manifest = None
        non_target_max = {}; started = time.monotonic()
        try:
            env.reset(seed=0); env.select_object(case['object_id']); env.set_task_budget(120)
            manifest = env.manifest(); dump(folder / 'initial.json', {'case': case, 'manifest': manifest,
                'qpos': env.data.qpos, 'qvel': env.data.qvel, 'selection': env.selection()})
            Image.fromarray(env.render('operator')).save(folder / 'start.jpg', quality=90)
            rec = Recorder(env, 'tamp', root=folder / 'episode', record_images=False, seed=0,
                details={'case': case, 'selection': env.selection(), 'source_sha256': hashes, 'timing_mode': 'measured_delay'})
            rec.manifest['task'] = 'Directed registered-object handover: ' + case['object_id']; dump(rec.path / 'manifest.json', rec.manifest)
            before = rec.observation(env)
            protected = {ident: env.data.xpos[row['body']].copy() for ident, row in env.registry.items() if ident != env.object_id}
            non_target_max = {ident: 0. for ident in protected}
            def apply(action, diag, is_hold=False):
                nonlocal before
                _, _, term, trunc, info = env.hold_tamp(action, diag['planning_timing']['duration_s']) if is_hold else env.step_tamp(action)
                after = rec.observation(env)
                diag['selected_object_contacts'] = {'held': dict(env.evaluation['held']), 'touching': dict(env.evaluation['touching']),
                    'hold_seconds': dict(env.contacts.hold_seconds), 'object_id': env.object_id,
                    'position_m': env.data.xpos[env.bottle_id].copy()}
                diag['registered_objects'] = {ident: {'position_m': env.data.xpos[row['body']].copy(),
                    'velocity': env.data.qvel[row['dofadr']:row['dofadr'] + 6].copy()} for ident, row in env.registry.items()}
                rec.append(before, action, after, info, terminated=term, truncated=trunc, diagnostics=diag)
                elapsed = (before['sim_time'], after['sim_time']); before = after
                if term or trunc: raise RuntimeError('native physics/declared budget terminated current task')
                for ident, origin in protected.items():
                    drift = float(np.linalg.norm(env.data.xpos[env.registry[ident]['body']] - origin))
                    non_target_max[ident] = max(non_target_max[ident], drift)
                    if drift > .003: raise RuntimeError('non-target object displacement exceeded3mm: ' + ident)
                return elapsed
            def execute(kind, task, parameters):
                nonlocal planner
                timing = PlanningTiming('measured_delay'); planner = None
                def native():
                    nonlocal planner
                    planner = kind(env, task, parameters=parameters)
                    yield from planner.actions()
                for action, diag, is_hold in timing.actions(env, native(), rec.event):
                    start_s, end_s = apply(action, diag, is_hold); timing.committed(is_hold, start_s, end_s)
                feedback = {**planner.result, 'planning_timing': timing.summary()}
                if not feedback.get('primitive_success'): raise RuntimeError('no native skill receipt')
                steps.append({'task': task, 'feedback': feedback, 'parameters': parameters})
                dump(folder / 'progress.json', {'steps': steps, 'phase': phase, 'sim_time': float(env.data.time)})
            phase = 'pick'
            pick_kind = ObjectRelayPickPlanner if case['object_id'] == 'can' else ObjectPlanner
            pick_parameters = {'object_id': case['object_id'], 'arm': case['donor']}
            if pick_kind is ObjectRelayPickPlanner: pick_parameters['next_handover'] = {'to_arm': case['receiver']}
            execute(pick_kind, 'pick_object', pick_parameters)
            Image.fromarray(env.render('operator')).save(folder / 'picked.jpg', quality=90)
            phase = 'handover'
            relay_parameters = {'object_id': case['object_id'], 'from_arm': case['donor'], 'to_arm': case['receiver']}
            if not args.relay_only: relay_parameters['next_place'] = {'surface': 'table', 'target_xy_m': case['target_xy_m']}
            execute(ObjectHandoverPlanner, 'handover_object', relay_parameters)
            Image.fromarray(env.render('operator')).save(folder / 'transferred.jpg', quality=90)
            phase = 'hold'; quality = GraspStability(env); hold_started = float(env.data.time)
            for _ in range(300): apply(env.last_action.copy(), {'phase': '10s-receiver-hold'}); quality.read()
            steps.append({'task': 'hold', 'actual_seconds': float(env.data.time) - hold_started, 'feedback': quality.summary()})
            Image.fromarray(env.render('operator')).save(folder / 'held.jpg', quality=90)
            if quality.summary()['arms'][case['receiver']]['status'] != 'stable_within_audit_window':
                raise RuntimeError('original 10s hold audit failed; no downstream place')
            if args.gpu_transit and not args.relay_only:
                phase = 'gpu-transit'
                arm = case['receiver']; hand = env.data.xmat[env.ee_ids[arm]].reshape(3, 3)
                relative = hand.T @ env.data.xmat[env.bottle_id].reshape(3, 3)
                offset = hand.T @ (env.data.xpos[env.bottle_id] - env.data.xpos[env.ee_ids[arm]])
                yaw = np.arctan2(env.data.xmat[env.bottle_id].reshape(3, 3)[1, 0], env.data.xmat[env.bottle_id].reshape(3, 3)[0, 0])
                angle = planner.result['future_place']['yaw_offset_rad']; theta = yaw + angle
                upright = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
                rotation = upright @ relative.T; target = np.r_[case['target_xy_m'], env.placement_surface('table')['top_z'] + .20]
                tcp = target + rotation @ (np.array([0, .006, .075]) - offset)
                quat = np.empty(4); mujoco.mju_mat2Quat(quat, rotation.ravel())
                execute(ObjectCuRoboPlanner, 'move_to_pose', {'arm': arm, 'position_m': tcp.tolist(),
                    'quaternion_xyzw': quat[[1, 2, 3, 0]].tolist(), 'frame': 'world', 'planner': 'curobo', 'object_id': case['object_id']})
            if not args.relay_only:
                phase = 'place'
                execute(ObjectPlanner, 'place_object', {'object_id': case['object_id'], 'arm': case['receiver'],
                    'receptacle_id': 'table', 'target_xy_m': case['target_xy_m'], 'position_tolerance_m': .01})
            status = 'completed'
        except Exception as exc:
            status = 'error'; error = {'code': getattr(exc, 'code', 'execution_error'), 'message': str(exc)}
        finally:
            relay = next((s['feedback'] for s in steps if s['task'] == 'handover_object'), {})
            place = next((s['feedback'] for s in steps if s['task'] == 'place_object'), {})
            checks = {'verified_directed_handover': relay.get('verification_source') == 'native_receiver_contact_then_donor_release'
                    and relay.get('held_by') == case['receiver'] and relay.get('object_id') == case['object_id']
                    and relay.get('independent_hold_seconds', 0) >= .999,
                'receiver_10s_hold': bool(quality) and quality.summary()['arms'][case['receiver']]['status'] == 'stable_within_audit_window',
                'downstream_original_object_placement': place.get('primitive_success') is True
                    and place.get('verification_source') == 'native_support_release_and_stability'
                    and place.get('object_id') == case['object_id'] and place.get('position_error_m', 1) <= .01,
                'model_evidence_unchanged': manifest is not None and env.manifest()['model_evidence'] == manifest['model_evidence']}
            checks['non_target_objects_undisturbed'] = all(v <= .003 for v in non_target_max.values())
            if args.relay_only: checks.pop('downstream_original_object_placement')
            result = {'case': case, 'status': status, 'passed': all(checks.values()), 'checks': checks,
                'steps': steps, 'error': error, 'phase': phase, 'failed_feedback': planner.result if error and planner else None,
                'hold': quality.summary() if quality else None, 'sim_time': float(env.data.time),
                'wall_seconds': time.monotonic() - started, 'evaluation': env.evaluation, 'hardware_access': False, 'transfer_ready': False}
            result['non_target_max_displacement_m'] = non_target_max
            dump(folder / 'result.json', result)
            dump(folder / 'final-native-state.json', {'qpos': env.data.qpos, 'qvel': env.data.qvel,
                'ctrl': env.data.ctrl, 'sim_time': float(env.data.time), 'selection': env.selection(),
                'scope': 'read-only final diagnostic snapshot; never restored into an execution'})
            if rec and not rec._closed: rec.finish(status, env.info(), error=error, metrics={'object_handover': result})
            Image.fromarray(env.render('operator')).save(folder / 'final.jpg', quality=90)
            env.close(); rows.append(result); dump(out / 'results.json', {'complete': len(rows) == len(cases), 'cases': rows})
            print(json.dumps({'case': case['name'], 'passed': result['passed'], 'checks': checks, 'phase': phase,
                'error': error, 'sim_time': result['sim_time']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
