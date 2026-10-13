"""Predeclared non-upright starts, unchanged native pick/hold/upright-place gates."""
import argparse
import hashlib
import time

import mujoco
import numpy as np
from PIL import Image

from sim.environment import ROOT
from sim.recording import Recorder, dump
from sim.oriented_grasp import OrientedPlanner, CONTRACT
from sim.object_planner import ObjectPlanner
from sim.planning_timing import PlanningTiming
from sim.decision_timing import GraspStability
from harness.object_handover import ProtectedObjectEnv

CASES = [
    {'name': 'can-right-horizontal', 'object_id': 'can', 'arm': 'right', 'origin_xy_m': [.4, -.32], 'yaw_rad': 0., 'axis_rotation': 'y', 'target_xy_m': [.58, -.28]},
    {'name': 'can-left-oblique', 'object_id': 'can', 'arm': 'left', 'origin_xy_m': [.30, .16], 'yaw_rad': .6, 'axis_rotation': 'y', 'target_xy_m': [.42, .18]},
    {'name': 'block-left-horizontal', 'object_id': 'block', 'arm': 'left', 'origin_xy_m': [.45, .21], 'yaw_rad': 0., 'axis_rotation': 'x', 'target_xy_m': [.55, .08]},
    {'name': 'bottle-right-horizontal', 'object_id': 'bottle', 'arm': 'right', 'origin_xy_m': [.33, -.20], 'yaw_rad': 0., 'axis_rotation': 'y', 'target_xy_m': [.56, -.20]},
]


def initialize(env, case):
    """Declare the initial fixture at t=0, before recording or any control."""
    env.reset(seed=0); row = env.registry[case['object_id']]; address = row['qposadr']
    angle = np.pi/2; c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]]) if case['axis_rotation'] == 'y' else np.array([[1., 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(case['yaw_rad']), np.sin(case['yaw_rad']); rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ rotation
    env.data.qpos[address:address+2] = case['origin_xy_m']
    mujoco.mju_mat2Quat(env.data.qpos[address+3:address+7], rotation.ravel()); mujoco.mj_forward(env.model, env.data)
    from sim.scene import bounds
    bottom = min(bounds(env.model, env.data, g)[0][2] for g in row['geoms'])
    table = env.placement_surface('table')['top_z']; env.data.qpos[address+2] += table - bottom
    mujoco.mj_forward(env.model, env.data); env.select_object(case['object_id']); env._protect()
    env.evaluation = env._evaluate(0.); env.set_task_budget(120)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='try-1'); parser.add_argument('--baseline', action='store_true'); parser.add_argument('--gpu-upright',action='store_true'); parser.add_argument('--gpu-recovery',action='store_true'); parser.add_argument('--lookahead',action='store_true');parser.add_argument('--pick-hint-gpu',action='store_true');args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple immutable experiment name required')
    out = ROOT / 'runs/20261009-poses' / args.name; out.mkdir(parents=True, exist_ok=False)
    files = [str(p.relative_to(ROOT)) for folder in ('sim', 'harness') for p in sorted((ROOT/folder).glob('*.py'))]
    files += ['tools/check_pose_grasp.py']; hashes = {}
    for name in files:
        source = ROOT/name; frozen = out/'source'/name; frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(source.read_bytes()); hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = {'cases': CASES, 'source_sha256': hashes, 'proposal_contract': CONTRACT,
        'planner': 'ObjectPlanner baseline' if args.baseline else 'OrientedPlanner',
        'time_budget_s': 120, 'timing_mode': 'measured_delay', 'hold_seconds': 10,
        'hold_limits': {'position_m': .001, 'rotation_rad': .02},
        'placement_tolerance_m': .010, 'placement_stability_s': 1., 'original_upright_place_required': True,
        'gpu_upright_before_place':args.gpu_upright,'gpu_recovery_after_direct_placement_failure':args.gpu_recovery,
        'gpu_recovery_clearance_m':[.12,.08,.05] if args.gpu_recovery else [],
        'fresh_replanning_after_wait_rejection':2 if args.gpu_recovery else 0,
        'grasp_placement_lookahead':args.lookahead,
        'explicit_gpu_transit_using_pick_hint':args.pick_hint_gpu,
        'initial_pose_edits_only_before_first_recorded_control': True, 'no_reset_or_pose_edits_after_task_start': True,
        'no_physics_grasp_placement_threshold_changes': True, 'hardware_access': False, 'transfer_ready': False}
    dump(out/'protocol.json', protocol); rows = []
    for case in CASES:
        folder = out/case['name']; folder.mkdir(); env = ProtectedObjectEnv(images=False)
        rec = None; planner = None; steps = []; error = None; phase = 'initial'; quality = None; manifest = None; held_s = 0.; attempts=[]
        started = time.monotonic()
        try:
            initialize(env, case); manifest = env.manifest()
            initial = {ident: env.data.xpos[value['body']].copy() for ident, value in env.registry.items()}
            dump(folder/'initial.json', {'case': case, 'manifest': manifest, 'qpos': env.data.qpos, 'qvel': env.data.qvel,
                'initial_objects': initial, 'axis': env.data.xmat[env.bottle_id].reshape(3,3)[:,2]})
            Image.fromarray(env.render('operator')).save(folder/'start.jpg', quality=90)
            rec = Recorder(env, 'tamp', root=folder/'episode', record_images=False, seed=0,
                details={**protocol, 'case': case, 'selection': env.selection()})
            before = rec.observation(env)
            def apply(action, diagnostic, hold=False):
                nonlocal before
                _, _, term, trunc, info = env.hold_tamp(action, diagnostic['planning_timing']['duration_s']) if hold else env.step_tamp(action)
                after = rec.observation(env)
                diagnostic['object_pose'] = {'position_m': env.data.xpos[env.bottle_id].copy(),
                    'rotation': env.data.xmat[env.bottle_id].reshape(3,3).copy(), 'held': dict(env.evaluation['held'])}
                rec.append(before, action, after, info, terminated=term, truncated=trunc, diagnostics=diagnostic); before = after
                if term or trunc: raise RuntimeError('native physics or original task budget terminated')
            def execute(cls,native_task,parameters,label):
                nonlocal planner
                for attempt in range(1,4 if args.gpu_recovery else 2):
                    timing=PlanningTiming('measured_delay');planner=None;commands=0
                    def native():
                        nonlocal planner
                        planner=cls(env,native_task,parameters=parameters)
                        yield from planner.actions()
                    try:
                        for action,diagnostic,is_hold in timing.actions(env,native(),rec.event):
                            start=float(env.data.time);apply(action,diagnostic,hold=is_hold)
                            timing.committed(is_hold,start,float(env.data.time));commands+=1
                            dump(folder/'progress.json',{'phase':phase,'sim_time':float(env.data.time),'commands':commands,
                                'planner':timing.summary(),'steps':steps,'attempts':attempts})
                        if not planner.result.get('primitive_success'):raise RuntimeError('missing original native skill success')
                        steps.append({'task':label,'commands':commands,'feedback':copy_value(planner.result),'planning':timing.summary()})
                        return
                    except Exception as exc:
                        attempts.append({'task':label,'attempt':attempt,'parameters':parameters,'sim_time':float(env.data.time),
                            'error':{'code':getattr(exc,'code',type(exc).__name__),'message':str(exc)},
                            'feedback':copy_value(planner.result) if planner else None,'planning':timing.summary()})
                        if (not args.gpu_recovery or getattr(exc,'code',None)!='planning_state_changed' or attempt>=3
                            or env._done or env.evaluation['invalid']):raise
                        # A stale proposal was never dispatched. Reconstruct
                        # from the CURRENT state in this same original task.
            tasks=['pick_object','hold']+(['upright'] if args.gpu_upright else [])+(['pick-hint-transit'] if args.pick_hint_gpu else [])+['place_object']
            for task in tasks:
                phase = task
                if task == 'hold':
                    quality = GraspStability(env); begin = float(env.data.time); command = env.last_action.copy()
                    for _ in range(300): apply(command, {'phase':'10s-hold'}); quality.read()
                    held_s = float(env.data.time)-begin; steps.append({'task':task,'seconds':held_s,'feedback':quality.summary()})
                    Image.fromarray(env.render('operator')).save(folder/'held.jpg', quality=90)
                    if quality.summary()['arms'][case['arm']]['status'] != 'stable_within_audit_window': raise RuntimeError('original full-hold audit failed')
                    continue
                parameters = {'object_id':case['object_id'],'arm':case['arm']}
                if task=='pick_object' and args.lookahead:parameters['next_placement']={'surface':'table','target_xy_m':case['target_xy_m']}
                if task == 'place_object': parameters.update(receptacle_id='table', target_xy_m=case['target_xy_m'],position_tolerance_m=.01)
                cls = ObjectPlanner if args.baseline else OrientedPlanner
                native_task=task
                if task in ('upright', 'pick-hint-transit'):
                    from sim.object_curobo import ObjectCuRoboPlanner
                    from harness.object_transport import pose_for_object
                    from sim.scene import SceneDirectory,quaternion
                    body=env.ee_ids[case['arm']];hand=env.data.xmat[body].reshape(3,3)
                    tcp={'position_m':(env.data.xpos[body]+hand@np.array([0.,.006,.075])).tolist(),'quaternion_xyzw':quaternion(hand)}
                    obj=next(row for row in SceneDirectory(env).snapshot('native:'+str(env.tick))['objects'] if row['object_id']==case['object_id'])
                    if task == 'pick-hint-transit':
                        future=steps[0]['feedback'].get('future_placement')
                        if not future or future['target_xy_m']!=case['target_xy_m']:raise RuntimeError('missing original-goal pick hint')
                        measured=np.asarray(obj['rotation_matrix']);yaw=float(np.arctan2(measured[1,0],measured[0,0]))
                        delta=float(np.arctan2(np.sin(future['yaw_rad']-yaw),np.cos(future['yaw_rad']-yaw)))
                        desired=np.r_[case['target_xy_m'],env.placement_surface('table')['top_z']+future['transit_height_m']]
                        pose=pose_for_object(tcp,obj,desired,delta)
                    else:
                        desired=np.r_[case['target_xy_m'],env.placement_surface('table')['top_z']+.25]
                        pose=pose_for_object(tcp,obj,desired,0.)
                    parameters={**pose,'object_id':case['object_id'],'arm':case['arm'],'frame':'world'}
                    native_task='move_to_pose';cls=ObjectCuRoboPlanner
                try:execute(cls,native_task,parameters,task)
                except Exception as exc:
                    if (not args.gpu_recovery or task!='place_object' or getattr(exc,'code',None)!='placement_unreachable'
                        or env._done or env.evaluation['invalid'] or not env.evaluation['held'][case['arm']]):raise
                    from sim.object_curobo import ObjectCuRoboPlanner
                    from harness.object_transport import pose_for_object
                    from sim.scene import SceneDirectory,quaternion
                    recovered=False
                    for clearance in (.12,.08,.05):
                        body=env.ee_ids[case['arm']];hand=env.data.xmat[body].reshape(3,3)
                        tcp={'position_m':(env.data.xpos[body]+hand@np.array([0.,.006,.075])).tolist(),'quaternion_xyzw':quaternion(hand)}
                        obj=next(row for row in SceneDirectory(env).snapshot('native:'+str(env.tick))['objects'] if row['object_id']==case['object_id'])
                        measured=np.asarray(obj['rotation_matrix']);yaw=float(np.arctan2(measured[1,0],measured[0,0]))
                        desired=np.r_[case['target_xy_m'],env.placement_surface('table')['top_z']+clearance]
                        pose=pose_for_object(tcp,obj,desired,-yaw)
                        motion={**pose,'object_id':case['object_id'],'arm':case['arm'],'frame':'world'}
                        try:execute(ObjectCuRoboPlanner,'move_to_pose',motion,'gpu-upright-recovery');recovered=True;break
                        except Exception as failure:
                            if env._done or env.evaluation['invalid'] or getattr(failure,'code',None) not in ('curobo_no_path','object_gpu_rejected'):raise
                    if not recovered:raise RuntimeError('all declared GPU upright recovery hypotheses failed; original goal retained')
                    execute(cls,native_task,parameters,task)
                if task == 'pick_object':Image.fromarray(env.render('operator')).save(folder/'picked.jpg',quality=90)
            status='completed'
        except Exception as exc:
            status='error';error={'code':getattr(exc,'code',type(exc).__name__),'message':str(exc)}
        finally:
            final={ident:env.data.xpos[value['body']].copy() for ident,value in env.registry.items()}
            position_error=float(np.linalg.norm(final[case['object_id']][:2]-case['target_xy_m']))
            feedback=steps[-1]['feedback'] if steps else {}
            checks={'actual_contact_lift':bool(steps and steps[0]['feedback'].get('primitive_success')),
                'full10s_hold':bool(quality and held_s>=9.9 and quality.summary()['arms'][case['arm']]['status']=='stable_within_audit_window'),
                'original_upright_place_release_stability':bool(feedback.get('primitive_success') and feedback.get('verification_source')=='native_support_release_and_stability' and env.evaluation.get('placement',{}).get('confirmed')),
                'original_target':position_error<=.010,'scene_valid':not env.evaluation['invalid'],
                'compiled_physics_unchanged':bool(manifest and env.manifest()['model_evidence']==manifest['model_evidence']),
                'within_original120s':float(env.data.time)<=120}
            result={'case':case,'status':status,'passed':all(checks.values()),'checks':checks,'error':error,'phase':phase,
                'steps':steps,'attempts':attempts,'failed_feedback':planner.result if planner and status=='error' else None,
                'initial_objects':initial if manifest else None,'final_objects':final,'position_error_m':position_error,
                'final_evaluation':env.evaluation,'hold':quality.summary() if quality else None,'actual_hold_seconds':held_s,
                'sim_time':float(env.data.time),'wall_seconds':time.monotonic()-started,'hardware_access':False,'transfer_ready':False}
            if rec:rec.finish(status,env.info(),error=str(error) if error else None,metrics={'oriented_grasp':result})
            Image.fromarray(env.render('operator')).save(folder/'final.jpg',quality=90);dump(folder/'result.json',result)
            rows.append(result);dump(out/'results.json',{'complete':len(rows)==len(CASES),'cases':rows})
            print(case['name'],status,error,float(env.data.time),position_error,flush=True);env.close()


def copy_value(value):
    import copy
    return copy.deepcopy(value)


if __name__=='__main__':main()
