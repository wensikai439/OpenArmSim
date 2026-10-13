"""Immutable bottle direction/forward tasks; native motion and original gates.

Synthetic starting fixtures are declared before the first command. No reset,
object pose edit, physics change or goal substitution is allowed in a task.
"""
import argparse
import copy
import hashlib
import json
import re
import time

import mujoco
import numpy as np
from PIL import Image

from sim.environment import ROOT
from sim.object_planner import ObjectPlanner
from sim.object_handover import ObjectHandoverPlanner, ObjectRelayPickPlanner
from sim.bottle_routing import BottleHandoverPlanner, BottlePickPlanner, BottlePlacementPlanner
from sim.object_curobo import ObjectCuRoboPlanner, metadata
from sim.planning_timing import PlanningTiming
from sim.decision_timing import GraspStability
from sim.oriented_grasp import initial_conflicts
from sim.recording import Recorder, dump
from harness.object_handover import ProtectedObjectEnv

CASES = [
    {'name': 'original-left-right', 'arm': 'left', 'receiver': 'right',
     'origin_xy_m': None, 'target_xy_m': [.56, -.13]},
    {'name': 'left-right', 'arm': 'left', 'receiver': 'right',
     'origin_xy_m': [.44, .18], 'target_xy_m': [.56, -.13]},
    {'name': 'right-forward', 'arm': 'right', 'receiver': None,
     'origin_xy_m': None, 'target_xy_m': [.583, -.173]},
    {'name': 'left-forward', 'arm': 'left', 'receiver': None,
     'origin_xy_m': [.44, .18], 'target_xy_m': [.52, .18]},
]


def load_cases(path=None, selected=None):
    """Read declared fixtures; never repair or filter a failed physical start."""
    metadata=None
    if path:
        payload=path.read_bytes();data=json.loads(payload)
        if data.get('schema')!='openarm-bottle-batch/1':raise ValueError('unknown bottle batch schema')
        values=data.get('cases');metadata={'path':str(path),'sha256':hashlib.sha256(payload).hexdigest(),
            'description':data.get('description',''), 'cases_declared_before_execution':True}
    else:values=copy.deepcopy(CASES)
    if not isinstance(values,list) or not 1<=len(values)<=32:raise ValueError('declare1–32 cases')
    names=set()
    for row in values:
        if not isinstance(row,dict) or set(row)!={'name','arm','receiver','origin_xy_m','target_xy_m'}:
            raise ValueError('case fields must be name/arm/receiver/origin_xy_m/target_xy_m')
        name=row['name']
        if not isinstance(name,str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',name) or len(name)>48 or name in names:
            raise ValueError('unique simple case names required')
        names.add(name)
        if row['arm'] not in ('left','right') or row['receiver'] not in (None,'left','right') or row['receiver']==row['arm']:
            raise ValueError('distinct valid arms required')
        # The production optional provider supports left→right; keep the old
        # right→left demo separate rather than test an unconnected new route.
        if row['receiver']=='left':raise ValueError('this batch targets the connected left-to-right provider')
        for field in ('origin_xy_m','target_xy_m'):
            xy=row[field]
            if field=='origin_xy_m' and xy is None:continue
            if (not isinstance(xy,list) or len(xy)!=2 or any(type(x) not in (int,float) for x in xy)
                or not np.isfinite(xy).all()):raise ValueError('finite two-dimensional XY required')
    if selected and selected not in names:raise ValueError('requested case is not in the declared batch')
    return [copy.deepcopy(row) for row in values if not selected or row['name']==selected],metadata


def initialize(env, case):
    env.reset(seed=0)
    if case['origin_xy_m'] is not None:
        assert env.data.time == 0 and env.tick == 0
        address = env.registry['bottle']['qposadr']
        env.data.qpos[address:address + 2] = case['origin_xy_m']
        mujoco.mj_forward(env.model, env.data)
        env.evaluation = env._evaluate(0.)
    env._protect()
    env.set_task_budget(120)


def freeze(out):
    names = [str(p.relative_to(ROOT)) for d in ('sim', 'harness')
             for p in sorted((ROOT / d).glob('*.py'))]
    names += ['tools/check_bottle_routes.py', 'tools/object_gpu.py', 'tools/scene_gpu.py']
    hashes = {}
    for name in names:
        payload = (ROOT / name).read_bytes(); target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return hashes


def execute_gpu_with_recovery(env, execute, motion, max_replans, apply, feedback, record):
    """Bounded same-target development recovery, never a reset or success flag.

    A stale correction proposal can follow earlier executed motion. Preserve
    that attempt, hold the last applied command for1native second, then let a
    fresh planner observe. The original absolute pose and task clock stay fixed.
    """
    locked=copy.deepcopy(motion);attempts=[]
    for attempt in range(max_replans+1):
        try:
            execute(ObjectCuRoboPlanner,'move_to_pose',copy.deepcopy(locked))
            return attempts
        except Exception as exc:
            if getattr(exc,'code',None)!='planning_state_changed':raise
            attempts.append({'attempt':attempt,'failure_code':exc.code,
                'feedback':copy.deepcopy(feedback()),'sim_time':float(env.data.time),
                'original_absolute_motion':copy.deepcopy(locked),
                'scope':'raw stopped proposal; previous GPU commands may already have executed; never a completed motion receipt'})
            record(attempts)
            if attempt>=max_replans:raise
            arm=locked['arm']
            if env._done or not env.evaluation['held'][arm]:
                raise RuntimeError('same-goal GPU recovery requires a valid current native hold')
            dt=float(env.model.opt.timestep)
            remaining=round(1./dt)
            chunk=max(1,round(1./env.profile['control_hz']/dt))
            command=env.last_action.copy()
            while remaining:
                units=min(chunk,remaining)
                apply(command,{'phase':'same-goal-settle',
                    'planning_timing':{'duration_s':units*dt},
                    'original_absolute_motion':copy.deepcopy(locked)},True)
                remaining-=units
                if env._done or not env.evaluation['held'][arm]:
                    raise RuntimeError('native hold lost or task ended while settling; no further motion')
    raise AssertionError('bounded recovery must return or raise')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='baseline')
    parser.add_argument('--strategy', choices=('legacy', 'generic', 'goal', 'robust', 'fast', 'adaptive', 'refined', 'margin', 'task'), default='legacy')
    parser.add_argument('--case', help='one declared case name')
    parser.add_argument('--batch',type=lambda value: (ROOT/value).resolve(),help='predeclared case JSON')
    parser.add_argument('--output',type=lambda value: (ROOT/value).resolve(),help='new immutable directory inside runs/')
    parser.add_argument('--placement',choices=('current','cartesian','constrained','connected','predictive'),default='current',help='optional native table descent provider')
    parser.add_argument('--gpu-retries',type=int,choices=(0,1,2),default=0,
        help='predeclare bounded same-target replans only after planning_state_changed; old default remains0')
    parser.add_argument('--handover-ik',choices=('full','kinematic'),default='full',
        help='optional original bottle relay with minimal scratch IK iterations; original full collision gates')
    parser.add_argument('--gpu-cache', action='store_true', help='explicit scene-aware8005worker; old backend unchanged')
    args = parser.parse_args()
    if not args.name.replace('-', '').isalnum(): raise ValueError('simple immutable run name required')
    cases,batch=load_cases(args.batch,args.case)
    out = args.output or ROOT / 'runs/20261009-bottle' / args.name
    if args.output and (not out.is_relative_to((ROOT/'runs').resolve()) or out==(ROOT/'runs').resolve()):
        raise ValueError('output must be a new child of runs/')
    out.mkdir(parents=True, exist_ok=False)
    source = freeze(out)
    if args.gpu_cache:
        from sim.scene_curobo import install_isolated as scene_gpu
        from sim.cache_curobo import install_isolated as cache_gpu, CacheCuRoboPlanner
        scene_gpu(); cache_gpu()
        global ObjectCuRoboPlanner
        ObjectCuRoboPlanner = CacheCuRoboPlanner
        name='tools/scene_cache.py'; payload=(ROOT/name).read_bytes()
        target=out/'source'/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(payload)
        source[name]=hashlib.sha256(payload).hexdigest()
    backend = metadata()
    protocol = {'cases': cases, 'source_sha256': source, 'strategy': args.strategy,
        'placement_provider':args.placement,
        'handover_IK_pipeline':args.handover_ik,
        'GPU': backend, 'time_budget_s': 120, 'timing_mode': 'measured_delay',
        'gpu_worker_port':8005 if args.gpu_cache else 8004,
        'receiver_or_single_arm_hold_seconds': 10, 'hold_window_s': 2,
        'hold_limits': {'position_m': .001, 'rotation_rad': .02},
        'placement_tolerance_m': .01, 'non_target_displacement_limit_m': .003,
        'forward_definition': 'explicit per-case initial/target world XY' if batch else 'original bottle world X + 0.08m; world Y unchanged',
        'batch_source':batch,
        'fixtures': 'explicit synthetic upright development starts; not measured real scenes',
        'gpu_retry_policy':{'max_replans':args.gpu_retries,'eligible_failure':'planning_state_changed',
            'native_settle_seconds':1.,'original_absolute_TCP_target_preserved':True,
            'same_task_clock_and120s_budget':True,'failed_attempts_retained':True},
        'no_reset_or_pose_edit_after_task_start': True,
        'no_physics_or_acceptance_change': True, 'hardware_access': False, 'transfer_ready': False}
    if args.strategy == 'task':
        protocol['proposal_difference'] = {
            'original_sampler_first': True, 'baseline': 'current MarginBottlePick including directed-relay refinements',
            'fallback': 'continuous upright single-arm placement yaw after original exhaustion; directed relays stay current provider',
            'fallback_placement_collision': 'actual ConnectedBottlePlacement native predicate; pick-only extra12mm not used for this placement lookahead',
            'original_grasp_joint_margin_rad': .020, 'original_task_goal_and_native_scores_unchanged': True}
    if args.placement == 'predictive':
        protocol['execution_validation_difference'] = {
            'trigger': 'completed original checked path and feedback, then correction failure',
            'changed_check': 'bounded biased actuator increment validated by copied native150ms dynamics rather than ideal command-pose FK',
            'maximum_command_increment_rad': .008, 'native_collision_depth_m': .0005,
            'preview_state_never_restored': True, 'original_task_goal_release_support_stability_scores_unchanged': True,
            'controller_scope': 'stateless native position-PD only; no source-controller or callback support',
            'candidate_not_deployed': True}
    dump(out / 'protocol.json', protocol)
    rows = []
    goal_strategy = args.strategy in ('goal', 'robust', 'fast', 'adaptive', 'refined', 'margin', 'task')
    goal_pick = BottlePickPlanner
    if args.strategy == 'robust':
        from sim.robust_grasp import RobustBottlePick
        goal_pick = RobustBottlePick
    elif args.strategy == 'margin':
        from sim.margin_grasp import MarginBottlePick
        goal_pick = MarginBottlePick
    elif args.strategy == 'task':
        from sim.task_grasp import TaskBottlePick
        goal_pick = TaskBottlePick
    elif args.strategy in ('fast', 'adaptive', 'refined'):
        from sim.adaptive_grasp import FastBottlePick, AdaptiveBottlePick, RefinedBottlePick
        goal_pick = {'fast': FastBottlePick, 'adaptive': AdaptiveBottlePick, 'refined': RefinedBottlePick}[args.strategy]
    for case in cases:
        folder = out / case['name']; folder.mkdir()
        env = ProtectedObjectEnv(images=False); rec = None; planner = None
        steps = []; phase = 'initial'; error = None; hold = None; hold_seconds = 0; recovery_attempts=[];phase_states=[]
        manifest = None; started = time.monotonic(); audit_all_stable = True
        try:
            initialize(env, case); manifest = env.manifest()
            conflicts = initial_conflicts(env)
            dump(folder / 'initial.json', {'case': case, 'manifest': manifest,
                'qpos': env.data.qpos, 'qvel': env.data.qvel, 'native_conflicts': conflicts,
                'bottle_origin_m': env.data.xpos[env.bottle_id]})
            Image.fromarray(env.render('operator')).save(folder / 'start.jpg', quality=90)
            if conflicts: raise RuntimeError('original starting fixture has native overlap: ' + str(conflicts))
            rec = Recorder(env, 'tamp', root=folder / 'episode', record_images=False, seed=0,
                details={**protocol, 'case': case})
            before = rec.observation(env)
            def save_phase_state(label):
                phase_states.append({'phase':label,'sim_time':float(env.data.time),
                    'qpos':env.data.qpos.copy(),'qvel':env.data.qvel.copy(),
                    'ctrl':env.data.ctrl.copy(),'last_action':env.last_action.copy(),
                    'held':copy.deepcopy(env.evaluation['held']),
                    'touching':copy.deepcopy(env.evaluation['touching']),
                    'native_object_position_m':env.data.xpos[env.bottle_id].copy(),
                    'native_object_rotation':env.data.xmat[env.bottle_id].copy(),
                    'scope':'private read-only native phase snapshot; never a state restoration or success flag'})
                dump(folder/'phase-states.json',phase_states)
            def apply(action, diagnostic, is_hold=False):
                nonlocal before
                result = env.hold_tamp(action, diagnostic['planning_timing']['duration_s']) if is_hold else env.step_tamp(action)
                _, _, term, trunc, info = result
                after = rec.observation(env)
                diagnostic['native_object'] = {'position_m': env.data.xpos[env.bottle_id].copy(),
                    'held': dict(env.evaluation['held']), 'touching': dict(env.evaluation['touching']),
                    'hold_seconds': dict(env.contacts.hold_seconds)}
                rec.append(before, action, after, info, terminated=term, truncated=trunc, diagnostics=diagnostic)
                before = after
                if term or trunc: raise RuntimeError('native task invalid or original budget exhausted')
            def execute(cls, task, parameters):
                nonlocal planner
                timing = PlanningTiming('measured_delay'); planner = None
                def actions():
                    nonlocal planner
                    planner = cls(env, task, parameters=parameters)
                    # Private disk feedback survives a long blocked search.
                    # It never changes physics, goals, task stages or UI text.
                    planner.search_feedback = lambda value: dump(folder / 'search.json',
                        {'phase': phase, 'sim_time': float(env.data.time), 'search': value,
                         'source': 'scratch proposal progress; not an execution receipt'})
                    yield from planner.actions()
                for action, diagnostic, is_hold in timing.actions(env, actions(), rec.event):
                    begin = float(env.data.time); apply(action, diagnostic, is_hold)
                    timing.committed(is_hold, begin, float(env.data.time))
                    dump(folder / 'progress.json', {'phase': phase, 'sim_time': float(env.data.time),
                        'steps': steps, 'planning': timing.summary()})
                if not planner.result.get('primitive_success'): raise RuntimeError('native primitive receipt missing')
                steps.append({'task': task, 'parameters': parameters, 'feedback': copy.deepcopy(planner.result),
                    'planning': timing.summary()})
                save_phase_state(task)
                dump(folder / 'progress.json', {'phase': phase, 'sim_time': float(env.data.time), 'steps': steps})
            phase = 'pick'
            params = {'object_id': 'bottle', 'arm': case['arm']}
            pick = ObjectPlanner
            if case['receiver'] and (args.strategy=='generic' or goal_strategy):
                pick = goal_pick if goal_strategy else ObjectRelayPickPlanner
                params['next_handover'] = {'to_arm': case['receiver']}
            elif goal_strategy:
                pick = goal_pick
                params['next_placement']={'surface':'table','target_xy_m':case['target_xy_m'],'position_tolerance_m':.01}
            execute(pick, 'pick_object', params)
            Image.fromarray(env.render('operator')).save(folder / 'picked.jpg', quality=90)
            arm = case['arm']
            if case['receiver']:
                phase = 'handover'
                params = {'object_id': 'bottle', 'from_arm': case['arm'], 'to_arm': case['receiver']}
                if args.strategy=='generic' or goal_strategy:
                    params['next_place'] = {'surface': 'table', 'target_xy_m': case['target_xy_m']}
                relay_cls = BottleHandoverPlanner if goal_strategy else ObjectHandoverPlanner if args.strategy=='generic' else ObjectPlanner
                if goal_strategy and args.handover_ik=='kinematic':
                    from sim.fast_handover import FastBottleHandover
                    relay_cls=FastBottleHandover
                execute(relay_cls,'handover_object' if args.strategy=='generic' or goal_strategy else 'handover_bottle',params)
                arm = case['receiver']
                Image.fromarray(env.render('operator')).save(folder / 'transferred.jpg', quality=90)
            phase = 'hold'; hold = GraspStability(env); begin = float(env.data.time)
            command = env.last_action.copy()
            for _ in range(300):
                apply(command, {'phase': '10s-native-hold'}); hold.read()
                if float(env.data.time) - begin >= 2:
                    audit_all_stable &= hold.summary()['arms'][arm]['status'] == 'stable_within_audit_window'
            hold_seconds = float(env.data.time) - begin
            steps.append({'task': 'hold', 'seconds': hold_seconds, 'feedback': hold.summary(),
                'all_full_windows_stable': audit_all_stable})
            save_phase_state('hold')
            Image.fromarray(env.render('operator')).save(folder / 'held.jpg', quality=90)
            if not audit_all_stable: raise RuntimeError('an original full native hold window failed')
            phase = 'gpu-transit'
            hand = env.data.xmat[env.ee_ids[arm]].reshape(3, 3)
            obj = env.data.xmat[env.bottle_id].reshape(3, 3)
            relative = hand.T @ obj
            offset = hand.T @ (env.data.xpos[env.bottle_id] - env.data.xpos[env.ee_ids[arm]])
            yaw = np.arctan2(obj[1, 0], obj[0, 0])
            relay = next((s['feedback'] for s in steps if s['task'] in ('handover_object','handover_bottle')), {})
            angle = relay.get('future_place', {}).get('yaw_offset_rad', 0.)
            if not case['receiver'] and goal_strategy:
                future=next(s['feedback'] for s in steps if s['task']=='pick_object').get('future_placement',{})
                desired=future.get('yaw_rad')
                if desired is not None:angle=float(np.arctan2(np.sin(desired-yaw),np.cos(desired-yaw)))
            theta = yaw + angle
            upright = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
            rotation = upright @ relative.T
            base = np.r_[case['target_xy_m'], env.placement_surface('table')['top_z'] + .20]
            tcp = base + rotation @ (np.array([0, .006, .075]) - offset)
            quat = np.empty(4); mujoco.mju_mat2Quat(quat, rotation.ravel())
            locked_motion={'arm': arm, 'object_id': 'bottle', 'position_m': tcp.tolist(),
                'quaternion_xyzw': quat[[1,2,3,0]].tolist(), 'frame': 'world', 'planner': 'curobo'}
            def save_recovery(attempts):
                recovery_attempts[:]=copy.deepcopy(attempts)
                dump(folder/'recovery.json',recovery_attempts)
            execute_gpu_with_recovery(env,execute,locked_motion,args.gpu_retries,apply,
                lambda:planner.result,save_recovery)
            phase = 'place'
            placement=BottlePlacementPlanner if goal_strategy else ObjectPlanner
            if args.placement=='cartesian':
                from sim.cartesian_place import CartesianBottlePlacement
                placement=CartesianBottlePlacement
            elif args.placement=='constrained':
                from sim.constrained_place import ConstrainedBottlePlacement
                placement=ConstrainedBottlePlacement
            elif args.placement=='connected':
                from sim.connected_place import ConnectedBottlePlacement
                placement=ConnectedBottlePlacement
            elif args.placement=='predictive':
                from sim.predictive_place import PredictiveBottlePlacement
                placement=PredictiveBottlePlacement
            execute(placement, 'place_object', {'object_id': 'bottle', 'arm': arm, 'receptacle_id': 'table',
                'target_xy_m': case['target_xy_m'], 'position_tolerance_m': .01})
            status = 'completed'
        except Exception as exc:
            status = 'error'; error = {'code': getattr(exc, 'code', type(exc).__name__), 'message': str(exc)}
        finally:
            relay = next((s['feedback'] for s in steps if s['task'] in ('handover_object','handover_bottle')), {})
            place = next((s['feedback'] for s in steps if s['task'] == 'place_object'), {})
            transit = next((s['feedback'] for s in steps if s['task'] == 'move_to_pose'), {})
            checks = {'original_native_pick': any(s['task'] == 'pick_object' for s in steps),
                '10s_native_hold': hold_seconds >= 9.999 and audit_all_stable and bool(hold)
                    and hold.summary()['arms'][case['receiver'] or case['arm']]['status'] == 'stable_within_audit_window',
                'actual_GPU_transport': transit.get('primitive_success') is True
                    and transit.get('planner') == 'official cuRobo V2 GPU IK/trajectory optimization'
                    and transit.get('native_path_revalidation') is True and transit.get('fallback') is False
                    and transit.get('worker_port') == (8005 if args.gpu_cache else 8004) and bool(transit.get('plans'))
                    and transit.get('backend_metadata', {}).get('ready') is True
                    and transit.get('backend_metadata', {}).get('worker_source_sha256') == source['tools/object_gpu.py']
                    and transit.get('verification_source') == 'robot_forward_kinematics'
                    and transit.get('position_error_m', 1) <= .012 and transit.get('rotation_error_rad', 1) <= .08,
                'original_native_placement': place.get('primitive_success') is True
                    and place.get('verification_source') == 'native_support_release_and_stability'
                    and place.get('position_error_m', 1) <= .01,
                'unchanged_model': manifest is not None and env.manifest()['model_evidence'] == manifest['model_evidence'],
                'unselected_objects_undisturbed': env.evaluation.get('non_target_object_guard', {}).get('passed') is True}
            if case['receiver']:
                checks['native_directed_handover'] = relay.get('primitive_success') is True and relay.get('held_by') == case['receiver']
                checks['independent_receiver_hold'] = relay.get('verification_source') == 'native_receiver_contact_then_donor_release'
            row = {'case': case, 'status': status, 'passed': all(checks.values()), 'checks': checks,
                'phase': phase, 'error': error, 'steps': steps, 'failed_feedback': planner.result if error and planner else None,
                'recovery_attempts':recovery_attempts,
                'sim_time': float(env.data.time), 'wall_seconds': time.monotonic() - started,
                'hold_seconds': hold_seconds, 'hold': hold.summary() if hold else None,
                'evaluation': env.evaluation, 'hardware_access': False, 'transfer_ready': False}
            dump(folder / 'result.json', row)
            dump(folder / 'final-native-state.json', {'qpos': env.data.qpos, 'qvel': env.data.qvel,
                'ctrl': env.data.ctrl, 'sim_time': float(env.data.time)})
            if rec and not rec._closed: rec.finish(status, env.info(), error=error, metrics={'bottle_route': row})
            Image.fromarray(env.render('operator')).save(folder / 'final.jpg', quality=90)
            env.close(); rows.append(row); dump(out / 'results.json', {'complete': len(rows) == len(cases), 'cases': rows})
            print({'case': case['name'], 'passed': row['passed'], 'phase': phase, 'error': error,
                'checks': checks, 'sim_time': row['sim_time']}, flush=True)


if __name__ == '__main__': main()
