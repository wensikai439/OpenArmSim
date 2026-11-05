"""A natural language forward request; actual API, original native gates."""
import argparse
import hashlib
import time

from sim.environment import ROOT
from sim.recording import dump
from tools.check_task_queue import setup
from tools.check_task_api import collect

PROMPT = ('请用右手把瓶子向机器人正前方放8厘米，搬运时使用cuRobo，其他物体不动。'
    '这是同一项仿真任务，总预算120秒，计入模型与规划等待。'
    '不要重置、加时或改变原目标，松手后复查。')


def gpu_evidence(record, worker_hash,worker_port=8004):
    moves=[r for r in record['receipts'] if r['execution']['skill'] in ('move_to_pose','move_relative')]
    completed=[r for r in moves if r['feedback'].get('primitive_success') is True]
    if not completed:return False
    # Tool choice is the model's. A native stale-proposal rejection is allowed
    # only as an archived retry; it is never scored as executed transport.
    if any(r['feedback'].get('primitive_success') is not True
        and r['feedback'].get('failure_code')!='planning_state_changed' for r in moves):return False
    return all(gpu_motion_evidence(r,worker_hash,worker_port) for r in completed)


def gpu_motion_evidence(r,worker_hash,worker_port=8004):
    f=r['feedback'];meta=f.get('backend_metadata',{})
    return bool(r['arguments'].get('planner')=='curobo' and f.get('payload_state')=='single_hold'
        and f.get('planner')=='official cuRobo V2 GPU IK/trajectory optimization'
        and f.get('native_path_revalidation') is True and f.get('fallback') is False
        and f.get('worker_port')==worker_port and meta.get('ready') is True
        and meta.get('worker_source_sha256')==worker_hash and bool(f.get('plans'))
        and f.get('primitive_success') is True and f.get('verification_source')=='robot_forward_kinematics'
        and f.get('position_error_m',1)<=.012 and f.get('rotation_error_rad',1)<=.08)


def route_evidence(receipts,directed,requested_arm):
    # Preserve all attempts in the record. A rejected stale proposal may be
    # retried, but must keep the same object, arm and original placement goal.
    side='right' if directed else requested_arm
    placed=[r for r in receipts if r['execution']['skill']=='place'
        and r['feedback'].get('primitive_success') is True]
    for r in receipts:
        if r['feedback'].get('primitive_success') is True:continue
        if r['feedback'].get('failure_code')!='planning_state_changed':return False
        if r['arguments'].get('object_id','bottle')!='bottle':return False
        if r['execution']['skill']=='place':
            if not placed or r['arguments'].get('arm')!=side:return False
            keys=('object_id','receptacle_id','arm','target_xy_m','offset_xy_m')
            if any(r['arguments'].get(k)!=placed[-1]['arguments'].get(k) for k in keys):return False
    receipts=[r for r in receipts if r['feedback'].get('primitive_success') is True]
    names=[r['execution']['skill'] for r in receipts]
    if not names or names[0]!='pick' or names[-1]!='place':return False
    middle=names[1:-1]
    if directed:
        if not middle or middle[0]!='handover':return False
        middle=middle[1:]
    if not middle or any(s not in ('move_relative','move_to_pose') for s in middle):return False
    pick=receipts[0]['feedback']
    moves=[r for r in receipts if r['execution']['skill'] in ('move_relative','move_to_pose')]
    return bool(pick.get('primitive_success') is True and pick.get('held_by')==('left' if directed else requested_arm)
        and pick.get('verification_source')=='native_two_finger_contact_and_lift'
        and all(r['arguments'].get('arm')==side and r['feedback'].get('object_id')=='bottle' for r in moves))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='api-forward')
    parser.add_argument('--case',choices=('forward','forward-left','forward-left-a','left-right','original-left-right','cross-relay-a','left-sideways'),default='forward')
    parser.add_argument('--provider',choices=('current','bottle','connected','fast','scene','intent','margin','free','task'),default='current')
    parser.add_argument('--scene',choices=('current','empty'),default='current')
    parser.add_argument('--gpu-cache',action='store_true',help='explicit independent8005exact-world cache candidate')
    parser.add_argument('--compact-context',action='store_true',help='model-only diagnostic projection; raw native receipts untouched')
    parser.add_argument('--stage-tools',action='store_true',help='receipt-derived stage routing; model still declares the plan')
    parser.add_argument('--live-context',action='store_true',help='bounded current root-task ledger at each model request')
    parser.add_argument('--focused-context',action='store_true',help='replace old text dialogue with the bound current-task ledger')
    parser.add_argument('--bind-atomic',action='store_true',help='bind declared gripper stages to the named model tool; never invent its arguments')
    parser.add_argument('--output',type=lambda x:(ROOT/x).resolve(),help='new immutable folder inside runs/');args=parser.parse_args()
    if not args.name.replace('-','').isalnum():raise ValueError('simple immutable run name required')
    out=args.output or ROOT/'runs/20261009-bottle'/args.name
    if not out.is_relative_to((ROOT/'runs').resolve()) or out==(ROOT/'runs').resolve():raise ValueError('new output folder inside runs/ required')
    out.mkdir(parents=True,exist_ok=False)
    setup()
    from harness.poses import PoseHarness,PoseWorkspace,install_isolated,service_factory
    from harness.llm import ModelSettings
    from harness.vault import SecretStore
    from harness.provider import install_isolated as provider
    install_isolated();factory=service_factory()
    harness_class=PoseHarness
    if args.provider in ('bottle','connected','fast','scene','intent','margin','free','task'):
        from harness.bottle import BottleHarness,install_isolated as bottle,service_factory as bottle_factory
        bottle();factory=bottle_factory();harness_class=BottleHarness
    if args.provider in ('connected','fast','scene','intent','margin','free','task'):
        from harness.connected_placement import install_isolated as connected
        factory=connected(factory)
    if args.provider in ('fast','scene','intent','margin','free','task'):
        from harness.fast_grasp import install_isolated as efficient
        factory=efficient(factory)
    if args.provider in ('margin','free','task'):
        from harness.margin_grasp import install_isolated as margin
        factory=margin(factory)
    if args.provider in ('free','task'):
        from harness.free_transport import install_isolated as free
        factory=free(factory)
    if args.provider in ('scene','intent','margin','free','task'):
        from sim.scene_curobo import install_isolated as scene_gpu
        scene_gpu()
    if args.gpu_cache:
        if args.provider not in ('scene','intent','margin','free','task'):raise ValueError('scene-aware provider required for GPU cache candidate')
        from sim.cache_curobo import install_isolated as cache_gpu
        cache_gpu()
    if args.provider in ('intent','margin','free','task'):
        from harness.intent import install_isolated as intent
        harness_class=intent(harness_class)
    if args.provider=='task':
        from harness.task_placement import install_isolated as task_placement
        factory=task_placement(factory)
    if args.stage_tools or args.live_context:
        if args.provider not in ('intent','margin','free','task'):raise ValueError('isolated Bottle/intent chain required for stage context')
        from harness.stages import install_isolated as stages
        if args.focused_context and not args.live_context:raise ValueError('bound live context required before pruning old dialogue')
        if args.bind_atomic and not args.live_context:raise ValueError('bound live context required before atomic tool choice')
        harness_class=intent(stages(reveal_by_stage=args.stage_tools,live_context=args.live_context,
            focused_context=args.focused_context,bind_atomic=args.bind_atomic))
    if args.compact_context:
        from harness.compact import install_isolated as compact
        harness_class=compact(harness_class)
    if args.scene=='empty':
        if args.provider not in ('fast','scene','intent','margin','free','task'):raise ValueError('empty optional-scene check requires the isolated fast provider')
        from sim.scene_library import LibraryEnv, service_factory as library
        import sim.service as native
        factory=library(factory)
        class EmptyEnv(LibraryEnv):
            def __init__(self, *, selection=None, **kwargs):
                super().__init__(selection=[] if selection is None else selection, **kwargs)
        native.OpenArmEnv=EmptyEnv
    requested=PROMPT;target=[.583,-.173]
    requested_arm='left' if args.case in ('forward-left','forward-left-a','left-sideways') else 'right'
    if args.case in ('forward-left','forward-left-a','left-sideways'):
        requested=PROMPT.replace('右手','左手');target=[.495,.185] if args.case=='forward-left-a' else [.52,.18]
    if args.case=='left-sideways':
        requested=('用左手抓起瓶子并连续保持实际夹持10秒，再使用cuRobo搬运，放到桌面world XY [0.435,0.115]米。'
            '其他物体不动。总预算120秒，计入模型与规划等待；不要重置、加时、换手或改变原目标，松手后复查。')
        target=[.435,.115]
    directed=args.case in ('left-right','original-left-right','cross-relay-a')
    if directed:
        requested=('用左手抓起瓶子交给右手，右手独立保持实际夹持10秒，再经cuRobo放到桌面world XY [0.56,-0.13]米，松手确认稳定。'
            '其他物体不动。本次总预算120秒，计入模型与规划等待；不要重置、加时或改变原目标。')
        target=[.56,-.13]
        if args.case=='cross-relay-a':
            target=[.57,-.09];requested=requested.replace('[0.56,-0.13]','[0.57,-0.09]')
    if args.case in ('left-right','forward-left','forward-left-a','cross-relay-a','left-sideways'):
        fixture_name='left-forward' if args.case=='forward-left' else 'left-right'
        base=factory
        class FixtureQueue:
            def __init__(self,queue,owner):self.queue,self.owner=queue,owner
            def put(self,item):return self.queue.put(item)
            def get(self,*a,**kw):
                while True:
                    command,values,future=self.queue.get(*a,**kw)
                    if command!='_initial_bottle_fixture':return command,values,future
                    if not future.set_running_or_notify_cancel():continue
                    try:
                        owner=self.owner
                        if owner.job or owner.decision_wait or owner.env.tick or owner.env.data.time:
                            raise ValueError('fixture is initialisation only; no active task may be reset')
                        owner.env.require_current_sources()
                        from tools.check_bottle_routes import CASES,initialize,load_cases
                        case_path=ROOT/'runs/20261009-generalize/cases.json'
                        if args.case in ('forward-left-a','cross-relay-a','left-sideways') and hashlib.sha256(case_path.read_bytes()).hexdigest()!=hashes['runs/20261009-generalize/cases.json']:
                            raise ValueError('declared initial fixture changed after source freeze')
                        selected='left-forward-a' if args.case=='forward-left-a' else 'cross-relay-a' if args.case=='cross-relay-a' else 'left-sideways' if args.case=='left-sideways' else None
                        cases=load_cases(case_path,selected)[0] if selected else CASES
                        fixture=cases[0] if selected else next(c for c in cases if c['name']==fixture_name)
                        initialize(owner.env,fixture);owner._publish()
                        future.set_result({'initialised':True,'at_time_zero':True})
                    except Exception as exc:future.set_exception(exc)
        class FixtureService(base):
            def _worker(self):
                self.commands=FixtureQueue(self.commands,self)
                return super()._worker()
        factory=FixtureService
    names=[str(p.relative_to(ROOT)) for d in ('sim','harness') for p in sorted((ROOT/d).glob('*.py'))]
    names+=['tools/check_bottle_api.py','tools/check_bottle_routes.py','tools/check_task_api.py','tools/check_task_queue.py','tools/object_gpu.py','tools/scene_gpu.py','server.py']
    if args.gpu_cache:names+=['tools/scene_cache.py']
    if args.case in ('forward-left-a','cross-relay-a','left-sideways'):names.append('runs/20261009-generalize/cases.json')
    hashes={}
    for name in names:
        value=(ROOT/name).read_bytes();source_target=out/'source'/name;source_target.parent.mkdir(parents=True,exist_ok=True)
        source_target.write_bytes(value);hashes[name]=hashlib.sha256(value).hexdigest()
    settings=ModelSettings(store=SecretStore())
    if not settings.public()['configured']:raise ValueError('saved API unavailable')
    dump(out/'protocol.json',{'prompt':requested,'case':args.case,'provider':args.provider,'scene':args.scene,'source_sha256':hashes,'time_budget_s':120,
        'model_context_projection':args.compact_context,'raw_native_evidence_and_original_motion_guards_unchanged':True,
        'reveal_tools_by_stage':args.stage_tools,'live_task_context':args.live_context,
        'focused_context':args.focused_context,
        'bind_model_declared_atomic_stage':args.bind_atomic,
        'gpu_worker_port':8005 if args.gpu_cache else 8004,
        'timing_mode':'measured_delay','original_bottle_xy_m':[.415,.185] if args.case=='forward-left-a' else [.525,-.2] if args.case=='cross-relay-a' else [.435,.185] if args.case=='left-sideways' else [.44,.18] if args.case in ('left-right','forward-left') else [.503,-.173],
        'original_target_xy_m':target,'no_prescribed_tool_sequence_in_prompt':True,
        'world_target_explicit_in_prompt':directed or args.case=='left-sideways',
        'fixture':'declared synthetic initialisation before task; not a measured real scene',
        'accepted_transport_tools':['move_to_pose','move_relative'],
        'failed_motion_retry_policy':'only native planning_state_changed; raw rejection retained; placement retries keep the same original arguments; rejected attempt never counts as completed motion',
        'no_task_reset_goal_substitution_or_physics_change':True,'hardware_access':False,'transfer_ready':False})
    service=factory();workspace=PoseWorkspace();harness=None;root=None;error=None;started=time.monotonic()
    try:
        if args.case in ('left-right','forward-left','forward-left-a','cross-relay-a','left-sideways'):service.call('_initial_bottle_fixture')
        initial=service.snapshot();manifest=service.call('manifest')
        for camera in ('operator','head','left','right'):(out/(camera+'-start.jpg')).write_bytes(service.frame(camera))
        harness=harness_class(service,workspace=workspace,settings=settings);provider(harness)
        root=harness.chat(requested)['turn_id'];last=None
        while True:
            state=service.snapshot();summary=workspace.summary(state)
            public=workspace.public('execute',state['observation_id'].split(':')[0],state['memory_context'])
            turns=[t for t in public['turns'] if t['id']==root or t.get('parent')==root]
            key=(summary['revision'],state['observation_id'])
            if key!=last:
                dump(out/'progress.json',{'root':root,'terminal':state,'agent':summary,'turns':turns});last=key
            if (not state.get('job') and not summary['busy'] and not harness.pending_review
                and not harness.pending_operations and turns and all(t['status'] not in ('running','waiting') for t in turns)):break
            if time.monotonic()-started>600:raise TimeoutError('original owner exceeded600s; retain partial evidence')
            time.sleep(.5)
    except Exception as exc:error={'code':type(exc).__name__,'message':str(exc)}
    finally:
        if root:
            record=collect(service,workspace,root);state=record['terminal'];plan=state['agent'].get('plan') or {}
            dump(out/'record.json',record)
            receipts=record['receipts'];motions=[r['execution']['skill'] for r in receipts]
            goal=plan.get('placement_goal') or {};xy=goal.get('target_xy_m',[])
            pick=next((r['feedback'] for r in receipts if r['execution']['skill']=='pick'),{})
            place=next((r['feedback'] for r in reversed(receipts) if r['execution']['skill']=='place'),{})
            events=[e for t in record['turns'] for e in t['trace'] if e['kind']=='tool']
            terminal_manifest=service.call('manifest')
            checks={'actual_model_API':record['actual_api_calls']>0,'model_created_own_plan':bool(plan.get('steps')),
                'exact_original_goal':goal.get('object_id')=='bottle' and goal.get('surface')=='table'
                    and len(xy)==2 and max(abs(x-y) for x,y in zip(xy,target))<1e-8,
                'native_requested_skills':route_evidence(receipts,directed,requested_arm),
                'actual_GPU':gpu_evidence(record,hashes['tools/object_gpu.py'],8005 if args.gpu_cache else 8004),
                'original_release_support_stability':place.get('primitive_success') is True
                    and place.get('verification_source')=='native_support_release_and_stability'
                    and place.get('placement',{}).get('confirmed') is True,
                'original_budget_and_timing':plan.get('time_budget_s')==120 and plan.get('timing_mode')=='measured_delay'
                    and state['observation']['sim_time']<=120 and not state['diagnostics']['evaluation']['invalid'],
                'unselected_objects_undisturbed':state['diagnostics']['evaluation']['non_target_object_guard']['passed'],
                'no_reset_or_plan_replacement':not any(e['name']=='prepare_simulation' for e in events)
                    and sum(e['name']=='set_plan' and e['status']=='completed' for e in events)==1,
                'same_compiled_physics':manifest['model_evidence']==terminal_manifest['model_evidence'],
                'all_native_stages_completed':all(s['status']=='done' for s in plan.get('steps',[])),
                'formal_review':plan.get('verdict') in ('success','failure','uncertain')}
            if directed:
                relay=next((r['feedback'] for r in receipts if r['execution']['skill']=='handover'),{})
                hold=relay.get('hold_audit',{})
                checks['native_left_right_relay']=relay.get('primitive_success') is True and relay.get('from_arm')=='left'
                checks['native_left_right_relay'] &= relay.get('to_arm')==relay.get('held_by')=='right' and relay.get('execution_binding',{}).get('valid') is True
                checks['native_independent_receiver_hold']=relay.get('verification_source')=='native_receiver_contact_then_donor_release'
                checks['actual10s_receiver_hold']=hold.get('requested_seconds')==10 and hold.get('actual_seconds',0)>=9.998
                checks['actual10s_receiver_hold'] &= hold.get('continuous_full_windows_stable') is True
            if args.case=='left-sideways':
                hold=pick.get('hold_audit',{})
                checks['actual10s_independent_pick_hold']=(hold.get('requested_seconds')==10
                    and hold.get('actual_seconds',0)>=9.998 and hold.get('continuous_full_windows_stable') is True
                    and hold.get('audit_completed') is True and hold.get('full_window_checks',0)>0)
                checks['actual_task_pose_and_native_control_provider']=(pick.get('task_pose_fallback',{}).get('eligible_original_candidates',0)>0
                    and place.get('preview_servo_contract',{}).get('maximum_actuator_increment_rad')==.008
                    and bool(place.get('private_native_servos')))
            if args.gpu_cache:
                moves=[r['feedback'] for r in receipts if r['execution']['skill'] in ('move_to_pose','move_relative')
                    and r['feedback'].get('primitive_success') is True]
                checks['exact_world_cache_provenance']=bool(moves) and all(
                    f.get('backend_metadata',{}).get('scene_cache_worker_source_sha256')==hashes['tools/scene_cache.py']
                    and f.get('backend_metadata',{}).get('exact_world_cache_source_sha256')==hashes['sim/scene_cache.py']
                    and f.get('backend_metadata',{}).get('worker_port')==8005
                    and all(p.get('cache',{}).get('exact_world_update',{}).get('same_gpu_buffers') is True
                        and p.get('cache',{}).get('exact_world_update',{}).get('same_as_fresh_official_scene_storage') is True
                        for p in f.get('plans',[])) for f in moves)
            result={**record,'passed':all(checks.values()),'checks':checks,'error':error,
                'initial_manifest':manifest,'terminal_manifest':terminal_manifest,'wall_seconds':time.monotonic()-started}
        else:result={'passed':False,'error':error,'root_created':False}
        for camera in ('operator','head','left','right'):(out/(camera+'-final.jpg')).write_bytes(service.frame(camera))
        dump(out/'result.json',result)
        print({'passed':result['passed'],'error':error,'calls':result.get('actual_api_calls'),
            'checks':result.get('checks')},flush=True)
        if harness:harness.close()
        service.close();workspace.db.close()


if __name__=='__main__':main()
