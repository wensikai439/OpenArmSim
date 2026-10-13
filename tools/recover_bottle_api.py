"""Read-only reassessment of completed actions after collector scoring failed.

No simulation, API calls, fabricated SQL receipts or raw archive rewrites.
Native execution files and the final persisted model trace are authoritative.
"""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261009-bottle/api-left-right'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',default='recovery');options=parser.parse_args()
    if not options.revision.replace('-','').isalnum():raise ValueError('simple recovery revision required')
    archive=OUT/options.revision;archive.mkdir(exist_ok=False)
    progress=read(OUT/'progress.json');protocol=read(OUT/'protocol.json')
    plan=progress['agent']['plan'];root=plan['turn_id']
    terminal={**progress['terminal'],'agent':progress['agent']}
    assert not terminal.get('job') and not progress['agent']['busy']
    assert all(t['status']=='completed' for t in progress['turns'])
    shutil.copy2(OUT/'progress.json',archive/'progress.json')
    records={};raw_hashes={}
    for path in (ROOT/'runs').glob('20261009-*/manifest.json'):
        manifest=read(path);details=manifest.get('details',{})
        if details.get('initiator',{}).get('plan_turn_id')!=root:continue
        ident=details.get('execution_id')
        if not ident:continue
        summary_path=path.parent/'summary.json';summary=read(summary_path)
        assert ident not in records
        records[ident]=(path,manifest,summary_path,summary)
    events=[e for t in progress['turns'] for e in t['trace'] if e['kind']=='tool']
    steps=[]
    for execution in plan['execution_tools']:
        ident=execution['id'];skill=execution['skill'];path,manifest,summary_path,summary=records[ident]
        matching=[e for e in events if e['name']==skill and e['status']=='completed'
            and e.get('detail',{}).get('result',{}).get('execution_id')==ident]
        assert len(matching)==1
        detail=matching[0]['detail'];args=detail.get('dispatch_arguments',detail['arguments'])
        feedback=summary['metrics']['tamp'];parameters=manifest['details']['tamp']['parameters']
        for key in ('arm','from_arm','to_arm','object_id','hold_seconds','planner'):
            if key in args:assert parameters.get(key)==args[key],key
        assert manifest['details']['initiator']['plan_turn_id']==root
        assert summary['status']=='primitive_completed' and feedback.get('primitive_success') is True
        for original,kind in ((path,'manifest'),(summary_path,'summary')):
            target=archive/(skill+'-'+kind+'.json');shutil.copy2(original,target)
            raw_hashes[str(original.relative_to(ROOT))]=sha(original)
        steps.append({'task':skill,'parameters':args,'feedback':feedback,
            'execution_id':ident,'native_run_id':path.parent.name,
            'binding_source':'independent match of persisted model dispatch, native execution ID, initiator and native parameters',
            'manifest':manifest,'summary':summary})
    assert [s['task'] for s in steps]==['pick','handover','move_to_pose','place']
    pick,relay,gpu,place=[s['feedback'] for s in steps]
    hold=relay.get('hold_audit',{});meta=gpu.get('backend_metadata',{})
    original_goal=plan['placement_goal']
    # The same shadowed variable also corrupted the protocol's target field.
    # Read the PRE-EXISTING frozen fixture declaration, without executing it,
    # and corroborate it with the original explicit natural-language goal.
    tree=ast.parse((OUT/'source/tools/check_bottle_routes.py').read_text())
    assignment=next(node for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CASES' for t in node.targets))
    case=next(c for c in ast.literal_eval(assignment.value) if c['name']=='left-right')
    expected=case['target_xy_m'];assert expected==[.56,-.13]
    assert '[0.56,-0.13]' in protocol['prompt']
    worker_sha=protocol['source_sha256']['tools/object_gpu.py']
    assert sha(OUT/'source/tools/object_gpu.py')==worker_sha
    initial=steps[0]['manifest'];evidence=[s['manifest']['model_evidence'] for s in steps]
    checks={
        'actual_model_API':sum(e['kind']=='api' for t in progress['turns'] for e in t['trace'])>0,
        'exact_original_goal':original_goal['object_id']=='bottle' and original_goal['surface']=='table'
            and original_goal['target_xy_m']==expected and original_goal['anchor']['object_xy_m']==[.44,.18],
        'native_dispatch_bindings':len(steps)==4 and all(s['feedback'].get('object_id')=='bottle' for s in steps),
        'native_left_pick':pick.get('held_by')=='left' and pick.get('verification_source')=='native_two_finger_contact_and_lift',
        'native_left_right_relay':relay.get('from_arm')=='left' and relay.get('to_arm')==relay.get('held_by')=='right'
            and relay.get('verification_source')=='native_receiver_contact_then_donor_release' and relay.get('independent_hold_seconds',0)>=.999,
        'actual10s_receiver_hold':hold.get('requested_seconds')==10 and hold.get('actual_seconds',0)>=9.998
            and hold.get('continuous_full_windows_stable') is True,
        'actual_GPU':gpu.get('planner')=='official cuRobo V2 GPU IK/trajectory optimization'
            and gpu.get('native_path_revalidation') is True and gpu.get('fallback') is False
            and gpu.get('worker_port')==8004 and bool(gpu.get('plans')) and meta.get('ready') is True
            and meta.get('worker_source_sha256')==worker_sha and gpu.get('verification_source')=='robot_forward_kinematics'
            and gpu.get('position_error_m',1)<=.012 and gpu.get('rotation_error_rad',1)<=.08,
        'original_release_support_stability':place.get('verification_source')=='native_support_release_and_stability'
            and place.get('placement',{}).get('confirmed') is True and place.get('position_error_m',1)<=.01,
        'original120s_measured_clock':plan['time_budget_s']==120 and plan['timing_mode']=='measured_delay'
            and terminal['observation']['sim_time']<=120 and not terminal['diagnostics']['evaluation']['invalid'],
        'unselected_objects_undisturbed':all(s['summary']['evaluation']['non_target_object_guard']['passed'] for s in steps),
        'no_reset_or_plan_replacement':not any(e['name']=='prepare_simulation' for e in events)
            and sum(e['name']=='set_plan' and e['status']=='completed' for e in events)==1,
        'same_compiled_physics':all(e==evidence[0] for e in evidence),
        'all_stages_completed':all(s['status']=='done' for s in plan['steps']),
        'formal_review':plan['verdict'] in ('success','failure','uncertain'),
    }
    frames=terminal['keyframes'][-1]
    sources={'overview':frames['url'],**frames['sensor_urls']}
    for camera,url in sources.items():
        source=ROOT/url.lstrip('/');target=OUT/(camera+'-final.jpg')
        shutil.copy2(source,target);raw_hashes[str(source.relative_to(ROOT))]=sha(source)
    result={'recovered_after_collector_failure':True,'passed':all(checks.values()),'checks':checks,
        'actual_api_calls':sum(e['kind']=='api' for t in progress['turns'] for e in t['trace']),
        'terminal':terminal,'turns':progress['turns'],'steps':steps,'initial_manifest':initial,
        'raw_source_sha256':raw_hashes,'source_progress_sha256':sha(OUT/'progress.json'),
        'image_sources':sources,'hardware_access':False,'transfer_ready':False,
        'collector_error':'TypeError: target XY list was shadowed by a source Path during freezing; scoring failed after all actions and formal review completed.',
        'recovery_scope':'Only persisted model/native artifacts independently reassessed. In-memory SQL receipts were not recovered or fabricated. No new API calls or robot actions; original task and gates unchanged.'}
    result['goal_check_source']='Frozen pre-existing left-right CASES plus original explicit prompt; corrupted protocol target Path is preserved unchanged.'
    save(archive/'assessment.json',result)
    if options.revision=='recovery':save(OUT/'assessment.json',result)
    save(archive/'source.json',{'raw_source_sha256':raw_hashes,'progress_sha256':sha(OUT/'progress.json'),
        'protocol_sha256':sha(OUT/'protocol.json'),'no_raw_archive_changes':True})
    print({'passed':result['passed'],'calls':result['actual_api_calls'],'checks':checks,
        'position_error_mm':place['position_error_m']*1000,'sim_time':terminal['observation']['sim_time']})


if __name__=='__main__':main()
