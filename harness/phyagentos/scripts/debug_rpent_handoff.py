"""Project-local route debugging with unchanged RPent/PiperAirHandoff primitives."""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
RP=Path(str(Path(os.environ.get("RPENT_ROOT", str(Path(__file__).resolve().parents[1] / "external/rpent"))) / 'src/RPent'))
SIM=Path(str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_robocasa")))))
sys.path[:0]=[str(RP),str(SIM/'src')]
import numpy as np
from scipy.spatial.transform import Rotation
from piper_robocasa.air_handoff_task import PiperAirHandoffTask
from piper_robocasa.pose_preflight import check_pose_path, interpolate_rotation
ROOT=Path(str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser()
p.add_argument('label')
p.add_argument('--stop-after', choices=['transport','complete'], default='complete')
a=p.parse_args()
if not a.label or not all(c.isalnum() or c in '-_' for c in a.label):raise ValueError('unique alphanumeric label required')
out=ROOT/'runs/ik-handoff-20261004'/a.label
out.mkdir(parents=True,exist_ok=False)
records=[];snapshots=[];probes=[];task=None;error=None;completion=False

def save(name,value):
    (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def source_hashes():
    paths=[RP/'robots/piper_sim/env_server.py',SIM/'src/piper_robocasa/air_handoff_task.py',SIM/'src/piper_robocasa/photo_relay_task.py',SIM/'src/piper_robocasa/pose_preflight.py',SIM/'scenes/piper_air_handoff_v1/scene_spec.json',RP.parents[1]/'reports/horizontal_grasp_20261001/physical_validation_01/plan.json',Path(__file__)]
    return [{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]

def snapshot(label):
    frames=task.capture_frames(('cam_right','overview'))
    for f in frames:
        f['absolute_path']=str(task.output/f['path'])
        assert hashlib.sha256(Path(f['absolute_path']).read_bytes()).hexdigest()==f['sha256']
    state=task.observe()
    snapshots.append({'label':label,'frames':frames,'state':state})
    save('snapshots.json',snapshots)
    return state

def probe(side,target,rotation,label):
    before=task.data.qpos.copy();steps=task.ws.step_count
    result=check_pose_path(task.ws,side,target,rotation)
    assert np.array_equal(before,task.data.qpos) and steps==task.ws.step_count
    probes.append({'label':label,'side':side,'target_world_m':list(target),'target_rotation_rad':list(rotation) if rotation is not None else None,'verification':result,'read_only_confirmed':True})
    save('preflight-records.json',probes)
    print(json.dumps({'probe':label,'accepted':result['accepted'],'reason':result.get('reason'),'position_error_m':result.get('position_error_m'),'rotation_error_rad':result.get('rotation_error_rad')},ensure_ascii=False),flush=True)
    return result

def act(name,payload,label):
    action={'id':name,'payload':payload}
    preflight=None
    if name=='move_arm':
        task._validate_composable_action(action)
        preflight=probe(payload['side'],payload['target_world_m'],payload.get('target_rotation_rad'),label)
        if not preflight['accepted']:raise RuntimeError('No motion dispatched: '+label+': '+preflight['reason'])
    record={'sequence':len(records)+1,'label':label,'action':action,'preflight':preflight,'requested_at':time.time(),'result':None}
    records.append(record);save('action-records.json',records)
    result=task.step(action)
    record.update(result=result,completed_at=time.time());save('action-records.json',records)
    state=snapshot(label)
    print(json.dumps({'action':len(records),'label':label,'outcome':result['outcome'],'control_steps':result.get('control_steps'),'bottle':state['bottle_world_m'],'assist':state['assist_active']},ensure_ascii=False),flush=True)
    if result.get('primitive_completed') is not True:raise RuntimeError('Native action failed; no replay: '+str(result.get('error')))
    return state

def move(side,target,rotation,label):
    start=task.tcp(side);initial=Rotation.from_matrix(task.data.site_xmat[task.ws.eef_ids[side]].reshape(3,3)).as_rotvec()
    final=initial if rotation is None else np.asarray(rotation)
    angle=(Rotation.from_rotvec(final)*Rotation.from_rotvec(initial).inv()).magnitude()
    count=max(1,math.ceil(float(np.linalg.norm(np.asarray(target)-start))/.15),math.ceil(angle/.6))
    for k in range(1,count+1):
        fraction=k/count
        act('move_arm',{'side':side,'target_world_m':(start+(np.asarray(target)-start)*fraction).tolist(),'target_rotation_rad':interpolate_rotation(initial,final,fraction).tolist()},label+'-'+str(k))

def carry(side,root,label,yaw_delta_deg=0):
    state=task.observe();offset=np.asarray(state['tcp_world_m'][side])-state['bottle_world_m']
    yaw=Rotation.from_euler('Z',yaw_delta_deg,degrees=True)
    rotation=(yaw*Rotation.from_rotvec(state['tcp_rotation_rad'][side])).as_rotvec().tolist()
    move(side,(np.asarray(root)+yaw.apply(offset)).tolist(),rotation,label)

save('source-manifest-before.json',source_hashes())
try:
    task=PiperAirHandoffTask(out/'native',render=True,run_id='paos-ik-'+a.label)
    initial=snapshot('initial')
    reference=RP.parents[1]/'reports/horizontal_grasp_20261001/physical_validation_01/plan.json'
    plan=json.loads(reference.read_text());save('reference-plan.json',{'path':str(reference),'sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'plan':plan})
    assert np.linalg.norm(np.asarray(initial['bottle_world_m'])-[.38,-.42,.72])<1e-5
    for label,item in plan:act(item['id'],{k:v for k,v in item.items() if k!='id'},label)
    state=task.observe()
    assert state['assist_active']=={'right':True,'left':False} and not state['bottle_worktable_contact']
    rb=np.asarray(state['bottle_rotation_matrix']);rt=Rotation.from_rotvec(state['tcp_rotation_rad']['right']).as_matrix()
    offset=rb.T@(np.asarray(state['tcp_world_m']['right'])-state['bottle_world_m'])
    root_target=np.array([.32,-.025,.84])
    chosen=None
    for yaw in [0,15,30,45,60,75,90,-15,-30,-45]:
        rz=Rotation.from_euler('Z',yaw,degrees=True).as_matrix()
        target=root_target+rz@rb@offset
        rotation=Rotation.from_matrix(rz@rt).as_rotvec()
        result=probe('right',target.tolist(),rotation.tolist(),'transport-candidate-yaw-'+str(yaw))
        if result['accepted']:
            chosen={'yaw_deg':yaw,'root_target_world_m':root_target.tolist(),'target_world_m':target.tolist(),'target_rotation_rad':rotation.tolist()};break
    if chosen is None:raise RuntimeError('No current native IK-verified transport candidate in the bounded yaw set')
    save('chosen-transport.json',chosen)
    move('right',chosen['target_world_m'],chosen['target_rotation_rad'],'right-transport')
    state=snapshot('transport-completed')
    assert state['assist_active']['right'] and not state['bottle_worktable_contact']
    if a.stop_after=='complete':
        act('set_gripper',{'side':'left','open_or_close':'open'},'left-open')
        move('left',[.24,.18,.84],None,'left-retreat')
        transition=Rotation.from_euler('ZY',[-17.5,140],degrees=True).as_rotvec().tolist()
        flat=Rotation.from_euler('ZY',[-17.5,90],degrees=True).as_rotvec().tolist()
        lr=Rotation.from_euler('ZY',[-45,90],degrees=True).as_rotvec().tolist()
        move('left',[.26,.18,.97],transition,'left-orientation-staging')
        move('left',[.43,.18,.97],transition,'left-travel-tilted')
        move('left',[.43,.18,.97],flat,'left-flatten-staging')
        bottle=task.bottle();target=(bottle+np.array([0,-.01,.08])).tolist()
        move('left',[target[0],.12,target[2]],lr,'left-pregrasp')
        move('left',target,lr,'left-contact-approach')
        act('grasp_bottle',{'side':'left'},'left-grasp')
        assert task.receiver_ready=='left'
        act('set_gripper',{'side':'right','open_or_close':'open'},'donor-open-transfer')
        state=snapshot('handoff-completed')
        assert state['assist_active']=={'right':False,'left':True}
        assert state['composable_transfers'][-1]['qpos_change']==0
        assert not state['composable_transfers'][-1]['table_contact']
        rr=state['tcp_rotation_rad']['right']
        move('right',[.28,-.20,1.04],rr,'donor-retreat-shared')
        carry('left',[.43,.18,.90],'receiver-leave-shared',27.5)
        carry('left',[.43,.18,.96],'receiver-lift-for-bucket-clearance')
        carry('left',[.34,.42,.96],'receiver-over-bucket',30)
        carry('left',[.34,.42,.87],'receiver-lower-bucket')
        act('release_bottle',{'side':'left'},'receiver-release-physical-settling')
        final=snapshot('final')
        completion=bool(final['composable_transfers'] and final['bucket_bottom_contact'] and final['bucket_xy_error_m']<.055 and final['bottle_speed_norm']<.05 and not any(final['assist_active'].values()) and not final['failed'] and final['collision_stop'] is None)
        if not completion:raise RuntimeError('Physical handoff/bucket settling criteria not satisfied')
        save('validated-route.json',{'source':'actually completed native composable actions, same factory initial layout','actions':[{'label':r['label'],**r['action']} for r in records],'initial_bottle_world_m':initial['bottle_world_m'],'physical_complete':completion})
except BaseException as exc:
    error={'type':type(exc).__name__,'message':str(exc)}
    print(json.dumps({'error':error},ensure_ascii=False),flush=True)
finally:
    if task is not None:
        final=task.observe();save('summary.json',{'label':a.label,'physical_complete':completion,'requested_stop_after':a.stop_after,'error':error,'action_count':len(records),'llm_calls':0,'hardware_commands':0,'state':final})
        task.close()
    after=source_hashes();save('source-manifest-after.json',after)
    before=json.loads((out/'source-manifest-before.json').read_text())
    save('source-integrity.json',{'unchanged':before==after,'files':after})
if error:sys.exit(1)
