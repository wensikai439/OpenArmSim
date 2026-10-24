"""Local connection preparation. No SSH, camera, CAN or actuator interface."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ORDER=[n for side in ('left','right') for n in
       [*[f'{side}_joint{i}' for i in range(1,8)],f'{side}_gripper']]


def read(path):return json.loads(Path(path).read_text())


def finite_vector(value,size):
    return (isinstance(value,list) and len(value)==size
        and all(type(v) in (int,float) for v in value) and np.isfinite(value).all())


def validate(config,observation,*,now=None,fixture=False):
    """Missing or stale evidence blocks even shadow planning; no imagined pose."""
    gaps=[];now=time.time() if now is None else now
    if config.get('schema')!='openarm-connection/1':gaps.append('unknown connection schema')
    if config.get('hardware_enabled') is not False:gaps.append('this package has no hardware execution; hardware_enabled must be false')
    if observation.get('schema')!='openarm-observation/1':gaps.append('unknown observation schema')
    expected='synthetic_fixture' if fixture else 'measured_perception'
    if observation.get('source')!=expected:gaps.append('expected '+expected+'; simulator truth cannot impersonate a measured camera pose')
    if observation.get('frame')!='robot_base':gaps.append('observation frame must be robot_base')
    timestamp=observation.get('timestamp_unix_s')
    if type(timestamp) not in (int,float) or not np.isfinite(timestamp):gaps.append('finite timestamp required')
    elif not fixture and not -.001<=now-timestamp<=config['limits']['max_age_s']:gaps.append('observation stale or in future')
    if observation.get('state_order')!=ORDER or not finite_vector(observation.get('state_rad'),16):
        gaps.append('state must be finite left7,left_gripper,right7,right_gripper')
    for ident in ('bottle','basket'):
        pose=observation.get('objects',{}).get(ident,{}) or {}
        if pose.get('pose_reference')!='model_origin':gaps.append(ident+' model-origin transform missing')
        if not finite_vector(pose.get('position_m'),3):gaps.append(ident+' measured position missing')
        quat=pose.get('quaternion_xyzw')
        if not finite_vector(quat,4) or abs(np.linalg.norm(quat)-1)>.001:gaps.append(ident+' unit quaternion missing')
        sigma=pose.get('position_std_m')
        if type(sigma) not in (int,float) or not np.isfinite(sigma) or not 0<=sigma<=config['limits']['max_position_std_m']:
            gaps.append(ident+' pose uncertainty missing or excessive')
    for side in ('left','right'):
        hold=observation.get('hold_evidence',{}).get(side,{})
        if hold.get('status')!='empty' or not hold.get('source'):
            gaps.append(side+' fresh empty-hand evidence required for this entry skill')
    transform=config.get('T_world_from_base')
    try:
        m=np.asarray(transform,float)
        if (m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1])
            or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-6)
            or abs(np.linalg.det(m[:3,:3])-1)>1e-6):raise ValueError()
    except (ValueError,TypeError):gaps.append('invalid base-to-world rigid transform')
    return gaps


def preflight(config):
    manifest=read(ROOT/'skills/bottle_recovery/manifest.json')
    program=ROOT/'skills/bottle_recovery/program.py'
    gaps=[]
    if config.get('hardware_enabled') is not False:gaps.append('hardware_enabled must remain false')
    if hashlib.sha256(program.read_bytes()).hexdigest()!=manifest['program_sha256']:gaps.append('skill program hash mismatch')
    if config.get('perception_service') is None:gaps.append('bottle/basket pose and hold-evidence service not connected')
    for name in ('kinematics','cameras','controller','grippers','paired_trials'):
        if config.get('evidence',{}).get(name) is None:gaps.append(name+' independent evidence not supplied')
    return {'package_ready':not any('hash mismatch' in s for s in gaps),'hardware_access':False,
        'hardware_execution_available':False,'transfer_ready':False,'missing':gaps,
        'capabilities':{'recorded_state_replay':True,'snapshot_shadow_planning':True,
            'live_perception':False,'robot_motion':False},
        'controller_note':'reference/source/inputs/controller.py constructor enables motors; this entry never imports it'}


def replay(episode):
    data=read(ROOT/'reference/real_trajectory.json')['episodes'][str(episode)]
    clock=read(ROOT/'reference/acquisition.json')['episodes'][str(episode)]['recorded_seconds']
    frames=[]
    for index in np.unique(np.linspace(0,data['frames']-1,6,dtype=int)):
        state=[v for side in ('left','right') for v in data[side+'_joints'][index]+[data[side+'_gripper'][index]]]
        frames.append({'frame':int(index),'recorded_seconds':clock[index],'state_rad':state})
    return {'source':'previously exported real encoder recording, local files only',
        'episode':episode,'state_order':ORDER,'frames':data['frames'],'samples':frames,
        'pose_and_contact_evidence_present':False,'hardware_access':False,
        'scope':'state/clock contract replay; does not execute skill or certify calibration'}


def fixture_observation(config,world):
    from scipy.spatial.transform import Rotation
    m=np.asarray(config['T_world_from_base']);inv=np.linalg.inv(m)
    objects={}
    for ident,body in (('bottle',world.env.bottle_id),('basket',world.env.box_id)):
        objects[ident]={'position_m':(inv@np.r_[world.env.data.xpos[body],1])[:3].tolist(),
            'quaternion_xyzw':Rotation.from_matrix(inv[:3,:3]@world.env.data.xmat[body].reshape(3,3)).as_quat().tolist(),
            'position_std_m':0.,'pose_reference':'model_origin'}
    return {'schema':'openarm-observation/1','source':'synthetic_fixture','frame':'robot_base',
        'timestamp_unix_s':time.time(),'state_order':ORDER,'state_rad':world.env.state().tolist(),
        'objects':objects,'hold_evidence':{s:{'status':'empty','source':'synthetic_fixture'} for s in ('left','right')}}


def shadow(config,observation,output,*,fixture=False):
    from .aspire.runtime import dump, ProgramExecutor, ToolRegistry, PublicTrace
    gaps=validate(config,observation,fixture=fixture)
    if gaps:return {'status':'blocked','missing':gaps,'hardware_access':False,'motion_commands':0}
    from .aspire.precision import PrecisionWorld
    from .aspire.recovery import SegmentPlanner
    import mujoco
    from scipy.spatial.transform import Rotation
    from sim.alignment import recorded_pose
    class ShadowWorld(PrecisionWorld):
        def capture(self,label):pass
    m=np.asarray(config['T_world_from_base']);poses={}
    for ident,p in observation['objects'].items():
        poses[ident]=(m@np.r_[p['position_m'],1])[:3],m[:3,:3]@Rotation.from_quat(p['quaternion_xyzw']).as_matrix()
    xy=poses['bottle'][0][:2].tolist()
    world=ShadowWorld({'seed':0,'bottle_xy_m':xy,'fault':'none','motion_profile':'precision'},output)
    planned=[]
    try:
        # A diagnostic planning MODEL is initialized once from the snapshot.
        # No physics step, live world mutation or device command is performed.
        recorded_pose(world.env,observation['state_rad'])
        for ident in ('bottle','basket'):
            pos,rot=poses[ident];quat=Rotation.from_matrix(rot).as_quat()
            if ident=='bottle':
                adr=world.env.registry['bottle']['qposadr'];world.env.data.qpos[adr:adr+7]=np.r_[pos,quat[3],quat[:3]]
            else:
                world.env.model.body_pos[world.env.box_id]=pos
                world.env.model.body_quat[world.env.box_id]=np.r_[quat[3],quat[:3]]
        mujoco.mj_forward(world.env.model,world.env.data)
        # Basket parameters belong to the immutable initial planning instance.
        from sim.model_evidence import ParameterAudit,compiled_evidence
        world.env._parameter_audit=ParameterAudit(world.env.model)
        world.env._model_evidence['compiled']=compiled_evidence(world.env.model)
        world.env.evaluation=world.env._evaluate(0.);world.env._protect()
        def observe():
            o=world._observation();o.update(source=observation['source'],held_by=[],touching={'left':False,'right':False},
                gripper_motor_feedback_rad={'left':observation['state_rad'][7],'right':observation['state_rad'][15]})
            return o
        def propose_grasps(exclude_ids=()):return world.propose_grasps(exclude_ids)
        def pregrasp(grasp_id):
            g=world._grasp(grasp_id);p=SegmentPlanner(world.env,mode='pregrasp',grasp=g)
            p._sync();p.rotation=np.asarray(g['rotation']);goal=p._ik(np.asarray(g['pregrasp_m']))
            path=p._path(goal)
            planned.append({'tool':'pregrasp','grasp_id':grasp_id,'joint_path_rad':[q.tolist() for q in path],
                'path_is_execution_command':False,'planning_model_calibrated':False})
            return {'ok':False,'done':True,'error_code':'shadow_execution_disabled'}
        def disabled(*args,**kwargs):return {'ok':False,'done':True,'error_code':'shadow_execution_disabled'}
        registry=ToolRegistry(shared_public_names=world.PUBLIC,public_extension_allowlist=())
        for name in world.PUBLIC:
            registry.register({'observe':observe,'propose_grasps':propose_grasps,'pregrasp':pregrasp}.get(name,disabled),
                name=name,public=True,layer='shared_high_level',capability='snapshot planning only, no motion')
        result=ProgramExecutor(registry,recorder=PublicTrace(output/'shadow-trace.jsonl')).execute_program(
            (ROOT/'skills/bottle_recovery/program.py').read_text())
        data={'status':'planned_execution_disabled' if planned else 'planning_failed',
            'program_ok':result.ok,'program_stop':str(result.error) if result.error else None,
            'source':observation['source'],'plans':planned,'hardware_access':False,'motion_commands':0,
            'native_physics_steps':world.env.tick,'transfer_ready':False}
        dump(output/'observation.json',observation);dump(output/'shadow.json',data);return data
    finally:world.close()


def main(argv=None):
    p=argparse.ArgumentParser(description='OpenArm 本地接入准备；没有真机运动命令')
    p.add_argument('command',choices=('preflight','replay','shadow','demo'))
    p.add_argument('--config',type=Path,default=ROOT/'deploy/config.json')
    p.add_argument('--observation',type=Path,default=ROOT/'deploy/observation.example.json')
    p.add_argument('--episode',type=int,choices=(0,1,2),default=0)
    a=p.parse_args(argv);config=read(a.config)
    if a.command=='preflight':result=preflight(config)
    elif a.command=='replay':result=replay(a.episode)
    else:
        output=ROOT/'runs'/(time.strftime('%Y%m%d-%H%M%S')+'-shadow')
        if a.command=='demo':
            from .aspire.precision import PrecisionWorld
            # One backend per process: use a plain isolated model for the fixture.
            from sim.scene_library import LibraryEnv
            from types import SimpleNamespace
            env=LibraryEnv(images=False,selection=[])
            try:env.reset(seed=0);observation=fixture_observation(config,SimpleNamespace(env=env))
            finally:env.close()
        else:observation=read(a.observation)
        result=shadow(config,observation,output,fixture=a.command=='demo')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 2 if result.get('status') in ('blocked','planning_failed') else 0


if __name__=='__main__':raise SystemExit(main())
