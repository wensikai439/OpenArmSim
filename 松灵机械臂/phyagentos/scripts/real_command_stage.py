"""Prepare one explicitly specified relative expert stage from fresh observations.
Reads native CAN feedback only; register/prepare never dispatches an action.
"""
from pathlib import Path
import argparse,json,time,sys,select,struct,fcntl,shutil,math,os
CTRL=Path(str(Path(os.environ.get("PIPER_CONTROL_ROOT", str(Path(__file__).resolve().parents[1] / "external/robot-control")))))
PAOS=Path(str(Path(__file__).resolve().parents[1]))
OUT=Path(os.environ['PAOS_COMMAND_RUN']).resolve()
if not OUT.is_relative_to((PAOS/'runs/real-commands').resolve()):raise ValueError('Invalid real command destination')
sys.path[:0]=[str(CTRL/'tools'),str(PAOS/'bundles/piper-real-runtime/profiles/runtime')]
from replay_handoff import Port
from right_follow import open_can,CAN,ip
from dual_follow import channels_from_config
from runtime_bridge import Bridge
import native_replay_bridge

def observe():
 lock=(CTRL/'run/unified_runtime.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 channels=channels_from_config(('left','right'));roles=('follower_left','follower_right')
 before=json.loads(ip('-j','-s','link','show','type','can'))
 ports={r:Port(open_can(channels[r],tuple(range(0x2A1,0x2A9))+tuple(range(0x261,0x267))),r) for r in roles}
 for p in ports.values():p.native_coordinates=True
 try:
  until=time.monotonic()+1.
  while time.monotonic()<until:
   ready,_,_=select.select([p.follower for p in ports.values()],[],[],.02)
   for p in ports.values():
    if p.follower in ready:
     for i in range(64):
      try:raw=p.follower.recv(CAN.size)
      except BlockingIOError:break
      p.received(raw,time.monotonic())
  now=time.monotonic();fb={}
  for role,p in ports.items():
   ids=tuple(range(0x2A1,0x2A9))+tuple(range(0x261,0x267))
   if any(now-p.rx_stamps.get(cid,float('-inf'))>.1 for cid in ids):raise TimeoutError(role+': no fresh complete feedback')
   if not p.enabled() or p.cache['follower'][0x2A1][0]!=1:raise RuntimeError(role+': not enabled CAN follower')
   pose=[]
   for cid in (0x2A2,0x2A3,0x2A4):pose.extend(struct.unpack('>ii',p.cache['follower'][cid]))
   fb[role]=dict(p.snapshot(),status_raw=p.cache['follower'][0x2A1].hex(),
    flange_pose_m_deg=[v/1000000 if i<3 else v/1000 for i,v in enumerate(pose)])
 finally:
  for p in ports.values():p.follower.close()
  lock.close()
 after=json.loads(ip('-j','-s','link','show','type','can'));tx={x['ifname']:x['stats64']['tx']['packets'] for x in before}
 delta={x['ifname']:x['stats64']['tx']['packets']-tx[x['ifname']] for x in after}
 if any(delta.values()):raise RuntimeError('Another sender active during passive planning; inspect owners')
 return dict(captured_wall_s=time.time(),feedback=fb,can_tx_delta=delta,read_only=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('prepared_id')
 p.add_argument('--right-delta',default='0,0,0,0,0,0');p.add_argument('--left-delta',default='0,0,0,0,0,0')
 p.add_argument('--keep-right-closed',action='store_true');p.add_argument('--keep-left-closed',action='store_true')
 p.add_argument('--right-gripper',type=float);p.add_argument('--left-gripper',type=float);p.add_argument('--seconds',type=float,default=8.)
 args=p.parse_args();dest=OUT/args.stage;dest.mkdir(exist_ok=False)
 now=time.time();scene=[]
 for cam in ('cam_left','cam_middle','cam_right'):
  meta=json.loads((OUT/(cam+'-latest.json')).read_text())
  if not 0<=now-meta['saved_wall_s']<2.:raise RuntimeError(cam+': current image unavailable')
  shutil.copy2(OUT/(cam+'-latest.jpg'),dest/(cam+'-before.jpg'))
  (dest/(cam+'-before.json')).write_text(json.dumps(meta,indent=2));scene.append(str(dest/(cam+'-before.jpg')))
 obs=observe();(dest/'feedback-before.json').write_text(json.dumps(obs,indent=2))
 frames=[]
 offsets={s:[float(v) for v in getattr(args,s+'_delta').split(',')] for s in ('left','right')}
 for t,scale in [(0.,0.),(args.seconds,1.),(args.seconds+1.,1.)]:
  row=dict(t_s=t)
  for s in ('left','right'):
   row[s+'_deg']=[v*scale for v in offsets[s]]
   start=obs['feedback']['follower_'+s]['measured_gripper_m'];end=getattr(args,s+'_gripper')
   if end is None:end=start
   row[s+'_gripper_m']=(0. if getattr(args,'keep_'+s+'_closed') else start+(end-start)*scale);row[s+'_gripper_force_mn']=1000
  frames.append(row)
 data=dict(schema='piper-four-arm-replay-v1',execution_mode='follower-relative-expert',expert_stage=args.stage,
  source=dict(origin='current_feedback_and_live_scene',task=json.loads((OUT/'command.json').read_text())['content'],reference_role='current_user_command',
   reference=str(OUT/'command.json'),output_joint_unit='deg',gripper_unit='m'),
  keep_closed_sides=[s for s in ('left','right') if getattr(args,'keep_'+s+'_closed')],
  mapping={'leader_left':'left','follower_left':'left','leader_right':'right','follower_right':'right'},
  coordinate_policy='zeroed-native-width',interpolation='linear',alignment_speed_deg_s=3.,grippers=True,rate=1.,
  replay_peak_joint_speed_deg_s=max(abs(v)/args.seconds for q in offsets.values() for v in q),
  plan_origin=dict(observed_wall_s=obs['captured_wall_s'],scene_evidence=scene,
   feedback_deg={r:v['measured_deg'] for r,v in obs['feedback'].items()}),frames=frames)
 target=dest/'stage.json';target.write_text(json.dumps(data,indent=2))
 b=Bridge();registered=native_replay_bridge.register(b,args.prepared_id,str(target))
 prepared=b.prepare_task(args.prepared_id,'four_can_replay',{'registered_replay_id':registered['registered_replay_id']})
 result=dict(prepared=prepared,registered=registered,hardware_action_submissions=0)
 (dest/'preparation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
