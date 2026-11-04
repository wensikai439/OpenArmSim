"""Actual native ACT policy evaluation; no official Task/verdict or dataset admission."""
from pathlib import Path
import argparse,json,os,sys,types,subprocess,selectors,time,hashlib
import numpy as np,mujoco,cv2
from PIL import Image
from scipy.spatial.transform import Rotation
ROOT=Path(str(Path(__file__).resolve().parents[1]))
ASSET=Path(str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_robocasa")))))
sys.path.insert(0,str(ASSET/"src"));sys.path.insert(0,str(Path(os.environ.get("RPENT_ROOT", str(Path(__file__).resolve().parents[1] / "external/rpent"))) / 'src/RPent'))
sys.path.insert(0,str(ROOT/"bundles/rpent-piper-sim/profiles/rpent"))
# This batch pins two exact historical modules while reusing the shared asset package.
SOURCE_LOCK_PATH=os.environ.get("PAOS_FROZEN_SIM_SOURCE_LOCK")
SOURCE_LOCK=None
if SOURCE_LOCK_PATH:
    SOURCE_LOCK=json.loads(Path(SOURCE_LOCK_PATH).read_text())
    assert SOURCE_LOCK["asset_root"]==str(ASSET)
    for name in ("air_handoff_task","pose_preflight"):
        entry=SOURCE_LOCK["modules"][name]
        source=Path(entry["path"])
        assert source.parent==Path(SOURCE_LOCK_PATH).parent/"piper_robocasa"
        assert hashlib.sha256(source.read_bytes()).hexdigest()==entry["sha256"],name
    import piper_robocasa
    piper_robocasa.__path__.insert(0,str(Path(SOURCE_LOCK_PATH).parent/"piper_robocasa"))
from joint_expert_adapter import install
from native_shim import CooperativeTask
from piper_robocasa.pose_preflight import cross_arm_penetration,static_environment_penetration
from run_sim_expert import physical_success
SOURCE_LOAD={"asset_root":str(ASSET),"frozen_source_lock":SOURCE_LOCK_PATH,"modules":{}}
for name in ("air_handoff_task","pose_preflight"):
    module=sys.modules["piper_robocasa."+name]
    path=Path(module.__file__).resolve()
    row={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    if SOURCE_LOCK:
        entry=SOURCE_LOCK["modules"][name]
        assert row=={"path":entry["path"],"sha256":entry["sha256"]},name
    SOURCE_LOAD["modules"][name]=row
SOURCE_LOAD["resolved_asset_root"]=str(sys.modules["piper_robocasa.air_handoff_task"].ROOT)
assert SOURCE_LOAD["resolved_asset_root"]==str(ASSET)
def save(path,value):path.write_text(json.dumps(value,indent=2,default=str)+"\n")
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def response(proc,selector,transcript):
 while True:
  if not selector.select(timeout=30):raise TimeoutError("ACT inference worker timeout; no action replay")
  line=proc.stdout.readline().decode("utf-8")
  if not line:raise RuntimeError("ACT worker exited; "+str(proc.poll()))
  transcript.write(line);transcript.flush()
  try:row=json.loads(line)
  except json.JSONDecodeError:continue
  if row.get("type") in ("ready","action"):return row
def physical_step(ws,action=None):
 row={"transition_index":int(ws.step_count),"source":"upstream_ACT_prediction_to_consumed_absolute_joint_targets",
  "pre":ws._sample(),"policy_action14":ws.policy_action14.tolist(),"gripper_task_modes":dict(ws.binary_task_modes)}
 command=ws.policy_command.copy();count=int(round(.05/ws.model.opt.timestep));done=0
 try:
  row.update(command_native=command.tolist(),command_created_before_step=True)
  for i in range(count):
   ws.robot.control(command,policy_step=i==0)
   if i==0:
    consumed={s:ws.robot.part_controllers[s].goal_qpos.copy() for s in ("left","right")}
    for s in consumed:np.testing.assert_array_equal(consumed[s],command[slice(*ws.robot._action_split_indexes[s])])
    row["controller_consumed_joint_targets_rad"]={s:q.tolist() for s,q in consumed.items()}
    row["ctrl_first"]=ws.data.ctrl.tolist()
   ws.sim.step();done+=1;ws.safety()
  ws.step_count+=1;ws.joint_command[:]=command
  row["completed"]=True
 except Exception as exc:
  ws.paused=True;ws.last_error=str(exc);row.update(completed=False,error=str(exc));raise
 finally:
  row.update(physics_substeps_done=done,physics_substeps_expected=count,post=ws._sample())
  ws.joint_trace.write(json.dumps(row,separators=(",",":"))+"\n");ws.joint_trace.flush()
def update_binary(task,modes):
 ws=task.ws
 # Apply OPEN effects before CLOSE effects, matching contact-triggered native transfer/release.
 for s in ("left","right"):
  other="right" if s=="left" else "left"
  if modes[s]!=1:continue
  held=task._held_sides()
  if held[s] and task.receiver_ready==other:
   task._attach_measured(other,donor=s);task.receiver_ready=None
  elif held[s]:
   before=task.data.qpos.copy();task.data.eq_active[task.eq[s]]=0;mujoco.mj_forward(task.model,task.data)
   np.testing.assert_array_equal(before,task.data.qpos)
   task.events.append({"kind":"composable_release","side":s,"qpos_change":0,"time":float(task.data.time)})
  if task.receiver_ready==s:task.receiver_ready=None
 for s in ("left","right"):
  idx=ws.robot._action_split_indexes[s+"_gripper"][0]
  ws.binary_task_modes[s]=modes[s]
  if modes[s]==1:width=.1
  else:
   width=.05*(1-float(ws.action[idx]));contacts=task._finger_contacts_now(s)
   if not contacts:width=max(task.BINARY_CLOSED_CLEARANCE_M,width-.002)
  ws.action[idx]=1-width/.05
 for s in ("left","right"):
  if modes[s]!=0:continue
  contacts=task._finger_contacts_now(s);held=task._held_sides()
  if not contacts or held[s]:continue
  geometry=task._tabletop_approach()
  if geometry.get("active"):
   matrix=task.data.site_xmat[ws.eef_ids[s]].reshape(3,3);limit=np.sin(geometry["max_grasp_tilt_rad"])
   if abs(matrix[2,2])>limit or abs(matrix[2,1])>limit:raise RuntimeError("Policy contacted tabletop bottle without required horizontal fingers")
  other="right" if s=="left" else "left"
  if held[other]:task.receiver_ready=s
  else:task._attach_measured(s)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,required=True)
 ap.add_argument("--scene-config",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
 ap.add_argument("--max-control-steps",type=int,default=4500);a=ap.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 save(out/"source-load.json",SOURCE_LOAD)
 os.environ["PAOS_GRIPPER_MODE"]="binary_endpoints"
 install();task=None;proc=None;error=None;success=False;complete_steps=0;ever_held=False
 stderr=(out/"inference.stderr.log").open("x");transcript=(out/"inference.stdout.jsonl").open("x")
 try:
  task=CooperativeTask(out/"sim",render=True,run_id=out.name,config_path=a.scene_config)
  save(out/"initial-observation.json",task.observe());ws=task.ws
  ws.step=types.MethodType(physical_step,ws)
  task.composable_started=True;task.composable_warning_baseline=np.array([w.number for w in task.data.warning])
  task.phase="ACT_policy_control";initial_free_bottle=task.bottle().copy()
  cmd=[str(Path(os.environ.get("PIPER_TRAINING_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_training_project"))) / '.venvs/act/cu128/bin/python'),
   str(ROOT/"scripts/infer_binary_act_checkpoint.py"),"--checkpoint",str(a.checkpoint.resolve())]
  save(out/"inference.command.json",{"argv":cmd,"cwd":str(ROOT)})
  env=os.environ.copy();env.pop("PYTHONPATH",None);env.pop("PYTHONHOME",None)
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
   text=False,bufsize=0)
  sel=selectors.DefaultSelector();sel.register(proc.stdout,selectors.EVENT_READ)
  ready=response(proc,sel,transcript);assert ready["type"]=="ready" and ready["state"]==12 and ready["action"]==14
  save(out/"policy-contract.json",{"worker":ready,"checkpoint":str(a.checkpoint.resolve()),
   "checkpoint_sha256":digest(a.checkpoint/"model.safetensors"),"control_hz":20,"physics_dt_s":float(task.model.opt.timestep),
   "assistance":"same measured bilateral-contact assisted semantics; incremental binary close/contact stop",
   "binary_decode_threshold":.5,"prediction_clipped_or_expert_corrected":False,
   "joint_bounds_and_max_delta_rad":.5,"formal_verdict":None,"dataset_admission":False})
  initial_transfer_count=len(task.composable_transfers)
  predictions=(out/"predictions.jsonl").open("x")
  for n in range(a.max_control_steps):
   o=task.observe()
   if physical_success(o,initial_transfer_count):success=True;break
   paths={}
   folder=out/"policy-frames"/f"{n:06d}";folder.mkdir(parents=True)
   for camera in ("cam_left","cam_middle","cam_right"):
    rgb=cv2.resize(ws.frame(camera),(320,240),interpolation=cv2.INTER_AREA)
    p=folder/(camera+".jpg");Image.fromarray(rgb).save(p,quality=90);paths[camera]=str(p)
   request={"control_step":n,"state12":[*o["joint_positions_rad"]["left"],*o["joint_positions_rad"]["right"]],
    "rgb_jpeg_paths":paths}
   proc.stdin.write((json.dumps(request)+"\n").encode("utf-8"));proc.stdin.flush()
   pred=response(proc,sel,transcript);assert pred["type"]=="action" and pred["control_step"]==n
   act=np.asarray(pred["action14"],dtype=float);assert act.shape==(14,) and np.isfinite(act).all()
   goals={"left":act[:6],"right":act[7:13]};modes={"left":int(act[6]>=.5),"right":int(act[13]>=.5)}
   predictions.write(json.dumps({"control_step":n,"source_observation":request,"prediction":pred,"binary_modes":modes})+"\n");predictions.flush()
   for s,q in goals.items():
    limits=task.model.jnt_range[ws.joint_ids[s]]
    if ((q<limits[:,0])|(q>limits[:,1])).any():raise RuntimeError("Policy joint target outside unchanged model bounds: "+s)
    if np.max(np.abs(q-task.data.qpos[ws.qindexes[s]]))>.5:raise RuntimeError("Policy joint target exceeds unchanged 0.5rad bound: "+s)
   start=task.data.qpos.copy();scratch=mujoco.MjData(task.model)
   count=max(1,int(np.ceil(max(np.max(np.abs(goals[s]-start[ws.qindexes[s]])) for s in goals)/.05)))
   for f in np.linspace(0.,1.,count+1)[1:]:
    scratch.qpos[:]=start
    for s,q in goals.items():scratch.qpos[ws.qindexes[s]]=start[ws.qindexes[s]]*(1-f)+q*f
    mujoco.mj_fwdPosition(task.model,scratch)
    collision=(cross_arm_penetration(task.model,scratch)
      or static_environment_penetration(task.model,scratch,"left")
      or static_environment_penetration(task.model,scratch,"right"))
    if collision is not None:raise RuntimeError("Policy joint path rejected before action: "+str(collision))
   update_binary(task,modes)
   ws.policy_command=ws.joint_command.copy();ws.policy_action14=act
   for s,q in goals.items():
    ws.policy_command[slice(*ws.robot._action_split_indexes[s])]=q
    ws.policy_command[ws.robot._action_split_indexes[s+"_gripper"][0]]=ws.action[ws.robot._action_split_indexes[s+"_gripper"][0]]
   if any(task._held_sides().values()):ever_held=True
   task._free_bottle_motion_reference=None if ever_held else initial_free_bottle
   task.tick();complete_steps+=1
   if n%100==0:print(json.dumps({"complete_control_steps":complete_steps,"physical_goal_met":False}),flush=True)
  predictions.close()
  final=task.observe();success=success or physical_success(final,initial_transfer_count)
 except Exception as exc:
  error={"type":type(exc).__name__,"message":str(exc)}
 finally:
  if task:
   final=task.observe();save(out/"final-observation.json",final);task.save_artifacts();task.close()
  if proc:
   if proc.poll() is None:
    try:proc.stdin.write(b'{"type":"stop"}\n');proc.stdin.flush();proc.wait(timeout=10)
    except Exception:proc.terminate();proc.wait(timeout=10)
   save(out/"inference.exit.json",{"exit_code":proc.returncode})
  stderr.close();transcript.close()
  summary={"mode":"actual native learned-policy evaluation; no official Task/verdict or dataset admission",
   "physical_goal_met":bool(success),"complete_control_steps":complete_steps,"error":error,
   "max_control_steps":a.max_control_steps,"model_requests":0,"hardware_connected":False}
  save(out/"summary.json",summary);print(json.dumps(summary),flush=True)
 return 0 if success and error is None else 1
if __name__=="__main__":raise SystemExit(main())
