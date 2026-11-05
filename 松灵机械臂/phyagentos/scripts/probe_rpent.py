"""Explicit finite native RPent SIM smoke; owns and closes its own child."""
import base64,hashlib,json,os,subprocess,time,urllib.request,uuid
from pathlib import Path
ROOT=Path(str(Path(__file__).resolve().parents[1]));RP=Path(str(Path(os.environ.get("RPENT_ROOT", str(Path(__file__).resolve().parents[1] / "external/rpent"))) / 'src/RPent'));SIM=Path(str(Path(os.environ.get("PIPER_SIM_ROOT", str(Path(__file__).resolve().parents[1] / "external/piper_robocasa")))))
run=ROOT/'runs/rpent-air-native-01';run.mkdir(exist_ok=True)
instance='paos-air-native-'+uuid.uuid4().hex[:10]
cmd=[str(SIM/'.venv/bin/python'),str(RP/'robots/piper_sim/env_server.py'),'--transport','http','--host','127.0.0.1','--port','19022','--instance-id',instance,'--output',str(run/'sim'),'--a-source-root',str(SIM/'src'),'--task-variant','air-handoff','--render','--enable-actions','--parent-watch']
env=os.environ.copy();env.update(MUJOCO_GL='egl',PYTHONPATH=str(RP)+os.pathsep+str(SIM/'src'))
log=(run/'server.log').open('w');process=subprocess.Popen(cmd,env=env,cwd=RP,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def rpc(method,**kwargs):
 req=urllib.request.Request('http://127.0.0.1:19022/call',data=json.dumps({'method':method,'args':[],'kwargs':kwargs}).encode(),headers={'Content-Type':'application/json'})
 with opener.open(req,timeout=120) as response:data=json.load(response)
 if data.get('ok') is not True:raise RuntimeError(data)
 return data['result']
def capture(label):
 frames=rpc('env.capture_frames',cameras=['cam_right','overview'])
 for f in frames['frames']:
  rgb=f.pop('rgb');f['rgb_sha256']=hashlib.sha256(base64.b64decode(rgb['__ndarray__'])).hexdigest();f['rgb_shape']=rgb['shape']
 (run/(label+'-frames.json')).write_text(json.dumps(frames,indent=2))
 state=rpc('env.observe');(run/(label+'-state.json')).write_text(json.dumps(state,indent=2));return state
report={'command':cmd,'instance_id':instance,'pid':process.pid,'actions':[]}
try:
 for _ in range(60):
  if process.poll() is not None:raise RuntimeError('child exited '+str(process.returncode))
  try:report['meta']=rpc('env.get_env_meta');break
  except (OSError,urllib.error.URLError):time.sleep(.5)
 else:raise RuntimeError('startup deadline')
 initial=capture('initial')
 for side in ['right','left']:
  targets=list(initial['tcp_world_m'][side]);targets[2]+=.02
  for action in [{'id':'set_gripper','payload':{'side':side,'open_or_close':'open'}},{'id':'move_arm','payload':{'side':side,'target_world_m':targets}}]:
   result=rpc('env.step_action',action=action);report['actions'].append({'action':action,'result':result});print(side,action['id'],result.get('primitive_completed'),result.get('outcome'))
   if result.get('primitive_completed') is not True:raise RuntimeError('native primitive incomplete')
 final=capture('final');report.update(initial_physics_step=initial['physics_step'],final_physics_step=final['physics_step'],initial_sim_time_s=initial['sim_time_s'],final_sim_time_s=final['sim_time_s'],task_success=final['task_success'],tcp_delta_m={s:[b-a for a,b in zip(initial['tcp_world_m'][s],final['tcp_world_m'][s])] for s in ['left','right']})
except BaseException as error:report['error']=str(error);raise
finally:
 if process.poll() is None:
  try:report['shutdown']=rpc('shutdown')
  finally:process.stdin.close()
 report['exit_code']=process.wait(timeout=30);log.close();(run/'report.json').write_text(json.dumps(report,indent=2))
print('REPORT',run/'report.json',report['tcp_delta_m'])
