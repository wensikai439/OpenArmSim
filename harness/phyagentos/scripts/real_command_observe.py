"""Bounded three-view observation only, using original camera sources; no robot sender."""
from pathlib import Path
import sys,json,time,os,subprocess,fcntl,signal
import numpy as np,cv2
CTRL=Path(str(Path(os.environ.get("PIPER_CONTROL_ROOT", str(Path(__file__).resolve().parents[1] / "external/robot-control")))))
OUT=Path(os.environ['PAOS_COMMAND_CAMERA_RUN']).resolve()
if not OUT.is_relative_to(Path(str(Path(__file__).resolve().parents[1] / 'runs/real-commands')).resolve()):raise ValueError('Invalid camera output')
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(CTRL))
from piper_baseline.hil.compact_recording import CompactRecorder,ExistingCameraRecorder
from piper_baseline.camera_runner import commands,stop_children
class Live(CompactRecorder):
 def __init__(self,*a,**kw):
  self.latest={};self.last_saved={};super().__init__(*a,**kw)
 def image(self,camera,**value):
  super().image(camera,**value)
  now=time.monotonic()
  if now-self.last_saved.get(camera,0)<.5:return
  self.last_saved[camera]=now
  rgb=np.frombuffer(value['pixels'],np.uint8).reshape(value['height'],value['width'],3)
  tmp=OUT/(camera+'.tmp.jpg');cv2.imwrite(str(tmp),rgb[:,:,::-1]);tmp.replace(OUT/(camera+'-latest.jpg'))
  self.latest[camera]={k:v for k,v in value.items() if k!='pixels'}
  self.latest[camera]['saved_wall_s']=time.time()
  p=OUT/(camera+'.tmp.json');p.write_text(json.dumps(self.latest[camera]));p.replace(OUT/(camera+'-latest.json'))
lock=(CTRL/'run/cameras.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rec=Live(OUT/'scene-recording',video=True,grippers=False,fps=30.)
cams=ExistingCameraRecorder(rec,211,start_sources=False)
children=[];logs=[]
def stop(sig,frm):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 cams.start()
 cfg=json.loads((CTRL/'config/act_offline.json').read_text())
 env=dict(os.environ,ROS_DOMAIN_ID='211',ROS_LOCALHOST_ONLY='0',PIPER_RUNTIME_OWNER='paos-real-command-console')
 for name,command in commands(cfg).items():
  log=(OUT/(name+'-camera.log')).open('x');logs.append(log)
  children.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
 until=time.monotonic()+1800
 while time.monotonic()<until:
  if any(p.poll() is not None for p in children):raise RuntimeError('Owned camera child exited')
  p=OUT/'camera-status.tmp';p.write_text(json.dumps(dict(rec.status(),latest=rec.latest,wall_s=time.time()),default=str));p.replace(OUT/'camera-status.json')
  time.sleep(.5)
except KeyboardInterrupt:pass
finally:
 stop_children(children)
 closed=cams.close();writer=rec.close();lock.close()
 (OUT/'camera-final.json').write_text(json.dumps(dict(rec.status(),camera_closed=closed,writer_closed=writer),default=str,indent=2))
 for log in logs:log.close()
