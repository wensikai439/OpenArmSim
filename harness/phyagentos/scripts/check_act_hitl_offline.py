import os
import argparse,json,subprocess,time,hashlib,sys,importlib.metadata
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
CONTROL = Path(os.environ.get('PIPER_CONTROL_ROOT', str(Path(__file__).resolve().parents[1]/'external/robot-control')))
sys.path.insert(0, str(CONTROL))
from piper_baseline.model_config import resolve_config
from piper_baseline.hil.policy import RemotePolicy, child_command
from piper_baseline.shadow_protocol import deny_can
from piper_baseline.act_contract import CAMERAS, validate_contract

deny_can()
parser=argparse.ArgumentParser(description='Real recorded ACT observation through native HITL IPC; CAN denied.')
parser.add_argument('--output', type=Path, required=True)
out=parser.parse_args().output.resolve()
out.mkdir(parents=True, exist_ok=False)
cfg=resolve_config(str(Path(os.environ.get("PIPER_CONTROL_ROOT", str(Path(__file__).resolve().parents[1] / "external/robot-control"))) / 'config/models/act.json'))
dataset=Path(cfg['dataset']); info=json.loads((dataset/'meta/info.json').read_text())
validate_contract(cfg,info)
episodes=dataset/'meta/episodes/chunk-000/file-000.parquet'
meta=pq.read_table(episodes).slice(0,1).to_pylist()[0]
data=dataset/info['data_path'].format(chunk_index=meta['data/chunk_index'],file_index=meta['data/file_index'])
row=pq.read_table(data,columns=['episode_index','frame_index','timestamp','observation.state']).slice(0,1).to_pylist()[0]
assert row['episode_index']==meta['episode_index'] and row['frame_index']==0
images={};sources={}
for camera in CAMERAS:
 key='observation.images.'+camera
 video=dataset/info['video_path'].format(video_key=key,chunk_index=meta['videos/'+key+'/chunk_index'],file_index=meta['videos/'+key+'/file_index'])
 with av.open(str(video)) as stream:
  frame=next(stream.decode(video=0)); rgb=frame.to_ndarray(format='rgb24')
  expected=meta['videos/'+key+'/from_timestamp']+row['timestamp']
  assert abs(float(frame.time)-expected)<=1/30+1e-6
 images[camera]=rgb
 sources[camera]={'video':str(video),'time_s':float(frame.time),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'shape':list(rgb.shape)}
policy=RemotePolicy(out/'act-offline.sock')
log=(out/'act-policy-worker.log').open('w')
worker=None
started=time.monotonic()
try:
 command=child_command(cfg,policy.path,'paos-offline-act-check-20261021')
 worker=subprocess.Popen(command,cwd=cfg['project_root'],stdout=log,stderr=subprocess.STDOUT)
 policy.reset()
 first=policy.select(row['observation.state'],images)
 policy.reset()
 second=policy.select(row['observation.state'],images)
 assert np.isfinite(first).all() and len(first)==14
 delta=float(np.max(np.abs(np.asarray(first)-np.asarray(second))))
 assert delta<1e-5,delta
 result={'scope':'real_recorded_observation_native_HITL_IPC_no_CAN',
  'native_entry':'piper_baseline.hil.policy','checkpoint':cfg['checkpoint'],'python':cfg['python'],
  'dataset':str(dataset),'source_parquet':str(data),'episode_index':row['episode_index'],'frame_index':row['frame_index'],
  'state':row['observation.state'],'images':sources,'action':first,'action_dimension':14,
  'fresh_prediction_epochs':2,'reset_max_abs_difference':delta,'elapsed_s':time.monotonic()-started,
  'motion_command_sent':False,'hardware_task_executed':False,'physical_task_success':None,
  'versions':{name:importlib.metadata.version(name) for name in ['lerobot','torch','torchvision','numpy','av']}}
finally:
 policy.close()
 if worker is not None:
  try: worker.wait(timeout=10)
  except subprocess.TimeoutExpired:
   worker.terminate();worker.wait(timeout=5)
 log.close()
result['worker_exit_code']=worker.returncode
result['worker_pid']=worker.pid
assert worker.returncode==0,worker.returncode
(out/'act-offline-inference.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:result[k] for k in ['scope','action_dimension','reset_max_abs_difference','elapsed_s','worker_exit_code']}))
