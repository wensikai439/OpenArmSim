"""Offline local IK around current feedback using the official PiPER MDH.
No sender; native owner receives the resulting bounded joint increments separately.
"""
from pathlib import Path
import ast,json,math,sys,argparse,importlib.util,os
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
OUT=Path(os.environ['PAOS_COMMAND_RUN']).resolve()
spec=importlib.util.spec_from_file_location('current_stage',Path(__file__).resolve().parent/'real_command_stage.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
tree=ast.parse((stage.CTRL/'upstream/pyAgxArm/pyAgxArm/api/constants.py').read_text())
mdh=next(ast.literal_eval(n.value)['piper'] for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROBOT_MDH_PRESET' for t in n.targets))
def fk(q):
 T=np.eye(4)
 for (d,a,al,off),v in zip(mdh,q):
  th=v+off;c,s=np.cos(th),np.sin(th);ca,sa=np.cos(al),np.sin(al)
  T=T@np.array([[c,-s,0,a],[ca*s,ca*c,-sa,-sa*d],[sa*s,sa*c,ca,ca*d],[0,0,0,1]])
 return T

def solve(obs,side,translation,frame='base'):
 q0=np.radians(obs['feedback']['follower_'+side]['measured_deg']);T0=fk(q0);goal=T0.copy()
 goal[:3,3]+=np.array(translation) if frame=='base' else T0[:3,:3]@np.array(translation)
 def err(q):
  T=fk(q);return np.r_[(T[:3,3]-goal[:3,3])/.0005,Rotation.from_matrix(goal[:3,:3].T@T[:3,:3]).as_rotvec()/.02,(q-q0)*np.array([1,1,1,10,1,10])]
 limit=math.radians(25.)
 opt=least_squares(err,q0,bounds=(q0-limit,q0+limit),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=350)
 T=fk(opt.x);poserr=np.linalg.norm(T[:3,3]-goal[:3,3]);rot=np.linalg.norm(Rotation.from_matrix(goal[:3,:3].T@T[:3,:3]).as_rotvec())
 if poserr>.0005 or rot>.03:raise RuntimeError('No bounded local IK solution: '+str((poserr,rot)))
 path=[fk(q0+u*(opt.x-q0)) for u in np.linspace(0,1,51)]
 dev=max(float(np.linalg.norm(t[:3,3]-(T0[:3,3]+u*(goal[:3,3]-T0[:3,3])))) for t,u in zip(path,np.linspace(0,1,51)))
 if dev>.005:raise RuntimeError('Joint interpolation deviates over 5mm from local Cartesian segment')
 return dict(side=side,relative_frame=frame,requested_translation_m=translation,current_q_deg=np.degrees(q0).tolist(),
  current_flange_m=T0[:3,3].tolist(),target_flange_m=T[:3,3].tolist(),delta_deg=np.degrees(opt.x-q0).tolist(),
  position_residual_m=float(poserr),orientation_residual_rad=float(rot),max_path_deviation_m=dev,
  native_feedback=obs,ik='scipy bounded least_squares, official pyAgxArm PiPER MDH, current feedback seed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--side',choices=['left','right'],default='right');p.add_argument('--translation',required=True);p.add_argument('--frame',choices=['base','tool'],default='base');args=p.parse_args()
 obs=stage.observe();v=solve(obs,args.side,[float(x) for x in args.translation.split(',')],args.frame)
 dest=OUT/(args.name+'-ik.json');dest.write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='native_feedback'},indent=2))
