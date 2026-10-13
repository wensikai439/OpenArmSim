"""Compare public models with source-URDF FK using only numeric SSH results.

The remote URDF remains remote. The remote program reads existing files and
returns hashes and error magnitudes, not source code or robot parameters.
No hardware libraries, writes, caches, or device interfaces are used remotely.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sim.reference_phases import landmarks

REMOTE = r'''
import json,sys,math,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
p=json.loads(sys.stdin.readline())
source=Path('/home/sj/Documents/moqi_workspace_20251119_211400/openarm/openarm_description/urdf/robot/openarm_bimanual.urdf')
raw=source.read_bytes(); robot=ET.fromstring(raw)
joints=list(robot.findall('joint'))
parents={j.find('child').get('link'):j for j in joints}
def ident(): return [[float(i==j) for j in range(4)] for i in range(4)]
def mm(a,b): return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def rotation(axis,theta):
 r=ident(); length=math.sqrt(sum(v*v for v in axis)); axis=[v/length for v in axis]
 x,y,z=axis;c=math.cos(theta);s=math.sin(theta);v=1-c
 r[:3]=[[c+x*x*v,x*y*v-z*s,x*z*v+y*s,0.],[y*x*v+z*s,c+y*y*v,y*z*v-x*s,0.],[z*x*v-y*s,z*y*v+x*s,c+z*z*v,0.]]
 return r
def inv(t):
 r=ident()
 for i in range(3):
  for j in range(3): r[i][j]=t[j][i]
  r[i][3]=-sum(t[j][i]*t[j][3] for j in range(3))
 return r
def parse(value,default): return [float(v) for v in (value or default).split()]
names={}
for side in ('left','right'):
 names[side]=[]
 for n in range(1,8):
  found=[j for j in joints if j.get('name').endswith(side+'_joint'+str(n))]
  assert len(found)==1
  names[side].append(found[0])
records={}
for ep,item in p['sources'].items():
 file=Path(item['path']);assert file.is_relative_to(Path('/home/sj/Documents/moqi_workspace_20251119_211400/record_data'))
 data=file.read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256']
 records[ep]=json.loads(data)
rows=[]
for case in p['cases']:
 q=records[str(case['episode'])][case['frame']]['observations']['qpos']
 assert len(q)==14
 values={j.get('name'):q[7*i+n] for i,side in enumerate(('left','right')) for n,j in enumerate(names[side])}
 cache={}
 def fk(link):
  if link in cache:return cache[link]
  if link not in parents:return ident()
  joint=parents[link];origin=joint.find('origin');t=ident()
  xyz=parse(origin.get('xyz') if origin is not None else None,'0 0 0')
  rpy=parse(origin.get('rpy') if origin is not None else None,'0 0 0')
  t=mm(mm(rotation([0,0,1],rpy[2]),rotation([0,1,0],rpy[1])),rotation([1,0,0],rpy[0]))
  for i in range(3):t[i][3]=xyz[i]
  kind=joint.get('type');axis=joint.find('axis');a=parse(axis.get('xyz') if axis is not None else None,'1 0 0')
  amount=values.get(joint.get('name'),0.)
  if kind in ('revolute','continuous'):t=mm(t,rotation(a,amount))
  elif kind=='prismatic':
   move=ident()
   for i in range(3):move[i][3]=a[i]*amount
   t=mm(t,move)
  cache[link]=mm(fk(joint.find('parent').get('link')),t)
  return cache[link]
 result={'episode':case['episode'],'frame':case['frame'],'models':{}}
 for model,sides in case['models'].items():
  result['models'][model]={}
  for side,prediction in sides.items():
   first,last=[names[side][n].find('child').get('link') for n in (0,6)]
   truth=mm(inv(fk(first)),fk(last))
   distance=math.sqrt(sum((truth[i][3]-prediction[i][3])**2 for i in range(3)))
   cosine=(sum(truth[i][j]*prediction[i][j] for i in range(3) for j in range(3))-1)/2
   result['models'][model][side]={'position_error_m':distance,'rotation_error_deg':math.degrees(math.acos(max(-1.,min(1.,cosine))))}
 rows.append(result)
print(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),'cases':rows,
 'scope':'relative FK from joint1 child to joint7 child; current source version, not a real TCP measurement or capture-version certification'}))
'''


def main():
    reference = json.loads((ROOT/'reference/real_trajectory.json').read_text())['episodes']
    clock = json.loads((ROOT/'reference/acquisition.json').read_text())
    models = {name:mujoco.MjModel.from_xml_path(str(ROOT/'openarm_mujoco'/name/'openarm_bimanual.xml')) for name in ('v1','v2')}
    cases=[]
    for ep,row in reference.items():
        events=landmarks(row)['events']
        for frame in (0,events['right_close'],events['left_close'],row['frames']-1):
            case={'episode':int(ep),'frame':frame,'models':{}}
            for version,model in models.items():
                data=mujoco.MjData(model)
                for side in ('left','right'):
                    for i,q in enumerate(row[f'{side}_joints'][frame],1):
                        data.qpos[model.joint(f'openarm_{side}_joint{i}').qposadr[0]]=q
                mujoco.mj_forward(model,data)
                case['models'][version]={}
                for side in ('left','right'):
                    first,last=[int(model.jnt_bodyid[model.joint(f'openarm_{side}_joint{i}').id]) for i in (1,7)]
                    r=data.xmat[first].reshape(3,3).T
                    t=np.eye(4);t[:3,:3]=r@data.xmat[last].reshape(3,3)
                    t[:3,3]=r@(data.xpos[last]-data.xpos[first])
                    case['models'][version][side]=t.tolist()
            cases.append(case)
    request={'cases':cases,'sources':{ep:{k:r[k] for k in ('path','sha256')} for ep,r in clock['episodes'].items()}}
    # Program on argv, data on stdin; no remote temporary file or source export.
    import shlex
    command='python3 -B -c '+shlex.quote(REMOTE)
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
        '-o','ControlMaster=no','-o','ControlPath=none','-T','data-cs',command],
        input=json.dumps(request),text=True,capture_output=True,timeout=60,check=True)
    result=json.loads(r.stdout)
    if len(result['cases']) != len(cases):raise ValueError('incomplete numeric kinematics comparison')
    output=ROOT/'runs'/(time.strftime('%Y%m%d-%H%M%S')+'-kinematics-'+uuid.uuid4().hex[:6])
    output.mkdir()
    result.update({'schema':'openarm-kinematics-audit/1','status':'completed','kind':'kinematics',
        'source_exported':False,'hardware_access':False,'transfer_ready':False,
        'reference_sha256':hashlib.sha256((ROOT/'reference/real_trajectory.json').read_bytes()).hexdigest(),
        'public_model_sha256':{v:hashlib.sha256((ROOT/'openarm_mujoco'/v/'openarm_bimanual.xml').read_bytes()).hexdigest() for v in models}})
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'path':str(output),'max_errors':{
        v:{s:{key:max(row['models'][v][s][key] for row in result['cases']) for key in ('position_error_m','rotation_error_deg')}
           for s in ('left','right')} for v in models}},indent=2))


if __name__=='__main__':main()
