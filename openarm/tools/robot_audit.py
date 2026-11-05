"""Read-only SSH source audit; writes only local, small deployment contracts."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=r'''
from pathlib import Path
import hashlib,json
root=Path('/home/sj/Desktop/wuji/lerobot_20260116/openarm_deploy')
names=['model_launchers/configs/act_startup_target.json','openarm/openarm_controller.py',
 'model_launchers/shared/joint_feedback.py','model_launchers/shared/joint_initialization.py',
 'model_launchers/models/dp/deploy_dp_dual_joint.py','pyroki_/robot_ik_solver.py','pyroki_/config/robot.yaml']
startup=json.loads((root/names[0]).read_text())
urdf=Path('/home/sj/Desktop/lerobot/openarm_deploy/openarm/openarm_description/urdf/robot/openarm_bimanual.urdf')
data={'schema':'openarm-hardware-reference/1','deploy_root':str(root),
 'source_hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
 'startup_target_file':names[0],'startup_target':startup['target'],'feature_names':startup['feature_names'],
 'startup_episode':startup['selected_episode'],'startup_interpolation_steps':400,
 'startup_interval_s':.01,'startup_final_hold_s':5.,
 'urdf':str(urdf),'urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest(),
 'state_order':'left7,left_gripper_motor,right7,right_gripper_motor',
 'cameras':{'head':{'device':'/dev/video12','resolution':[1280,960],'fps':30},
 'left':{'serial':'150622074105','resolution':[640,480],'fps':30},
 'right':{'serial':'236422072385','resolution':[640,480],'fps':30}},
 'read_only':True,'hardware_access':False,'startup_executed':False,
 'calibration_warning':'existing calib/{left,right}.yaml binds VR controller to wrist, not optical camera to robot base',
 'solver_warning':'existing IK uses internal wxyz+xyz despite a conflicting xyz+xyzw comment; never pass arrays by assumed order'}
protected=json.loads((root.parent/'LLMHarness/records/protected-before.json').read_text())
print(json.dumps({'hardware':data,'protected':protected}))
'''


def main():
    result=subprocess.run(['ssh','-o','BatchMode=yes','data-cs','PYTHONDONTWRITEBYTECODE=1 python3 -B -'],
        input=SCRIPT,text=True,capture_output=True,check=True)
    data=json.loads(result.stdout)
    (ROOT/'deploy/hardware.json').write_text(json.dumps(data['hardware'],ensure_ascii=False,indent=2)+'\n')
    out=ROOT/'runs/20261024-ready'
    (out/'protected.json').write_text(json.dumps(data['protected'],indent=2)+'\n')
    print(json.dumps({'read_only_ssh':True,'hardware_access':False,'pinned_sources':len(data['hardware']['source_hashes'])}))


if __name__=='__main__':main()
