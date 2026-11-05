"""Dependency/geometry preparation and explicitly requested no-motion capture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def dependencies():
    # Module imports only. No Camera, OpenArm, pipeline/context or motor object.
    import cv2
    import numpy
    import pyrealsense2
    import openarm_can
    import fcl
    from scipy.optimize import least_squares
    if not hasattr(cv2.aruco,'ArucoDetector'):raise RuntimeError('OpenCV ArUco detector missing')
    return {'python':sys.version.split()[0],'environment':sys.prefix,
        'modules':{'opencv':cv2.__version__,'numpy':numpy.__version__,
            'realsense':str(pyrealsense2.__file__),'can':str(openarm_can.__file__),'ik':callable(least_squares),'fcl':str(fcl.__file__)},
        'device_objects_created':0,'devices_started':False,'motion_commands':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['deps','geometry','digest','capture'],nargs='?',default='deps')
    p.add_argument('--confirm-on-site',action='store_true');p.add_argument('--camera',choices=['left','right'],default='right')
    p.add_argument('--seconds',type=float,default=10);a=p.parse_args()
    hardware=json.loads((ROOT/'config/hardware.json').read_text())
    if a.mode=='deps':result=dependencies()
    elif a.mode=='geometry':
        from planning import mesh_envelopes
        path=ROOT/'config/collision.json'
        if path.exists():raise ValueError('collision config already exists; refusing to overwrite measurements')
        model=mesh_envelopes(hardware);path.write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
        result={'capsules':len(model['capsules']),'config':'config/collision.json','physical_trial_passed':False,'devices_started':False}
    elif a.mode=='digest':
        from calibration import check,subject
        cal=json.loads((ROOT/'config/calibration.json').read_text());path=ROOT/'config/collision.json'
        collision=json.loads(path.read_text()) if path.is_file() else {}
        from calibration import implementation
        result={'subject_sha256':subject(cal,hardware,collision,implementation(ROOT)),
            'missing':check(cal,hardware,collision,ROOT),'devices_started':False}
    else:
        if not a.confirm_on_site or not 0<a.seconds<=60:raise ValueError('capture requires --confirm-on-site and 0 < seconds <= 60')
        from devices import Camera,LegacyArm
        from service import Journal
        import cv2
        import numpy as np
        folder=ROOT/'records'/('capture-'+time.strftime('%Y%m%d-%H%M%S'))
        journal=Journal(folder,{'mode':'raw_device_capture','camera':a.camera,'motors_enabled':False,
            'timestamp_qualification':'raw device clock and host arrival; not synchronized object poses'})
        camera=Camera(hardware['cameras'][a.camera],journal.record);driver=LegacyArm(hardware,journal.record);count=0
        try:
            camera.open(operator_confirmed=True);driver.open(operator_confirmed=True)
            deadline=time.monotonic()+a.seconds
            while time.monotonic()<deadline:
                before=driver.read();frame=camera.read(diagnostic=True);after=driver.read();count+=1
                name=str(count)
                if not cv2.imwrite(str(folder/(name+'.jpg')),frame.pop('bgr')):raise RuntimeError('image logging failed')
                np.save(folder/(name+'-depth.npy'),frame.pop('depth_m'),allow_pickle=False)
                frame={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in frame.items()}
                journal.record('raw_calibration_frame',{'frame':frame,'joints_before':before,'joints_after':after,'image':name+'.jpg','depth':name+'-depth.npy'})
                time.sleep(.15)
        finally:
            try:camera.close()
            finally:
                try:driver.close()
                finally:journal.close()
        result={'record':str(folder),'frames':count,'motors_enabled':False,'motion_commands':0,'calibration_accepted':False}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
