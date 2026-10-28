"""Commissioning checks with synthetic geometry and inert device doubles."""
import io
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np
from scipy.spatial.transform import Rotation

from . import handeye as he
from . import calib_math as cm


class FakePlanner:
    def __init__(self):
        self.model={'hulls':[{'link':'wrist','vertices':[[-.05,-.02,-.02],[.05,.02,.02]]}]}
        self.kin=types.SimpleNamespace(pose=self.pose)
    def joints(self,q):return q
    def pose(self,link,q,cache):return cm.rigid(Rotation.from_euler('yz',[q[13],q[14]]).as_matrix(),[.4,-.2,.5])
    def valid(self,q,scene):return bool(np.max(abs(q))<=1)


class HandEyeChecks(unittest.TestCase):
    def test_wrist_envelope_freezes_other_arm_and_both_jaws(self):
        e=he.WristEnvelope(FakePlanner(),np.zeros(16),'right')
        self.assertEqual(len(e.targets()),9)
        for index in [0,7,8,12,15]:
            q=np.zeros(16);q[index]=.0001;self.assertFalse(e.valid(q))
        q=np.zeros(16);q[13]=.101;self.assertFalse(e.valid(q))

    def test_self_collision_check_includes_path_interior(self):
        p=FakePlanner();p.valid=lambda q,s:not .03<q[13]<.04
        e=he.WristEnvelope(p,np.zeros(16),'right');q=np.zeros(16);q[13]=.08
        self.assertTrue(e.valid(q));self.assertFalse(e.validate_path([np.zeros(16),q],{}))

    def test_no_confirmation_never_opens_can_or_enables_a_motor(self):
        class Camera:
            def __init__(self,*a):pass
            def open(self,**kw):pass
            def close(self):pass
        def forbidden(*a,**k):self.fail('CAN or actuator path reached without explicit commissioning confirmation')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir()
            (root/'config/hardware.json').write_text(json.dumps({'cameras':{'right':{'serial':'test'}}}))
            (root/'config/calibration.json').write_text('{}')
            frame={'clock_domain':'global_time','arrival_unix_s':time.time(),'device_timestamp_ms':time.time()*1000}
            with patch.object(he,'preflight',return_value={'missing':[]}),patch.object(he,'Camera',Camera),\
                 patch.object(he,'preview',return_value=({'reprojection_rms_px':.4},frame)),patch.object(he,'LegacyArm',forbidden),\
                 redirect_stdout(io.StringIO()):
                r=he.run(root);self.assertEqual(r['status'],'preview_only')
                with self.assertRaisesRegex(RuntimeError,'confirm-wrist-clearance'):
                    he.run(root,execute=True,confirmed=False)

    def test_checkerboard_reversal_keeps_the_same_physical_frame(self):
        import cv2
        board=cm.calibration_board(he.BOARD);K=np.array([[600.,0,320],[0,600.,240],[0,0,1.]])
        obj=np.zeros((48,3),np.float32);obj[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)*.02
        rv=np.array([.3,-.1,.05]);tv=np.array([-.06,-.04,.5]);pixels,_=cv2.projectPoints(obj,rv,tv,K,np.zeros(5))
        frame={'bgr':np.ones((480,640,3),np.uint8)*127,'K':K,'distortion':np.zeros(5)}
        with patch.object(cv2,'findChessboardCorners',side_effect=[(True,pixels.copy()),(True,pixels[::-1].copy())]),\
             patch.object(cv2,'cornerSubPix',side_effect=lambda gray,c,*a:c):
            first=cm.board_pose(frame,board);second=cm.board_pose(frame,board)
        np.testing.assert_allclose(first['T_camera_board'],second['T_camera_board'],atol=1e-6)
        self.assertEqual(first['markers'],48)

    def test_small_two_axis_samples_are_identifiable_but_one_axis_is_rejected(self):
        X=cm.rigid(Rotation.from_euler('xyz',[.2,-.1,.3]).as_matrix(),[.03,-.02,.05])
        B=cm.rigid(Rotation.from_euler('xyz',[1.,.2,.1]).as_matrix(),[.45,-.1,.3])
        a=[]
        for axis in ['x','y']:
            for angle in [-.1,-.08,0.,.08,.1]:a.append(cm.rigid(Rotation.from_euler(axis,angle).as_matrix(),[.4,-.2,.5]))
        b=[np.linalg.inv(p@X)@B for p in a]
        fit=he.validate_solution(a,b)
        self.assertTrue(fit['stable']);np.testing.assert_allclose(fit['T_link_camera'],X,atol=1e-5)
        with self.assertRaisesRegex(ValueError,'one axis'):
            cm.solve_hand_eye(a[:5],b[:5])


if __name__=='__main__':unittest.main()
