"""No-device checks for failure gates, geometry and durable audit records."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import numpy as np

from .devices import Camera,LegacyArm
from .evidence import HoldEvidence,Kinematics,grasp_region,relative_target,transform
from .service import Journal,Service,audit_log,serve
from .robot import Robot
from .motion import Servo


class MeasuredEvidenceChecks(unittest.TestCase):
    def test_relative_target_follows_object_rotation_and_translation(self):
        obj=np.array([[0,-1,0,.5],[1,0,0,-.2],[0,0,1,.1],[0,0,0,1.]])
        offset=np.eye(4);offset[0,3]=.03
        np.testing.assert_allclose(relative_target(obj,offset)[:3,3],[.5,-.17,.1])
        invalid=np.eye(4);invalid[0,0]=2
        with self.assertRaises(ValueError):transform(invalid)

    def test_urdf_fk_uses_joint_names_and_origins(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'robot.urdf'
            p.write_text('<robot name="r"><link name="base"/><link name="hand"/>'
                '<joint name="joint" type="revolute"><parent link="base"/><child link="hand"/>'
                '<origin xyz="0 0 1"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/></joint></robot>')
            t=Kinematics(p).pose('hand',{'joint':np.pi/2})
            np.testing.assert_allclose(t[:3,3],[0,0,1]);np.testing.assert_allclose(t[:2,:2],[[0,-1],[1,0]],atol=1e-8)

    def hold(self):
        return HoldEvidence({side:{'object_center_bounds_tool_m':[[-.1]*3,[.1]*3],
            'contact_motor_range':[-.8,-.4],'contact_torque_range':[.2,2.],
            'maximum_slip_m':.003,'maximum_rotation_rad':.03} for side in ['left','right']})

    def sample(self,h,stamp,torque=.5,x=0.):
        obj=np.eye(4);obj[0,3]=x
        return h.read('right',object_pose=obj,tool_pose=np.eye(4),
            feedback={'position':-.6,'torque':torque,'timestamp_unix_s':stamp},stamp=stamp,now=stamp+.01)

    def test_closed_jaw_without_effort_or_duration_is_not_hold(self):
        h=self.hold()
        self.assertFalse(self.sample(h,100.)['held'])
        self.assertFalse(self.sample(h,100.04,torque=0.)['held'])
        for s in [100.08,100.12,100.16,100.20]:r=self.sample(h,s)
        self.assertTrue(r['held'])

    def test_duplicate_cached_samples_and_slip_do_not_accumulate_hold(self):
        h=self.hold()
        for _ in range(20):r=self.sample(h,100.)
        self.assertFalse(r['held'])
        for s,x in [(100.04,0.),(100.08,0.),(100.12,.01)]:r=self.sample(h,s,x=x)
        self.assertFalse(r['held'])

    def test_region_includes_full_extent_and_uncertainty(self):
        pads=[[x,y,z] for x in [-.01,.01] for y in [-.01,.01] for z in [.04,.06]]
        r=grasp_region(np.eye(4),np.eye(4),height_m=.2,body_bottom_m=0.,pad_points_tool=pads,uncertainty_m=.01)
        self.assertAlmostEqual(r['fraction_min'],.15);self.assertAlmostEqual(r['fraction_max'],.35)

    def test_device_construction_and_rejected_open_never_import_sdk(self):
        import sys
        before=set(sys.modules)
        arm=LegacyArm({'startup_target':[0.]*16},lambda *_:None)
        cam=Camera({},lambda *_:None)
        for device in [arm,cam]:
            with self.assertRaises(RuntimeError):device.open()
        self.assertFalse({'pyrealsense2','openarm_can'} & (set(sys.modules)-before))


class RealServiceChecks(unittest.TestCase):
    def test_servo_continues_both_arm_hold_while_api_is_idle(self):
        class Controller:
            _activated_sides=['left','right']
            def hold_last_validated_position(self):pass
        class Driver:
            controller=Controller()
            q=np.zeros(16)
            def __init__(self):self.sent=[]
            def read(self):return {'q':self.q.tolist(),'timestamp_unix_s':time.time()}
            def send(self,q,state):self.q=q.copy();self.sent.append(q.copy())
        class Planner:
            identity='offline_test_fixture'
            def validate_path(self,path,scene):
                self.called=True
                return True
        driver=Driver();planner=Planner()
        limits={s:(np.full(7,-2.),np.full(7,2.)) for s in ['left','right']}
        limits.update(left_gripper=(-1.,1.),right_gripper=(-1.,1.))
        servo=Servo(driver,limits,lambda *_:None)
        try:
            servo.start();points=[]
            for x in [.005,.01,.015]:
                q=np.zeros(16);q[8]=x;points.append(q)
            servo.follow(points,planner,{'revision':1})
            n=len(driver.sent);time.sleep(.14)
            self.assertGreaterEqual(len(driver.sent),n+3)
            self.assertTrue(planner.called)
            for p in points:self.assertTrue(any(np.array_equal(p,q) for q in driver.sent))
            self.assertIsNone(servo.failure)
            with self.assertRaises(ValueError):servo.follow([np.full(16,.5)],planner,{'revision':1})
        finally:servo.close()

    def test_journal_chain_detects_modification_and_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            j=Journal(Path(folder)/'record',{'mode':'offline_fixture','api_key':'never-log'})
            j.record('command_requested',{'target':[1.],'authorization':'secret'});j.close()
            self.assertEqual(audit_log(j.folder)['events'],3)
            p=j.folder/'events.jsonl';text=p.read_text()
            self.assertNotIn('never-log',text);self.assertNotIn('secret',text)
            p.write_text(text.replace('command_requested','command_changed'))
            with self.assertRaises(ValueError):audit_log(j.folder)

    def test_disabled_http_service_refuses_live_observations_and_posts(self):
        with tempfile.TemporaryDirectory() as folder:
            j=Journal(Path(folder)/'run',{'mode':'offline_fixture'})
            service=Service({'robot_contract':'openarm-body-v4/1'},
                {'calibration_verified':False,'missing':['physical acceptance missing']},j)
            server=serve(service,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base='http://127.0.0.1:'+str(server.server_port)
            try:
                self.assertFalse(Robot({'robot_service':base},execute=True).status()['ready'])
                with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(base+'/observation')
                req=urllib.request.Request(base+'/tools/lift',data=b'{}',headers={'Content-Type':'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as exc:urllib.request.urlopen(req)
                self.assertEqual(exc.exception.code,403)
                with self.assertRaises(ValueError):service.tool('lift',{'request_id':'x'})
                self.assertEqual(service.calls,0)
            finally:server.shutdown();server.server_close();thread.join(2);j.close()


if __name__=='__main__':unittest.main()
