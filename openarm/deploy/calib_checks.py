"""No-device checks for calibration, cuRobo planning and online task poses.
Synthetic data only; none of this is a physical trial."""
import json
from pathlib import Path
import time
import types
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import calib
from . import calib_math as cm
from .calibration import check
from .planning import Planner

ROOT=Path(__file__).resolve().parents[1]


def pose(p,r=None):
    t=np.eye(4);t[:3,3]=p
    if r is not None:t[:3,:3]=r
    return t


class MathChecks(unittest.TestCase):
    def test_marker_frames_follow_printed_up_arrow_and_outward_normal(self):
        obj=pose([.4,-.2,.3],R.from_euler('z',.7).as_matrix())
        marker=obj@np.linalg.inv(cm.side_marker(.1,.035))
        np.testing.assert_allclose(marker[:3,3],obj[:3,3]+obj[:3,0]*.035+[0,0,.1],atol=1e-12)
        np.testing.assert_allclose(marker[:3,1],[0,0,1],atol=1e-12)
        np.testing.assert_allclose(marker[:3,2],obj[:3,0],atol=1e-12)
        top=obj@np.linalg.inv(cm.top_marker(.2));np.testing.assert_allclose(top[:3,2],[0,0,1],atol=1e-12)

    def test_printed_sheets_are_detectable_and_carry_print_resolution(self):
        import cv2
        pages=cm.marker_sheets(dpi=150);self.assertEqual(len(pages),2)
        self.assertIn(b'pHYs',cm.png_dpi(pages[0],150)[:80])
        _,d=cm.aruco();det=cv2.aruco.ArucoDetector(d,cv2.aruco.DetectorParameters())
        ids=set(det.detectMarkers(pages[0])[1].flatten());self.assertEqual(ids,set(range(10,22)))
        self.assertEqual(set(det.detectMarkers(pages[1])[1].flatten()),{cm.BOTTLE_ID,cm.BASKET_ID})

    def test_grip_thresholds_separate_contact_from_air_and_reject_overlap(self):
        g=cm.grip('right',open_motor=-.94,limits=(-1.,.05),free=[.01,.02],empty={'position':.05,'torque':.01},
            stall={'position':-.40,'torque':.14},squeeze=[{'position':-.41,'torque':.9}]*5,center_tool=[0,0,-.02])
        self.assertLess(g['contact_torque_range'][0],.14);self.assertGreater(g['contact_torque_range'][1],.9)
        self.assertFalse(g['contact_motor_range'][0]<=.05<=g['contact_motor_range'][1])
        with self.assertRaisesRegex(ValueError,'overlap'):
            cm.grip('left',open_motor=-.9,limits=(-1.,.05),free=[.01],empty={'position':-.38,'torque':.2},
                stall={'position':-.40,'torque':.14},squeeze=[{'position':-.41,'torque':.9}],center_tool=[0,0,0])

    def test_clock_fit_recovers_exposure_offset_from_moving_views(self):
        rng=np.random.default_rng(3);board=pose([.5,0,.3],R.from_euler('y',-1.2).as_matrix())
        def camera(q):return pose([0,0,.4],R.from_euler('z',q[0]).as_matrix()@R.from_euler('y',1.2).as_matrix())
        t0=1.8e9;times=t0+np.arange(0,6,1/30);truth=-.004  # exposure = device clock + truth
        feedback=[(t,np.array([.25*np.sin(1.3*(t-t0))])) for t in np.arange(t0-.5,t0+6.5,1/100)]
        views=[];device=[];arrival=[]
        for t in times:
            d=(t-truth)*1000;cam=camera([.25*np.sin(1.3*(t-t0))]);rel=np.linalg.inv(cam)@board
            noise=R.from_rotvec(rng.normal(0,.0015,3)).as_matrix()
            views.append({'device_ms':d,'R_camera_board':rel[:3,:3]@noise});device.append(d);arrival.append(t+.025+rng.uniform(0,.01))
        clock=cm.clock_envelope(device,arrival)
        fit=cm.time_offset(views,feedback,camera,board,clock=clock)
        # Offset and slope are one model: compare predicted exposure instants.
        for d,t in [(device[0],times[0]),(device[-1],times[-1])]:
            self.assertLess(abs(d*clock['seconds_per_ms']+fit['offset_unix_s']-t),max(.004,fit['maximum_error_s']))
        self.assertLess(fit['maximum_error_s'],.02)

    def test_jaw_sweep_and_camera_hull_load_into_planner(self):
        hw=json.loads((ROOT/'config/hardware.json').read_text())
        if not Path(hw['urdf']).is_file() or not (ROOT/'config/collision.json').is_file():self.skipTest('robot URDF not on this host')
        col=json.loads((ROOT/'config/collision.json').read_text());cal=json.loads((ROOT/'config/calibration.json').read_text())
        cm.jaw_sweeps(col,hw['urdf']);cm.camera_hull(col,'openarm_right_hand',pose([.02,0,.06]),[.1,.045,.045])
        col['full_jaw_sweep_verified']=True
        self.assertNotIn('measured full-range finger collision volumes',check(cal,hw,col,ROOT))
        p=Planner(hw,calib.session_calibration(cal),col);q=np.array(hw['startup_target'])
        scene={'revision':0,'timestamp_unix_s':time.time(),'table_z_m':-10,'obstacles':[],'carried':None,'contact_links':[],
            'T_base_basket':np.eye(4).tolist(),'allow_basket_support':False}
        self.assertTrue(p.valid(q,scene))
        far=pose([.02,0,.06]);far[:3,3]=[0,-.3,0]   # a camera hull through the forearm must be rejected
        cm.camera_hull(col,'openarm_right_hand',far,[.1,.045,.045]);self.assertFalse(Planner(hw,calib.session_calibration(cal),col).valid(q,scene))

    def test_curobo_paths_pass_fcl_recheck_and_servo_limits(self):
        hw=json.loads((ROOT/'config/hardware.json').read_text())
        if not (ROOT/'config/curobo.yml').is_file() or not Path(hw['urdf']).is_file():self.skipTest('cuRobo config not on this host')
        from .cumotion import CuMotion
        from .runtime import Runtime
        col=json.loads((ROOT/'config/collision.json').read_text());cm.jaw_sweeps(col,hw['urdf'])
        cal=calib.session_calibration(json.loads((ROOT/'config/calibration.json').read_text()),.28)
        cal['objects']['bottle'].update(body_height_m=.2,body_bottom_m=0.,body_radius_m=.032)
        cal['objects']['basket'].update(inner_size_m=[.16,.2,.1],floor_z_m=0.,wall_radius_m=.003)
        p=Planner(hw,cal,col);q=np.array(hw['startup_target']);start=p.tools(q)
        bottle=pose([.28,-.22,.28],R.from_euler('z',np.pi).as_matrix());basket=pose([.42,.20,.285],R.from_euler('z',np.pi).as_matrix())
        def scene(qq,carried=None,contact=()):
            tools=p.tools(qq);now=time.time();b=bottle if carried is None else tools[carried['arm']]@np.array(carried['T_tool_object'])
            state={'object_poses':{'bottle':b.tolist(),'basket':basket.tolist()},'held_by':[] if carried is None else [carried['arm']],
                'pose_revision':1,'timestamp_unix_s':now,'exposure_aligned_tools':{s:{'T_base_tool':t.tolist(),'timestamp_unix_s':now} for s,t in tools.items()}}
            return Runtime.scene(types.SimpleNamespace(cal=cal,operation=None),state,contact=contact)
        target=pose([.30,-.04,.50],R.from_euler('z',-.4).as_matrix()@start['left'][:3,:3])
        t0=time.time();p.global_planner=CuMotion(ROOT,hw,cal);p.global_planner.warm(q,.28);print('cuRobo warm-up %.1fs'%(time.time()-t0))
        t0=time.time()
        path=p._global(q,{'left':target},scene(q));print('cuRobo left reach %.1fs, %d ticks'%(time.time()-t0,len(path)))
        self.assertLess(np.linalg.norm(p.tools(path[-1])['left'][:3,3]-target[:3,3]),.001)
        t0=time.time();back=p._global(path[-1],{'left':start['left']},scene(path[-1]));took=time.time()-t0
        print('cuRobo warm plan %.1fs'%took);self.assertLess(took,15.)
        np.testing.assert_allclose(path[-1][8:],q[8:])                         # the other arm stays locked
        self.assertLessEqual(max(np.max(abs(np.diff([q,*path],axis=0)),axis=1)),.01)  # servo step limit
        # Carrying, after the lift: the sphere cover protrudes ~1 radius past the
        # bottle ends, so global moves start clear of supports (lifts stay local).
        cal['table_top_z_m']=.20
        grasp=np.linalg.inv(start['right'])@pose(start['right'][:3,3]+[0,0,-.06]);carried={'arm':'right','T_tool_object':grasp.tolist()}
        up=start['right'].copy();up[:3,3]+=[.15,.10,.08];t0=time.time()
        path=p._global(q,{'right':up},scene(q,carried,('right',)));print('cuRobo carried move %.1fs'%(time.time()-t0))
        self.assertLess(np.linalg.norm(p.tools(path[-1])['right'][:3,3]-up[:3,3]),.001)


    def test_autoplan_finds_an_executable_task_without_moving(self):
        hw=json.loads((ROOT/'config/hardware.json').read_text())
        if not (ROOT/'config/curobo.yml').is_file() or not Path(hw['urdf']).is_file():self.skipTest('cuRobo config not on this host')
        from .cumotion import CuMotion
        from .autoplan import Search
        col=json.loads((ROOT/'config/collision.json').read_text());cm.jaw_sweeps(col,hw['urdf'])
        cal=calib.session_calibration(json.loads((ROOT/'config/calibration.json').read_text()),.28)
        cal['objects']['bottle'].update(body_height_m=.2,body_bottom_m=0.,body_radius_m=.032)
        cal['objects']['basket'].update(inner_size_m=[.16,.2,.1],floor_z_m=0.,wall_radius_m=.003)
        p=Planner(hw,cal,col);p.global_planner=CuMotion(ROOT,hw,cal);q=np.array(hw['startup_target']);p.global_planner.warm(q,.28)
        objects={'bottle':pose([.28,-.22,.28],R.from_euler('z',np.pi).as_matrix()),'basket':pose([.42,.20,.285],R.from_euler('z',np.pi).as_matrix())}
        t0=time.time();r=Search(p,cal,q,objects).run();print('autoplan %.0fs'%(time.time()-t0))
        self.assertLessEqual(np.linalg.norm(r['handover_offset_m']),.3);self.assertGreaterEqual(np.linalg.norm(r['donor_retreat_base_m']),.01)
        self.assertLess(abs(np.asarray(r['T_basket_bottle'])[2,3]-.002),1e-6)

if __name__=='__main__':unittest.main()
