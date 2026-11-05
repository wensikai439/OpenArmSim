"""Projection estimation, missing evidence, and safe handover program checks."""
import ast
from pathlib import Path
import unittest

import numpy as np

from .runtime import ROOT, ProgramExecutor, ToolRegistry
from .visual_geometry import fit_camera, project, triangulate, basket_opening
from .visual_relay import smooth_joint_path


class RelayChecks(unittest.TestCase):
    def test_smooth_path_actual_sample_limits_and_rest_boundaries(self):
        table=np.array([[0,0,0],[.15,.03,-.02],[.22,.13,.01],[.35,.16,.10]])
        samples,profile=smooth_joint_path(table,velocity=.6,acceleration=2.,jerk=20.)
        np.testing.assert_allclose(samples[-1],table[-1],atol=1e-12)
        q=np.vstack([np.repeat(table[:1],3,axis=0),samples,np.repeat(table[-1:],3,axis=0)])
        for order,limit in ((1,.6),(2,2.),(3,20.)):
            derivative=np.diff(q,n=order,axis=0)*30**order
            self.assertLess(np.max(abs(derivative)),limit*1.08)
        self.assertGreater(profile['seconds'],.2)

    def test_duplicate_and_stationary_waypoints(self):
        table=np.array([[.1,-.2],[.1,-.2],[.2,.1],[.2,.1]])
        q,_=smooth_joint_path(table)
        self.assertTrue(np.isfinite(q).all())
        np.testing.assert_allclose(q[-1],table[-1])
        still,profile=smooth_joint_path(np.repeat(table[:1],4,axis=0))
        np.testing.assert_allclose(still,np.repeat(table[:1],6,axis=0))
        self.assertTrue(profile['stationary'])

    def test_smooth_path_preserves_waypoint_envelope_at_extrema(self):
        table=np.array([[0,.3],[.19,.31],[.2,-.15],[.195,-.2],[.199,-.19]])
        samples,_=smooth_joint_path(table)
        self.assertTrue(np.all(samples>=np.min(table,axis=0)-1e-12))
        self.assertTrue(np.all(samples<=np.max(table,axis=0)+1e-12))

    def test_noisy_independent_projection_and_triangulation(self):
        rng=np.random.default_rng(2);xyz=rng.normal(size=(120,3))*.1+[.45,0,1.3]
        truth=[np.array([[700,0,300,10],[0,750,240,-100],[.1,.2,1,0.]]),
               np.array([[600,200,340,-40],[-100,650,250,-120],[-.2,.15,1,0.]])]
        inferred=[]
        for camera in truth:
            fit,quality=fit_camera(xyz,project(camera,xyz)+rng.normal(0,.12,(len(xyz),2)))
            inferred.append(fit);self.assertLess(quality['p95_px'],.5)
        target=np.array([.48,.25,1.1]);pixels=np.concatenate([project(p,[target])[0] for p in truth])
        recovered,error=triangulate(inferred,pixels)
        self.assertLess(np.linalg.norm(recovered-target),.005)
        self.assertLess(error,1.)

    def test_inconsistent_multiview_targets_rejected(self):
        p1=np.array([[800,0,640,0],[0,800,480,0],[0,0,1,0.]])
        p2=p1.copy();p2[0,3]=-80
        with self.assertRaises(ValueError):triangulate([p1,p2],[640,480,620,650])

    def test_missing_basket_and_insufficient_robot_samples_rejected(self):
        with self.assertRaises(ValueError):basket_opening(np.full((960,1280,3),200,np.uint8))
        with self.assertRaises(ValueError):fit_camera(np.ones((4,3)),np.ones((4,2)))

    def test_missing_receiver_evidence_prevents_donor_release(self):
        state={'left':[.39,.36,1.24],'right':[.5,-.21,1.14]};motors={'left':-1.26,'right':-1.26};calls=[]
        def observe():
            return {'image_targets':{'bottle_lower':[600,500,650,500]},'robot':{
                side:{'tcp_m':list(state[side]),'gripper_motor_rad':motors[side]} for side in state}}
        def grip(arm,opened):
            calls.append(('gripper',arm,opened));motors[arm]=-1.26 if opened else -.5
            return {'motor_rad':motors[arm]}
        def move(arm,position_m,**kwargs):state[arm]=list(position_m);return {'ok':True}
        def relative(arm,delta_m):state[arm]=(np.array(state[arm])+delta_m).tolist();return {'ok':True}
        functions={'observe':observe,'set_gripper':grip,'move_tcp':move,'move_relative':relative,
            'grip_evidence':lambda arm:{'blocked_closure':False},
            'approach_options':lambda **kw:{'ok':True,'pre_m':kw['target_m'],'target_m':kw['target_m'],'yaw_deg':-85,'pitch_deg':-5},
            **{name:(lambda **kw:{'ok':True}) for name in ('label','orient','visual_align','wait','check_grasp')}}
        registry=ToolRegistry(shared_public_names=tuple(functions),public_extension_allowlist=())
        for name,fn in functions.items():registry.register(fn,name=name,public=True,layer='shared_atomic',capability='test response')
        result=ProgramExecutor(registry).execute_program((ROOT/'skills/bottle_visual/relay/program.py').read_text())
        self.assertFalse(result.ok)
        # The initial open is allowed; there must be no release after closing.
        right=[c[2] for c in calls if c[:2]==('gripper','right')]
        self.assertEqual(right,[True,False])

    def test_no_legacy_motion_or_simulator_camera_reads(self):
        forbidden={'pick','handover','place','step_tamp','TampPlanner','StableWorld','SegmentPlanner'}
        for path in ('harness/aspire/visual_relay.py','harness/aspire/visual_geometry.py','skills/bottle_visual/relay/program.py'):
            tree=ast.parse((ROOT/path).read_text())
            calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
                   if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
            self.assertFalse(calls&forbidden)
            attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
            self.assertFalse(attrs&{'cam_xpos','cam_xmat','cam_intrinsic'})

    def test_shared_home_plane_and_bounded_tracking_recovery(self):
        for error in ('TCP tracking error exceeds 8mm','native termination'):
            state={'left':[.39,.359,1.24],'right':[.335,-.406,1.18]}
            motors={'left':-1.26,'right':-1.26};targets=[];labels=[];aligned=[]
            def observe():
                return {'image_targets':{'bottle_lower':[600,500,650,500]},'robot':{
                    side:{'tcp_m':list(state[side]),'gripper_motor_rad':motors[side]} for side in state}}
            def align(**kwargs):
                aligned.append(True)
                if len(aligned)==2:state['right']=[.5,-.188,1.06]
                return {'ok':True}
            def grip(arm,opened):motors[arm]=-1.26 if opened else -.5;return {'motor_rad':motors[arm]}
            def move(arm,position_m,**kwargs):
                if arm=='right':
                    targets.append(list(position_m));state[arm]=list(position_m)
                    if len(targets)==1:raise ValueError(error)
                else:state[arm]=list(position_m)
                return {'ok':True}
            def relative(arm,delta_m):state[arm]=(np.array(state[arm])+delta_m).tolist();return {'ok':True}
            def label(name):labels.append(name);return {'ok':True}
            functions={'observe':observe,'set_gripper':grip,'move_tcp':move,'move_relative':relative,
                'label':label,'visual_align':align,'grip_evidence':lambda arm:{'blocked_closure':arm=='right'},
                'approach_options':lambda **kw:{'ok':True,'pre_m':kw['target_m'],'target_m':kw['target_m'],'yaw_deg':-85,'pitch_deg':-5},
                **{name:(lambda **kw:{'ok':True}) for name in ('orient','wait','check_grasp')}}
            registry=ToolRegistry(shared_public_names=tuple(functions),public_extension_allowlist=())
            for name,fn in functions.items():registry.register(fn,name=name,public=True,layer='shared_atomic',capability='test response')
            result=ProgramExecutor(registry).execute_program((ROOT/'skills/bottle_visual/relay/program.py').read_text())
            self.assertFalse(result.ok)  # Receiver is deliberately unconfirmed.
            home_mid=(.359-.406)/2
            self.assertAlmostEqual(targets[0][1],(-.188+home_mid)/2)
            if error.startswith('TCP'):
                self.assertEqual(len(targets),2)  # No repeated lift/transport.
                self.assertAlmostEqual(targets[1][1],home_mid)
                self.assertIn('transport_replan_from_actual',labels)
            else:
                self.assertEqual(len(targets),1)
                self.assertNotIn('transport_replan_from_actual',labels)


if __name__=='__main__':unittest.main()
