"""Fault injection with fake motors/cameras only; never import a device SDK."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import sys
from unittest.mock import patch

import numpy as np

from .calibration import check
from .devices import exposure_time,LegacyArm
from .evidence import HoldEvidence
from .motion import Servo,retime
from .planning import segment_distance
from .robot import Robot,validate_observation
from .runtime import Runtime
from .service import Service,serve


def observation():
    return {'source':'measured_perception','frame':'robot_base','timestamp_unix_s':time.time(),
        'held_by':[],'touching':{'left':False,'right':False},'upright':True,'bottle_center_m':[.5,-.1,.2],
        'pose_revision':1,'gripper_motor_feedback_rad':{'left':-.9,'right':-.9},'gripper_open_rad':{'left':-.9,'right':-.9},
        'checkpoint_id':None,'current_grasp_id':None,'last_failure':None,'completed_steps':[],
        'operation':None,'remaining_seconds':120.,'terminal':False,'invalid':False,
        'hold_evidence_verified':True,'object_pose_verified':True}


class Controller:
    _activated_sides=['left','right']
    def __init__(self):self.holds=0
    def hold_last_validated_position(self):self.holds+=1


class Driver:
    def __init__(self):self.controller=Controller();self.q=np.zeros(16);self.sent=[]
    def read(self):return {'q':self.q.tolist(),'timestamp_unix_s':time.time()}
    def send(self,q,state):self.q=q.copy();self.sent.append(q.copy())


def servo(driver,record=lambda *_:None):
    limits={s:(np.full(7,-2.),np.full(7,2.)) for s in ['left','right']}
    limits.update(left_gripper=(-1.,1.),right_gripper=(-1.,1.))
    return Servo(driver,limits,record)


class AcceptPlanner:
    identity='fake-no-hardware'
    def validate_path(self,path,scene):return True


class RuntimeChecks(unittest.TestCase):
    def test_startup_rejects_unchecked_path_or_state_drift_before_motor_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model_launchers/models/dp/deploy_dp_dual_joint.py';p.parent.mkdir(parents=True)
            names=['move_arms_to_initial_position','read_joint_state','get_current_state','joints_to_state_vector',
                'parse_dp_action','initial_joint_state','log_initialization_error']
            p.write_text('\n'.join('def '+n+'(*args,**kwargs): pass' for n in names))
            for accepted,drift in [(False,0.),(True,.02)]:
                class ControllerFixture:
                    activated=False
                    def remember_validated_state(self,q):pass
                    def activate_at_current_position(self,*a,**k):self.activated=True
                class Planner:
                    def validate_path(self,path,scene):return accepted
                arm=LegacyArm({'startup_target':[0.]*16,'deploy_root':tmp},lambda *_:None)
                arm.controller=ControllerFixture();calls=0
                def read():
                    nonlocal calls
                    calls+=1;return {'q':[0. if calls==1 else drift]*16}
                arm.read=read
                limits={s:([-2.]*7,[2.]*7) for s in ['left','right']}
                with self.assertRaises(RuntimeError):arm.initialize(limits,lambda:False,planner=Planner(),scene={})
                self.assertFalse(arm.controller.activated)

    def test_official_executor_runs_provider_protocol_and_receiver_recovery(self):
        # A kinematic TEST DOUBLE, not a simulation or hardware success claim.
        # The actual Runtime tools/state machine run; only sensing/motors/IK
        # are replaced. This catches frozen-program/adapter contract drift.
        base=Path(__file__).resolve().parents[1]
        runtime=base/'app/aspire/src'
        if not runtime.is_dir():runtime=Path('/home/wsk/ASPIRE/libero-pro/source/src')
        sys.path.insert(0,str(runtime))
        from cap_harness.registry import ToolRegistry
        from cap_harness.runtime import ProgramExecutor
        program=base/'app/skill/legacy.py'
        if not program.is_file():program=base/'skills/bottle_recovery/program.py'
        class Fixture(Runtime):
            def __init__(self,x,fail_receiver):
                self.required_holds=set();self.candidates={};self.checkpoints={};self.completed=[];self.last_failure=None
                self.operation=None;self.current_grasp=None;self.held=[];self.left_closes=0;self.fail_receiver=fail_receiver
                self.record=lambda *_:None
                self.bottle=np.eye(4);self.bottle[:3,3]=[x,-.1,.3];self.basket=np.eye(4);self.basket[:3,3]=[.4,.2,.1]
                self.cal={'motion':{'right_grasps':[{'id':'g0','T_object_tool':np.eye(4).tolist()}],
                    'pregrasp_m':.06,'lift_m':.08,'T_base_handover_bottle':np.eye(4).tolist(),
                    'T_bottle_left_tool':np.eye(4).tolist(),'T_basket_bottle':np.eye(4).tolist(),
                    'collision_margin_m':.005},'support_tolerance_m':.008,
                    'objects':{'basket':{'inner_size_m':[.2,.2,.1]},'bottle':{'body_radius_m':.03}}}
                self.poses={s:np.eye(4) for s in ['left','right']}
                class Planner:
                    def tools(_,q):return copy.deepcopy(self.poses)
                class ServoFixture:
                    lock=threading.RLock();target=np.zeros(16);state={'q':[0.]*16}
                self.planner=Planner();self.servo=ServoFixture()
            def _ambient_guard(self):pass
            def observe(self):
                o=observation();o.update(held_by=list(self.held),touching={s:s in self.held for s in ['left','right']},
                    object_poses={'bottle':self.bottle.tolist(),'basket':self.basket.tolist()},
                    checkpoint_id=next(reversed(self.checkpoints),None),current_grasp_id=self.current_grasp,
                    last_failure=self.last_failure,operation=copy.deepcopy(self.operation),completed_steps=list(self.completed),
                    grasp_region_evidence={'right':{'verified':True,'source':'measured_geometry','part':'body','fraction_min':.2,'fraction_max':.4}},
                    support_evidence={'table':{'verified':True,'stationary':True},'basket':{'verified':True,'stationary':True}})
                return o
            def _move(self,targets,**kwargs):
                if self.held:
                    side='left' if self.held==['left'] else 'right'
                    if side in targets:self.bottle=targets[side]@np.linalg.inv(self.poses[side])@self.bottle
                self.poses.update(copy.deepcopy(targets))
            def _jaw(self,side,opened):
                if opened:
                    if side in self.held:self.held.remove(side)
                else:
                    if side=='left':
                        self.left_closes+=1
                        if self.fail_receiver and self.left_closes==1:return
                    if side not in self.held:self.held.append(side)
            def _wait_hold(self,side,wanted=True):return self.observe()
        try:
            for x,failed in [(.42,False),(.55,True)]:
                fixture=Fixture(x,failed);registry=ToolRegistry(shared_public_names=Runtime.tools,public_extension_allowlist=())
                for name in Runtime.tools:registry.register(getattr(fixture,name),name=name,public=True,layer='shared_high_level',capability='kinematic test double; no devices')
                result=ProgramExecutor(registry).execute_program(program.read_text())
                self.assertTrue(result.ok,str(result.error));self.assertEqual(fixture.held,[])
                self.assertEqual(fixture.completed,['handover','placement'])
                if failed:self.assertEqual(fixture.left_closes,2)
        finally:sys.path.remove(str(runtime))

    def test_slow_slip_remains_latched_after_short_window_becomes_stable(self):
        config={s:{'object_center_bounds_tool_m':[[-.2]*3,[.2]*3],
            'contact_motor_range':[-.8,-.4],'contact_torque_range':[.2,2.],
            'maximum_slip_m':.003,'maximum_rotation_rad':.03,'open_motor':-.95} for s in ['left','right']}
        hold=HoldEvidence(config)
        for i in range(181):
            stamp=1000+i/30;pose=np.eye(4);pose[0,3]=min(i,90)*.0002
            out=hold.read('right',object_pose=pose,tool_pose=np.eye(4),
                feedback={'timestamp_unix_s':stamp,'position':-.6,'torque':.5},stamp=stamp,now=stamp)
        self.assertFalse(out['held']);self.assertTrue(out['slip_latched']);self.assertGreater(out['cumulative_slip_m'],.003)

    def test_guard_failure_stops_background_commands(self):
        d=Driver();s=servo(d);calls=0
        def guard():
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('synthetic object lost')
        try:
            s.start();p=np.zeros(16);p[0]=.005
            with self.assertRaisesRegex(RuntimeError,'object lost'):
                s.follow([p,p*2],AcceptPlanner(),{'revision':1},guard=guard)
            n=len(d.sent);time.sleep(.08)
            self.assertEqual(len(d.sent),n);self.assertTrue(s.stop_event.is_set());self.assertGreaterEqual(d.controller.holds,1)
        finally:s.close()

    def test_rejected_or_ambiguous_planner_never_dispatches_path(self):
        for value in [False,None,{'ok':True}]:
            class Planner:
                identity='fake'
                def validate_path(self,path,scene):return value
            d=Driver();s=servo(d)
            try:
                s.start();p=np.zeros(16);p[0]=.005
                with self.assertRaises(ValueError):s.follow([p],Planner(),{'revision':1})
                self.assertTrue(all(np.array_equal(q,np.zeros(16)) for q in d.sent))
            finally:s.close()

    def test_failed_logger_cannot_skip_hold_attempt(self):
        class Broken(Driver):
            def read(self):raise RuntimeError('synthetic failed feedback')
        def disk_full(*_):raise OSError('synthetic disk full')
        d=Broken();s=servo(d,disk_full);s._loop()
        self.assertIsNotNone(s.failure);self.assertEqual(d.controller.holds,1)

    def test_independent_watchdog_latches_during_blocked_feedback(self):
        release=threading.Event()
        class Blocked(Driver):
            reads=0
            def read(self):
                self.reads+=1
                if self.reads>1:release.wait(1)
                return super().read()
        d=Blocked();s=servo(d)
        try:
            s.start();time.sleep(.22)
            self.assertTrue(s.stop_event.is_set());self.assertIn('watchdog',s.failure)
            release.set();s.thread.join(1);self.assertEqual(d.sent,[])
        finally:release.set();s.close()

    def test_quintic_timing_respects_velocity_acceleration_and_small_steps(self):
        start=np.zeros(16);goal=np.full(16,.4);q=np.array([start,*retime(start,goal)])
        velocity=np.diff(q,axis=0)*30;acceleration=np.diff(velocity,axis=0)*30
        self.assertLessEqual(np.max(abs(velocity)),.15+1e-6)
        self.assertLessEqual(np.max(abs(acceleration)),.3+1e-4)
        self.assertLess(np.max(abs(np.diff(q,axis=0))),.01)
        np.testing.assert_allclose(q[-1],goal)

    def test_arrival_timestamp_cannot_make_stale_exposure_fresh(self):
        model={'serial':'test','domain':'test_clock','seconds_per_ms':.001,'offset_unix_s':100.,'maximum_error_s':.005,'valid_until_unix_s':200.}
        self.assertAlmostEqual(exposure_time(1000.,'test_clock',model,serial='test',now=101.01),101.)
        for ms,domain in [(500.,'test_clock'),(1000.,'wrong')]:
            with self.assertRaises(RuntimeError):exposure_time(ms,domain,model,serial='test',now=101.01)

    def test_nonfinite_nested_or_negative_budget_observation_rejected(self):
        for key,value in [('bottle_center_m',[float('nan'),0,0]),('remaining_seconds',-1),('pose_revision',0)]:
            o=observation();o[key]=value
            with self.assertRaises(ValueError):validate_observation(o)
        o=observation();o['optional_data']={'invalid':float('inf')}
        with self.assertRaises(ValueError):validate_observation(o)

    def test_empty_gripper_dictionary_cannot_pass_preflight(self):
        path=Path(__file__).parent/'calibration.json'
        if not path.is_file():path=Path(__file__).resolve().parents[1]/'config/calibration.json'
        cal=json.loads(path.read_text())
        cal['grippers']={'left':{},'right':{}}
        with tempfile.TemporaryDirectory() as tmp:missing=check(cal,{}, {},tmp)
        self.assertIn('left complete grip calibration',missing);self.assertIn('right complete grip calibration',missing)

    def test_capsule_distance_including_degenerate_parallel_and_crossing(self):
        self.assertAlmostEqual(segment_distance([0,0,0],[1,0,0],[.5,-1,0],[.5,1,0]),0.)
        self.assertAlmostEqual(segment_distance([0,0,0],[1,0,0],[0,1,0],[1,1,0]),1.)
        self.assertAlmostEqual(segment_distance([0,0,0],[0,0,0],[1,0,0],[2,0,0]),1.)

    def test_all_declared_real_tools_have_provider_methods(self):
        self.assertTrue(all(callable(getattr(Runtime,name,None)) for name in Runtime.tools))

    def test_http_token_exclusive_session_and_abort(self):
        class Journal:
            def record(self,*_):pass
        class Provider:
            tools=Runtime.tools;stopped=False
            def observe(self):return observation()
            def call(self,*args,**kwargs):return {'candidates':[]}
            def stop(self,reason):self.stopped=True
        provider=Provider();service=Service({'robot_contract':'openarm-body-v4/1'},
            {'calibration_verified':True,'missing':[]},Journal(),provider)
        server=serve(service,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.runtime').mkdir();token=root/'.runtime/session-token';token.write_text(service.token);token.chmod(0o600)
            base='http://127.0.0.1:'+str(server.server_port)
            try:
                with patch(Robot.__module__+'.ROOT',root):
                    a=Robot({'robot_service':base},execute=True);b=Robot({'robot_service':base},execute=True)
                    a.begin()
                    with self.assertRaises(RuntimeError):b.begin()
                    self.assertEqual(a.call('propose_grasps'),{'candidates':[]})
                    a.end();b.begin();b.abort()
                    self.assertTrue(provider.stopped);self.assertFalse(a.status()['ready']);b.end()
            finally:
                server.shutdown();server.server_close();thread.join(2)
                # Some target hosts omit dot-directories from enumeration.
                # These are the exact test-owned paths, never a broad deletion.
                if token.is_file():token.unlink()
                token.parent.rmdir()


if __name__=='__main__':unittest.main()
