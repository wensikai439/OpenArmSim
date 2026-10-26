"""No-device regressions for startup, coarse/fine reach, effort and failure logs."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import numpy as np

from . import robot as client
from .motion import retime
from .runtime import Runtime
from .relative import goal_error
from .relative_checks import Fixture,pose
from .service import setup_groups
from . import service
try:
    from .runtime_checks import Driver,servo,AcceptPlanner
except ImportError:
    from .checks import Driver,servo,AcceptPlanner


class MotionChecks(unittest.TestCase):
    def ready_fixture(self):
        f=Fixture();f.hardware={'startup_target':[0.]*16}
        f.servo.state={'q':[0.]*16,'timestamp_unix_s':time.time()}
        f.servo.target=np.zeros(16)
        return f

    def test_every_session_rechecks_measured_act_pose_and_keeps_old_anchor_on_rejection(self):
        f=self.ready_fixture();f.begin_session();anchor=copy.deepcopy(f.start_tools)
        f.servo.state={'q':[.1]+[0.]*15,'timestamp_unix_s':time.time()}
        f.poses['right']=pose(.9,-.1,.2)
        with self.assertRaisesRegex(RuntimeError,'EACH task'):f.begin_session()
        np.testing.assert_array_equal(anchor['right'],f.start_tools['right'])
    def test_stale_feedback_contact_or_pending_target_cannot_start_task(self):
        for reason in ['stale','contact','pending']:
            f=self.ready_fixture()
            if reason=='stale':f.servo.state['timestamp_unix_s']-=.2
            elif reason=='contact':f.held=['right']
            else:f.servo.target[0]=.1
            with self.assertRaises(RuntimeError):f.begin_session()

    def test_coarse_reach_converges_then_fine_steps_remain_bounded(self):
        for distance in [.03,.06,.08,.12]:
            f=Fixture();f.poses['right']=pose(0,0,0);target=pose(0,0,distance)
            fine=0;done=False
            for _ in range(8):
                before=f.poses['right'].copy();done=f._relative_step('right',target)
                if done:break
                event=[v for k,v in f.events if k=='relative_goal'][-1]
                if event['phase']=='fine':
                    fine+=1;self.assertLessEqual(goal_error(before,f.poses['right'])[0],.006000001)
            self.assertTrue(done);self.assertLessEqual(len(f.moves),5);self.assertGreater(fine,0)
            self.assertLessEqual(goal_error(f.poses['right'],target)[0],.002)
    def test_coarse_fine_tolerance_band_avoids_zero_length_replanning_loop(self):
        f=Fixture();f.poses['right']=pose(0,0,0);target=pose(0,0,.018000000000000002)
        for _ in range(8):
            if f._relative_step('right',target):break
        self.assertLess(goal_error(f.poses['right'],target)[0],.002)

    def test_sampled_reference_velocity_acceleration_jerk_respect_limits(self):
        for distance in [.001,.01,.1,.4]:
            start=np.zeros(16);goal=start.copy();goal[8]=distance
            points=np.array([start]*4+retime(start,goal,velocity=.1,acceleration=.2,jerk=2.)+[goal]*4)
            for order,bound in [(1,.1),(2,.2),(3,2.)]:
                derivative=np.diff(points,n=order,axis=0)*30**order
                self.assertLessEqual(np.max(abs(derivative)),bound+1e-6)
            np.testing.assert_allclose(points[-1],goal)
    def test_contact_stop_discards_remaining_closing_targets_but_keeps_hold_command(self):
        d=Driver();s=servo(d);read_count=0
        def held():
            nonlocal read_count
            read_count+=1;return read_count>=2
        try:
            s.start();points=[]
            for value in [.01,.02,.03]:
                q=np.zeros(16);q[15]=value;points.append(q)
            s.follow(points,AcceptPlanner(),{'revision':1},until=held)
            self.assertAlmostEqual(s.target[15],.01)
            self.assertFalse(any(q[15]>.010001 for q in d.sent))
        finally:s.close()
    def test_fresh_over_effort_feedback_prevents_next_send(self):
        class EffortDriver(Driver):
            def read(self):return {'q':self.q.tolist(),'timestamp_unix_s':time.time(),'torque':2.}
        d=EffortDriver();s=servo(d)
        def guard():
            if s.state and s.state.get('torque',0)>1.:raise RuntimeError('effort ceiling')
        s.guard=guard
        try:
            s.start();s.thread.join(1)
            self.assertIn('effort ceiling',s.failure);self.assertEqual(d.sent,[])
        finally:s.close()
    def test_runtime_effort_guard_applies_even_without_required_hold(self):
        r=object.__new__(Runtime);r.failure=None;r.stop_event=threading.Event();r.lock=threading.RLock()
        r.snapshot={'timestamp_unix_s':time.time(),'held_by':[]};r.required_holds=set()
        r.cal={'grippers':{s:{'contact_torque_range':[.1,1.]} for s in ['left','right']}}
        class State:pass
        r.servo=State();r.servo.state={'timestamp_unix_s':time.time(),
            'motors':{s:[{'torque':0.}]*7+[{'torque':2. if s=='right' else 0.}] for s in ['left','right']}}
        with self.assertRaisesRegex(RuntimeError,'effort ceiling'):r._ambient_guard()
    def test_carried_object_uses_exposure_aligned_tool_not_latest_robot_pose(self):
        f=Fixture();f.held=['right']
        f.cal.update(table_top_z_m=0.,objects={
            'bottle':{'body_bottom_m':0.,'body_height_m':.2,'body_radius_m':.03},
            'basket':{'inner_size_m':[.2,.2,.15],'floor_z_m':0.,'wall_radius_m':.01}})
        f.cal['motion']['contact_links']={'left':[],'right':[]}
        state=f.observe();stamp=state['timestamp_unix_s']
        aligned=pose(.4,-.12,.3)
        state['exposure_aligned_tools']={'right':{'T_base_tool':aligned.tolist(),'timestamp_unix_s':stamp}}
        f.poses['right']=pose(.9,-.12,.3)
        scene=Runtime.scene(f,state)
        self.assertAlmostEqual(scene['carried']['T_tool_object'][0][3],.1)
        state['exposure_aligned_tools']['right']['timestamp_unix_s']-=.006
        with self.assertRaises(RuntimeError):Runtime.scene(f,state)

    def test_original_error_and_last_evidence_survive_abort_and_cleanup_failure(self):
        class Broken:
            execute=True;tools=('observe','propose_grasps');motion_requests_attempted=0;aborted=False
            last_observation={'tag':'last valid before stop'}
            def status(self):return {'ready':True,'missing':[]}
            def observe(self):
                if self.aborted:raise RuntimeError('SECONDARY stopped observation')
                return copy.deepcopy(self.last_observation)
            def call(self,name,*a,**k):
                if name=='observe':return self.observe()
                raise RuntimeError('ORIGINAL proposal failure')
            def begin(self):pass
            def abort(self):self.aborted=True
            def end(self):raise RuntimeError('SECONDARY cleanup failure')
        base=Path(__file__).resolve().parents[1];official=base/'app/aspire/src'
        if not official.is_dir():official=Path('/home/wsk/ASPIRE/libero-pro/source/src')
        sys.path.insert(0,str(official))
        try:
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);(root/'app/skill').mkdir(parents=True);(root/'config').mkdir();(root/'records').mkdir()
                program=b'observe()\npropose_grasps()\n';(root/'app/skill/program.py').write_bytes(program)
                (root/'config/robot.json').write_text(json.dumps({'program_sha256':hashlib.sha256(program).hexdigest()}))
                with patch.object(client,'ROOT',root):result=client.run_skill(Broken(),root/'records')
                self.assertIn('ORIGINAL proposal failure',result['error']);self.assertIsNone(result['task_success'])
                self.assertEqual(result['last_valid_observation']['tag'],'last valid before stop')
                self.assertIsNone(result['final_observation']);self.assertIn('SECONDARY',result['cleanup_errors'][0])
                self.assertTrue((root/'records/skill-result.json').is_file())
        finally:sys.path.remove(str(official))
    def test_diagnostic_groups_preserve_all_missing_evidence(self):
        missing=['table geometry','camera extrinsic transform','left complete grip calibration',
            'camera_extrinsics bound physical acceptance','native CAN extension missing/changed']
        groups=setup_groups(missing)
        self.assertCountEqual([v for g in groups.values() for v in g['missing']],missing)
        self.assertIn('physical_acceptance',groups);self.assertIn('software',groups)
    def test_missing_acceptance_is_reported_before_any_device_provider_import(self):
        readiness={'calibration_verified':False,'missing':['camera extrinsic transform'],
            'setup_groups':setup_groups(['camera extrinsic transform'])}
        output=io.StringIO()
        with patch.object(service,'verify'),patch.object(service,'preflight',return_value=readiness),\
                patch.object(sys,'argv',['service','service','--devices','--confirm-on-site']),redirect_stdout(output):
            self.assertEqual(service.main(),2)
        result=json.loads(output.getvalue());self.assertFalse(result['devices_started'])
        self.assertEqual(result['motion_commands'],0);self.assertIn('perception',result['setup_groups'])


if __name__=='__main__':unittest.main()
