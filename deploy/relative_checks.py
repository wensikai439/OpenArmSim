"""Relative-goal and failure-path tests. Test doubles are NOT physical trials."""
import copy
from pathlib import Path
import sys
import threading
import tempfile
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from .evidence import transform
from .relative import RelativeRuntime,bounded_target,goal_error,transfer_goal,CONTRACT
from .robot import Robot,RELATIVE_TOOLS
from .calibration import check,implementation
try:
    from .runtime_checks import observation
except ImportError:
    from .checks import observation


def pose(x=0.,y=0.,z=0.):
    t=np.eye(4);t[:3,3]=[x,y,z];return t


class Fixture(RelativeRuntime):
    """Only kinematic state/hold doubles; no physics, driver, SDK or real scoring."""
    def __init__(self,x=.5,fail_close=False,fail_receiver=False):
        self.required_holds=set();self.candidates={};self.checkpoints={};self.completed=[]
        self.last_failure=None;self.operation=None;self.current_grasp=None;self.held=[]
        self.right_closes=0;self.left_closes=0;self.fail_close=fail_close;self.fail_receiver=fail_receiver
        self.events=[];self.moves=[];self.reject_motion=False;self.record=lambda k,v:self.events.append((k,v))
        self.bottle=pose(x,-.12,.3);self.basket=pose(.44,.2,.1)
        self.poses={'left':pose(.4,.2,.35),'right':pose(.4,-.2,.35)}
        self.cal={'motion':{'right_grasps':[{'id':'g0','T_object_tool':np.eye(4).tolist()}],
            'pregrasp_m':.06,'lift_m':.08,'T_bottle_left_tool':np.eye(4).tolist(),
            'T_basket_bottle':np.eye(4).tolist(),'collision_margin_m':.005},'support_tolerance_m':.008,
            'objects':{'basket':{'inner_size_m':[.2,.2,.1]},'bottle':{'body_radius_m':.03}}}
        self.relative={'correction_step_m':.006,'correction_tolerance_m':.002,
            'rotation_tolerance_rad':.02,'maximum_target_shift_m':.02,
            'handover_offset_m':[.04,0.,.08],'donor_retreat_base_m':[0.,-.06,0.]}
        self.start_tools=copy.deepcopy(self.poses);self.transfer_target=None;self.donor_withdrawn=False
        class Planner:
            def tools(_,q):return copy.deepcopy(self.poses)
        class Servo:
            lock=threading.RLock();target=np.zeros(16);state={'q':[0.]*16}
        self.planner=Planner();self.servo=Servo()
    def _ambient_guard(self):pass
    def observe(self):
        s=observation();s.update(held_by=list(self.held),touching={v:v in self.held for v in ['left','right']},
            object_poses={'bottle':self.bottle.tolist(),'basket':self.basket.tolist()},
            checkpoint_id=next(reversed(self.checkpoints),None),operation=copy.deepcopy(self.operation),
            current_grasp_id=self.current_grasp,last_failure=self.last_failure,completed_steps=list(self.completed),
            grasp_region_evidence={'right':{'verified':True,'source':'measured_geometry','part':'body','fraction_min':.2,'fraction_max':.4}},
            support_evidence={n:{'verified':True,'stationary':True} for n in ['table','basket']})
        return s
    def _move(self,targets,**kwargs):
        if self.reject_motion:raise RuntimeError('injected collision rejection; no dispatch')
        self.moves.append(copy.deepcopy(targets))
        if self.held:
            side='left' if self.held==['left'] else 'right'
            if side in targets:self.bottle=targets[side]@np.linalg.inv(self.poses[side])@self.bottle
        self.poses.update(copy.deepcopy(targets))
    def _jaw(self,side,opened):
        if opened:
            if side in self.held:self.held.remove(side)
        else:
            if side=='right':
                self.right_closes+=1
                if self.fail_close and self.right_closes==1:return
            else:
                self.left_closes+=1
                if self.fail_receiver and self.left_closes==1:return
            if side not in self.held:self.held.append(side)
    def _wait_hold(self,side,wanted=True):return self.observe()


class RelativeChecks(unittest.TestCase):
    def test_correction_bounds_translation_and_rotation_for_72_targets(self):
        for x in np.linspace(-.09,.09,6):
            for y in [-.04,0.,.04]:
                for yaw in [-90,-10,10,90]:
                    target=pose(x,y,.03);target[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
                    step=bounded_target(np.eye(4),target)
                    d,r=goal_error(np.eye(4),step)
                    self.assertLessEqual(d,.006000001);self.assertLessEqual(r,.030000001)
                    transform(step)
    def test_small_target_is_reached_without_overshoot(self):
        target=pose(.001,.002,-.001)
        np.testing.assert_allclose(bounded_target(np.eye(4),target),target)
    def test_nonfinite_and_excess_correction_are_rejected(self):
        t=np.eye(4);t[0,3]=np.nan
        for target,step in [(t,.006),(np.eye(4),.007),(np.eye(4),0.)]:
            with self.assertRaises(ValueError):bounded_target(np.eye(4),target,step)
    def test_handover_tracks_measured_session_anchor_and_preserves_bottle_rotation(self):
        b=pose(.5,-.1,.2);b[:3,:3]=Rotation.from_euler('z',.1).as_matrix()
        tools={'left':pose(.4,.2,.3),'right':pose(.4,-.2,.3)}
        a=transfer_goal(b,tools,[.03,0,.1])
        np.testing.assert_allclose(a[:3,3],[.43,0,.4]);np.testing.assert_allclose(a[:3,:3],b[:3,:3])
        delta=np.array([.04,-.03,.01])
        for p in tools.values():p[:3,3]+=delta
        np.testing.assert_allclose(transfer_goal(b,tools,[.03,0,.1])[:3,3],a[:3,3]+delta)
    def test_alignment_follows_new_object_measurement_and_measured_tcp(self):
        f=Fixture();f.propose_grasps();f.pregrasp('g0');old=f.bottle.copy()
        f.poses['right']=f.bottle.copy();f.poses['right'][2,3]-=.012
        f.bottle[0,3]+=.004
        before=f.poses['right'].copy();f.servo.target[:]=1.  # misleading old command
        result=f.align_relative('g0')
        self.assertFalse(result['done'])
        target=[v for k,v in f.events if k=='relative_goal'][-1]['target']
        self.assertAlmostEqual(target[0][3],old[0,3]+.004)
        self.assertLessEqual(goal_error(before,f.poses['right'])[0],.006000001)
        self.assertEqual([v for k,v in f.events if k=='relative_goal'][-1]['phase'],'fine')
    def test_large_target_jump_and_collision_failure_dispatch_nothing(self):
        for jump,reject in [(.021,False),(0.,True)]:
            f=Fixture();f.propose_grasps();f.pregrasp('g0');n=len(f.moves)
            f.bottle[0,3]+=jump;f.reject_motion=reject
            with self.assertRaises(RuntimeError):f.align_relative('g0')
            self.assertEqual(len(f.moves),n)
    def test_right_close_requires_fresh_alignment_not_previous_tool_success(self):
        f=Fixture();f.propose_grasps();f.pregrasp('g0')
        with self.assertRaises(RuntimeError):f.close_gripper('right')
        self.assertEqual(f.right_closes,0)
        f.poses['right']=f.bottle.copy();self.assertTrue(f.close_gripper('right')['ok'])
    def test_donor_withdrawal_and_placement_require_left_hold(self):
        f=Fixture()
        with self.assertRaises(RuntimeError):f.withdraw_donor()
        self.assertEqual(f.moves,[])
        f.held=['left'];f.start_place()
        with self.assertRaisesRegex(RuntimeError,'withdrawal'):f.advance()
        self.assertEqual(f.moves,[])
    def test_receiver_failure_keeps_donor_closed(self):
        f=Fixture(fail_receiver=True);f.held=['right'];f.start_transfer()
        for _ in range(64):
            r=f.advance()
            if not r['ok']:break
        self.assertFalse(r['ok']);self.assertEqual(f.held,['right'])
    def test_official_executor_full_relative_program_and_recovery(self):
        base=Path(__file__).resolve().parents[1]
        runtime=base/'app/aspire/src'
        if not runtime.is_dir():runtime=Path('/home/wsk/ASPIRE/libero-pro/source/src')
        program=base/'app/skill/program.py'
        if not program.is_file():program=base/'deploy/relative_program.py'
        sys.path.insert(0,str(runtime))
        try:
            from cap_harness.registry import ToolRegistry
            from cap_harness.runtime import ProgramExecutor
            for x,close,receiver in [(.46,False,False),(.50,True,False),(.54,False,True)]:
                f=Fixture(x,close,receiver)
                registry=ToolRegistry(shared_public_names=f.tools,public_extension_allowlist=())
                for name in f.tools:registry.register(getattr(f,name),name=name,public=True,
                    layer='shared_high_level',capability='kinematic test double, no hardware')
                result=ProgramExecutor(registry).execute_program(program.read_text())
                self.assertTrue(result.ok,str(result.error));self.assertEqual(f.held,[])
                self.assertEqual(f.completed,['handover','placement']);self.assertTrue(f.donor_withdrawn)
                self.assertGreater(len([1 for k,v in f.events if k=='relative_goal']),10)
                if close:self.assertEqual(f.right_closes,2)
                if receiver:self.assertEqual(f.left_closes,2)
        finally:sys.path.remove(str(runtime))
    def test_client_requires_relative_tool_contract_and_no_motion_by_default(self):
        calls=[];r=Robot({'robot_contract':CONTRACT,'robot_service':'http://127.0.0.1:8772'},
            transport=lambda *a,**k:calls.append(a))
        self.assertEqual(r.tools,RELATIVE_TOOLS)
        with self.assertRaises(RuntimeError):r.call('align_relative','g0')
        self.assertEqual(calls,[])
    def test_relative_contract_cannot_silently_accept_legacy_only_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir()
            (root/'config/robot.json').write_text(json.dumps({'robot_contract':CONTRACT}))
            path=Path(__file__).parent/'calibration.json'
            if not path.is_file():path=Path(__file__).resolve().parents[1]/'config/calibration.json'
            cal=json.loads(path.read_text());cal['motion'].pop('relative',None)
            self.assertIn('measured relative grasp/transfer/place and motion limits',check(cal,{}, {},root))
    def test_physical_acceptance_digest_binds_active_skill_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'app/skill').mkdir(parents=True);(root/'config').mkdir()
            program=root/'app/skill/program.py';program.write_text('observe()')
            (root/'config/robot.json').write_text(json.dumps({'robot_contract':CONTRACT}))
            first=implementation(root);program.write_text('open_gripper()')
            self.assertNotEqual(first,implementation(root))


if __name__=='__main__':unittest.main()
