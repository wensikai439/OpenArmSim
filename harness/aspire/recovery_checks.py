"""Recovery chooses from measured state; releases and resets stay guarded."""
from types import SimpleNamespace
import unittest

from .runtime import ROOT, ProgramExecutor
from .recovery import RecoveryWorld, SegmentPlanner
from sim.tamp import PlanningFailure


class FeedbackWorld:
    """Contract double: no physics-success claims are derived from these tests."""
    PUBLIC=RecoveryWorld.PUBLIC
    def __init__(self, fault='none'):
        self.fault=fault;self.used=False;self.held=[];self.calls=[];self.phase=None
        self.revision=0;self.grasp=None;self.closed=False;self.retreated=False

    def observe(self):
        return dict(held_by=self.held[:],touching={s:s in self.held for s in ('left','right')},
            upright=True,operation=None,terminal=False,invalid=False,last_failure=None,
            pose_revision=self.revision,current_grasp_id=self.grasp,checkpoint_id='checked-tcp',
            completed_steps=(),remaining_seconds=120,
            gripper_motor_feedback_rad={'left':-1.2,'right':.05 if self.closed else -1.2},
            gripper_open_rad={'left':-1.2,'right':-1.2})

    def propose_grasps(self,exclude_ids=()):
        self.calls.append('propose')
        return {'candidates':[{'id':'observed-'+str(self.revision)+'-'+str(len(self.calls))}],
            'pose_revision':self.revision}

    def pregrasp(self,grasp_id):
        if self.held:raise ValueError('cannot pregrasp while holding')
        self.grasp=grasp_id;self.calls.append('pregrasp');return self.ok()

    def approach(self,grasp_id):self.calls.append('approach');return self.ok()

    @staticmethod
    def ok():return dict(ok=True,done=True,error_code=None)

    def close_gripper(self,arm='right'):
        self.calls.append('close_'+arm)
        if arm=='left':
            if self.held!=['right']:raise ValueError('donor required')
            self.held=['left','right'];return self.ok()
        if not self.used and self.fault in ('command','pose','collision'):
            self.used=True;self.closed=self.fault!='command';self.revision+=self.fault=='pose'
            return dict(ok=False,done=True,error_code='start_collision' if self.fault=='collision' else 'grasp_not_confirmed')
        if self.fault=='pose' and not self.retreated:raise ValueError('must physically retreat after miss')
        self.closed=True;self.held=['right'];return self.ok()

    def open_gripper(self,arm='right'):
        if arm in self.held:raise ValueError('held release forbidden')
        self.calls.append('open_'+arm);self.closed=False;return self.ok()

    def retreat(self,checkpoint_id):
        if checkpoint_id!='checked-tcp' or self.held:raise ValueError('bad retreat')
        self.calls.append('retreat');self.retreated=True;return self.ok()

    def verify_hold(self,arm='right'):return {'ok':arm in self.held}
    def lift(self):
        if self.held!=['right']:raise ValueError('no right hold')
        self.calls.append('lift');return self.ok()

    def start_transfer(self,placement_aware=True):
        if self.held!=['right']:raise ValueError('right donor required')
        self.calls.append('transfer');self.phase='transfer';return {'ok':True,'done':False}

    def advance(self):
        if self.phase=='transfer':
            self.phase=None
            if self.fault=='receiver':return dict(ok=False,done=True,error_code='dual_hold_not_confirmed')
            self.held=['left'];return self.ok()
        if self.phase=='place':self.phase=None;self.held=[];return self.ok()
        raise ValueError('no active operation')

    def finish_transfer(self):
        if set(self.held)!={'left','right'}:raise ValueError('dual hold required')
        self.calls.append('finish_transfer');self.held=['left'];return self.ok()

    def start_place(self):
        if self.held!=['left']:raise ValueError('left hold required')
        self.calls.append('place');self.phase='place';return {'ok':True,'done':False}


class RecoveryChecks(unittest.TestCase):
    def run_skill(self,fault):
        w=FeedbackWorld(fault)
        r=ProgramExecutor(RecoveryWorld.registry(w)).execute_program(
            (ROOT/'skills/bottle_recovery/program.py').read_text())
        return r,w

    def test_nominal_is_not_retried(self):
        r,w=self.run_skill('none');self.assertTrue(r.ok,r.error)
        self.assertEqual(w.calls.count('close_right'),1)
        self.assertNotIn('retreat',w.calls);self.assertNotIn('finish_transfer',w.calls)

    def test_open_motor_retries_close_without_repeating_motion(self):
        r,w=self.run_skill('command');self.assertTrue(r.ok,r.error)
        self.assertEqual(w.calls.count('close_right'),2)
        self.assertEqual(w.calls.count('pregrasp'),1)

    def test_closed_empty_jaw_retreats_and_reproposes(self):
        r,w=self.run_skill('pose');self.assertTrue(r.ok,r.error)
        self.assertEqual(w.calls.count('retreat'),1)
        self.assertEqual(w.calls.count('propose'),2)
        self.assertLess(w.calls.index('open_right'),w.calls.index('retreat'))

    def test_receiver_failure_keeps_donor_then_closes_receiver(self):
        r,w=self.run_skill('receiver');self.assertTrue(r.ok,r.error)
        self.assertEqual(w.calls.count('transfer'),1)
        self.assertLess(w.calls.index('close_left'),w.calls.index('finish_transfer'))
        self.assertLess(w.calls.index('finish_transfer'),w.calls.index('place'))

    def test_unknown_collision_stops_without_blind_recovery(self):
        r,w=self.run_skill('collision');self.assertFalse(r.ok)
        for call in ('lift','retreat','open_right','transfer','place'):self.assertNotIn(call,w.calls)

    def test_motion_guard_forbids_terminal_resume(self):
        w=object.__new__(RecoveryWorld);w.calls=0
        w.env=SimpleNamespace(require_current_sources=lambda:None,_done=True,evaluation={'invalid':False})
        with self.assertRaisesRegex(ValueError,'terminal'):w._guard()
        w._guard(readonly=True)

    def test_stale_grasp_and_held_checkpoint_are_rejected(self):
        w=object.__new__(RecoveryWorld);w.catalogue={'old':{'pose_revision':0}};w.pose_revision=1
        with self.assertRaisesRegex(ValueError,'stale'):w._grasp('old')
        w.calls=0;w.checkpoint={'id':'last'}
        w.env=SimpleNamespace(require_current_sources=lambda:None,_done=False,
            evaluation={'invalid':False,'held':{'left':False,'right':True}})
        with self.assertRaisesRegex(ValueError,'held object'):w.retreat('last')

    def test_held_release_and_incomplete_dual_hold_fail_before_motion(self):
        for mode,held,code in (('gripper',{'left':False,'right':True},'release_forbidden'),
            ('finish_transfer',{'left':False,'right':True},'dual_hold_required'),
            ('pregrasp',{'left':True,'right':False},'already_holding')):
            p=object.__new__(SegmentPlanner);p.mode=mode;p.side='right';p.fraction=1
            p.env=SimpleNamespace(evaluation={'held':held})
            with self.assertRaises(PlanningFailure) as caught:next(p.actions())
            self.assertEqual(caught.exception.code,code)

    def test_no_reset_or_hidden_score_tool(self):
        self.assertEqual(set(RecoveryWorld.PUBLIC),{'observe','propose_grasps','pregrasp','approach',
            'close_gripper','open_gripper','verify_hold','lift','retreat','start_transfer','advance',
            'finish_transfer','start_place'})


if __name__=='__main__':unittest.main()
