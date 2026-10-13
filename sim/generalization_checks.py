"""Directed transfer, recovery and no hidden live-state edits."""
import unittest
import numpy as np
from .environment import OpenArmEnv
from .tamp import TampPlanner,PlanningFailure

class GeneralizationChecks(unittest.TestCase):
    def setUp(self):
        self.env=OpenArmEnv(images=False);self.env.reset(seed=0)
    def tearDown(self):self.env.close()
    def drive(self,task,args):
        env=self.env;planner=TampPlanner(env,task,parameters=args);count=0
        actions=planner.actions()
        while True:
            q=env.data.qpos.copy();v=env.data.qvel.copy()
            try:a,_=next(actions)
            except StopIteration:break
            np.testing.assert_array_equal(q,env.data.qpos);np.testing.assert_array_equal(v,env.data.qvel)
            _,_,terminal,truncated,_=env.step_tamp(a);count+=1
            self.assertFalse(terminal or truncated);self.assertLess(count,1200)
        self.assertTrue(planner.result['primitive_success']);return planner
    def test_left_pick_then_transfer_to_right_in_original_scene(self):
        p=self.drive('pick_bottle',{'arm':'left'})
        self.assertTrue(self.env.evaluation['held']['left']);self.assertEqual(p.result['held_by'],'left')
        p=self.drive('handover_bottle',{'from_arm':'left','to_arm':'right'})
        self.assertTrue(self.env.evaluation['held']['right']);self.assertFalse(self.env.evaluation['touching']['left'])
        self.assertEqual(p.result['verification_source'],'native_receiver_contact_then_donor_release')
        self.assertFalse(self.env.evaluation['success'])  # legacy basket task remains separate
    def test_new_task_preserves_state_and_can_transfer_a_body_grasp_back(self):
        self.drive('pick_bottle',{});self.drive('handover_bottle',{})
        env=self.env;q=env.data.qpos.copy();v=env.data.qvel.copy();a=env.last_action.copy();t=env.data.time
        env.prepare_task()
        np.testing.assert_array_equal(q,env.data.qpos);np.testing.assert_array_equal(v,env.data.qvel)
        np.testing.assert_array_equal(a,env.last_action);self.assertEqual(t,env.data.time)
        self.assertTrue(env.evaluation['held']['left'])
        self.drive('handover_bottle',{'from_arm':'left','to_arm':'right'})
        self.assertTrue(env.evaluation['held']['right']);self.assertFalse(env.evaluation['touching']['left'])
    def test_invalid_state_and_same_arm_transfer_reject_without_motion(self):
        q=self.env.data.qpos.copy();self.env.evaluation['invalid']=True
        with self.assertRaises(ValueError):self.env.prepare_task()
        np.testing.assert_array_equal(q,self.env.data.qpos)
        self.env.evaluation['invalid']=False
        with self.assertRaises(PlanningFailure):TampPlanner(self.env,'handover_bottle',parameters={'from_arm':'left','to_arm':'left'})
        np.testing.assert_array_equal(q,self.env.data.qpos)

if __name__=='__main__':unittest.main()
