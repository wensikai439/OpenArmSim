"""Independent gripper commands and generic-vs-benchmark termination."""
import unittest
from unittest.mock import patch
import numpy as np
from .environment import OpenArmEnv
from .tamp import TampPlanner

class GripperChecks(unittest.TestCase):
    def setUp(self):self.env=OpenArmEnv(images=False);self.env.reset(seed=0)
    def tearDown(self):self.env.close()
    def test_composed_open_partial_close_preserves_arm_and_other_gripper_commands(self):
        env=self.env
        for side,fraction in [('left',1.),('right',.5),('left',0.)]:
            original=env.last_action.copy();planner=TampPlanner(env,'set_gripper',parameters={'arm':side,'opening_fraction':fraction})
            chosen=7 if side=='left' else 15;others=np.delete(np.arange(16),chosen)
            iterator=planner.actions()
            while True:
                before=env.data.qpos.copy()
                try:action,_=next(iterator)
                except StopIteration:break
                np.testing.assert_array_equal(before,env.data.qpos)
                np.testing.assert_array_equal(action[others],original[others])
                _,_,terminal,truncated,_=env.step_tamp(action)
                self.assertFalse(terminal or truncated)
            self.assertTrue(planner.result['primitive_success'])
            self.assertEqual(planner.result['verification_source'],'robot_gripper_encoder_settling')
            self.assertFalse(any(env.evaluation['held'].values()))
            self.assertFalse(env.evaluation['success'])

    def test_invalid_fraction_rejects_without_motion(self):
        q=self.env.data.qpos.copy()
        for fraction in (-.1,1.1,True,float('nan')):
            with self.assertRaises(ValueError):TampPlanner(self.env,'set_gripper',parameters={'arm':'right','opening_fraction':fraction})
        np.testing.assert_array_equal(q,self.env.data.qpos);self.assertEqual(self.env.tick,0)

    def test_generic_budget_is_declared_before_motion_and_cannot_extend_an_attempt(self):
        env=self.env;q=env.data.qpos.copy();v=env.data.qvel.copy()
        env.set_task_budget(60)
        np.testing.assert_array_equal(q,env.data.qpos);np.testing.assert_array_equal(v,env.data.qvel)
        self.assertEqual(env.task_window()['active_seconds'],60)
        env.step_tamp(env.last_action.copy())
        with self.assertRaisesRegex(ValueError,'失败后加时'):env.set_task_budget(90)
        for value in (True,60.0,30,180):
            with self.assertRaises(ValueError):env.set_task_budget(value)
        env.prepare_task();self.assertEqual(env.task_window()['requested_seconds'],40)

    def test_legacy_limit_remains_40_and_generic_60_does_not_stop_at_40(self):
        env=self.env;env.set_task_budget(60);env.data.time=39.966;env.tick=1199
        self.assertFalse(env.step_tamp(env.last_action.copy())[3])
        env.reset();env.set_task_budget(60);env.task_termination='legacy_benchmark'
        env.data.time=39.966;env.tick=1199
        self.assertTrue(env.step_tamp(env.last_action.copy())[3])

    def test_placement_withdrawal_also_clears_closed_jaw(self):
        env=self.env;target=(env.data.xpos[env.bottle_id,:2]+[.08,0]).tolist()
        for task,params in [('pick_bottle',{'arm':'left'}),('handover_bottle',{'from_arm':'left','to_arm':'right'}),
                            ('place_bottle',{'arm':'right','receptacle_id':'table','target_xy_m':target})]:
            planner=TampPlanner(env,task,parameters=params)
            for action,_ in planner.actions():
                _,_,terminal,truncated,_=env.step_tamp(action)
                self.assertFalse(terminal or truncated)
        env.prepare_task();before=env.data.xpos[env.bottle_id].copy()
        planner=TampPlanner(env,'set_gripper',parameters={'arm':'right','opening_fraction':0.})
        for action,_ in planner.actions():env.step_tamp(action)
        self.assertFalse(any(env.evaluation['touching'].values()))
        self.assertFalse(any(env.evaluation['held'].values()))
        self.assertLess(np.linalg.norm(env.data.xpos[env.bottle_id]-before),.001)

    def test_old_basket_score_cannot_end_generic_skill_but_invalid_state_still_stops(self):
        env=self.env
        TampPlanner(env,'set_gripper',parameters={'arm':'right','opening_fraction':1.})
        evaluation={**env.evaluation,'success':True,'invalid':False}
        with patch.object(env,'_evaluate',return_value=evaluation):
            self.assertFalse(env.step_tamp(env.last_action.copy())[2])
        evaluation['invalid']=True
        with patch.object(env,'_evaluate',return_value=evaluation):
            self.assertTrue(env.step_tamp(env.last_action.copy())[2])
        env.reset(seed=0)
        evaluation={**env.evaluation,'success':True}
        with patch.object(env,'_evaluate',return_value=evaluation):
            self.assertTrue(env.step_tamp(env.last_action.copy())[2])

if __name__=='__main__':unittest.main()
