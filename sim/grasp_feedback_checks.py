"""Recent-window evidence, contact history and passive native integration."""
import unittest

import numpy as np

from .grasp_feedback import GraspFeedback
from .environment import OpenArmEnv
from .tamp import TampPlanner


class GraspFeedbackChecks(unittest.TestCase):
    def test_contact_and_short_duration_cannot_claim_stability(self):
        monitor=GraspFeedback()
        for t in (0,1):monitor.sample('right',t,[0,0,0],np.eye(3),held=True,continuous_contact_s=t+.1)
        self.assertEqual(monitor.summary()['arms']['right']['status'],'insufficient_duration')
        monitor.sample('right',2,[0,0,0],np.eye(3),held=True,continuous_contact_s=2.1)
        report=monitor.summary();self.assertEqual(report['arms']['right']['status'],'stable_recent_window')
        self.assertFalse(report['changes_task_score']);self.assertFalse(report['transfer_ready'])
        self.assertIn('recent',report['scope'])

    def test_out_and_back_slip_and_rotation_do_not_hide_in_equal_endpoints(self):
        for rotation in (False,True):
            monitor=GraspFeedback()
            for i,t in enumerate((0,1,2)):
                p=[0,.0024 if i==1 and not rotation else 0,0]
                angle=.025 if i==1 and rotation else 0
                r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
                monitor.sample('right',t,p,r,held=True,continuous_contact_s=t+.1)
            self.assertEqual(monitor.summary()['arms']['right']['status'],'sliding')

    def test_interrupted_contact_inside_interval_starts_new_evidence(self):
        monitor=GraspFeedback()
        for t in (0,1,2):monitor.sample('right',t,[0,0,0],np.eye(3),held=True,continuous_contact_s=t+.1)
        monitor.sample('right',2.25,[0,0,0],np.eye(3),held=True,continuous_contact_s=.12)
        result=monitor.summary()['arms']['right']
        self.assertEqual(result['status'],'insufficient_duration');self.assertEqual(result['samples'],1)
        self.assertEqual(result['contact_episode'],2)

    def test_observation_cannot_inflate_duration_or_mutate_cached_evidence(self):
        monitor=GraspFeedback()
        monitor.sample('right',0,[0,0,0],np.eye(3),held=True,continuous_contact_s=.1)
        for _ in range(12):monitor.sample('right',0,[.1,0,0],np.eye(3),held=True,continuous_contact_s=5)
        result=monitor.summary();self.assertEqual(result['arms']['right']['samples'],1)
        result['arms']['right']['status']='forged'
        self.assertEqual(monitor.summary()['arms']['right']['status'],'insufficient_duration')
        with self.assertRaises(ValueError):monitor.sample('right',-1,[0,0,0],np.eye(3),held=True,continuous_contact_s=2)

    def test_release_clears_recent_window_and_preserves_contact_identity(self):
        monitor=GraspFeedback()
        for t in (0,1,2):monitor.sample('right',t,[0,0,0],np.eye(3),held=True,continuous_contact_s=t+.1)
        monitor.sample('right',2.1,[0,0,0],np.eye(3),held=False,continuous_contact_s=0)
        self.assertEqual(monitor.summary()['arms']['right']['status'],'not_held')
        monitor.sample('right',2.2,[0,0,0],np.eye(3),held=True,continuous_contact_s=.1)
        self.assertEqual(monitor.summary()['arms']['right']['contact_episode'],2)

    def test_native_monitor_is_passive_and_task_prepare_preserves_it(self):
        env=OpenArmEnv(images=False)
        try:
            env.reset(seed=0);env.set_task_budget(120)
            planner=TampPlanner(env,'pick_bottle',parameters={'arm':'right'})
            for action,_ in planner.actions():env.step_tamp(action)
            self.assertTrue(planner.result['primitive_success']);self.assertTrue(env.evaluation['held']['right'])
            target=env.last_action.copy()
            for _ in range(61):env.step_tamp(target)
            report=env.diagnostics()['manipulation_scene']['grasp_quality']
            self.assertEqual(report['arms']['right']['status'],'sliding')
            self.assertTrue(env.evaluation['held']['right'])
            before=(env.data.qpos.copy(),env.data.qvel.copy(),env.last_action.copy(),env.grasp_feedback.summary())
            for _ in range(10):env.info();env.diagnostics()
            env.prepare_task()
            self.assertTrue(np.array_equal(env.data.qpos,before[0]));self.assertTrue(np.array_equal(env.data.qvel,before[1]))
            self.assertTrue(np.array_equal(env.last_action,before[2]));self.assertEqual(env.grasp_feedback.summary(),before[3])
            env.reset(seed=0)
            self.assertEqual(env.grasp_feedback.summary()['arms']['right']['status'],'not_held')
        finally:env.close()


if __name__=='__main__':unittest.main()
