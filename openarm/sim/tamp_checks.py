"""Physical primitive, planner isolation, and bounded cancellation checks."""
import unittest
import numpy as np
import mujoco
from unittest.mock import patch

from .environment import OpenArmEnv
from .tamp import TampPlanner, PlanningFailure


class TampChecks(unittest.TestCase):
    def setUp(self):
        self.env=OpenArmEnv(images=False)
        self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()

    def test_pick_moves_free_bottle_by_physics_and_preserves_full_task_score(self):
        env=self.env
        initial_bottom=env._bottle_bounds()[0][2]
        planner=TampPlanner(env,'pick_bottle')
        actions=planner.actions()
        steps=0
        while True:
            before=env.data.qpos.copy()
            try:
                action,_=next(actions)
            except StopIteration:
                break
            np.testing.assert_array_equal(before,env.data.qpos)
            _,_,terminal,truncated,_=env.step(action)
            self.assertFalse(terminal or truncated)
            steps+=1
            self.assertLess(steps,900)
        self.assertTrue(planner.result['primitive_success'])
        self.assertTrue(env.evaluation['held']['right'])
        self.assertGreater(env._bottle_bounds()[0][2]-initial_bottom,.06)
        self.assertFalse(env.evaluation['success'])
        self.assertEqual([e['stage'] for e in env.events],[0,1,2])
        self.assertLess(env.evaluation['max_penetration_m'],.001)
        self.assertEqual(env.model.jnt_type[env.model.joint('bottle_free').id],mujoco.mjtJoint.mjJNT_FREE)
        held=env.data.qpos.copy()
        with self.assertRaisesRegex(PlanningFailure,'已经'):
            TampPlanner(env,'pick_bottle')
        np.testing.assert_array_equal(held,env.data.qpos)

    def test_collision_is_checked_before_motion_and_cancel_is_bounded(self):
        env=self.env;planner=TampPlanner(env,'reach_bottle')
        addr=env.model.joint('bottle_free').qposadr[0]
        env.data.qpos[addr:addr+3]=env.data.xpos[env.ee_ids['right']]
        mujoco.mj_forward(env.model,env.data)
        before=env.data.qpos.copy()
        with self.assertRaisesRegex(PlanningFailure,'碰撞'):
            planner._path(env.data.qpos[env.qids['right']].copy())
        np.testing.assert_array_equal(before,env.data.qpos)
        planner.cancel=lambda:True
        with self.assertRaisesRegex(PlanningFailure,'停止'):
            next(planner.actions())
        np.testing.assert_array_equal(before,env.data.qpos)

    def test_separate_operators_complete_handover_with_native_contact_only(self):
        env=self.env
        for task in ('handover_bottle','place_bottle'):
            before=env.data.qpos.copy()
            with self.assertRaises(PlanningFailure):TampPlanner(env,task)
            np.testing.assert_array_equal(before,env.data.qpos)
        for task in ('pick_bottle','handover_bottle','place_bottle'):
            planner=TampPlanner(env,task)
            actions=planner.actions()
            while not env._done:
                before=env.data.qpos.copy()
                try:action,_=next(actions)
                except StopIteration:break
                # All planner FK, carried-body estimates and collision search
                # must leave the live state untouched until the physics step.
                np.testing.assert_array_equal(before,env.data.qpos)
                env.step_tamp(action)
            if task!='place_bottle':
                self.assertFalse(env._done)
                self.assertTrue(planner.result['primitive_success'])
        self.assertTrue(env.evaluation['success'])
        self.assertEqual([e['stage'] for e in env.events],list(range(7)))
        self.assertFalse(any(env.evaluation['touching'].values()))
        self.assertTrue(env.evaluation['supported_by_basket'])
        self.assertGreaterEqual(env.evaluation['stable_seconds'],1.)
        self.assertLess(env.evaluation['max_penetration_m'],.005)

    def test_native_finger_range_does_not_expand_policy_commands(self):
        env=self.env
        action=env.last_action.copy();action[[7,15]]=-1.26
        before=env.data.qpos.copy()
        with self.assertRaises(ValueError):env.step(action)
        np.testing.assert_array_equal(before,env.data.qpos)
        env.step_tamp(action)
        action[7]=-1.30
        with self.assertRaises(ValueError):env.step_tamp(action)

    def test_runtime_manifest_keeps_loaded_hash_and_rejects_disk_drift(self):
        env=self.env
        original=env.manifest()['hashes']
        from .environment import digest
        def changed(path):
            return 'changed-after-start' if str(path).endswith('/sim/tamp.py') else digest(path)
        with patch('sim.environment.digest',side_effect=changed):
            manifest=env.manifest()
            self.assertEqual(manifest['hashes'],original)
            self.assertEqual(manifest['source_status']['changed_files'],['sim/tamp.py'])
            with self.assertRaisesRegex(ValueError,'重启'):env.require_current_sources()
