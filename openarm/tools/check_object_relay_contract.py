"""Protection, primitive receiver frames and unchanged native parameters."""
import copy
import unittest

import mujoco
import numpy as np

from harness.object_handover import ProtectedObjectEnv, ObjectRelayHarness
from sim.objects import ObjectEnv
from sim.object_handover import receiver_candidates


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.env = ProtectedObjectEnv(images=False)
    @classmethod
    def tearDownClass(cls): cls.env.close()

    def setUp(self): self.env.reset(seed=0)

    def test_guard_does_not_change_compiled_physics_or_controller(self):
        original = ObjectEnv(images=False)
        try:
            self.assertEqual(original.manifest()['model_evidence'], self.env.manifest()['model_evidence'])
            for field in ('body_mass', 'body_inertia', 'geom_size', 'geom_friction', 'jnt_range',
                'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'cam_pos', 'cam_quat', 'cam_fovy'):
                np.testing.assert_array_equal(getattr(original.model, field), getattr(self.env.model, field))
            self.assertEqual(original.profile, self.env.profile)
        finally: original.close()

    def test_non_target_drift_fails_without_mutating_the_live_environment(self):
        env = self.env; env.select_object('block')
        before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time)
        shadow = copy.copy(env); shadow.data = mujoco.MjData(env.model)
        mujoco.mj_copyData(shadow.data, env.model, env.data)
        shadow.data.qpos[env.registry['bottle']['qposadr']] += .004
        mujoco.mj_forward(env.model, shadow.data)
        report = shadow._evaluate(0.)
        self.assertTrue(report['invalid']); self.assertFalse(report['non_target_object_guard']['passed'])
        self.assertIn('bottle', report['failure_reasons'][-1])
        for old, new in zip(before[:3], (env.data.qpos, env.data.qvel, env.data.ctrl)): np.testing.assert_array_equal(old, new)
        self.assertEqual(before[3], env.data.time)
        selected = copy.copy(env); selected.data = mujoco.MjData(env.model)
        mujoco.mj_copyData(selected.data, env.model, env.data)
        selected.data.qpos[env.registry['block']['qposadr']] += .02; mujoco.mj_forward(env.model, selected.data)
        self.assertTrue(selected._evaluate(0.)['non_target_object_guard']['passed'])

    def test_receiver_candidates_use_actual_shape_and_proper_rotation_frames(self):
        env = self.env
        for ident in ('block', 'can'):
            env.select_object(ident); before = env.data.qpos.copy()
            values = list(receiver_candidates(env, env.data, 'left', 'right')); self.assertTrue(values)
            for value in values:
                rotation = value['rotation']; np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1.)
                self.assertLessEqual(value['required_opening_m'], sum(np.ptp(f.range) for f in env.fingers['right']))
                self.assertEqual(value['geom'], env.model.geom(next(iter(env.bottle_geoms))).name)
            np.testing.assert_array_equal(before, env.data.qpos)

    def test_invalid_scene_blocks_a_second_motion_before_worker_dispatch(self):
        class InvalidService:
            def snapshot(self): return {'diagnostics': {'evaluation': {'invalid': True}}}
            def call(self, *args, **kwargs): raise AssertionError('invalid scene must not dispatch another native job')
        harness = ObjectRelayHarness.__new__(ObjectRelayHarness); harness.service = InvalidService()
        with self.assertRaisesRegex(ValueError, '不能继续运动'):
            harness._invoke('handover', {'object_id': 'can', 'from_arm': 'right', 'to_arm': 'left'}, 'execute')


if __name__ == '__main__': unittest.main(verbosity=2)
