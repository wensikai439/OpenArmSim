"""Hold validator keeps native motor commands and rejects missing hold proof."""
import copy
import unittest

import numpy as np

from sim.oriented_grasp import OrientedPlanner, HoldAudit
from sim.tamp import PlanningFailure
from harness.object_handover import ProtectedObjectEnv
from tools.check_pose_grasp import initialize, CASES


class HoldChecks(unittest.TestCase):
    def test_native_requested_hold_and_original_controller_match(self):
        with ProtectedObjectEnv(images=False) as env, ProtectedObjectEnv(images=False) as reference:
            for cell in (env, reference): initialize(cell, CASES[2])
            a = HoldAudit(OrientedPlanner(env, 'pick_object', parameters={'object_id': 'block', 'arm': 'left'}), 2.)
            b = OrientedPlanner(reference, 'pick_object', parameters={'object_id': 'block', 'arm': 'left'})
            measured, original = a.actions(), b.actions()
            for action, _ in original:
                checked, _ = next(measured); np.testing.assert_array_equal(action, checked)
                reference.step_tamp(action); env.step_tamp(checked)
            previous = env.last_action.copy(); steps = 0
            for checked, _ in measured:
                np.testing.assert_array_equal(checked, previous)
                reference.step_tamp(previous); env.step_tamp(checked); steps += 1
                for key in ('qpos', 'qvel', 'ctrl'): np.testing.assert_array_equal(getattr(env.data, key), getattr(reference.data, key))
            self.assertEqual(steps, 60); self.assertTrue(a.result['primitive_success'])
            self.assertGreaterEqual(a.result['hold_audit']['actual_seconds'], 1.999)
            self.assertEqual(a.result['hold_audit']['arms']['left']['status'], 'stable_within_audit_window')
            self.assertEqual(env.data.time, reference.data.time)
            self.assertEqual(env.manifest()['model_evidence'], reference.manifest()['model_evidence'])

    def test_invalid_duration_or_no_confirmed_pick_never_emits_extra_control(self):
        from types import SimpleNamespace
        for seconds in (0, 1, -1, 11, True, float('nan')):
            with self.assertRaises(ValueError): HoldAudit(SimpleNamespace(), seconds)
        class FailedPick:
            result = {'primitive_success': False}
            def actions(self): yield from ()
        audit = HoldAudit(FailedPick(), 10.)
        with self.assertRaises(PlanningFailure): next(audit.actions())


if __name__ == '__main__': unittest.main()
