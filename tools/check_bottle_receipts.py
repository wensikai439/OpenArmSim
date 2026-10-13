"""Failure-preserving receipt validation with same-goal stale retries."""
import copy
import unittest

from tools.check_bottle_api import route_evidence


def receipt(skill, arm='right', success=True, code=None):
    args = {'object_id': 'bottle', 'arm': arm}
    feedback = {'primitive_success': success, 'failure_code': code, 'object_id': 'bottle'}
    if skill == 'pick':
        feedback.update(held_by=arm, verification_source='native_two_finger_contact_and_lift')
    if skill == 'place':
        args.update(receptacle_id='table', target_xy_m=[.56, -.13])
    return {'execution': {'skill': skill}, 'arguments': args, 'feedback': feedback}


class ReceiptChecks(unittest.TestCase):
    def route(self):
        return [receipt('pick', 'left'), receipt('handover'), receipt('move_to_pose'), receipt('place')]

    def test_original_route_preserved(self):
        self.assertTrue(route_evidence(self.route(), True, 'left'))

    def test_same_goal_stale_placement_retry_does_not_drop_or_mutate_failure(self):
        values = self.route()
        values.insert(-1, receipt('place', success=False, code='planning_state_changed'))
        before = copy.deepcopy(values)
        self.assertTrue(route_evidence(values, True, 'left'))
        self.assertEqual(values, before)
        self.assertFalse(values[-2]['feedback']['primitive_success'])

    def test_wrong_goal_or_arm_on_failed_placement_is_rejected(self):
        for key, value in [('target_xy_m', [.6, -.13]), ('arm', 'left'), ('object_id', 'can')]:
            values = self.route()
            failed = receipt('place', success=False, code='planning_state_changed')
            failed['arguments'][key] = value
            values.insert(-1, failed)
            self.assertFalse(route_evidence(values, True, 'left'))

    def test_other_failure_codes_missing_final_release_and_extra_completed_placements_rejected(self):
        for code in ['grasp_lost', 'object_gpu_rejected', None]:
            values = self.route()
            values.insert(-1, receipt('place', success=False, code=code))
            self.assertFalse(route_evidence(values, True, 'left'))
        values = self.route()
        values[-1] = receipt('place', success=False, code='planning_state_changed')
        self.assertFalse(route_evidence(values, True, 'left'))
        values = self.route()
        values.append(receipt('place'))
        self.assertFalse(route_evidence(values, True, 'left'))


if __name__ == '__main__':
    unittest.main()
