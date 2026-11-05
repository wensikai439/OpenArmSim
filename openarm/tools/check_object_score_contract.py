"""Adversarial checks of the real UI task's read-only semantic scorer."""
import copy
import json
import unittest

from .check_object_score import OUT, score


class ScoreChecks(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((OUT/'workbench.json').read_text())
        self.protocol = json.loads((OUT/'workbench.protocol.json').read_text())

    def test_equivalent_absolute_target_and_relative_goal_use_original_object_anchor(self):
        self.assertTrue(score(self.record,self.protocol)['passed'])
        record = copy.deepcopy(self.record); goal = record['terminal']['agent']['plan']['placement_goal']
        goal['offset_xy_m'] = [.06,0]
        self.assertTrue(score(record,self.protocol)['passed'])
        goal['target_xy_m'][0] += 1.1e-6
        self.assertFalse(score(record,self.protocol)['passed'])

    def test_goal_anchor_wrong_object_receipt_source_arm_or_release_gate_is_rejected(self):
        for kind in ('anchor','object','source','arm','release','tolerance','error'):
            record = copy.deepcopy(self.record)
            if kind=='anchor': record['terminal']['agent']['plan']['placement_goal']['anchor']['object_xy_m'][0] += .01
            elif kind=='object': record['receipts'][0]['feedback']['object_id']='block'
            elif kind=='source': record['receipts'][1]['feedback']['verification_source']='robot_forward_kinematics'
            elif kind=='arm': record['receipts'][0]['arguments']['arm']='left'
            elif kind=='release': record['receipts'][1]['feedback']['placement']['confirmed']=False
            elif kind=='tolerance': record['receipts'][1]['feedback']['placement']['tolerance_m']=.011
            elif kind=='error': record['receipts'][1]['feedback']['position_error_m']=.010001
            self.assertFalse(score(record,self.protocol)['passed'],kind)

    def test_extra_motion_pending_stage_or_unbound_execution_cannot_pass_from_cached_checks(self):
        for kind in ('extra','stage','execution'):
            record = copy.deepcopy(self.record)
            if kind=='extra': record['turns'][0]['trace'].append({'kind':'tool','name':'move_relative','status':'completed'})
            elif kind=='stage': record['terminal']['agent']['plan']['steps'][0]['status']='pending'
            elif kind=='execution': record['receipts'][0]['execution']['id']='wrong'
            self.assertFalse(score(record,self.protocol)['passed'],kind)


if __name__=='__main__':unittest.main(verbosity=2)
