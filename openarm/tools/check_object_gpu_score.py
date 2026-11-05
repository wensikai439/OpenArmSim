"""Adversarial checks against retained real UI receipts, no new robot actions."""
import copy
import json
import unittest

from sim.environment import ROOT
from tools.check_object_gpu_ui import score

OUT = ROOT / 'runs/20261009-object-gpu'


class ScoreChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((OUT / 'workbench.json').read_text())
        cls.protocol = json.loads((OUT / 'workbench.protocol.json').read_text())
        assert score(cls.record, cls.protocol)['passed']

    def test_wrong_GPU_object_backend_source_or_fallback_cannot_pass(self):
        for kind in ['object', 'source', 'backend', 'fallback', 'native_check']:
            record = copy.deepcopy(self.record); motion = record['receipts'][1]
            if kind == 'object': motion['feedback']['plans'][0]['attachment']['object_id'] = 'block'
            if kind == 'source': motion['feedback']['plans'][0]['metadata']['worker_source_sha256'] = 'wrong'
            if kind == 'backend': motion['arguments']['planner'] = 'tamp'
            if kind == 'fallback': motion['feedback']['fallback'] = True
            if kind == 'native_check': motion['feedback']['native_path_revalidation'] = False
            self.assertFalse(score(record, self.protocol)['passed'], kind)

    def test_changed_goal_wrong_execution_and_extra_action_cannot_pass_cached_green(self):
        for kind in ['goal', 'execution', 'extra', 'release']:
            record = copy.deepcopy(self.record)
            if kind == 'goal': record['terminal']['agent']['plan']['placement_goal']['target_xy_m'][0] += .02
            if kind == 'execution': record['receipts'][1]['execution']['id'] = 'wrong'
            if kind == 'extra':
                action = next(e for t in record['turns'] for e in t['trace'] if e['kind'] == 'tool' and e['name'] == 'move_relative' and e['status'] == 'completed')
                record['turns'][-1]['trace'].append(copy.deepcopy(action))
            if kind == 'release': record['receipts'][-1]['feedback']['placement']['confirmed'] = False
            self.assertFalse(score(record, self.protocol)['passed'], kind)


if __name__ == '__main__': unittest.main(verbosity=2)
