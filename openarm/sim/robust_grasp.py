"""Optional grasp lookahead that reserves joint margin for loaded feedback.

This is proposal filtering, not an enlarged workspace or changed task score.
The legacy grasp provider and original native collision/hold gates are retained.
"""
import numpy as np

from .bottle_routing import BottlePickPlanner
from .tamp import PlanningFailure


class RobustBottlePick(BottlePickPlanner):
    def _ik(self, target, allow_fingers=False, **kwargs):
        progress = self.result.setdefault('private_search_progress', {
            'source': 'native proposal search only; never executed success evidence',
            'IK_calls': 0, 'rejected_codes': {}, 'placement_joint_margin_rad': .04})
        progress['IK_calls'] += 1
        progress.update(arm=self.side, phase=self.phase)
        try:
            q = super()._ik(target, allow_fingers, **kwargs)
            if (self.phase == 'place' and self.carried is not None
                    and self.parameters.get('next_placement')):
                limits = np.asarray([self.env.model.joint(f'openarm_{self.side}_joint{i}').range
                                     for i in range(1, 8)])
                slack = float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min())
                progress['latest_placement_joint_margin_rad'] = slack
                if slack < .04:
                    raise PlanningFailure('placement_near_joint_stop',
                        'Placement lookahead leaves less than0.04rad joint margin for loaded feedback')
            return q
        except PlanningFailure as exc:
            counts = progress['rejected_codes']
            counts[exc.code] = counts.get(exc.code, 0) + 1
            progress['latest_failure_code'] = exc.code
            raise
