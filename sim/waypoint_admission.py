"""Opt-in unloaded IK admission when static current-history headroom fails.

Current SourceMIT history cannot prove future endpoint motor infeasibility.
Allow the original collision-checked unloaded pick pose to reach the original
bounded native dispatcher; loaded headroom, retiming, force/range, contact and
task gates remain mandatory. No future dynamics or physical success is claimed.
"""
import numpy as np
from .tamp import TampPlanner, PlanningFailure


def install_isolated(*, allow_loaded=False):
    """Use instead of waypoint_ik.install in a fresh candidate process."""
    from .waypoint_ik import install_isolated as static_poses
    if type(allow_loaded) is not bool: raise ValueError('explicit boolean admission scope required')
    geometric = TampPlanner._ik
    static_poses()
    proposed = TampPlanner._ik
    def inverse(self, target, allow_fingers=False, *, position_tolerance=.0015):
        try:
            return proposed(self, target, allow_fingers, position_tolerance=position_tolerance)
        except PlanningFailure as exc:
            allowed_task = self.task in ('pick_bottle', 'pick_object', 'reach_bottle', 'reach_object')
            if (exc.code != 'controller_pose_unreachable'
                or not allow_loaded and (self.carried is not None or self.phase != 'approach' or not allowed_task)
                or not hasattr(self.env, 'waypoint_adapter')):
                raise
            # Recompute with the original complete IK/collision implementation,
            # without treating the static proposal's pose as a proven solution.
            q = geometric(self, target, allow_fingers, position_tolerance=position_tolerance)
            self.result.setdefault('private_unloaded_admissions', []).append({
                'arm': self.side, 'phase': self.phase, 'target_m': np.asarray(target).tolist(),
                'geometric_q_rad': q.tolist(), 'static_proposal_failure': exc.code,
                'source_calls': dict(self.env.source_controller.calls),
                'scope': 'geometric proposal admitted for bounded execution; future motor feasibility unproven',
                'original_static_rejection_retained': True, 'joint_and_motor_limits_unchanged': True,
                'loaded_pose_admission_unchanged': not allow_loaded, 'carried_geometry': self.carried is not None,
                'actual_native_dispatch_limits_mandatory': True, 'hardware_access': False, 'transfer_ready': False})
            return q
    TampPlanner._ik = inverse
