"""Native relay IK and explicit empty-donor clearance for parameterized skills.

The legacy full demonstration keeps its original donor withdrawal. Modern
right-to-left handover searches collision-checked empty-donor postures only
after the receiver physically holds the bottle and the donor has released it.
"""
import numpy as np

from .adaptive_grasp import KinematicIK
from .bottle_routing import BottleHandoverPlanner
from .tamp import TampPlanner, PlanningFailure


class FastBottleHandover(KinematicIK,BottleHandoverPlanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.result['ik_pipeline_contract']={
            'iterations':'mj_kinematics then mj_comPos, scratch MjData only',
            'acceptance_and_path_edges':'original full native forward/collision/clearance',
            'proposal_solver_tolerances_joint_bounds_unchanged':True,
            'scope':'original bottle handover; compute efficiency only',
            'live_state_written_by_solver':False,'hardware_access':False,'transfer_ready':False}


class ForwardBottleHandover(KinematicIK, TampPlanner):
    def __init__(self, env, task, **kwargs):
        super().__init__(env, task, **kwargs)
        # Preserve the selected native object's identity through the modern
        # route, so the workspace can verify this exact directed transfer.
        self.result['object_id'] = env.object_id

    def _retreat_after_forward_handover(self, point):
        if not self.env.evaluation['held']['left'] or self.env.evaluation['held']['right']:
            raise PlanningFailure('receiver_hold_required', '先确认接收手独立持瓶，才能规划空手撤离')
        # An empty donor need not preserve its grasp orientation. Joint limits,
        # full native collisions and every path edge remain the original gates.
        self.rotation = None
        rejected = []
        for delta in ([.04, -.10, -.08], [.04, -.14, -.04], [0., -.16, .03],
                      [.08, -.12, 0.], [.04, -.18, -.04]):
            target = point + np.asarray(delta)
            try:
                self._sync()
                goal = self._ik(target, position_tolerance=.0015)
                self._path(goal)
            except PlanningFailure as error:
                if error.code not in {'goal_collision', 'ik_unreachable', 'path_not_found', 'start_collision'}:
                    raise
                rejected.append({'delta_m': delta, 'failure_code': error.code,
                                 'diagnostics': self.result.get('planning_diagnostics')})
                continue
            self.result['donor_retreat_search'] = {'selected_delta_m': delta, 'rejected': rejected,
                'orientation': 'unconstrained empty donor', 'collision_gates_unchanged': True}
            yield from self._move(target)
            if self.env.evaluation['touching']['right'] or not self.env.evaluation['held']['left']:
                raise PlanningFailure('handover_not_confirmed', '实际撤离后仍有供物手接触或接收手失去夹持')
            return
        self.result['donor_retreat_search'] = {'selected_delta_m': None, 'rejected': rejected}
        raise PlanningFailure('retreat_unreachable', '当前交接姿态未找到保持接收手夹持的无碰撞撤离路径')
