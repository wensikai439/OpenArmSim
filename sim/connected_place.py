"""Place proposals must include a connected, upright loaded descent.

The former provider accepts the first feasible pair of endpoints. A distant IK
branch can require a large RRT detour and disturb the native grasp. This optional
provider checks the transition before selecting a yaw, while retaining all old
physics, goal, contact, release and stable-support acceptance conditions.
"""
import numpy as np

from .bottle_routing import BottlePlacementPlanner
from .tamp import PlanningFailure


class ConnectedBottlePlacement(BottlePlacementPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proposal_high = None
        self._executing_move = False
        self.result['placement_path_contract'] = {
            'source': 'native collision geometry and measured hand-object relation',
            'endpoint_selection': 'connected approach plus direct loaded lowering before choosing yaw',
            'additional_proposal_constraint': 'carried bottle axis cosine>=0.98 along joint edges',
            'original_physics_goal_contact_release_stability_gates_unchanged': True,
            'scope': 'registered upright bottle on table; not a physical calibration',
            'hardware_access': False, 'transfer_ready': False}

    def _active(self):
        return (self.env.object_id == 'bottle' and self.phase == 'place'
                and getattr(self, 'placement_surface', None) == 'table' and self.carried is not None)

    def _edge(self, start, end, allow_fingers=False):
        if not super()._edge(start, end, allow_fingers):
            return False
        if not self._active():
            return True
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / .025)))
        for index in range(count + 1):
            self._at(start + (end - start) * index / count)
            upright = float(self.data.xmat[self.env.bottle_id].reshape(3, 3)[2, 2])
            if upright < .98:
                self.last_collision = {'kind': 'carried_axis_constraint', 'upright_cosine': upright,
                    'minimum_cosine': .98, 'source': 'additional scratch loaded-path constraint; not a physical collision'}
                return False
        return True

    def _ik(self, target, allow_fingers=False, **kwargs):
        q = super()._ik(target, allow_fingers, **kwargs)
        if (not self._active() or self._executing_move or getattr(self, '_pre_release_servo', False)
                or kwargs.get('position_tolerance', .0015) <= .0002):
            return q
        self._at(q)
        from .scene import bounds
        bottom = min(bounds(self.env.model, self.data, geom)[0][2] for geom in self.env.bottle_geoms)
        table = self.env.placement_surface('table')['top_z']
        if bottom > table + .040:
            # Prove a connected upright route from the current hold to the
            # high pose, not merely an endpoint. All RRT edges use this guard.
            path = self._path(q, allow_fingers)
            self._proposal_high = q.copy()
            self.ik_seeds[self.side].insert(0, q.copy())
            self.result['connected_placement_proposal'] = {
                'high_q_rad': q.tolist(), 'approach_path_nodes': len(path),
                'source': 'scratch-only; real loaded motion must still pass'}
        else:
            if self._proposal_high is None or not self._edge(self._proposal_high, q, allow_fingers):
                self.result['planning_diagnostics'] = {'arm': self.side, 'phase': self.phase,
                    'location': 'high_to_low_placement_proposal', 'blocked_pairs': [self.last_collision]}
                raise PlanningFailure('lowering_path_rejected',
                    'Feasible endpoints lack a direct upright collision-free loaded descent; trying another yaw')
            self.result['connected_placement_proposal'].update(low_q_rad=q.tolist(),
                maximum_descent_joint_change_rad=float(np.max(np.abs(q - self._proposal_high))),
                connected_upright_lowering=True)
        return q

    def _move(self, target, allow_fingers=False, **kwargs):
        previous = self._executing_move
        self._executing_move = True
        try:
            yield from super()._move(target, allow_fingers, **kwargs)
        finally:
            self._executing_move = previous
