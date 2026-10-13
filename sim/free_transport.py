"""Empty optional scenes need robot/support-derived transport proposals.

Existing obstacle corridors stay untouched. These are finite proposals only;
all original pose, collision, contact, clearance and budget gates still apply.
The simulation process has one physics owner; scoped function substitution
does not change any model, data, actuator, object registry or physical state.
"""
import copy
from unittest.mock import patch

import numpy as np

from . import bottle_routing
from .bottle_routing import CONTRACT, bypass_origins as original_origins
from .fast_handover import FastBottleHandover
from .margin_grasp import MarginBottlePick
from .scene import bounds
from .tamp import PlanningFailure


def transport_origins(env, data):
    values = original_origins(env, data)
    if any(ident != 'bottle' for ident in env.registry):
        return values
    origin = data.xpos[env.bottle_id].copy()
    low = np.min([bounds(env.model, data, int(g))[0] for g in env.bottle_geoms], axis=0)
    high = np.max([bounds(env.model, data, int(g))[1] for g in env.bottle_geoms], axis=0)
    table_low, table_high = bounds(env.model, data, env.model.geom('cell_table_col').id)
    shoulder_x = float(np.mean([data.xanchor[env.model.joint(f'openarm_{arm}_joint1').id, 0]
        for arm in ('left', 'right')]))
    back_extent, front_extent = origin[0] - low[0], high[0] - origin[0]
    padding = CONTRACT['transport_proposal_clearance_m']
    minimum = max(table_low[0] + back_extent + padding, shoulder_x + back_extent + padding)
    maximum = table_high[0] - front_extent - padding
    width = max(high[0] - low[0], 1e-6)
    # Half an actual object width, bounded to16 proposed centers. This uses
    # current geometry rather than an absent can's remembered world pose.
    count = min(16, max(1, int(np.ceil(max(0., origin[0] - minimum) / (width / 2)))))
    for index in range(1, count + 1):
        x = origin[0] - index * width / 2
        if not minimum <= x <= maximum:
            continue
        goal = origin.copy()
        goal[0] = x
        values.append({'origin_m': goal, 'around_object': None, 'side': 'empty_base_corridor',
            'proposal_source': 'current bottle bounds, native table and shoulder anchors'})
    return values


class FreeBottlePick(MarginBottlePick):
    def _future_transport(self, grasp):
        if self._obstacles:
            return super()._future_transport(grasp)
        self.result['free_transport_proposals'] = {
            'source': 'current native object bounds, support bounds and shoulder anchors',
            'scope': 'empty optional scene; scratch proposals only, original native gates',
            'scene_objects_unchanged': True, 'hardware_access': False}
        # The original optional-object clearance is for OTHER free objects.
        # A supporting table cannot be inserted into that list: the carried
        # bottle starts touching it, so every lift edge would be rejected at
        # its first sample. The native table collision/contact checks remain.
        saved = (self.carried, self.rotation.copy(), self.phase, copy.deepcopy(self.ik_seeds))
        rejected = []
        try:
            self._at(grasp)
            hand = self.data.xmat[self.body].reshape(3, 3).copy()
            origin = self.data.xpos[self.env.bottle_id].copy()
            obj = self.data.xmat[self.env.bottle_id].reshape(3, 3).copy()
            self.carried = (hand.T @ (origin - self.data.xpos[self.body]), hand.T @ obj)
            self.phase = 'transfer'
            top = float(self.env.placement_surface('table')['top_z'])
            lower = min(bounds(self.env.model, self.data, int(g))[0][2] for g in self.env.bottle_geoms)
            elevated = origin + [0, 0, top + CONTRACT['transport_vertical_clearance_m'] - lower]
            goals = [{**p, 'origin_m': p['origin_m'] + [0, 0, .12]}
                for p in transport_origins(self.env, self.data)]
            goals.append({'origin_m': elevated, 'side': 'elevated', 'around_object': None})
            approach = float(np.arctan2(hand[1, 2], hand[0, 2]))
            limits = np.asarray([self.env.model.joint(f'openarm_{self.side}_joint{i}').range for i in range(1, 8)])
            for proposed in goals:
                for yaw in dict.fromkeys([0., -approach, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]):
                    c, s = np.cos(yaw), np.sin(yaw)
                    self.rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ hand
                    try:
                        target = proposed['origin_m'] + self.rotation @ (self.local_point - self.carried[0])
                        q = self._ik(target, True)
                        margin = float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min())
                        if margin < .04:
                            raise PlanningFailure('transport_near_joint_stop', 'Transport hypothesis leaves less than0.04rad joint slack')
                        if not self._edge(grasp, q, True):
                            raise PlanningFailure('transport_lift_edge_collision', 'Connected native transport hypothesis collides')
                        return {'goal_object_origin_m': proposed['origin_m'].tolist(), 'yaw_offset_rad': yaw,
                            'joint_slack_rad': margin, 'route': proposed['side'], 'around_object': None,
                            'source': 'scratch current geometry/robot/support corridor; actual lift/hold/relay mandatory',
                            'rejected': rejected}
                    except PlanningFailure as exc:
                        if exc.code in ('cancelled', 'planning_timeout'):
                            raise
                        rejected.append({'route': proposed['side'], 'yaw_offset_rad': yaw, 'code': exc.code,
                            'diagnostics': copy.deepcopy(self.result.get('planning_diagnostics'))})
            self.result['future_transport_search'] = {'rejected': rejected}
            raise PlanningFailure('transport_clearance_unreachable', 'No connected native empty-scene corridor with joint slack')
        finally:
            self.carried, self.rotation, self.phase, self.ik_seeds = saved


class FreeBottleHandover(FastBottleHandover):
    def _handover(self):
        if any(ident != 'bottle' for ident in self.env.registry):
            yield from super()._handover()
            return
        self.result['free_transport_proposals'] = {
            'source': 'current native object bounds, support bounds and shoulder anchors',
            'scope': 'empty optional scene; native actuator paths, original gates',
            'scene_objects_unchanged': True, 'hardware_access': False}
        with patch.object(bottle_routing, 'bypass_origins', transport_origins):
            yield from super()._handover()
