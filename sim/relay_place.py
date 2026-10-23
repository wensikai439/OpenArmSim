"""Optional task-aware relay and upright basket placement; demo unchanged.

The measured hand-object transform determines every target. Before choosing a
yaw, prove both the loaded approach and a connected descent. Only
scratch geometry is used for proposals; native actuators/contact decide outcome.
"""
import copy

import numpy as np

from .bottle_routing import BottlePlacementPlanner
from .scene import bounds
from .tamp import PlanningFailure


class RelayBasketPlacement(BottlePlacementPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proposal_high = None
        self._executing_move = False
        self.result['relay_placement'] = {
            'provider': 'upright basket pose with connected loaded descent',
            'source': 'current simulator geometry and measured hand-object relation',
            'physics_collision_release_and_score_unchanged': True,
            'hardware_access': False, 'transfer_ready': False}

    def _placement_upright(self):
        return True

    def _placement_yaws(self):
        # Solve the task's free yaw at the difficult low pose first. The same
        # yaw must then pass high-pose IK, the approach and connected descent.
        from .task_grasp import upright_solution
        measured=self.env.data.xmat[self.env.bottle_id].reshape(3,3)
        current_yaw=float(np.arctan2(measured[1,0],measured[0,0]))
        floor=self.env.placement_surface('basket')['top_z']
        goal=np.r_[self.placement_xy,floor+.019]
        preferred=[]
        try:
            q=upright_solution(self,goal,.0015)
            self.ik_seeds[self.side].insert(0,q.copy())
            preferred=[self.result['last_task_placement_pose']['yaw_rad']-current_yaw]
        except PlanningFailure as exc:
            if exc.code in ('cancelled','planning_timeout'): raise
            self.result['relay_placement']['free_yaw_failure']=exc.code
        return tuple(dict.fromkeys([*preferred,*super()._placement_yaws(),*np.linspace(-np.pi,np.pi,49)[:-1]]))

    def _ik(self, target, allow_fingers=False, **kwargs):
        q = super()._ik(target, allow_fingers, **kwargs)
        if (self.phase != 'place' or self.carried is None or self._executing_move
                or kwargs.get('position_tolerance', .0015) <= .0002):
            return q
        self._at(q)
        bottom = min(bounds(self.env.model, self.data, geom)[0][2] for geom in self.env.bottle_geoms)
        floor = self.env.placement_surface('basket')['top_z']
        if bottom > floor + .040:
            self._path(q, allow_fingers)
            self._proposal_high = q.copy()
            self.ik_seeds[self.side].insert(0, q.copy())
            self.result['relay_placement']['high_q_rad'] = q.tolist()
        else:
            high = self._proposal_high
            passed = high is not None and self._edge(high, q, allow_fingers)
            if passed:
                count=max(1,int(np.ceil(np.max(abs(q-high))/.025)))
                for index in range(count+1):
                    self._at(high+(q-high)*index/count)
                    if self.data.xmat[self.env.bottle_id].reshape(3,3)[2,2] < .98:
                        passed=False
                        break
            if not passed:
                self.result['planning_diagnostics'] = {'location': 'connected_basket_descent',
                    'blocked_pairs': [copy.deepcopy(self.last_collision)]}
                raise PlanningFailure('lowering_path_rejected', 'No connected collision-free loaded basket descent')
            self.result['relay_placement'].update(low_q_rad=q.tolist(), connected_upright_descent=True)
        return q

    def _move(self, target, allow_fingers=False, **kwargs):
        previous = self._executing_move
        self._executing_move = True
        try:
            yield from super()._move(target, allow_fingers, **kwargs)
        finally:
            self._executing_move = previous


def compose_planner(previous):
    class Router:
        CONTRACT = {**previous.CONTRACT, 'shape_aware_basket': 'optional shape-aware basket clearance'}

        def __new__(cls, env, task, **kwargs):
            params = kwargs.get('parameters') or {}
            if (env.object_id=='bottle' and task=='handover_bottle'
                    and params.get('placement_aware') is True):
                from .free_transport import FreeBottleHandover
                from .bottle_routing import ContinuousHoldAudit
                kwargs={**kwargs,'parameters':{**params,'next_place':{'surface':'basket',
                    'target_xy_m':env.data.xpos[env.box_id,:2].tolist()}}}
                planner=FreeBottleHandover(env,task,**kwargs)
                return ContinuousHoldAudit(planner,params['hold_seconds']) if 'hold_seconds' in params else planner
            if (env.object_id == 'bottle' and task == 'place_bottle'
                    and params.get('receptacle_id') == 'basket' and params.get('shape_clearance') is True):
                return RelayBasketPlacement(env, task, **kwargs)
            return previous(env, task, **kwargs)
    return Router
