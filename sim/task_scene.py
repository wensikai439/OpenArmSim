"""Passive all-object placement evidence; original integration stays intact."""
import copy

import numpy as np

from harness.object_handover import ProtectedObjectEnv
from .scene import bounds
from .contacts import Contacts


class TaskScene(ProtectedObjectEnv):
    def __init__(self, *args, **kwargs):
        self.rest_seconds = {}; self.object_rest = {}; self.task_goal_ledger = None
        super().__init__(*args, **kwargs)
        self.geom_objects = {int(g): ident for ident, row in self.registry.items() for g in row['geoms']}
        self.finger_geoms = {}
        for geom in range(self.model.ngeom):
            body = self.model.body(int(self.model.geom_bodyid[geom])).name or ''
            for arm in ('left', 'right'):
                if 'openarm_' + arm + '_' in body and 'finger' in body: self.finger_geoms[geom] = arm
        # The normal selected-body sampler omits other bodies' table contacts.
        # This separate READ-ONLY sampler includes every registered free body.
        self.all_contacts = Contacts(self.model)
        self.all_contacts.is_bottle = np.isin(self.model.geom_bodyid, [row['body'] for row in self.registry.values()])
        # Consume solved evidence at each original substep; no controller,
        # force, geometry, integration or acceptance limit is changed.
        for row in self.registry.values():
            previous = row['contacts'].sample
            def sample(contact, dt, timestamp, *, stable=False, placement_stable=False, original=previous):
                original(contact, dt, timestamp, stable=stable, placement_stable=placement_stable)
                self._sample_objects(self.all_contacts.read(self.data), dt)
            row['contacts'].sample = sample

    def reset(self, **kwargs):
        self.rest_seconds = {}; self.object_rest = {}; self.task_goal_ledger = None
        return super().reset(**kwargs)

    def declare_goals(self, root, goals):
        if self.task_goal_ledger and self.task_goal_ledger['root'] == root:
            raise ValueError('original native task goal ledger cannot be replaced')
        self.task_goal_ledger = {'root': root, 'goals': copy.deepcopy(goals),
            'start_sim_time': float(self.data.time), 'task_start_sim_time': self._task_start,
            'task_budget_s': self._task_budget}
        self.rest_seconds = {}; self.object_rest = {}

    def _sample_objects(self, contact, dt):
        table = self.placement_surface('table'); basket = self.placement_surface('basket')
        states = {ident: {'table_supported': False, 'basket_supported': False, 'touching': {'left': False, 'right': False}}
            for ident in self.registry}
        for (a, b), pair in contact['pairs'].items():
            if pair['normal_force_N'] <= .05: continue
            for obj, other in ((a, b), (b, a)):
                ident = self.geom_objects.get(int(obj))
                if ident is None: continue
                states[ident]['table_supported'] |= other == self.table_id
                states[ident]['basket_supported'] |= other == self.box_floor_id
                if other in self.finger_geoms: states[ident]['touching'][self.finger_geoms[other]] = True
        for ident, row in self.registry.items():
            state = states[ident]; body = row['body']; velocity = self.data.qvel[row['dofadr']:row['dofadr'] + 6]
            boxes = [bounds(self.model, self.data, int(geom)) for geom in row['geoms']]
            lo = np.min([box[0] for box in boxes], axis=0); hi = np.max([box[1] for box in boxes], axis=0)
            upright = bool(self.data.xmat[body].reshape(3, 3)[2, 2] >= .98)
            quiet = np.linalg.norm(velocity[:3]) < .02 and np.linalg.norm(velocity[3:]) < .1 and not any(state['touching'].values())
            for surface, area in (('table', table), ('basket', basket)):
                ready = (quiet and state[surface + '_supported'] and (surface != 'table' or upright)
                    and np.all(lo[:2] > area['xy_min']) and np.all(hi[:2] < area['xy_max']))
                goal = next((goal for goal in (self.task_goal_ledger or {}).get('goals', []) if goal['object_id'] == ident), None)
                if goal:
                    ready = ready and goal['surface'] == surface and np.linalg.norm(self.data.xpos[body, :2] - goal['target_xy_m']) <= goal['tolerance_m']
                key = (ident, surface)
                self.rest_seconds[key] = self.rest_seconds.get(key, 0.) + dt if ready else 0.
            self.object_rest[ident] = {**state, 'object_id': ident, 'origin_m': self.data.xpos[body].tolist(),
                'upright': upright, 'world_bounds': {'min_m': lo.tolist(), 'max_m': hi.tolist()},
                'linear_speed_m_s': float(np.linalg.norm(velocity[:3])), 'angular_speed_rad_s': float(np.linalg.norm(velocity[3:])),
                'stable_seconds': {surface: self.rest_seconds.get((ident, surface), 0.) for surface in ('table', 'basket')}}

    def diagnostics(self):
        value = super().diagnostics()
        value['task_objects'] = {'source': 'native contact pairs and free-body velocity at every original solved physics substep',
            'objects': copy.deepcopy(self.object_rest), 'transfer_ready': False,
            'goal_ledger': copy.deepcopy(self.task_goal_ledger),
            'scope': 'passive multi-object support/release/stability evidence; not visual or physical calibration'}
        return value
