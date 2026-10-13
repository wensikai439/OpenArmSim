"""Isolated object-selected native environment; no changes to the main cell.

New assets are declared synthetic development objects, never measurements.
The legacy planner's body/geoms aliases are bound to one registered free body;
selected-joint integration and exact primitive bounds replace bottle-only
assumptions. Selection changes bookkeeping only, never qpos/qvel/actuators.
"""
import copy
from unittest.mock import patch

import mujoco
import numpy as np

from . import environment as environment_module
from .environment import OpenArmEnv, SIDES
from .contacts import Contacts
from .grasp_feedback import GraspFeedback
from .scene import SceneDirectory, bounds
from .placement import placement_tolerance
from tools.contact_candidate import CandidateSpec, DERIVATION

ASSETS = [
    {'object_id': 'block', 'label': '长方体', 'type': 'box', 'size_m': [.024, .021, .09],
        'geom_pos_m': [0., 0., .09], 'position_m': [.49, .075, 1.005], 'mass_kg': .20,
        'rgba': [.57, .65, .58, 1.]},
    {'object_id': 'can', 'label': '圆柱', 'type': 'cylinder', 'size_m': [.024, .09, 0.],
        'geom_pos_m': [0., 0., .09], 'position_m': [.503, -.345, 1.005], 'mass_kg': .20,
        'rgba': [.65, .49, .36, 1.]},
]
CONTRACT = {'schema': 'registered-object-cell/1', 'source': 'simulator_ground_truth',
    'assets': 'explicit synthetic development primitives; not reconstructed real objects',
    'selection': 'one registered free rigid body; forbidden while any object touches a gripper',
    'contact_acceptance': 'original two-finger force >0.05N and 100ms dwell; lift >=60mm',
    'placement_acceptance': 'original support/release/whole footprint/upright/1s stability gates',
    'contact_candidate': DERIVATION, 'hardware_access': False, 'transfer_ready': False}


class ObjectEnv(OpenArmEnv):
    def __init__(self, *, assets=None, **kwargs):
        assets = copy.deepcopy(ASSETS if assets is None else assets)
        names = {row['object_id'] for row in assets}
        if len(names) != len(assets) or names & {'bottle', 'basket', 'table'}:
            raise ValueError('duplicate/reserved development object ID')
        class Spec:
            @staticmethod
            def from_string(*args, **kw):
                spec = CandidateSpec.from_string(*args, **kw)
                for row in assets:
                    body = spec.worldbody.add_body(name='task_' + row['object_id'], pos=row['position_m'])
                    body.add_freejoint(name=row['object_id'] + '_free')
                    body.add_geom(name=row['object_id'] + '_shape',
                        type={'box': mujoco.mjtGeom.mjGEOM_BOX, 'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER}[row['type']],
                        size=row['size_m'], pos=row['geom_pos_m'], mass=row['mass_kg'], rgba=row['rgba'],
                        friction=[1., .01, .01], condim=4, contype=1, conaffinity=1)
                return spec
        class Compiler:
            MjSpec = Spec
            def __getattr__(self, name): return getattr(mujoco, name)
        profile = dict(kwargs.pop('profile', {}) or {})
        profile.update(contact_model='finite_patch_candidate', contact_derivation=DERIVATION)
        if assets: profile['object_registry'] = {**CONTRACT, 'declared_assets': assets}
        self.object_id = 'bottle'
        with patch.object(environment_module, 'mujoco', Compiler()):
            super().__init__(profile=profile, **kwargs)
        self.directory = SceneDirectory(self)
        self.registry = {}
        for ident, row in self.directory.rows.items():
            if 'free_joint' not in row: continue
            body = row['body']; joint = self.model.joint(row['free_joint'])
            geoms = [g for g in np.flatnonzero(self.model.geom_bodyid == body)
                if self.model.geom_contype[g] or self.model.geom_conaffinity[g]]
            if not geoms: raise ValueError('object lacks native collision geometry')
            self.registry[ident] = {'body': body, 'joint': joint.id, 'qposadr': int(joint.qposadr[0]),
                'dofadr': int(joint.dofadr[0]), 'geoms': np.asarray(geoms),
                'contacts': Contacts(self.model, body, self.box_floor_id, self.table_id)}
        self.registry['bottle']['contacts'] = self.contacts
        self.asset_contract = {**CONTRACT, 'declared_assets': assets}

    def reset(self, **kwargs):
        self.object_id = 'bottle'; row = self.registry['bottle']
        self.bottle_id = row['body']; self.bottle_geoms = row['geoms']; self.contacts = row['contacts']
        for row in self.registry.values(): row['contacts'].reset()
        return super().reset(**kwargs)

    def select_object(self, ident):
        if ident not in self.registry: raise ValueError('unregistered free object')
        if ident == self.object_id: return self.selection()
        if self._done: raise ValueError('finished task cannot change object identity')
        if any(any(row['contacts'].read(self.data)['touching'].values()) for row in self.registry.values()):
            raise ValueError('a gripper touches a scene object; retain its identity until released')
        before = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(), self.data.time,
            self.last_action.copy())
        row = self.registry[ident]
        self.object_id = ident; self.bottle_id = row['body']; self.bottle_geoms = row['geoms']; self.contacts = row['contacts']
        self.contacts.reset(); self.placement_target = None; self.grasp_feedback = GraspFeedback()
        self._dwell = self._bad_dwell = self._max_penetration = self._interval_penetration = 0.
        self.evaluation = self._evaluate(0.); self.grasp_feedback.read(self)
        for old, new in zip(before[:3], (self.data.qpos, self.data.qvel, self.data.ctrl)):
            np.testing.assert_array_equal(old, new)
        assert before[3] == self.data.time
        np.testing.assert_array_equal(before[4], self.last_action)
        return self.selection()

    def selection(self):
        row = self.registry[self.object_id]
        return {'object_id': self.object_id, 'body': self.model.body(row['body']).name,
            'free_joint': self.model.joint(row['joint']).name, 'geometry': [self.model.geom(g).name for g in row['geoms']],
            'source': 'registered native object identity; no physical state changed', 'transfer_ready': False}

    def manifest(self):
        value = super().manifest()
        if hasattr(self, 'registry'):
            value['selected_object'] = self.selection(); value['object_contract'] = self.asset_contract
        return value

    def _bottle_bounds(self):
        if self.object_id == 'bottle': return super()._bottle_bounds()
        boxes = [bounds(self.model, self.data, g) for g in self.bottle_geoms]
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)

    def set_placement_target(self, surface, xy, tolerance_m=None):
        if self.object_id == 'bottle': return super().set_placement_target(surface, xy, tolerance_m)
        xy = np.asarray(xy, dtype=float); area = self.placement_surface(surface)
        lo, hi = self._bottle_bounds(); origin = self.data.xpos[self.bottle_id]
        margin = np.maximum(origin[:2] - lo[:2], hi[:2] - origin[:2]) + .010
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.any(xy < area['xy_min'] + margin) or np.any(xy > area['xy_max'] - margin):
            raise ValueError('selected object footprint exceeds support surface')
        tolerance = placement_tolerance(surface, xy) if tolerance_m is None else tolerance_m
        if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or not .0005 <= tolerance <= placement_tolerance(surface, xy):
            raise ValueError('invalid object placement tolerance')
        self.placement_target = {'object_id': self.object_id, 'surface': surface, 'xy_m': xy.tolist(), 'tolerance_m': float(tolerance)}
        self.contacts.placement_seconds = 0.

    def _advance(self, action, target, mode, *, native_fingers=False):
        if self.object_id == 'bottle': return super()._advance(action, target, mode, native_fingers=native_fingers)
        # Same native integration/control/force/contact gates as the base cell.
        # The sole changed data address is the SELECTED object's free velocity.
        if self._done: raise RuntimeError('episode finished; reset before stepping')
        if self._clock_mode not in (None, mode): raise ValueError('reset before switching control clock')
        if not np.isfinite(target) or not self.model.opt.timestep / 2 < target - self.data.time <= .25:
            raise ValueError('target time must advance by at most250ms')
        applied = self.validate_action(action, native_fingers=native_fingers)
        self._clock_mode = mode; self._set_command(applied)
        before = float(self.data.time); self.contacts.begin(before); self._interval_penetration = 0.; self.tick += 1
        did = self.registry[self.object_id]['dofadr']
        while self.data.time + self.model.opt.timestep / 2 < target:
            timestamp = float(self.data.time); velocity = self.data.qvel[did:did + 6].copy()
            if self.profile['gravity_compensation'] == 'bias':
                for side in SIDES: self.data.qfrc_applied[self.dids[side]] = self.data.qfrc_bias[self.dids[side]]
            mujoco.mj_step(self.model, self.data)
            contact = self.contacts.read(self.data)
            self.contacts.sample(contact, float(self.model.opt.timestep), timestamp,
                stable=self._stable_placement(contact, velocity), placement_stable=self._placement_condition(contact, velocity))
        mujoco.mj_forward(self.model, self.data); contact = self.contacts.read(self.data)
        velocity = self.data.qvel[did:did + 6]
        self.contacts.sample(contact, 0., float(self.data.time), stable=self._stable_placement(contact, velocity),
            placement_stable=self._placement_condition(contact, velocity))
        self._interval_penetration = self.contacts.penetration; self.last_action = applied.copy()
        if not np.isfinite(self.data.qpos).all(): raise RuntimeError('non-finite physics state')
        self.evaluation = self._evaluate(float(self.data.time) - before); self.grasp_feedback.read(self)
        terminated = self.evaluation['invalid']
        truncated = self.data.time + self.model.opt.timestep / 2 >= self._task_start + self._task_budget
        self._done = terminated or truncated
        return self.observe(), 0., terminated, truncated, self.info()

    def _evaluate(self, dt):
        if self.object_id == 'bottle': return super()._evaluate(dt)
        contact_held, touching, supported, penetration, force, robot_contacts = self._contacts()
        held = {side: self.contacts.hold_seconds[side] >= .1 and contact_held[side] for side in SIDES}
        penetration = max(penetration, self._interval_penetration); lo, hi = self._bottle_bounds()
        box = self.data.xpos[self.box_id]
        inside = bool(np.all(lo[:2] > box[:2] - [.100, .084]) and np.all(hi[:2] < box[:2] + [.100, .084]))
        did = self.registry[self.object_id]['dofadr']; velocity = self.data.qvel[did:did + 6]
        self._max_penetration = max(self._max_penetration, penetration)
        invalid = bool(lo[2] < .80 or self.contacts.deep_penetration)
        reasons = (['selected object fell below work surface'] if lo[2] < .80 else []) + (
            ['penetration exceeded10mm continuously for100ms'] if self.contacts.deep_penetration else [])
        return {'object_id': self.object_id, 'success': False, 'legacy_score_applicable': False, 'invalid': invalid,
            'held': held, 'touching': touching, 'failure_reasons': reasons, 'inside_basket': inside,
            'supported_by_basket': supported, 'stable_seconds': round(self.contacts.stable_seconds, 4),
            'linear_speed_m_s': float(np.linalg.norm(velocity[:3])), 'angular_speed_rad_s': float(np.linalg.norm(velocity[3:])),
            'penetration_m': penetration, 'max_penetration_m': self._max_penetration,
            'peak_normal_force_N': max(force, self.contacts.peak_force), 'robot_contact_count': robot_contacts,
            'contact_window': self.contacts.summary(), 'placement': {**(self.placement_target or {}),
                'stable_seconds': round(self.contacts.placement_seconds, 4),
                'confirmed': bool(self.placement_target and self.contacts.placement_seconds >= 1. and not invalid)},
            'evidence_source': 'selected-object native contact evaluator; not camera measurement', 'transfer_ready': False}

    def diagnostics(self):
        value = super().diagnostics()
        scene = value['manipulation_scene']; scene['object_id'] = self.object_id
        scene['selection'] = self.selection(); scene['world_bounds'] = {'min_m': self._bottle_bounds()[0].tolist(),
            'max_m': self._bottle_bounds()[1].tolist()}
        value['selected_object_world'] = value.pop('bottle_world')
        return value
