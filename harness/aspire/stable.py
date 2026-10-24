"""Right donor payload support and paired TCP/command diagnostics, local only.

Keeps the admitted v4 provider available. No mass, gain, friction, joint state,
collision test, time budget or grasp-success predicate is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .placement import PlacementWorld
from .runtime import ROOT, dump
from sim.tamp import PlanningFailure


class StableWorld(PlacementWorld):
    def __init__(self, case, output):
        super().__init__(case, output)
        self.right_offset = np.zeros(7)
        self.tracking_rows = []
        self.scratch = mujoco.MjData(self.env.model)
        self.source_hash = hashlib.sha256(Path(__file__).read_bytes() +
            self.source_hash.encode()).hexdigest()

    def _step(self, action, diagnostic):
        env = self.env
        nominal = np.array(action, copy=True)
        if np.array_equal(nominal, env.last_action):
            nominal[8:15] -= self.right_offset
            nominal[:7] -= self.payload_offset
        target = np.zeros(7)
        # The donor carries the load until release is actually measured. The
        # existing left compensation starts only after independent left hold.
        if self.case.get('right_payload_compensation', True) and env.evaluation['held']['right']:
            jac = np.zeros((3, env.model.nv)); rot = np.zeros_like(jac)
            mujoco.mj_jac(env.model, env.data, jac, rot,
                          env.data.xipos[env.bottle_id], env.ee_ids['right'])
            force = -float(env.model.body_subtreemass[env.bottle_id]) * env.model.opt.gravity
            target = (jac[:, env.dids['right']].T @ force) / env.model.actuator_gainprm[env.aids['right'], 0]
            if np.max(abs(target)) > .20:
                raise PlanningFailure('payload_compensation_limit', 'Donor offset exceeds0.20rad')
        self.right_offset += np.clip(target - self.right_offset, -.05, .05)
        requested = self.right_offset.copy()
        self.right_offset = np.clip(self.right_offset,
            env.action_space.low[8:15] - nominal[8:15], env.action_space.high[8:15] - nominal[8:15])
        applied = nominal.copy(); applied[8:15] += self.right_offset
        done = super()._step(applied, {**diagnostic,
            'right_payload_offset_rad': self.right_offset.tolist(),
            'right_payload_saturation_rad': (requested - self.right_offset).tolist(),
            'nominal_right_target_rad': nominal[8:15].tolist()})
        # BodyWorld removes its left offset; also remove ours before a new
        # planner receives base_action. Repeated dwell must never accumulate it.
        self.nominal[8:15] -= self.right_offset
        self.scratch.qpos[:] = env.data.qpos
        for side, offset in [('left', 0), ('right', 8)]:
            self.scratch.qpos[env.qids[side]] = self.nominal[offset:offset+7]
        mujoco.mj_kinematics(env.model, self.scratch)
        row = {'tick': env.tick, 'phase': diagnostic.get('phase'), 'kind': self.active['kind']}
        for side in ('left', 'right'):
            body = env.ee_ids[side]; local = np.array([0., .006, .075])
            actual = env.data.xpos[body] + env.data.xmat[body].reshape(3, 3) @ local
            commanded = self.scratch.xpos[body] + self.scratch.xmat[body].reshape(3, 3) @ local
            row[side] = {'held': bool(env.evaluation['held'][side]),
                'tcp_m': actual.tolist(), 'nominal_tcp_m': commanded.tolist(),
                'vertical_tracking_error_m': float(commanded[2] - actual[2]),
                'joint_reference_offset_rad': (self.right_offset if side == 'right' else self.payload_offset).tolist(),
                'object_in_hand_m': (env.data.xmat[body].reshape(3, 3).T @
                    (env.data.xpos[env.bottle_id] - env.data.xpos[body])).tolist()}
        self.tracking_rows.append(row)
        return done

    def result(self):
        result = super().result()
        hold = [r['right'] for r in self.tracking_rows
                if r['kind'] == 'lift' and r['phase'] == 'hold' and r['right']['held']]
        residual = max([0., *[r['vertical_tracking_error_m'] for r in hold]])
        # Report actual decline separately from absolute target error. A moving
        # Cartesian target is not mislabeled as passive sag.
        decline = max([0., *[hold[0]['tcp_m'][2] - r['tcp_m'][2] for r in hold]]) if hold else None
        drift = max([abs(r['nominal_tcp_m'][2]-hold[0]['nominal_tcp_m'][2]) for r in hold], default=0.)
        passed = bool(hold) and drift < .0001 and residual <= .005 and decline <= .005
        result['donor_support'] = {'enabled': self.case.get('right_payload_compensation', True),
            'right_hold_samples': len(hold), 'max_vertical_tracking_error_m': residual,
            'max_stationary_decline_m': decline, 'command_height_drift_m': drift,
            'limit_m': .005, 'passed': passed,
            'source': 'candidate simulator gravity/Jacobian; not a real controller calibration',
            'physical_parameters_changed': False, 'hardware_access': False}
        result['task_success'] = bool(result['task_success'] and passed)
        return result

    def close(self):
        dump(self.output / 'tracking.json', self.tracking_rows)
        super().close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--program', type=Path, required=True); a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(ROOT / 'runs'): p.error('local output belongs in runs/')
    from .recovery_campaign import run_trial
    run_trial(json.loads(a.case.read_text()), out, a.program, 'frozen', world_type=StableWorld)
