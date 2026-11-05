"""Optional bounded native settling before an independent hold audit.

Settling is extra physical execution, not a skipped failed audit. A complete
stable two-second current window is required, followed by the unchanged caller
hold audit. Contact loss, invalid physics or timeout reject immediately.
"""
from collections import deque
import math
import numpy as np
from .decision_timing import GraspStability
from .tamp import PlanningFailure


class HoldSettling:
    def __init__(self, env, side, *, maximum_seconds=4.):
        if side not in ('left', 'right') or maximum_seconds != 4.:
            raise ValueError('declared arm and4s settling budget required')
        if not env.evaluation['held'][side] or env.evaluation['held']['right' if side == 'left' else 'left']:
            raise PlanningFailure('single_hold_required', 'Settling needs independent native two-finger contact')
        self.env, self.side = env, side
        self.started = float(env.data.time); self.maximum_seconds = maximum_seconds
        self.rows = deque(); self.measurements = []; self.ready = False
        self.reference = env.waypoint_adapter.reference.copy() if env.source_controller is not None else env.last_action.copy()

    def read(self):
        env = self.env
        if env._done or env.evaluation['invalid']:
            raise PlanningFailure('settling_task_ended', 'Original task became invalid or exhausted during extra settling')
        if not env.evaluation['held'][self.side]:
            raise PlanningFailure('grasp_lost', 'Native contact lost during settling; cannot start hold audit')
        body = env.ee_ids[self.side]; rotation = env.data.xmat[body].reshape(3,3)
        self.rows.append((float(env.data.time), rotation.T@(env.data.xpos[env.bottle_id]-env.data.xpos[body]),
            rotation.T@env.data.xmat[env.bottle_id].reshape(3,3)))
        while len(self.rows) > 1 and self.rows[-1][0]-self.rows[1][0] >= GraspStability.window_seconds-1e-6:
            self.rows.popleft()
        span = self.rows[-1][0]-self.rows[0][0]
        if span < GraspStability.window_seconds-1e-6: return
        positions = np.asarray([r[1] for r in self.rows]); matrices = np.asarray([r[2] for r in self.rows]).reshape(len(self.rows),9)
        drift = float(np.max(np.linalg.norm(positions[:,None]-positions[None,:],axis=2)))
        angle = float(np.arccos(np.clip((np.min(matrices@matrices.T)-1)/2,-1,1)))
        self.ready = drift <= GraspStability.position_limit_m and angle <= GraspStability.rotation_limit_rad
        self.measurements.append({'elapsed_s':float(env.data.time)-self.started, 'window_s':span,
            'relative_drift_m':drift,'relative_rotation_rad':angle,'current_window_ready':self.ready})

    def actions(self):
        self.read()
        for index in range(math.ceil(self.maximum_seconds*self.env.profile['control_hz'])):
            if self.ready: return
            # This is preparation, not the caller's independent hold audit.
            # The outer ContinuousHoldAudit must start only after we finish.
            yield self.reference.copy(), {'phase':'pre-hold-settling', 'settling_validation':'native_contact_and_relative_pose',
                'settling_command_index':index,'extra_native_time_counted':True,
                'scope':'settling only; no independent10s hold success yet'}
            self.read()
        if self.ready: return
        raise PlanningFailure('grasp_settling_timeout', 'No complete stable native window within declared4s')

    def summary(self):
        return {'schema':'native-pre-hold-settling/1','arm':self.side,'ready_for_independent_audit':self.ready,
            'actual_seconds':float(self.env.data.time)-self.started,'maximum_seconds':self.maximum_seconds,
            'required_window_seconds':GraspStability.window_seconds,
            'limits':{'relative_position_m':GraspStability.position_limit_m,'relative_rotation_rad':GraspStability.rotation_limit_rad},
            'windows':self.measurements,'original_hold_audit_still_required':True,
            'source':'native relative-pose and sustained contact; oracle simulation diagnostic',
            'hardware_access':False,'transfer_ready':False}


def execute(env, side, apply, record):
    settling = HoldSettling(env, side)
    try:
        for reference, diagnostic in settling.actions(): apply(reference, diagnostic, False)
    finally: record(settling.summary())
