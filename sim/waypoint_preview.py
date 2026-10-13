"""Isolated reference-command preview for the audited SourceMIT and waypoint adapter.

The production PD predictor is delegated without changing its implementation.
No simulated state or controller history is ever restored into the live owner.
"""
import copy
import time
import mujoco
import numpy as np

from sim.waypoint_control import WaypointAdapter, feasible_command
from sim.controller import SourceMIT
from sim.environment import OpenArmEnv
from sim.model_evidence import ParameterAudit
from sim.predictive_place import NativePreview as OriginalPreview
from sim.tamp import PlanningFailure


class RetimedPreview(OriginalPreview):
    def __init__(self, env):
        self.env = env
        self.original = None
        self.source_model = env.model
        if env.source_controller is None:
            self.original = OriginalPreview(env)
            self.model = env.model
            self.data = self.original.data
            return
        self._supported()
        if env.model.nplugin:
            raise PlanningFailure('preview_plugin_unsupported', 'Stateful plugins are not covered')
        # Full 258MB assets are copied once, not per servo proposal. Existing
        # ParameterAudit covers small mutable arrays and options; bulk assets
        # have no live mutation tool and require a newly compiled environment.
        self.audit = ParameterAudit(env.model)
        self.parameter_fields = self.audit.fields()
        self.model = copy.deepcopy(env.model)
        self.data = mujoco.MjData(self.model)
        self.command_context = None

    def _supported(self):
        env = self.env
        if (type(env.source_controller) is not SourceMIT
                or env.controller_mode != 'source_mit_candidate'
                or env.profile['gravity_compensation'] != 'none'
                or mujoco.get_mjcb_control() is not None
                or env._set_command.__func__ is not OpenArmEnv._set_command):
            raise PlanningFailure('preview_controller_unsupported',
                'Only the reviewed SourceMIT and original native command implementation are supported')

    def run(self, action, side, *, seconds=.15, guard=True):
        if self.original is not None:
            if self.env.source_controller is not None:
                raise PlanningFailure('preview_controller_changed', 'Rebuild preview after controller replacement')
            return self.original.run(action, side, seconds=seconds, guard=guard)
        env = self.env
        started = time.monotonic()
        self._supported()
        if side not in ('left', 'right'):
            raise ValueError('unknown preview arm')
        if env.model is not self.source_model or self.audit.fields() != self.parameter_fields:
            raise PlanningFailure('preview_model_changed', 'Native parameters changed; rebuild isolated predictor')
        if env._clock_mode not in (None, 'periodic'):
            raise PlanningFailure('preview_clock_unsupported', 'Source preview requires the current periodic command clock')
        dt = float(self.model.opt.timestep)
        if not np.isfinite(seconds) or not dt <= seconds <= .2:
            raise ValueError('Preview horizon must be1physics step to200ms')
        # Validate with the original live-state contract before copying anything.
        action = env.validate_action(action, native_fingers=True)
        # The affine feedforward column is volatile command state, explicitly
        # excluded from ParameterAudit. Copy it from the owner at every call.
        self.model.actuator_biasprm[:, 0] = env.model.actuator_biasprm[:, 0]
        mujoco.mj_copyData(self.data, self.model, env.data)
        context = copy.copy(env)
        context.model = self.model
        context.data = self.data
        context.source_controller = copy.deepcopy(env.source_controller)
        context.controller_feedback = copy.deepcopy(env.controller_feedback)
        self.command_context = context
        initial_calls = dict(context.source_controller.calls)
        data = self.data
        finish = float(data.time) + round(seconds/dt)*dt
        tick = env.tick
        offset = getattr(env, '_periodic_offset', 0.)
        samples = 0
        native_commands = []
        reference_adapter = None
        segment_anchor = env.waypoint_adapter.reference.copy()
        retiming_receipts = []
        failure = None
        minimum_upright = 1.
        maximum_drift = 0.

        def check():
            nonlocal failure, minimum_upright, maximum_drift
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                failure = {'kind':'non_finite_native_prediction'}
                return
            upright = float(data.xmat[env.bottle_id].reshape(3,3)[2,2])
            minimum_upright = min(minimum_upright, upright)
            if not guard:
                return
            pair = self.collision(side)
            if pair:
                failure = {'kind':'native_collision', **pair}
                return
            if upright < .98:
                failure = {'kind':'carried_axis_constraint', 'upright_cosine':upright}
                return
            for arm in ('left', 'right'):
                limits = np.asarray([self.model.joint(f'openarm_{arm}_joint{i}').range for i in range(1,8)])
                q = data.qpos[env.qids[arm]]
                if np.max(np.maximum(limits[:,0]-q, q-limits[:,1])) > .01:
                    failure = {'kind':'native_joint_guard', 'arm':arm}
                    return
            for ident, origin in getattr(env, 'protected_objects', {}).items():
                drift = float(np.linalg.norm(data.xpos[env.registry[ident]['body']] - origin))
                maximum_drift = max(maximum_drift, drift)
                if drift > .003:
                    failure = {'kind':'non_target_object_displaced', 'object_id':ident, 'distance_m':drift}
                    return

        check()
        while failure is None and data.time + dt/2 < finish:
            tick += 1
            endpoint = min(finish, tick/env.profile['control_hz'] + offset)
            if endpoint <= data.time + dt/2:
                raise ValueError('Preview requires the existing periodic control clock')
            try:
                # Use reviewed original command arithmetic, not a second force
                # formula. Validate each prospective measured-state jump too.
                if reference_adapter is None:
                    reference_adapter = WaypointAdapter(context)
                    reference_adapter.reference = env.waypoint_adapter.reference.copy()
                native, receipt, complete = feasible_command(reference_adapter, action, anchor=segment_anchor)
                retiming_receipts.append(receipt['retiming'])
                context._set_command(native)
                context.last_action = native.copy()
                native_commands.append(native.tolist())
            except (ValueError, PlanningFailure) as exc:
                failure = {'kind':'native_command_rejected', 'message':str(exc)}
                break
            while failure is None and data.time + dt/2 < endpoint:
                mujoco.mj_step(self.model, data)
                samples += 1
                check()
            if failure is None:
                mujoco.mj_forward(self.model, data)
                check()
        return {'accepted':failure is None, 'failure':failure, 'requested_horizon_s':seconds,
                'integrated_seconds':float(data.time-env.data.time), 'physics_steps':samples,
                'minimum_upright_cosine':minimum_upright, 'maximum_non_target_drift_m':maximum_drift,
                'seconds':time.monotonic()-started,
                'source':'copied native1ms dynamics and copied audited SourceMIT history',
                'controller_calls':{s:context.source_controller.calls[s]-initial_calls[s] for s in initial_calls},
                'future_command_schedule':'retime pending geometric reference using prospective history and recompute native motor target at each original30Hz endpoint; final partial interval truncated',
                'input_semantics':'nominal TAMP joint reference, not replay or ACT native motor command',
                'prospective_native_commands':native_commands,
                'prospective_retiming':retiming_receipts,
                'model_copy':'once at construction; dynamic bias column synchronized per proposal; small mutable parameter changes rejected',
                'live_state_written':False, 'hardware_access':False, 'transfer_ready':False}
