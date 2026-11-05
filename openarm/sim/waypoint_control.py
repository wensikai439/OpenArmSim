"""Opt-in TAMP references for the audited SourceMIT simulator candidate.

Normal policy/replay motor commands and default PD execution are unchanged.
Retiming keeps every original nominal waypoint pending, obeys original motor
and force limits and native task budgets, and never writes physical state.
Use install_isolated() only inside a separate explicitly configured process.
This adapter is a control design, not identified physical parameters.
"""
import copy
import time
import mujoco
import numpy as np
from sim.controller import SourceMIT
from sim.environment import OpenArmEnv
from sim.tamp import PlanningFailure, TampPlanner

class WaypointAdapter:
    def __init__(self, env):
        if (type(env.source_controller) is not SourceMIT or env.controller_mode!='source_mit_candidate'
                or env.profile['gravity_compensation']!='none'
                or env._set_command.__func__ is not OpenArmEnv._set_command
                or mujoco.get_mjcb_control() is not None):
            raise PlanningFailure('tracking_controller_unsupported','Only reviewed native SourceMIT is supported')
        self.env=env
        self.reference=env.last_action.copy()
        self.last_receipt=None

    @staticmethod
    def effort(target,q,velocity,kp,kd,previous,accumulated,limits):
        error=np.clip(target-q,-.3,.3)
        total=kp*(target-q)+kd*(error-previous-velocity)+np.clip(accumulated+error,-10.,10.)
        return np.clip(total,limits[:,0],limits[:,1])

    def map(self, reference):
        env=self.env
        reference=env.validate_action(reference,native_fingers=True)
        output=reference.copy()
        arms={}
        for side,start in [('left',0),('right',8)]:
            aids=env.aids[side];q=env.data.qpos[env.qids[side]];v=env.data.qvel[env.dids[side]]
            kp=env.model.actuator_gainprm[aids,0];kd=-env.model.actuator_biasprm[aids,2]
            if np.any(kp<=0):raise PlanningFailure('tracking_gain_unsupported','Position stiffness must be positive')
            previous=env.source_controller.previous[side]
            accumulated=env.source_controller.accumulated[side]
            force_limits=env.model.actuator_forcerange[aids]
            # Reference is a geometric waypoint. Request the current model's
            # local position-PD effort + bias; this changes motor commands, not
            # parameters. The old PD path clips PD before adding its bias.
            desired=np.clip(kp*(reference[start:start+7]-q)-kd*v,
                            force_limits[:,0],force_limits[:,1])+env.data.qfrc_bias[env.dids[side]]
            low=env.action_space.low[start:start+7].astype(float)
            high=env.action_space.high[start:start+7].astype(float)
            lower_effort=self.effort(low,q,v,kp,kd,previous,accumulated,force_limits)
            upper_effort=self.effort(high,q,v,kp,kd,previous,accumulated,force_limits)
            if np.any(desired<lower_effort-1e-8) or np.any(desired>upper_effort+1e-8):
                raise PlanningFailure('tracking_effort_unreachable','Waypoint effort is unavailable inside original joint/force limits')
            for _ in range(40):
                midpoint=(low+high)*.5
                value=self.effort(midpoint,q,v,kp,kd,previous,accumulated,force_limits)
                below=value<desired
                low=np.where(below,midpoint,low);high=np.where(below,high,midpoint)
            target=((low+high)*.5).astype(np.float32)
            actual=self.effort(target,q,v,kp,kd,previous,accumulated,force_limits)
            difference=float(np.max(abs(actual-desired)))
            if difference>3e-5:
                raise PlanningFailure('tracking_effort_rounding','Motor target cannot represent requested local effort precisely')
            output[start:start+7]=target
            arms[side]={'desired_effort_Nm':desired.tolist(),'predicted_effort_Nm':actual.tolist(),
                        'max_effort_error_Nm':difference,'target_bias_rad':(target-reference[start:start+7]).tolist(),
                        'source_calls_before':env.source_controller.calls[side]}
        output=env.validate_action(output,native_fingers=True)
        self.last_receipt={'schema':'source-waypoint-adapter/1','reference_action':reference.tolist(),
            'native_motor_action':output.tolist(),'arms':arms,'time_s':float(env.data.time),
            'scope':'instantaneous controller-law inversion for geometric reference, not calibrated dynamics or long-horizon guarantee',
            'plant_source_law_unchanged':True,'qfrc_applied_added':False,'model_or_state_written':False,
            'policy_and_replay_unchanged':True,'hardware_access':False,'transfer_ready':False}
        self.reference=reference.copy()
        return output


MAX_COMMANDS = 60


def segment_interval(adapter, anchor, requested):
    """Exact scalar feasible interval for the original PD+bias reference law.

    Each clipped PD torque is monotonic in the segment fraction. Intersect all
    14 original motor-command effort envelopes. This can select an interior
    prefix even if a previously dispatched reference is now infeasible.
    """
    env = adapter.env; low, high = 0., 1.; steepest = 1.
    for side, start in [('left', 0), ('right', 8)]:
        ids = env.aids[side]; dids = env.dids[side]
        q = env.data.qpos[env.qids[side]]; velocity = env.data.qvel[dids]
        kp = env.model.actuator_gainprm[ids, 0]; kd = -env.model.actuator_biasprm[ids, 2]
        limits = env.model.actuator_forcerange[ids]
        previous = env.source_controller.previous[side]; accumulated = env.source_controller.accumulated[side]
        lower = adapter.effort(env.action_space.low[start:start+7], q, velocity, kp, kd, previous, accumulated, limits)
        upper = adapter.effort(env.action_space.high[start:start+7], q, velocity, kp, kd, previous, accumulated, limits)
        offset = kp*(anchor[start:start+7]-q)-kd*velocity
        slope = kp*(requested[start:start+7]-anchor[start:start+7])
        lower_pd = lower-env.data.qfrc_bias[dids]; upper_pd = upper-env.data.qfrc_bias[dids]
        for j in range(7):
            if lower_pd[j] > limits[j, 1] or upper_pd[j] < limits[j, 0]:
                raise PlanningFailure('retiming_segment_unreachable', 'No effort feasible anywhere in the original segment')
            minimum = lower_pd[j] if lower_pd[j] > limits[j, 0] else -np.inf
            maximum = upper_pd[j] if upper_pd[j] < limits[j, 1] else np.inf
            a, b = float(offset[j]), float(slope[j]); steepest = max(steepest, abs(b))
            if abs(b) < 1e-14:
                if not minimum <= a <= maximum:
                    raise PlanningFailure('retiming_segment_unreachable', 'Unchanged joint prevents any feasible original-segment progress')
                continue
            edges = sorted(((minimum-a)/b, (maximum-a)/b))
            low = max(low, edges[0]); high = min(high, edges[1])
    if low > high:
        raise PlanningFailure('retiming_segment_unreachable', 'Native joint effort envelopes have no common checked-segment fraction')
    # Numerical inset is stricter than the original envelope. It never widens
    # command bounds, and the original mapper independently checks the result.
    inset = min((high-low)/2., 1e-5/steepest)
    return max(low, high-inset), (low, high)


def feasible_command(adapter, requested, *, anchor=None):
    started = time.monotonic()
    try:
        command = adapter.map(requested)
        complete = True; fraction = 1.; trigger = None
    except PlanningFailure as exc:
        if exc.code != 'tracking_effort_unreachable':
            raise
        trigger = exc.code
        previous = adapter.reference.copy() if anchor is None else np.asarray(anchor, dtype=np.float32).copy()
        difference = np.asarray(requested, dtype=np.float32)-previous
        fraction, interval = segment_interval(adapter, previous, np.asarray(requested, dtype=np.float32))
        candidate = previous+difference*fraction
        command = adapter.map(candidate)
        complete = False
    receipt = copy.deepcopy(adapter.last_receipt)
    receipt['retiming'] = {
        'requested_reference': np.asarray(requested).tolist(),
        'applied_reference': adapter.reference.tolist(),
        'fraction_of_original_segment': fraction,
        'original_reference_dispatched': complete,
        'trigger': trigger, 'search': 'none' if complete else 'analytic intersection of14 original motor-effort intervals',
        'compute_seconds': time.monotonic()-started,
        'motor_target_clipped': False, 'joint_and_force_limits_unchanged': True,
        'same_requested_target_pending_until_full_dispatch': True,
    }
    return command, receipt, complete


def commands(env, requested, diagnostic, is_hold=False):
    if env.source_controller is None or is_hold:
        action, detail = prepare_waypoint(env, requested, diagnostic, is_hold)
        yield action, detail, is_hold
        return
    adapter = env.waypoint_adapter
    target = adapter.reference.copy() if diagnostic.get('phase') == '10s-native-hold' else np.asarray(requested).copy()
    anchor = adapter.reference.copy()
    for index in range(MAX_COMMANDS):
        action, receipt, complete = feasible_command(adapter, target, anchor=anchor)
        detail = copy.deepcopy(diagnostic)
        detail['waypoint_adapter'] = receipt
        detail['waypoint_adapter']['retiming']['command_index_for_original_waypoint'] = index
        detail['waypoint_adapter']['retiming']['maximum_commands_per_original_waypoint'] = MAX_COMMANDS
        yield action, detail, False
        if complete:
            return
        if env._done:
            raise PlanningFailure('retiming_task_ended', 'Original native task ended before the pending waypoint')
    raise PlanningFailure('retiming_timeout', 'Original reference not dispatchable in bounded native steps; no next waypoint')


def prepare_waypoint(env, action, diagnostic, is_hold):
    if env.source_controller is None:
        return action, diagnostic
    if is_hold:
        # The existing planner clock contract holds the exact previous motor
        # target. Never adapt it a second time or replace it with a new target.
        return action, diagnostic
    adapter = env.waypoint_adapter
    reference = adapter.reference.copy() if diagnostic.get('phase') == '10s-native-hold' else action
    started = time.monotonic()
    native = adapter.map(reference)
    detail = copy.deepcopy(diagnostic)
    detail['waypoint_adapter'] = copy.deepcopy(adapter.last_receipt)
    detail['waypoint_adapter']['compute_seconds'] = time.monotonic() - started
    detail['waypoint_adapter']['timing_scope'] = 'conversion inside existing33ms command period; no hardware real-time guarantee'
    return native, detail


def install_isolated():
    original = TampPlanner.__init__

    def initialize(self, env, *args, **kwargs):
        original(self, env, *args, **kwargs)
        if env.source_controller is None:
            return
        cold_reset = env.tick == 0 and float(env.data.time) == 0. and all(n == 0 for n in env.source_controller.calls.values())
        if not hasattr(env, 'waypoint_adapter') or cold_reset:
            env.waypoint_adapter = WaypointAdapter(env)
        self.base_action = env.waypoint_adapter.reference.copy()
        self.result['command_adapter'] = {
            'schema': 'source-waypoint-adapter/1',
            'planner_input': 'nominal joint reference',
            'executed_output': 'native SourceMIT motor target',
            'planning_wait': 'exact prior native target retained',
            'source_plant_state_law_preserved': True,
            'no_actuator_force_or_joint_range_widening': True,
            'policy_and_recorded_replay_unchanged': True,
            'hardware_access': False, 'transfer_ready': False,
        }
    TampPlanner.__init__ = initialize

