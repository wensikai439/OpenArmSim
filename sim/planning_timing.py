"""Single-owner discrete-event integration of native planner wall duration.

The planner computes against a stationary state. Before applying its proposal,
the owner integrates the elapsed computation holding the previous actuator target
and rejects stale proposals. This is a latency sensitivity model, not a real-time
controller or a calibrated hardware clock. No concurrent live MuJoCo readers.
"""
import math
import time

import numpy as np

from .tamp import PlanningFailure


CONTRACT = {
    'schema': 'planning-timing/1',
    'method': 'single-owner discrete-event integration before proposal dispatch',
    'scope': 'planner construction and each generator resume, including IK, collision search, GPU worker round trips and feedback publishing',
    'waiting_action': 'exact previous applied native target; no new target, reset, weld or state overwrite',
    'quantization': 'cumulative wall duration on the existing 1ms physics grid; final remainder rounded up; bounded by one physics step; command cadence remains 30Hz',
    'omissions': ['initial LLM planning before first plan declaration', 'camera exposure and CAN timing',
                  'real controller/watchdog behavior', 'continuous streaming during blocking computation'],
    'real_time_validated': False, 'transfer_ready': False,
}


def state(env):
    tcp = {}
    for side, body in env.ee_ids.items():
        rotation = env.data.xmat[body].reshape(3, 3)
        tcp[side] = (env.data.xpos[body] + rotation @ np.array([0., .006, .075]), rotation.copy())
    return {'tick': env.tick, 'sim_time': float(env.data.time), 'tcp': tcp,
            'joints': env.data.qpos.copy(), 'held': dict(env.evaluation['held']),
            'object': env.data.xpos[env.bottle_id].copy(),
            'object_rotation': env.data.xmat[env.bottle_id].reshape(3, 3).copy()}


def rotation_distance(a, b):
    return float(np.arccos(np.clip((np.trace(a @ b.T)-1)/2, -1, 1)))


def revalidate(env, before):
    after = state(env)
    tcp = max(float(np.linalg.norm(after['tcp'][s][0]-before['tcp'][s][0])) for s in after['tcp'])
    angle = max(rotation_distance(after['tcp'][s][1], before['tcp'][s][1]) for s in after['tcp'])
    object_drift = float(np.linalg.norm(after['object']-before['object']))
    object_rotation = rotation_distance(after['object_rotation'], before['object_rotation'])
    joints = max(float(np.max(abs(after['joints'][env.qids[s]]-before['joints'][env.qids[s]]))) for s in env.qids)
    reason = ('episode_finished' if env._done or env.evaluation['invalid'] else
              'holding_state_changed' if after['held'] != before['held'] else
              'scene_drift' if tcp > .005 or angle > .05 or object_drift > .003
              or object_rotation > .05 or joints > .02 else None)
    return {'accepted': reason is None, 'reason': reason, 'source_tick': before['tick'], 'current_tick': env.tick,
            'tcp_drift_m': tcp, 'tcp_rotation_rad': angle, 'joint_drift_rad': joints,
            'object_drift_m': object_drift, 'object_rotation_rad': object_rotation,
            'limits': {'tcp_m': .005, 'tcp_rotation_rad': .05, 'joint_rad': .02,
                       'object_m': .003, 'object_rotation_rad': .05},
            'source': 'simulator_oracle; rejects stale paths, not physical localization accuracy',
            'hardware_access': False, 'transfer_ready': False}


class PlanningTiming:
    def __init__(self, mode='paused', *, clock=time.monotonic):
        if mode not in ('paused', 'measured_delay'):
            raise ValueError('unknown planning timing mode')
        self.mode, self.clock = mode, clock
        self.resumes = self.commands = self.wait_ticks = 0
        self.wait_units=0
        self.wall_seconds = self.integrated_seconds = 0.
        self.last_revalidation = None

    def contract(self):
        return {**CONTRACT, 'mode': self.mode,
                'delay_source': 'measured local planner duration; no injected diagnostic delay'}

    def summary(self):
        return {**self.contract(), 'generator_resumes': self.resumes, 'native_commands': self.commands,
                'planner_wall_seconds': self.wall_seconds, 'hold_transitions': self.wait_ticks,
                'hold_physics_steps':self.wait_units,
                'integrated_hold_seconds': self.integrated_seconds,
                'unintegrated_seconds': max(0., self.wall_seconds-self.integrated_seconds),
                'rounding_overhead_seconds': max(0., self.integrated_seconds-self.wall_seconds),
                'last_revalidation': self.last_revalidation}

    def committed(self, is_hold, before_seconds, after_seconds):
        if is_hold:
            self.wait_ticks += 1
            self.integrated_seconds += after_seconds-before_seconds
            self.wait_units += round((after_seconds-before_seconds)/self.physics_dt)
        else:
            self.commands += 1

    def actions(self, env, source, event, cancelled=lambda: False):
        hz = env.profile['control_hz']
        self.physics_dt=float(env.model.opt.timestep)
        chunk_units=max(1,round(1/hz/self.physics_dt))
        while True:
            if cancelled():
                raise PlanningFailure('cancelled', '规划已停止，未执行待返回的动作')
            before, held_target = state(env), env.last_action.copy()
            started = self.clock()
            exhausted, failure, proposal = False, None, None
            try:
                proposal = next(source)
            except StopIteration:
                exhausted = True
            except Exception as exc:
                failure = exc
            elapsed = self.clock()-started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('planner clock moved backward or became nonfinite')
            self.resumes += 1
            self.wall_seconds += elapsed
            # A cheap resume must not create a 33ms pause mid-trajectory.
            target_units = (math.ceil(self.wall_seconds/self.physics_dt-1e-10) if exhausted or failure else
                            math.floor(self.wall_seconds/self.physics_dt+1e-10)) if self.mode == 'measured_delay' else 0
            units=max(0,target_units-self.wait_units)
            waits=math.ceil(units/chunk_units)
            details = {'resume': self.resumes, 'source_tick': before['tick'],
                       'source_sim_time': before['sim_time'], 'wall_seconds': elapsed,
                       'accumulated_wall_seconds': self.wall_seconds, 'hold_ticks': waits,
                       'hold_physics_steps':units,
                       'response_valid': failure is None, 'exhausted': exhausted, 'mode': self.mode}
            # Only material waits and terminal/error responses create events.
            if waits or exhausted or failure:
                event({'type': 'native_planning', **details})
            for index in range(waits):
                if cancelled():
                    raise PlanningFailure('cancelled', '规划等待已停止，禁止下发返回动作')
                duration=min(chunk_units,target_units-self.wait_units)*self.physics_dt
                yield held_target.copy(), {'planning_timing': {**details, 'action_source': 'planning_hold',
                                                               'wait_index': index,'duration_s':duration}}, True
                if env._done or env.evaluation['invalid']:
                    return
            if cancelled():
                raise PlanningFailure('cancelled', '规划已停止，禁止下发返回动作')
            if failure is not None:
                raise failure
            self.last_revalidation = revalidate(env, before)
            if not self.last_revalidation['accepted']:
                event({'type': 'planning_state_changed', **self.last_revalidation})
                exc = PlanningFailure('planning_state_changed', '规划等待期间场景变化，旧路径未执行；重新观察后再规划')
                exc.revalidation = self.last_revalidation
                raise exc
            if exhausted:
                return
            action, diagnostic = proposal
            yield action, {**diagnostic, 'planning_timing': {**details, 'action_source': 'native_command',
                'source_age_sim_seconds': float(env.data.time)-before['sim_time'],
                'revalidation': self.last_revalidation}}, False
