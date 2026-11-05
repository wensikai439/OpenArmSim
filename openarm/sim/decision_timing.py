"""Owner-thread physics during an LLM round trip; never applies its new action."""
import math
import time
import uuid
from pathlib import Path

import numpy as np

from .environment import ROOT
from .recording import Recorder

CONTRACT = {
    'mode': 'measured_delay', 'method': 'single physics owner; hold while API runs, round final partial control tick up',
    'waiting_action': 'exact previous applied native actuator target; no new target or feedback re-limiting',
    'initial_planning': 'paused until the first accepted plan declares this mode; earlier API waits are excluded',
    'scope': 'LLM request/response and local scheduling; low-level planning computation is not modeled',
    'real_time_validated': False, 'transfer_ready': False,
}


class ObservationChanged(ValueError):
    """A rejected proposal with measured drift; no motion has been started."""
    def __init__(self, feedback):
        self.feedback = feedback
        label='模型使用了早于本次请求的旧观测' if feedback['reason']=='stale_observation' else '模型等待期间场景变化：'+str(feedback['reason'])
        super().__init__(label+'；本次未执行动作，使用更新后的观测重新决策')


class GraspStability:
    """Read-only native relative-pose audit, independent of contact/lift score."""
    window_seconds, position_limit_m, rotation_limit_rad = 2., .001, .02

    def __init__(self, env):
        self.env = env
        self.samples = {s: [] for s in ('left', 'right')}
        self.max_drift = {s: 0. for s in self.samples}
        self.max_rotation = dict(self.max_drift)
        self.contact_lost = {s: False for s in self.samples}
        self.initial_held = dict(env.evaluation['held'])
        self.max_sample_span = 0.
        self.read()

    def read(self):
        env = self.env
        for side, rows in self.samples.items():
            if not self.initial_held[side]:
                continue
            self.contact_lost[side] |= not env.evaluation['held'][side]
            body = env.ee_ids[side]
            hand = env.data.xmat[body].reshape(3, 3)
            relative = hand.T @ (env.data.xpos[env.bottle_id]-env.data.xpos[body])
            rotation = hand.T @ env.data.xmat[env.bottle_id].reshape(3, 3)
            rows.append((float(env.data.time), relative.copy(), rotation.copy()))
            # Keep one boundary sample. Floating arithmetic must not evict a
            # valid nominal two-second boundary and leave an incomplete window.
            while len(rows)>1 and rows[-1][0]-rows[1][0]>=self.window_seconds-1e-6:
                rows.pop(0)
            if rows[-1][0]-rows[0][0]>=self.window_seconds-1e-6:
                self.max_sample_span = max(self.max_sample_span, rows[-1][0]-rows[0][0])
                # An object can slide/rotate out and back within a window.
                # Endpoint subtraction misses it; use the observed pose range.
                positions=np.asarray([row[1] for row in rows])
                rotations=np.asarray([row[2] for row in rows]).reshape(len(rows),9)
                drift=float(np.max(np.linalg.norm(positions[:,None]-positions[None,:],axis=2)))
                minimum_trace=float(np.min(rotations@rotations.T))
                angle=float(np.arccos(np.clip((minimum_trace-1)/2,-1,1)))
                self.max_drift[side] = max(self.max_drift[side], drift)
                self.max_rotation[side] = max(self.max_rotation[side], angle)

    def summary(self):
        result = {}
        for side, rows in self.samples.items():
            duration = rows[-1][0]-rows[0][0] if rows else 0.
            state = ('not_held_at_start' if not self.initial_held[side] else
                     'contact_lost' if self.contact_lost[side] else
                     'insufficient_duration' if duration<self.window_seconds-1e-6 else
                     'sliding' if self.max_drift[side]>self.position_limit_m or self.max_rotation[side]>self.rotation_limit_rad else
                     'stable_within_audit_window')
            result[side] = {'status': state, 'max_window_relative_drift_m': self.max_drift[side],
                            'max_window_relative_rotation_rad': self.max_rotation[side]}
        return {'schema': 'grasp-stability/2', 'window_seconds': self.window_seconds,
                'metric':'maximum sampled relative-position and rotation range within each rolling window; includes one boundary sample',
                'maximum_sample_span_seconds':self.max_sample_span,
                'limits': {'relative_position_m': self.position_limit_m, 'relative_rotation_rad': self.rotation_limit_rad},
                'arms': result, 'source': 'simulator_ground_truth',
                'scope': 'diagnostic only; sustained contacts/lift do not establish stationary object-in-hand',
                'hardware_access': False, 'transfer_ready': False}


class DecisionWait:
    def __init__(self, env, root, request_id, observation_id, *, clock=time.monotonic, output=None):
        uuid.UUID(root)
        if type(request_id) is not int or not 1 <= request_id <= 32:
            raise ValueError('invalid decision request identity')
        if env._done or env.evaluation['invalid'] or env._clock_mode not in (None, 'periodic'):
            raise ValueError('decision wait requires a valid ongoing periodic task')
        self.env, self.clock = env, clock
        self.started, self.start_tick, self.start_sim = clock(), env.tick, float(env.data.time)
        self.target = env.last_action.copy()
        self.grasp = GraspStability(env)
        self.status, self.steps = 'running', 0
        self.error = None
        self.details = {**CONTRACT, 'root': root, 'request_id': request_id, 'source_observation_id': observation_id,
                        'task_window': env.task_window(), 'native_finger_contract': True}
        folder = Path(output) if output else ROOT/'runs'/f'{time.strftime("%Y%m%d")}-agent-{root[:8]}'/'waits'
        self.recorder = Recorder(env, 'wait', root=folder, record_images=False,
                                 seed=int(env.np_random_seed), details=self.details)
        self.before = self.recorder.observation(env)
        self.closed = False

    def advance(self, *, final=False, cancelled=False):
        if cancelled:
            self.status = 'cancelled'
        elapsed = max(0., self.clock()-self.started)
        ticks = math.ceil(elapsed*self.env.profile['control_hz']-1e-10) if final else math.floor(elapsed*self.env.profile['control_hz'])
        while self.status == 'running' and self.steps < ticks:
            try:
                _, _, terminal, truncated, info = self.env.step_tamp(self.target)
            except Exception as exc:
                self.status='physics_error';self.error=str(exc)
                self.recorder.event({'type':'decision_physics_error','error':self.error})
                break
            after = self.recorder.observation(self.env)
            self.grasp.read()
            self.recorder.append(self.before, self.target, after, info, terminated=terminal, truncated=truncated,
                                 diagnostics={'decision_wait': {**self.details, 'hold_index': self.steps,
                                     'source_age_sim_seconds': after['sim_time']-self.start_sim}})
            self.before, self.steps = after, self.steps+1
            if terminal or truncated:
                self.status = 'physics_invalid' if info['evaluation']['invalid'] else 'budget_exhausted'
        return self.summary()

    def summary(self):
        return {**self.details, 'status': self.status, 'run_id': self.recorder.path.name,
                'run_path': str(self.recorder.path.relative_to(ROOT)) if self.recorder.path.is_relative_to(ROOT) else str(self.recorder.path),
                'hold_transitions': self.steps, 'start_tick': self.start_tick, 'end_tick': self.env.tick,
                'start_sim_seconds': self.start_sim, 'end_sim_seconds': float(self.env.data.time),
                'integrated_seconds': float(self.env.data.time)-self.start_sim,
                'wall_seconds': max(0., self.clock()-self.started),
                'grasp_stability': self.grasp.summary(),
                'error':self.error,'task_failed': self.status in {'physics_invalid', 'physics_error', 'budget_exhausted', 'cancelled'}}

    def finish(self, *, cancelled=False):
        if not self.closed:
            self.advance(final=True, cancelled=cancelled)
            if self.status == 'running':
                self.status = 'completed'
            self.recorder.finish(self.status, self.env.info(), metrics={'decision_timing': self.summary()})
            self.closed = True
        return self.summary()


def revalidation(before, after):
    """Conservative software drift gate; not a physical localization claim."""
    reason = None
    if before['observation_id'].split(':')[0] != after['observation_id'].split(':')[0]:
        reason = 'epoch_changed'
    if after.get('observation', {}).get('episode_finished'):
        reason = 'task_finished_during_decision'
    if (after.get('decision_timing') or {}).get('task_failed'):
        reason = 'decision_wait_failed'
    distances, angles = [], []
    for side in ('left', 'right'):
        old = before['observation']['tcp'][side]
        new = after['observation']['tcp'][side]
        distances.append(float(np.linalg.norm(np.array(old['position_m'])-new['position_m'])))
        a, b = np.array(old['quaternion_xyzw']), np.array(new['quaternion_xyzw'])
        cosine = min(1., abs(float(a@b))/max(1e-12, float(np.linalg.norm(a)*np.linalg.norm(b))))
        angles.append(2*math.acos(cosine))
    old = before['diagnostics']['manipulation_scene']
    new = after['diagnostics']['manipulation_scene']
    movement = float(np.linalg.norm(np.array(old['position_m'])-new['position_m']))
    if old.get('held_by') != new.get('held_by'):
        reason = 'holding_state_changed'
    if abs(float(old.get('upright',0))-float(new.get('upright',0))) > .01:
        reason = 'object_orientation_changed'
    if max(distances) > .005 or max(angles) > .05 or movement > .003:
        reason = reason or 'scene_drift'
    return {'accepted': reason is None, 'reason': reason,
            'source_observation_id': before['observation_id'], 'current_observation_id': after['observation_id'],
            'tcp_drift_m': max(distances), 'tcp_rotation_rad': max(angles), 'object_drift_m': movement,
            'limits': {'tcp_m': .005, 'tcp_rotation_rad': .05, 'object_m': .003},
            'source': 'simulator_oracle_drift_guard; not calibrated perception', 'transfer_ready': False}


def rebind_observation(arguments, before, after, *, current_query=False, failure_report=False):
    if 'observation_id' not in arguments:
        return dict(arguments), None
    if arguments['observation_id'] != before['observation_id']:
        if failure_report and arguments.get('verdict') in {'uncertain','failure'}:
            # A non-success report may describe older evidence from this task.
            # It never authorizes motion or claims a fresh visual assessment.
            requested=str(arguments['observation_id']).rsplit(':',1)
            current=str(after['observation_id']).rsplit(':',1)
            source=str(before['observation_id']).rsplit(':',1)
            if (len(requested)==len(current)==len(source)==2 and requested[0]==current[0]==source[0]
                    and requested[1].isdigit() and current[1].isdigit()
                    and int(requested[1])<=int(current[1])):
                return {**arguments,'observation_id':after['observation_id']},{**revalidation(before,after),
                    'requested_observation_id':arguments['observation_id'],'report_only':True,
                    'historical_evidence':True,'motion_authorized':False,
                    'dispatch_scope':'non-success assessment of this task; older camera evidence is labelled'}
        raise ObservationChanged({**revalidation(before,after), 'accepted':False, 'reason':'stale_observation',
                                  'requested_observation_id':arguments['observation_id']})
    drift=revalidation(before,after)
    if not drift['accepted'] and not current_query and not failure_report:
        raise ObservationChanged(drift)
    return {**arguments,'observation_id':after['observation_id']},drift
