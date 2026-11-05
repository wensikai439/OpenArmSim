"""Frozen ACT rollout with private diagnostics; no LLM, hardware or UI mutation."""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import mujoco
from PIL import Image

from harness.policy_trace import TracingClient, TracingTiming
from sim.environment import OpenArmEnv, ROOT
from sim.policy import load_config
from sim.recording import Recorder, dump


def sample(env, requested=None, applied=None):
    """Privileged geometry belongs to the evaluator, never the ACT observation."""
    tcp = {}
    for side, body in env.ee_ids.items():
        rotation = env.data.xmat[body].reshape(3, 3)
        tcp[side] = (env.data.xpos[body] + rotation @ np.array([0., .006, .075])).tolist()
    contacts = env.contacts.read(env.data)
    window = env.contacts.summary()
    return {
        'tick': env.tick, 'sim_seconds': float(env.data.time), 'state': env.state().tolist(),
        'requested': None if requested is None else np.asarray(requested).tolist(),
        'applied': None if applied is None else np.asarray(applied).tolist(),
        'tcp_m': tcp, 'bottle_m': env.data.xpos[env.bottle_id].tolist(),
        'basket_m': env.data.xpos[env.box_id].tolist(),
        'bottle_upright_cos': float(env.data.xmat[env.bottle_id].reshape(3, 3)[2, 2]),
        'held': contacts['held'], 'touching': contacts['touching'],
        'hold_seconds': dict(window['hold_seconds']),
        'penetration_m': float(window['max_penetration_m']),
        'peak_contact_force_N': float(window['peak_normal_force_N']),
        'invalid': bool(env.evaluation['invalid']), 'task_success': bool(env.evaluation['success']),
    }


def summarize(rows):
    if not rows:
        raise ValueError('at least the initial observation is required')
    phases = {}; after = -1
    for name, column, close in [('right_close', 15, True), ('left_close', 7, True),
                                ('right_release', 15, False), ('left_release', 7, False)]:
        for i in range(after + 1, len(rows)):
            action = rows[i]['requested']
            if action is not None and (action[column] > -.2 if close else action[column] < -.7):
                phases[name] = {'index': i, 'sim_seconds': rows[i]['sim_seconds']}; after = i; break
        else:
            break
    distance = lambda row, side: float(np.linalg.norm(np.subtract(row['tcp_m'][side], row['bottle_m'])))
    right = [distance(r, 'right') for r in rows]
    physical = {side: {'two_finger_hold_seen': any(r['held'][side] for r in rows),
        'max_continuous_hold_seconds': max(r['hold_seconds'][side] for r in rows)} for side in ('left', 'right')}
    initial_z = rows[0]['bottle_m'][2]
    physical['max_bottle_lift_m'] = max(r['bottle_m'][2] - initial_z for r in rows)
    physical['task_success'] = any(r['task_success'] for r in rows)
    return {'schema': 'act-private-feedback/1',
        'scope': 'command trend is descriptive; only native contact/lift/placement establishes physical success',
        'oracle_to_policy': False, 'ordered_command_phases': phases,
        'right_tcp_initial_bottle_distance_m': right[0], 'right_tcp_min_bottle_distance_m': min(right),
        'min_interhand_tcp_distance_m': min(float(np.linalg.norm(np.subtract(r['tcp_m']['left'], r['tcp_m']['right']))) for r in rows),
        'physical': physical, 'max_penetration_m': max(r['penetration_m'] for r in rows)}


def frames(env, folder, label):
    folder.mkdir(exist_ok=True)
    for camera in ('operator', 'head', 'left', 'right'):
        Image.fromarray(env.render(camera)).save(folder / f'{label}-{camera}.jpg', quality=88)


def initialize(env, mode):
    if mode == 'reference':
        return {'source': 'original reference episode reset', 'state': env.state().tolist()}
    root = ROOT/'runs/20261023-act/reference'
    target = json.loads((root/'act_startup_target.json').read_text())['target']
    state = list(target)
    if mode == 'feedback':
        text = (root/'real-run.log').read_text()
        for side, start in [('left', 0), ('right', 8)]:
            match = re.search(r'\[JOINT_DIAG\] '+side+r' current=\[([^]]+)\]', text)
            state[start:start+7] = np.fromstring(match[1], sep=' ').tolist()
        match = re.search(r'\[JOINT_DIAG\] gripper current=\[([^]]+)\]', text)
        state[7], state[15] = np.fromstring(match[1].replace(',', ' '), sep=' ')
    if len(state) != 16 or not np.isfinite(state).all():
        raise ValueError('invalid exported ACT startup state')
    # A declared reset condition, applied ONCE before time advances. No object,
    # pose or policy output is ever overwritten during the rollout.
    for side, start in [('left', 0), ('right', 8)]:
        joints = np.asarray(state[start:start+7])
        bounds = np.array([env.model.joint(f'openarm_{side}_joint{i}').range for i in range(1,8)])
        if np.any(joints < bounds[:,0]) or np.any(joints > bounds[:,1]):
            raise ValueError('real startup exceeds simulated joint limits')
        env.data.qpos[env.qids[side]] = joints
        value = env.grippers[side].to_joint(state[start+7])
        for joint in env.fingers[side]:
            if not min(joint.range) <= value <= max(joint.range):
                raise ValueError('real startup exceeds simulated finger range')
            env.data.qpos[joint.qposadr[0]] = value
    env.last_action = np.asarray(target, np.float32)
    env._set_command(env.last_action, update_controller=False)
    mujoco.mj_forward(env.model, env.data)
    env.evaluation = env._evaluate(0.)
    return {'source': 'exported real ACT '+mode, 'state': env.state().tolist(),
        'command': target, 'log_feedback_precision_rad': .00005 if mode == 'feedback' else None}


def run(args):
    folder = ROOT / 'runs' / '20261023-act' / args.name
    folder.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / 'harness/policy.json')
    config['max_steps'] = args.steps
    env = OpenArmEnv(images=False)
    recorder = None; rows = []; status = 'error'; error = None
    try:
        env.reset(seed=args.seed, options={'episode': args.episode})
        initialization = initialize(env, args.startup)
        recorder = Recorder(env, 'act', record_images=False, seed=args.seed, root=folder,
            details={'policy': config, 'timing_mode': 'paused', 'episode': args.episode,
                'scope': 'private geometry is never fed to the policy',
                'initialization': initialization, 'native_gripper_adapter': args.native_gripper,
                'physical_parameters_changed': False, 'ui_changed': False, 'hardware_access': False})
        dump(folder / 'protocol.json', {'case': args.name, 'episode': args.episode,
            'seed': args.seed, 'policy': config, 'source': env.manifest(),
            'initialization': initialization, 'native_gripper_adapter': args.native_gripper,
            'real_behavior_reference': 'user qualitative ACT trend; recorded demonstrations are separate evidence'})
        rows.append(sample(env)); frames(env, folder / 'frames', '0000')
        before = recorder.observation(env)
        timing = TracingTiming(config['timing_mode'])
        class ObservationAdapter:
            def __getattr__(self, name): return getattr(env, name)
            def state(self): return env.state()
            def render(self, name): return env.render(name)
            def prepare_policy_action(self, action):
                return env.prepare_policy_action(action, native_fingers=args.native_gripper)
        observation = ObservationAdapter()
        with TracingClient(config, seed=args.seed) as client:
            for requested, diagnostic, is_hold in timing.actions(observation, client, config, recorder.event):
                applied, controller = env.prepare_policy_action(requested, native_fingers=args.native_gripper)
                before_seconds = float(env.data.time)
                _, _, terminal, truncated, info = env.step_policy(applied, native_fingers=args.native_gripper)
                timing.committed(is_hold, before_seconds, float(env.data.time))
                after = recorder.observation(env)
                recorder.append(before, applied, after, info, terminated=terminal, truncated=truncated,
                    diagnostics={**diagnostic, 'controller': controller, 'requested_action': requested})
                before = after
                row = sample(env, requested, applied); rows.append(row)
                if env.tick % 60 == 0:
                    frames(env, folder / 'frames', f'{env.tick:04d}')
                    print(json.dumps({'tick': env.tick, 'seconds': float(env.data.time),
                        'held': row['held'], 'bottle_m': row['bottle_m']}, ensure_ascii=False), flush=True)
                if terminal or truncated:
                    status = 'success' if info['evaluation']['success'] else 'invalid' if info['evaluation']['invalid'] else 'truncated'
                    break
            else:
                status = 'budget_exhausted'
    except Exception as exc:
        error = str(exc)
    finally:
        if recorder:
            frames(env, folder / 'frames', 'final')
            feedback = summarize(rows)
            dump(folder / 'feedback.json', {'summary': feedback, 'samples': rows})
            result = {'status': status, 'error': error, 'steps': recorder.count,
                'sim_seconds': float(env.data.time), 'native_run': recorder.path.name,
                'feedback': feedback, 'timing': timing.summary() if 'timing' in locals() else None,
                'transfer_ready': False}
            recorder.finish(status, env.info(), error=error, metrics=result)
            dump(folder / 'result.json', result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--episode', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--startup', choices=('reference', 'target', 'feedback'), default='reference')
    parser.add_argument('--native-gripper', action='store_true')
    args = parser.parse_args()
    if not args.name.replace('-', '').isalnum() or not 1 <= args.steps <= 900:
        parser.error('use a simple case name and at most the original 900 actions')
    run(args)
