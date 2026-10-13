"""Re-execute an archived native route and inspect loaded tracking privately.

Replays declared actions and planning holds at their original clock. This is a
diagnostic of the same simulation, never a new task score or physical calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from harness.object_handover import ProtectedObjectEnv
from sim.environment import ROOT
from sim.recording import dump
from tools.check_bottle_routes import initialize


def relation(env, side):
    hand = env.data.xmat[env.ee_ids[side]].reshape(3, 3)
    return (hand.T @ (env.data.xpos[env.bottle_id] - env.data.xpos[env.ee_ids[side]]),
            hand.T @ env.data.xmat[env.bottle_id].reshape(3, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=lambda x: (ROOT / x).resolve())
    parser.add_argument('--output', required=True, type=lambda x: (ROOT / x).resolve())
    args = parser.parse_args()
    if not args.output.is_relative_to((ROOT / 'runs').resolve()):
        raise ValueError('diagnostics belong in a new runs/ folder')
    result = json.loads((args.case / 'result.json').read_text())
    episode = next((args.case / 'episode').iterdir())
    archived = json.loads((episode / 'manifest.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('tools/inspect_motion.py', 'tools/check_bottle_routes.py'):
        p = args.output / 'source' / name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    env = ProtectedObjectEnv(images=False)
    segments = []; maximum_error = 0.; maximum_time_error = 0.; active = None
    side = result['case']['receiver'] or result['case']['arm']
    counters = {}; previous_grasp = None; baseline_grasp = None
    try:
        initialize(env, result['case'])
        if env.manifest()['model_evidence'] != archived['model_evidence']:
            raise ValueError('compiled model, inputs or runtime parameters differ from archived route')
        for line in (episode / 'transitions.jsonl').read_text().splitlines():
            row = json.loads(line); diagnostic = row.get('diagnostics') or {}
            phase = diagnostic.get('phase', 'planning')
            timing = diagnostic.get('planning_timing') or {}
            # A planning hold preserves the exact previous motor target.
            command = np.asarray(row['action'])
            if timing.get('action_source') == 'planning_hold':
                env.hold_tamp(command, timing['duration_s'])
            else:
                env.step_tamp(command)
            expected = row['next_observation']
            error = float(np.max(np.abs(env.state() - expected['state'])))
            time_error = abs(float(env.data.time) - expected['sim_time'])
            maximum_error = max(maximum_error, error); maximum_time_error = max(maximum_time_error, time_error)
            if error > 1e-6 or time_error > 1e-9 or env.tick != expected['tick']:
                raise ValueError(f'archive replay differs at transition {row["index"]}: state={error}, time={time_error}')
            kind = ('wait' if timing.get('action_source') == 'planning_hold' else
                    'settling' if diagnostic.get('settling') else 'command')
            key = (phase, kind, diagnostic.get('feedback_correction'))
            if kind == 'wait': continue  # keep original waits in physics, avoid splitting every motion row
            if key != active:
                active = key
                counters[phase] = counters.get(phase, 0) + 1
                segments.append({'phase': phase, 'kind': kind, 'correction': key[2], 'start_tick': env.tick,
                    'start_seconds': float(env.data.time), 'samples': []})
            hand = env.data.xmat[env.ee_ids[side]].reshape(3, 3)
            tcp = env.data.xpos[env.ee_ids[side]] + hand @ np.array([0., .006, .075])
            offset, relative = relation(env, side)
            if env.evaluation['held'][side] and baseline_grasp is None:
                baseline_grasp = (offset.copy(), relative.copy())
            angle = lambda a, b: float(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1)))
            velocity = env.data.qvel[env.registry['bottle']['dofadr']:env.registry['bottle']['dofadr'] + 6]
            sample = {'tick': env.tick, 'sim_time': float(env.data.time), 'q_rad': env.data.qpos[env.qids[side]].copy(),
                'q_target_rad': command[:7] if side == 'left' else command[8:15],
                'q_error_rad': command[:7] - env.data.qpos[env.qids[side]] if side == 'left'
                    else command[8:15] - env.data.qpos[env.qids[side]],
                'tcp_m': tcp, 'object_m': env.data.xpos[env.bottle_id].copy(),
                'bottom_z_m': float(env._bottle_bounds()[0][2]),
                'object_upright_cosine': float(env.data.xmat[env.bottle_id].reshape(3, 3)[2, 2]),
                'linear_speed_m_s': float(np.linalg.norm(velocity[:3])), 'held': bool(env.evaluation['held'][side]),
                'hand_object_offset_m': offset,
                'relative_change_since_first_hold_m': float(np.linalg.norm(offset - baseline_grasp[0])) if baseline_grasp else None,
                'relative_rotation_since_first_hold_rad': angle(relative, baseline_grasp[1]) if baseline_grasp else None,
                'relative_increment_m': float(np.linalg.norm(offset - previous_grasp[0])) if previous_grasp else None,
                'relative_rotation_increment_rad': angle(relative, previous_grasp[1]) if previous_grasp else None}
            previous_grasp = (offset.copy(), relative.copy())
            segments[-1]['samples'].append(sample)
            # One exact integration-state checkpoint before/after each settling
            # period permits short scratch dynamics probes without another run.
            if phase == 'place' and kind == 'settling':
                ident = f'place-{counters[phase]}'
                state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
                packed = np.empty(mujoco.mj_stateSize(env.model, state_spec))
                mujoco.mj_getState(env.model, env.data, packed, state_spec)
                np.savez(args.output / (ident + '.npz'), integration_state=packed, state_spec=state_spec,
                    qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(),
                    last_action=env.last_action.copy(), tick=env.tick, sim_time=float(env.data.time))
        # Keep all geometric feedback rows for analysis, separate from model memory.
        dump(args.output / 'segments.json', segments)
        dump(args.output / 'summary.json', {'same_archived_commands_and_planning_holds': True,
            'source_case': str(args.case.relative_to(ROOT)), 'source_run': archived['run_id'],
            'manifest_sha256': hashlib.sha256((episode / 'manifest.json').read_bytes()).hexdigest(),
            'trajectory_sha256': hashlib.sha256((episode / 'transitions.jsonl').read_bytes()).hexdigest(),
            'maximum_robot_state_error': maximum_error, 'maximum_clock_error_s': maximum_time_error,
            'model_evidence': env.manifest()['model_evidence'], 'segments': len(segments),
            'hardware_access': False, 'transfer_ready': False, 'scope': 'simulation replay diagnostic only'})
        print({'same_archived_simulation': True, 'max_state_error': maximum_error, 'segments': len(segments)}, flush=True)
    finally:
        env.close()


if __name__ == '__main__': main()
