"""Fixed gripper-coordinate comparison against the preserved local implementation.

No fitting, external model calls, SSH, hardware, or runtime-parameter writes.
The archived implementation is local project code, never robot source code.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np

from .environment import ROOT, OpenArmEnv, digest
from .recording import dump
from .replay import ReferenceReplay

BASELINE = ROOT / '.history/20261001-032134-gripper-contract/sim'
VARIANTS = ('legacy', 'shared_coordinates', 'shared_motor_gain')


def rollout(env, episode):
    env.reset(seed=0, options={'episode': episode})
    replay = ReferenceReplay(env, episode)
    states, times = [env.state()], [float(env.data.time)]
    error, outcome = None, 'reference_exhausted'
    while len(states) <= replay.intervals:
        try:
            _, (_, _, terminated, truncated, _), _ = replay.advance()
        except (ValueError, RuntimeError) as exc:
            error, outcome = str(exc), 'error'
            break
        states.append(env.state())
        times.append(float(env.data.time))
        if terminated or truncated:
            outcome = ('success' if env.evaluation['success'] else
                       'invalid' if env.evaluation['invalid'] else 'truncated')
            break
    return {'states': np.asarray(states), 'times': np.asarray(times),
            'status': outcome, 'error': error, 'evaluation': env.evaluation,
            'available_samples': len(states), 'reference_samples': replay.intervals + 1}


def time_error(prediction, target, times):
    squared = (prediction - target)**2
    if len(times) < 2:
        raise ValueError('a scored comparison needs at least two observations')
    duration = float(times[-1] - times[0])
    integral = np.sum(.5 * (squared[:-1] + squared[1:]) * np.diff(times)[:, None], axis=0)
    return np.sqrt(integral / duration), integral, duration


def build():
    legacy_source = (BASELINE / 'environment.py').read_bytes()
    legacy_profile = json.loads((BASELINE / 'profile.json').read_text())
    output = ROOT / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + '-gripper-' + uuid.uuid4().hex[:6])
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    (inputs / 'legacy_environment.py').write_bytes(legacy_source)
    dump(inputs / 'legacy_profile.json', legacy_profile)
    for source, name in ((Path(__file__), 'study.py'), (ROOT/'sim/environment.py', 'environment.py'),
                         (ROOT/'sim/gripper.py', 'gripper.py'), (ROOT/'sim/profile.json', 'profile.json'),
                         (ROOT/'calibrate.json', 'calibrate.json'), (ROOT/'tools/plot_gripper.py', 'plot.py')):
        (inputs / name).write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location('sim._archived_gripper_environment', inputs / 'legacy_environment.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.ROOT = ROOT  # Both versions use exactly the same assets and recordings.
    manifest = {'schema': 'openarm-gripper-study/1', 'kind': 'gripper', 'status': 'running',
        'run_id': output.name, 'variants': list(VARIANTS), 'episodes': [0, 1, 2],
        'selection': 'no fitting or selection by scores; algebraic correction with an ablation of gain conversion',
        'scope': 'development replay of recorded commands, not a policy benchmark or measured transmission',
        'inputs': {p.name: digest(p) for p in inputs.iterdir()},
        'source_hashes': {p: digest(ROOT/p) for p in ('sim/environment.py', 'sim/gripper.py',
            'sim/gripper_study.py', 'sim/profile.json', 'calibrate.json', 'sim/contacts.py',
            'task/bottle_to_box.xml', 'reference/real_trajectory.json', 'reference/acquisition.json',
            'sim/replay.py', 'sim/reference_clock.py', 'tools/plot_gripper.py')},
        'transfer_ready': False, 'physically_calibrated': False,
        'limitations': [
            'Transmission endpoints and gripper geometry remain candidates, not physical measurements.',
            'Inspected controller source has not been bound to the recording-time revision.',
            'Same recorded logger timestamps and object placement for each variant; sensor/CAN alignment remains unmeasured.',
            'Scores use the common observed prefix per episode; termination and coverage are reported separately.',
            'Only the gripper coordinate/gain conversion changes; arm controller and scene stay at their prior candidates.']}
    dump(output/'manifest.json', manifest)
    arrays, episodes, totals = {}, [], {v: np.zeros(2) for v in VARIANTS}
    total_duration = 0.
    try:
        for episode in (0, 1, 2):
            rows = {}
            for variant in VARIANTS:
                env = old.OpenArmEnv(images=False, profile=legacy_profile) if variant == 'legacy' else OpenArmEnv(images=False,
                    profile={**legacy_profile, 'robot_version':'v2', 'gripper_force_limit_value':7., 'gripper_force_limit_unit':'N·m'})
                try:
                    if variant == 'legacy':
                        env.profile = legacy_profile.copy()
                    elif variant == 'shared_coordinates':
                        for side in ('left', 'right'):
                            aid = env.model.actuator(f'{side}_finger1_ctrl').id
                            kp = env.calibration['actuators']['kp']['gripper']
                            env.model.actuator_gainprm[aid, 0] = kp
                            env.model.actuator_biasprm[aid, 1:3] = [-kp, 0.]
                    if variant == 'shared_motor_gain' and episode == 0:
                        manifest['model'] = env.manifest()
                    rows[variant] = rollout(env, episode)
                    reference = env.reference[str(episode)]
                finally:
                    env.close()
            target = np.asarray([reference[f'{s}_gripper'] for s in ('left', 'right')]).T
            commanded = np.asarray([reference[f'{s}_action_gripper'] for s in ('left', 'right')]).T
            sample_count = min(r['available_samples'] for r in rows.values())
            times = rows['shared_motor_gain']['times'][:sample_count]
            scored = {'episode': episode, 'common_samples': sample_count,
                      'scored_seconds': float(times[-1]), 'variants': {}}
            arrays[f'ep{episode}_target'] = target
            arrays[f'ep{episode}_command'] = commanded
            for variant, row in rows.items():
                arrays[f'ep{episode}_{variant}_state'] = row['states']
                arrays[f'ep{episode}_{variant}_time'] = row['times']
                if not np.allclose(row['times'][:sample_count], times, rtol=0, atol=1e-9):
                    raise ValueError('comparison clocks diverged')
                rmse, integral, duration = time_error(row['states'][:sample_count, [7, 15]], target[:sample_count], times)
                totals[variant] += integral
                scored['variants'][variant] = {
                    k: v for k, v in row.items() if k not in ('states', 'times')}
                scored['variants'][variant]['gripper_time_weighted_rmse_rad'] = rmse.tolist()
            total_duration += duration
            episodes.append(scored)
            print(json.dumps({'episode': episode, 'samples': sample_count,
                'status': {v: r['status'] for v, r in rows.items()}}), flush=True)
        summary = {'schema': manifest['schema'], 'status': 'completed', 'kind': 'gripper',
            'run_id': output.name, 'episodes': episodes, 'score_seconds': total_duration,
            'aggregate_rmse_rad': {v: np.sqrt(s/total_duration).tolist() for v, s in totals.items()},
            'coordinate_example': {'target_motor_rad': -.9, 'legacy_equilibrium_feedback_rad': -1.26,
                                   'corrected_equilibrium_feedback_rad': -.9},
            'current_variant': 'shared_motor_gain', 'physically_calibrated': False, 'transfer_ready': False,
            'limits': manifest['limitations'], 'report_url': f'/runs/{output.name}/index.html'}
        np.savez_compressed(output/'series.npz', **arrays)
        dump(output/'summary.json', summary)
        config = json.loads((ROOT/'sim/learning.json').read_text())
        process_env = {k: os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH') if k in os.environ}
        subprocess.run([config['plot_python'], '-B', str(ROOT/'tools/plot_gripper.py'), str(output)],
                       cwd=ROOT, env=process_env, timeout=60, check=True)
        manifest['outputs'] = {name: digest(output/name) for name in (
            'summary.json', 'series.npz', 'gripper.png', 'index.html')}
        manifest['status'] = 'completed'
        print(json.dumps({'report': summary['report_url'], 'rmse': summary['aggregate_rmse_rad']}), flush=True)
        return summary
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        dump(output/'manifest.json', manifest)


def latest():
    for path in sorted((ROOT/'runs').glob('*-gripper-*/summary.json'), reverse=True):
        result = json.loads(path.read_text())
        manifest = json.loads((path.parent/'manifest.json').read_text())
        if result.get('status') == manifest.get('status') == 'completed':
            if set(manifest.get('outputs', {})) != {'summary.json', 'series.npz', 'gripper.png', 'index.html'}:
                continue
            if any(digest(path.parent/name) != value for name,value in manifest['outputs'].items()):
                raise ValueError('gripper study output changed after verification')
            result['matches_current_model'] = (all(digest(ROOT/p) == h for p,h in manifest['source_hashes'].items())
                and json.loads((ROOT/'sim/profile.json').read_text()).get('robot_version','v2') == 'v2')
            return result
    return None
