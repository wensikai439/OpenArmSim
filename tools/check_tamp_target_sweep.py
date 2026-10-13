"""Repeat a bounded native TAMP target/arm sweep and archive each outcome.

This is an oracle-geometry simulator diagnostic. It varies robot start state,
arm and table target while keeping the registered bottle and scene fixed. It
does not exercise language understanding, perception, GPU policies or hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.environment import OpenArmEnv
from sim.relocation import RelocationPlanner
from sim.tamp import PlanningFailure


CASES = {
    'front_8cm': [0.08, 0.0],
    'left_4cm': [0.0, 0.04],
    'right_4cm': [0.0, -0.04],
    'back_4cm': [-0.04, 0.0],
}
SOURCES = ('sim/environment.py', 'sim/relocation.py', 'sim/tamp.py', 'sim/placement.py')
_MODEL_IDENTITY: dict | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(env: OpenArmEnv, planner: RelocationPlanner, *, max_steps: int) -> int:
    actions = planner.actions()
    steps = 0
    while True:
        qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
        try:
            action, _ = next(actions)
        except StopIteration:
            return steps
        np.testing.assert_array_equal(qpos, env.data.qpos)
        np.testing.assert_array_equal(qvel, env.data.qvel)
        _, _, terminated, truncated, _ = env.step_tamp(action)
        steps += 1
        if terminated or truncated:
            raise RuntimeError('simulation terminated or truncated before skill receipt')
        if steps > max_steps:
            raise RuntimeError('bounded diagnostic step budget exceeded')


def run_case(episode: int, arm: str, label: str, offset: list[float]) -> dict:
    global _MODEL_IDENTITY
    env = OpenArmEnv(images=False)
    env.reset(seed=0, options={'episode': episode})
    env.set_task_budget(120)
    if _MODEL_IDENTITY is None:
        evidence = env._model_evidence
        compiled, inputs = evidence['compiled'], evidence['inputs']
        _MODEL_IDENTITY = {
            'engine': compiled['engine'],
            'compiled_model_sha256': compiled['sha256'],
            'input_resources_sha256': inputs['content_sha256'],
            'physically_calibrated': compiled['physically_calibrated'],
            'transfer_ready': compiled['transfer_ready'],
        }
    origin = env.data.xpos[env.bottle_id, :2].copy()
    initial_joints = {
        side: env.data.qpos[env.qids[side]].astype(float).tolist()
        for side in ('left', 'right')
    }
    base = {'robot_start_episode': episode, 'arm': arm, 'case': label,
            'offset_xy_m': offset, 'initial_bottle_xy_m': origin.tolist(),
            'initial_arm_joints_rad': initial_joints}
    t0 = time.monotonic()
    first: dict = {'status': 'not_started'}
    recovery: dict | None = None
    planner = None
    try:
        params = {'object_id': 'bottle', 'arm': arm, 'receptacle_id': 'table',
                  'offset_xy_m': offset}
        planner = RelocationPlanner(env, 'relocate_object', parameters=params)
        try:
            steps = execute(env, planner, max_steps=7200)
            first = {'status': planner.result.get('status'), 'steps': steps,
                     'placement_confirmed': planner.result.get('placement', {}).get('confirmed'),
                     'position_error_m': planner.result.get('position_error_m'),
                     'route': [{'skill': row['skill'], 'status': row['status'],
                                'failure_code': row.get('failure_code')}
                               for row in planner.result.get('substeps', [])]}
        except Exception as exc:
            first = {'status': 'failed', 'failure_code': getattr(exc, 'code', type(exc).__name__),
                     'message': str(exc), 'tick': env.tick,
                     'held_by': next((side for side, held in env.evaluation['held'].items() if held), None),
                     'route': [{'skill': row['skill'], 'status': row['status'],
                                'failure_code': row.get('failure_code')}
                               for row in planner.result.get('substeps', [])]}
            held_by = first['held_by']
            if (held_by == arm and first['failure_code'] in RelocationPlanner.RECOVERABLE
                    and not env._done and not env.evaluation['invalid']):
                recovery_planner = RelocationPlanner(env, 'relocate_object', parameters={
                    **params, 'allow_handover_recovery': True})
                try:
                    steps = execute(env, recovery_planner, max_steps=7200)
                    recovery = {'status': recovery_planner.result.get('status'),
                                'success': recovery_planner.result.get('primitive_success') is True,
                                'recovery_attempted': recovery_planner.result.get('recovery_attempted'),
                                'final_arm': recovery_planner.result.get('arm'),
                                'position_error_m': recovery_planner.result.get('position_error_m'),
                                'placement_confirmed': recovery_planner.result.get('placement', {}).get('confirmed'),
                                'steps': steps,
                                'route': [{'skill': row['skill'], 'status': row['status'],
                                           'failure_code': row.get('failure_code')}
                                          for row in recovery_planner.result.get('substeps', [])]}
                except Exception as recovery_exc:
                    recovery = {'status': 'failed',
                                'failure_code': getattr(recovery_exc, 'code', type(recovery_exc).__name__),
                                'message': str(recovery_exc), 'tick': env.tick,
                                'route': [{'skill': row['skill'], 'status': row['status'],
                                           'failure_code': row.get('failure_code')}
                                          for row in recovery_planner.result.get('substeps', [])]}
        actual = env.data.xpos[env.bottle_id, :2]
        return {**base, 'first_attempt': first, 'same_state_recovery': recovery,
                'final_bottle_xy_m': actual.astype(float).tolist(), 'final_tick': env.tick,
                'elapsed_wall_seconds': round(time.monotonic() - t0, 3)}
    finally:
        env.close()


def build_report() -> dict:
    cases = [run_case(ep, arm, label, offset)
             for ep in (0, 1, 2)
             for arm in ('left', 'right')
             for label, offset in CASES.items()]
    def passed(row: dict) -> bool:
        first = row['first_attempt']
        if first.get('status') == 'completed' and first.get('placement_confirmed') is True:
            return True
        recovery = row.get('same_state_recovery') or {}
        return recovery.get('status') == 'completed' and recovery.get('placement_confirmed') is True

    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return {
        'schema': 'oracle-relocation-target-sweep/1',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'code_revision': revision,
        'source_sha256': {name: sha256(ROOT / name) for name in SOURCES},
        'simulation_model': _MODEL_IDENTITY,
        'scope': {
            'physics': 'native MuJoCo joint/actuator/contact execution with independent support/release/stability receipt',
            'object': 'single registered upright bottle; fixed world pose and fixed table geometry',
            'variation': 'two arms, four table offsets, three robot joint start states selected from recorded episode starts 0/1/2',
            'recovery': 'only after a recoverable planning failure while the requested arm is independently confirmed holding; one same-state handover-to-other-arm retry with the original target unchanged',
            'excluded': ['LLM/API', 'camera perception', 'object pose variation', 'obstacle variation', 'GPU/cuRobo', 'ACT/π0.5', 'physical calibration', 'real robot'],
            'interpretation': 'diagnostic coverage of a narrow known-object action family; not an arbitrary-task generalization rate or sim-to-real claim',
        },
        'parameters': {'seed': 0, 'task_budget_seconds': 120, 'offsets_xy_m': CASES,
                       'arms': ['left', 'right'], 'robot_start_episodes': [0, 1, 2],
                       'placement_success': 'native confirmed placement at locked target; recovery allowed only in separate field'},
        'counts': {
            'cases': len(cases),
            'first_attempt_successes': sum(c['first_attempt'].get('status') == 'completed' and c['first_attempt'].get('placement_confirmed') is True for c in cases),
            'first_attempt_failures': sum(c['first_attempt'].get('status') != 'completed' or c['first_attempt'].get('placement_confirmed') is not True for c in cases),
            'recovery_attempts': sum(c.get('same_state_recovery') is not None for c in cases),
            'recovery_successes': sum((c.get('same_state_recovery') or {}).get('status') == 'completed' and (c.get('same_state_recovery') or {}).get('placement_confirmed') is True for c in cases),
            'eventual_successes': sum(passed(c) for c in cases),
        },
        'cases': cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/20261013-tamp-target-sweep/result.json')
    args = parser.parse_args()
    report = build_report()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'completed', 'output': str(output), 'counts': report['counts'],
                      'code_revision': report['code_revision']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
