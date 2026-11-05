"""Measured-delay, native relocation after object-centered grasp search.

The bottle offset belongs to the declared initial condition. No task-time
reset, pose edit, physical-parameter change or scoring change is permitted.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from sim.environment import ROOT
from sim.recording import dump
from sim.grasp import CONTRACT, install_isolated as install_grasp
from harness.composite import install_isolated as install_composite
from tools.contact_candidate import ContactCandidate, DERIVATION

OUT = ROOT / 'runs/20261008-ik'
CASES = {
    'original-left': {'delta_m': [0., 0., 0.], 'arm': 'left', 'via_arm': 'right', 'distance_m': .06},
    'shifted-left': {'delta_m': [.015, -.02, 0.], 'arm': 'left', 'via_arm': 'right', 'distance_m': .06},
    'original-right': {'delta_m': [0., 0., 0.], 'arm': 'right', 'via_arm': None, 'distance_m': .08},
}


def install(case):
    import sim.service as service
    install_composite()
    install_grasp()

    class InitialScene(ContactCandidate):
        def reset(self, **kwargs):
            super().reset(**kwargs)
            address = self.model.joint('bottle_free').qposadr[0]
            self.data.qpos[address:address + 3] += case['delta_m']
            mujoco.mj_forward(self.model, self.data)
            self.evaluation = self._evaluate(0.)
            return self.observe(), self.info()

    service.OpenArmEnv = InitialScene
    return service


def evidence(case):
    return {'contact_candidate': DERIVATION, 'grasp_sampler': CONTRACT,
        'declared_initial_condition': case, 'sampler_sha256': hashlib.sha256((ROOT / 'sim/grasp.py').read_bytes()).hexdigest(),
        'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'physics_parameters_changed_during_task': False, 'scoring_thresholds_changed': False,
        'main_page_commands': 0, 'hardware_access': False, 'transfer_ready': False}


def native():
    target = OUT / 'routes.json'
    if target.exists(): raise ValueError('preserve the previous declared route study')
    protocol = {'cases': CASES, 'timing_mode': 'measured_delay', 'budget_s': 120,
        'seed': 0, 'episode': 0, 'cold_preparation_counted_as_physics_wait': True,
        'initial_condition_only_before_task': True, 'actual_LLM_API': False,
        'scope': 'three declared development cases; no statistical or physical transfer claim',
        'evidence': evidence(CASES)}
    dump(target.with_suffix('.protocol.json'), protocol)
    rows = []
    for name, case in CASES.items():
        module = install(case)
        service = module.SimulationService()
        try:
            service.call('reset', episode=0, seed=0)
            service.call('task_budget', seconds=120)
            initial = service.snapshot()
            manifest = service.call('manifest')
            parameters = {'object_id': 'bottle', 'arm': case['arm'], 'receptacle_id': 'table',
                'offset_xy_m': [case['distance_m'], 0.], 'transport_planner': 'curobo',
                'transport_delta_m': [.015, 0., 0.]}
            if case['via_arm']: parameters['via_arm'] = case['via_arm']
            started = time.monotonic()
            receipt = service.call('tamp', task='relocate_object', skill='relocate', parameters=parameters,
                observation_id=initial['observation_id'], timing_mode='measured_delay', realtime=False)
            deadline = time.monotonic() + 300
            while True:
                state = service.snapshot()
                if not state.get('job'): break
                if time.monotonic() > deadline: raise TimeoutError('same native owner needs inspection; no restart')
                time.sleep(.1)
            last = state['last_run']
            summary = json.loads((ROOT / 'runs' / last['run_id'] / 'summary.json').read_text())
            feedback = last['metrics'].get('tamp') or {}
            substeps = feedback.get('substeps', [])
            verified_curobo = any(r.get('backend') == 'curobo' and r['status'] == 'completed'
                and r['feedback'].get('native_path_revalidation') and r['feedback'].get('fallback') is False for r in substeps)
            verified_transfer = not case['via_arm'] or any(r['skill'] == 'handover' and r['status'] == 'completed'
                and r['parameters'].get('from_arm') == case['arm'] and r['parameters'].get('to_arm') == case['via_arm'] for r in substeps)
            goal = np.array(initial['diagnostics']['manipulation_scene']['position_m'][:2]) + [case['distance_m'], 0.]
            declared = np.array(feedback.get('target_xy_m', []))
            target_unchanged = bool(declared.shape == (2,) and np.linalg.norm(goal - declared) < 1e-6)
            passed = bool(feedback.get('primitive_success') and verified_curobo and verified_transfer and target_unchanged
                and manifest['model_evidence'] == service.call('manifest')['model_evidence'])
            row = {'case': name, 'passed': passed, 'parameters': parameters, 'initial': initial,
                'receipt': receipt, 'status': last['status'], 'error': last['error'], 'run_id': last['run_id'],
                'wall_seconds': time.monotonic() - started, 'duration_sim_seconds': summary['duration_sim_seconds'],
                'metrics': last['metrics'], 'terminal_evaluation': state['diagnostics']['evaluation'],
                'verified_curobo': bool(verified_curobo), 'verified_transfer': bool(verified_transfer),
                'target_unchanged': target_unchanged, 'manifest': manifest,
                'model_evidence_unchanged': manifest['model_evidence'] == service.call('manifest')['model_evidence']}
            rows.append(row)
            dump(target, {'protocol': protocol, 'cases': rows, 'complete': len(rows) == len(CASES)})
            print(json.dumps({k: row[k] for k in ('case', 'passed', 'status', 'error', 'duration_sim_seconds')}, ensure_ascii=False), flush=True)
        finally:
            service.close()


def api(case_name, output=None):
    case = CASES[case_name]
    install(case)
    import harness.agent as agent
    from harness.candidate import CandidateHarness, install_isolated as install_candidate
    from harness.provider import install_isolated as install_provider
    from tools import check_cache

    install_candidate()
    class CheckedHarness(CandidateHarness):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install_provider(self)

    agent.Harness = CheckedHarness  # only this independent process
    check_cache.OUT = OUT
    check_cache.isolated_api(output or 'api-' + case_name + '.json', arm=case['arm'], via_arm=case['via_arm'],
        distance_m=case['distance_m'], experiment_details={**evidence(case),
            'provider_mode': 'documented non-thinking, unchanged 1600 token output cap',
            'composite_stage_receipts': 'same verified native child binding'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['native', 'api'], required=True)
    parser.add_argument('--case', choices=CASES, default='shifted-left')
    parser.add_argument('--output')
    arguments = parser.parse_args()
    if arguments.output and ('/' in arguments.output or not arguments.output.endswith('.json')):
        raise ValueError('simple JSON filename required')
    native() if arguments.mode == 'native' else api(arguments.case, arguments.output)
