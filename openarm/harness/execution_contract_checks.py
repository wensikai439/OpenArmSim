"""Evidence and dependency-boundary regressions; no model or GPU requests."""
import copy
from dataclasses import FrozenInstanceError
import json
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from harness.execution_contracts import completion_record


class CompletionContractChecks(unittest.TestCase):
    def record(self, recorded=True):
        self.result = {'run_id': 'run-1', 'status': 'primitive_completed',
                       'metrics': {'tamp': {'primitive_success': False}}}
        self.evaluation = {'success': False, 'failure_reasons': ['not released']}
        self.frames = [{'observation_id': 'epoch:2', 'sim_time': .1,
                        'sensor_urls': {'head': '/frame/head.jpg'},
                        'debug_object_truth': [1, 2, 3]},
                       {'url': '/debug.jpg', 'label': 'operator-only'}]
        return completion_record(
            epoch='epoch', execution_id='execution-1', last_run=self.result,
            evaluation=self.evaluation, keyframes=self.frames,
            terminal={'observation_id': 'epoch:3',
                      'observation': {'tick': 3, 'state': [0, 1]}},
            recorded=recorded)

    def test_terminal_evidence_is_detached_from_mutable_execution_state(self):
        receipt = self.record()
        saved = copy.deepcopy(receipt)
        self.result['metrics']['tamp']['primitive_success'] = True
        self.evaluation['failure_reasons'].clear()
        self.frames[0]['sensor_urls']['head'] = '/different.jpg'
        self.assertEqual(receipt, saved)
        self.assertFalse(receipt['result']['metrics']['tamp']['primitive_success'])
        self.assertEqual(receipt['execution_id'], 'execution-1')
        self.assertEqual(receipt['observation']['observation_id'], 'epoch:3')

    def test_camera_receipt_does_not_copy_debug_truth_or_operator_only_frames(self):
        receipt = self.record()
        self.assertEqual(len(receipt['camera_evidence']), 1)
        self.assertNotIn('debug_object_truth', receipt['camera_evidence'][0])
        self.assertNotIn('evaluation', receipt['observation'])
        self.assertFalse(receipt['evaluation']['success'])

    def test_unrecorded_attempt_cannot_reuse_previous_camera_evidence(self):
        self.assertEqual(self.record(recorded=False)['camera_evidence'], [])

    def test_contract_import_does_not_load_simulator_or_model_runtime(self):
        script = ('import sys,json; import harness.execution_contracts; '
                  'print(json.dumps([x for x in sys.modules '
                  'if x == "mujoco" or x == "torch" or x.startswith("sim.")]))')
        value = subprocess.check_output([sys.executable, '-B', '-c', script], text=True)
        self.assertEqual(json.loads(value), [])


class ProviderBoundaryChecks(unittest.TestCase):
    def test_existing_entrypoint_uses_calling_thread_and_instance_providers(self):
        from sim import service
        owner = service.SimulationService.__new__(service.SimulationService)
        owner.policy_client_factory = lambda *a, **k: None
        owner.policy_timing_class = lambda mode: mode
        seen = []
        def run(host, kind, args, providers):
            seen.append((host, kind, args, providers, threading.get_ident()))
        caller = threading.get_ident()
        with patch.object(service, 'run_execution', side_effect=run):
            owner._run('policy', {'observation_id': 'epoch:0'})
        host, kind, args, providers, thread = seen[0]
        self.assertIs(host, owner)
        self.assertEqual((kind, args, thread), ('policy', {'observation_id': 'epoch:0'}, caller))
        self.assertIs(providers.policy_client, owner.policy_client_factory)
        self.assertIs(providers.policy_timing, owner.policy_timing_class)
        with self.assertRaises(FrozenInstanceError):
            providers.policy_client = object()

    def test_legacy_default_hooks_stay_late_bound_between_jobs(self):
        from sim import service
        owner = service.SimulationService.__new__(service.SimulationService)
        seen = []
        with patch.object(service, 'run_execution', side_effect=lambda h, k, a, p: seen.append(p)):
            with patch.object(service, 'PolicyTiming', lambda mode: 'first'):
                owner._run('policy', {})
            with patch.object(service, 'PolicyTiming', lambda mode: 'second'):
                owner._run('policy', {})
        self.assertEqual(seen[0].policy_timing('paused'), 'first')
        self.assertEqual(seen[1].policy_timing('paused'), 'second')


if __name__ == '__main__':
    unittest.main()
