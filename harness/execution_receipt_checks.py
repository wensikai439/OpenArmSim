"""Durable execution contracts retain task identity and never resume after restart."""
import tempfile
import unittest
from pathlib import Path

from .workspace import Workspace
from sim.service import execution_lifecycle, planner_provenance


class ExecutionReceiptChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'workspace.sqlite3'
        self.context = {'robot': 'openarm-v1', 'contract': 'test'}
        self.steps = [{'id': 'move', 'title': '前移', 'checkpoint': 'tool_completed', 'skill': 'move_relative'}]

    def _plan(self, work):
        return work.plan('root-turn', 'epoch-a', 'test-model', self.steps,
            placement_goal={'object_id': 'bottle', 'surface': 'table', 'target_xy_m': [.5, -.1]},
            time_budget_s=60, timing_mode='paused')

    def _contract(self, task_id):
        return {'schema': 'openarm-execution/1', 'task_id': task_id,
            'root_execution_id': 'root-turn', 'execution_id': 'exec-1',
            'observation_id': 'epoch-a:12', 'epoch': 'epoch-a',
            'skill_id': 'move_relative', 'skill_version': '1.0', 'backend_id': 'curobo',
            'object_id': 'bottle', 'arms': ['right'], 'frame': 'world',
            'target': {'delta_m': [.08, 0, 0]},
            'locked_goal': {'object_id': 'bottle', 'surface': 'table', 'target_xy_m': [.5, -.1]},
            'time_budget_s': 60, 'timing_mode': 'paused', 'status': 'accepted'}

    def test_task_identity_survives_replanning_and_terminal_receipt_is_bound(self):
        work = Workspace(self.path)
        try:
            self._plan(work)
            task_id = work.active_plan['task_id']
            self._plan(work)
            self.assertEqual(work.active_plan['task_id'], task_id)
            self.assertEqual(work.active_plan['time_budget_s'], 60)
            self.assertEqual(work.active_plan['placement_goal']['target_xy_m'], [.5, -.1])
            contract = self._contract(task_id)
            work.bind('root-turn', 'exec-1', 'execute', '前移瓶子', 'move_relative', 'move_relative',
                'test-model', context=self.context, request_contract=contract)
            work.complete('exec-1', 'run-1')
            work.record_feedback('exec-1', 'execute', 'epoch-a', '前移瓶子', 'move_relative',
                {'arm': 'right', 'delta_m': [.08, 0, 0]},
                {'primitive_success': True, 'verification_source': 'robot_forward_kinematics'},
                self.context, 'simulator_oracle', evidence={'request_contract': contract,
                    'after': {'observation_id': 'epoch-a:48'}})
            receipt = work.execution_receipt('execute', 'exec-1', context=self.context, include_oracle=True)
            self.assertEqual(receipt['status'], 'succeeded')
            self.assertEqual(receipt['request_contract']['status'], 'succeeded')
            self.assertEqual(receipt['request_contract']['task_id'], task_id)
            self.assertEqual(receipt['request_contract']['root_execution_id'], 'root-turn')
            self.assertEqual(receipt['request_contract']['observation_id'], 'epoch-a:12')
            self.assertEqual(receipt['request_contract']['terminal_observation_id'], 'epoch-a:48')
            self.assertEqual(receipt['request_contract']['locked_goal']['target_xy_m'], [.5, -.1])
            self.assertEqual(receipt['request_contract']['backend_id'], 'curobo')
        finally:
            work.db.close()

    def test_restart_turns_an_inflight_accepted_request_into_nonresumable_interruption(self):
        first = Workspace(self.path)
        self._plan(first)
        task_id = first.active_plan['task_id']
        first.bind('root-turn', 'exec-1', 'execute', '前移瓶子', 'move_relative', 'move_relative',
            'test-model', context=self.context, request_contract=self._contract(task_id))
        first.db.close()
        reopened = Workspace(self.path)
        try:
            receipt = reopened.execution_receipt('execute', 'exec-1', context=self.context)
            self.assertEqual(receipt['status'], 'interrupted')
            self.assertEqual(receipt['request_contract']['status'], 'interrupted')
            self.assertEqual(receipt['request_contract']['task_id'], task_id)
            self.assertIsNone(receipt['feedback'])
            self.assertEqual(reopened.active_plan, None)
        finally:
            reopened.db.close()

    def test_rejected_request_is_not_an_execution_success(self):
        work = Workspace(self.path)
        try:
            contract = self._contract('task-1')
            contract['status'] = 'rejected'
            work.record_feedback('rejected-1', 'execute', 'epoch-a', '不支持的目标', 'relocate',
                {'object_id': 'block'}, {'status': 'rejected', 'motion_started': False},
                self.context, 'contract_guard', evidence={'request_contract': contract, 'motion_started': False})
            receipt = work.execution_receipt('execute', 'rejected-1', context=self.context)
            self.assertEqual(receipt['status'], 'rejected')
            self.assertFalse(receipt['evidence']['motion_started'])
            self.assertFalse(receipt['feedback'].get('primitive_success', False))
        finally:
            work.db.close()

    def test_terminal_lifecycle_never_promotes_policy_exit_or_cancel(self):
        self.assertEqual(execution_lifecycle('tamp','primitive_completed',
            {'tamp':{'primitive_success':True}}),'succeeded')
        self.assertEqual(execution_lifecycle('tamp','primitive_completed',
            {'tamp':{'primitive_success':False}}),'failed')
        self.assertEqual(execution_lifecycle('tamp','cancelled',
            {'tamp':{'primitive_success':True}}),'cancelled')
        self.assertEqual(execution_lifecycle('policy','success',{}),'unverified')

    def test_planner_receipt_names_selected_factory_and_concrete_implementation(self):
        class Router: pass
        class ConnectedPlanner:
            CONTRACT={'planner':'tamp','proposal':'connected-path'}
        result=planner_provenance(Router,ConnectedPlanner(),'tamp','place_bottle')
        self.assertEqual(result['requested_factory']['class'],'ExecutionReceiptChecks.test_planner_receipt_names_selected_factory_and_concrete_implementation.<locals>.Router')
        self.assertEqual(result['implementation']['class'],'ExecutionReceiptChecks.test_planner_receipt_names_selected_factory_and_concrete_implementation.<locals>.ConnectedPlanner')
        self.assertEqual(result['backend'],'tamp')
        self.assertIn('not a performance',result['scope'])


if __name__ == '__main__':
    unittest.main()
