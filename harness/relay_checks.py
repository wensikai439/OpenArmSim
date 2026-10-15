"""Modern donor clearance keeps legacy routing and native rejection guards."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sim.fast_handover import ForwardBottleHandover
from sim.tamp import PlanningFailure
from .margin_grasp import compose_planner


class RelayChecks(unittest.TestCase):
    def test_forward_receipt_identity_satisfies_directed_workspace_binding(self):
        from .bottle import BottleWorkspace
        def initialize(planner, env, task, **kwargs):
            planner.result = {'task': task}
        with patch('sim.tamp.TampPlanner.__init__', initialize):
            planner = ForwardBottleHandover(SimpleNamespace(object_id='bottle'), 'handover_bottle')
        feedback = {**planner.result, 'status': 'completed', 'primitive_success': True,
                    'verification_source': 'native_receiver_contact_then_donor_release',
                    'from_arm': 'right', 'to_arm': 'left', 'held_by': 'left'}
        work = BottleWorkspace()
        try:
            arguments = {'object_id': 'bottle', 'from_arm': 'right', 'to_arm': 'left'}
            work.record_feedback('relay', 'execute', 'episode', '交接', 'handover', arguments,
                                 feedback, evidence_kind='simulator_oracle')
            row = work.db.execute('SELECT outcome,feedback FROM experiences WHERE id=?', ('relay',)).fetchone()
            import json
            stored = json.loads(row['feedback'])
            self.assertEqual(row['outcome'], 'success')
            self.assertTrue(stored['execution_binding']['valid'])
            stage = {'checkpoint': 'handover_confirmed', 'skill': 'handover'}
            plan = {'placement_goal': {'object_id': 'bottle'}}
            self.assertTrue(work._step_done(stage, stored, plan))
            wrong_object = {**stored, 'object_id': 'can'}
            self.assertFalse(work._step_done(stage, wrong_object, plan))
        finally:
            work.db.close()

    def test_composite_uses_injected_native_primitives_without_changing_gpu_choice(self):
        from tools.object_workbench import bind_relocation_primitives
        from sim.tamp import TampPlanner
        calls = []
        class Previous:
            def _skill(self, name, parameters, planner_type=TampPlanner):
                calls.append((name, planner_type))
        primitive = object()
        explicit_gpu = object()
        bound = bind_relocation_primitives(Previous, primitive)
        bound()._skill('handover', {})
        bound()._skill('move', {}, explicit_gpu)
        self.assertEqual(calls, [('handover', primitive), ('move', explicit_gpu)])
        self.assertIs(bind_relocation_primitives(bound, primitive), bound)
        self.assertNotIn('_bound_primitive_factory', Previous.__dict__)

    def test_fixed_full_demo_keeps_previous_provider(self):
        previous = Mock()
        previous.CONTRACT = {}
        previous.MARGIN_GRASP_ROUTER = False
        env = SimpleNamespace(object_id='bottle', bottle_id=0,
                              data=SimpleNamespace(xmat=np.eye(3).reshape(1, 9)))
        with patch('sim.fast_handover.ForwardBottleHandover') as modern:
            router = compose_planner(previous)
            router(env, 'handover_to_basket', parameters={})
        previous.assert_called_once_with(env, 'handover_to_basket', parameters={})
        modern.assert_not_called()

    def make_planner(self, donor_held=False):
        planner = object.__new__(ForwardBottleHandover)
        planner.env = SimpleNamespace(evaluation={'held': {'left': True, 'right': donor_held},
                                                  'touching': {'left': True, 'right': False}})
        planner.rotation = np.eye(3)
        planner.result = {}
        planner._sync = Mock()
        planner._ik = Mock(return_value=np.zeros(7))
        planner._path = Mock()
        planner._move = Mock(return_value=iter(['native-command']))
        return planner

    def test_donor_cannot_move_before_receiver_only_hold(self):
        planner = self.make_planner(donor_held=True)
        with self.assertRaises(PlanningFailure):
            list(planner._retreat_after_forward_handover(np.zeros(3)))
        planner._ik.assert_not_called()
        planner._move.assert_not_called()

    def test_rejected_collision_proposal_never_executes(self):
        planner = self.make_planner()
        planner._ik.side_effect = [PlanningFailure('goal_collision', 'blocked'), np.zeros(7)]
        self.assertEqual(list(planner._retreat_after_forward_handover(np.zeros(3))), ['native-command'])
        self.assertEqual(planner._ik.call_count, 2)
        planner._path.assert_called_once()
        planner._move.assert_called_once()
        np.testing.assert_allclose(planner._move.call_args.args[0], [.04, -.14, -.04])
        self.assertTrue(planner.result['donor_retreat_search']['collision_gates_unchanged'])

    def test_native_execution_failure_is_not_retried_as_another_proposal(self):
        planner = self.make_planner()
        planner._move.side_effect = PlanningFailure('grasp_lost', 'native hold lost')
        with self.assertRaises(PlanningFailure):
            list(planner._retreat_after_forward_handover(np.zeros(3)))
        self.assertEqual(planner._ik.call_count, 1)
        planner._move.assert_called_once()


if __name__ == '__main__':
    unittest.main()
