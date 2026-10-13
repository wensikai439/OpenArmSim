"""Optional router preserves upright relay and binds only the locked goal."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from harness.object_handover import ProtectedObjectEnv, RelayPlanner
from harness.poses import PoseHarness, service_factory
from sim.oriented_grasp import OrientedPlanner
from tools.check_pose_grasp import CASES, initialize


class PoseRouterChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.check_task_queue import setup
        setup(); cls.env = ProtectedObjectEnv(images=False)

    @classmethod
    def tearDownClass(cls): cls.env.close()

    def test_upright_pick_keeps_existing_relay_planner_and_native_state(self):
        import sim.service as service
        old_env, old_planner = service.OpenArmEnv, service.TampPlanner
        self.addCleanup(setattr, service, 'OpenArmEnv', old_env)
        self.addCleanup(setattr, service, 'TampPlanner', old_planner)
        service_factory(); env = self.env; env.reset(seed=0); env.select_object('can')
        before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time]
        planner = service.TampPlanner(env, 'pick_bottle', parameters={'object_id': 'can', 'arm': 'right'})
        expected = RelayPlanner(env, 'pick_bottle', parameters={'object_id': 'can', 'arm': 'right'})
        self.assertIs(type(planner), type(expected))
        intent = {'object_id': 'can', 'arm': 'right', 'next_handover': {'to_arm': 'left'}}
        planned = service.TampPlanner(env, 'pick_bottle', parameters=intent)
        self.assertIs(type(planned), type(RelayPlanner(env, 'pick_bottle', parameters=intent)))
        for old, new in zip(before[:3], (env.data.qpos, env.data.qvel, env.data.ctrl)): np.testing.assert_array_equal(old, new)
        self.assertEqual(before[3], env.data.time)

    def test_non_upright_route_is_optional_and_keeps_original_physics(self):
        import sim.service as service
        old_env, old_planner = service.OpenArmEnv, service.TampPlanner
        self.addCleanup(setattr, service, 'OpenArmEnv', old_env)
        self.addCleanup(setattr, service, 'TampPlanner', old_planner)
        service_factory(); env = self.env; initialize(env, CASES[2])
        original = copy.deepcopy(env.manifest()['model_evidence'])
        planner = service.TampPlanner(env, 'pick_bottle', parameters={'object_id': 'block', 'arm': 'left'})
        self.assertIsInstance(planner, OrientedPlanner)
        self.assertEqual(original, env.manifest()['model_evidence'])

    def test_lookahead_context_copies_the_same_goal_without_automatic_motion(self):
        import sim.service as service
        old_planner = service.TampPlanner; self.addCleanup(setattr, service, 'TampPlanner', old_planner)
        class DummyService:
            def call(self, command, **arguments): return {'command': command, 'arguments': arguments}
        with patch('harness.task_queue.service_factory', return_value=DummyService):
            factory = service_factory()
        source = {'task': 'pick_bottle', 'parameters': {'object_id': 'block', 'arm': 'left'},
            'initiator': {'pose_pick_lookahead': {'surface': 'table', 'target_xy_m': [.55, .08], 'position_tolerance_m': .01}}}
        before = copy.deepcopy(source); result = factory().call('tamp', **source)
        self.assertEqual(source, before)
        self.assertEqual(result['arguments']['parameters']['next_placement'], source['initiator']['pose_pick_lookahead'])
        self.assertEqual(result['command'], 'tamp')
        self.assertEqual(factory().call('capture', label='fixture')['arguments'], {'label': 'fixture'})

    def test_pick_hint_is_bound_to_this_native_object_goal_epoch_and_holder(self):
        from types import SimpleNamespace
        feedback = {'object_id': 'block', 'primitive_success': True, 'held_by': 'left',
            'verification_source': 'native_two_finger_contact_and_lift',
            'grasp_start': {'orientation_class': 'non_upright'},
            'future_placement': {'target_xy_m': [.55, .08], 'yaw_rad': .2}}
        receipt = {'feedback': feedback, 'evidence_kind': 'simulator_oracle',
            'evidence': {'after': {'observation_id': 'same:123'}}}
        goal = {'object_id': 'block', 'surface': 'table', 'target_xy_m': [.55, .08]}
        snapshot = {'observation_id': 'same:200', 'memory_context': {'contract': 'fixture'},
            'diagnostics': {'scene_directory': {'objects': [{'object_id': 'block',
                'rotation_matrix': [[np.cos(.4), -np.sin(.4), 0.], [np.sin(.4), np.cos(.4), 0.], [0., 0., 1.]]}]}}}
        harness = PoseHarness.__new__(PoseHarness)
        harness.workspace = SimpleNamespace(active_plan={'execution_tools': [{'skill': 'pick', 'id': 'p'}]},
            execution_receipt=lambda *args, **kwargs: copy.deepcopy(receipt))
        original = copy.deepcopy((snapshot, goal, receipt))
        angle, ident = harness._lookahead(snapshot, goal, 'left')
        self.assertAlmostEqual(angle, -.2); self.assertEqual(ident, 'p')
        self.assertEqual(original, (snapshot, goal, receipt))
        for field, wrong in [('object_id', 'can'), ('held_by', 'right'), ('primitive_success', False),
            ('verification_source', 'model')]:
            saved = copy.deepcopy(feedback); feedback[field] = wrong
            self.assertEqual(harness._lookahead(snapshot, goal, 'left'), (0., None))
            feedback.clear(); feedback.update(saved)
        for value in ('old:123', ''):
            receipt['evidence']['after']['observation_id'] = value
            self.assertEqual(harness._lookahead(snapshot, goal, 'left'), (0., None))
        receipt['evidence']['after']['observation_id'] = 'same:123'
        receipt['evidence_kind'] = 'model_hypothesis'
        self.assertEqual(harness._lookahead(snapshot, goal, 'left'), (0., None))
        receipt['evidence_kind'] = 'simulator_oracle'
        feedback['future_placement']['target_xy_m'] = [.56, .08]
        self.assertEqual(harness._lookahead(snapshot, goal, 'left'), (0., None))

    def test_intervening_handover_never_reuses_the_pick_grasp_hint(self):
        from types import SimpleNamespace
        harness = PoseHarness.__new__(PoseHarness)
        def missing(*args, **kwargs): raise ValueError('no compatible native handover receipt')
        harness.workspace = SimpleNamespace(active_plan={'execution_tools': [
            {'skill': 'pick', 'id': 'p'}, {'skill': 'handover', 'id': 'h'}]}, execution_receipt=missing)
        self.assertEqual(harness._lookahead({'memory_context': {}}, {'surface': 'table'}, 'left'), (0., None))


if __name__ == '__main__': unittest.main()
