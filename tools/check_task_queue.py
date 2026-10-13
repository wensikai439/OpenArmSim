"""Multi-goal identity, clock/physics preservation and native scoring gates."""
import copy
import time
import unittest

import numpy as np

from harness.object_handover import ProtectedObjectEnv
from sim.task_scene import TaskScene
from sim.scene import SceneDirectory
from harness.task_queue import TaskHarness, TaskWorkspace, assess_tasks, object_snapshot
from harness.objects import resolve_goal


def setup():
    from harness.composite import install_isolated as composite
    from sim.grasp import install_isolated as grasp
    from harness.candidate import install_isolated as candidate
    from harness.scene import install_isolated as scene
    from harness.object_gpu import install_isolated as gpu
    from harness.object_handover import install_isolated as relay
    from harness.object_transport import install_isolated as transport
    from harness.task_queue import install_isolated as tasks
    composite(); grasp(); candidate(); scene(); gpu(); relay(); transport(); tasks()


class QueueChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.env = TaskScene(images=False); setup()
    @classmethod
    def tearDownClass(cls): cls.env.close()

    def setUp(self):
        env = self.env; env.reset(seed=0); env.select_object('can')
        self.snapshot = {'observation_id': 'initial:0', 'memory_context': {'contract': 'fixture'},
            'diagnostics': env.diagnostics(), 'job': None,
            'observation': {'sim_time': 0., 'state': env.state().tolist(), 'tcp': {}, 'task_window': env.task_window()}}
        self.snapshot['diagnostics']['scene_directory'] = SceneDirectory(env).snapshot('initial:0')
        self.goals = [resolve_goal(spec, object_snapshot(self.snapshot, spec['object_id'])) for spec in (
            {'object_id': 'can', 'surface': 'table', 'offset_xy_m': [.08, 0.]},
            {'object_id': 'block', 'surface': 'table', 'target_xy_m': [.503, -.345]})]
        self.workspace = TaskWorkspace(); self.addCleanup(self.workspace.db.close)
        self.workspace.active_plan = {'turn_id': 'root', 'epoch': 'initial', 'object_goals': copy.deepcopy(self.goals),
            'object_index': 0, 'placement_goal': copy.deepcopy(self.goals[0]),
            'task_object_anchors': {row['object_id']: row['origin_m'] for row in self.snapshot['diagnostics']['scene_directory']['objects'] if row['object_id'] in ('bottle', 'block', 'can')},
            'steps': [{'id': 'can_place', 'title': '放圆柱', 'object_id': 'can', 'checkpoint': 'placement_confirmed', 'skill': 'place'}],
            'stage_bindings': {'can_place': 'certificate'}, 'execution_tools': []}
        self.args = {'observation_id': 'initial:0', 'object_id': 'block'}
        self.harness = TaskHarness.__new__(TaskHarness); self.harness.workspace = self.workspace
        self.harness.observed = {'initial:0': time.monotonic()}; self.harness.sessions = {'root': {'selected_object': 'can'}}
        outer = self
        class ReadOnlyService:
            def snapshot(self): return outer.snapshot
            def call(self, *args, **kwargs): raise AssertionError('rejected transition must never dispatch a native command')
        self.harness.service = ReadOnlyService()

    def test_passive_monitor_does_not_change_physics_or_native_commands(self):
        env = self.env; baseline = ProtectedObjectEnv(images=False)
        try:
            env.reset(seed=0); baseline.reset(seed=0)
            self.assertEqual(env.manifest()['model_evidence'], baseline.manifest()['model_evidence'])
            for key in ('body_mass', 'body_inertia', 'geom_friction', 'actuator_gainprm', 'actuator_biasprm', 'cam_pos', 'cam_quat'):
                np.testing.assert_array_equal(getattr(env.model, key), getattr(baseline.model, key))
            for _ in range(32):
                env.step_tamp(env.last_action.copy()); baseline.step_tamp(baseline.last_action.copy())
                for key in ('qpos', 'qvel', 'ctrl'): np.testing.assert_array_equal(getattr(env.data, key), getattr(baseline.data, key))
            self.assertEqual(env.data.time, baseline.data.time)
            self.assertTrue(all(row['table_supported'] for row in env.object_rest.values()))
            self.assertTrue(all(row['stable_seconds']['table'] >= 1. for row in env.object_rest.values()))
        finally: baseline.close()

    def test_original_targets_anchor_all_objects_without_mutation(self):
        before = copy.deepcopy(self.snapshot)
        self.assertAlmostEqual(self.goals[0]['target_xy_m'][0], .583)
        self.assertEqual(self.goals[1]['target_xy_m'], [.503, -.345]); self.assertEqual(self.snapshot, before)
        self.assertEqual(self.goals[1]['anchor']['object_xy_m'], [.49, .075])

    def test_wrong_object_receipt_cannot_complete_another_objects_stage(self):
        goal = self.goals[0]
        feedback = {'object_id': 'block', 'primitive_success': True, 'verification_source': 'native_support_release_and_stability',
            'placement': {'confirmed': True, 'surface': 'table', 'xy_m': goal['target_xy_m'], 'tolerance_m': goal['tolerance_m']}, 'position_error_m': 0.}
        self.assertFalse(self.workspace._step_done(self.workspace.active_plan['steps'][0], feedback, self.workspace.active_plan))

    def test_model_busy_does_not_mark_an_undispatched_object_stage_running(self):
        work = self.workspace; goal = self.goals[0]
        feedback = {'object_id': 'can', 'primitive_success': True, 'verification_source': 'native_support_release_and_stability',
            'placement': {'confirmed': True, 'surface': 'table', 'xy_m': goal['target_xy_m'], 'tolerance_m': goal['tolerance_m']}, 'position_error_m': 0.}
        work.record_feedback('certificate', 'execute', 'initial', 'fixture', 'place', {'object_id': 'can', 'arm': 'right'}, feedback,
            self.snapshot['memory_context'], 'simulator_oracle', {'after': {'observation_id': 'initial:0'}})
        work.active_plan.update(execution_id='certificate', verdict=None)
        work.active_plan['steps'].append({'id': 'block_pick', 'title': '抓长方体', 'object_id': 'block', 'checkpoint': 'grasp_confirmed', 'skill': 'pick'})
        self.snapshot['last_run'] = {'execution_id': 'certificate', 'status': 'primitive_completed'}
        work.busy = True
        plan = work.plan_status(self.snapshot)
        self.assertEqual(plan['steps'][0]['status'], 'done'); self.assertEqual(plan['steps'][1]['status'], 'pending')

    def test_unreleased_failed_unknown_and_incomplete_goals_cannot_switch(self):
        original = copy.deepcopy(self.snapshot)
        for kind, message in [('unknown', 'next declared'), ('touching', 'released'), ('invalid', 'valid'), ('incomplete', 'success receipt')]:
            self.snapshot = copy.deepcopy(original); args = copy.deepcopy(self.args)
            if kind == 'unknown': args['object_id'] = 'bottle'
            if kind == 'touching': self.snapshot['diagnostics']['evaluation']['touching']['right'] = True
            if kind == 'invalid': self.snapshot['diagnostics']['evaluation']['invalid'] = True
            with self.assertRaisesRegex(ValueError, message): self.harness._advance_object(args, 'execute', {'turn_id': 'root'})

    def test_all_objects_need_current_target_support_and_native_ledger(self):
        plan = self.workspace.active_plan
        for goal in plan['object_goals']: goal['placement_execution_id'] = goal['object_id'] + '-receipt'
        self.snapshot['diagnostics']['task_objects'] = {'goal_ledger': {'root': 'root', 'goals': copy.deepcopy(self.goals)},
            'objects': {goal['object_id']: {'origin_m': goal['target_xy_m'] + [1.], 'touching': {'left': False, 'right': False},
                'stable_seconds': {'table': 1.1}} for goal in self.goals}}
        self.assertTrue(assess_tasks(plan, self.snapshot)['confirmed'])
        saved = copy.deepcopy(self.snapshot)
        self.snapshot['diagnostics']['task_objects']['objects']['can']['stable_seconds']['table'] = .5
        self.assertFalse(assess_tasks(plan, self.snapshot)['confirmed'])
        self.snapshot = copy.deepcopy(saved); self.snapshot['diagnostics']['task_objects']['objects']['can']['origin_m'][0] += .02
        self.assertFalse(assess_tasks(plan, self.snapshot)['confirmed'])
        self.snapshot = copy.deepcopy(saved); self.snapshot['diagnostics']['task_objects']['goal_ledger']['goals'][0]['target_xy_m'][0] += .02
        self.assertFalse(assess_tasks(plan, self.snapshot)['confirmed'])
        self.snapshot = copy.deepcopy(saved); next(row for row in self.snapshot['diagnostics']['scene_directory']['objects'] if row['object_id'] == 'bottle')['origin_m'][0] += .004
        self.assertFalse(assess_tasks(plan, self.snapshot)['confirmed'])

    def test_native_stability_never_promotes_rest_at_the_wrong_original_goal(self):
        env = self.env; env.declare_goals('root', self.goals)
        before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time]
        with self.assertRaisesRegex(ValueError, 'cannot be replaced'): env.declare_goals('root', self.goals)
        for old, new in zip(before[:3], (env.data.qpos, env.data.qvel, env.data.ctrl)): np.testing.assert_array_equal(old, new)
        self.assertEqual(env.data.time, before[3])
        for _ in range(32): env.step_tamp(env.last_action.copy())
        self.assertEqual(env.object_rest['can']['stable_seconds']['table'], 0.)
        self.assertEqual(env.object_rest['block']['stable_seconds']['table'], 0.)


if __name__ == '__main__': unittest.main(verbosity=2)
