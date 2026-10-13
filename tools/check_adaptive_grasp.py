"""Native pose/Jacobian/collision equivalence and immutable search checks."""
import copy
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from harness.object_handover import ProtectedObjectEnv
from sim.adaptive_grasp import AdaptiveBottlePick, FastBottlePick, RefinedBottlePick, roll_alternatives
from sim.bottle_routing import BottlePickPlanner
from sim.grasp import candidates, cylinder_bands
from sim.tamp import PlanningFailure


class NativeChecks(unittest.TestCase):
    def setUp(self):
        self.env = ProtectedObjectEnv(images=False); self.addCleanup(self.env.close)
        self.env.reset(seed=0)
        self.before = (self.env.data.qpos.copy(), self.env.data.qvel.copy(),
                       self.env.data.ctrl.copy(), self.env.data.time, self.env.manifest()['model_evidence'])

    def check_live(self):
        for a, b in zip(self.before[:3], (self.env.data.qpos, self.env.data.qvel, self.env.data.ctrl)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(self.env.data.time, self.before[3])
        self.assertEqual(self.env.manifest()['model_evidence'], self.before[4])

    def planners(self, arm):
        parameters = {'object_id': 'bottle', 'arm': arm}
        a = BottlePickPlanner(self.env, 'pick_object', parameters=parameters)
        b = FastBottlePick(self.env, 'pick_object', parameters=parameters)
        for planner in (a, b): planner._arm(arm); planner._sync()
        return a, b

    def test_minimal_pipeline_matches_full_pose_jacobian_and_carried_geometry(self):
        rng = np.random.default_rng(12)
        for side in ('left', 'right'):
            full, fast = self.planners(side)
            limits = np.asarray([self.env.model.joint(f'openarm_{side}_joint{i}').range for i in range(1, 8)])
            for carrying in (False, True):
                for planner in (full, fast):
                    planner._sync()
                    if carrying:
                        hand = planner.data.xmat[planner.body].reshape(3, 3)
                        planner.carried = (hand.T @ (planner.data.xpos[self.env.bottle_id] - planner.data.xpos[planner.body]),
                                           hand.T @ planner.data.xmat[self.env.bottle_id].reshape(3, 3))
                    else: planner.carried = None
                for _ in range(12):
                    q = rng.uniform(limits[:, 0], limits[:, 1])
                    full._at(q); fast._kinematics_only = True; fast._at(q)
                    for name in ('xpos', 'xmat', 'xipos', 'geom_xpos', 'geom_xmat', 'cdof'):
                        np.testing.assert_allclose(getattr(full.data, name), getattr(fast.data, name), atol=2e-13, rtol=0)
                    jac = []
                    for planner in (full, fast):
                        jp, jr = np.zeros((3, self.env.model.nv)), np.zeros((3, self.env.model.nv))
                        point, _ = planner._pose()
                        mujoco.mj_jac(self.env.model, planner.data, jp, jr, point, planner.body)
                        jac.append((jp, jr))
                    np.testing.assert_allclose(jac[0], jac[1], atol=2e-13, rtol=0)
                    # Collision forces a FULL fresh native query even when the
                    # previous scratch query left stale contact data behind.
                    self.assertEqual(full._collision(q, True), fast._collision(q, True))
                    self.assertEqual(full.last_collision, fast.last_collision)
                    self.assertEqual(full.data.ncon, fast.data.ncon)
        self.check_live()

    def test_jacobian_agrees_with_independent_finite_difference(self):
        for side in ('left', 'right'):
            _, planner = self.planners(side); planner._kinematics_only = True
            q = self.env.data.qpos[planner.qids].copy(); planner._at(q)
            point, _ = planner._pose(); jp = np.zeros((3, self.env.model.nv))
            mujoco.mj_jac(self.env.model, planner.data, jp, None, point, planner.body)
            numerical = np.empty((3, 7))
            for i in range(7):
                plus, minus = q.copy(), q.copy(); plus[i] += 1e-6; minus[i] -= 1e-6
                planner._at(plus); p, _ = planner._pose(); p = p.copy()
                planner._at(minus); m, _ = planner._pose()
                numerical[:, i] = (p - m) / 2e-6
            np.testing.assert_allclose(jp[:, planner.dids], numerical, atol=2e-9, rtol=0)
        self.check_live()

    def test_same_original_solver_gives_same_acceptance_and_rejection(self):
        timings = {'full': [], 'minimal': []}
        for side in ('left', 'right'):
            for unreachable in (False, True):
                planners = self.planners(side); output = []
                point, rotation = planners[0]._pose(); rotation = rotation.copy()
                target = point + ([0, 0, .008] if not unreachable else [2., 0, 0])
                for label, planner in zip(timings, planners):
                    planner.rotation = rotation.copy(); start = time.monotonic()
                    try: output.append(('solution', planner._ik(target)))
                    except PlanningFailure as exc: output.append((exc.code, copy.deepcopy(planner.result.get('planning_diagnostics'))))
                    timings[label].append(time.monotonic() - start)
                self.assertEqual(output[0][0], output[1][0])
                if output[0][0] == 'solution': np.testing.assert_allclose(output[0][1], output[1][1], atol=2e-11, rtol=0)
                else: self.assertEqual(output[0][1], output[1][1])
        print('native IK timing', json.dumps(timings), flush=True)
        self.check_live()

    def test_roll_adds_proper_rotations_without_changing_originals_or_approach(self):
        planner = AdaptiveBottlePick(self.env, 'pick_object', parameters={'arm': 'left'})
        planner._arm('left'); planner._sync()
        original = list(candidates(cylinder_bands(self.env.model, planner.data, self.env.bottle_id)))
        values = roll_alternatives(original); self.assertEqual(len(values), 2 * len(original))
        for a, b, rolled in zip(original, values[:len(original)], values[len(original):]):
            np.testing.assert_array_equal(a['rotation'], b['rotation'])
            for key in ('grasp_m', 'pregrasp_m'):
                np.testing.assert_array_equal(a[key], rolled[key])
            np.testing.assert_array_equal(a['rotation'][:, 2], rolled['rotation'][:, 2])
            np.testing.assert_allclose(rolled['rotation'].T @ rolled['rotation'], np.eye(3), atol=1e-14)
            self.assertAlmostEqual(float(np.linalg.det(rolled['rotation'])), 1.)
        self.check_live()

    def test_accepted_query_clears_prior_failure_and_reports_actual_joint_slack(self):
        _, planner = self.planners('left')
        point, rotation = planner._pose(); point = point.copy(); rotation = rotation.copy()
        planner.rotation = rotation
        with self.assertRaises(PlanningFailure): planner._ik(point + [2., 0, 0])
        self.assertIn('planning_diagnostics', planner.result)
        q = planner._ik(point + [0, 0, .008])
        self.assertNotIn('planning_diagnostics', planner.result)
        latest = planner.result['private_search_progress']['last_accepted_ik']
        limits = np.asarray([self.env.model.joint(f'openarm_left_joint{i}').range for i in range(1, 8)])
        slack = np.minimum(q-limits[:, 0], limits[:, 1]-q)
        self.assertEqual(latest['minimum_joint_slack_rad'], float(slack.min()))
        self.assertEqual(latest['limiting_joint'], f'openarm_left_joint{np.argmin(slack)+1}')
        np.testing.assert_array_equal(latest['q_rad'], q)
        self.check_live()

    def test_interruption_preserves_private_search_progress_without_live_changes(self):
        planner = AdaptiveBottlePick(self.env, 'pick_object', parameters={'arm': 'left'}, cancel=lambda: True)
        planner._arm('left')
        with self.assertRaises(PlanningFailure) as error: planner._choose_pick_pose(None)
        self.assertEqual(error.exception.code, 'cancelled')
        self.assertEqual(planner.result['private_search_progress']['candidate_index'], 0)
        self.assertEqual(planner.result['private_search_progress']['total_candidates'], 576)
        self.assertIn('grasp_search', planner.result); self.assertIsNone(planner.planning_qpos)
        self.check_live()

    def test_refinement_uses_native_band_dimensions_and_has_no_recursive_expansion(self):
        planner = RefinedBottlePick(self.env, 'pick_object', parameters={'arm': 'left'})
        planner._arm('left'); planner._sync()
        band = cylinder_bands(self.env.model, planner.data, self.env.bottle_id)[-1]
        value = roll_alternatives(list(candidates([band])))[0]
        q = self.env.data.qpos[planner.qids].copy()
        planner.result['private_search_progress'] = {}
        rows = planner._refinements(value, 'next_place_unreachable', q)
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertLessEqual(abs(row['axial_shift_m']), .4 * band['half_height_m'])
            np.testing.assert_allclose(row['rotation'].T @ row['rotation'], np.eye(3), atol=1e-14)
            np.testing.assert_allclose(row['pregrasp_m'], row['grasp_m'] - .10 * row['rotation'][:, 2], atol=1e-14)
            self.assertEqual(planner._refinements(row, 'next_place_unreachable', q), [])
        self.assertEqual(planner._refinements(value, 'ik_unreachable', None), [])
        self.assertNotIn('primitive_success', planner.result)
        self.check_live()


    def test_empty_optional_scene_uses_real_table_and_restores_scratch_obstacles_on_failure(self):
        from sim.scene_library import LibraryEnv
        env = LibraryEnv(selection=[], images=False)
        try:
            env.reset(seed=0)
            planner = FastBottlePick(env, 'pick_object', parameters={'arm': 'left', 'next_handover': {'to_arm': 'right'}})
            obstacles = planner._obstacles; self.assertEqual(obstacles, [])
            before = env.data.qpos.copy()
            def fail(owner, q):
                self.assertEqual(owner._obstacles, [env.model.geom('cell_table_col').id])
                raise PlanningFailure('probe_failure', 'only scratch proposal failed')
            with patch.object(BottlePickPlanner, '_future_transport', fail):
                with self.assertRaises(PlanningFailure): planner._future_transport(np.zeros(7))
            self.assertIs(planner._obstacles, obstacles)
            np.testing.assert_array_equal(env.data.qpos, before)
            self.assertEqual(env.data.time, 0.)
        finally: env.close()


class RoutingChecks(unittest.TestCase):
    def test_only_original_goal_conditioned_bottle_provider_is_replaced(self):
        import sim.service as service
        from harness.fast_grasp import install_isolated
        class Old:
            CONTRACT = {'original': True}
            def __new__(cls, env, task, **kwargs): return ('old', task, copy.deepcopy(kwargs))
        env = SimpleNamespace(object_id='bottle', bottle_id=0, data=SimpleNamespace(xmat=np.eye(3).reshape(1, 9)))
        with patch.object(service, 'TampPlanner', Old), patch('sim.adaptive_grasp.FastBottlePick', return_value='fast'), \
                patch('sim.oriented_grasp.HoldAudit', side_effect=lambda p, s: ('hold', p, s)):
            marker = object(); self.assertIs(install_isolated(marker), marker)
            for params in ({'arm': 'right', 'bottle_single_arm_placement': True},
                           {'arm': 'left', 'next_handover': {'to_arm': 'right'}}):
                before = copy.deepcopy(params)
                self.assertEqual(service.TampPlanner(env, 'pick_bottle', parameters=params), 'fast')
                self.assertEqual(params, before)
            self.assertEqual(service.TampPlanner(env, 'pick_bottle', parameters={
                'arm': 'left', 'next_handover': {'to_arm': 'right'}, 'hold_seconds': 10}), ('hold', 'fast', 10))
            for task, params in (('handover_to_basket', {}), ('pick_bottle', {}), ('pick_bottle', {'arm': 'right'}),
                                 ('place_bottle', {'receptacle_id': 'table'})):
                self.assertEqual(service.TampPlanner(env, task, parameters=params)[0], 'old')
            env.object_id = 'block'
            self.assertEqual(service.TampPlanner(env, 'pick_bottle', parameters={'bottle_single_arm_placement': True})[0], 'old')


if __name__ == '__main__': unittest.main()
