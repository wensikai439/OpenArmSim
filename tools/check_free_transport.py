"""Real empty-scene regression and preserved occupied/native collision gates."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from sim import bottle_routing
from sim.free_transport import FreeBottlePick, transport_origins
from sim.margin_grasp import MarginBottlePick
from sim.scene_library import LibraryEnv
from sim.tamp import PlanningFailure


class FreeTransportChecks(unittest.TestCase):
    def env(self, selection):
        env = LibraryEnv(selection=selection, images=False)
        self.addCleanup(env.close)
        env.reset(seed=0)
        return env

    def planner(self, env, cls=FreeBottlePick, **kwargs):
        p = cls(env, 'pick_object', parameters={'object_id': 'bottle', 'arm': 'left',
            'next_handover': {'to_arm': 'right'}}, **kwargs)
        p._arm('left')
        p._sync()
        return p

    def test_real_empty_scene_first_grasp_has_connected_native_corridor_without_live_changes(self):
        env = self.env([])
        before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time,
            env.manifest()['model_evidence'])
        p = self.planner(env)
        p._choose_pick_pose(None)
        self.assertEqual(p.result['grasp_search']['candidate_index'], 0)
        self.assertGreaterEqual(p.result['future_transport']['joint_slack_rad'], .04)
        self.assertEqual(p.result['future_transport']['route'], 'empty_base_corridor')
        self.assertEqual(p._obstacles, [])
        self.assertNotIn('empty_scene_transport_reference', p.result)
        for a, b in zip(before[:3], (env.data.qpos, env.data.qvel, env.data.ctrl)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(before[3], env.data.time)
        self.assertEqual(before[4], env.manifest()['model_evidence'])

    def test_occupied_scene_keeps_original_corridors_grasp_and_joint_solutions(self):
        env = self.env(['can', 'block'])
        self.assertEqual([v['side'] for v in transport_origins(env, env.data)],
            [v['side'] for v in bottle_routing.bypass_origins(env, env.data)])
        results = []
        for cls in (MarginBottlePick, FreeBottlePick):
            p = self.planner(env, cls)
            p._choose_pick_pose(None)
            results.append((p.result['grasp_search'], p.result['future_transport']))
            self.assertNotIn('free_transport_proposals', p.result)
        self.assertEqual(results[0], results[1])

    def test_native_collision_results_and_real_table_collisions_stay_enforced(self):
        env = self.env([])
        full, candidate = self.planner(env, MarginBottlePick), self.planner(env)
        rng = np.random.default_rng(51)
        limits = np.asarray([env.model.joint(f'openarm_left_joint{i}').range for i in range(1, 8)])
        table_hits = 0
        for _ in range(80):
            q = rng.uniform(limits[:, 0], limits[:, 1])
            a, b = full._collision(q, True), candidate._collision(q, True)
            self.assertEqual(a, b)
            self.assertEqual(full.last_collision, candidate.last_collision)
            table_hits += any(env.table_id in (c.geom1, c.geom2) for c in candidate.data.contact)
        self.assertGreater(table_hits, 0)

    def test_cancelled_empty_query_restores_state_and_keeps_global_origin_function(self):
        env = self.env([])
        p = self.planner(env, cancel=lambda: True)
        original = bottle_routing.bypass_origins
        saved = (p.carried, p.rotation.copy(), p.phase, copy.deepcopy(p.ik_seeds))
        with self.assertRaises(PlanningFailure) as error:
            p._future_transport(env.data.qpos[p.qids].copy())
        self.assertEqual(error.exception.code, 'cancelled')
        self.assertIs(bottle_routing.bypass_origins, original)
        self.assertEqual(p._obstacles, [])
        self.assertIs(p.carried, saved[0])
        np.testing.assert_array_equal(p.rotation, saved[1])
        self.assertEqual(p.phase, saved[2])


if __name__ == '__main__':
    unittest.main()
