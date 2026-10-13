"""Check physical inclusion and original robot gates, not visual-only toggles."""
import copy
import unittest
import threading
from types import SimpleNamespace

import mujoco
import numpy as np

from sim.scene_library import LibraryEnv, ITEMS, validate_selection, commit_scene
from harness.object_handover import ProtectedObjectEnv
from sim.oriented_grasp import initial_conflicts
from sim.curobo import scene


class SceneChecks(unittest.TestCase):
    def test_commit_handles_initial_owner_and_rolls_back_failed_publish(self):
        old = SimpleNamespace(close=lambda: None)
        candidate = object()
        owner = SimpleNamespace(env=old, epoch='old', keyframes=['keep'], decision_receipt=None,
            frame_history={'old:0': {}}, cache={'observation_id': 'old:0'}, frames={},
            applied_view_revision=0, lock=threading.Lock())
        def broken_publish():
            self.assertIs(owner.scene_transition['cache']['observation_id'], owner.cache['observation_id'])
            owner.cache = {'observation_id': 'must-not-leak'}
            raise ValueError('injected publish failure')
        owner._publish = broken_publish
        with self.assertRaises(ValueError): commit_scene(owner, candidate)
        self.assertIs(owner.env, old); self.assertEqual(owner.epoch, 'old')
        self.assertEqual(owner.cache, {'observation_id': 'old:0'})
        self.assertEqual(owner.keyframes, ['keep']); self.assertEqual(owner.frame_history, {'old:0': {}})
        self.assertIsNone(owner.scene_transition)
        owner._publish = lambda: None
        commit_scene(owner, candidate)
        self.assertIs(owner.env, candidate); self.assertNotEqual(owner.epoch, 'old')
        self.assertIsNone(owner.last_run)
    def test_selection_is_bounded_and_deterministic(self):
        self.assertEqual(validate_selection(['block', 'can']), ['can', 'block'])
        for invalid in (None, 'can', ['can', 'can'], [True], ['../../secret'], ['bottle']):
            with self.assertRaises(ValueError): validate_selection(invalid)

    def test_no_optional_objects_have_no_ghost_bodies(self):
        env = LibraryEnv(selection=[], images=False)
        try:
            env.reset(seed=0)
            self.assertEqual(set(env.registry), {'bottle'})
            self.assertFalse(initial_conflicts(env))
            for item in ITEMS:
                self.assertEqual(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'task_'+item['object_id']), -1)
        finally: env.close()

    def test_default_robot_and_physics_remain_original(self):
        old = ProtectedObjectEnv(images=False); env = LibraryEnv(images=False)
        try:
            old.reset(seed=0); env.reset(seed=0)
            self.assertEqual(old.manifest()['model_evidence']['compiled'], env.manifest()['model_evidence']['compiled'])
            for name in ('actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'body_mass', 'geom_friction'):
                np.testing.assert_array_equal(getattr(old.model, name), getattr(env.model, name))
            np.testing.assert_array_equal(old.data.qpos, env.data.qpos)
            for _ in range(12):
                action=old.last_action.copy();old.step_tamp(action);env.step_tamp(action)
                for name in ('qpos','qvel','ctrl'):
                    np.testing.assert_array_equal(getattr(old.data,name),getattr(env.data,name))
                self.assertEqual(old.data.time,env.data.time)
            self.assertEqual(set(env.diagnostics()['task_objects']['objects']),set(env.registry))
        finally: old.close(); env.close()

    def test_optional_scene_keeps_native_goal_ledger_and_new_scene_starts_without_old_goals(self):
        env=LibraryEnv(images=False);empty=LibraryEnv(selection=[],images=False)
        try:
            env.reset(seed=0);empty.reset(seed=0)
            before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),env.data.time)
            goals=[{'object_id':'can','surface':'table','target_xy_m':[.583,-.345],'tolerance_m':.01},
                {'object_id':'block','surface':'table','target_xy_m':[.503,-.345],'tolerance_m':.01}]
            env.declare_goals('original-task',goals)
            for a,b in zip(before[:3],(env.data.qpos,env.data.qvel,env.data.ctrl)):np.testing.assert_array_equal(a,b)
            self.assertEqual(before[3],env.data.time)
            self.assertEqual(env.diagnostics()['task_objects']['goal_ledger']['goals'],goals)
            with self.assertRaises(ValueError):env.declare_goals('original-task',goals)
            self.assertIsNone(empty.diagnostics()['task_objects']['goal_ledger'])
            self.assertEqual(set(empty.registry),{'bottle'})
        finally:env.close();empty.close()

    def test_all_items_have_native_collision_and_conservative_GPU_world(self):
        ids = [item['object_id'] for item in ITEMS]; frozen = copy.deepcopy(ITEMS)
        env = LibraryEnv(selection=ids, images=False)
        try:
            env.reset(seed=0); self.assertFalse(initial_conflicts(env))
            self.assertEqual(set(env.registry), {'bottle', *ids})
            world = scene(env)['cuboid']
            for ident in ids:
                row = env.registry[ident]
                self.assertGreater(env.model.body_mass[row['body']], 0)
                for g in row['geoms']:
                    self.assertTrue(env.model.geom_contype[g])
                    self.assertIn(env.model.geom(g).name, world)
            self.assertEqual(len(env.registry['cup']['geoms']), 16)
            for _ in range(30):
                _, _, terminated, _, _ = env.step_tamp(env.last_action.copy())
                self.assertFalse(terminated)
            self.assertTrue(env.evaluation['non_target_object_guard']['passed'])
            self.assertFalse(env._parameter_audit.summary()['changed_fields'])
            self.assertEqual(ITEMS, frozen)
        finally: env.close()


if __name__ == '__main__': unittest.main()
