"""Optional inventory guards and native-bound request integrity, CPU only."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from sim.scene_library import LibraryEnv, ITEMS
from sim.curobo import scene
from sim.object_curobo import attachment
from sim import scene_curobo as native
from sim.tamp import PlanningFailure
from tools.scene_gpu import validate_inventory
from tools.object_gpu import validate_attachment


class SceneChecks(unittest.TestCase):
    def environment(self, selected):
        env = LibraryEnv(selection=selected, images=False); env.reset(seed=0)
        self.addCleanup(env.close); return env

    def test_empty_one_two_and_all_optional_objects_keep_actual_nonselected_geometry(self):
        for selected in ([], ['can'], ['can', 'block'], [x['object_id'] for x in ITEMS]):
            env = self.environment(selected); before = env.data.qpos.copy()
            registry = native.inventory(env); world = scene(env, carrying=True); cover = attachment(env, 'right')
            spheres = validate_inventory(registry, cover, world)
            self.assertEqual(len(spheres), len(cover['spheres_tcp']))
            for ident, names in registry.items():
                for name in names: self.assertEqual(name in world['cuboid'], ident != 'bottle')
            validate_inventory(registry, None, scene(env))
            np.testing.assert_array_equal(before, env.data.qpos); self.assertEqual(env.data.time, 0.)

    def test_active_obstacle_omission_wrong_inventory_duplicate_payload_and_fixed_support_rejected(self):
        env = self.environment(['can']); registry = native.inventory(env)
        world = scene(env, carrying=True); cover = attachment(env, 'right'); cases = []
        wrong = copy.deepcopy(world); wrong['cuboid'].pop('can_shape'); cases.append((registry, cover, wrong))
        wrong = copy.deepcopy(registry); wrong.pop('can'); cases.append((wrong, cover, world))
        wrong = copy.deepcopy(world); wrong['cuboid'].pop('cell_table_col'); cases.append((registry, cover, wrong))
        wrong = copy.deepcopy(cover); wrong['object_id'] = 'block'; cases.append((registry, wrong, world))
        wrong = copy.deepcopy(registry); wrong['bottle'] = ['bottle_body']; cases.append((wrong, cover, world))
        wrong = copy.deepcopy(world); wrong['cuboid']['bottle_body'] = wrong['cuboid']['can_shape']; cases.append((registry, cover, wrong))
        for inventory, carried, geometry in cases:
            with self.assertRaises(ValueError): validate_inventory(inventory, carried, geometry)

    def test_legacy_three_object_contract_is_retained_and_empty_failure_is_reproduced(self):
        env = self.environment(['can', 'block']); world = scene(env, carrying=True); cover = attachment(env, 'right')
        np.testing.assert_array_equal(validate_attachment(cover, world), validate_inventory(native.inventory(env), cover, world))
        env = self.environment([]); world = scene(env, carrying=True); cover = attachment(env, 'right')
        with self.assertRaises(ValueError): validate_attachment(cover, world)
        validate_inventory(native.inventory(env), cover, world)

    def test_requests_bind_owner_registry_without_mutating_input_and_require_acknowledgment(self):
        env = self.environment([]); token = native.CURRENT_ENV.set(env)
        payload = {'scene': scene(env, carrying=True), 'attachment': attachment(env, 'right')}; before = copy.deepcopy(payload)
        try:
            def response(path, data, timeout):
                self.assertEqual(data['native_scene_inventory'], native.inventory(env))
                return {'success': True, 'native_scene_inventory': copy.deepcopy(data['native_scene_inventory'])}
            with patch.object(native, 'BASE_CALL', response): native.call('/plan', payload)
            self.assertEqual(payload, before)
            with patch.object(native, 'BASE_CALL', return_value={'success': True}):
                with self.assertRaises(PlanningFailure) as error: native.call('/plan', payload)
                self.assertEqual(error.exception.code, 'scene_inventory_unacknowledged')
        finally: native.CURRENT_ENV.reset(token)
        self.assertIsNone(native.CURRENT_ENV.get())


if __name__ == '__main__': unittest.main()
