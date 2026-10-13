"""CPU-only cover/identity guards; no GPU claim or live platform command."""
import copy
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from sim.objects import ObjectEnv
from sim.object_curobo import attachment
from sim.curobo import scene
from tools.object_gpu import GEOMS, validate_attachment
from harness.object_gpu import consolidate_gpu, ObjectGPUWorkspace


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.env = ObjectEnv(images=False); cls.env.reset(seed=0)
    @classmethod
    def tearDownClass(cls): cls.env.close()

    def test_cover_contains_full_box_corners_and_cylinder_volume_under_rotations(self):
        env = self.env
        for ident in ('bottle', 'block', 'can'):
            env.select_object(ident)
            for angle in (0., .43, -.7):
                data = mujoco.MjData(env.model); mujoco.mj_copyData(data, env.model, env.data)
                address = env.registry[ident]['qposadr']
                data.qpos[address + 3:address + 7] = [np.cos(angle / 2), 0., np.sin(angle / 2), 0.]
                mujoco.mj_forward(env.model, data)
                shadow = copy.copy(env); shadow.data = data
                value = attachment(shadow, 'left'); spheres = np.array(value['spheres_tcp'])
                hand = data.xmat[env.ee_ids['left']].reshape(3, 3)
                tcp = data.xpos[env.ee_ids['left']] + hand @ np.array([0, .006, .075])
                self.assertEqual(value['object_id'], ident); self.assertEqual(value['removed_world_geoms'], GEOMS[ident])
                self.assertLessEqual(len(spheres), 32)
                for geom in env.bottle_geoms:
                    size = env.model.geom_size[geom]; kind = env.model.geom_type[geom]
                    half = size[2] if kind == mujoco.mjtGeom.mjGEOM_BOX else size[1]
                    if kind == mujoco.mjtGeom.mjGEOM_BOX:
                        points = np.array([[x, y, z] for x in (-size[0], 0., size[0])
                            for y in (-size[1], 0., size[1]) for z in np.linspace(-half, half, 97)])
                    else:
                        points = np.array([[r * np.cos(t), r * np.sin(t), z] for r in (0., size[0])
                            for t in np.linspace(0, 2 * np.pi, 32) for z in np.linspace(-half, half, 97)])
                    world = data.geom_xpos[geom] + points @ data.geom_xmat[geom].reshape(3, 3).T
                    local = (world - tcp) @ hand
                    gap = np.linalg.norm(local[:, None, :] - spheres[None, :, :3], axis=2) - spheres[None, :, 3]
                    self.assertLessEqual(float(gap.min(axis=1).max()), 1e-12)

    def test_world_keeps_unselected_real_objects_and_removes_only_carried_geoms(self):
        for ident in ('bottle', 'block', 'can'):
            self.env.select_object(ident); world = scene(self.env, carrying=True); value = attachment(self.env, 'right')
            validate_attachment(value, world)
            for other, names in GEOMS.items():
                for name in names: self.assertEqual(name in world['cuboid'], other != ident)

    def test_wrong_identity_duplicate_world_hidden_obstacle_and_unbounded_cover_rejected(self):
        self.env.select_object('block'); value = attachment(self.env, 'left'); world = scene(self.env, carrying=True)
        cases = []
        wrong = copy.deepcopy(value); wrong['object_id'] = 'can'; cases.append((wrong, world))
        wrong = copy.deepcopy(value); wrong['spheres_tcp'][0][3] = .101; cases.append((wrong, world))
        wrong = copy.deepcopy(value); wrong['spheres_tcp'] = wrong['spheres_tcp'] * 3; cases.append((wrong, world))
        wrong = copy.deepcopy(world); wrong['cuboid'].pop('bottle_neck'); cases.append((value, wrong))
        wrong = copy.deepcopy(world); wrong['cuboid']['block_shape'] = wrong['cuboid']['can_shape']; cases.append((value, wrong))
        for attachment_value, scene_value in cases:
            with self.assertRaises(ValueError): validate_attachment(attachment_value, scene_value)


class HarnessContracts(unittest.TestCase):
    def test_empty_and_different_carried_objects_do_not_merge_motion_memories(self):
        rows = []
        for index, (ident, payload) in enumerate([('block', 'single_hold')] * 2 + [('can', 'single_hold')] * 2 + [('block', 'empty')] * 2):
            rows.append({'id': str(index), 'tool': 'move_relative', 'outcome': 'success',
                'arguments': {'arm': 'left', 'planner': 'curobo'}, 'context': 'same-cell',
                'feedback': {'object_id': ident, 'payload_state': payload, 'primitive_success': True,
                    'verification_source': 'robot_forward_kinematics', 'position_error_m': .003},
                'evidence_kind': 'simulator_oracle', 'created': float(index)})
        original = copy.deepcopy(rows); lessons = consolidate_gpu(rows)
        self.assertEqual(rows, original); self.assertEqual(len(lessons), 3)
        self.assertEqual({(r['object_id'], r['payload_state']) for r in lessons},
            {('block', 'single_hold'), ('can', 'single_hold'), ('block', 'empty')})
        self.assertTrue(all(r['repeat_count'] == 2 and r['context'] == 'same-cell' for r in lessons))
        self.assertEqual(len(consolidate_gpu(rows, [lessons[0]['id']])), 2)

    def test_wrong_object_TCP_receipt_cannot_complete_locked_object_motion(self):
        step = {'checkpoint': 'motion_reached', 'skill': 'move_relative'}
        plan = {'placement_goal': {'object_id': 'block'}}
        good = {'object_id': 'block', 'primitive_success': True, 'verification_source': 'robot_forward_kinematics',
            'position_error_m': .003, 'rotation_error_rad': .03}
        self.assertTrue(ObjectGPUWorkspace._step_done(step, good, plan))
        self.assertFalse(ObjectGPUWorkspace._step_done(step, {**good, 'object_id': 'can'}, plan))
        self.assertFalse(ObjectGPUWorkspace._step_done(step, {**good, 'verification_source': 'model'}, plan))


if __name__ == '__main__': unittest.main(verbosity=2)
