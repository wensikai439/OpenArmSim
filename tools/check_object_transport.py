"""Held-object TCP equations and pre-dispatch identity/hold/drift gates."""
import copy
import time
import unittest

import mujoco
import numpy as np

from harness.object_transport import ObjectTransportHarness, pose_for_object, rotation_from_tcp
from harness.object_memory import ObjectMemoryWorkspace
from sim.scene import SceneDirectory, quaternion
from harness.object_handover import ProtectedObjectEnv


class TransportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.env = ProtectedObjectEnv(images=False)
    @classmethod
    def tearDownClass(cls): cls.env.close()

    def setUp(self):
        env = self.env; env.reset(seed=0); env.select_object('block')
        self.token = 'current:0'; directory = SceneDirectory(env).snapshot(self.token)
        # Hold is an isolated interface fixture, not a simulated successful grasp.
        diag = env.diagnostics(); diag['scene_directory'] = directory
        diag['evaluation']['held'] = {'left': False, 'right': True}
        diag['manipulation_scene']['held_by'] = ['right']
        tcp = env.data.site('right_ee_control_point')
        block = next(row for row in directory['objects'] if row['object_id'] == 'block')
        fixture_tcp = (np.array(block['origin_m']) + [.01, -.06, .09]).tolist()
        self.snapshot = {'observation_id': self.token, 'memory_context': {'contract': 'fixture'},
            'diagnostics': diag, 'observation': {'tcp': {'right': {'position_m': fixture_tcp,
            'quaternion_xyzw': quaternion(tcp.xmat.reshape(3, 3))}}}, 'job': None}
        outer = self
        class ReadOnlyService:
            def snapshot(self): return outer.snapshot
            def call(self, *args, **kwargs): raise AssertionError('resolver must never dispatch a native command')
        self.harness = ObjectTransportHarness.__new__(ObjectTransportHarness)
        self.harness.service = ReadOnlyService(); self.harness.observed = {self.token: time.monotonic()}
        self.harness._log = lambda *args: None; self.harness.workspace = ObjectMemoryWorkspace()
        self.addCleanup(self.harness.workspace.db.close)
        self.goal = {'object_id': 'block', 'surface': 'table', 'target_xy_m': [.58, -.05]}
        self.harness.workspace.active_plan = {'turn_id': 'task', 'epoch': 'current', 'placement_goal': copy.deepcopy(self.goal), 'execution_tools': []}
        self.context = {'turn_id': 'task'}
        self.args = {'observation_id': self.token, 'object_id': 'block', 'arm': 'right', 'planner': 'curobo'}

    def resolve(self): return self.harness._invoke('resolve_placement_transit', self.args, 'execute', self.context)

    def test_rigid_grasp_round_trip_with_tilt_and_offset(self):
        tcp = {'position_m': [.4, -.15, 1.25], 'quaternion_xyzw': quaternion(np.array([[0, 0, 1.], [1., 0, 0], [0, 1., 0]]))}
        q = np.array([.9, .1, .2, .3]); q /= np.linalg.norm(q); r = np.empty(9)
        mujoco.mju_quat2Mat(r, q); obj = {'origin_m': [.5, -.1, 1.15], 'rotation_matrix': r.reshape(3, 3).tolist()}
        goal = np.array([.58, -.05, 1.205]); result = pose_for_object(tcp, obj, goal, -.7)
        relative_position = rotation_from_tcp(tcp).T @ (np.array(obj['origin_m']) - tcp['position_m'])
        relative_rotation = rotation_from_tcp(tcp).T @ obj['rotation_matrix']
        new_tcp = {'position_m': result['position_m'], 'quaternion_xyzw': result['quaternion_xyzw']}
        np.testing.assert_allclose(np.array(result['position_m']) + rotation_from_tcp(new_tcp) @ relative_position, goal, atol=1e-12)
        np.testing.assert_allclose(rotation_from_tcp(new_tcp) @ relative_rotation, result['object_target_rotation_matrix'], atol=1e-12)
        self.assertAlmostEqual(result['object_target_rotation_matrix'][2][2], 1.)

    def test_resolution_is_read_only_and_bound_to_current_original_goal(self):
        env = self.env; before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time]
        result = self.resolve(); self.assertFalse(result['motion_started']); self.assertFalse(result['transfer_ready'])
        self.assertEqual(result['motion_request']['planner'], 'curobo'); self.assertEqual(result['locked_placement_goal'], self.goal)
        self.assertEqual(result['carried_reference']['object_id'], 'block')
        np.testing.assert_allclose(result['object_transport']['object_target_origin_m'][:2], self.goal['target_xy_m'])
        self.assertEqual(self.harness._targets(self.context)[result['target_id']], result)
        for old, new in zip(before[:3], (env.data.qpos, env.data.qvel, env.data.ctrl)): np.testing.assert_array_equal(old, new)
        self.assertEqual(env.data.time, before[3])

    def test_empty_dual_other_arm_and_wrong_identity_are_rejected(self):
        saved = copy.deepcopy(self.snapshot)
        for held in ({'left': False, 'right': False}, {'left': True, 'right': True}, {'left': True, 'right': False}):
            self.snapshot['diagnostics']['evaluation']['held'] = held
            with self.assertRaisesRegex(ValueError, 'alone hold'): self.resolve()
        self.snapshot = saved; self.args['object_id'] = 'can'
        with self.assertRaisesRegex(ValueError, 'identity'): self.resolve()

    def test_stale_epoch_observation_goal_and_running_scene_are_rejected(self):
        self.args['observation_id'] = 'old:0'
        with self.assertRaisesRegex(ValueError, '新观测'): self.resolve()
        self.args['observation_id'] = self.token
        self.harness.workspace.active_plan['epoch'] = 'old'
        with self.assertRaisesRegex(ValueError, 'original table'): self.resolve()
        self.harness.workspace.active_plan['epoch'] = 'current'; self.snapshot['job'] = {'state': 'running'}
        with self.assertRaisesRegex(ValueError, 'idle'): self.resolve()

    def test_changed_hold_object_pose_tcp_and_goal_reject_before_dispatch(self):
        resolved = self.resolve(); original = copy.deepcopy(self.snapshot)
        request = resolved['motion_request']
        cases = [('hold', 'hold changed'), ('object', 'reference drifted'), ('tcp', 'grasp TCP changed'), ('goal', 'goal changed')]
        for kind, message in cases:
            self.snapshot = copy.deepcopy(original); self.harness.workspace.active_plan['placement_goal'] = copy.deepcopy(self.goal)
            if kind == 'hold': self.snapshot['diagnostics']['evaluation']['held']['right'] = False
            if kind == 'object': next(r for r in self.snapshot['diagnostics']['scene_directory']['objects'] if r['object_id'] == 'block')['origin_m'][0] += .004
            if kind == 'tcp': self.snapshot['observation']['tcp']['right']['position_m'][0] += .006
            if kind == 'goal': self.harness.workspace.active_plan['placement_goal']['target_xy_m'][0] += .01
            with self.assertRaisesRegex(ValueError, message): self.harness._invoke('move_to_pose', request, 'execute', self.context)

    def test_lookahead_requires_bound_receipt_same_epoch_object_arm_and_goal(self):
        workspace = self.harness.workspace; workspace.active_plan['execution_tools'] = [{'skill': 'handover', 'id': 'relay'}]
        good = {'primitive_success': True, 'object_id': 'block', 'from_arm': 'left', 'to_arm': 'right', 'held_by': 'right',
            'verification_source': 'native_receiver_contact_then_donor_release', 'future_place': {'surface': 'table', 'target_xy_m': [.58, -.05], 'yaw_offset_rad': -.7}}
        args = {'object_id': 'block', 'from_arm': 'left', 'to_arm': 'right'}
        evidence = {'after': {'observation_id': 'current:1'}}
        def save(value, proof): workspace.record_feedback('relay', 'execute', 'current', 'fixture', 'handover', args, value, self.snapshot['memory_context'], 'simulator_oracle', proof)
        save(good, evidence); self.assertEqual(self.harness._lookahead(self.snapshot, self.goal, 'right'), (-.7, 'relay'))
        save({**good, 'object_id': 'can'}, evidence); self.assertEqual(self.harness._lookahead(self.snapshot, self.goal, 'right'), (0., None))
        save(good, {'after': {'observation_id': 'old:1'}}); self.assertEqual(self.harness._lookahead(self.snapshot, self.goal, 'right'), (0., None))
        save({**good, 'future_place': {**good['future_place'], 'target_xy_m': [.59, -.05]}}, evidence)
        self.assertEqual(self.harness._lookahead(self.snapshot, self.goal, 'right'), (0., None))


if __name__ == '__main__': unittest.main(verbosity=2)
