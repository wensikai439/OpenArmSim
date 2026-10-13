"""Scene geometry, oracle boundaries and resolved-target dispatch guards."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from sim.scene import SceneDirectory, target, bounds
from harness.scene import SceneHarness, install_isolated
from harness.candidate import CandidateHarness
from harness.workspace import Workspace


class SceneChecks(unittest.TestCase):
    def setUp(self):
        xml = '''<mujoco><worldbody>
          <geom name="cell_table_col" type="box" pos="0 0 0" size="1 1 .05"/>
          <body name="target_box" pos=".3 .2 .1"><geom type="box" size=".1 .2 .03"/></body>
          <body name="task_cube" pos=".2 -.1 .15" quat=".7071067811865476 0 0 .7071067811865476"><freejoint/><geom name="cube" type="box" size=".04 .02 .03" mass=".1"/></body>
          <body name="task_ball" pos=".4 -.2 .2"><freejoint/><geom name="ball" type="sphere" size=".025" mass=".1"/></body>
        </worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.env = SimpleNamespace(model=model, data=data)
        self.directory = SceneDirectory(self.env).snapshot('scene:0')
        self.tcp = {'frame': 'world', 'position_m': [0., 0., .4], 'quaternion_xyzw': [0., 0., 0., 1.]}
        self.arguments = {'observation_id': 'scene:0', 'arm': 'left', 'object_id': 'cube',
            'reference_point': 'origin', 'offset_frame': 'object', 'offset_m': [.1, 0., .08], 'planner': 'tamp'}

    def test_new_free_shapes_discovered_without_bottle_grasp_claims(self):
        rows = {row['object_id']: row for row in self.directory['objects']}
        self.assertEqual(set(rows), {'table', 'basket', 'cube', 'ball'})
        self.assertIsNone(rows['cube']['affordances']['pick_provider'])
        self.assertIsNone(rows['ball']['affordances']['pick_provider'])
        self.assertTrue(rows['cube']['dynamic'])
        self.assertEqual(rows['cube']['physical_calibration'], 'unverified')
        self.assertTrue(self.directory['contains_privileged_object_state'])

    def test_rotated_box_bounds_and_sphere_bounds_use_native_geometry(self):
        rows = {row['object_id']: row for row in self.directory['objects']}
        np.testing.assert_allclose(rows['cube']['world_bounds']['min_m'], [.18, -.14, .12], atol=1e-12)
        np.testing.assert_allclose(rows['cube']['world_bounds']['max_m'], [.22, -.06, .18], atol=1e-12)
        np.testing.assert_allclose(rows['ball']['world_bounds']['min_m'], [.375, -.225, .175], atol=1e-12)

    def test_object_axes_rotate_offsets_world_axes_do_not_and_no_live_state_mutates(self):
        q = self.env.data.qpos.copy(); v = self.env.data.qvel.copy()
        local = target(self.directory, self.arguments, self.tcp)
        world = target(self.directory, {**self.arguments, 'offset_frame': 'world'}, self.tcp)
        np.testing.assert_allclose(local['motion_request']['position_m'], [.2, 0., .23], atol=1e-12)
        np.testing.assert_allclose(world['motion_request']['position_m'], [.3, -.1, .23], atol=1e-12)
        self.assertFalse(local['motion_started'])
        np.testing.assert_array_equal(q, self.env.data.qpos); np.testing.assert_array_equal(v, self.env.data.qvel)

    def test_top_center_is_bounds_top_and_unsupported_plane_is_not_invented_box(self):
        result = target(self.directory, {**self.arguments, 'reference_point': 'top_center',
            'offset_frame': 'world', 'offset_m': [0., 0., .1]}, self.tcp)
        np.testing.assert_allclose(result['motion_request']['position_m'], [.2, -.1, .28], atol=1e-12)
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        with self.assertRaises(ValueError): bounds(model, data, 0)
        for extras in ({'object_id': 'unknown'}, {'offset_m': [0., 0., .501]}, {'offset_m': [float('nan'), 0., 0.]},
            {'quaternion_xyzw': [0., 0., 0., 0.]}, {'offset_frame': 'camera'}):
            with self.assertRaises(ValueError): target(self.directory, {**self.arguments, **extras}, self.tcp)

    def harness(self):
        install_isolated()
        value = object.__new__(SceneHarness); value.sessions = {'root': {}}
        snapshot = {'observation_id': 'scene:0', 'diagnostics': {'scene_directory': copy.deepcopy(self.directory)},
            'observation': {'tcp': {'left': copy.deepcopy(self.tcp)}}}
        value.service = SimpleNamespace(snapshot=lambda: copy.deepcopy(snapshot))
        return value, snapshot

    def test_resolved_target_must_match_arm_position_rotation_planner_and_epoch_before_dispatch(self):
        harness, snapshot = self.harness(); context = {'type': 'llm', 'plan_turn_id': 'root', 'turn_id': 'turn'}
        with patch.object(CandidateHarness, '_invoke', return_value={'observation_id': 'scene:0'}) as parent:
            resolved = harness._invoke('resolve_scene_target', self.arguments, 'execute', context)
        accepted = {'state': 'started', 'execution_id': 'native'}
        with patch.object(CandidateHarness, '_invoke', return_value=accepted) as parent:
            motion = resolved['motion_request']
            for changed in ({'arm': 'right'}, {'position_m': [.3, 0., .23]},
                {'quaternion_xyzw': [0., 0., 1., 0.]}, {'planner': 'curobo'}, {'scene_target_id': 'x' * 20}):
                with self.assertRaises(ValueError): harness._invoke('move_to_pose', {**motion, **changed}, 'execute', context)
                parent.assert_not_called()
            snapshot['observation_id'] = 'other:0'
            with self.assertRaises(ValueError): harness._invoke('move_to_pose', motion, 'execute', context)
            parent.assert_not_called()
            snapshot['observation_id'] = 'scene:0'
            harness._invoke('move_to_pose', {**motion, 'quaternion_xyzw': [-x for x in motion['quaternion_xyzw']]}, 'execute', context)
            parent.assert_called_once()
            self.assertNotIn('scene_target_id', parent.call_args.args[1])
            self.assertEqual(parent.call_args.args[3]['scene_target']['target_id'], resolved['target_id'])

    def test_reference_motion_invalidates_old_target_before_parent_dispatch(self):
        harness, snapshot = self.harness(); context = {'type': 'llm', 'plan_turn_id': 'root'}
        with patch.object(CandidateHarness, '_invoke', return_value={'observation_id': 'scene:0'}):
            result = harness._invoke('resolve_scene_target', self.arguments, 'execute', context)
        cube = next(row for row in snapshot['diagnostics']['scene_directory']['objects'] if row['object_id'] == 'cube')
        cube['origin_m'][0] += .0031
        with patch.object(CandidateHarness, '_invoke') as parent:
            with self.assertRaises(ValueError): harness._invoke('move_to_pose', result['motion_request'], 'execute', context)
            parent.assert_not_called()

    def test_pose_resolution_alone_cannot_complete_a_motion_stage(self):
        result = target(self.directory, self.arguments, self.tcp)
        step = {'checkpoint': 'motion_reached', 'skill': 'move_to_pose'}
        self.assertFalse(Workspace._step_done(step, result, {}))

    def test_catalogue_keeps_standard_signature_and_lists_resolved_target_schema(self):
        from harness.scene import catalogue
        install_isolated()
        value = catalogue({'diagnostics': {'scene_directory': self.directory}}, 'act')
        names = {skill['id']: skill for skill in value['skills']}
        self.assertIn('scene_target_id', names['move_to_pose']['parameters']['properties'])
        self.assertFalse(names['resolve_scene_target']['starts_motion'])
        self.assertEqual({row['object_id'] for row in value['declared_assets']}, {'table', 'basket', 'cube', 'ball'})

    def test_progress_keeps_both_receipts_separate_from_durable_memory_and_requires_oracle(self):
        harness, snapshot = self.harness()
        plan = {'turn_id': 'root', 'steps': [
            {'id': 'r', 'checkpoint': 'motion_reached', 'skill': 'move_to_pose', 'status': 'done'},
            {'id': 'l', 'checkpoint': 'motion_reached', 'skill': 'move_to_pose', 'status': 'done'},
            {'id': 'review', 'checkpoint': 'visual_review', 'status': 'pending'}],
            'stage_bindings': {'r': 'one', 'l': 'two'},
            'execution_tools': [{'id': 'one', 'skill': 'move_to_pose'}, {'id': 'two', 'skill': 'move_to_pose'}]}
        reads = []
        def receipt(role, ident, **kwargs):
            reads.append((ident, kwargs))
            return {'feedback': {'primitive_success': True, 'verification_source': 'robot_forward_kinematics',
                'position_error_m': .001, 'raw_pose': [7, 8, 9]}, 'evidence': {'before': [7, 8, 9]}}
        harness.workspace = SimpleNamespace(plan_status=lambda state: copy.deepcopy(plan), execution_receipt=receipt)
        harness.active_session = 'root'; harness.privileged_epoch = None
        safe = harness._task_progress(snapshot)
        self.assertTrue(safe['all_planned_robot_stages_done']); self.assertFalse(safe['receipts']); self.assertFalse(reads)
        harness.privileged_epoch = 'scene'
        full = harness._task_progress(snapshot)
        self.assertEqual([r['execution_id'] for r in full['receipts']], ['one', 'two'])
        self.assertTrue(full['contains_privileged_object_state']); self.assertFalse(full['pending_robot_stages'])
        self.assertNotIn('raw_pose', str(full)); self.assertNotIn('before', str(full))
        self.assertIsNone(harness._task_progress(snapshot, {'plan_turn_id': 'new-root'}))

    def test_completed_scene_stages_reject_repeat_without_dispatch_failed_stage_can_recover(self):
        harness, snapshot = self.harness(); context = {'type': 'llm', 'plan_turn_id': 'root'}
        with patch.object(CandidateHarness, '_invoke', return_value={'observation_id': 'scene:0'}):
            resolved = harness._invoke('resolve_scene_target', self.arguments, 'execute', context)
        plan = {'turn_id': 'root', 'steps': [{'id': 'one', 'skill': 'move_to_pose',
            'checkpoint': 'motion_reached', 'status': 'done'}]}
        harness.workspace = SimpleNamespace(plan_status=lambda state: copy.deepcopy(plan))
        harness.privileged_epoch = None
        with patch.object(CandidateHarness, '_invoke', return_value={'state': 'started', 'execution_id': 'native'}) as parent:
            with self.assertRaises(ValueError): harness._invoke('move_to_pose', resolved['motion_request'], 'execute', context)
            parent.assert_not_called()
            plan['steps'][0]['status'] = 'blocked'
            harness._invoke('move_to_pose', resolved['motion_request'], 'execute', context)
            parent.assert_called_once()

    def test_pending_execution_without_terminal_receipt_does_not_invent_success(self):
        harness, snapshot = self.harness(); harness.active_session = 'root'; harness.privileged_epoch = 'scene'
        plan = {'turn_id': 'root', 'steps': [{'id': 'one', 'skill': 'move_to_pose', 'status': 'running'}],
            'execution_tools': [{'id': 'pending', 'skill': 'move_to_pose'}]}
        def missing(*args, **kwargs): raise ValueError('terminal receipt is not yet recorded')
        harness.workspace = SimpleNamespace(plan_status=lambda state: plan, execution_receipt=missing)
        progress = harness._task_progress(snapshot)
        self.assertFalse(progress['receipts']); self.assertFalse(progress['all_planned_robot_stages_done'])
        self.assertEqual(progress['pending_robot_stages'], ['one'])


class SceneScoreChecks(unittest.TestCase):
    def fixture(self):
        from tools.check_scene import CASES
        resolved = []; motions = []; receipts = []
        for index, case in enumerate(CASES):
            ident = str(index) * 20
            arguments = {'arm': case['arm'], 'position_m': [.3, .2, .4], 'quaternion_xyzw': [0., 0., 0., 1.],
                'planner': case['planner'], 'scene_target_id': ident}
            resolved.append({'target_id': ident, 'reference': {'object_id': case['object_id']},
                'reference_point': case['reference_point'], 'offset_frame': case['offset_frame'],
                'offset_m': case['offset_m'], 'motion_request': copy.deepcopy(arguments)})
            motions.append({'name': 'move_to_pose', 'kind': 'tool', 'status': 'completed',
                'detail': {'arguments': copy.deepcopy(arguments), 'result': {'execution_id': 'native' + str(index)}}})
            feedback = {'primitive_success': True, 'verification_source': 'robot_forward_kinematics',
                'position_error_m': .001, 'rotation_error_rad': .01}
            if case['planner'] == 'curobo': feedback.update(fallback=False, native_path_revalidation=True,
                planner='official cuRobo V2 GPU IK/trajectory optimization', plans=[{'success': True}])
            receipts.append({'execution': {'id': 'native' + str(index)}, 'feedback': feedback})
        turns = [{'status': 'completed', 'trace': copy.deepcopy(motions)}]
        plan = {'steps': [{'status': 'done'}, {'status': 'done'}], 'timing_mode': 'measured_delay',
            'time_budget_s': 120, 'verdict': 'success'}
        return receipts, turns, plan, resolved, motions

    def test_extra_unused_read_only_proposal_is_not_an_extra_action(self):
        from tools.check_scene import score_api
        values = self.fixture(); proposal = copy.deepcopy(values[3][-1]); proposal['target_id'] = 'x' * 20
        values[3].append(proposal)
        self.assertTrue(score_api(*values)['passed'])

    def test_binary_float_serialization_uses_dispatch_precision_not_exact_list_equality(self):
        from tools.check_scene import score_api
        values = self.fixture()
        values[3][0]['motion_request']['position_m'][1] = -.23299999999999998
        values[4][0]['detail']['arguments']['position_m'][1] = -.233
        self.assertTrue(score_api(*values)['passed'])
        values[4][0]['detail']['arguments']['position_m'][1] += 1.1e-9
        self.assertFalse(score_api(*values)['passed'])

    def test_changed_target_backend_receipt_or_missing_native_gpu_evidence_is_rejected(self):
        from tools.check_scene import score_api
        for change in ('object', 'offset', 'arm', 'pose', 'planner', 'receipt', 'gpu', 'position_error'):
            values = self.fixture(); receipts, turns, plan, resolved, motions = values
            if change == 'object': resolved[0]['reference']['object_id'] = 'basket'
            elif change == 'offset': resolved[0]['offset_m'] = [0, 0, 0]
            elif change == 'arm': motions[0]['detail']['arguments']['arm'] = 'left'
            elif change == 'pose': motions[0]['detail']['arguments']['position_m'] = [0, 0, 0]
            elif change == 'planner': motions[1]['detail']['arguments']['planner'] = 'tamp'
            elif change == 'receipt': receipts[1]['execution']['id'] = 'old'
            elif change == 'gpu': receipts[1]['feedback']['plans'] = []
            elif change == 'position_error': receipts[0]['feedback']['position_error_m'] = .013
            self.assertFalse(score_api(*values)['passed'], change)

    def test_extra_manipulation_unfinished_stage_or_same_used_target_does_not_pass(self):
        from tools.check_scene import score_api
        for change in ('pick', 'stage', 'target'):
            values = self.fixture(); receipts, turns, plan, resolved, motions = values
            if change == 'pick': turns[0]['trace'].append({'kind': 'tool', 'name': 'pick', 'status': 'completed'})
            elif change == 'stage': plan['steps'][1]['status'] = 'pending'
            elif change == 'target': motions[1]['detail']['arguments']['scene_target_id'] = resolved[0]['target_id']
            self.assertFalse(score_api(*values)['passed'], change)


if __name__ == '__main__': unittest.main(verbosity=2)
