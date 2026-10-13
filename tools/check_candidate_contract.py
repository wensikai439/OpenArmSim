"""Adversarial candidate backend binding and memory-archive boundaries."""
import copy
import json
import unittest
from unittest.mock import patch

from harness.candidate import CandidateHarness, CandidateWorkspace, install_isolated, validate_candidate_plan
from harness.composite import CompositeHarness

CONTEXT = {'robot': 'v1', 'contract': 'candidate-test', 'policy_contract': 'one'}


class CandidateChecks(unittest.TestCase):
    def test_fixed_demo_direction_events_cannot_validate_general_api_stages(self):
        for checkpoint in ('right_grasp', 'right_lift', 'handover', 'left_grasp', 'left_release_at_basket', 'basket'):
            with self.assertRaises(ValueError):
                validate_candidate_plan({'steps': [{'id': 'one', 'title': '右手独立持瓶', 'checkpoint': checkpoint, 'skill': 'handover'}]})
        values = {'steps': [{'id': 'one', 'title': '右手独立持瓶', 'checkpoint': 'handover_confirmed', 'skill': 'handover'}]}
        self.assertEqual(validate_candidate_plan(values), values['steps'])
        with self.assertRaises(ValueError):
            validate_candidate_plan({'steps': [{'id': 'one', 'title': '完成', 'checkpoint': 'tool_completed'}]})

    def harness(self):
        value = object.__new__(CandidateHarness)
        value.sessions = {'root': {'actions': 0}}
        return value

    def test_accepted_relocation_binds_backend_across_plan_revisions(self):
        harness = self.harness()
        context = {'type': 'llm', 'plan_turn_id': 'root', 'turn_id': 'child'}
        with patch.object(CompositeHarness, '_invoke', return_value={'state': 'started', 'execution_id': 'one'}) as parent:
            result = harness._invoke('relocate', {'transport_planner': 'curobo'}, 'execute', context)
            self.assertEqual(result['bound_transport_planner'], 'curobo')
            harness._invoke('set_plan', {}, 'execute', context)
            for tool, args in [('relocate', {'transport_planner': 'tamp'}),
                ('move_relative', {'planner': 'tamp'}), ('move_to_pose', {})]:
                before = parent.call_count
                with self.assertRaises(ValueError): harness._invoke(tool, args, 'execute', context)
                self.assertEqual(parent.call_count, before)  # zero dispatch
            harness._invoke('relocate', {'transport_planner': 'curobo'}, 'execute', context)
            harness._invoke('pick', {'arm': 'left'}, 'execute', context)
            harness._invoke('finish', {'verdict': 'failure'}, 'execute', context)

    def test_unstarted_or_direct_call_does_not_create_binding(self):
        harness = self.harness()
        context = {'type': 'llm', 'plan_turn_id': 'root'}
        with patch.object(CompositeHarness, '_invoke', return_value={'tool_failed': True}):
            harness._invoke('relocate', {'transport_planner': 'curobo'}, 'execute', context)
        self.assertNotIn('transport_backend', harness.sessions['root'])
        with patch.object(CompositeHarness, '_invoke', return_value={'state': 'started', 'execution_id': 'one'}):
            harness._invoke('relocate', {'transport_planner': 'tamp'}, 'execute')
        self.assertNotIn('transport_backend', harness.sessions['root'])

    def test_new_user_goal_has_independent_backend_selection(self):
        harness = self.harness()
        harness.sessions['root']['transport_backend'] = 'curobo'
        harness.sessions['new'] = {}
        with patch.object(CompositeHarness, '_invoke', return_value={'state': 'started', 'execution_id': 'one'}):
            harness._invoke('relocate', {'transport_planner': 'tamp'}, 'execute',
                {'type': 'llm', 'plan_turn_id': 'new'})
        self.assertEqual(harness.sessions['root']['transport_backend'], 'curobo')
        self.assertEqual(harness.sessions['new']['transport_backend'], 'tamp')

    def test_user_memory_has_pointers_model_index_has_no_raw_pose_archive_is_intact(self):
        workspace = CandidateWorkspace()
        try:
            for index in range(3):
                workspace.record_feedback(str(index), 'execute', 'e', 'duplicate long prompt', 'pick', {'arm': 'left'},
                    {'primitive_success': True, 'verification_source': 'native_two_finger_contact_and_lift'}, CONTEXT,
                    'simulator_oracle', evidence={'run_id': 'native-' + str(index),
                        'before': {'state': [7, 8, 9]}, 'after': {'tcp': [1, 2, 3]}})
            before = copy.deepcopy(workspace.memory('execute', 'e', context=CONTEXT, compact=True))
            public = workspace.public('execute', 'e', CONTEXT)['memory']
            self.assertFalse(public['experiences']); self.assertFalse(public['episodes'])
            self.assertEqual(len(public['knowledge']['lessons']), 1)
            self.assertNotIn('state', json.dumps(public['knowledge']))
            self.assertNotIn('before', json.dumps(public['knowledge']))
            self.assertEqual(public['knowledge']['lessons'][0]['repeat_count'], 3)
            self.assertTrue(public['knowledge']['lessons'][0]['sources'][0]['run_id'])
            self.assertEqual(before, workspace.memory('execute', 'e', context=CONTEXT, compact=True))
            with self.assertRaises(ValueError): workspace.execution_receipt('execute', '0', context=CONTEXT)
            receipt = workspace.execution_receipt('execute', '0', context=CONTEXT, include_oracle=True)
            self.assertEqual(receipt['evidence']['before']['state'], [7, 8, 9])
        finally: workspace.db.close()

    def test_grasp_exhaustion_is_one_scoped_pose_free_lesson(self):
        install_isolated()
        workspace = CandidateWorkspace()
        try:
            for index in range(2):
                workspace.record_feedback(str(index), 'execute', 'e', 'failed pick', 'pick', {'arm': 'left'},
                    {'failure_code': 'grasp_search_exhausted', 'primitive_success': False}, CONTEXT,
                    'simulator_oracle', evidence={'before': {'state': [7, 8, 9]}})
            lessons = workspace.memory('execute', 'e', context=CONTEXT, compact=True)['knowledge']['index']
            self.assertEqual(len(lessons), 1)
            self.assertEqual(lessons[0]['repeat_count'], 2)
            self.assertIn('不证明物理不可达', lessons[0]['summary'])
            self.assertNotIn('state', json.dumps(lessons))
        finally: workspace.db.close()


if __name__ == '__main__': unittest.main(verbosity=2)
