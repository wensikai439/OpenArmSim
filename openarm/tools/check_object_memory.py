"""Decision-index boundaries, applicability, forgetting and receipt binding."""
import copy
import json
import unittest

from harness.object_memory import ObjectMemoryWorkspace, consolidate, index_entry, install_isolated

CONTEXT = {'robot': 'v1', 'contract': 'same', 'policy_contract': 'same'}


class MemoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): install_isolated()

    @staticmethod
    def rows():
        return [{'id': str(i), 'tool': 'move_relative', 'outcome': 'success', 'arguments': {'arm': 'left', 'planner': 'curobo'},
            'feedback': {'object_id': obj, 'payload_state': payload, 'primitive_success': True,
                'verification_source': 'robot_forward_kinematics', 'position_error_m': .003, 'private_pose': [.1, .2, 1.]},
            'evidence_kind': 'simulator_oracle', 'context': 'same', 'created': float(i),
            'evidence': {'run_id': 'raw-' + str(i), 'camera_keyframes': [{'sensor_urls': {'head': '/runs/raw/head.jpg'}}]}}
            for i, (obj, payload) in enumerate([('block', 'single_hold')] * 2 + [('can', 'single_hold')] * 2 + [('block', 'empty')] * 2)]

    def test_scope_survives_index_without_raw_coordinates_or_camera_paths(self):
        rows = self.rows(); original = copy.deepcopy(rows); lessons = consolidate(rows)
        self.assertEqual(rows, original); self.assertEqual(len(lessons), 3)
        index = [index_entry(item) for item in lessons]
        self.assertEqual({(item['object_id'], item['payload_state']) for item in index},
            {('block', 'single_hold'), ('can', 'single_hold'), ('block', 'empty')})
        self.assertTrue(all(item['conditions']['arm'] == 'left' for item in index))
        self.assertTrue(all(item['protocol_id'] == 'object_motion' for item in index))
        text = json.dumps(index); self.assertNotIn('private_pose', text); self.assertNotIn('/runs/', text)
        self.assertNotIn('sources', text); self.assertTrue(all(len(item['summary']) < 100 for item in index))

    def test_selected_object_filter_runs_before_top_k_and_preserves_archive(self):
        work = ObjectMemoryWorkspace(); self.addCleanup(work.db.close)
        rows = self.rows()[:2]
        codes = ['goal_collision', 'ik_unreachable', 'grasp_lost', 'dual_hold_not_confirmed', 'receiver_grasp_lost',
            'planning_state_changed', 'placement_unreachable', 'placement_not_confirmed', 'orientation_tracking_error',
            'tracking_error', 'release_pose_not_reached', 'retreat_not_clear']
        for i, code in enumerate(codes):
            rows.append({'id': 'can-' + str(i), 'tool': 'pick', 'arguments': {'object_id': 'can', 'arm': 'left'},
                'feedback': {'object_id': 'can', 'failure_code': code}, 'evidence_kind': 'simulator_oracle', 'evidence': {}})
        for row in rows:
            work.record_feedback(row['id'], 'execute', 'e', 'fixture', row['tool'], row['arguments'],
                row['feedback'], CONTEXT, row['evidence_kind'], row['evidence'])
        work.index_object_id = 'block'; memory = work.memory('execute', 'e', context=CONTEXT, compact=True)
        self.assertEqual(len(memory['knowledge']['index']), 1)
        self.assertEqual(memory['knowledge']['index'][0]['object_id'], 'block')
        self.assertEqual(memory['knowledge']['index'][0]['repeat_count'], 2)
        self.assertLessEqual(memory['retrieval']['index_bytes'], 8192)
        self.assertEqual(work.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0], len(rows))
        public = work.public('execute', 'e', CONTEXT)
        self.assertTrue(any(item['object_id'] == 'can' for item in public['memory']['knowledge']['lessons']))
        work.delete_memory('execute', memory['knowledge']['index'][0]['id'])
        self.assertEqual(work.memory('execute', 'e', context=CONTEXT, compact=True)['knowledge']['index'], [])

    def test_archive_is_separate_and_forgetting_cannot_resurrect_a_method(self):
        work = ObjectMemoryWorkspace(); self.addCleanup(work.db.close)
        for item in self.rows():
            work.record_feedback(item['id'], 'execute', 'e', 'move', item['tool'], item['arguments'],
                item['feedback'], CONTEXT, item['evidence_kind'], item['evidence'])
        before = work.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0]
        memory = work.memory('execute', 'e', context=CONTEXT, compact=True)
        self.assertEqual(len(memory['knowledge']['index']), 3); self.assertEqual(memory['experiences'], [])
        self.assertEqual(set(memory['knowledge']['protocols']), {'object_motion'})
        ident = memory['knowledge']['index'][0]['id']; work.delete_memory('execute', ident)
        self.assertEqual(len(work.memory('execute', 'e', context=CONTEXT, compact=True)['knowledge']['index']), 2)
        self.assertEqual(work.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0], before)
        with self.assertRaises(ValueError): work.execution_receipt('execute', '0', context=CONTEXT)
        self.assertEqual(work.execution_receipt('execute', '0', context=CONTEXT, include_oracle=True)['feedback']['private_pose'], [.1, .2, 1.])

    def test_original_goal_is_preserved_in_failure_guidance(self):
        row = self.rows()[0]; row.update(outcome='failure')
        row['feedback'] = {'object_id': 'block', 'failure_code': 'goal_collision'}
        lesson = consolidate([row])[0]
        self.assertIn('保留锁定的原目标', lesson['summary']); self.assertNotIn('调整目标', lesson['summary'])

    def test_critical_native_displacement_is_one_lesson_and_raw_feedback_stays_intact(self):
        rows = self.rows()[:2]
        for item in rows:
            item.update(tool='handover', outcome='failure')
            item['arguments'] = {'object_id': 'can', 'from_arm': 'right', 'to_arm': 'left'}
            item['feedback'] = {'failure_code': 'invalid', 'message': 'non-target object displaced: bottle'}
        original = copy.deepcopy(rows); lessons = consolidate(rows)
        self.assertEqual(rows, original); self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]['failure_code'], 'non_target_object_displaced')
        self.assertEqual(lessons[0]['repeat_count'], 2); self.assertIn('停止本任务', lessons[0]['summary'])
        rows[0]['evidence_kind'] = 'model_hypothesis'; rows[0]['outcome'] = 'unverified'
        self.assertEqual(consolidate(rows)[0]['repeat_count'], 1)

    def test_different_grasp_intents_have_distinct_applicability_and_titles(self):
        rows = self.rows()[:4]
        for i, item in enumerate(rows):
            item.update(tool='pick'); item['arguments'] = {'object_id': 'can', 'arm': 'right'}
            item['feedback'] = {'object_id': 'can', 'primitive_success': True, 'verification_source': 'native_two_finger_contact_and_lift'}
            if i > 1: item['arguments']['next_handover'] = {'to_arm': 'left'}
        lessons = consolidate(rows); self.assertEqual(len(lessons), 2)
        self.assertEqual(len({item['title'] for item in lessons}), 2)
        self.assertEqual({item['repeat_count'] for item in lessons}, {2})
        self.assertEqual(sum('next_handover' in item['conditions'] for item in lessons), 1)

    def test_directed_receipt_cannot_complete_a_different_object_or_direction(self):
        work = ObjectMemoryWorkspace(); self.addCleanup(work.db.close)
        args = {'object_id': 'block', 'from_arm': 'left', 'to_arm': 'right'}
        good = {'object_id': 'block', 'from_arm': 'left', 'to_arm': 'right', 'held_by': 'right',
            'primitive_success': True, 'verification_source': 'native_receiver_contact_then_donor_release'}
        step = {'checkpoint': 'handover_confirmed', 'skill': 'handover'}; plan = {'placement_goal': {'object_id': 'block'}}
        self.assertTrue(work._step_done(step, good, plan))
        for key, value in [('object_id', 'can'), ('to_arm', 'left'), ('held_by', 'left')]:
            wrong = {**good, key: value}; saved = copy.deepcopy(wrong)
            work.record_feedback(key, 'execute', 'e', 'transfer', 'handover', args, wrong, CONTEXT, 'simulator_oracle')
            row = work.db.execute('SELECT feedback,outcome FROM experiences WHERE id=?', (key,)).fetchone()
            stored = json.loads(row['feedback']); self.assertFalse(stored['execution_binding']['valid'])
            self.assertEqual(row['outcome'], 'failure'); self.assertFalse(work._step_done(step, stored, plan)); self.assertEqual(wrong, saved)


if __name__ == '__main__': unittest.main(verbosity=2)
