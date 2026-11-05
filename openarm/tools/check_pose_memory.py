"""Pose applicability, bounded default index and unmodified raw evidence."""
import copy
import json
import unittest

from harness.poses import PoseWorkspace, consolidate, install_isolated

CONTEXT = {'robot': 'v1', 'contract': 'same', 'policy_contract': 'same'}


class PoseMemoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.check_task_queue import setup
        setup(); install_isolated()

    @staticmethod
    def rows():
        rows = []
        for i in range(6):
            native = i != 5
            feedback = {'object_id': 'block', 'primitive_success': True,
                'verification_source': 'native_two_finger_contact_and_lift'}
            if i >= 2:
                feedback['grasp_start'] = {'orientation_class': 'non_upright',
                    'private_axis': [.7, .7, 0.]}
            rows.append({'id': str(i), 'tool': 'pick', 'outcome': 'success' if native else 'unverified',
                'arguments': {'object_id': 'block', 'arm': 'left'}, 'feedback': feedback,
                'context': 'same-cell', 'created': float(i),
                'evidence_kind': 'simulator_oracle' if native else 'model_hypothesis',
                'evidence': {'run_id': 'raw-' + str(i), 'camera': '/runs/raw/head.jpg'}})
        return rows

    def test_native_orientation_split_is_deduplicated_without_mutating_receipts(self):
        rows = self.rows(); before = copy.deepcopy(rows); lessons = consolidate(rows)
        self.assertEqual(before, rows); self.assertEqual(len(lessons), 2)
        self.assertEqual({x['repeat_count'] for x in lessons}, {2, 3})
        self.assertEqual(len({x['id'] for x in lessons}), 2)
        oriented = next(x for x in lessons if x['conditions'].get('start_orientation'))
        self.assertEqual(oriented['conditions']['start_orientation'], 'non_upright')
        self.assertEqual(oriented['context'], 'same-cell')
        self.assertIn('倒卧/倾斜抓取', oriented['title'])

    def test_compact_index_and_detail_share_scope_and_forgetting_id(self):
        work = PoseWorkspace(); self.addCleanup(work.db.close)
        for row in self.rows():
            work.record_feedback(row['id'], 'execute', 'e', 'fixture', row['tool'], row['arguments'],
                row['feedback'], CONTEXT, row['evidence_kind'], row['evidence'])
        work.index_object_id = 'block'
        compact = work.memory('execute', 'e', context=CONTEXT, compact=True)
        index = compact['knowledge']['index']; self.assertEqual(len(index), 2)
        self.assertEqual(compact['experiences'], []); self.assertEqual(compact['episodes'], [])
        self.assertLessEqual(compact['retrieval']['index_bytes'], 8192)
        serialized = json.dumps(index); self.assertNotIn('private_axis', serialized)
        self.assertNotIn('/runs/', serialized); self.assertNotIn('raw-', serialized)
        oriented = next(x for x in index if x['conditions'].get('start_orientation'))
        with self.assertRaises(ValueError):
            work.knowledge_detail('execute', 'e', oriented['id'], context=CONTEXT)
        detail = work.knowledge_detail('execute', 'e', oriented['id'], context=CONTEXT, include_oracle=True)
        self.assertEqual(detail['conditions'], oriented['conditions'])
        work.delete_memory('execute', oriented['id'])
        after = work.memory('execute', 'e', context=CONTEXT, compact=True)['knowledge']['index']
        self.assertEqual(len(after), 1); self.assertNotEqual(after[0]['id'], oriented['id'])
        self.assertEqual(work.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0], 6)

    def test_search_failure_is_one_key_lesson_not_every_rejected_proposal(self):
        rows = self.rows()[2:5]
        for row in rows:
            row['outcome'] = 'failure'; row['feedback'].update(primitive_success=False,
                failure_code='grasp_search_exhausted', grasp_search={'rejected': [{'index': i} for i in range(180)]})
        lessons = consolidate(rows); self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]['repeat_count'], 3)
        self.assertIn('不证明物理不可达', lessons[0]['summary'])


if __name__ == '__main__': unittest.main()
