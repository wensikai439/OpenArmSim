"""Archive facts must survive presentation without being upgraded or rebound."""
import unittest
from harness.records import archive_overview


class ArchiveChecks(unittest.TestCase):
    def test_native_skill_is_not_task_success(self):
        result = archive_overview('run', {}, {'status': 'primitive_completed'})
        self.assertEqual(result['status'], 'primitive_completed')
        self.assertNotIn('success', result)

    def test_terminal_summary_wins(self):
        result = archive_overview('run', {'status': 'running'}, {'status': 'error'})
        self.assertEqual(result['status'], 'error')

    def test_running_without_summary_stays_running(self):
        self.assertEqual(archive_overview('run', {'status': 'running'}, {})['status'], 'running')

    def test_unknown_outcome_is_preserved(self):
        self.assertEqual(archive_overview('run', {}, {'status': 'unknown'})['status'], 'unknown')

    def test_wrong_archive_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            archive_overview('one', {'run_id': 'two'}, {})

    def test_bad_measurements_are_not_zero(self):
        for value in (None, True, float('nan'), float('inf'), -1):
            result = archive_overview('run', {'started_at_unix': value}, {'wall_seconds': value})
            self.assertIsNone(result['wall_seconds'])
            self.assertIsNone(result['started_at'])

    def test_public_summary_does_not_publish_full_manifest(self):
        result = archive_overview('run', {'hashes': {'private': 'abc'}, 'details': {
            'tamp': {'task': 'move_relative', 'parameters': {'arm': 'right', 'private': 'abc'}}}}, {})
        self.assertEqual(result['parameters'], {'arm': 'right'})
        self.assertNotIn('hashes', result)

    def test_old_optional_metadata_cannot_break_the_catalogue(self):
        for details in (None, [], {'tamp': []}, {'tamp': {'parameters': None}, 'initiator': []}):
            self.assertEqual(archive_overview('run', {'details': details}, {})['parameters'], {})


if __name__ == '__main__':
    unittest.main()
