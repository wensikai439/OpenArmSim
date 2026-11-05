"""Service synchronization must preserve live tasks and survive UI sessions."""
import unittest
from unittest.mock import patch

from tools import platform_service as platform


class PlatformChecks(unittest.TestCase):
    def test_changed_idle_runtime_restarts_without_a_separate_stop(self):
        with patch.object(platform, 'health_json', return_value={'source_current': False}), \
                patch.object(platform, 'status_json', return_value={'job': None, 'agent': None, 'learning_job': None}), \
                patch.object(platform, 'systemctl') as control:
            platform.synchronize_service(True)
        control.assert_called_once_with('restart', platform.UNIT)

    def test_changed_runtime_never_interrupts_live_owners(self):
        for status in ({'job': {'id': 'motion'}}, {'agent': {'busy': True}},
                       {'learning_job': {'running': True}}, {'decision_wait': True}):
            with self.subTest(status=status), \
                    patch.object(platform, 'health_json', return_value={'source_current': False}), \
                    patch.object(platform, 'status_json', return_value=status), \
                    patch.object(platform, 'systemctl') as control:
                with self.assertRaises(RuntimeError):
                    platform.synchronize_service(True)
                control.assert_not_called()

    def test_current_runtime_is_not_restarted(self):
        with patch.object(platform, 'health_json', return_value={'source_current': True}), \
                patch.object(platform, 'status_json') as snapshot, \
                patch.object(platform, 'systemctl') as control:
            platform.synchronize_service(True)
        snapshot.assert_not_called()
        control.assert_called_once_with('start', platform.UNIT)

    def test_cached_health_cannot_hide_changed_code(self):
        health = {'source_current': True, 'source_revision_sha256': 'loaded-code'}
        with patch.object(platform, 'health_json', return_value=health), \
                patch.object(platform, 'fingerprint', return_value=('changed-code', 267)), \
                patch.object(platform, 'status_json', return_value={}), \
                patch.object(platform, 'systemctl') as control:
            platform.synchronize_service(True)
        control.assert_called_once_with('restart', platform.UNIT)

    def test_cached_health_never_reports_old_code_as_ready(self):
        health = {'state': 'ready', 'backend': 'native-mujoco', 'source_current': True,
                  'source_revision_sha256': 'loaded-code'}
        with patch.object(platform, 'health_json', return_value=health), \
                patch.object(platform, 'fingerprint', return_value=('changed-code', 267)):
            self.assertFalse(platform.ready())

    def test_missing_runtime_starts_without_a_stale_http_probe(self):
        with patch.object(platform, 'health_json') as health, \
                patch.object(platform, 'systemctl') as control:
            platform.synchronize_service(False)
        health.assert_not_called()
        control.assert_called_once_with('start', platform.UNIT)

    def test_observation_timeout_is_not_authorization_to_restart(self):
        with patch.object(platform, 'health_json', side_effect=TimeoutError), \
                patch.object(platform, 'systemctl') as control:
            with self.assertRaises(TimeoutError):
                platform.synchronize_service(True)
        control.assert_not_called()

    def test_broken_status_can_only_reload_after_independent_idle_health(self):
        from urllib.error import HTTPError
        health = {'source_current': False, 'task_active': False, 'agent_busy': False, 'learning_active': False}
        with patch.object(platform, 'health_json', return_value=health), \
                patch.object(platform, 'status_json', side_effect=HTTPError('http://localhost', 500, 'broken', {}, None)), \
                patch.object(platform, 'systemctl') as control:
            platform.synchronize_service(True)
        control.assert_called_once_with('restart', platform.UNIT)

    def test_broken_status_never_infers_idle_from_missing_health_fields(self):
        from urllib.error import HTTPError
        with patch.object(platform, 'health_json', return_value={'source_current': False}), \
                patch.object(platform, 'status_json', side_effect=HTTPError('http://localhost', 500, 'broken', {}, None)), \
                patch.object(platform, 'systemctl') as control:
            with self.assertRaises(RuntimeError):
                platform.synchronize_service(True)
        control.assert_not_called()

    def test_persistent_entry_starts_a_clean_table_and_can_restore_after_login(self):
        rendered = platform.render_unit(platform.SOURCE)
        self.assertIn('Environment=OPENARM_SCENE_OBJECTS=[]', rendered)
        self.assertIn('WantedBy=default.target', rendered)
        self.assertIn('Restart=always', rendered)
        self.assertNotIn('@OPENARMSIM_', rendered)


if __name__ == '__main__':
    unittest.main()
