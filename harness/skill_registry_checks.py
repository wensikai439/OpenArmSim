"""Runtime skill registry agrees with the exact schemas offered to the model."""
import copy
import unittest

from .skill_registry import reconcile_catalogue


class SkillRegistryChecks(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {'type': 'function', 'function': {'name': 'pick', 'description': 'pick the selected object',
                'parameters': {'type': 'object', 'properties': {
                    'object_id': {'type': 'string', 'enum': ['bottle', 'block']},
                    'arm': {'type': 'string', 'enum': ['left', 'right']}}, 'required': ['object_id', 'arm']}}},
            {'type': 'function', 'function': {'name': 'move_to_pose', 'description': 'move TCP',
                'parameters': {'type': 'object', 'properties': {'position_m': {'type': 'array'},
                    'frame': {'type': 'string', 'enum': ['world']}}, 'required': ['position_m', 'frame']}}},
            {'type': 'function', 'function': {'name': 'run_policy', 'description': 'run a policy',
                'parameters': {'type': 'object', 'properties': {'backend': {'type': 'string', 'enum': ['act']}}}}},
        ]
        self.catalogue = {'adapters': [
            {'id': 'tamp', 'ready': True}, {'id': 'curobo', 'ready': True, 'connection_ready': True},
            {'id': 'act', 'ready': True, 'connection_ready': False, 'performance_verified': False}],
            'skills': [{'id': 'pick', 'kind': 'registered_object', 'backends': ['tamp'],
                'parameters': {'stale': True}}, {'id': 'removed_stale_skill', 'backends': ['tamp']}],
        }
        self.snapshot = {'policy_status': {'connection_ready': False}}

    def test_registry_schema_scope_and_catalogue_are_from_live_allowlist(self):
        value = reconcile_catalogue(self.catalogue, self.tools, self.snapshot)
        skills = {item['id']: item for item in value['skills']}
        self.assertEqual(set(skills), {'pick', 'move_to_pose', 'run_policy'})
        self.assertEqual(skills['pick']['parameters'], self.tools[0]['function']['parameters'])
        self.assertEqual(skills['pick']['scope']['object_ids'], ['bottle', 'block'])
        self.assertEqual(skills['pick']['scope']['arms'], ['left', 'right'])
        self.assertTrue(skills['pick']['starts_motion'])
        self.assertEqual(skills['move_to_pose']['scope']['coordinate_frame'], 'world')
        self.assertTrue(skills['move_to_pose']['available'])
        self.assertFalse(skills['run_policy']['available'])
        self.assertEqual(value['skill_registry']['source'], 'live_model_tool_allowlist')
        self.assertEqual(value['skill_registry']['unavailable'], ['run_policy'])
        self.assertEqual(self.catalogue['skills'][0]['parameters'], {'stale': True})

    def test_runtime_connection_is_not_reported_as_task_performance(self):
        value = reconcile_catalogue(self.catalogue, self.tools, self.snapshot)
        policy = next(item for item in value['skills'] if item['id'] == 'run_policy')
        self.assertFalse(policy['providers'][0]['connection_ready'])
        self.assertFalse(policy['providers'][0]['performance_verified'])
        adapters = {item['id']: item for item in value['adapters']}
        self.assertFalse(adapters['act']['ready'])
        self.assertTrue(adapters['act']['configured_ready'])
        self.assertFalse(adapters['act']['connection_ready'])
        self.assertFalse(adapters['act']['performance_verified'])

    def test_policy_provider_scope_comes_from_its_live_tool_schema(self):
        tools = copy.deepcopy(self.tools)
        tools[-1]['function']['parameters']['properties']['backend']['enum'] = ['act']
        value = reconcile_catalogue(self.catalogue, tools, self.snapshot)
        policy = next(item for item in value['skills'] if item['id'] == 'run_policy')
        self.assertEqual(policy['backends'], ['act'])
        self.assertEqual([provider['id'] for provider in policy['providers']], ['act'])

    def test_active_act_endpoint_does_not_make_pi05_look_connected(self):
        catalogue = copy.deepcopy(self.catalogue)
        catalogue['adapters'][2]['connection_ready'] = True
        catalogue['adapters'][2]['ready'] = True
        tools = copy.deepcopy(self.tools)
        tools[-1]['function']['parameters']['properties']['backend']['enum'] = ['pi05', 'act']
        self.snapshot = {'policy_type': 'act', 'policy_status': {'connection_ready': True}}
        value = reconcile_catalogue(catalogue, tools, self.snapshot)
        policy = next(item for item in value['skills'] if item['id'] == 'run_policy')
        providers = {item['id']: item for item in policy['providers']}
        self.assertTrue(providers['act']['connection_ready'])
        self.assertFalse(providers['pi05']['connection_ready'])
        self.assertFalse(providers['pi05']['callable'])


if __name__ == '__main__':
    unittest.main()
