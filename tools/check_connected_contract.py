"""New placement routing preserves the old demo, other objects and tool ABI."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from harness import agent
from harness.connected_placement import install_isolated
import sim.service as service


class RoutingChecks(unittest.TestCase):
    def setUp(self):
        self.calls = []
        outer = self
        class Original:
            CONTRACT = {'original': True}
            def __new__(cls, env, task, **kwargs):
                outer.calls.append((env.object_id, task, copy.deepcopy(kwargs)))
                return 'old'
        self.original_tools = copy.deepcopy(agent.SKILL_TOOLS)
        self.original_controls = copy.deepcopy(agent.EXECUTION_TOOLS)
        def catalogue(snapshot, policy_type='pi05'):
            return {'skills': [{'id': t['function']['name'], 'description': t['function']['description'],
                'parameters': copy.deepcopy(t['function']['parameters'])} for t in agent.SKILL_TOOLS]}
        for owner, name, value in ((service, 'TampPlanner', Original), (agent, 'catalogue', catalogue),
            (agent, 'SKILL_TOOLS', copy.deepcopy(self.original_tools)),
            (agent, 'EXECUTION_TOOLS', copy.deepcopy(self.original_controls))):
            handle = patch.object(owner, name, value); handle.start(); self.addCleanup(handle.stop)
        handle = patch('sim.connected_place.ConnectedBottlePlacement', lambda *a, **k: 'connected')
        handle.start(); self.addCleanup(handle.stop)
        marker = object(); self.assertIs(install_isolated(marker), marker)

    def test_old_demo_and_other_shapes_keep_the_original_provider(self):
        for ident, task, params in (
            ('bottle', 'handover_to_basket', {}),
            ('bottle', 'place_bottle', {'receptacle_id': 'basket'}),
            ('block', 'place_bottle', {'receptacle_id': 'table'}),
            ('can', 'place_bottle', {'receptacle_id': 'table'}),
            ('bottle', 'pick_bottle', {'arm': 'right'})):
            self.assertEqual(service.TampPlanner(SimpleNamespace(object_id=ident), task, parameters=params), 'old')
        self.assertEqual(len(self.calls), 5)

    def test_only_bottle_table_placement_uses_connected_provider(self):
        args = {'receptacle_id': 'table', 'target_xy_m': [.495, .185], 'arm': 'left'}
        before = copy.deepcopy(args)
        self.assertEqual(service.TampPlanner(SimpleNamespace(object_id='bottle'), 'place_bottle', parameters=args), 'connected')
        self.assertEqual(args, before); self.assertEqual(self.calls, [])

    def test_tool_parameters_and_other_tool_descriptions_remain_unchanged(self):
        current = agent.catalogue({})['skills']
        for original in self.original_tools:
            f = original['function']; item = next(x for x in current if x['id'] == f['name'])
            self.assertEqual(item['parameters'], f['parameters'])
            if f['name'] != 'place': self.assertEqual(item['description'], f['description'])
        self.assertTrue(agent.catalogue({})['bottle_table_placement']['original_demo_preserved'])

    def test_timing_descriptions_preserve_supported_choices_and_conditional_defaults(self):
        old = next(t['function'] for t in self.original_controls if t['function']['name'] == 'set_plan')
        new = next(t['function'] for t in agent.EXECUTION_TOOLS if t['function']['name'] == 'set_plan')
        old_fields = old['parameters']['properties']; fields = new['parameters']['properties']
        for name in ('time_budget_s', 'timing_mode'):
            self.assertEqual({k: v for k, v in fields[name].items() if k != 'description'},
                             {k: v for k, v in old_fields[name].items() if k != 'description'})
        self.assertIn('when omitted', fields['time_budget_s']['description'])
        self.assertIn('default to120s', fields['time_budget_s']['description'])
        self.assertIn('default to measured_delay', fields['timing_mode']['description'])
        self.assertEqual(new['parameters']['required'], old['parameters']['required'])


if __name__ == '__main__': unittest.main()
