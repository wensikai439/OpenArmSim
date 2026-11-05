"""Dispatch regression only; native task proof lives in immutable experiments."""
import copy
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from . import agent
from .task_placement import compose_planner, install_isolated
import sim.service as service
from sim.bottle_routing import ContinuousHoldAudit


class Routes(TestCase):
    def setUp(self):
        old = service.TampPlanner; tools = agent.SKILL_TOOLS; catalogue = agent.catalogue
        self.addCleanup(setattr, service, 'TampPlanner', old)
        self.addCleanup(setattr, agent, 'SKILL_TOOLS', tools)
        self.addCleanup(setattr, agent, 'catalogue', catalogue)
        class Previous:
            CONTRACT = {'original_demo':True}
            def __new__(cls, env, task, **kwargs):return {'original':task,'parameters':copy.deepcopy(kwargs.get('parameters'))}
        class Pick:
            def __init__(self, env, task, **kwargs):self.env,self.result=env,{'pick':True}
        class Place:
            def __init__(self, env, task, **kwargs):self.result={'preview_servo_contract':{'not_deployed':True}}
        service.TampPlanner=Previous
        self.previous=Previous
        agent.catalogue=lambda snapshot,policy_type='pi05':{'skills':[{'id':x['function']['name']} for x in tools]}
        self.pick = Pick; self.place = Place
        with patch('sim.task_grasp.TaskBottlePick',Pick),patch('sim.predictive_place.PredictiveBottlePlacement',Place):
            self.factory=object();self.assertIs(install_isolated(self.factory,deployed=True),self.factory)
        self.env=SimpleNamespace(object_id='bottle',bottle_id=0,data=SimpleNamespace(xmat=np.eye(3).reshape(1,9)))

    def test_single_arm_table_intent_uses_candidate(self):
        value=service.TampPlanner(self.env,'pick_bottle',parameters={'bottle_single_arm_placement':True})
        self.assertIsInstance(value,self.pick)

    def test_requested_hold_retains_continuous_window_validator(self):
        value=service.TampPlanner(self.env,'pick_bottle',parameters={'bottle_single_arm_placement':True,'hold_seconds':10})
        self.assertIsInstance(value,ContinuousHoldAudit)
        self.assertIsInstance(value.planner,self.pick)

    def test_table_place_declares_actual_deployment(self):
        value=service.TampPlanner(self.env,'place_bottle',parameters={'receptacle_id':'table'})
        self.assertIsInstance(value,self.place)
        self.assertFalse(value.result['preview_servo_contract']['not_deployed'])

    def test_original_demo_relay_and_policy_do_not_change(self):
        for task in ('handover_to_basket','handover_bottle','run_policy'):
            self.assertEqual(service.TampPlanner(self.env,task,parameters={})['original'],task)

    def test_basket_and_other_registered_objects_do_not_change(self):
        self.assertEqual(service.TampPlanner(self.env,'place_bottle',parameters={'receptacle_id':'basket'})['original'],'place_bottle')
        self.env.object_id='block'
        self.assertEqual(service.TampPlanner(self.env,'pick_bottle',parameters={'bottle_single_arm_placement':True})['original'],'pick_bottle')

    def test_catalogue_keeps_tools_and_qualifies_known_geometry(self):
        value=agent.catalogue({})
        self.assertIn('run_policy',[x['id'] for x in value['skills']])
        self.assertIn('handover',[x['id'] for x in value['skills']])
        self.assertFalse(value['bottle_task_placement']['transfer_ready'])
        self.assertIn('known simulator geometry',value['bottle_task_placement']['scope'])

    def test_planner_composer_is_instance_local_and_matches_legacy_routes(self):
        module_default=service.TampPlanner
        with patch('sim.task_grasp.TaskBottlePick',self.pick), \
             patch('sim.predictive_place.PredictiveBottlePlacement',self.place):
            local=compose_planner(self.previous,deployed=True)
            self.assertIs(service.TampPlanner,module_default)
            self.assertIsInstance(local(self.env,'pick_bottle',parameters={
                'bottle_single_arm_placement':True}),self.pick)
            placed=local(self.env,'place_bottle',parameters={'receptacle_id':'table'})
            self.assertIsInstance(placed,self.place)
            self.assertFalse(placed.result['preview_servo_contract']['not_deployed'])
            self.assertEqual(local(self.env,'handover_to_basket',parameters={})['original'],
                self.previous(self.env,'handover_to_basket')['original'])
        self.assertIs(service.TampPlanner,module_default)


if __name__=='__main__':main()
