"""Audit planner-provider precedence before simplifying the workbench stack."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


class PlannerStackChecks(unittest.TestCase):
    def test_free_transport_composer_matches_legacy_router_without_global_patch(self):
        from harness.free_transport import compose_planner, install_isolated
        from sim import service

        original=service.TampPlanner
        pick=Mock(return_value='free-pick')
        handover=Mock(return_value='free-handover')

        class Base:
            CONTRACT={'planner':'fixture'}
            def __new__(cls,env,task,**kwargs):return f'base:{task}'

        def environment(*,occupied=False, upright=True):
            rotation=np.eye(3)
            rotation[2,2]=1. if upright else .5
            return SimpleNamespace(object_id='bottle',bottle_id=0,
                data=SimpleNamespace(xmat=np.asarray([rotation.reshape(-1)])),
                registry=({'bottle':{}} if not occupied else {'bottle':{},'can':{}}))

        pick_args={'parameters':{'arm':'left','next_handover':{'to_arm':'right'}}}
        handover_args={'parameters':{'from_arm':'left','to_arm':'right'}}
        cases=[(environment(),'pick_bottle',pick_args),
            (environment(),'handover_object',handover_args),
            (environment(occupied=True),'pick_bottle',pick_args),
            (environment(upright=False),'pick_bottle',pick_args)]
        try:
            with patch('sim.free_transport.FreeBottlePick',pick), \
                 patch('sim.free_transport.FreeBottleHandover',handover):
                service.TampPlanner=Base
                install_isolated(object())
                legacy=[service.TampPlanner(env,task,**args) for env,task,args in cases]
                service.TampPlanner=Base
                Router=compose_planner(Base)
                local=[Router(env,task,**args) for env,task,args in cases]
                self.assertEqual(local,legacy)
                self.assertEqual(local,['free-pick','free-handover','base:pick_bottle','base:pick_bottle'])
                self.assertIs(compose_planner(Router),Router)
                self.assertIs(service.TampPlanner,Base)
        finally:
            service.TampPlanner=original

    def test_fast_grasp_provider_is_shadowed_by_later_specific_routes(self):
        from harness import agent
        from harness.connected_placement import install_isolated as connected
        from harness.fast_grasp import install_isolated as fast_grasp
        from harness.free_transport import install_isolated as free_transport
        from harness.margin_grasp import install_isolated as margin_grasp
        from harness.task_placement import install_isolated as task_placement
        from sim import service

        original=service.TampPlanner
        fast=Mock(return_value='fast')
        single=Mock(return_value='single-arm-placement')
        empty=Mock(return_value='empty-scene-relay')
        occupied=Mock(return_value='occupied-scene-relay')

        class Base:
            CONTRACT={'planner':'fixture'}
            def __new__(cls,*args,**kwargs): return 'base'

        base_tools=copy.deepcopy(agent.EXECUTION_TOOLS)
        base_skills=copy.deepcopy(agent.SKILL_TOOLS)
        base_catalogue=agent.catalogue
        try:
            with patch.object(agent,'EXECUTION_TOOLS',copy.deepcopy(base_tools)), \
                 patch.object(agent,'SKILL_TOOLS',copy.deepcopy(base_skills)), \
                 patch.object(agent,'catalogue',base_catalogue), \
                 patch('sim.adaptive_grasp.FastBottlePick',fast), \
                 patch('sim.margin_grasp.MarginBottlePick',occupied), \
                 patch('sim.fast_handover.FastBottleHandover',return_value='fast-relay'), \
                 patch('sim.free_transport.FreeBottlePick',empty), \
                 patch('sim.free_transport.FreeBottleHandover',return_value='empty-relay'), \
                 patch('sim.task_grasp.TaskBottlePick',single), \
                 patch('sim.predictive_place.PredictiveBottlePlacement',return_value='place'):
                def environment(*,occupied_scene):
                    return type('Env',(),{
                        'object_id':'bottle','bottle_id':0,
                        'data':type('Data',(),{'xmat':np.asarray([np.eye(3).reshape(-1)])})(),
                        'registry':({'bottle':{}} if not occupied_scene else {'bottle':{},'can':{}}),
                    })()

                def build(include_fast_grasp):
                    service.TampPlanner=Base
                    agent.EXECUTION_TOOLS=copy.deepcopy(base_tools)
                    agent.SKILL_TOOLS=copy.deepcopy(base_skills)
                    agent.catalogue=base_catalogue
                    for install in (connected,*((fast_grasp,) if include_fast_grasp else ()),
                                    margin_grasp,free_transport):
                        install(object())
                    task_placement(object(),deployed=True)
                    single_arm=service.TampPlanner(environment(occupied_scene=False),'pick_bottle',parameters={
                        'arm':'right','bottle_single_arm_placement':True})
                    empty_scene=service.TampPlanner(environment(occupied_scene=False),'pick_bottle',parameters={
                        'arm':'left','next_handover':{'to_arm':'right'}})
                    occupied_scene=service.TampPlanner(environment(occupied_scene=True),'pick_bottle',parameters={
                        'arm':'left','next_handover':{'to_arm':'right'}})
                    return single_arm,empty_scene,occupied_scene

                before=build(include_fast_grasp=True)
                after=build(include_fast_grasp=False)
                self.assertEqual(before,('single-arm-placement','empty-scene-relay','occupied-scene-relay'))
                self.assertEqual(after,before)
                fast.assert_not_called()
        finally:
            service.TampPlanner=original


if __name__=='__main__': unittest.main()
