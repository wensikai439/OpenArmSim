"""Meaningful adapter gates: real executor, immutable fixtures and independent scoring."""
import unittest

import numpy as np

from .runtime import (ProgramExecutor, ToolRegistry, SOURCE, OpenArmTrial,
                      evaluate_route, fixture_environment, validate_case)


class AspireChecks(unittest.TestCase):
    def test_official_executor_blocks_imports_private_access_and_unregistered_reset(self):
        import cap_harness.runtime as module
        self.assertTrue(str(module.__file__).startswith(str(SOURCE)))
        registry=ToolRegistry(shared_public_names=('observe',),public_extension_allowlist=())
        registry.register(lambda:{'held_by':[]},name='observe',public=True,layer='shared_atomic',capability='observation')
        executor=ProgramExecutor(registry)
        self.assertTrue(executor.execute_program('result = observe()').ok)
        for code in ('import os','result = observe.__globals__','reset()'):
            self.assertFalse(executor.execute_program(code).ok,code)

    def test_final_basket_pose_or_wrong_direction_cannot_replace_native_route(self):
        final={'inside_basket':True,'supported_by_basket':True,'placement':{'confirmed':True},
               'held':{'left':False,'right':False},'invalid':False}
        rows=[{'tool':'pick','arguments':{'arm':'right'},'ok':True},
              {'tool':'handover','arguments':{'from_arm':'right','to_arm':'left'},'ok':True},
              {'tool':'place','arguments':{'arm':'left','receptacle_id':'basket'},'ok':True}]
        self.assertTrue(evaluate_route(rows,final,program_ok=True)['task_success'])
        self.assertFalse(evaluate_route([],final,program_ok=True)['task_success'])
        rows[1]['arguments']={'from_arm':'left','to_arm':'right'}
        self.assertFalse(evaluate_route(rows,final,program_ok=True)['task_success'])
        rows[1]['arguments']={'from_arm':'right','to_arm':'left'}
        for condition in ('invalid','held'):
            altered={**final,condition:True if condition=='invalid' else {'left':True,'right':False}}
            self.assertFalse(evaluate_route(rows,altered,program_ok=True)['task_success'])
        self.assertFalse(evaluate_route(rows,final,program_ok=False)['task_success'])

    def test_case_is_initial_condition_only_and_does_not_change_robot_or_physics(self):
        from sim.scene_library import LibraryEnv
        case={'seed':52,'bottle_xy_m':[.52,-.19]}
        base=LibraryEnv(images=False,selection=[])
        variant=fixture_environment(LibraryEnv,case)(images=False,selection=[])
        try:
            base.reset(seed=52);variant.reset(seed=52)
            address=variant.registry['bottle']['qposadr']
            np.testing.assert_allclose(variant.data.qpos[address:address+2],case['bottle_xy_m'])
            np.testing.assert_array_equal(base.state(),variant.state())
            np.testing.assert_array_equal(base.model.geom_friction,variant.model.geom_friction)
            np.testing.assert_array_equal(base.model.body_mass,variant.model.body_mass)
            before=variant.data.qpos.copy()
            variant.step(variant.last_action.copy())
            self.assertEqual(variant.tick,1)
            self.assertGreater(variant.data.time,0.)
            self.assertAlmostEqual(variant.data.qpos[address],before[address],places=6)
        finally:
            base.close();variant.close()

    def test_relative_motion_uses_actual_native_delta_contract(self):
        host=object.__new__(OpenArmTrial)
        host.execute=lambda tool,task,args:args
        row=host.move_relative('right',0,0,.03)
        self.assertEqual(row,{'arm':'right','delta_m':[0,0,.03],'frame':'world'})
        with self.assertRaises(ValueError):host.move_relative('right',0,0,.2)
        for case in ({'seed':1,'bottle_xy_m':[float('nan'),-.17]}, {'seed':1,'bottle_xy_m':[.5,.4]}):
            with self.assertRaises(ValueError):validate_case(case)


if __name__=='__main__':unittest.main()
