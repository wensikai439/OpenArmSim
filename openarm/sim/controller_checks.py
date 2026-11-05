"""Compare numerical command state to audited source without hardware imports."""
import ast
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
from .controller import SourceMIT,ROOT
from .environment import OpenArmEnv


def audited_stub():
    tree=ast.parse((ROOT/'reference/source/inputs/controller.py').read_text())
    klass=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='OpenArmController')
    method=next(node for node in klass.body if isinstance(node,ast.FunctionDef) and node.name=='set_position')
    # Execute ONLY the reviewed arithmetic method with local motor stubs.
    # No imports, constructors, CAN libraries, network or physical devices.
    namespace={'np':np,'oa':SimpleNamespace(MITParam=lambda *args:tuple(args))}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<audited set_position>','exec'),namespace)
    class Arm:
        def __init__(self):self.commands=None;self.gripper=SimpleNamespace(mit_control_all=lambda commands:None)
        def get_arm(self):return self
        def get_gripper(self):return self.gripper
        def mit_control_all(self,commands):self.commands=commands
        def recv_all(self):pass
    obj=SimpleNamespace(left_arm=Arm(),right_arm=Arm(),kp=np.array([60,50,50,50,30,30,30]),
        kv=np.array([6,6,6,6,1.5,1.5,1.5]),acc_error_limit=10.,
        left_last_error=np.zeros(7),right_last_error=np.zeros(7),
        left_acc_error=np.zeros(7),right_acc_error=np.zeros(7))
    return obj,namespace['set_position']


class ControllerChecks(unittest.TestCase):
    def test_emitted_dq_tau_equal_original_source_for_both_arms_and_saturation(self):
        expected,method=audited_stub();candidate=SourceMIT();rng=np.random.default_rng(3);saturated=False
        for i in range(100):
            side='left' if i%3 else 'right';arm=getattr(expected,side+'_arm')
            target=rng.uniform(-3,3,7);measured=target-.4 if i<80 else target+rng.normal(0,.2,7)
            method(expected,arm,target,.05,measured,-.5)
            result=candidate.update(side,target,measured)
            emitted=np.asarray(arm.commands)
            np.testing.assert_array_equal(emitted[:,2],result['position_rad'])
            np.testing.assert_array_equal(emitted[:,3],result['velocity_input'])
            np.testing.assert_array_equal(emitted[:,4],result['feedforward_input'])
            saturated |= bool(np.any(candidate.accumulated[side]==10.))
        self.assertTrue(saturated)

    def test_native_total_force_law_is_limited_and_no_extra_applied_torque(self):
        env=OpenArmEnv(images=False,profile={'controller_mode':'source_mit_candidate','gravity_compensation':'none'})
        try:
            env.reset(seed=0);q=env.data.qpos.copy();v=env.data.qvel.copy();action=env.last_action.copy()
            action[:7]+=.30;action[8:15]-=.30
            for _ in range(100):env._set_command(action)
            mujoco.mj_forward(env.model,env.data)
            np.testing.assert_array_equal(q,env.data.qpos);np.testing.assert_array_equal(v,env.data.qvel)
            for side,start in [('left',0),('right',8)]:
                f=env.controller_feedback[side];aids=env.aids[side];kp=env.model.actuator_gainprm[aids,0];kd=-env.model.actuator_biasprm[aids,2]
                expected=kp*(action[start:start+7]-env.data.qpos[env.qids[side]])+kd*(f['velocity_input']-env.data.qvel[env.dids[side]])+f['feedforward_input']
                expected=np.clip(expected,*env.model.actuator_forcerange[aids].T)
                np.testing.assert_allclose(env.data.actuator_force[aids],expected,atol=1e-10)
                self.assertTrue(np.any(abs(expected)==env.model.actuator_forcerange[aids,1]))
            np.testing.assert_array_equal(env.data.qfrc_applied,np.zeros(env.model.nv))
        finally:env.close()

    def test_command_clock_preserved_across_task_window_but_cleared_on_reset(self):
        env=OpenArmEnv(images=False,profile={'controller_mode':'source_mit_candidate','gravity_compensation':'none'})
        try:
            env.reset(seed=0);self.assertEqual(env.source_controller.calls['left'],0)
            env.step_tamp(env.last_action.copy());self.assertEqual(env.contacts.samples,33)
            self.assertEqual(env.source_controller.calls['left'],1)
            saved=env.source_controller.accumulated['left'].copy();env.prepare_task()
            np.testing.assert_array_equal(env.source_controller.accumulated['left'],saved)
            self.assertEqual(env.source_controller.calls['left'],1)
            env.reset(seed=0);self.assertEqual(env.source_controller.calls['left'],0)
            np.testing.assert_array_equal(env.source_controller.accumulated['left'],np.zeros(7))
        finally:env.close()

    def test_modes_are_explicit_and_do_not_modify_default_profile(self):
        raw=(ROOT/'sim/profile.json').read_bytes()
        with self.assertRaises(ValueError):OpenArmEnv(images=False,profile={'controller_mode':'source_mit_candidate'})
        env=OpenArmEnv(images=False)
        try:self.assertIsNone(env.source_controller)
        finally:env.close()
        self.assertEqual((ROOT/'sim/profile.json').read_bytes(),raw)

if __name__=='__main__':unittest.main()
