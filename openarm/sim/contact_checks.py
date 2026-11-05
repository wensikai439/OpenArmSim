"""Temporal contact evidence: real transient collision and continuous dwell."""
import unittest
import mujoco
import numpy as np

from .contacts import Contacts
from .environment import OpenArmEnv

TRANSIENT = '''<mujoco><option timestep="0.001" gravity="0 0 0" integrator="implicitfast"/>
<worldbody><geom name="floor" type="plane" size="1 1 .1"/>
<body name="openarm_right_test" pos="0 0 .03"><freejoint/>
<geom name="ball" type="sphere" size=".01" mass=".01" solref=".002 .05" solimp=".95 .99 .001"/>
</body></worldbody></mujoco>'''


class ContactChecks(unittest.TestCase):
    def test_real_transient_collision_survives_endpoint_disappearance(self):
        model = mujoco.MjModel.from_xml_string(TRANSIENT)
        data = mujoco.MjData(model)
        data.qvel[2] = -4.
        monitor = Contacts(model)
        control = mujoco.MjData(model)
        control.qvel[2] = -4.
        for _ in range(33):
            t = float(data.time)
            mujoco.mj_step(model, data)
            monitor.sample(monitor.read(data), .001, t)
            mujoco.mj_step(model, control)
        np.testing.assert_array_equal(data.qpos, control.qpos)
        np.testing.assert_array_equal(data.qvel, control.qvel)
        mujoco.mj_forward(model, data)
        final = monitor.read(data)
        monitor.sample(final, 0., data.time)
        summary = monitor.summary()
        self.assertEqual(summary["endpoint_robot_external_force_N"], 0.)
        self.assertGreater(summary["robot_external_impulse_Ns"], .04)
        self.assertGreater(summary["peak_normal_force_N"], 6.)
        self.assertAlmostEqual(summary["robot_external_seconds"], .013)
        self.assertEqual(summary["samples"], 33)
        self.assertFalse(monitor.deep_penetration)

    def test_blinking_hold_and_support_cannot_accumulate_continuous_dwell(self):
        model = mujoco.MjModel.from_xml_string(TRANSIENT)
        monitor = Contacts(model)
        c = monitor.read(mujoco.MjData(model))
        c["held"]["right"] = True
        for _ in range(90): monitor.sample(c, .001, 0., stable=True)
        c["held"]["right"] = False
        monitor.sample(c, .001, .09, stable=False)
        c["held"]["right"] = True
        for _ in range(20): monitor.sample(c, .001, .091, stable=True)
        self.assertAlmostEqual(monitor.hold_seconds["right"], .02)
        self.assertAlmostEqual(monitor.stable_seconds, .02)
        monitor.begin(.111)
        self.assertAlmostEqual(monitor.hold_seconds["right"], .02)
        monitor.reset()
        self.assertEqual(monitor.hold_seconds["right"], 0.)

    def test_deep_penetration_requires_unbroken_physics_samples(self):
        model = mujoco.MjModel.from_xml_string(TRANSIENT)
        monitor = Contacts(model)
        c = monitor.read(mujoco.MjData(model))
        c["penetration_m"] = .012
        for _ in range(90): monitor.sample(c, .001, 0.)
        c["penetration_m"] = 0.
        monitor.sample(c, .001, .09)
        c["penetration_m"] = .012
        for _ in range(20): monitor.sample(c, .001, .091)
        self.assertFalse(monitor.deep_penetration)
        for _ in range(80): monitor.sample(c, .001, .111)
        self.assertTrue(monitor.deep_penetration)

    def test_control_windows_cover_all_33_34_33_physics_steps(self):
        env = OpenArmEnv(images=False)
        try:
            env.reset(seed=0)
            counts = []
            for frame in range(3):
                before = env.data.time
                env.step(env.reference_action(0, frame))
                window = env.evaluation["contact_window"]
                counts.append(window["samples"])
                self.assertAlmostEqual(window["duration_s"], env.data.time-before)
            self.assertEqual(counts, [33, 34, 33])
        finally:
            env.close()
