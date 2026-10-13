"""Operator views must not alter robot state or model-visible camera pixels."""
import unittest
import time
import numpy as np

from .environment import OpenArmEnv
from .service import SimulationService
from .view import FRONT,validate_view


class ViewChecks(unittest.TestCase):
    def test_operator_render_preserves_sensor_pixels_and_physics(self):
        env=OpenArmEnv(images=False)
        try:
            env.reset(seed=0)
            before={c:env.render(c) for c in ('head','left','right')}
            qpos,qvel,campos,camquat=env.data.qpos.copy(),env.data.qvel.copy(),env.model.cam_pos.copy(),env.model.cam_quat.copy()
            front=env.render('operator',view=FRONT)
            free=env.render('operator',view={**FRONT,'mode':'free','azimuth':130.,'elevation':-35.,'distance':1.7})
            self.assertGreater(np.mean(abs(front.astype(float)-free)),1.)
            for c in before:np.testing.assert_array_equal(before[c],env.render(c))
            np.testing.assert_array_equal(qpos,env.data.qpos);np.testing.assert_array_equal(qvel,env.data.qvel)
            np.testing.assert_array_equal(campos,env.model.cam_pos);np.testing.assert_array_equal(camquat,env.model.cam_quat)
            self.assertEqual(env.tick,0);self.assertEqual(env.data.time,0.)
        finally:env.close()

    def test_live_view_updates_do_not_invalidate_observation_or_sensor_cache(self):
        service=SimulationService()
        try:
            before=service.snapshot();sensor={c:service.frame(c) for c in ('head','left','right')}
            old=service.frame('operator')
            accepted=service.set_view({**FRONT,'mode':'free','azimuth':135.})
            deadline=time.monotonic()+3
            while service.snapshot()['viewer']['revision']<accepted['revision'] and time.monotonic()<deadline:time.sleep(.02)
            after=service.snapshot()
            self.assertEqual(after['viewer']['revision'],accepted['revision'])
            self.assertEqual(after['observation_id'],before['observation_id'])
            self.assertEqual(after['observation'],before['observation'])
            self.assertNotEqual(old,service.frame('operator'))
            for c in sensor:self.assertEqual(sensor[c],service.frame(c))
            self.assertEqual(after['observation']['cameras'],['head','left','right'])
        finally:service.close()

    def test_view_rejects_invalid_coordinates_and_unknown_parameters(self):
        for bad in ({'mode':'arbitrary'},{**FRONT,'extra':1},{**FRONT,'mode':'free','distance':float('nan')},
                    {**FRONT,'mode':'free','azimuth':True},{**FRONT,'mode':'free','lookat':[0,0,100]}):
            with self.assertRaises(ValueError):validate_view(bad)
