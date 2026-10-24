"""Real-only contract checks: no simulator or robot driver imports."""
import unittest
from .robot import Robot,validate_observation,clean


class RobotChecks(unittest.TestCase):
    def test_unconfigured_service_cannot_be_executed(self):
        r=Robot({'robot_service':None},execute=True)
        self.assertFalse(r.status()['ready'])
        self.assertEqual(r.calls,0)

    def test_external_or_credentialed_robot_service_is_rejected(self):
        for url in ['https://example.com','http://user@localhost:8000','http://127.0.0.1:8000/tools']:
            with self.assertRaises(ValueError):Robot({'robot_service':url})

    def test_no_motion_without_explicit_session_option(self):
        calls=[]
        r=Robot({'robot_service':'http://127.0.0.1:8000'},transport=lambda *a,**k:calls.append(a))
        with self.assertRaises(RuntimeError):r.call('close_gripper',arm='right')
        self.assertEqual(calls,[])

    def test_simulator_replay_stale_and_nonfinite_observations_are_rejected(self):
        for source,stamp in [('synthetic_fixture',100.),('measured_perception',99.),
                             ('measured_perception',float('nan'))]:
            with self.assertRaises(ValueError):validate_observation({'source':source,'frame':'robot_base',
                'timestamp_unix_s':stamp},now=100.)

    def test_credentials_are_not_retained_in_record(self):
        self.assertEqual(clean({'api_key':'secret','authorization':'secret','ok':True}),{'ok':True})


if __name__=='__main__':unittest.main()
