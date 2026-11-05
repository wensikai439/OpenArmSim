"""Real-only contract checks: no simulator or robot driver imports."""
import unittest
from .robot import Robot,validate_observation,clean,right_region_error


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

    def test_measured_interval_must_fit_entire_lower_half(self):
        def state(low,high):return {'grasp_region_evidence':{'right':{'verified':True,
            'source':'measured_geometry','part':'body','fraction_min':low,'fraction_max':high}}}
        self.assertIsNone(right_region_error(state(.30,.5)))
        self.assertEqual(right_region_error(state(.40,.58)),'grasp_region_mismatch')
        self.assertEqual(right_region_error(state(.49,.51)),'grasp_region_mismatch')
        self.assertEqual(right_region_error(state(float('nan'),.4)),'grasp_region_unverified')
        self.assertEqual(right_region_error({}),'grasp_region_unverified')

    def test_terminal_or_invalid_evidence_prevents_any_post(self):
        calls=[]
        r=Robot({'robot_service':'http://127.0.0.1:8000'},execute=True,
                transport=lambda *a,**k:calls.append(a))
        r.observe=lambda:{'terminal':True,'invalid':False,'held_by':[]}
        with self.assertRaises(RuntimeError):r.call('approach','g0')
        self.assertEqual(calls,[])
        self.assertEqual(r.motion_requests_attempted,0)

    def test_missing_region_prevents_lift_and_unverified_support_prevents_release(self):
        calls=[]
        r=Robot({'robot_service':'http://127.0.0.1:8000'},execute=True,
                transport=lambda *a,**k:calls.append(a))
        r.observe=lambda:{'terminal':False,'invalid':False,'held_by':['right']}
        self.assertEqual(r.call('lift')['error_code'],'grasp_region_unverified')
        self.assertEqual(r.call('release_on_support')['error_code'],'supported_release_required')
        self.assertEqual(calls,[])

    def test_timeout_after_dispatch_remains_a_possible_motion(self):
        def timeout(*args,**kwargs):raise RuntimeError('service timeout')
        r=Robot({'robot_service':'http://127.0.0.1:8000'},execute=True,transport=timeout)
        r.observe=lambda:{'terminal':False,'invalid':False,'held_by':[]}
        with self.assertRaises(RuntimeError):r.call('approach','g0')
        self.assertEqual(r.motion_requests_attempted,1)


if __name__=='__main__':unittest.main()
