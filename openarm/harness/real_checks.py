"""Boundary checks for offline connection tooling, not hardware validation."""
import copy
import unittest

from .real import ROOT, ORDER, read, validate, preflight, replay


class RealConnectionChecks(unittest.TestCase):
    def setUp(self):
        self.config=read(ROOT/'deploy/config.json')
        self.observation={'schema':'openarm-observation/1','source':'measured_perception',
            'frame':'robot_base','timestamp_unix_s':100.,'state_order':ORDER[:],
            'state_rad':[0.]*16,'objects':{name:{'pose_reference':'model_origin',
                'position_m':[.3,0.,.36],'quaternion_xyzw':[0.,0.,0.,1.],
                'position_std_m':.001} for name in ('bottle','basket')},
            'hold_evidence':{s:{'status':'empty','source':'test_evidence'} for s in ('left','right')}}

    def test_complete_snapshot_passes_only_shadow_input_contract(self):
        self.assertEqual(validate(self.config,self.observation,now=100.1),[])
        self.assertFalse(preflight(self.config)['hardware_execution_available'])

    def test_stale_future_and_nonfinite_time_block(self):
        for stamp in (99.,101.,float('nan')):
            o=copy.deepcopy(self.observation);o['timestamp_unix_s']=stamp
            self.assertTrue(validate(self.config,o,now=100.))

    def test_hardware_flag_is_not_an_enable_switch(self):
        c=copy.deepcopy(self.config);c['hardware_enabled']=True
        self.assertTrue(validate(c,self.observation,now=100.))

    def test_simulator_source_requires_explicit_fixture_mode(self):
        o=copy.deepcopy(self.observation);o['source']='synthetic_fixture'
        self.assertTrue(validate(self.config,o,now=100.))
        self.assertEqual(validate(self.config,o,now=100.,fixture=True),[])

    def test_unknown_hold_is_not_empty_hand(self):
        o=copy.deepcopy(self.observation);o['hold_evidence']['right']['status']='unknown'
        self.assertTrue(validate(self.config,o,now=100.))

    def test_bad_quaternion_uncertainty_and_order_block(self):
        for field,value in (('quaternion_xyzw',[0.,0.,0.,0.]),('position_std_m',float('nan'))):
            o=copy.deepcopy(self.observation);o['objects']['bottle'][field]=value
            self.assertTrue(validate(self.config,o,now=100.))
        o=copy.deepcopy(self.observation);o['state_order']=ORDER[::-1]
        self.assertTrue(validate(self.config,o,now=100.))

    def test_joint_replay_does_not_claim_pose_or_motion(self):
        r=replay(0);self.assertFalse(r['hardware_access'])
        self.assertFalse(r['pose_and_contact_evidence_present'])
        self.assertEqual(len(r['samples'][0]['state_rad']),16)

    def test_default_template_reports_missing_perception(self):
        r=preflight(self.config);self.assertTrue(r['missing'])
        self.assertFalse(r['transfer_ready'])
        self.assertTrue(validate(self.config,read(ROOT/'deploy/observation.example.json')))


if __name__=='__main__':unittest.main()
