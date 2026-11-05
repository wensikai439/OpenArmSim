"""Placement cannot pass from a closed hand, wrong target or blinking support."""
import unittest
import numpy as np
from .environment import OpenArmEnv
from .tamp import TampPlanner,PlanningFailure

class PlacementChecks(unittest.TestCase):
    def setUp(self):self.env=OpenArmEnv(images=False);self.env.reset(seed=0)
    def tearDown(self):self.env.close()
    def test_empty_hand_and_outside_surface_are_rejected_without_motion(self):
        q=self.env.data.qpos.copy()
        with self.assertRaises(PlanningFailure):TampPlanner(self.env,'place_bottle',parameters={'arm':'right','receptacle_id':'table','offset_xy_m':[.08,0]})
        with self.assertRaises(ValueError):self.env.set_placement_target('table',[1.1,0])
        np.testing.assert_array_equal(q,self.env.data.qpos)
    def test_target_footprint_support_contact_and_release_are_independent(self):
        env=self.env;env.set_placement_target('table',env.data.xpos[env.bottle_id,:2])
        contact=env.contacts.read(env.data);contact['table_supported']=True
        self.assertTrue(env._placement_condition(contact,np.zeros(6)))
        contact['touching']['right']=True
        self.assertFalse(env._placement_condition(contact,np.zeros(6)))
        contact['touching']['right']=False;contact['table_supported']=False
        self.assertFalse(env._placement_condition(contact,np.zeros(6)))
        contact['table_supported']=True;env.set_placement_target('table',[.6,-.173])
        self.assertFalse(env._placement_condition(contact,np.zeros(6)))
    def test_stability_is_continuous_at_physics_resolution_and_reset_expires_goal(self):
        env=self.env;contact=env.contacts.read(env.data)
        for _ in range(900):env.contacts.sample(contact,.001,0,placement_stable=True)
        env.contacts.sample(contact,.001,0,placement_stable=False)
        for _ in range(200):env.contacts.sample(contact,.001,0,placement_stable=True)
        self.assertAlmostEqual(env.contacts.placement_seconds,.2)
        env.set_placement_target('table',[.6,-.173]);env.reset(seed=0)
        self.assertIsNone(env.placement_target);self.assertFalse(env.evaluation['placement']['confirmed'])

    def test_one_centimeter_goal_rejects_unmoved_body_at_native_resolution(self):
        env=self.env;target=env.data.xpos[env.bottle_id,:2]+[.01,0]
        env.set_placement_target('table',target,.0025)
        contact=env.contacts.read(env.data);contact['table_supported']=True
        self.assertFalse(env._placement_condition(contact,np.zeros(6)))
        with self.assertRaises(ValueError):env.set_placement_target('table',target,.025)

if __name__=='__main__':unittest.main()
