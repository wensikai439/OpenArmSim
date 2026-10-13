"""Native composition, frozen object goals, abort and backend boundaries."""
import unittest
from unittest.mock import patch

import numpy as np

from .environment import OpenArmEnv
from .relocation import RelocationPlanner
from .tamp import PlanningFailure
from harness.workspace import validate_plan,Workspace
from harness.skills import SKILL_TOOLS,validate_arguments


class RelocationChecks(unittest.TestCase):
    def setUp(self):
        self.env=OpenArmEnv(images=False);self.env.reset(seed=0);self.env.set_task_budget(60)

    def tearDown(self):self.env.close()

    def run_native(self,params,feedback=lambda value:None,cancel=lambda:False):
        planner=RelocationPlanner(self.env,'relocate_object',parameters=params,feedback=feedback,cancel=cancel)
        iterator=planner.actions()
        while True:
            q=self.env.data.qpos.copy()
            try:action,_=next(iterator)
            except StopIteration:break
            np.testing.assert_array_equal(q,self.env.data.qpos)
            _,_,term,trunc,_=self.env.step_tamp(action)
            self.assertFalse(term or trunc)
        return planner

    def test_parameterized_forward_relocation_has_real_grasp_and_independent_placement(self):
        start=self.env.data.xpos[self.env.bottle_id].copy()
        planner=self.run_native({'object_id':'bottle','arm':'right','receptacle_id':'table','offset_xy_m':[.08,0]})
        self.assertEqual([s['skill'] for s in planner.result['substeps']],['pick','place'])
        self.assertTrue(all(s['status']=='completed' for s in planner.result['substeps']))
        self.assertTrue(planner.result['primitive_success'])
        self.assertEqual(planner.result['verification_source'],'native_support_release_and_stability')
        self.assertLess(np.linalg.norm(self.env.data.xpos[self.env.bottle_id,:2]-start[:2]-[.08,0]),.01)
        self.assertTrue(planner.result['placement']['confirmed'])
        self.assertFalse(any(self.env.evaluation['touching'].values()))
        self.assertFalse(self.env.evaluation['success'])

    def test_left_to_right_then_original_goal_with_no_scene_reset(self):
        start=self.env.data.xpos[self.env.bottle_id,:2].copy()
        planner=self.run_native({'object_id':'bottle','arm':'left','via_arm':'right','receptacle_id':'table','offset_xy_m':[.06,0]})
        self.assertEqual([s['skill'] for s in planner.result['substeps']],['pick','handover','place'])
        transfer=planner.result['substeps'][1]['feedback']
        self.assertEqual(transfer['held_by'],'right')
        self.assertEqual(transfer['verification_source'],'native_receiver_contact_then_donor_release')
        np.testing.assert_allclose(planner.result['target_xy_m'],start+[.06,0],rtol=0,atol=1e-12)
        self.assertTrue(planner.result['placement']['confirmed'])

    def test_invalid_target_and_unavailable_explicit_backend_start_no_motion(self):
        original=self.env.data.qpos.copy()
        for params in [{'object_id':'bottle','arm':'right','receptacle_id':'table','target_xy_m':[10.,0.]},
                       {'object_id':'bottle','arm':'right','via_arm':'right','receptacle_id':'table','offset_xy_m':[.08,0]}]:
            with self.assertRaises((ValueError,PlanningFailure)):RelocationPlanner(self.env,'relocate_object',parameters=params)
            np.testing.assert_array_equal(original,self.env.data.qpos)
        with patch('sim.relocation.metadata',return_value={'ready':False}):
            with self.assertRaisesRegex(PlanningFailure,'cuRobo'):
                RelocationPlanner(self.env,'relocate_object',parameters={'object_id':'bottle','arm':'right',
                    'receptacle_id':'table','offset_xy_m':[.08,0],'transport_planner':'curobo','transport_delta_m':[.03,0,0]})
        self.assertEqual(self.env.tick,0)

    def test_cancellation_between_subskills_never_releases_or_places(self):
        stopped=False
        def feedback(value):
            nonlocal stopped
            if value.get('composite_step_complete')=='pick':stopped=True
        with self.assertRaisesRegex(PlanningFailure,'停止'):
            self.run_native({'object_id':'bottle','arm':'right','receptacle_id':'table','offset_xy_m':[.08,0]},feedback,lambda:stopped)
        self.assertTrue(self.env.evaluation['held']['right'])
        self.assertFalse(self.env.evaluation['placement']['confirmed'])

    def test_relocation_cannot_fake_grasp_checkpoint_or_complete_from_pick_only(self):
        with self.assertRaises(ValueError):validate_plan({'steps':[{'id':'grip','title':'抓取','checkpoint':'grasp_confirmed','skill':'relocate'}]})
        steps=validate_plan({'steps':[{'id':'goal','title':'搬到目标','checkpoint':'placement_confirmed','skill':'relocate'}]})
        self.assertEqual(steps[0]['skill'],'relocate')
        self.assertFalse(Workspace._step_done(steps[0],{'primitive_success':True,'verification_source':'native_two_finger_contact_and_lift'},{}))
        schema=next(t['function']['parameters'] for t in SKILL_TOOLS if t['function']['name']=='relocate')
        with self.assertRaises(ValueError):validate_arguments(schema,{'object_id':'bottle','arm':'right','receptacle_id':'table','observation_id':'e:0','allow_handover_recovery':1})
