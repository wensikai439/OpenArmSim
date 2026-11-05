"""Provider routing only; native performance uses the independent batches."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from harness.margin_grasp import install_isolated


class RoutingChecks(unittest.TestCase):
    def setUp(self):
        self.env=SimpleNamespace(object_id='bottle',bottle_id=0,data=SimpleNamespace(xmat=np.eye(3).reshape(1,9)))
    def check(self,task,params,expected,*,ident='bottle',upright=True):
        import sim.service as service
        env=self.env;env.object_id=ident;env.data.xmat[0,8]=1 if upright else 0
        class Old:
            CONTRACT={'original':True}
            def __new__(cls,e,t,**kw):return ('original',t,copy.deepcopy(kw))
        before=copy.deepcopy(params);factory=object()
        with patch.object(service,'TampPlanner',Old),patch('sim.margin_grasp.MarginBottlePick',return_value='refined') as pick, \
                patch('sim.fast_handover.FastBottleHandover',return_value='fast-relay') as relay, \
                patch('sim.bottle_routing.ContinuousHoldAudit',return_value='continuous-relay') as continuous, \
                patch('sim.oriented_grasp.HoldAudit',return_value='audited') as audit:
            self.assertIs(install_isolated(factory),factory)
            result=service.TampPlanner(env,task,parameters=params)
            self.assertEqual(result if isinstance(result,str) else result[0],expected)
            self.assertEqual(params,before)
            if expected=='original':self.assertFalse(pick.called or relay.called or audit.called)
            if expected=='audited':audit.assert_called_once_with('refined',10)
            if expected=='continuous-relay':continuous.assert_called_once_with('fast-relay',10)
    def test_directed_pick_and_goal_placement_use_refinement_provider(self):
        self.check('pick_bottle',{'arm':'left','next_handover':{'to_arm':'right'}},'refined')
        self.check('pick_bottle',{'arm':'right','bottle_single_arm_placement':True},'refined')
    def test_directed_relay_and_continuous_hold_wrapper(self):
        self.check('handover_bottle',{'from_arm':'left','to_arm':'right'},'fast-relay')
        self.check('handover_object',{'from_arm':'left','to_arm':'right'},'fast-relay')
        self.check('pick_bottle',{'arm':'left','next_handover':{'to_arm':'right'},'hold_seconds':10},'audited')
        self.check('handover_bottle',{'from_arm':'left','to_arm':'right','hold_seconds':10},'continuous-relay')
    def test_old_demo_right_pick_and_motion_tools_are_preserved(self):
        for task,params in [('handover_to_basket',{}),('pick_bottle',{'arm':'right'}),
            ('handover_bottle',{'from_arm':'right','to_arm':'left'}),
            ('move_to_pose',{'arm':'right'}),('place_bottle',{'arm':'left'})]:
            self.check(task,params,'original')
    def test_other_shapes_and_non_upright_route_are_preserved(self):
        params={'arm':'left','next_handover':{'to_arm':'right'}}
        for ident in ('block','can','cup','book'):self.check('pick_bottle',params,'original',ident=ident)
        self.check('pick_bottle',params,'original',upright=False)


if __name__=='__main__':unittest.main()
