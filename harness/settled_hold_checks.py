"""Scoped optional pre-hold settling composition checks."""
import unittest
from unittest.mock import patch

from harness.settled_hold import SettledPlanner, compose_planner
from sim.tamp import PlanningFailure


class SettledHoldChecks(unittest.TestCase):
    def test_composition_is_local_idempotent_and_still_requires_explicit_hold(self):
        from sim import service
        original=service.TampPlanner
        inner=object()

        class Audit:
            def __init__(self,planner): self.planner=planner

        class Base:
            CONTRACT={'planner':'fixture'}
            def __new__(cls,*args,**kwargs): return Audit(inner)

        with patch('sim.bottle_routing.ContinuousHoldAudit',Audit):
            Router=compose_planner(Base)
            self.assertIs(compose_planner(Router),Router)
            self.assertIs(service.TampPlanner,original)
            no_request=Router(type('Env',(),{'source_controller':None})(),'pick',parameters={})
            self.assertIs(no_request.planner,inner)
            requested=Router(type('Env',(),{'source_controller':None})(),'pick',
                parameters={'hold_seconds':10})
            self.assertIsInstance(requested.planner,SettledPlanner)
            self.assertIs(requested.planner.planner,inner)
            trained=Router(type('Env',(),{'source_controller':object()})(),'pick',
                parameters={'hold_seconds':10})
            self.assertIs(trained.planner,inner)
        self.assertIs(service.TampPlanner,original)

    def test_settling_keeps_success_gated_and_records_its_separate_receipt(self):
        class Inner:
            env=object()
            result={'primitive_success':True,'held_by':'left'}
            def actions(self): yield 'native-motion',{}

        class FakeSettling:
            def __init__(self,env,side): self.env,self.side=env,side
            def actions(self): yield 'settling-motion',{'phase':'pre-hold-settling'}
            def summary(self): return {'arm':self.side,'ready_for_independent_audit':True}

        with patch('harness.settled_hold.HoldSettling',FakeSettling):
            planner=SettledPlanner(Inner())
            self.assertEqual(list(planner.actions()),[
                ('native-motion',{}),('settling-motion',{'phase':'pre-hold-settling'})])
            self.assertTrue(planner.result['primitive_success'])
            self.assertEqual(planner.result['status'],'completed')
            self.assertEqual(planner.result['pre_hold_settling']['arm'],'left')

            class Failed(Inner):
                result={'primitive_success':False,'held_by':None}
            with self.assertRaisesRegex(PlanningFailure,'unconfirmed grasp'):
                list(SettledPlanner(Failed()).actions())


if __name__=='__main__': unittest.main()
