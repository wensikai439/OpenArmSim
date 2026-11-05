"""Atomic policy role compatibility, preserved guards and unknown outcomes."""
import copy
import unittest
from types import SimpleNamespace

from .intent import extract,progress,install_isolated as intent
from .policy_route import admit,install_isolated


class RouteChecks(unittest.TestCase):
    def test_left_final_placement_and_full_right_left_route_are_admitted(self):
        for goal in ('由左手把瓶子放入白盒。','用右手抓起瓶子交给左手，由左手放入白盒。'):
            c=extract(goal);s=progress(c,[]);old=copy.deepcopy(s)
            result=admit(c,s);self.assertTrue(result['admission_only']);self.assertFalse(result['completion_verified'])
            self.assertEqual(s,old);self.assertEqual(progress(c,[])['verified_requirements'],0)

    def test_reverse_wrong_final_hand_wrong_object_and_protected_object_rejected(self):
        for goal in ('用左手抓起瓶子交给右手。','由右手把瓶子放入白盒。',
            '用右手抓起杯子。','瓶子不动。'):
            c=extract(goal)
            with self.assertRaises(ValueError,msg=goal):admit(c,progress(c,[]))

    def test_repeated_required_action_is_not_silently_deduplicated(self):
        c=extract('由左手把瓶子放入白盒。');s=progress(c,[]);s['remaining']*=2
        with self.assertRaises(ValueError):admit(c,s)

    def harness(self,goal,*,capability='atomic_right_to_left_handover',lower_error=None):
        class Lower:
            def __init__(self):
                self.service=SimpleNamespace(policy_config={'capability':capability});self.called=[]
                self.sessions={'root':{'goal':goal}}
            def _invoke(self,name,args,role,context=None):
                self.called.append((name,args))
                if lower_error:raise ValueError(lower_error)
                return {'execution_id':'actual-native-id'}
        Intent=intent(Lower)
        Intent._role_records=lambda self,root:[]
        return install_isolated(Intent)()

    def test_lower_dispatch_and_arguments_are_preserved_without_completion(self):
        h=self.harness('由左手把瓶子放入白盒。');args={'backend':'act','skill_id':'atomic_right_to_left_handover','observation_id':'e:0'}
        result=h._invoke('run_policy',args,'execute',{'type':'llm','turn_id':'root'})
        self.assertEqual(h.called,[('run_policy',args)]);self.assertIs(h.called[0][1],args)
        self.assertEqual(result['execution_id'],'actual-native-id');self.assertFalse(result['policy_role_admission']['completion_verified'])

    def test_lower_native_guard_still_rejects_before_policy_motion(self):
        h=self.harness('由左手把瓶子放入白盒。',lower_error='stale observation')
        with self.assertRaisesRegex(ValueError,'stale observation'):
            h._invoke('run_policy',{'backend':'act'},'execute',{'type':'llm','turn_id':'root'})

    def test_reverse_role_is_rejected_before_lower_dispatch(self):
        h=self.harness('用左手抓起瓶子交给右手。')
        with self.assertRaises(ValueError):h._invoke('run_policy',{'backend':'act'},'execute',{'type':'llm','turn_id':'root'})
        self.assertEqual(h.called,[])

    def test_unreviewed_capability_keeps_original_admission(self):
        h=self.harness('由左手把瓶子放入白盒。',capability='unknown')
        with self.assertRaises(ValueError):h._invoke('run_policy',{'backend':'act'},'execute',{'type':'llm','turn_id':'root'})
        self.assertEqual(h.called,[])


if __name__=='__main__':unittest.main()
