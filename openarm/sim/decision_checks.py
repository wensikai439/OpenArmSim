"""Native decision waits and controlled-provider failure checks, no real API."""
import copy
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from .environment import OpenArmEnv, ROOT
from .service import SimulationService
from .decision_timing import DecisionWait, GraspStability, ObservationChanged, revalidation, rebind_observation
from harness.failures import CallFailures
from .tamp import TampPlanner
from harness.agent import Harness
from harness.workspace import Workspace
from harness.llm import ModelSettings
from harness.goal_checks import snapshot as fixture
from harness.goals import resolve_placement, assess_placement
from harness.checks import LocalState


def drift_fixture():
    value=fixture()
    value['observation']['tcp']={side:{'position_m':[.4,0,1.2],'quaternion_xyzw':[0,0,0,1]} for side in ('left','right')}
    value['diagnostics']['manipulation_scene'].update(upright=True,held_by=None)
    return value


class DecisionChecks(unittest.TestCase):
    def test_rejected_drift_contains_measurements_and_does_not_use_format_budget(self):
        before=drift_fixture();after=copy.deepcopy(before);after['observation_id']='episode:30'
        after['diagnostics']['manipulation_scene']['position_m'][0]+=.004
        failures=CallFailures();failures.record('bad-json',ValueError('invalid JSON'),malformed=True)
        for i in range(2):
            try:rebind_observation({'observation_id':'episode:0','arm':'right'},before,after)
            except ObservationChanged as exc:
                result=failures.record(str(i),exc)
                self.assertFalse(result['motion_started'])
                self.assertAlmostEqual(result['revalidation']['object_drift_m'],.004)
                self.assertEqual(result['revalidation']['current_observation_id'],'episode:30')
                self.assertEqual(result['failure_kind'],'observation_refresh')
            else:self.fail('stale proposal was not rejected')
        self.assertFalse(failures.exhausted)
        failures.record('third',ObservationChanged(revalidation(before,after)))
        self.assertTrue(failures.exhausted)

    def test_relative_stability_audit_sees_sliding_despite_bilateral_contact(self):
        # Independent synthetic rigid-body fixture, never a robot rollout.
        class Data:
            time=0.;xpos=np.zeros((3,3));xmat=np.tile(np.eye(3).ravel(),(3,1))
        class Env:
            data=Data();ee_ids={'left':0,'right':1};bottle_id=2
            evaluation={'held':{'left':False,'right':True}}
        env=Env();audit=GraspStability(env)
        self.assertEqual(audit.summary()['arms']['right']['status'],'insufficient_duration')
        env.data.time=2.;env.data.xpos[2,2]=-.002;audit.read()
        result=audit.summary()
        self.assertEqual(result['arms']['right']['status'],'sliding')
        self.assertAlmostEqual(result['arms']['right']['max_window_relative_drift_m'],.002)
        self.assertTrue(env.evaluation['held']['right'])
        env.data.time=4.;env.data.xpos[:]+=np.array([.4,-.2,.1]);audit.read()
        self.assertAlmostEqual(audit.summary()['arms']['right']['max_window_relative_drift_m'],.002)
        env.evaluation['held']['right']=False;env.data.time=4.1;audit.read()
        self.assertEqual(audit.summary()['arms']['right']['status'],'contact_lost')

    def test_stability_audit_detects_motion_that_returns_to_its_start(self):
        # Controlled rigid-body poses; this is a diagnostic contract fixture.
        class Data:
            time=0.;xpos=np.zeros((3,3));xmat=np.tile(np.eye(3).ravel(),(3,1))
        class Env:
            data=Data();ee_ids={'left':0,'right':1};bottle_id=2
            evaluation={'held':{'left':False,'right':True}}
        env=Env();audit=GraspStability(env)
        for stamp,delta in ((1.,.0006),(1.5,-.0006),(2.,0.)):
            env.data.time=stamp;env.data.xpos[2,0]=delta;audit.read()
        result=audit.summary()['arms']['right']
        self.assertEqual(result['status'],'sliding')
        self.assertAlmostEqual(result['max_window_relative_drift_m'],.0012)
        # Pure rotation returns to its initial angle as well.
        env=Env();env.data=Data();env.data.xpos=np.zeros((3,3));env.data.xmat=np.tile(np.eye(3).ravel(),(3,1));audit=GraspStability(env)
        for stamp,theta in ((1.,.012),(1.5,-.012),(2.,0.)):
            env.data.time=stamp;c,s=np.cos(theta),np.sin(theta)
            env.data.xmat[2]=np.array([[c,-s,0],[s,c,0],[0,0,1]]).ravel();audit.read()
        result=audit.summary()['arms']['right']
        self.assertEqual(result['status'],'sliding')
        self.assertAlmostEqual(result['max_window_relative_rotation_rad'],.024)

    def test_stability_window_tolerates_floating_boundary_and_retains_late_drift(self):
        class Data:
            time=0.;xpos=np.zeros((3,3));xmat=np.tile(np.eye(3).ravel(),(3,1))
        class Env:
            data=Data();ee_ids={'left':0,'right':1};bottle_id=2
            evaluation={'held':{'left':False,'right':True}}
        env=Env();audit=GraspStability(env)
        for i in range(1,181):
            env.data.time=i/30+1e-10
            env.data.xpos[2,0]=0 if i<95 else .002
            audit.read()
        result=audit.summary()
        self.assertEqual(result['arms']['right']['status'],'sliding')
        self.assertAlmostEqual(result['arms']['right']['max_window_relative_drift_m'],.002)
        self.assertLessEqual(len(audit.samples['right']),62)
        self.assertLessEqual(result['maximum_sample_span_seconds'],2+1/30+1e-6)

    def test_rebinding_requires_this_request_source_and_logs_both_observations(self):
        before=drift_fixture();after=copy.deepcopy(before);after['observation_id']='episode:30'
        args={'observation_id':'episode:0','arm':'right','delta_m':[0,0,.01]}
        rebound,receipt=rebind_observation(args,before,after)
        self.assertEqual(rebound['observation_id'],'episode:30');self.assertEqual(args['observation_id'],'episode:0')
        self.assertEqual(receipt['source_observation_id'],'episode:0')
        with self.assertRaisesRegex(ValueError,'旧观测'):rebind_observation({**args,'observation_id':'episode:older'},before,after)
        after['decision_timing']={'task_failed':True}
        with self.assertRaisesRegex(ValueError,'未执行动作'):rebind_observation(args,before,after)
        self.assertEqual(rebind_observation(args,before,after,failure_report=True)[0]['observation_id'],'episode:30')

    def test_non_success_report_can_label_old_evidence_without_authorizing_motion(self):
        before=drift_fixture();before['observation_id']='episode:20';after=copy.deepcopy(before);after['observation_id']='episode:30'
        args={'observation_id':'episode:5','verdict':'uncertain'}
        rebound,evidence=rebind_observation(args,before,after,failure_report=True)
        self.assertEqual(rebound['observation_id'],'episode:30');self.assertEqual(args['observation_id'],'episode:5')
        self.assertTrue(evidence['historical_evidence']);self.assertFalse(evidence['motion_authorized'])
        with self.assertRaises(ObservationChanged):rebind_observation(args,before,after)
        with self.assertRaises(ObservationChanged):rebind_observation({**args,'verdict':'success'},before,after,failure_report=True)
        for old in ('other:5','episode:31','episode:bad'):
            with self.assertRaises(ObservationChanged):rebind_observation({**args,'observation_id':old},before,after,failure_report=True)

    def test_accepted_finish_is_terminal_without_another_provider_request(self):
        # Controlled provider; verifies terminal-loop behavior, not LLM skill.
        state=LocalState();state.cache['last_run']={'execution_id':'done','status':'primitive_completed','run_id':'fixture'}
        class Store:
            def load(self):return {'base_url':'http://127.0.0.1:1234/v1','model':'fixture','api_key':'test-placeholder','vision':True}
            def save(self,value):pass
        agent=Harness(state,settings=ModelSettings(Store()));calls=[]
        def request(*args,**kwargs):
            calls.append(True)
            if len(calls)>1:raise AssertionError('accepted finish must not issue another API request')
            payload={'choices':[{'message':{'role':'assistant','tool_calls':[{'id':'f','type':'function','function':{
                'name':'finish','arguments':json.dumps({'observation_id':'episode:0','verdict':'uncertain','reason':'fixture cannot establish success'})}}]}}]}
            class Response:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read(self,_):return json.dumps(payload).encode()
            return Response()
        with patch('harness.agent.urllib.request.urlopen',side_effect=request):result=agent.chat('Review this completed fixture')
        self.assertEqual(len(calls),1)
        self.assertTrue(result['tools'][0]['result']['assessment_recorded'])
        self.assertIn('不确定',result['message'])
        self.assertEqual(agent.workspace.db.execute('SELECT status FROM turns WHERE id=?',(result['turn_id'],)).fetchone()[0],'completed')

    def test_first_plan_locks_timing_and_keeps_learned_policy_contract(self):
        class State(LocalState):
            def call(self,name,**args):
                self.calls.append((name,args));return {'accepted':True}
        state=State();state.cache=drift_fixture();agent=Harness(state)
        context={'type':'llm','turn_id':'t','plan_turn_id':'t','model':'test'}
        args={'observation_id':'episode:0','steps':[{'id':'m','title':'右臂移动','checkpoint':'motion_reached','skill':'move_relative'}],
              'time_budget_s':120,'timing_mode':'measured_delay'}
        with patch.object(agent,'_log'):
            agent.invoke('observe',{});agent.invoke('set_plan',args,context=context)
            self.assertEqual(agent.workspace.active_plan['timing_mode'],'measured_delay')
            with self.assertRaisesRegex(ValueError,'已锁定'):agent.invoke('set_plan',{**args,'timing_mode':'paused'},context=context)
            with self.assertRaisesRegex(ValueError,'独立策略计时'):agent.invoke('set_plan',{**args,'steps':[{'id':'v','title':'策略','checkpoint':'tool_completed','skill':'run_policy'}]},context=context)

    def test_wait_is_exact_native_hold_and_final_rounding_is_bounded(self):
        env=OpenArmEnv(images=False);env.reset()
        try:
            action=env.last_action.copy();clock=[0.]
            with tempfile.TemporaryDirectory() as out:
                wait=DecisionWait(env,'a'*32,1,'e:0',clock=lambda:clock[0],output=out)
                clock[0]=.101;wait.advance();self.assertEqual(wait.steps,3)
                report=wait.finish();self.assertEqual(report['hold_transitions'],4)
                self.assertLess(report['integrated_seconds']-.101,1/30+.001)
                actual=env.data.qpos.copy()
                for row in wait.recorder.path.joinpath('transitions.jsonl').read_text().splitlines():
                    np.testing.assert_array_equal(json.loads(row)['action'],action)
                env.reset()
                for _ in range(4):env.step_tamp(action)
                np.testing.assert_array_equal(actual,env.data.qpos)
        finally:env.close()

    def test_native_grasp_continues_to_hold_during_two_seconds_of_wait(self):
        env=OpenArmEnv(images=False);env.reset();env.set_task_budget(60)
        try:
            p=TampPlanner(env,'pick_bottle',parameters={'arm':'right'})
            for action,_ in p.actions():env.step_tamp(action)
            self.assertTrue(p.result['primitive_success'])
            target=env.last_action.copy();clock=[0.]
            with tempfile.TemporaryDirectory() as out:
                wait=DecisionWait(env,'b'*32,2,'e:'+str(env.tick),clock=lambda:clock[0],output=out)
                clock[0]=2.;report=wait.finish()
                self.assertFalse(report['task_failed']);self.assertEqual(report['hold_transitions'],60)
                np.testing.assert_array_equal(env.last_action,target)
                self.assertTrue(env.evaluation['held']['right'])
                # Current candidate can maintain bilateral contacts yet creep.
                # Do not let a native held=True conceal that measured defect.
                self.assertEqual(report['grasp_stability']['arms']['right']['status'],'sliding')
        finally:env.close()

    def test_deadline_stops_wait_without_stepping_past_it(self):
        env=OpenArmEnv(images=False);env.reset();env.set_task_budget(40)
        try:
            env.data.time=39.967;env.tick=1199;clock=[0.]
            with tempfile.TemporaryDirectory() as out:
                wait=DecisionWait(env,'c'*32,3,'e:1199',clock=lambda:clock[0],output=out)
                clock[0]=1.;report=wait.finish()
                self.assertEqual(report['status'],'budget_exhausted');self.assertEqual(report['hold_transitions'],1)
                self.assertLessEqual(env.data.time,40.001)
        finally:env.close()

    def test_scene_drift_epoch_and_hold_changes_reject_old_decisions(self):
        before=drift_fixture();after=copy.deepcopy(before);after['observation_id']='episode:30'
        self.assertTrue(revalidation(before,after)['accepted'])
        for modify in ('tcp','object','hold','epoch','orientation'):
            x=copy.deepcopy(after)
            if modify=='tcp':x['observation']['tcp']['left']['position_m'][2]+=.006
            if modify=='object':x['diagnostics']['manipulation_scene']['position_m'][0]+=.004
            if modify=='hold':x['diagnostics']['manipulation_scene']['held_by']='left'
            if modify=='epoch':x['observation_id']='different:30'
            if modify=='orientation':x['diagnostics']['manipulation_scene']['upright']=False
            self.assertFalse(revalidation(before,x)['accepted'],modify)

    def test_wait_deadline_invalidates_an_otherwise_stable_object_goal(self):
        s=fixture();goal=resolve_placement({'object_id':'bottle','surface':'table','offset_xy_m':[.01,0]},s)
        s['diagnostics']['manipulation_scene']['position_m'][:2]=goal['target_xy_m']
        s['diagnostics']['evaluation'].update(invalid=False,touching={'left':False,'right':False},
            placement={'surface':'table','xy_m':goal['target_xy_m'],'tolerance_m':goal['tolerance_m'],'confirmed':True})
        self.assertTrue(assess_placement(goal,s)['confirmed'])
        s['decision_timing']={'task_failed':True,'status':'budget_exhausted'}
        self.assertFalse(assess_placement(goal,s)['confirmed'])

    def test_timing_conditions_have_separate_skill_outcome_counts(self):
        work=Workspace();context={'contract':'one'}
        for i,mode in enumerate(('paused','measured_delay')):
            work.record_feedback(str(i),'execute','e','hold','pick',{'arm':'right'},
                {'primitive_success':True,'verification_source':'native_two_finger_contact_and_lift','decision_timing_mode':mode},context,'simulator_oracle')
        profiles=work.capability_profile('execute',context=context,include_oracle=True)
        self.assertEqual({p['timing_mode'] for p in profiles},{'paused','measured_delay'})
        self.assertTrue(all(p['scored_attempts']==1 for p in profiles))

    def controlled_request(self, *, stop=False):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='.decision-check-') as out:
            class Wait(DecisionWait):
                def __init__(self,*args,**kwargs):super().__init__(*args,output=out,**kwargs)
            class Store:
                def load(self):return {'base_url':'http://127.0.0.1:1234/v1','model':'controlled-test','api_key':'non-secret-placeholder','vision':True}
                def save(self,value):pass
            service=SimulationService();agent=Harness(service,settings=ModelSettings(Store()));calls=[0]
            initial_target=service.env.last_action.copy()
            def response(request,**kwargs):
                calls[0]+=1
                if calls[0]==1:
                    message={'tool_calls':[{'id':'plan','type':'function','function':{'name':'set_plan',
                        'arguments':json.dumps({'observation_id':service.snapshot()['observation_id'],'time_budget_s':60,
                            'timing_mode':'measured_delay','steps':[{'id':'move','title':'右手上移','checkpoint':'motion_reached','skill':'move_relative'}]})}}]}
                else:
                    deadline=time.monotonic()+3
                    while service.snapshot()['observation']['tick']<2 and time.monotonic()<deadline:threading.Event().wait(.01)
                    self.assertGreaterEqual(service.snapshot()['observation']['tick'],2)
                    if stop:
                        agent.invoke('stop',{})
                        message={'tool_calls':[{'id':'late','type':'function','function':{'name':'move_relative',
                            'arguments':json.dumps({'observation_id':'irrelevant','arm':'right','frame':'world','delta_m':[0,0,.01]})}}]}
                    else:raise TimeoutError('controlled provider timeout')
                class Response:
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def read(self,_):return json.dumps({'choices':[{'message':message}]}).encode()
                return Response()
            try:
                with patch('sim.service.DecisionWait',Wait),patch('harness.agent.urllib.request.urlopen',side_effect=response),patch.object(agent,'_log'):
                    if stop:
                        result=agent.chat('受控接口测试，声明等待后移动')
                        self.assertNotIn('move_relative',[c['name'] for c in result['tools']])
                    else:
                        with self.assertRaisesRegex(RuntimeError,'没有切换为脚本'):agent.chat('受控接口测试，声明等待后移动')
                s=service.snapshot();self.assertIsNone(s['job']);self.assertIsNone(s['last_run'])
                self.assertGreater(s['observation']['tick'],0)
                self.assertEqual(s['decision_timing']['status'],'cancelled' if stop else 'completed')
                np.testing.assert_array_equal(initial_target,service.env.last_action)
            finally:agent.close();service.close()

    def test_api_timeout_accounts_for_wait_but_applies_no_new_motion(self):self.controlled_request()

    def test_stop_during_api_discards_the_late_motion_response(self):self.controlled_request(stop=True)


if __name__=='__main__':unittest.main()
