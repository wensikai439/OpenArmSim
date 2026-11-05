"""Object target identity and evidence survive changed pose and recovery plan."""
import copy
import unittest
from .goals import resolve_placement,bind_place,preserve_completed,assess_placement,object_state_from_feedback
from .workspace import Workspace
from .agent import Harness
from .checks import LocalState
from unittest.mock import patch

def snapshot():
    s=LocalState().snapshot();s['diagnostics']['manipulation_scene']={'object_id':'bottle',
        'position_m':[.50,-.17,1.005],'radius_m':.028,
        'surfaces':{'table':{'xy_min':[0,-.55],'xy_max':[.82,.55]},
                    'basket':{'xy_min':[.37,.27],'xy_max':[.57,.44]}}}
    return s

class GoalChecks(unittest.TestCase):
    def test_model_budget_is_locked_on_first_plan_and_policy_contract_cannot_expand(self):
        class BudgetState(LocalState):
            def call(self,name,**args):
                if name=='task_budget':
                    self.calls.append((name,args));self.cache['observation']['task_window']={'requested_seconds':args['seconds']}
                    return {'locked':True}
                return super().call(name,**args)
        state=BudgetState();state.cache=snapshot();agent=Harness(state)
        context={'type':'llm','turn_id':'t','plan_turn_id':'t','model':'test'}
        steps=[{'id':'move','title':'末端移动','checkpoint':'motion_reached','skill':'move_relative'}]
        with patch.object(agent,'_log'):
            agent.invoke('observe',{})
            args={'observation_id':'episode:0','steps':steps,'time_budget_s':60}
            agent.invoke('set_plan',args,context=context)
            self.assertEqual(agent.workspace.active_plan['time_budget_s'],60)
            self.assertEqual(len(state.calls),1)
            with self.assertRaisesRegex(ValueError,'不能加时'):agent.invoke('set_plan',{**args,'time_budget_s':90},context=context)
            agent.sessions['t']['actions']=1
            agent.invoke('set_plan',args,context=context);self.assertEqual(len(state.calls),1)
            policy=[{'id':'vla','title':'策略执行','checkpoint':'tool_completed','skill':'run_policy'}]
            with self.assertRaisesRegex(ValueError,'40 秒'):agent.invoke('set_plan',{**args,'steps':policy},context=context)

    def test_task_camera_trail_keeps_failed_attempt_and_never_mixes_epochs_or_claimed_success(self):
        work=Workspace();s=snapshot();steps=[{'id':'a','title':'抓取','checkpoint':'grasp_confirmed','skill':'pick'},
            {'id':'b','title':'放置','checkpoint':'placement_confirmed','skill':'place'}]
        work.plan('t','episode','test',steps)
        def record(ident,skill,epoch,feedback):
            work.bind('t',ident,'execute','goal',skill,skill,'test')
            work.record_feedback(ident,'execute',epoch,'goal',skill,{'arm':'left','receptacle_id':'table'},feedback,
                evidence={'camera_keyframes':[{'observation_id':epoch+':0','sim_time':0.,'url':'/runs/'+ident+'/before.jpg'},
                    {'observation_id':epoch+':10','sim_time':.3,'url':'/runs/'+ident+'/after.jpg'}]})
            s['last_run']={'execution_id':ident,'status':'primitive_completed'}
        record('pick','pick','episode',{'primitive_success':True,'verification_source':'native_two_finger_contact_and_lift','held_by':'left'})
        record('bad','place','episode',{'primitive_success':False,'failure_code':'ik_unreachable'})
        frames=work.summary(s)['goal_keyframes'];self.assertEqual([f['label'] for f in frames],['任务起点','左手抓起','放置未完成'])
        record('claim','pick','episode',{'primitive_success':True,'verification_source':'model_claim'})
        self.assertEqual(work.summary(s)['goal_keyframes'][-1]['outcome'],'failure')
        record('old','pick','old-epoch',{'primitive_success':True})
        self.assertFalse(any(f['execution_id']=='old' for f in work.summary(s)['goal_keyframes']))
    def test_small_translation_cannot_pass_at_unchanged_position_or_loose_native_tolerance(self):
        s=snapshot();goal=resolve_placement({'object_id':'bottle','surface':'table','offset_xy_m':[.01,0]},s)
        self.assertAlmostEqual(goal['tolerance_m'],.0025)
        self.assertEqual(bind_place({'object_id':'bottle','receptacle_id':'table'},goal)['position_tolerance_m'],goal['tolerance_m'])
        s['diagnostics']['evaluation']={'placement':{'confirmed':True,'surface':'table','xy_m':goal['target_xy_m'],'tolerance_m':goal['tolerance_m']}}
        self.assertFalse(assess_placement(goal,s)['confirmed'])
        s['diagnostics']['manipulation_scene']['position_m']=[.509,-.17,1.005]
        self.assertTrue(assess_placement(goal,s)['confirmed'])
        s['diagnostics']['evaluation']['placement']['tolerance_m']=.025
        self.assertFalse(assess_placement(goal,s)['confirmed'])

    def test_late_plan_is_anchored_to_user_start_and_reset_invalidates_it(self):
        initial=snapshot();current=snapshot();current['observation_id']='episode:300'
        current['diagnostics']['manipulation_scene']['position_m']=[.44,.06,1.15]
        spec={'object_id':'bottle','surface':'table','offset_xy_m':[.06,0]}
        goal=resolve_placement(spec,current,initial_snapshot=initial)
        self.assertEqual(goal['target_xy_m'],[.56,-.17]);self.assertEqual(goal['anchor']['observation_id'],'episode:0')
        current['observation_id']='new-epoch:0'
        with self.assertRaisesRegex(ValueError,'初始观测'):resolve_placement(spec,current,initial_snapshot=initial)

    def test_finish_cannot_skip_pending_primitive_or_claim_success_after_failure(self):
        state=LocalState();state.cache=snapshot();agent=Harness(state)
        steps=[{'id':'move','title':'左手上移','checkpoint':'motion_reached','skill':'move_relative'}]
        agent.workspace.plan('t','episode','test',steps);agent.workspace.bind('t','run','execute','move','move_relative','move_relative','test')
        state.cache['last_run']={'execution_id':'run','status':'primitive_completed','run_id':'run'}
        with patch.object(agent,'_log'):
            agent.invoke('observe',{})
            args={'observation_id':'episode:0','verdict':'uncertain','reason':'Next action not executed yet'}
            with self.assertRaisesRegex(ValueError,'计划动作尚未完成'):agent.invoke('finish',args)
            state.cache['last_run']['status']='error'
            with self.assertRaisesRegex(ValueError,'计划动作尚未完成'):agent.invoke('finish',{**args,'verdict':'success'})
            self.assertTrue(agent.invoke('finish',{**args,'verdict':'failure'})['assessment_recorded'])

    def test_scene_memory_uses_actual_arm_and_surface_and_never_promotes_failed_receipt(self):
        grasp={'primitive_success':True,'verification_source':'native_two_finger_contact_and_lift','held_by':'left'}
        self.assertEqual(object_state_from_feedback('pick',grasp)['arm'],'left')
        transfer={**grasp,'verification_source':'native_receiver_contact_then_donor_release','held_by':'right'}
        self.assertEqual(object_state_from_feedback('handover',transfer)['arm'],'right')
        place={'primitive_success':True,'verification_source':'native_support_release_and_stability',
               'placement':{'surface':'table','confirmed':True}}
        state=object_state_from_feedback('place',place)
        self.assertEqual(state['state'],'on_surface');self.assertEqual(state['container_id'],'table')
        self.assertIsNone(object_state_from_feedback('place',{**place,'primitive_success':False}))
        self.assertIsNone(object_state_from_feedback('pick',{**grasp,'held_by':None}))

    def test_two_verified_drift_refusals_allow_uncertain_report_but_never_success(self):
        state=LocalState();state.cache=snapshot();agent=Harness(state);work=agent.workspace
        settings={'model':'fixture','vision':True}
        root=work.start('execute','episode','goal',settings,False,None)
        steps=[{'id':'move','title':'持瓶移动','checkpoint':'motion_reached','skill':'move_relative'}]
        work.plan(root,'episode','fixture',steps);work.bind(root,'run','execute','goal','pick','pick','fixture')
        state.cache['last_run']={'execution_id':'run','status':'primitive_completed','run_id':'run'}
        def refusal(reason,epoch='episode',name='move_relative'):
            event=work.event(root,'tool',name)
            work.end_event(event,{'result':{'failure_kind':'observation_refresh','motion_started':False,
                'revalidation':{'accepted':False,'reason':reason,'source_observation_id':epoch+':0','current_observation_id':epoch+':30'}}},'error')
        args={'observation_id':'episode:0','verdict':'uncertain','reason':'verified source drift prevented motion'}
        with patch.object(agent,'_log'):
            agent.invoke('observe',{})
            refusal('scene_drift',epoch='old');refusal('scene_drift',name='inspect_manipulation');refusal('stale_observation')
            self.assertFalse(work.rejected_motion_feedback(root,'episode',{'move_relative'}))
            refusal('scene_drift')
            with self.assertRaisesRegex(ValueError,'计划动作尚未完成'):agent.invoke('finish',args)
            refusal('holding_state_changed')
            result=agent.invoke('finish',args)
            self.assertTrue(result['assessment_recorded']);self.assertEqual(result['reported_verdict'],'uncertain')
            self.assertEqual(result['report_evidence']['early_stop_cause'],'repeated_observation_drift')
            self.assertEqual(work.active_plan['verdict'],'uncertain')
            with self.assertRaisesRegex(ValueError,'计划动作尚未完成'):agent.invoke('finish',{**args,'verdict':'success'})

    def test_final_goal_checks_current_position_and_target_not_legacy_basket_score(self):
        s=snapshot();goal=resolve_placement({'object_id':'bottle','surface':'table','offset_xy_m':[.08,0]},s)
        s['diagnostics']['evaluation']={'success':True,'placement':{'confirmed':True,'surface':'table','xy_m':[.48,-.17]}}
        self.assertFalse(assess_placement(goal,s)['confirmed'])
        s['diagnostics']['evaluation']={'success':False,'touching':{'left':False,'right':False},
            'placement':{'confirmed':True,'surface':'table','xy_m':goal['target_xy_m'],'tolerance_m':goal['tolerance_m'],'stable_seconds':1.1}}
        self.assertFalse(assess_placement(goal,s)['confirmed'])
        s['diagnostics']['manipulation_scene']['position_m']=[.575,-.17,1.005]
        self.assertTrue(assess_placement(goal,s)['confirmed'])
        s['diagnostics']['evaluation']['invalid']=True
        self.assertFalse(assess_placement(goal,s)['confirmed'])

    def test_relative_goal_does_not_move_after_handover_and_replanning(self):
        s=snapshot();spec={'object_id':'bottle','surface':'table','offset_xy_m':[.08,0]}
        first=resolve_placement(spec,s)
        s['observation_id']='episode:300';s['diagnostics']['manipulation_scene']['position_m']=[.44,.06,1.15]
        after=resolve_placement(spec,s,first)
        self.assertEqual(first,after)
        args={'observation_id':'episode:300','object_id':'bottle','receptacle_id':'table','arm':'right'}
        self.assertEqual(bind_place(args,after)['target_xy_m'],[.58,-.17])
        self.assertEqual(bind_place({**args,'offset_xy_m':[.08,0]},after)['target_xy_m'],[.58,-.17])
        with self.assertRaises(ValueError):bind_place({**args,'target_xy_m':[.52,.06]},after)
        with self.assertRaises(ValueError):resolve_placement({**spec,'offset_xy_m':[.02,0]},s,first)

    def test_achieved_stages_cannot_be_erased_or_relabelled_by_replan(self):
        previous={'steps':[{'id':'pick','title':'左手抓起','checkpoint':'grasp_confirmed','skill':'pick','status':'done'},
                           {'id':'place','title':'直接放置','checkpoint':'placement_confirmed','skill':'place','status':'blocked'}]}
        requested=[{'id':'pick','title':'右手放入篮','checkpoint':'tool_completed'},
                   {'id':'pass','title':'交给右手','checkpoint':'handover_confirmed','skill':'handover'},
                   {'id':'place','title':'右手放置','checkpoint':'placement_confirmed','skill':'place'}]
        merged=preserve_completed(previous,requested)
        self.assertEqual(merged[0]['title'],'左手抓起');self.assertEqual(len(merged),4)
        self.assertEqual(merged[1]['id'],'pick_replan1')
        self.assertNotIn('status',merged[0])

    def test_model_renumbering_keeps_done_grasp_and_adds_new_handover(self):
        previous={'steps':[{'id':'s1','title':'观察','checkpoint':'observation','status':'done'},
            {'id':'s2','title':'左手抓起','checkpoint':'grasp_confirmed','skill':'pick','status':'done'}]}
        requested=[{'id':'s1','title':'抓起','checkpoint':'grasp_confirmed','skill':'pick'},
            {'id':'s2','title':'交给右手','checkpoint':'handover_confirmed','skill':'handover'}]
        result=preserve_completed(previous,requested)
        self.assertEqual(len(result),3);self.assertEqual(result[-1]['skill'],'handover')
        self.assertEqual(result[-1]['id'],'s2_replan1')

    def test_retry_receipt_replaces_failed_attempt_without_completing_next_repeated_skill(self):
        work=Workspace();s=snapshot()
        steps=[{'id':'a','title':'第一段','checkpoint':'motion_reached','skill':'move_relative','occurrence':1},
               {'id':'b','title':'第二段','checkpoint':'motion_reached','skill':'move_relative','occurrence':2}]
        work.plan('t','episode','test',steps)
        def bind(ident,feedback):
            work.bind('t',ident,'execute','move','move_relative','move_relative','test')
            work.record_feedback(ident,'execute','episode','move','move_relative',{},feedback)
            s['last_run']={'execution_id':ident,'status':'primitive_completed'}
        bind('fail',{'primitive_success':False,'failure_code':'tracking_error'})
        feedback={'primitive_success':True,'verification_source':'robot_forward_kinematics','position_error_m':.001,'rotation_error_rad':.002}
        bind('retry',feedback)
        status=work.plan_status(s)
        self.assertEqual(status['steps'][0]['status'],'done');self.assertNotEqual(status['steps'][1]['status'],'done')
        bind('next',feedback)
        self.assertTrue(all(step['status']=='done' for step in work.plan_status(s)['steps']))
        self.assertEqual(work.active_plan['stage_bindings'],{'a':'retry','b':'next'})

    def test_wrong_native_target_cannot_complete_bound_placement_phase(self):
        work=Workspace();s=snapshot();goal=resolve_placement({'object_id':'bottle','surface':'table','offset_xy_m':[.08,0]},s)
        steps=[{'id':'place','title':'放到前方','checkpoint':'placement_confirmed','skill':'place'}]
        work.plan('t','episode','test',steps,placement_goal=goal);work.bind('t','run','execute','forward','place','place','test')
        s['last_run']={'execution_id':'run','status':'primitive_completed'}
        f={'primitive_success':True,'verification_source':'native_support_release_and_stability','position_error_m':.005,
           'placement':{'confirmed':True,'surface':'table','xy_m':[.48,-.17],'tolerance_m':goal['tolerance_m']}}
        work.record_feedback('run','execute','episode','forward','place',{},f)
        self.assertNotEqual(work.plan_status(s)['steps'][0]['status'],'done')
        f['placement']['xy_m']=goal['target_xy_m'];work.record_feedback('run','execute','episode','forward','place',{},f)
        self.assertEqual(work.plan_status(s)['steps'][0]['status'],'done')
        f['placement']['tolerance_m']=.025;work.record_feedback('run','execute','episode','forward','place',{},f)
        self.assertNotEqual(work.plan_status(s)['steps'][0]['status'],'done')

    def test_model_requires_explicit_oracle_and_valid_goal_before_place_plan(self):
        state=LocalState();state.cache=snapshot();agent=Harness(state)
        context={'type':'llm','turn_id':'t','plan_turn_id':'t','model':'test'}
        steps=[{'id':'place','title':'放到前方','checkpoint':'placement_confirmed','skill':'place'}]
        with patch.object(agent,'_log'):
            agent.invoke('observe',{})
            with self.assertRaisesRegex(ValueError,'placement_goal'):agent.invoke('set_plan',{'observation_id':'episode:0','steps':steps},context=context)
            args={'observation_id':'episode:0','steps':steps,'placement_goal':{'object_id':'bottle','surface':'table','offset_xy_m':[.08,0]}}
            with self.assertRaisesRegex(ValueError,'oracle'):agent.invoke('set_plan',args,context=context)
            agent.invoke('inspect_manipulation',{'observation_id':'episode:0'})
            agent.invoke('set_plan',args,context=context)
            self.assertEqual(agent.workspace.active_plan['placement_goal']['target_xy_m'],[.58,-.17])

if __name__=='__main__':unittest.main()
