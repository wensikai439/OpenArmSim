"""Regressions for a real empty-arm API failure and original-pose identity."""
import copy
import json
from unittest import TestCase, main

from sim.environment import ROOT
from .stages import pending_action, require_next_action, start_anchor, route_tools, adapt_context, bind_atomic_choice


class StageChecks(TestCase):
    def setUp(self):
        record=json.loads((ROOT/'runs/20261009-context/api-arm/result.json').read_text())
        self.steps=copy.deepcopy(record['terminal']['agent']['plan']['steps'])
        for step in self.steps:step['status']='pending'
        self.steps[3]['skill']='move_to_pose'
        self.plan={'steps':self.steps}

    def test_completed_relative_does_not_block_absolute_return(self):
        for step in self.steps[:3]:step['status']='done'
        require_next_action(self.plan,'move_to_pose')
        self.assertEqual(pending_action(self.plan)['id'],'back')

    def test_gripper_close_cannot_be_skipped(self):
        for step in self.steps[:2]:step['status']='done'
        with self.assertRaisesRegex(ValueError,'set_gripper'):require_next_action(self.plan,'move_to_pose')
        require_next_action(self.plan,'set_gripper')

    def test_no_repeated_up_after_completed_up(self):
        self.steps[0]['status']='done'
        with self.assertRaises(ValueError):require_next_action(self.plan,'move_relative')

    def test_failed_stage_cannot_be_silently_skipped(self):
        self.steps[0]['status']='failed'
        with self.assertRaises(ValueError):require_next_action(self.plan,'set_gripper')

    def test_finished_robot_actions_require_review(self):
        for step in self.steps[:4]:step['status']='done'
        self.assertIsNone(pending_action(self.plan))
        with self.assertRaises(ValueError):require_next_action(self.plan,'move_to_pose')

    def test_gate_is_read_only(self):
        original=copy.deepcopy(self.plan);pending_action(self.plan)
        with self.assertRaises(ValueError):require_next_action(self.plan,'move_to_pose')
        self.assertEqual(original,self.plan)

    def test_start_anchor_retains_root_pose_not_later_pose(self):
        session={'initial_snapshot':{'observation_id':'original:0','observation':{'tcp':{'right':{'position_m':[.3,-.4,1.18]}}}}}
        original=copy.deepcopy(session);anchor=start_anchor(session,'original')
        self.assertEqual(anchor['tcp'],original['initial_snapshot']['observation']['tcp'])
        anchor['tcp']['right']['position_m'][2]=1.21
        self.assertEqual(session,original)

    def test_wrong_epoch_never_reuses_start_pose(self):
        session={'initial_snapshot':{'observation_id':'old:0','observation':{'tcp':{'right':{}}}}}
        self.assertIsNone(start_anchor(session,'new'))

    def tools(self):
        return [{'type':'function','function':{'name':name,'description':'original',
            'parameters':{'constant':'unchanged'}}} for name in
            ('observe','get_run_status','set_plan','finish','stop','move_relative','move_to_pose',
             'set_gripper','pick','handover','place','run_policy','run_tamp')]

    def test_pending_gripper_reveals_gripper_not_return_or_finish(self):
        for step in self.steps[:2]:step['status']='done'
        visible,meta=route_tools(self.tools(),self.plan)
        names=[t['function']['name'] for t in visible]
        self.assertIn('set_gripper',names)
        self.assertNotIn('move_to_pose',names);self.assertNotIn('finish',names)
        self.assertEqual(meta['next_stage']['id'],'close')

    def test_pending_return_reveals_absolute_pose_not_completed_relative(self):
        for step in self.steps[:3]:step['status']='done'
        visible,meta=route_tools(self.tools(),self.plan)
        names=[t['function']['name'] for t in visible]
        self.assertIn('move_to_pose',names);self.assertNotIn('move_relative',names)
        self.assertFalse(meta['finish_available'])

    def test_failure_reporting_stays_available_with_pending_stage(self):
        visible,meta=route_tools(self.tools(),self.plan,can_report_failure=True)
        self.assertIn('finish',[t['function']['name'] for t in visible]);self.assertTrue(meta['finish_available'])

    def test_finished_stages_reveal_review_and_hide_all_motion(self):
        for step in self.steps[:4]:step['status']='done'
        visible,meta=route_tools(self.tools(),self.plan)
        names=[t['function']['name'] for t in visible]
        self.assertIn('finish',names);self.assertNotIn('move_to_pose',names);self.assertNotIn('run_tamp',names)

    def test_initial_model_can_choose_its_own_plan_and_tools(self):
        original=self.tools();visible,meta=route_tools(original,None)
        self.assertEqual(visible,original);self.assertEqual(meta['hidden_tools'],[])

    def test_observation_replanning_stop_and_schemas_preserved(self):
        original=self.tools();before=copy.deepcopy(original)
        visible,meta=route_tools(original,self.plan)
        for name in ('observe','get_run_status','set_plan','stop'):
            self.assertEqual(next(t for t in visible if t['function']['name']==name),next(t for t in original if t['function']['name']==name))
        self.assertEqual(visible[-1]['function']['parameters'],{'constant':'unchanged'})
        self.assertEqual(original,before)

    def test_live_ledger_keeps_original_system_and_all_other_payload_fields(self):
        payload={'model':'same','max_tokens':1600,'tools':self.tools(),'messages':[
            {'role':'system','content':'original physical limits'},
            {'role':'user','content':[{'type':'image_url','image_url':{'url':'data:unchanged'}}]},
            {'role':'assistant','tool_calls':[{'id':'bound','function':{'name':'set_gripper','arguments':'{"opening_fraction":0}'}}]},
            {'role':'tool','tool_call_id':'bound','content':'original receipt'}]}
        before=copy.deepcopy(payload);after=adapt_context(payload,{'next_stage':{'id':'close'}})
        self.assertEqual(payload,before);self.assertTrue(after['messages'][0]['content'].startswith(payload['messages'][0]['content']))
        self.assertEqual(after['messages'][1:],payload['messages'][1:])
        for key in ('model','max_tokens','tools'):self.assertEqual(after[key],payload[key])
        self.assertIn('has NOT completed',after['messages'][0]['content'])

    def test_no_current_plan_keeps_packet_exact(self):
        payload={'messages':[{'role':'system','content':'original'}]}
        self.assertEqual(adapt_context(payload,None),payload)

    def test_missing_system_context_rejects_injection(self):
        with self.assertRaises(ValueError):adapt_context({'messages':[{'role':'user','content':'original'}]}, {'next_stage':{}})

    def test_old_text_dialogue_is_replaced_by_bound_live_ledger(self):
        payload={'messages':[{'role':'system','content':'original'},
            {'role':'user','content':'old task text'},{'role':'assistant','content':'old action started'},
            {'role':'user','content':[{'type':'text','text':'original goal\nCurrent observation:\nfresh'},
                {'type':'image_url','image_url':{'url':'data:unchanged'}}]},
            {'role':'assistant','tool_calls':[{'id':'this-turn'}]},
            {'role':'tool','tool_call_id':'this-turn','content':'this-turn receipt'}]}
        after=adapt_context(payload,{'next_stage':{'id':'close'}},focused=True)
        self.assertEqual(after['messages'][1:],payload['messages'][3:])
        self.assertEqual(len(payload['messages']),6)

    def test_unknown_tool_pair_before_marker_prevents_history_pruning(self):
        payload={'messages':[{'role':'system','content':'original'},
            {'role':'assistant','tool_calls':[{'id':'old'}]},
            {'role':'tool','tool_call_id':'old','content':'original receipt'},
            {'role':'user','content':'goal\nCurrent observation:\nfresh'}]}
        after=adapt_context(payload,{},focused=True)
        self.assertEqual(after['messages'][1:],payload['messages'][1:])

    def test_gripper_stage_hides_unrelated_pose_resolvers_only_in_focused_mode(self):
        self.steps[0]['status']='done'
        tools=self.tools()+[{'function':{'name':name,'description':'original','parameters':{}}}
            for name in ('resolve_scene_target','resolve_placement_transit')]
        visible,_=route_tools(tools,self.plan,focused=True)
        self.assertNotIn('resolve_scene_target',[t['function']['name'] for t in visible])
        self.assertIn('set_gripper',[t['function']['name'] for t in visible])

    def test_named_gripper_choice_changes_no_arguments_or_existing_payload(self):
        payload={'tools':self.tools(),'messages':[{'role':'user','content':'original task'}],'max_tokens':1600}
        before=copy.deepcopy(payload);ledger={'next_stage':{'skill':'set_gripper'},'native_job_active':False}
        after=bind_atomic_choice(payload,ledger)
        self.assertEqual(payload,before);self.assertEqual(after['tool_choice'],{'type':'function','function':{'name':'set_gripper'}})
        del after['tool_choice'];self.assertEqual(after,payload)

    def test_named_choice_never_forces_pose_or_failure_recovery(self):
        payload={'tools':self.tools()}
        for ledger in ({'next_stage':{'skill':'move_to_pose'}},
            {'next_stage':{'skill':'set_gripper'},'failure_reporting_available':True},
            {'next_stage':{'skill':'set_gripper'},'native_job_active':True}):
            self.assertEqual(bind_atomic_choice(payload,ledger),payload)

    def test_absent_tool_or_existing_choice_is_preserved(self):
        ledger={'next_stage':{'skill':'set_gripper'}}
        for payload in ({'tools':[]},{'tools':self.tools(),'tool_choice':'none'}):
            self.assertEqual(bind_atomic_choice(payload,ledger),payload)


if __name__=='__main__':main()
