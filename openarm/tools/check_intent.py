"""Original-role, recovery, native-evidence and task-isolation regressions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness.intent import extract, progress, check_action, native_events, proposed_events, install_isolated
from harness.task_queue import TaskWorkspace


def receipt(kind, *, arm=None, donor=None, receiver=None, ident='bottle', success=True, source=None, execution='e'):
    sources = {'pick':'native_two_finger_contact_and_lift',
        'handover':'native_receiver_contact_then_donor_release', 'place':'native_support_release_and_stability'}
    args = {'object_id': ident}; feedback = {'object_id': ident, 'status':'completed' if success else 'error',
        'primitive_success': success, 'verification_source': source or sources[kind]}
    if kind == 'handover':
        args.update(from_arm=donor, to_arm=receiver); feedback.update(args, held_by=receiver)
    else:
        args['arm'] = arm; feedback.update(arm=arm)
        if kind == 'pick': feedback['held_by'] = arm
        if kind == 'place': feedback['placement'] = {'confirmed':True}
    return {'skill':kind, 'execution_id':execution, 'arguments':args,
        'feedback':feedback, 'evidence_kind':'simulator_oracle'}


class RoleChecks(unittest.TestCase):
    def test_original_source_quotes_and_roles_in_both_directions(self):
        for first, second in [('左手','右手'), ('右手','左手')]:
            prompt=f'用{first}抓起瓶子交给{second}，由{second}放到桌面。'
            c=extract(prompt); kinds=[r['skill'] for r in c['requirements']]
            self.assertEqual(kinds,['pick','handover','place'])
            self.assertEqual(c['requirements'][0]['arm'],'left' if first=='左手' else 'right')
            for r in c['requirements']:
                a,b=r['source_range'];self.assertEqual(r['source_quote'],prompt[a:b])

    def test_free_arm_choice_and_explicit_placement_arm(self):
        unbound=extract('把瓶子向前放8厘米。')
        self.assertEqual(unbound['requirements'],[])
        for side in ('left','right'):check_action(unbound,progress(unbound,[]),{'skill':'pick','object_id':'bottle','arm':side})
        bound=extract('用右手把瓶子向机器人正前方放8厘米。')
        self.assertEqual(bound['requirements'][0]['skill'],'place')
        with self.assertRaises(ValueError):check_action(bound,progress(bound,[]),{'skill':'pick','object_id':'bottle','arm':'left'})

    def test_negative_quoted_unknown_and_ambiguous_text_not_guessed(self):
        for prompt in ['不要用左手抓起瓶子。','比如“左手抓起瓶子”，只是举例。','用左手抓起一个苹果。',
            '用左手抓起瓶子和杯子。','把瓶子从左手拿起到右手。',
            '如果瓶子在桌面，用左手抓起瓶子，否则用右手。']:
            self.assertEqual(extract(prompt)['requirements'],[],prompt)

    def test_registered_multi_object_order_and_protected_object(self):
        prompt='先用右手把圆柱can向前放8厘米，然后用左手夹起长方体block，交给右手，由右手放在圆柱原位。瓶子不动。'
        c=extract(prompt)
        self.assertEqual([(r['object_id'],r['skill']) for r in c['requirements']],
            [('can','place'),('block','pick'),('block','handover'),('block','place')])
        self.assertEqual(c['protected_objects'],['bottle'])
        self.assertEqual({r['object_id'] for r in extract('用左手抓起短圆柱can_short。')['requirements']},{'can_short'})
        with self.assertRaises(ValueError):check_action(c,progress(c,[]),{'skill':'pick','object_id':'bottle','arm':'right'})
        with self.assertRaises(ValueError):check_action(c,progress(c,[]),{'skill':'pick','object_id':'block','arm':'left'})

    def test_destination_container_never_becomes_the_picked_or_placed_item(self):
        for prompt in ['用右手把瓶子从杯子里抓起，右手把瓶子放在圆柱原位。',
            'With right hand grasp the bottle, right hand place the bottle in the cup.']:
            c=extract(prompt);self.assertEqual([r['object_id'] for r in c['requirements']],['bottle','bottle'],prompt)
        c=extract('用右手把瓶子和杯子抓起。');self.assertEqual(c['requirements'],[])

    def test_failed_pick_wrong_hand_and_reverse_transfer_never_satisfy_original(self):
        c=extract('用左手抓起瓶子交给右手。')
        old=[receipt('pick',arm='left',success=False),receipt('pick',arm='right'),
            receipt('handover',donor='right',receiver='left'),receipt('handover',donor='left',receiver='right')]
        self.assertEqual(progress(c,old)['verified_requirements'],0)
        with self.assertRaises(ValueError):check_action(c,progress(c,old),{'skill':'pick','object_id':'bottle','arm':'right'})
        with self.assertRaises(ValueError):check_action(c,progress(c,old),{'skill':'handover','object_id':'bottle','from_arm':'left','to_arm':'right'})

    def test_successful_prefix_allows_same_donor_regrasp_and_required_receiver(self):
        c=extract('用左手抓起瓶子交给右手，右手把瓶子放到桌面。')
        records=[receipt('pick',arm='left')];state=progress(c,records)
        check_action(c,state,{'skill':'pick','object_id':'bottle','arm':'left'})
        with self.assertRaises(ValueError):check_action(c,state,{'skill':'pick','object_id':'bottle','arm':'right'})
        records.append(receipt('handover',donor='left',receiver='right'));state=progress(c,records)
        check_action(c,state,{'skill':'pick','object_id':'bottle','arm':'right'})
        with self.assertRaises(ValueError):check_action(c,state,{'skill':'place','object_id':'bottle','arm':'left'})
        records.append(receipt('place',arm='right'));self.assertEqual(progress(c,records)['remaining'],[])

    def test_phase_flags_model_verdict_sensor_termination_wrong_object_not_receipts(self):
        base=receipt('pick',arm='left')
        variants=[]
        for change in [{'evidence_kind':'sensor_execution'}, {'feedback':{'primitive_success':True,'status':'completed','phase':'grasp_confirmed','reported_verdict':'success'}}]:
            variants.append({**base,**change})
        for key,value in [('object_id','cup'),('held_by','right'),('verification_source','model_visual_claim'),('status','cancelled')]:
            r=copy.deepcopy(base);r['feedback'][key]=value;variants.append(r)
        for r in variants:self.assertEqual(native_events(r),[])

    def test_composite_completed_child_survives_outer_failure_without_fabricating_rest(self):
        good=receipt('pick',arm='left');failed=receipt('handover',donor='left',receiver='right',success=False)
        r={'skill':'relocate','execution_id':'outer','arguments':{},'evidence_kind':'simulator_oracle',
            'feedback':{'status':'error','substeps':[{'skill':row['skill'],'status':row['feedback']['status'],
                'parameters':row['arguments'],'feedback':row['feedback']} for row in [good,failed]]}}
        c=extract('用左手抓起瓶子交给右手。');state=progress(c,[r])
        self.assertEqual(state['verified_requirements'],1);self.assertEqual(state['verified_events'][0]['execution_id'],'outer')
        self.assertEqual(state['remaining'][0]['skill'],'handover')

    def test_composite_and_policy_proposals_have_no_success_evidence(self):
        c=extract('用左手抓起瓶子交给右手。');state=progress(c,[])
        with self.assertRaises(ValueError):
            for event in proposed_events('run_policy',{'backend':'act'}):check_action(c,state,event)
        good=proposed_events('relocate',{'object_id':'bottle','arm':'left','via_arm':'right'})
        self.assertEqual([e['skill'] for e in good],['pick','handover','place'])
        self.assertEqual(progress(c,[])['verified_requirements'],0)


class HarnessChecks(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.workspace=TaskWorkspace(Path(tmp.name)/'test.sqlite3');self.addCleanup(self.workspace.db.close)
        class Base:
            def __init__(owner):
                owner.workspace=self.workspace;owner.sessions={'root':{'goal':'用左手抓起瓶子交给右手。'}};owner.called=[]
            def _invoke(owner,name,args,role,context=None):
                owner.called.append((name,args))
                if name=='set_plan':owner.workspace.plan('root','epoch','test',[{'id':'x','title':'测试','skill':'pick','checkpoint':'grasp_confirmed'}])
                return {'accepted':True}
        self.harness=install_isolated(Base)();self.context={'type':'llm','turn_id':'review','plan_turn_id':'root','model':'test'}

    def bind(self,record,*,root='root',epoch='epoch',after='epoch:1'):
        w=self.workspace
        if not w.active_plan:w.plan(root,epoch,'test',[])
        w.bind(root,record['execution_id'],'execute','original',record['skill'],'task','test')
        w.record_feedback(record['execution_id'],'execute',epoch,'original',record['skill'],record['arguments'],record['feedback'],
            evidence_kind=record['evidence_kind'],evidence={'after':{'observation_id':after}})

    def test_wrong_role_rejected_before_dispatch_and_manual_demo_unchanged(self):
        with self.assertRaises(ValueError):self.harness._invoke('pick',{'object_id':'bottle','arm':'right'},'execute',self.context)
        self.assertEqual(self.harness.called,[])
        self.harness._invoke('run_tamp',{'task':'handover_to_basket'},'execute',None)
        self.assertEqual(self.harness.called[0][0],'run_tamp')

    def test_review_prompt_model_titles_and_replanning_cannot_rewrite_task_contract(self):
        r=self.harness._invoke('set_plan',{'steps':[{'title':'右手抓取'}]},'execute',self.context)
        self.assertEqual(r['user_requirements']['contract']['requirements'][0]['arm'],'left')
        c=copy.deepcopy(self.workspace.active_plan['user_requirements']['contract'])
        self.harness._invoke('set_plan',{'steps':[{'title':'右手恢复'}]},'execute',self.context)
        self.assertEqual(self.workspace.active_plan['user_requirements']['contract'],c)
        row=self.workspace.db.execute('SELECT body FROM plans WHERE turn_id="root"').fetchone()
        self.assertEqual(json.loads(row['body'])['user_requirements']['contract'],c)

    def test_only_bound_same_task_receipts_and_matching_epoch_advance(self):
        r=receipt('pick',arm='left');self.bind(r,after='other:1')
        self.assertEqual(self.harness._role_records('root'),[])
        self.bind(receipt('pick',arm='left',execution='good'))
        self.assertEqual(progress(extract(self.harness.sessions['root']['goal']),self.harness._role_records('root'))['verified_requirements'],1)
        self.assertEqual(self.harness._role_records('another-root'),[])

    def test_success_gate_uses_original_roles_and_failure_report_remains_available(self):
        self.bind(receipt('pick',arm='right'))
        with self.assertRaises(ValueError):self.harness._invoke('finish',{'verdict':'success'},'execute',self.context)
        self.assertEqual(self.harness.called,[])
        r=self.harness._invoke('finish',{'verdict':'failure'},'execute',self.context)
        self.assertEqual(r['user_requirements']['verified_requirements'],0)

    def test_correct_receipts_allow_finish_and_new_goal_has_its_own_contract(self):
        self.bind(receipt('pick',arm='left'));self.bind(receipt('handover',donor='left',receiver='right',execution='relay'))
        r=self.harness._invoke('finish',{'verdict':'success'},'execute',self.context)
        self.assertEqual(r['user_requirements']['verified_requirements'],2)
        self.harness.sessions['next']={'goal':'用右手抓起瓶子。'}
        self.harness._invoke('pick',{'object_id':'bottle','arm':'right'},'execute',{**self.context,'plan_turn_id':'next'})
        self.assertEqual(self.harness.sessions['next']['user_role_contract']['requirements'][0]['arm'],'right')

    def test_compatible_policy_termination_does_not_create_role_verification(self):
        self.harness.sessions['root']['goal']='用右手抓起瓶子交给左手。'
        self.harness._invoke('run_policy',{'backend':'act'},'execute',self.context)
        policy={'skill':'run_policy','execution_id':'policy','arguments':{'backend':'act'},'evidence_kind':'sensor_execution',
            'feedback':{'status':'execution_finished'}}
        self.bind(policy)
        r=self.harness._invoke('finish',{'verdict':'success'},'execute',self.context)
        self.assertEqual(r['user_requirements']['verified_requirements'],0)
        self.assertIn('UNKNOWN',r['user_requirements']['learned_policy_role_verification'])

    def test_object_switch_preserves_prior_verified_requirements_without_old_task_reuse(self):
        from tools.check_task_api import PROMPT
        self.harness.sessions['root']['goal']=PROMPT
        self.bind(receipt('place',arm='right',ident='can',execution='can'),epoch='old',after='old:1')
        self.workspace.active_plan['epoch']='new'
        self.bind(receipt('pick',arm='left',ident='block',execution='block'),epoch='new',after='new:1')
        state=progress(extract(PROMPT),self.harness._role_records('root'))
        self.assertEqual(state['verified_requirements'],2)
        self.assertEqual(state['remaining'][0]['object_id'],'block')
        self.assertEqual(state['remaining'][0]['skill'],'handover')


class NativeChecks(unittest.TestCase):
    def test_wrong_role_tools_rejected_before_any_native_state_or_clock_change(self):
        from tools.object_workbench import build_backend
        from harness.poses import PoseWorkspace
        import numpy as np
        factory, Harness=build_backend();service=factory();workspace=PoseWorkspace();harness=None
        try:
            harness=Harness(service,workspace=workspace)
            snapshot=service.snapshot();prompt='用左手抓起瓶子交给右手。'
            root=workspace.start('execute',snapshot['observation_id'].split(':')[0],prompt,{'model':'offline-role-probe','vision':True})
            harness.sessions[root]={'goal':prompt,'initial_snapshot':snapshot,'actions':0}
            context={'type':'llm','model':'offline-role-probe','turn_id':root,'plan_turn_id':root}
            observed=harness.invoke('observe',{},context=context)
            harness.invoke('set_plan',{'observation_id':observed['observation_id'],'time_budget_s':120,'timing_mode':'measured_delay',
                'steps':[{'id':'pick','title':'左手抓起','checkpoint':'grasp_confirmed','skill':'pick'},
                    {'id':'relay','title':'交给右手','checkpoint':'handover_confirmed','skill':'handover'}]},context=context)
            before=service.snapshot();physical=(service.env.data.qpos.copy(),service.env.data.qvel.copy(),service.env.data.ctrl.copy(),service.env.data.time)
            calls=[('pick',{'object_id':'bottle','arm':'right'}),
                ('handover',{'object_id':'bottle','from_arm':'right','to_arm':'left'}),
                ('run_policy',{'backend':'act','skill_id':'atomic_right_to_left_handover'}),
                ('run_tamp',{'task':'handover_to_basket'}),
                ('relocate',{'object_id':'bottle','arm':'right','receptacle_id':'table'})]
            for name,args in calls:
                with self.assertRaisesRegex(ValueError,'原始用户要求'):
                    harness.invoke(name,{'observation_id':observed['observation_id'],**args},context=context)
                for old,new in zip(physical[:3],(service.env.data.qpos,service.env.data.qvel,service.env.data.ctrl)):
                    np.testing.assert_array_equal(old,new)
                self.assertEqual(service.env.data.time,physical[3]);self.assertIsNone(service.snapshot()['job'])
            self.assertEqual(service.snapshot()['observation_id'],before['observation_id'])
            self.assertEqual(harness.sessions[root]['actions'],0)
            self.assertEqual(workspace.active_plan.get('execution_tools',[]),[])
            print('native role probe: five rejected proposals, zero commands/clock changes',flush=True)
        finally:
            if harness:harness.close()
            service.close();workspace.db.close()


if __name__=='__main__':unittest.main()
