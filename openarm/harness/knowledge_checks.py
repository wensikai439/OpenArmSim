"""Knowledge quality, retrieval boundaries and forgetting without evidence loss."""
import json
import unittest
from .workspace import Workspace

C={'robot':'v1','contract':'candidate-a','policy_contract':'one'}

class KnowledgeChecks(unittest.TestCase):
    def test_relocation_index_shares_rules_and_records_only_confirmed_routes(self):
        work=Workspace()
        def record(ident,arm,offset,via=None,backend='tamp',failed_place=False):
            args={'arm':arm,'receptacle_id':'table','offset_xy_m':offset,'transport_planner':backend}
            steps=[{'skill':'pick','status':'completed','backend':'tamp','parameters':{'arm':arm}}]
            if failed_place:
                steps.append({'skill':'place','status':'failed','parameters':{'arm':arm}})
            if via:
                args['via_arm']=via
                steps.append({'skill':'handover','status':'completed','backend':'tamp',
                    'parameters':{'from_arm':arm,'to_arm':via}})
            if backend=='curobo':
                steps.append({'skill':'move_relative','status':'completed','backend':backend,
                    'parameters':{'arm':via or arm,'delta_m':[.03,0,0]}})
            steps.append({'skill':'place','status':'completed','backend':'tamp',
                'parameters':{'arm':via or arm,'receptacle_id':'table','target_xy_m':[.583,-.173]},
                'feedback':{'primitive_success':True,'verification_source':'native_support_release_and_stability'}})
            work.record_feedback(ident,'execute','e','duplicate prompt','relocate',args,
                {'primitive_success':True,'substeps':steps,'decision_timing_mode':'measured_delay',
                 'placement':{'confirmed':True},'verification_source':'native_support_release_and_stability'},
                C,'simulator_oracle')
        record('one','right',[.08,0]);record('two','right',[.06,0])
        record('transfer','left',[.06,0],via='right',failed_place=True)
        record('gpu','right',[.08,0],backend='curobo')
        work.record_feedback('fake','execute','e','claim','relocate',{'arm':'right'},
            {'primitive_success':True,'placement':{'confirmed':True},
             'verification_source':'native_support_release_and_stability',
             'substeps':[{'skill':'pick','status':'completed','parameters':{'arm':'right'}}]},C,'simulator_oracle')
        memory=work.memory('execute','e',context=C,compact=True)
        knowledge=memory['knowledge'];index=knowledge['index']
        self.assertEqual(len(index),3)
        self.assertEqual(len(knowledge['protocols']),1)
        self.assertEqual(len({i['summary'] for i in index}),3)
        self.assertEqual(max(i['repeat_count'] for i in index),2)
        self.assertTrue(any('左手交给右手 → 右手桌面放置' in i['summary'] for i in index))
        self.assertTrue(any('cuRobo携物运动' in i['summary'] for i in index))
        self.assertFalse(any('左手桌面放置' in i['summary'] for i in index))
        encoded=json.dumps(memory,ensure_ascii=False)
        for raw in ('target_xy_m','delta_m','offset_xy_m','duplicate prompt','fake'):
            self.assertNotIn(raw,encoded)
        self.assertEqual(work.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0],5)

    def test_one_method_lists_timing_conditions_without_importing_raw_records(self):
        work=Workspace()
        for i,mode in enumerate(('paused','paused','measured_delay')):
            work.record_feedback(str(i),'execute','e','pick','pick',{'arm':'right'},
                {'primitive_success':True,'decision_timing_mode':mode,
                 'verification_source':'native_two_finger_contact_and_lift','raw_pose':[.503,-.173,1.005]},C,'simulator_oracle')
        memory=work.memory('execute','e',context=C,compact=True)
        self.assertEqual(len(memory['knowledge']['index']),1)
        item=memory['knowledge']['index'][0]
        self.assertEqual(item['timing_conditions'],{'paused':2,'measured_delay':1})
        self.assertIn('未证明长期夹持',item['evidence_scope'])
        self.assertNotIn('raw_pose',json.dumps(memory))
        detail=work.knowledge_detail('execute','e',item['id'],context=C,include_oracle=True)
        self.assertEqual(detail['timing_conditions'],item['timing_conditions'])
        self.assertEqual(work.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0],3)

    def test_gripper_experience_has_own_summary_and_failure_causes_are_bounded(self):
        work=Workspace()
        work.record_feedback('jaw','execute','e','闭合右手','set_gripper',{'arm':'right'},
            {'primitive_success':True,'verification_source':'robot_gripper_encoder_settling'},C)
        for i in range(12):
            work.record_feedback(str(i),'execute','e','release','place',{'arm':'right','receptacle_id':'table'},
                {'failure_code':'release_pose_not_reached','primitive_success':False,'arbitrary_trace':'a'*5000},C)
        memory=work.memory('execute','e',context=C,skill='set_gripper',compact=True)
        self.assertEqual(len(memory['knowledge']['index']),1)
        self.assertIn('闭合不等于',memory['knowledge']['index'][0]['summary'])
        memory=work.memory('execute','e',context=C,skill='place',compact=True)
        self.assertEqual(len(memory['knowledge']['index']),1)
        self.assertEqual(memory['knowledge']['index'][0]['repeat_count'],12)
        self.assertNotIn('arbitrary_trace',json.dumps(memory))

    def test_targets_merge_into_one_procedure_and_no_pose_or_trace_enters_index(self):
        work=Workspace()
        for i in range(15):
            work.record_feedback(str(i),'execute','e','long duplicate prompt'*80,'move_relative',
                {'planner':'curobo','arm':'right','delta_m':[.001*i,0,.01],'frame':'world'},
                {'primitive_success':True,'position_error_m':.001,'private_pose':[7,8,9]},C,
                'simulator_oracle',evidence={'camera_keyframes':[{'sensor_urls':{'head':'/runs/raw/head.jpg'}}]})
        result=work.memory('execute','e',context=C,compact=True)
        index=result['knowledge']['index'];self.assertEqual(len(index),1)
        self.assertEqual(index[0]['repeat_count'],15)
        encoded=json.dumps(result);self.assertNotIn('delta_m',encoded);self.assertNotIn('private_pose',encoded)
        self.assertNotIn('/runs/raw/',encoded);self.assertNotIn('long duplicate prompt',encoded)
        self.assertFalse(result['experiences']);self.assertFalse(result['episodes'])
        with self.assertRaises(ValueError):work.knowledge_detail('execute','e',index[0]['id'],context=C)
        self.assertTrue(work.knowledge_detail('execute','e',index[0]['id'],context=C,include_oracle=True)['sources'])
        self.assertFalse(work.memory('execute','e',context={**C,'contract':'different'},compact=True)['knowledge']['index'])

    def test_one_small_move_and_model_verdict_do_not_become_knowledge(self):
        work=Workspace()
        work.record_feedback('small','execute','e','move','move_relative',{}, {'primitive_success':True},C)
        work.record_feedback('claim','execute','e','claim','handover',{}, {'verdict':'success'},C,'model_hypothesis')
        self.assertFalse(work.memory('execute','e',context=C,compact=True)['knowledge']['index'])
        self.assertEqual(work.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0],2)

    def test_directions_stay_separate_and_forgotten_lesson_does_not_reappear(self):
        work=Workspace()
        for donor,receiver in [('left','right'),('right','left')]:
            work.record_feedback(donor,'execute','e','transfer','handover',{'from_arm':donor,'to_arm':receiver},
                {'primitive_success':True,'held_by':receiver,'verification_source':'native_receiver_contact_then_donor_release'},C)
        index=work.memory('execute','e',context=C,compact=True)['knowledge']['index']
        self.assertEqual(len(index),2);self.assertNotEqual(index[0]['title'],index[1]['title'])
        work.delete_memory('execute',index[0]['id'])
        self.assertEqual(len(work.memory('execute','e',context=C,compact=True)['knowledge']['index']),1)
        self.assertEqual(work.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0],2)
        public=work.public('execute','e',C)['memory'];self.assertFalse(public['experiences']);self.assertFalse(public['episodes'])

    def test_placement_surface_and_failures_retain_useful_distinctions(self):
        work=Workspace()
        for surface in ('table','basket'):
            work.record_feedback(surface,'execute','e','place','place',{'arm':'right','receptacle_id':surface},
                {'primitive_success':True},C)
        work.record_feedback('blocked','execute','e','place','place',{'arm':'left','receptacle_id':'table'},
            {'failure_code':'placement_unreachable'},C)
        index=work.memory('execute','e',context=C,compact=True)['knowledge']['index']
        self.assertEqual(len(index),3)
        self.assertTrue(any('交给另一只手' in x['summary'] for x in index))
        self.assertEqual(sum('桌面' in x['title'] for x in index),2)

    def test_clear_memory_allows_new_evidence_to_be_learned(self):
        work=Workspace()
        def add():work.record_feedback('pick','execute','e','pick','pick',{'arm':'right'},{'primitive_success':True},C)
        add();ident=work.memory('execute','e',context=C,compact=True)['knowledge']['index'][0]['id']
        work.delete_memory('execute',ident);work.delete_memory('execute');add()
        self.assertEqual(len(work.memory('execute','e',context=C,compact=True)['knowledge']['index']),1)

if __name__=='__main__':unittest.main()
