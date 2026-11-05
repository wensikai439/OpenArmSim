"""Behavioral memory regressions: deletion, outcome banks and frozen receipts."""
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from .agent import Harness
from .checks import LocalState
from .workspace import Workspace
from sim.service import memory_context


CONTEXT = {'robot':'v1', 'contract':'geometry-a', 'policy_contract':'policy-a'}


class MemoryChecks(unittest.TestCase):
    def test_generic_failure_codes_do_not_merge_different_causes(self):
        work=Workspace()
        for ident,message in [('range','gripper output outside verified range'),('identity','checkpoint identity mismatch')]:
            work.record_feedback(ident,'execute','e','策略运行','run_policy',{'backend':'act'},
                {'status':'error','failure_code':'start_rejected','message':message},CONTEXT)
        recalled=work.memory('execute','e',context=CONTEXT,skill='run_policy',compact=True)
        self.assertEqual(len(recalled['knowledge']['index']),2)
        self.assertEqual({x['failure_code'] for x in recalled['knowledge']['index']},{'policy_gripper_range','policy_identity_mismatch'})

    def test_duplicate_receipts_do_not_crowd_distinct_failures_out_of_retrieval(self):
        work = Workspace()
        work.record_feedback('distinct','execute','e','抓取','pick',{},
                             {'failure_code':'goal_collision','planning_diagnostics':{'blocked_pairs':['table/finger']}},CONTEXT)
        for i in range(160):
            work.record_feedback('repeat-'+str(i),'execute','e','不同的长日志提示 '+str(i),'pick',{},
                                 {'failure_code':'grasp_lost'},CONTEXT)
        recalled = work.memory('execute','e',context=CONTEXT,skill='pick',compact=True)
        self.assertEqual(len(recalled['knowledge']['index']),2)
        repeated = next(e for e in recalled['knowledge']['index'] if e['failure_code']=='grasp_lost')
        self.assertEqual(repeated['repeat_count'],160)
        self.assertTrue(any(e['failure_code']=='goal_collision' for e in recalled['knowledge']['index']))
        raw = work.memory('execute','e',public=True,context=CONTEXT)
        self.assertEqual(len(raw['experiences']),161)
        self.assertEqual(len(raw['knowledge']['lessons']),2)
        self.assertTrue(all('summary' in item for item in raw['knowledge']['lessons']))
        self.assertEqual(work.db.execute('SELECT count(*) FROM experiences').fetchone()[0],161)

    def test_filtered_oracle_evidence_is_not_reported_as_missing_or_exposed_by_default(self):
        work = Workspace()
        evidence = {'before':{'observation_id':'e:0'}, 'after':{'observation_id':'e:1'},
                    'camera_keyframes':[{'sensor_urls':{'head':'/runs/private/head.jpg'}}]}
        work.record_feedback('oracle','execute','e','移动','move_relative',{},
                             {'primitive_success':True,'private_pose':[.1,.2,1.]},CONTEXT,
                             'simulator_oracle', evidence=evidence)
        filtered = work.memory('execute','e',context=CONTEXT,skill='move_relative',compact=True)
        self.assertFalse(filtered['experiences'])
        self.assertNotIn('private_pose',json.dumps(filtered))
        self.assertNotIn('/runs/private/head.jpg',json.dumps(filtered))
        self.assertIn('does NOT mean evidence was not saved',filtered['retrieval']['evidence_policy'])
        audit = work.memory('execute','e',context=CONTEXT,skill='move_relative',include_oracle=True,compact=True)
        self.assertFalse(audit['experiences'])  # one tiny movement is not durable knowledge
        with self.assertRaises(ValueError): work.execution_receipt('execute','oracle',context=CONTEXT)
        receipt = work.execution_receipt('execute','oracle',context=CONTEXT,include_oracle=True)
        self.assertEqual(receipt['evidence']['before']['observation_id'],'e:0')
        self.assertEqual(receipt['evidence']['after']['observation_id'],'e:1')
        self.assertEqual(receipt['evidence']['camera_keyframes'],evidence['camera_keyframes'])
        self.assertEqual(receipt['evidence_kind'],'simulator_oracle')
        self.assertEqual(work.memory('execute','e',public=True)['experiences'][0]['feedback']['private_pose'],[.1,.2,1.])

    def test_preference_edit_is_in_place_and_cannot_relabel_another_role_or_scene(self):
        work = Workspace()
        preference = work.save_note('execute', '旧偏好')
        other = work.save_note('research', '其他职责')
        scene = work.save_note('execute', '模型估计', scope='scene', epoch='e', source='model')
        before = work.db.execute('SELECT * FROM notes WHERE id=?', (preference['id'],)).fetchone()
        edited = work.update_preference('execute', preference['id'], ' 新偏好 ')
        after = work.db.execute('SELECT * FROM notes WHERE id=?', (preference['id'],)).fetchone()
        self.assertEqual(edited['id'], preference['id'])
        self.assertEqual(after['text'], '新偏好')
        for key in ('id', 'role', 'scope', 'source', 'created', 'context', 'evidence_kind'):
            self.assertEqual(after[key], before[key])
        for ident in (other['id'], scene['id'], 'missing'):
            with self.assertRaises(ValueError):
                work.update_preference('execute', ident, '不得替换')
        with self.assertRaises(ValueError):
            work.update_preference('execute', preference['id'], '')
        self.assertEqual(work.db.execute('SELECT COUNT(*) FROM notes').fetchone()[0], 3)
        self.assertEqual(work.memory('research', 'e')['notes'][0]['text'], '其他职责')

    def test_preference_edit_waits_for_inflight_work_and_clears_only_its_history(self):
        agent = Harness(LocalState())
        saved = agent.save_preference('execute', '旧偏好')
        agent.history['execute'].append({'role':'assistant', 'content':'引用旧偏好'})
        agent.history['research'].append({'role':'assistant', 'content':'其他职责'})
        agent.chat_lock.acquire()
        try:
            with self.assertRaisesRegex(ValueError, '任务进行中'):
                agent.save_preference('execute', '新偏好', saved['id'])
        finally:
            agent.chat_lock.release()
        self.assertEqual(agent.workspace.memory('execute', 'e')['notes'][0]['text'], '旧偏好')
        agent.pending_review = {'execution_id':'pending'}
        with self.assertRaisesRegex(ValueError, '任务进行中'):
            agent.save_preference('execute', '新偏好', saved['id'])
        self.assertFalse(agent.chat_lock.locked())
        agent.pending_review = None
        agent.pending_operations['pending'] = {'execution_id':'pending'}
        with self.assertRaisesRegex(ValueError, '任务进行中'):
            agent.save_preference('execute', '新偏好', saved['id'])
        with self.assertRaisesRegex(ValueError, '任务进行中'):
            agent.delete_memory('execute', saved['id'])
        agent.pending_operations.clear()
        result = agent.save_preference('execute', '新偏好', saved['id'])
        self.assertEqual(result['id'], saved['id'])
        self.assertEqual(len(agent.workspace.memory('execute', 'e')['notes']), 1)
        self.assertFalse(agent.history['execute'])
        self.assertTrue(agent.history['research'])

    def test_memory_compatibility_ignores_harness_tests_but_tracks_geometry_and_checkpoint(self):
        manifest = {'schema':'sim/1','engine':'mujoco','profile':{'camera':1},'physics_dt':.001,
                    'contract':{'action':'joint16'},'hashes':{'sim/environment.py':'a','task/cell.xml':'a',
                        'harness/workspace.py':'a','sim/timing_checks.py':'a','server.py':'a','sim/tamp.py':'a','sim/controller.py':'a',
                        'sim/cli.py':'a','sim/curobo_server.py':'a','sim/act_server.py':'a'}}
        before = memory_context(manifest, {'checkpoint':'one'}, 'v1')
        unrelated = {**manifest, 'hashes':{**manifest['hashes'],'harness/workspace.py':'b','sim/timing_checks.py':'b','server.py':'b','sim/tamp.py':'b',
            'sim/cli.py':'b','sim/curobo_server.py':'b','sim/act_server.py':'b'},
                     'source_status':{'restart_required':True}}
        self.assertEqual(before,memory_context(unrelated,{'checkpoint':'one'},'v1'))
        geometry = {**manifest,'hashes':{**manifest['hashes'],'task/cell.xml':'b'}}
        self.assertNotEqual(before['contract'],memory_context(geometry,{'checkpoint':'one'},'v1')['contract'])
        controller={**manifest,'hashes':{**manifest['hashes'],'sim/controller.py':'b'}}
        self.assertNotEqual(before['contract'],memory_context(controller,{'checkpoint':'one'},'v1')['contract'])
        checkpoint = memory_context(manifest,{'checkpoint':'two'},'v1')
        self.assertEqual(before['contract'],checkpoint['contract'])
        self.assertNotEqual(before['policy_contract'],checkpoint['policy_contract'])

    def test_native_context_migration_requires_matching_source_and_preserves_forgetting(self):
        from tools.migrate_memory import migrate,eligible
        old={'contract':'old','policy_contract':'same'};new={'contract':'new','policy_contract':'same'}
        hashes={p:'same' for p in ('sim/environment.py','sim/contacts.py','sim/gripper.py','sim/profile.json',
                                  'calibrate.json','task/bottle_to_box.xml','sim/tamp.py')}
        manifest={'schema':'sim','engine':'same','profile':{},'physics_dt':.001,'contract':{},'hashes':hashes}
        source={**manifest,'kind':'tamp'};work=Workspace()
        feedback={'primitive_success':True,'verification_source':'native_two_finger_contact_and_lift'}
        work.record_feedback('native','execute','e','pick','pick',{'arm':'right'},feedback,old,'simulator_oracle',evidence={'run_id':'native'})
        work.record_feedback('changed','execute','e','pick','pick',{'arm':'left'},feedback,old,'simulator_oracle',evidence={'run_id':'changed'})
        work.save_note('execute','preference',source='user')
        # Select the exact right-hand lesson, independent of recency ordering.
        ident=next(x['id'] for x in work.memory('execute','e',context=old,compact=True)['knowledge']['index'] if x['title'].endswith('右手'))
        work.delete_memory('execute',ident)
        def load(run):return (source if run=='native' else {**source,'hashes':{**hashes,'sim/contacts.py':'changed'}},feedback)
        result=migrate(work,old,new,manifest,load)
        self.assertEqual(result['migrated'],['native']);self.assertEqual(len(result['skipped']),1)
        self.assertEqual(result['forgetting_markers_preserved'],1)
        self.assertFalse(work.memory('execute','e',context=new,compact=True)['knowledge']['index'])
        self.assertEqual(work.db.execute('SELECT feedback FROM experiences WHERE id="native"').fetchone()[0],json.dumps(feedback,ensure_ascii=False))
        self.assertEqual(work.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0],2)
        self.assertEqual(len(work.memory('execute','e',context=new,compact=True)['notes']),1)
        self.assertFalse(eligible({**source,'hashes':{**hashes,'sim/tamp.py':'different'}},manifest))
        with self.assertRaises(ValueError):migrate(work,old,{**new,'policy_contract':'changed'},manifest,load)
        work.record_feedback('mismatch','execute','e','pick','pick',{'arm':'right'},feedback,old,'simulator_oracle',evidence={'run_id':'mismatch'})
        rejected=migrate(work,old,new,manifest,lambda run:(source,{**feedback,'primitive_success':False}))
        self.assertFalse(rejected['migrated'])
        self.assertTrue(any(x['reason']=='archived native feedback does not match receipt' for x in rejected['skipped']))

    def test_compact_model_memory_keeps_failure_and_evidence_without_mutating_archive(self):
        work = Workspace()
        for i in range(8):
            work.record_feedback(str(i),'execute','e','抓取瓶子','pick',{},
                {'failure_code':'goal_collision','message':'blocked','planning_diagnostics':{'blocked_pairs':['finger/table']},
                 'unused_trace':[{'long_debug':'x'*2000} for _ in range(15)]},CONTEXT,
                evidence={'before':{'observation_id':'e:0','state':[1.]*16},'after':{'observation_id':'e:1','state':[2.]*16},
                          'camera_keyframes':[{'sensor_urls':{'head':'/runs/check/head.jpg'}}]})
        full = work.memory('execute','e',context=CONTEXT)
        compact = work.memory('execute','e',context=CONTEXT,compact=True)
        self.assertLess(len(json.dumps(compact)),len(json.dumps(full))//3)
        self.assertEqual(len(compact['knowledge']['index']),1)
        item=compact['knowledge']['index'][0]
        self.assertEqual(item['failure_code'],'goal_collision')
        self.assertIn('关闭碰撞检查',item['summary'])
        self.assertNotIn('camera_keyframes',json.dumps(compact))
        detail=work.knowledge_detail('execute','e',item['id'],context=CONTEXT)
        self.assertTrue(detail['sources'][0]['evidence']['camera_keyframes'])
        self.assertIn('unused_trace',work.memory('execute','e',public=True)['experiences'][0]['feedback'])

    def test_clear_removes_working_copies_and_keeps_other_role(self):
        work = Workspace()
        for role in ('execute','research'):
            work.save_note(role, role+' preference')
            work.observe_working(role,'e',{'observation_id':'e:0'},CONTEXT)
            work.record_feedback(role,role,'e','pick bottle','pick',{}, {'failure_code':'grasp_lost'},CONTEXT)
            work.save_objects(role,'e','e:0',[{'object_id':'bottle','state':'visible'}],CONTEXT)
        work.delete_memory('execute')
        memory = work.memory('execute','e',context=CONTEXT)
        self.assertFalse(memory['notes']); self.assertFalse(memory['experiences'])
        self.assertFalse(memory['objects']); self.assertFalse(memory['capabilities'])
        self.assertFalse(memory['working'].get('observation')); self.assertFalse(memory['working'].get('failed_attempts'))
        self.assertEqual(len(work.memory('research','e',context=CONTEXT)['experiences']),1)

    def test_delete_one_receipt_prunes_derived_recipe_without_erasing_observation(self):
        work = Workspace(); work.observe_working('execute','e',{'observation_id':'e:4'},CONTEXT)
        for ident in ('one','two'):
            work.bind('root',ident,'execute','bottle','pick','pick','m',CONTEXT)
            work.complete(ident,'run-'+ident)
            work.record_feedback(ident,'execute','e','bottle','pick',{}, {'failure_code':'grasp_lost'},CONTEXT)
        steps = [{'skill':s} for s in ('pick','handover','place')]
        recipe = work.save_recipe('execute','bottle',steps,CONTEXT,'run-one')
        work.save_recipe('execute','bottle',steps,CONTEXT,'run-two')
        work.delete_memory('execute','one')
        memory = work.memory('execute','e',context=CONTEXT,include_oracle=True)
        self.assertEqual(memory['working']['observation']['observation_id'],'e:4')
        self.assertEqual([a['execution_id'] for a in memory['working']['actions']],['two'])
        self.assertEqual([a['execution_id'] for a in memory['working']['failed_attempts']],['two'])
        self.assertEqual(memory['recipes'][0]['id'],recipe)
        self.assertEqual(memory['recipes'][0]['body']['validated_runs'],['run-two'])
        work.delete_memory('execute','two')
        self.assertFalse(work.memory('execute','e',include_oracle=True)['recipes'])

    def test_harness_clear_forgets_prompt_history_and_rejects_inflight_clear(self):
        agent = Harness(LocalState())
        agent.history['execute'].append({'role':'assistant','content':'old remembered data'})
        agent.history['research'].append({'role':'assistant','content':'other role'})
        agent.workspace.save_note('execute','one')
        agent.chat_lock.acquire()
        try:
            with self.assertRaisesRegex(ValueError,'任务进行中'): agent.delete_memory('execute')
            self.assertEqual(len(agent.workspace.memory('execute','e')['notes']),1)
        finally: agent.chat_lock.release()
        agent.delete_memory('execute')
        self.assertFalse(agent.history['execute']); self.assertTrue(agent.history['research'])

    def test_outcomes_do_not_score_cancelled_policy_exit_or_model_claim(self):
        work = Workspace()
        cases = [('good',{'primitive_success':True},'sensor_execution','success'),
                 ('bad',{'failure_code':'grasp_lost'},'sensor_execution','failure'),
                 ('cancel',{'status':'cancelled','failure_code':'cancelled'},'sensor_execution','cancelled'),
                 ('exit',{'status':'execution_finished'},'sensor_execution','unverified'),
                 ('claim',{'primitive_success':True},'model_hypothesis','unverified'),
                 ('reject',{'status':'rejected','failure_code':'tool_contract_rejected'},'contract_guard','rejected')]
        for ident,feedback,source,_ in cases:
            work.record_feedback(ident,'execute','e','bottle','pick',{},feedback,CONTEXT,source)
        rows = work.memory('execute','e',public=True,context=CONTEXT)['experiences']
        self.assertEqual({r['id']:r['outcome'] for r in rows},{ident:expected for ident,_,_,expected in cases})
        profile = work.capability_profile('execute',context=CONTEXT)[0]
        self.assertEqual(profile['scored_attempts'],2); self.assertEqual(profile['observed_success_fraction'],.5)
        self.assertEqual(profile['counts']['unverified'],2)

    def test_old_compatible_success_is_not_crowded_out_by_new_failures_or_versions(self):
        work = Workspace()
        work.record_feedback('old-success','execute','old','抓取瓶子','pick',{}, {'primitive_success':True},CONTEXT)
        work.bind('old','old-episode','execute','抓取瓶子','pick','pick','m',CONTEXT)
        for i in range(330):
            work.record_feedback('failure-'+str(i),'execute','e','抓取瓶子','pick',{}, {'failure_code':'grasp_lost'},CONTEXT)
            work.record_feedback('wrong-'+str(i),'execute','e','抓取瓶子','pick',{}, {'primitive_success':True},{'contract':'wrong'})
            work.bind('new','wrong-episode-'+str(i),'execute','抓取瓶子','pick','pick','m',{'contract':'wrong'})
        memory = work.memory('execute','e',query='抓取瓶子',context=CONTEXT,skill='pick')
        self.assertIn('old-success',[e['id'] for e in memory['experiences']])
        self.assertTrue(any(e['outcome']=='failure' for e in memory['experiences']))
        self.assertLessEqual(len(memory['experiences']),5)
        self.assertEqual([e['id'] for e in memory['episodes']],['old-episode'])

    def test_skill_and_privilege_filters_apply_before_candidate_limits(self):
        work = Workspace()
        work.record_feedback('sensor','execute','old','抓取瓶子','pick',{}, {'failure_code':'grasp_lost'},CONTEXT)
        for i in range(320):
            work.record_feedback(str(i),'execute','e','抓取瓶子','pick',{},
                                 {'failure_code':'PRIVATE_ORACLE_FAILURE'},CONTEXT,'simulator_oracle')
            work.record_feedback('motion-'+str(i),'execute','e','抓取瓶子','move_relative',{}, {'primitive_success':True},CONTEXT)
        memory = work.memory('execute','e',query='抓取瓶子',context=CONTEXT,skill='pick')
        self.assertEqual([e['id'] for e in memory['experiences']],['sensor'])
        self.assertNotIn('PRIVATE_ORACLE_FAILURE',json.dumps(memory))
        self.assertEqual(memory['capabilities'][0]['counts']['failure'],1)

    def test_policy_revision_and_backend_samples_never_merge(self):
        work = Workspace()
        for backend in ('act','pi05'):
            work.record_feedback(backend,'execute','e','交接','run_policy',{'backend':backend,'skill_id':'handover'},
                                 {'status':'execution_finished'},CONTEXT)
        profiles = work.capability_profile('execute',context=CONTEXT)
        self.assertEqual({p['backend'] for p in profiles},{'act','pi05'})
        self.assertTrue(all(p['observed_success_fraction'] is None for p in profiles))
        self.assertFalse(work.memory('execute','e',context={**CONTEXT,'policy_contract':'policy-b'})['experiences'])

    def test_old_database_migrates_without_promoting_policy_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'memory.sqlite3'
            db = sqlite3.connect(path)
            db.execute('CREATE TABLE experiences (id TEXT PRIMARY KEY,role TEXT,epoch TEXT,goal TEXT,tool TEXT,arguments TEXT,context TEXT,evidence_kind TEXT,feedback TEXT,created REAL)')
            for ident,feedback in (('skill',{'primitive_success':True}),('policy',{'status':'execution_finished'})):
                db.execute('INSERT INTO experiences VALUES (?,?,?,?,?,?,?,?,?,?)',
                           (ident,'execute','e','goal','pick','{}','{}','sensor_execution',json.dumps(feedback),1.))
            db.commit(); db.close()
            work = Workspace(path)
            self.assertEqual({e['id']:e['outcome'] for e in work.memory('execute','e',public=True)['experiences']},
                             {'skill':'success','policy':'unverified'})

    def test_rejected_nonfinite_argument_cannot_poison_later_memory_json(self):
        state = LocalState(); agent = Harness(state); requests = []
        agent.configure_model({'base_url':'http://127.0.0.1:1234/v1','model':'test','api_key':'placeholder','vision':False})
        def response(request, **kwargs):
            requests.append(json.loads(request.data))
            msg = {'role':'assistant','content':'参数无效，动作未启动。'}
            if len(requests)==1:
                msg = {'role':'assistant','tool_calls':[{'id':'bad','type':'function','function':{
                    'name':'move_relative','arguments':json.dumps({'observation_id':'episode:0','arm':'right',
                        'delta_m':[float('nan'),0,0],'frame':'world'})}}]}
            class Reply:
                def __enter__(self): return self
                def __exit__(self,*args): pass
                def read(self,_): return json.dumps({'choices':[{'message':msg}]}).encode()
            return Reply()
        with patch('urllib.request.urlopen',side_effect=response),patch.object(agent,'_log'):
            result = agent.chat('检验无效移动参数')
        self.assertTrue(result['tools'][0]['result']['tool_failed']); self.assertFalse(state.calls)
        memory = agent.workspace.memory('execute','episode')
        encoded = json.dumps(memory,allow_nan=False)
        self.assertNotIn('NaN',encoded)
        self.assertEqual(memory['experiences'][0]['outcome'],'rejected')
        self.assertFalse(memory['experiences'][0]['evidence']['motion_started'])

    def test_reset_before_callback_keeps_old_terminal_evidence_without_resuming(self):
        state = LocalState(with_queue=True); agent = Harness(state)
        try:
            preference = agent.save_preference('execute', '保留用户偏好')
            original_feedback = agent.workspace.record_feedback
            guards = []
            def archive_feedback(*args, **kwargs):
                # The UI can issue a clear/edit after pending_review has
                # cleared but before this late completion reaches SQLite.
                for action in (lambda: agent.delete_memory('execute', preference['id']),
                               lambda: agent.save_preference('execute', '错误覆盖', preference['id'])):
                    try:
                        action()
                    except ValueError:
                        guards.append('blocked')
                    else:
                        guards.append('allowed')
                return original_feedback(*args, **kwargs)
            root = agent.workspace.start('execute','episode','抓瓶子',{'model':'m','vision':True})
            agent.workspace.bind(root,'old','execute','抓瓶子','pick','pick','m',CONTEXT)
            pending = {'epoch':'episode','execution_id':'old','prompt':'抓瓶子','turn_id':root,'root':root,
                       'tool':'pick','arguments':{},'context':CONTEXT,'continuation':True,
                       'before_observation':{'observation_id':'episode:0'}}
            agent.pending_review = pending; agent.pending_operations['old'] = pending
            agent.sessions[root] = {'failed':[],'sequence':[]}
            state.cache.update(observation_id='new:0',last_run={'execution_id':'other','status':'success',
                'metrics':{'tamp':{'primitive_success':True}}},memory_context=CONTEXT)
            with patch.object(agent,'_log'),patch.object(agent,'chat') as chat, \
                 patch.object(agent.workspace,'record_feedback',side_effect=archive_feedback):
                agent.invoke('observe',{})
                state.completed.put({'epoch':'episode','execution_id':'old','run_id':'old-run',
                    'result':{'execution_id':'old','status':'error','summary_url':'/runs/old-run/summary.json',
                              'metrics':{'tamp':{'primitive_success':False,'failure_code':'grasp_lost'}}},
                    'observation':{'observation_id':'episode:9','tick':9},'evaluation':{'success':False}})
                deadline = time.monotonic()+2
                while not agent.workspace.memory('execute','new',include_oracle=True)['experiences'] and time.monotonic()<deadline:
                    time.sleep(.01)
                receipt = agent.workspace.memory('execute','new',include_oracle=True)['experiences'][0]
                self.assertEqual(receipt['feedback']['failure_code'],'grasp_lost')
                self.assertEqual(receipt['evidence']['after']['observation_id'],'episode:9')
                self.assertEqual(guards, ['blocked', 'blocked'])
                self.assertEqual(agent.workspace.memory('execute','new')['notes'][0]['text'], '保留用户偏好')
                self.assertFalse(agent.workspace.memory('execute','new',include_oracle=True)['objects'])
                chat.assert_not_called()
        finally:
            agent.close(); agent.review_thread.join(1)


if __name__ == '__main__': unittest.main()
