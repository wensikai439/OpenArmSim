"""Plan provenance, execution binding, durable traces and memory isolation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .workspace import Workspace, validate_plan, redact
from .agent import Harness
from .checks import LocalState


STEPS = [{'id':'pick','title':'抓起瓶子','checkpoint':'right_lift'},
         {'id':'basket','title':'放入篮子','checkpoint':'basket'},
         {'id':'review','title':'视觉复查','checkpoint':'visual_review'}]
SETTINGS = {'model':'test-model','vision':False}


class WorkspaceChecks(unittest.TestCase):
    def test_requested_unsupported_or_absent_object_never_becomes_a_bottle(self):
        from .intent import extract, require_available_objects
        clean = {'scene_library': {'selected': []}}
        with self.assertRaisesRegex(ValueError, '尚未支持 cup'):
            require_available_objects(extract('左手拿起杯子'), clean)
        with self.assertRaisesRegex(ValueError, '当前场景没有 block'):
            require_available_objects(extract('右手抓起方块'), clean)
        require_available_objects(extract('右手抓起方块'), {'scene_library': {'selected': ['block']}})
        require_available_objects(extract('左手拿起瓶子'), clean)

    def model_request(self, results, *, code=None, budget=1, deadline=120):
        from unittest.mock import Mock
        from urllib.error import HTTPError
        import urllib.request
        agent = object.__new__(Harness)
        agent.workspace = Mock()
        agent.service = Mock()
        agent.request_transport = Mock()
        reply = Mock()
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = json.dumps({'choices': [{'message': {'content': 'ready'}}]}).encode()
        agent.request_transport.urlopen.side_effect = [HTTPError('https://example.test', x, 'error', {}, None)
                                                       if isinstance(x, int) else reply for x in results]
        session = {'llm_calls': budget}
        clock = [0.]
        def advance(seconds): clock[0] += seconds
        with patch('harness.agent.time.monotonic', side_effect=lambda: clock[0]), \
                patch('harness.agent.time.sleep', side_effect=advance):
            def run():
                return agent._model_reply(urllib.request.Request('https://example.test'), deadline=deadline,
                    turn='task', settings={'model': 'test', 'base_url': 'https://example.test'},
                    request_number=1, vision=True, session=session)
            if code is not None:
                with self.assertRaises(HTTPError) as caught: run()
                self.assertEqual(caught.exception.code, code)
            else:
                self.assertEqual(run(), {'content': 'ready'})
        agent.service.call.assert_not_called()
        return agent, session, clock[0]

    def test_transient_http_retry_is_audited_without_robot_tool_replay(self):
        agent, session, elapsed = self.model_request([503, None])
        self.assertEqual(agent.request_transport.urlopen.call_count, 2)
        self.assertEqual(session['api_retries'], 1)
        self.assertEqual(session['llm_calls'], 1)
        self.assertAlmostEqual(elapsed, 1.)
        attempts = [call for call in agent.workspace.event.call_args_list if call.args[1] == 'api']
        self.assertEqual(len(attempts), 2)

    def test_auth_balance_and_invalid_requests_are_never_retried(self):
        for code in (400, 401, 402, 403):
            with self.subTest(code=code):
                agent, session, _ = self.model_request([code], code=code)
                self.assertEqual(agent.request_transport.urlopen.call_count, 1)
                self.assertNotIn('api_retries', session)

    def test_transient_retries_remain_bounded_by_attempts_budget_and_deadline(self):
        for results, budget, deadline, count in (([503, 503, 503], 1, 120, 3),
                                                ([503], 32, 120, 1), ([503], 1, .5, 1)):
            with self.subTest(count=count, budget=budget, deadline=deadline):
                agent, session, _ = self.model_request(results, code=503, budget=budget, deadline=deadline)
                self.assertEqual(agent.request_transport.urlopen.call_count, count)

    def test_idle_composite_plan_cannot_match_an_absent_job_id(self):
        from .bottle import BottleWorkspace
        from .composite import validate_composite_plan
        work = BottleWorkspace()
        try:
            steps = validate_composite_plan({'steps': [
                {'id': 'pick', 'title': '抓取', 'checkpoint': 'grasp_confirmed',
                 'skill': 'relocate', 'subskill': 'pick', 'sub_occurrence': 1}]})
            work.plan('t', 'episode', 'test-model', steps)
            state = LocalState().snapshot()
            state['job'] = None
            plan = work.plan_status(state)
            self.assertEqual(plan['steps'][0]['status'], 'pending')
            self.assertIsNone(plan['execution_id'])
        finally:
            work.db.close()

    def test_orphan_wait_cleanup_requires_this_request_and_no_native_job(self):
        from unittest.mock import Mock
        agent = object.__new__(Harness)
        snapshot = {'state': 'running', 'job': None,
                    'decision_timing': {'root': 'task-a', 'request_id': 3}}
        agent.service = Mock()
        agent.service.snapshot.return_value = snapshot
        agent._release_orphan_decision('task-b', 3)
        agent._release_orphan_decision('task-a', 2)
        agent.service.call.assert_not_called()
        agent._release_orphan_decision('task-a', 3)
        self.assertEqual(agent.service.call.call_args_list[0].args, ('stop',))
        agent.service.call.assert_called_with('end_decision', root='task-a', request_id=3)
        agent.service.call.reset_mock()
        snapshot['job'] = {'execution_id': 'native-motion'}
        agent._release_orphan_decision('task-a', 3)
        agent.service.call.assert_not_called()

    def test_deployed_bottle_workspace_preserves_native_request_contract(self):
        from .bottle import BottleWorkspace
        work = BottleWorkspace()
        try:
            turn = work.start('execute', 'episode', '移动右手', SETTINGS)
            work.plan(turn, 'episode', 'test-model', [
                {'id': 'move', 'title': '移动右手', 'checkpoint': 'motion_reached', 'skill': 'move_to_pose'}])
            contract = {'task_id': 'demo-task', 'root_execution_id': turn, 'epoch': 'episode',
                        'observation_id': 'episode:0', 'backend_id': 'curobo',
                        'target': {'arm': 'right', 'position_m': [.35, -.4, 1.2]}}
            work.bind(turn, 'move-1', 'execute', '移动右手', 'move_to_pose', 'move_to_pose',
                      'test-model', request_contract=contract)
            contract['target']['position_m'][0] = 99
            row = work.db.execute('SELECT request_contract,task_id FROM episodes WHERE id=?', ('move-1',)).fetchone()
            stored = json.loads(row['request_contract'])
            self.assertEqual(row['task_id'], 'demo-task')
            self.assertEqual(stored['observation_id'], 'episode:0')
            self.assertEqual(stored['target']['position_m'], [.35, -.4, 1.2])
            self.assertEqual(work.active_plan['stage_bindings']['move'], 'move-1')
            self.assertEqual(work.active_plan['execution_tools'][0]['request_contract'], stored)
        finally:
            work.db.close()

    def test_read_only_batch_is_sequenced_and_mutation_batch_never_runs(self):
        def call(name,args,ident):
            return {'id':ident,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}
        for batch,expected in (([call('get_run_status',{},'a'),call('observe',{},'b')],2),
                               ([call('run_tamp',{'task':'pick_bottle','observation_id':'episode:0'},'a'),call('observe',{},'b')],0)):
            state=LocalState();agent=Harness(state)
            agent.configure_model({'base_url':'http://127.0.0.1:1234/v1','model':'test','api_key':'placeholder','vision':False})
            requests=[]
            def response(request,**kwargs):
                payload=json.loads(request.data);requests.append(payload)
                msg={'role':'assistant','tool_calls':batch} if len(requests)==1 else {'role':'assistant','content':'已读取结果。'}
                if len(requests)==2:
                    ids=[m['tool_call_id'] for m in payload['messages'] if m['role']=='tool']
                    self.assertEqual(ids,['a','b'])
                class Reply:
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def read(self,_):return json.dumps({'choices':[{'message':msg}]}).encode()
                return Reply()
            with patch('urllib.request.urlopen',side_effect=response),patch.object(agent,'_log'):
                result=agent.chat('查看场景')
            self.assertEqual(len(result['tools']),expected)
            self.assertFalse(state.calls)
            self.assertEqual(len(requests),2)

    def test_plan_requires_model_context_and_cannot_set_own_success(self):
        agent = Harness(LocalState())
        with patch.object(agent,'_log'):
            agent.invoke('observe',{})
            with self.assertRaises(ValueError):
                agent.invoke('set_plan',{'observation_id':'episode:0','steps':STEPS})
            with self.assertRaisesRegex(ValueError,'set_plan'):
                agent.invoke('run_tamp',{'observation_id':'episode:0','task':'pick_bottle'},
                             context={'type':'llm','turn_id':'t','model':'m'})
            for steps in ([{**STEPS[0],'status':'done'}], [STEPS[0],STEPS[0]], [], [{'id':'a','title':'a','checkpoint':'invented'}]):
                with self.assertRaises(ValueError):validate_plan({'steps':steps})
            agent.invoke('set_plan',{'observation_id':'episode:0','steps':STEPS}, context={'turn_id':'t','model':'m'})
            result=agent.invoke('run_tamp',{'observation_id':'episode:0','task':'pick_bottle'}, context={'type':'llm','turn_id':'t','model':'m'})
            self.assertEqual(result['state'],'started')

    def test_steps_follow_matching_execution_evidence_not_model_claims(self):
        work=Workspace();state=LocalState().snapshot()
        work.plan('t','episode','m',copy.deepcopy(STEPS))
        state['diagnostics']['events']=[{'stage':2},{'stage':6}]
        self.assertEqual([s['status'] for s in work.plan_status(state)['steps']],['pending']*3)
        work.bind('t','job-a','execute','goal','run_tamp','handover_to_basket','m')
        state['last_run'].update(execution_id='job-a',status='success')
        self.assertEqual([s['status'] for s in work.plan_status(state)['steps']],['done','done','pending'])
        work.assess('wrong-job','success')
        self.assertEqual(work.plan_status(state)['steps'][-1]['status'],'pending')
        work.assess('job-a','uncertain')
        self.assertEqual(work.plan_status(state)['steps'][-1]['status'],'uncertain')
        state['observation_id']='reset:0'
        self.assertIsNone(work.plan_status(state))

    def test_durable_records_restore_as_interrupted_without_resuming(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'agent.sqlite3';one=Workspace(path)
            turn=one.start('execute','old','task',SETTINGS)
            one.event(turn,'api','request',{'model':'test-model'})
            one.save_note('execute','简短回答')
            one.plan(turn,'old','m',copy.deepcopy(STEPS))
            two=Workspace(path);public=two.public('execute','new')
            self.assertEqual(public['turns'][0]['status'],'interrupted')
            self.assertEqual(public['turns'][0]['trace'][0]['status'],'interrupted')
            self.assertEqual(public['memory']['notes'][0]['text'],'简短回答')
            self.assertIsNone(two.active_plan)
            self.assertFalse(two.busy)

    def test_memory_expires_scene_notes_isolates_roles_and_can_delete(self):
        work=Workspace()
        work.save_note('research','RESEARCH_ONLY')
        work.save_note('execute','old coordinates',scope='scene',epoch='old',source='model')
        saved=work.save_note('execute','请简短回答')
        work.bind('t','job','execute','goal','run_tamp','pick_bottle','m')
        work.complete('job','run-a');work.assess('job','uncertain')
        current=work.memory('execute','new')
        self.assertNotIn('RESEARCH_ONLY',json.dumps(current))
        self.assertNotIn('old coordinates',json.dumps(current))
        self.assertEqual(current['episodes'][0]['verdict'],'uncertain')
        self.assertNotIn('simulator_success',json.dumps(current))
        self.assertEqual(len(work.memory('execute','old')['notes']),2)
        work.delete_memory('execute',saved['id'])
        self.assertEqual(work.memory('execute','new')['notes'],[])
        work.delete_memory('execute')
        self.assertEqual(work.memory('execute','new')['episodes'],[])
        self.assertEqual(len(work.memory('research','new')['notes']),1)

    def test_trace_excludes_credentials_images_and_private_reasoning(self):
        work=Workspace();turn=work.start('execute','e','prompt',SETTINGS)
        work.event(turn,'tool','test',{'api_key':'secret','headers':{'Authorization':'Bearer other'},
            'images':{'head':'data:image/jpeg;base64,ABCD'},'reasoning_content':'PRIVATE',
            'text':'sk-abcdefgh12345678','result':{'count':3}},'completed')
        encoded=json.dumps(work.public('execute','e'))
        for forbidden in ['secret','Bearer other','ABCD','PRIVATE','sk-abcdefgh']:
            self.assertNotIn(forbidden,encoded)
        self.assertIn('count',encoded)

    def test_failed_api_is_visible_and_does_not_execute_fallback(self):
        state=LocalState();agent=Harness(state)
        agent.configure_model({'base_url':'http://127.0.0.1:1234/v1','model':'test',
                               'api_key':'placeholder','vision':False})
        with patch.object(agent,'_log'),patch('urllib.request.urlopen',side_effect=TimeoutError):
            with self.assertRaises(RuntimeError):agent.chat('用 TAMP 抓瓶子')
        self.assertFalse(state.calls)
        turn=agent.workspace.public('execute','episode')['turns'][0]
        self.assertEqual(turn['status'],'error')
        self.assertEqual([e['status'] for e in turn['trace'] if e['kind']=='api'],['error'])
        self.assertFalse(agent.workspace.busy)
