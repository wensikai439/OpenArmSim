"""Session credentials, text-model tool permissions, and fresh-state feedback."""
import json
import unittest
import time
from unittest.mock import patch

from .agent import Harness
from .checks import LocalState
from .llm import ModelSettings


class ModelChecks(unittest.TestCase):
    def test_saved_api_survives_restart_and_can_edit_without_reentering_key(self):
        class Store:
            value=None
            def load(self):return self.value.copy() if self.value else None
            def save(self,value):self.value=value.copy()
        store=Store();settings=ModelSettings(store)
        good={'base_url':'https://example.org/v1','model':'first','api_key':'private-placeholder','vision':False}
        settings.configure(good)
        restarted=ModelSettings(store)
        self.assertEqual(restarted.snapshot(),good)
        public=restarted.configure({'base_url':good['base_url'],'model':'second','vision':True})
        self.assertEqual(restarted.snapshot()['api_key'],'private-placeholder')
        self.assertEqual(ModelSettings(store).snapshot()['model'],'second')
        self.assertNotIn('private-placeholder',json.dumps(public))
        with self.assertRaises(ValueError):
            restarted.configure({'base_url':'https://elsewhere.example/v1','model':'other','vision':False})
        self.assertEqual(ModelSettings(store).snapshot()['base_url'],good['base_url'])
        restarted.clear()
        with patch.dict('os.environ',{'OPENARM_LLM_KEY':'environment-placeholder'}):
            self.assertFalse(ModelSettings(store).public()['configured'])

    def test_failed_persistent_save_preserves_existing_settings(self):
        class Store:
            def load(self):return {'base_url':'https://example.org/v1','model':'first','api_key':'placeholder','vision':False}
            def save(self,value):raise RuntimeError('keyring unavailable')
        settings=ModelSettings(Store());before=settings.snapshot()
        with self.assertRaises(RuntimeError):
            settings.configure({'base_url':before['base_url'],'model':'second','vision':True})
        self.assertEqual(settings.snapshot(),before)
        with self.assertRaises(RuntimeError):settings.clear()
        self.assertEqual(settings.snapshot(),before)

    def test_secret_is_never_in_public_config_and_clearing_disables_fallback(self):
        settings=ModelSettings()
        good={'base_url':'https://example.org/v1','model':'model','api_key':'private-placeholder','vision':False}
        public=settings.configure(good)
        self.assertNotIn('private-placeholder',json.dumps(public))
        self.assertNotIn('api_key',public)
        self.assertTrue(public['configured'])
        for url in ('http://external.example/v1','https://user:password@example.org','https://example.org?key=secret'):
            with self.assertRaises(ValueError):
                settings.configure({**good,'base_url':url})
        settings.clear()
        self.assertFalse(settings.public()['configured'])

    def test_text_llm_can_route_to_tamp_but_not_vision_gated_vla(self):
        state=LocalState();agent=Harness(state)
        agent.configure_model({'base_url':'http://127.0.0.1:1234/v1','model':'contract-test',
                               'api_key':'private-placeholder','vision':False})
        requests=[]
        def response(request,**kwargs):
            payload=json.loads(request.data)
            names=[t['function']['name'] for t in payload['tools']]
            self.assertIn('pick',names);self.assertNotIn('run_policy',names)
            self.assertNotIn('image_url',json.dumps(payload))
            self.assertNotIn('OBJECT_TRUTH',json.dumps(payload))
            requests.append(payload)
            message={'role':'assistant','tool_calls':[{'id':'call1','type':'function',
                'function':{'name':'pick','arguments':json.dumps({'object_id':'bottle','arm':'right','observation_id':'episode:0'})}}]}
            if len(requests)==1:
                message={'role':'assistant','tool_calls':[{'id':'plan','type':'function','function':{'name':'set_plan','arguments':json.dumps({'observation_id':'episode:0','steps':[{'id':'pick','title':'抓起瓶子','checkpoint':'right_lift'}]})}}]}
            class Reply:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read(self,_):return json.dumps({'choices':[{'message':message}]}).encode()
            return Reply()
        with patch('urllib.request.urlopen',side_effect=response),patch.object(agent,'_log'):
            result=agent.chat('用 TAMP 抓起瓶子')
        self.assertEqual(state.calls,['tamp'])
        self.assertEqual([t['name'] for t in result['tools']],['set_plan','pick'])
        self.assertNotIn('private-placeholder',json.dumps(result))
        with self.assertRaises(ValueError):
            agent.invoke('run_tamp',{'task':'pick_bottle','observation_id':'old:0'})
        with self.assertRaises(ValueError):
            agent.invoke('run_tamp',{'task':'pick_bottle','observation_id':'episode:0'},role='research')

    def test_tamp_feedback_is_labelled_and_reset_drops_oracle_context(self):
        state=LocalState();agent=Harness(state)
        state.cache['last_run']={'kind':'tamp','run_id':'a','metrics':{'tamp':{'scene_source':'simulator_ground_truth',
            'handover_search':{'selected_bottle_world':[.46,-.02,1.215]}}}}
        with patch.object(agent,'_log'):
            result=agent.invoke('get_run_status',{})
            self.assertTrue(result['contains_privileged_object_state'])
            self.assertTrue(result['contains_privileged_primitive_feedback'])
            agent.invoke('observe',{})
            agent.history['execute'].append({'role':'assistant','content':'old oracle result'})
            state.cache['observation_id']='new-episode:0'
            agent.invoke('observe',{})
            self.assertFalse(agent.history['execute'])

    def test_completion_of_previous_operator_cannot_review_the_next_one(self):
        state=LocalState(with_queue=True);agent=Harness(state)
        agent.pending_review={'epoch':'episode','execution_id':'new-operation','prompt':'next'}
        try:
            with patch.object(agent,'chat',return_value={'message':'review'}) as chat:
                state.completed.put({'epoch':'episode','execution_id':'old-operation','run_id':'old-run'})
                state.completed.put({'epoch':'episode','execution_id':'new-operation','run_id':'new-run'})
                deadline=time.monotonic()+2
                while not agent.notifications() and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(len(agent.notifications()),1)
                self.assertEqual(agent.notifications()[0]['run_id'],'new-run')
                self.assertEqual(chat.call_count,1)
        finally:
            agent.close();agent.review_thread.join(1)
