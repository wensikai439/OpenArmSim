"""Runtime builder guard and layer-failure diagnostics."""
import copy
import json
import urllib.request
import unittest

from tools import object_workbench
from harness.bottle import VisibleRequests
from harness.compact import CompactRequests
from harness.provider import install_isolated as install_provider
from harness.stages import StageRequests


class RuntimeCompositionChecks(unittest.TestCase):
    def setUp(self):
        self.saved=copy.deepcopy(object_workbench._RUNTIME_TRACE)

    def tearDown(self):
        object_workbench._RUNTIME_TRACE.clear()
        object_workbench._RUNTIME_TRACE.update(self.saved)

    def test_builder_allows_one_process_local_installation_only(self):
        object_workbench._RUNTIME_TRACE.update(attempted=False,complete=False,profile=None,
            global_layers=[],factory_layers=[],failed_layer=None)
        object_workbench._claim_backend_build()
        with self.assertRaisesRegex(RuntimeError,'全新进程'):
            object_workbench._claim_backend_build()

    def test_failed_layer_names_the_partially_modified_process(self):
        object_workbench._RUNTIME_TRACE.update(attempted=True,complete=False,profile=None,
            global_layers=[],factory_layers=[],failed_layer=None)
        def fail(): raise ValueError('fixture')
        with self.assertRaisesRegex(RuntimeError,'fixture.layer'):
            object_workbench._run_layer('global_layers','fixture.layer',fail)
        self.assertEqual(object_workbench._RUNTIME_TRACE['failed_layer'],'fixture.layer')

    def test_request_middleware_is_instance_scoped_and_composes_in_order(self):
        class Workspace:
            def __init__(self): self.events=[]
            def event(self,*args): self.events.append(args)

        class Owner:
            active_session='task'
            stage_packet_audits=[]
            def __init__(self):
                self.workspace=Workspace()
                self.plan={'steps':[{'id':'pick','title':'抓瓶子','skill':'pick','status':'pending'}]}
                self.ledger={'schema':'current-task-ledger/1','root':'task','observation_id':'epoch:1',
                    'completed_stages':[],'next_stage':dict(self.plan['steps'][0]),
                    'remaining_stages':[dict(self.plan['steps'][0])],'native_job_active':False,
                    'failure_reporting_available':False}
            def current_ledger(self): return copy.deepcopy(self.ledger)
            def tools_for_request(self,tools):
                visible=[item for item in tools if item['function']['name']!='prepare_simulation']
                return visible,{'hidden_tools':['prepare_simulation'] if len(visible)!=len(tools) else []}

        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self,_limit):
                return json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]}).encode()

        class Capture:
            Request=urllib.request.Request
            def __init__(self): self.payloads=[]
            def urlopen(self,request,*args,**kwargs):
                self.payloads.append(json.loads(request.data));return Response()

        original_transport=urllib.request
        owner=Owner();harness=type('HarnessFixture',(),{})();harness.request_transport=Capture()
        capture=harness.request_transport
        harness.active_session=owner.active_session;harness.workspace=owner.workspace
        harness.request_transport.Request=urllib.request.Request
        install_provider(harness)
        self.assertIs(urllib.request,original_transport)
        self.assertIs(capture,harness.request_transport.base)
        installed=harness.request_transport
        install_provider(harness)
        self.assertIs(installed,harness.request_transport)
        other=type('HarnessFixture',(),{})();other.request_transport=urllib.request
        self.assertIs(other.request_transport,urllib.request)

        transport=harness.request_transport
        transport=CompactRequests(transport,owner)
        transport=StageRequests(transport,owner,focused=True,atomic=True)
        transport=VisibleRequests(transport,owner)
        sink=transport.base.base.base.base
        self.assertEqual([type(transport).__name__,type(transport.base).__name__,
            type(transport.base.base).__name__,type(transport.base.base.base).__name__],
            ['VisibleRequests','StageRequests','CompactRequests','CheckedRequests'])
        tools=[{'type':'function','function':{'name':name,'description':'fixture','parameters':{}}}
            for name in ('observe','prepare_simulation','pick','place')]
        from .stages import route_tools
        tools=route_tools(tools,owner.plan)[0]
        result={'tool_feedback':{'planner':'native','primitive_success':False,
            'trajectory':[[index]*7 for index in range(6)]}}
        payload={'model':'fixture','messages':[{'role':'system','content':'limits'},
            {'role':'tool','tool_call_id':'receipt','content':json.dumps(result)}],
            'tools':tools,'max_tokens':20}
        request=urllib.request.Request('https://provider.invalid/chat/completions',
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with transport.urlopen(request,timeout=1) as response: response.read(4096)

        posted=sink.payloads[0]
        names=[item['function']['name'] for item in posted['tools']]
        self.assertIn('pick',names);self.assertNotIn('place',names)
        self.assertNotIn('prepare_simulation',names)
        self.assertIn('Current harness task ledger',posted['messages'][0]['content'])
        feedback=json.loads(posted['messages'][1]['content'])['tool_feedback']
        self.assertIn('trajectory_summary',feedback)
        self.assertFalse(feedback['primitive_success'])
        self.assertEqual(owner.stage_packet_audits[-1]['ledger']['observation_id'],'epoch:1')


if __name__=='__main__':unittest.main()
