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
            global_layers=[],factory_layers=[],planner_layers=[],backend_layers=[],
            harness_layers=[],failed_layer=None)
        object_workbench._claim_backend_build()
        with self.assertRaisesRegex(RuntimeError,'全新进程'):
            object_workbench._claim_backend_build()

    def test_runtime_profile_is_derived_from_the_actual_harness_mro(self):
        class OuterHarness: pass
        class InnerHarness: pass
        OuterHarness.__module__='fixture.outer'
        InnerHarness.__module__='fixture.inner'
        OuterHarness.__qualname__='TransitHarness'
        InnerHarness.__qualname__='StageHarness'
        profile=object_workbench._harness_profile([OuterHarness,InnerHarness])
        self.assertEqual(profile,'Transit → Stage')

    def test_failed_layer_names_the_partially_modified_process(self):
        object_workbench._RUNTIME_TRACE.update(attempted=True,complete=False,profile=None,
            global_layers=[],factory_layers=[],backend_layers=[],harness_layers=[],failed_layer=None)
        def fail(): raise ValueError('fixture')
        with self.assertRaisesRegex(RuntimeError,'fixture.layer'):
            object_workbench._run_layer('global_layers','fixture.layer',fail)
        self.assertEqual(object_workbench._RUNTIME_TRACE['failed_layer'],'fixture.layer')

    def test_tamp_router_is_bound_to_service_and_wrapper_is_idempotent(self):
        class Router:
            CONTRACT={'fixture':True}
        class Relocation:
            CONTRACT={'fixture':'relocation'}
        class CuRobo:
            CONTRACT={'fixture':'curobo'}
        metadata=lambda: {'ready':True,'worker_port':8005}
        class BaseService:
            def __init__(self,**kwargs): self.dependencies=kwargs
        Bound=object_workbench.bind_tamp_planner(BaseService,Router,Relocation,CuRobo,metadata)
        self.assertIs(object_workbench.bind_tamp_planner(Bound,Router,Relocation,CuRobo,metadata),Bound)
        instance=Bound()
        self.assertIs(instance.dependencies['tamp_planner_factory'],Router)
        self.assertIs(instance.dependencies['relocation_planner_factory'],Relocation)
        self.assertIs(instance.dependencies['curobo_planner_factory'],CuRobo)
        self.assertIs(instance.dependencies['curobo_metadata_provider'],metadata)
        with self.assertRaisesRegex(ValueError,'already belongs'):
            Bound(tamp_planner_factory=object())
        with self.assertRaisesRegex(ValueError,'relocation planner'):
            Bound(relocation_planner_factory=object())
        with self.assertRaisesRegex(ValueError,'cuRobo planner'):
            Bound(curobo_planner_factory=object())
        with self.assertRaisesRegex(ValueError,'metadata provider'):
            Bound(curobo_metadata_provider=object())

    def test_transit_hint_composer_changes_only_the_selected_harness_subclass(self):
        from harness.transit_hint import install_isolated
        class Base:
            pass
        before=dict(Base.__dict__)
        Composed=install_isolated(Base)
        self.assertIsNot(Composed,Base)
        self.assertIs(Composed.__mro__[1],Base)
        self.assertTrue(Composed.RECEIPT_TRANSIT_HINTS)
        self.assertEqual(dict(Base.__dict__),before)

    def test_intent_composer_changes_only_the_selected_harness_subclass(self):
        from harness.intent import install_isolated
        class Base:
            pass
        before=dict(Base.__dict__)
        Composed=install_isolated(Base)
        self.assertIsNot(Composed,Base)
        self.assertIs(Composed.__mro__[1],Base)
        self.assertTrue(callable(Composed._intent))
        self.assertTrue(callable(Composed._role_records))
        self.assertEqual(dict(Base.__dict__),before)

    def test_policy_route_composer_changes_only_the_selected_harness_subclass(self):
        from harness.policy_route import install_isolated
        class Base:
            pass
        before=dict(Base.__dict__)
        Composed=install_isolated(Base)
        self.assertIsNot(Composed,Base)
        self.assertIs(Composed.__mro__[1],Base)
        self.assertIn('_invoke',Composed.__dict__)
        self.assertEqual(dict(Base.__dict__),before)

    def test_compact_composer_changes_only_the_selected_harness_subclass(self):
        from harness.compact import install_isolated
        class Base:
            def chat(self): pass
        before=dict(Base.__dict__)
        Composed=install_isolated(Base)
        self.assertIsNot(Composed,Base)
        self.assertIs(Composed.__mro__[1],Base)
        self.assertIn('chat',Composed.__dict__)
        self.assertEqual(dict(Base.__dict__),before)

    def test_stage_composer_changes_only_the_selected_harness_subclass(self):
        from harness.bottle import BottleHarness
        from harness.stages import install_isolated
        before=dict(BottleHarness.__dict__)
        Composed=install_isolated(reveal_by_stage=True,live_context=True,
            focused_context=True,bind_atomic=True)
        self.assertIsNot(Composed,BottleHarness)
        self.assertIs(Composed.__mro__[1],BottleHarness)
        self.assertTrue(callable(Composed.current_ledger))
        self.assertTrue(callable(Composed.tools_for_request))
        self.assertEqual(dict(BottleHarness.__dict__),before)

    def test_geometric_relocation_composer_is_local_and_keeps_the_selected_base(self):
        from sim import service
        from sim.grasp import compose_relocation
        original=service.RelocationPlanner
        class Base:
            def _skill(self,*args,**kwargs): return iter(())
        Composed=compose_relocation(Base)
        self.assertIsNot(Composed,Base)
        self.assertIs(Composed.__mro__[1],Base)
        self.assertIn('_skill',Composed.__dict__)
        self.assertIs(service.RelocationPlanner,original)

    def test_service_default_tamp_router_remains_late_bound(self):
        from sim import service
        original=service.TampPlanner
        class FixtureRouter: pass
        class LaterDefault: pass
        instance=service.SimulationService.__new__(service.SimulationService)
        instance.tamp_planner_factory=None
        try:
            service.TampPlanner=LaterDefault
            self.assertIs(instance._tamp_planner_factory(),LaterDefault)
            instance.tamp_planner_factory=FixtureRouter
            self.assertIs(instance._tamp_planner_factory(),FixtureRouter)
        finally:
            service.TampPlanner=original

    def test_service_default_relocation_router_is_late_bound_and_injection_wins(self):
        from sim import service
        original=service.RelocationPlanner
        class FixtureRouter: pass
        class LaterDefault: pass
        instance=service.SimulationService.__new__(service.SimulationService)
        instance.relocation_planner_factory=None
        try:
            service.RelocationPlanner=LaterDefault
            self.assertIs(instance._relocation_planner_factory(),LaterDefault)
            instance.relocation_planner_factory=FixtureRouter
            self.assertIs(instance._relocation_planner_factory(),FixtureRouter)
        finally:
            service.RelocationPlanner=original

    def test_service_default_curobo_router_is_late_bound_and_injection_wins(self):
        from sim import service
        original=service.CuRoboPlanner
        class FixtureRouter: pass
        class LaterDefault: pass
        instance=service.SimulationService.__new__(service.SimulationService)
        instance.curobo_planner_factory=None
        try:
            service.CuRoboPlanner=LaterDefault
            self.assertIs(instance._curobo_planner_factory(),LaterDefault)
            instance.curobo_planner_factory=FixtureRouter
            self.assertIs(instance._curobo_planner_factory(),FixtureRouter)
        finally:
            service.CuRoboPlanner=original

    def test_service_curobo_metadata_provider_is_instance_scoped(self):
        from sim import service
        from unittest.mock import patch
        expected={'ready':True,'worker_port':8005}
        instance=service.SimulationService.__new__(service.SimulationService)
        instance.curobo_metadata_provider=lambda: expected
        self.assertIs(instance._curobo_metadata(),expected)
        instance.curobo_metadata_provider=None
        with patch.object(service,'curobo_metadata',return_value={'worker_port':8004}) as fallback:
            self.assertEqual(instance._curobo_metadata(),{'worker_port':8004})
            fallback.assert_called_once_with()

    def test_service_recorder_is_instance_scoped_and_default_is_late_bound(self):
        from sim import service
        from unittest.mock import patch
        instance=service.SimulationService.__new__(service.SimulationService)
        selected=object();instance.recorder_class=selected
        self.assertIs(instance._recorder_class(),selected)
        instance.recorder_class=None
        fallback=object()
        with patch.object(service,'Recorder',fallback):
            self.assertIs(instance._recorder_class(),fallback)

    def test_object_gpu_defaults_are_scoped_and_explicit_service_backend_wins(self):
        from harness.object_gpu import object_gpu_dependencies
        object_planner=object();object_metadata=object()
        selected_planner=object();selected_metadata=object()
        original={'sentinel':'kept'}
        local=object_gpu_dependencies(original,object_planner,object_metadata)
        self.assertEqual(original,{'sentinel':'kept'})
        self.assertIs(local['curobo_planner_factory'],object_planner)
        self.assertIs(local['curobo_metadata_provider'],object_metadata)
        selected=object_gpu_dependencies({'curobo_planner_factory':selected_planner,
            'curobo_metadata_provider':selected_metadata},object_planner,object_metadata)
        self.assertIs(selected['curobo_planner_factory'],selected_planner)
        self.assertIs(selected['curobo_metadata_provider'],selected_metadata)

    def test_task_scene_default_is_scoped_and_outer_library_environment_wins(self):
        from harness.task_queue import task_scene_dependencies
        task_scene=object();library_env=object()
        original={'unrelated':'preserved'}
        local=task_scene_dependencies(original,task_scene)
        self.assertEqual(original,{'unrelated':'preserved'})
        self.assertIs(local['environment_factory'],task_scene)
        selected=task_scene_dependencies({'environment_factory':library_env},task_scene)
        self.assertIs(selected['environment_factory'],library_env)

    def test_protected_object_default_is_scoped_and_task_scene_override_wins(self):
        from harness.object_handover import relay_environment_dependencies
        protected=object();task_scene=object()
        self.assertIs(relay_environment_dependencies({},protected)['environment_factory'],protected)
        selected=relay_environment_dependencies({'environment_factory':task_scene},protected)
        self.assertIs(selected['environment_factory'],task_scene)

    def test_registered_object_default_is_scoped_and_protected_scene_override_wins(self):
        from harness.objects import object_service_dependencies
        object_env=object();protected=object();recorder=object();selected_recorder=object()
        defaults=object_service_dependencies({},object_env,recorder)
        self.assertIs(defaults['environment_factory'],object_env)
        self.assertIs(defaults['recorder_class'],recorder)
        selected=object_service_dependencies({'environment_factory':protected,
            'recorder_class':selected_recorder},object_env,recorder)
        self.assertIs(selected['environment_factory'],protected)
        self.assertIs(selected['recorder_class'],selected_recorder)

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
