"""Software protocol checks only. Native Gateway + memory-only Runtime; no real devices."""
import os
import asyncio,contextlib,json,socket,sys,tempfile,threading,time,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collections import namedtuple
ROOT=Path(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'bundles/piper-real-runtime/profiles/runtime'))
sys.path.insert(0,os.environ.get('PIPER_CONTROL_ROOT', str(ROOT/'external/robot-control')))
from runtime_bridge import Bridge,BridgeError,Client,check_program,digest,save
from piper_baseline.runtime_task_gateway import RuntimeTaskGateway
from provider import Endpoint

class MemoryOnlyRuntime:
    def __init__(self):
        self.runtime_instance_id='software-contract-fixture'
        self.stage=SimpleNamespace(value=2)
        self.bimanual_program_enabled=True;self.bimanual_program_active=False;self.bimanual_program_result=None
        self.model_failed=False;self.cycle_id=None;self.cycle_dir=None
        self.observation_socket=None;self.starts=0;self.stops=0;self.keep_active=False;self.release_ready=True
    def start_bimanual_program_task(self,**kwargs):
        self.starts+=1;self.bimanual_program_active=self.keep_active
        self.bimanual_program_result=None if self.keep_active else {'status':'completed','source':'memory_only_contract_fixture'}
        return {'source':'memory_only_contract_fixture'}
    def stop_bimanual_program(self):
        self.stops+=1;self.bimanual_program_active=False
        self.bimanual_program_result={'status':'cancelled','operator_cancelled':True,'source':'memory_only_contract_fixture'}
        return {'physical_stop':False}
    def read_runtime_state(self):
        return {'arms':{side:{'q_rad':[0.]*6,'feedback_timestamp':time.time()} for side in ['left','right']},'source':'memory_only_contract_fixture'}
    def task_release_evidence(self,**kwargs):return {'ready':self.release_ready,'source':'memory_only_contract_fixture'}

PROGRAM={'schema_version':'bimanual-demo/v1','request':'Software protocol fixture only; no physical task or success claim',
         'repeat':1,'prelude':[],'steps':[{'skill':'observe','args':{}},{'skill':'wait','args':{'duration_s':.1}}],'postlude':[]}

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='paos-real-check-');self.root=Path(self.temp.name)
        (self.root/'config').mkdir();(self.root/'prepared/real_tasks').mkdir(parents=True)
        save(self.root/'config/real_runtime.json',{'mode':'offline'})
        self.bridge=Bridge(self.root);self.runtime=MemoryOnlyRuntime()
        self.gateway=RuntimeTaskGateway(self.runtime,socket_path=self.root/'native.sock');self.gateway.enable()
        self.bridge.observe=lambda session='unused': {'source':'memory_only_contract_fixture','frames':[],'business_success':None}
        # Any accidental connection outside this one isolated socket aborts the check.
        original=socket.socket.connect;self.allowed={str(self.root/'native.sock')}
        def connect(peer,address):
            if address not in self.allowed:raise AssertionError('Non-isolated socket access forbidden in software checks: '+str(address))
            return original(peer,address)
        self.connect_patch=patch.object(socket.socket,'connect',connect);self.connect_patch.start()
    def tearDown(self):
        self.connect_patch.stop();self.gateway.disable();self.temp.cleanup()
    def ready(self):
        prepared=self.bridge.prepare('observe-fixture',PROGRAM)
        config={'mode':'live','runtime_instance_id':self.runtime.runtime_instance_id,'socket_path':str(self.root/'native.sock'),
                'release':{'release_id':'software-fixture-'+uuid.uuid4().hex,'prepared_id':'observe-fixture','sha256':prepared['sha256'],
                           'operator_confirmation_ref':'SOFTWARE_FIXTURE_NOT_FIELD_AUTHORIZATION','valid_until_unix':time.time()+60}}
        save(self.bridge.config_path,config);return config
    def run_task(self,key='invocation:attempt',event=None):return self.bridge.execute('observe-fixture',key,event or threading.Event())
    def test_offline_gate_before_all_native_io(self):
        with patch('socket.socket',side_effect=AssertionError('No I/O permitted')):
            caps=self.bridge.capabilities();self.assertFalse(caps['connected']);self.assertFalse(caps['motion_ready'])
            with self.assertRaisesRegex(BridgeError,'OFFLINE_ONLY'):self.bridge.client('test')
            with self.assertRaisesRegex(BridgeError,'OFFLINE_ONLY'):self.run_task()
        self.assertEqual(self.runtime.starts,0)
    def test_native_program_checker_rejects_raw_cartesian_parameters(self):
        bad=json.loads(json.dumps(PROGRAM));bad['steps']=[{'skill':'move_arm','args':{'target_world_m':[0,0,.4]}}]
        with self.assertRaisesRegex(BridgeError,'Native program rejection'):self.bridge.prepare('bad',bad)
        self.assertFalse(self.bridge.task_path('bad').exists());self.assertEqual(self.runtime.starts,0)
    def test_native_wire_completion_preserves_unknown_physical_evidence(self):
        self.ready();result=self.run_task()
        self.assertEqual(result['phase'],'completed');self.assertEqual(self.runtime.starts,1)
        self.assertEqual(result['native']['execution_rights'],'available')
        self.assertEqual(result['native']['evidence']['object'],'unknown');self.assertIsNone(result['business_success'])
        self.assertEqual(result['physical_stop'],'unknown')
    def test_exact_artifact_hash_is_bound_before_dispatch(self):
        self.ready();path=self.bridge.task_path('observe-fixture');path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(BridgeError,'exact prepared artifact'):self.run_task()
        self.assertEqual(self.runtime.starts,0)
    def test_runtime_identity_change_is_rejected(self):
        self.ready();self.runtime.runtime_instance_id='other-id';self.gateway.runtime_instance_id='other-id'
        with self.assertRaisesRegex(BridgeError,'runtime changed'):self.run_task()
        self.assertEqual(self.runtime.starts,0)
    def test_native_disabled_executor_is_not_bypassed(self):
        self.ready();self.runtime.bimanual_program_enabled=False
        with self.assertRaisesRegex(BridgeError,'disabled/unconfigured'):self.run_task()
        self.assertEqual(self.runtime.starts,0)
    def test_expired_release_and_one_use_budget(self):
        cfg=self.ready();cfg['release']['valid_until_unix']=time.time()-1;save(self.bridge.config_path,cfg)
        with self.assertRaisesRegex(BridgeError,'five-minute'):self.run_task()
        cfg['release']['valid_until_unix']=time.time()+60;save(self.bridge.config_path,cfg)
        self.run_task()
        with self.assertRaises(FileExistsError):self.run_task('different:attempt')
        self.assertEqual(self.runtime.starts,1)
    def test_lost_submit_reply_reconciles_without_replay(self):
        cfg=self.ready()
        class LostReplyClient(Client):
            def request(self,operation,**fields):
                result=super().request(operation,**fields)
                if operation=='submit_task':raise BridgeError('injected lost reply after native admission')
                return result
        self.bridge.client=lambda session:LostReplyClient(cfg['socket_path'],session,cfg['runtime_instance_id'])
        result=self.run_task();self.assertEqual(result['phase'],'completed');self.assertEqual(self.runtime.starts,1)
        with self.assertRaisesRegex(BridgeError,'never resubmit'):self.run_task()
    def test_unresolved_submit_stays_unknown_and_blocks_next_dispatch(self):
        cfg=self.ready()
        class NoReplyClient(Client):
            def request(self,operation,**fields):
                if operation=='submit_task':raise BridgeError('injected transport loss before outcome known')
                return super().request(operation,**fields)
        self.bridge.client=lambda session:NoReplyClient(cfg['socket_path'],session,cfg['runtime_instance_id'])
        result=self.run_task();self.assertEqual(result['phase'],'unknown');self.assertEqual(self.runtime.starts,0)
        with self.assertRaisesRegex(BridgeError,'unresolved owned task'):self.run_task('next:attempt')
    def test_owned_cancellation_waits_for_native_terminal_and_release(self):
        self.ready();self.runtime.keep_active=True;cancel=threading.Event();output={}
        thread=threading.Thread(target=lambda:output.update(self.run_task(event=cancel)));thread.start()
        deadline=time.monotonic()+5
        while self.runtime.starts==0 and time.monotonic()<deadline:time.sleep(.01)
        self.assertEqual(self.runtime.starts,1);cancel.set();thread.join(timeout=5)
        self.assertFalse(thread.is_alive());self.assertEqual(output['phase'],'cancelled')
        self.assertEqual(self.runtime.stops,1);self.assertEqual(output['native']['execution_rights'],'available')
        self.assertEqual(output['native']['evidence']['physical_stop'],'unknown')
    def test_recovery_does_not_release_terminal_without_native_release(self):
        self.ready();result=self.run_task();ref=digest(b'invocation:attempt')
        record=json.loads(self.bridge.receipt('invocation:attempt').read_text());record['phase']='unknown';save(self.bridge.receipt('invocation:attempt'),record)
        self.runtime.release_ready=False;self.gateway._task['execution_rights']='release_pending'
        recovered=self.bridge.recover(ref);self.assertEqual(recovered['phase'],'unknown')
        self.runtime.release_ready=True;recovered=self.bridge.recover(ref);self.assertEqual(recovered['phase'],'completed')
        self.assertEqual(self.runtime.starts,1)
    def test_forge_endpoint_preserves_unknown_after_unexpected_dispatch_error(self):
        self.ready();bridge=self.bridge
        bridge.preflight=lambda *a:None
        def fail(task_id,key,cancel):
            save(bridge.receipt(key),{'phase':'dispatch_pending'});raise ValueError('injected post-intent failure')
        bridge.execute=fail;endpoint=Endpoint(bridge=bridge);Key=namedtuple('Key','invocation_id attempt_id');key=Key('forge-fixture','attempt')
        events=SimpleNamespace(emit=lambda event:None)
        asyncio.run(endpoint.start(SimpleNamespace(arguments={'prepared_id':'observe-fixture'}),SimpleNamespace(execution_key=key),events))
        deadline=time.monotonic()+2
        while endpoint.done.empty() and time.monotonic()<deadline:time.sleep(.01)
        asyncio.run(endpoint.drain());self.assertEqual(endpoint.records[key]['result'].status,'unknown');self.assertEqual(endpoint.active,key)

    def test_native_binary_observation_contract_and_stale_frame_rejection(self):
        from runtime_bridge import snapshot,ORDER,CAMERAS
        from piper_baseline.shadow_protocol import receive,send
        path=str(self.root/'observation.sock');self.allowed.add(path)
        def one(stale):
            listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(path);listener.listen(1)
            stamp=time.time()-5 if stale else time.time()
            meta={'ok':True,'schema':'piper-shadow-v1','state':[0.]*14,'order':ORDER,'camera_names':CAMERAS,
                  'image_shape':[480,640,3],'encoding':'rgb8','snapshot_wall':stamp,
                  'state_stamps':{s+'_'+k:stamp for s in ['left','right'] for k in ['joints','gripper']},
                  'image_stamps':{c:stamp for c in CAMERAS}}
            payload=bytes([17])*(3*480*640*3)
            def serve():
                with listener.accept()[0] as peer:
                    request,raw=receive(peer);self.assertEqual(request,{'op':'observe'});send(peer,meta,payload)
            worker=threading.Thread(target=serve);worker.start()
            try:
                if stale:
                    with self.assertRaisesRegex(BridgeError,'Stale'):snapshot(path)
                else:
                    observed,pixels=snapshot(path);self.assertEqual(pixels,payload);self.assertEqual(observed['order'],ORDER)
            finally:
                worker.join(timeout=3);listener.close();Path(path).unlink()
        one(False);one(True)
    def test_recovering_an_old_task_cannot_release_another_active_action(self):
        bridge=self.bridge;bridge.recover=lambda ref:{'phase':'completed','native':{'execution_rights':'available'}}
        endpoint=Endpoint(bridge=bridge);key=('active','execution');endpoint.active=key;endpoint.records[key]={'reference':'a'*64}
        asyncio.run(endpoint.query(SimpleNamespace(arguments={'receipt_reference':'b'*64}),SimpleNamespace(operation='recover')))
        self.assertEqual(endpoint.active,key)


class PolicyTests(unittest.TestCase):
    """Bounded ACT protocol fixtures only; never a real checkpoint or inference."""
    setUpBase=Tests.setUp
    tearDownBase=Tests.tearDown
    def setUp(self):
        self.setUpBase()
        self.runtime.model_config={'model_id':'act','dataset':'SOFTWARE_FIXTURE_NOT_REAL_DATA'}
        self.runtime.model_active=False;self.runtime.model_result=None;self.started_options=None
        self.policy={'model_config_path':'/SOFTWARE_FIXTURE_NOT_MODEL.json','model_config_sha256':'a'*64,
            'expected_model_config_sha256':self.gateway._loaded_model_binding()['sha256'],
            'model_files':{'checkpoint_weights':{'path':'SOFTWARE_FIXTURE','sha256':'b'*64}},
            'dataset_metadata_sha256':'c'*64,'duration_s':25.0,'include_gripper':True,'total_budget_s':25.0}
        self.policy_patch=patch('runtime_bridge.check_policy',side_effect=lambda *args:dict(self.policy));self.policy_patch.start()
        def start(**options):
            self.runtime.starts+=1;self.started_options=options;self.runtime.model_active=self.runtime.keep_active
            self.runtime.model_result=None if self.runtime.keep_active else {'status':'completed','source':'memory_only_policy_fixture'}
            return {'source':'memory_only_policy_fixture'}
        def stop():
            self.runtime.stops+=1;self.runtime.model_active=False
            self.runtime.model_result={'status':'cancelled','operator_cancelled':True,'source':'memory_only_policy_fixture'}
            return {'physical_stop':False}
        self.runtime.start_act_task=start;self.runtime.stop_model=stop
    def tearDown(self):
        self.policy_patch.stop();self.tearDownBase()
    def ready_policy(self):
        prepared=self.bridge.prepare_policy('towel-fixture',self.policy['model_config_path'],25.0,True)
        save(self.bridge.config_path,{'mode':'live','runtime_instance_id':self.runtime.runtime_instance_id,
            'socket_path':str(self.root/'native.sock'),'release':{'release_id':'software-policy-'+uuid.uuid4().hex,
            'prepared_id':'towel-fixture','sha256':prepared['sha256'],'operator_confirmation_ref':'SOFTWARE_FIXTURE_NOT_FIELD_AUTHORIZATION',
            'valid_until_unix':time.time()+60}})
    def test_finite_policy_options_reach_native_gateway_without_success_claim(self):
        self.ready_policy();result=self.bridge.execute('towel-fixture','policy:one',threading.Event())
        self.assertEqual(result['phase'],'completed');self.assertEqual(self.runtime.starts,1)
        self.assertEqual(self.started_options['duration_s'],25.0);self.assertTrue(self.started_options['include_gripper'])
        self.assertEqual(result['dispatch_intent']['expected_model_config_sha256'],self.policy['expected_model_config_sha256'])
        self.assertIsNone(result['business_success']);self.assertEqual(result['native']['evidence']['object'],'unknown')
    def test_changed_loaded_model_is_rejected_before_start(self):
        self.ready_policy();self.runtime.model_config['checkpoint']='DIFFERENT_SOFTWARE_FIXTURE'
        with self.assertRaisesRegex(BridgeError,'loaded model configuration'):self.bridge.execute('towel-fixture','policy:one',threading.Event())
        self.assertEqual(self.runtime.starts,0)
    def test_changed_frozen_model_files_are_rejected(self):
        self.ready_policy();self.policy['model_files']={'checkpoint_weights':{'path':'SOFTWARE_FIXTURE','sha256':'d'*64}}
        with self.assertRaisesRegex(BridgeError,'model files changed'):self.bridge.execute('towel-fixture','policy:one',threading.Event())
        self.assertEqual(self.runtime.starts,0)
    def test_old_runtime_without_bounded_interface_is_rejected(self):
        self.ready_policy();capabilities=self.gateway._capabilities
        def old(request_id):
            response=capabilities(request_id);response['result']['task_types']['act_approved'].pop('finite_duration_s');return response
        self.gateway._capabilities=old
        with self.assertRaisesRegex(BridgeError,'finite policy interface is unavailable'):self.bridge.execute('towel-fixture','policy:one',threading.Event())
        self.assertEqual(self.runtime.starts,0)
    def test_policy_cancellation_reuses_native_stop_and_release(self):
        self.ready_policy();self.runtime.keep_active=True;cancel=threading.Event();output={}
        worker=threading.Thread(target=lambda:output.update(self.bridge.execute('towel-fixture','policy:one',cancel)))
        worker.start();deadline=time.monotonic()+5
        while self.runtime.starts==0 and time.monotonic()<deadline:time.sleep(.01)
        cancel.set();worker.join(timeout=5)
        self.assertFalse(worker.is_alive());self.assertEqual(output['phase'],'cancelled');self.assertEqual(self.runtime.stops,1)
        self.assertEqual(output['native']['execution_rights'],'available');self.assertIsNone(output['business_success'])
    def test_invalid_policy_duration_never_runs_config_checker(self):
        self.policy_patch.stop()
        from runtime_bridge import check_policy
        with patch('runtime_bridge.subprocess.run',side_effect=AssertionError('No subprocess permitted')):
            for duration in [0,-1,61,True,float('inf'),float('nan')]:
                with self.assertRaisesRegex(BridgeError,'duration_s'):check_policy('/not-a-model.json',duration,True)
        self.policy_patch.start()

if __name__=='__main__':unittest.main(verbosity=2)
