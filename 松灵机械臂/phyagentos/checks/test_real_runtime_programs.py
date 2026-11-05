"""Offline native UDS contract checks. Memory-only executor; no hardware success claim."""
import asyncio,json,sys,threading,time,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from collections import namedtuple
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import test_real_runtime_bridge as base
from runtime_bridge import Bridge,BridgeError,check_program,digest,save
from provider import Endpoint

SINGLE={'schema_version':'independent-pick-place/v1','request':'Offline protocol fixture only',
        'arm':'right','steps':[{'skill':'observe','args':{}},{'skill':'wait','args':{'duration_s':0.01}}]}

class ProgramChecks(unittest.TestCase):
    def setUp(self):
        base.Tests.setUp(self)
        self.runtime.stage.value=1
        self.runtime.single_arm_program_enabled=True;self.runtime.single_arm_program_active=False;self.runtime.single_arm_program_result=None
        self.runtime.visual_servo_enabled=True;self.runtime.visual_servo_active=False;self.runtime.visual_servo_result=None
        self.runtime.bounded_joint_step_config={'enabled':True,'step_rad':0.01,
            'joint_limits_rad':{f'joint{i}':[-1.,1.] for i in range(1,7)},
            'drift_tolerance_rad':0.02,'max_observation_age_s':0.5}
        self.started=None
        def start(kind,**args):
            self.runtime.starts+=1;self.started=args
            setattr(self.runtime,kind+'_active',self.runtime.keep_active)
            setattr(self.runtime,kind+'_result',None if self.runtime.keep_active else {'status':'completed','source':'memory_only_contract_fixture'})
            return {'source':'memory_only_contract_fixture'}
        def stop(kind):
            self.runtime.stops+=1;setattr(self.runtime,kind+'_active',False)
            setattr(self.runtime,kind+'_result',{'status':'cancelled','operator_cancelled':True,'source':'memory_only_contract_fixture'})
            return {'status':'cancelled','physical_stop':False}
        self.runtime.start_single_arm_program_task=lambda **args:start('single_arm_program',**args)
        self.runtime.start_visual_servo_task=lambda **args:start('visual_servo',**args)
        self.runtime.stop_single_arm_program=lambda:stop('single_arm_program')
        self.runtime.stop_visual_servo=lambda:stop('visual_servo')
    def tearDown(self):base.Tests.tearDown(self)
    def release(self,prepared):
        save(self.bridge.config_path,{'mode':'live','runtime_instance_id':self.runtime.runtime_instance_id,
            'socket_path':str(self.root/'native.sock'),'release':{'release_id':'software-'+uuid.uuid4().hex,
            'prepared_id':prepared['prepared_id'],'sha256':prepared['sha256'],
            'operator_confirmation_ref':'SOFTWARE_FIXTURE_NOT_FIELD_AUTHORIZATION',
            'valid_until_unix':time.time()+60,'field_review_id':'SOFTWARE_REVIEW_NOT_FIELD_APPROVAL','field_verified':True}})
    def execute(self,prepared,key='program:one'):
        self.release(prepared);return self.bridge.execute(prepared['prepared_id'],key,threading.Event())
    def test_preparation_is_offline_immutable_and_never_field_approval(self):
        with patch('socket.socket',side_effect=AssertionError('No native I/O allowed during preparation')):
            prepared=self.bridge.prepare_task('single','single_arm_program',program=SINGLE)
        artifact=json.loads(Path(prepared['path']).read_text())
        self.assertFalse(artifact['field_approved']);self.assertFalse(prepared['native_io_performed'])
        self.assertEqual(artifact['program_digest'],check_program(SINGLE,'single_arm_program')['program_digest'])
        self.assertEqual(check_program(SINGLE,'single_arm_program')['execution_budget_s'],120.)
        with self.assertRaises(FileExistsError):self.bridge.prepare('single',SINGLE)
    def test_single_arm_native_submission_and_field_review_at_hardware_stage(self):
        result=self.execute(self.bridge.prepare('single',SINGLE))
        self.assertEqual(result['phase'],'completed');self.assertEqual(self.runtime.starts,1)
        self.assertEqual(self.started['program_digest'],check_program(SINGLE,'single_arm_program')['program_digest'])
        self.assertEqual(self.gateway.task['task_type'],'single_arm_program')
        self.assertEqual(self.gateway.task['decision_provenance']['field_review_id'],'SOFTWARE_REVIEW_NOT_FIELD_APPROVAL')
        self.assertIsNone(result['business_success']);self.assertEqual(result['physical_stop'],'unknown')
    def test_missing_single_arm_field_review_rejects_before_submit(self):
        prepared=self.bridge.prepare('single',SINGLE);self.release(prepared)
        cfg=self.bridge.config();cfg['release'].pop('field_review_id');save(self.bridge.config_path,cfg)
        with self.assertRaisesRegex(BridgeError,'field_review_id'):self.bridge.execute('single','program:one',threading.Event())
        self.assertEqual(self.runtime.starts,0)
    def test_basic_actions_reach_native_gateway_and_bind_live_joint_feedback(self):
        for index,step in enumerate([{'skill':'move_right_tcp_relative','dx_m':0.005,'dy_m':0.,'dz_m':0.},
                {'skill':'set_gripper','position_m':0.},
                {'skill':'bounded_joint_step','candidate_id':'right_joint1_plus'}]):
            with self.subTest(skill=step['skill']):
                prepared=self.bridge.prepare_step('basic'+str(index),step);result=self.execute(prepared,'basic:'+str(index))
                self.assertEqual(result['phase'],'completed');self.assertEqual(self.gateway.task['task_type'],'visual_servo_step')
                self.assertEqual(self.started['step']['skill'],step['skill']);self.assertIsNone(result['business_success'])
                if step['skill']=='bounded_joint_step':
                    right=result['before']['state']['arms']['right'];sent=self.started['step']
                    self.assertEqual(sent['expected_q_rad'],right['q_rad'])
                    self.assertEqual(sent['expected_feedback_timestamp'],right['feedback_timestamp'])
                    self.assertNotIn('expected_q_rad',json.loads(Path(prepared['path']).read_text())['visual_servo_step'])
    def test_step_scope_and_native_limits_cannot_be_widened(self):
        bad=[{'skill':'move_right_tcp_relative','dx_m':0.011,'dy_m':0.,'dz_m':0.},
             {'skill':'move_right_tcp_relative','dx_m':0.005,'dy_m':0.,'dz_m':0.,'frame':'world'},
             {'skill':'set_gripper','position_m':0.,'side':'left'},
             {'skill':'bounded_joint_step','candidate_id':'right_joint1_plus','expected_q_rad':[0.]*6}]
        for index,step in enumerate(bad):
            with self.assertRaises(BridgeError):self.bridge.prepare_step('bad'+str(index),step)
        self.assertEqual(self.runtime.starts,0)
    def test_disabled_joint_step_is_not_dispatched(self):
        prepared=self.bridge.prepare_step('joint',{'skill':'bounded_joint_step','candidate_id':'right_joint1_plus'});self.release(prepared)
        self.runtime.bounded_joint_step_config['enabled']=False
        with self.assertRaisesRegex(BridgeError,'capability/candidate'):self.bridge.execute('joint','joint:one',threading.Event())
        self.assertEqual(self.runtime.starts,0)
    def test_explicit_native_rejection_is_failed_not_uncertain(self):
        prepared=self.bridge.prepare('single',SINGLE);self.release(prepared);self.gateway.enabled=False
        result=self.bridge.execute('single','rejected:one',threading.Event())
        self.assertEqual(result['phase'],'failed');self.assertEqual(result['native_rejection']['error']['code'],'demo_not_enabled')
        self.assertEqual(self.runtime.starts,0);self.assertIsNone(result['business_success'])
        self.gateway.enabled=True
    def test_manual_checkpoint_surfaces_in_forge_status_without_automatic_answer(self):
        prepared=self.bridge.prepare('single',SINGLE);self.release(prepared);self.runtime.keep_active=True
        self.runtime.single_arm_confirmation_pending=lambda **args:[{'run_id':args['task_id'],'step_index':1,'arm':'right','checkpoint':'after_lift'}]
        self.runtime.submit_single_arm_confirmation=lambda **args:self.fail('Agent must not answer a manual checkpoint')
        endpoint=Endpoint(bridge=self.bridge);Key=namedtuple('Key','invocation_id attempt_id');key=Key('single','confirmation')
        async def emit(event):pass
        asyncio.run(endpoint.start(SimpleNamespace(arguments={'prepared_id':'single'}),SimpleNamespace(execution_key=key,operation='run_task'),SimpleNamespace(emit=emit)))
        try:
            deadline=time.monotonic()+3
            while (self.runtime.starts==0 or not self.bridge.receipt('single:confirmation').exists()) and time.monotonic()<deadline:time.sleep(.01)
            deadline=time.monotonic()+3
            while True:
                status=asyncio.run(endpoint.status(key))
                if status.details.get('pending_confirmation'):break
                if time.monotonic()>deadline:self.fail('Pending confirmation not exposed')
                time.sleep(.01)
            self.assertEqual(status.details['pending_confirmation']['checkpoint'],'after_lift')
            self.assertEqual(status.phase,'running')
        finally:
            asyncio.run(endpoint.cancel(key));deadline=time.monotonic()+3
            while endpoint.done.empty() and time.monotonic()<deadline:time.sleep(.01)
            asyncio.run(endpoint.drain())
        self.assertEqual(self.runtime.stops,1)
    def test_operator_cli_confirmation_binds_pending_task_without_a_new_llm_tool(self):
        prepared=self.bridge.prepare('confirm',SINGLE);self.release(prepared);self.runtime.keep_active=True
        self.runtime.single_arm_confirmation_pending=lambda **args:[{'run_id':args['task_id'],'step_index':2,'arm':'right','checkpoint':'after_lift'}]
        answers=[]
        self.runtime.submit_single_arm_confirmation=lambda **args:answers.append(args) or {'accepted':True,'source':'memory_only_contract_fixture'}
        worker=threading.Thread(target=lambda:self.bridge.execute('confirm','operator:confirm',threading.Event()));worker.start()
        try:
            deadline=time.monotonic()+3
            while self.runtime.starts==0 and time.monotonic()<deadline:time.sleep(.01)
            ref=digest(b'operator:confirm')
            with self.assertRaisesRegex(BridgeError,'No matching pending'):
                self.bridge.confirm_owned(ref,3,'right','after_lift','yes','SOFTWARE_OPERATOR_FIXTURE')
            self.assertEqual(answers,[])
            result=self.bridge.confirm_owned(ref,2,'right','after_lift','unknown','SOFTWARE_OPERATOR_FIXTURE')
            self.assertTrue(result['confirmation']['confirmation']['accepted'])
            self.assertEqual(answers[0]['answer'],'unknown');self.assertEqual(answers[0]['task_id'],self.gateway.task['task_id'])
            spec=json.loads((base.ROOT/'bundles/piper-real-runtime/profiles/runtime/gateway.yaml').read_text())
            self.assertNotIn('confirm_task',[x['operation'] for x in spec['tools']['specs']])
        finally:
            self.bridge.cancel_owned(digest(b'operator:confirm'));worker.join(timeout=3)
        self.assertFalse(worker.is_alive())

    def test_owned_receipt_cancellation_after_provider_restart_never_replays(self):
        prepared=self.bridge.prepare_step('cancel',{'skill':'set_gripper','position_m':0.});self.release(prepared)
        self.runtime.keep_active=True;output={}
        worker=threading.Thread(target=lambda:output.update(self.bridge.execute('cancel','owned:cancel',threading.Event())));worker.start()
        deadline=time.monotonic()+3
        while self.runtime.starts==0 and time.monotonic()<deadline:time.sleep(.01)
        result=Bridge(self.root).cancel_owned(digest(b'owned:cancel'));worker.join(timeout=3)
        self.assertFalse(worker.is_alive());self.assertEqual(result['phase'],'cancelled')
        self.assertEqual(self.runtime.starts,1);self.assertEqual(self.runtime.stops,1)
        self.assertEqual(result['native']['execution_rights'],'available');self.assertIsNone(result['business_success'])

if __name__=='__main__':unittest.main(verbosity=2)
