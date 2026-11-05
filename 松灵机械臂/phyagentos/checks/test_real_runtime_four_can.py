"""Offline checks of native four-CAN adapter; no real devices or physical verdicts."""
import os
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
ROOT=Path(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'bundles/piper-real-runtime/profiles/runtime'))
from runtime_bridge import Bridge,BridgeError,save,digest
import native_replay_bridge as replay

TRAJECTORY={'schema':'piper-four-arm-replay-v1','source':{'kind':'SOFTWARE_TRAJECTORY_FIXTURE','output_joint_unit':'deg'},
    'mapping':{r:r.rsplit('_',1)[1] for r in ['leader_left','follower_left','leader_right','follower_right']},
    'coordinate_policy':'zeroed-native-width','grippers':False,'rate':1.,'replay_peak_joint_speed_deg_s':0.,
    'frames':[{'t_s':t,'left_deg':[0.]*6,'right_deg':[0.]*6} for t in [0.,1.]]}

class FourCanChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='paos-four-can-offline-')
        self.root=Path(self.temp.name);(self.root/'config').mkdir()
        save(self.root/'config/real_runtime.json',{'mode':'offline','transport':replay.TRANSPORT,'release':None})
        self.bridge=Bridge(self.root)
        self.source=self.root/'software-only-trajectory.json';save(self.source,TRAJECTORY)
    def tearDown(self):self.temp.cleanup()
    def prepare(self):
        replay.register(self.bridge,'software-only',self.source)
        return self.bridge.prepare_task('candidate',replay.TASK_TYPE,{'registered_replay_id':'software-only'})
    def test_actual_native_validator_accepts_frozen_trajectory_without_control(self):
        result=self.prepare();self.assertFalse(result['field_approved'])
        artifact=json.loads(Path(result['path']).read_text())
        self.assertEqual(artifact['task_type'],replay.TASK_TYPE)
        self.assertEqual(artifact['trajectory_sha256'],digest(self.source.read_bytes()))
        with self.assertRaises(FileExistsError):replay.register(self.bridge,'software-only',self.source)
    def test_native_rejection_cannot_be_prepared(self):
        save(self.source,{**TRAJECTORY,'mapping':{}})
        with self.assertRaisesRegex(BridgeError,'preview rejected'):replay.register(self.bridge,'bad',self.source)
        with self.assertRaisesRegex(BridgeError,'successful validation'):replay.prepare(self.bridge,'candidate',{'registered_replay_id':'bad'})
    def test_raw_targets_are_not_an_agent_program(self):
        for value in [TRAJECTORY,{'registered_replay_id':'../other'}, {'registered_replay_id':'x','raw_joints':[0]*6}]:
            with self.assertRaises(BridgeError):replay.prepare(self.bridge,'candidate',value)
    def test_mutated_registered_bytes_are_rejected(self):
        replay.register(self.bridge,'software-only',self.source)
        path=replay.registry(self.bridge,'software-only');path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(BridgeError,'changed'):replay.prepare(self.bridge,'candidate',{'registered_replay_id':'software-only'})
    def test_offline_gate_precedes_import_spawn_and_device_resolution(self):
        self.prepare()
        with patch.object(replay,'native_module',side_effect=AssertionError('No native worker allowed')):
            with self.assertRaisesRegex(BridgeError,'OFFLINE_ONLY'):self.bridge.execute('candidate','offline',threading.Event())
        self.assertFalse(self.bridge.receipt('offline').exists())
    def test_live_mode_without_exact_release_cannot_start_worker(self):
        self.prepare();save(self.bridge.config_path,{'mode':'live','transport':replay.TRANSPORT,'release':None})
        with patch.object(replay,'native_module',side_effect=AssertionError('No native worker allowed')):
            with self.assertRaisesRegex(BridgeError,'exact prepared artifact'):self.bridge.execute('candidate','no-release',threading.Event())
    def test_current_transport_rejects_legacy_task_and_uds_client(self):
        save(self.bridge.config_path,{'mode':'live','transport':replay.TRANSPORT,'release':None})
        self.bridge.store_task('legacy',{'task_type':'visual_servo_step'})
        with self.assertRaisesRegex(BridgeError,'not adapted'):self.bridge.execute('legacy','wrong-chain',threading.Event())
        with self.assertRaisesRegex(BridgeError,'legacy UDS'):self.bridge.client('wrong-chain')
    def test_inactive_worker_is_not_fresh_robot_feedback(self):
        value=replay.worker_snapshot(self.bridge)
        self.assertFalse(value['available']);self.assertFalse(value['feedback_available'])
        self.assertIsNone(value['business_success'])
    def test_uncertain_old_owner_is_not_replayed_or_promoted(self):
        key='software-only:unknown';ref=digest(key.encode())
        save(self.bridge.receipt(key),{'transport':replay.TRANSPORT,'phase':'running','worker_identity':None})
        value=replay.recover(self.bridge,ref);self.assertEqual(value['phase'],'unknown')
        self.assertFalse(value['worker_still_present'])
    def test_gateway_schema_accepts_id_reference_and_rejects_step_form(self):
        import jsonschema
        specs=json.loads((ROOT/'bundles/piper-real-runtime/profiles/runtime/gateway.yaml').read_text())['tools']['specs']
        schema=next(x['input_schema'] for x in specs if x['operation']=='prepare_task')
        jsonschema.validate({'prepared_id':'x','task_type':'four_can_replay','program':{'registered_replay_id':'x'}},schema)
        with self.assertRaises(jsonschema.ValidationError):jsonschema.validate({'prepared_id':'x','task_type':'four_can_replay','step':{}},schema)


    def software_release(self,prepared):
        save(self.bridge.config_path,{'mode':'live','transport':replay.TRANSPORT,'release':{
            'release_id':'SOFTWARE_ONLY_RELEASE','prepared_id':'candidate','sha256':prepared['sha256'],
            'valid_until_unix':time.time()+60,'operator_confirmation_ref':'SOFTWARE_FIXTURE_NOT_FIELD_AUTHORIZATION'}})
    def test_pre_cancelled_candidate_does_not_consume_or_start(self):
        prepared=self.prepare();self.software_release(prepared);native=replay.native_module()
        event=threading.Event();event.set()
        with patch.object(replay,'lock_available',return_value=True), patch.object(native,'require_prepared_can'), \
             patch.object(native.Worker,'start',side_effect=AssertionError('Cancelled task must not start')):
            result=self.bridge.execute('candidate','already-cancelled',event)
        self.assertEqual(result['phase'],'cancelled');self.assertFalse(result['native_dispatch_attempted'])
        self.assertFalse((self.bridge.run/'releases/SOFTWARE_ONLY_RELEASE.used').exists())
    def test_cursor_without_native_summary_never_becomes_success(self):
        # Memory-only lifecycle fixture for missing-evidence behavior; no physical claim.
        prepared=self.prepare();self.software_release(prepared);native=replay.native_module()
        class MemoryOnlyWorker:
            preview=False
            def __init__(self,*a,**k):self.process=None;self.stopped=False
            def build_command(self,*a):return ['SOFTWARE_FIXTURE','--prepared-can']
            def start(self,*a):
                from types import SimpleNamespace
                self.process=SimpleNamespace(pid=-999,poll=lambda:0)
            def stop(self):self.stopped=True
            def status(self):return {'state':'HOLD','cursor_s':1.,'running':not self.stopped,
                'error':None,'exit_code':0,'forced_stop':False,'recording':{'directory':None}}
        with patch.object(replay,'lock_available',return_value=True), patch.object(native,'require_prepared_can'), \
             patch.object(native,'Worker',MemoryOnlyWorker):
            result=self.bridge.execute('candidate','no-evidence',threading.Event())
        self.assertEqual(result['phase'],'unknown');self.assertIsNone(result['business_success'])
        self.assertIsNone(result['physical_success']);self.assertIsNone(result['native_summary'])

    def test_forge_preflight_routes_current_transport_without_legacy_client(self):
        prepared=self.prepare();self.software_release(prepared)
        with patch.object(replay,'lock_available',return_value=True), patch.object(self.bridge,'client',side_effect=AssertionError('Legacy UDS must not be used')):
            cfg,artifact,path,release_id=self.bridge.preflight('candidate','forge-preflight-software')
        self.assertEqual(artifact['transport'],replay.TRANSPORT)
        self.assertEqual(release_id,'SOFTWARE_ONLY_RELEASE')
        self.assertFalse(self.bridge.receipt('forge-preflight-software').exists())
    def test_forge_offline_preflight_cannot_reach_native_or_legacy_sender(self):
        self.prepare()
        with patch.object(self.bridge,'client',side_effect=AssertionError('No legacy sender')), patch.object(replay,'native_module',side_effect=AssertionError('No native worker')):
            with self.assertRaisesRegex(BridgeError,'OFFLINE_ONLY'):self.bridge.preflight('candidate','offline-forge')

if __name__=='__main__':unittest.main()
