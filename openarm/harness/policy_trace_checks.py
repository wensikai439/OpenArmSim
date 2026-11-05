"""Receipt integrity and noninterference checks; no models or hardware."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sim.environment import ROOT
from sim.policy import LeRobotClient
from sim.policy_wire import TASK, IMAGE_SHAPES, encode, decode
from .policy_trace import (TracingClient, TracingTiming, archive_response,
                           install_isolated, traced_client)


class TraceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='trace-check-',dir=ROOT/'runs')
        self.owner=SimpleNamespace(path=Path(self.tmp.name))
        self.observation={'state':np.zeros(16,np.float32),'task':np.array(TASK),
            'session':np.array('a'*32),'sequence':np.array(0,np.int64),
            **{k:np.zeros(s,np.uint8) for k,s in IMAGE_SHAPES.items()}}
        self.body=encode(self.observation)
        self.actions=np.zeros((32,16),np.float32);self.actions[:,[7,15]]=-.89
        self.capture={'input':self.body,'response':{'actions':self.actions.tolist(),'sequence':0}}
        self.bounds=[[-.9,-.9],[.05,.05]]

    def tearDown(self):self.tmp.cleanup()

    def record(self,valid=True):
        return archive_response(self.owner,{'request_id':0,'response_valid':valid},self.capture,
            horizon=16,gripper_bounds=self.bounds)

    def test_success_keeps_every_predicted_row_without_storing_pixel_bytes(self):
        trace=self.record()
        np.testing.assert_array_equal(np.load(self.owner.path/trace['actions_file']),self.actions)
        self.assertNotIn('rejected_input_file',trace)
        self.assertEqual(list((self.owner.path/'policy').iterdir()),[self.owner.path/'policy/0000-actions.npy'])
        self.assertEqual(trace['out_of_range_selected'],0);json.dumps(trace,allow_nan=False)

    def test_failed_executed_prefix_is_not_clipped_and_exact_input_roundtrips(self):
        self.actions[:6,15]=.0528;self.capture['response']['actions']=self.actions.tolist()
        trace=self.record(False)
        self.assertEqual(trace['out_of_range_selected'],6);self.assertEqual(trace['out_of_range_unused'],0)
        for k,v in decode((self.owner.path/trace['rejected_input_file']).read_bytes()).items():
            np.testing.assert_array_equal(v,self.observation[k])
        np.testing.assert_array_equal(np.load(self.owner.path/trace['actions_file']),self.actions)

    def test_unexecuted_tail_is_distinguished_and_original_capture_is_preserved(self):
        self.actions[20,7]=-.91;self.capture['response']['actions']=self.actions.tolist()
        before=json.dumps(self.capture['response']);trace=self.record(False)
        self.assertEqual(trace['out_of_range_selected'],0);self.assertEqual(trace['out_of_range_unused'],1)
        self.assertEqual(json.dumps(self.capture['response']),before)

    def test_nonfinite_response_is_archived_without_invalid_json_numbers(self):
        self.actions[0,7]=np.nan;self.capture['response']['actions']=self.actions.tolist()
        trace=self.record(False);self.assertFalse(trace['finite']);json.dumps(trace,allow_nan=False)
        self.assertTrue(np.isnan(np.load(self.owner.path/trace['actions_file'])[0,7]))

    def test_receipt_is_not_overwritten_or_written_outside_native_runs(self):
        self.record()
        trace=self.record();self.assertIn('do not overwrite',trace['response_trace_error'])
        with tempfile.TemporaryDirectory() as other:
            owner=SimpleNamespace(path=Path(other))
            with self.assertRaises(ValueError):
                archive_response(owner,{'request_id':0,'response_valid':False},self.capture,horizon=16,gripper_bounds=self.bounds)
            self.assertEqual(list(Path(other).iterdir()),[])

    def test_http_result_and_encoded_packet_are_returned_unchanged(self):
        client=TracingClient({});result=self.capture['response']
        with patch.object(LeRobotClient,'request',return_value=result) as request:
            self.assertIs(client.request('/infer',self.body,'application/x-npz'),result)
        request.assert_called_once_with('/infer',self.body,'application/x-npz')
        self.assertIs(client.response_capture['input'],self.body)

    def test_timing_retains_original_failure_and_does_not_step_physics(self):
        events=[]
        class Owner:
            path=self.owner.path
            def event(self,value):events.append(value)
        class Client:
            response_capture=self.capture
            def infer(self,*args):raise ValueError('original range rejection')
        client=Client();env=SimpleNamespace(profile={'control_hz':30},tick=0,data=SimpleNamespace(time=0.),
            action_space=SimpleNamespace(low=np.array([-2.]*7+[-.9]+[-2.]*7+[-.9]),
                high=np.array([2.]*7+[.05]+[2.]*7+[.05])),last_action=np.zeros(16,np.float32))
        actions=TracingTiming('paused').actions(env,client,{'max_steps':32,'execute_horizon':16},Owner().event)
        with self.assertRaisesRegex(ValueError,'original range rejection'):next(actions)
        self.assertEqual(len(events),1);self.assertIsNone(events[0]['candidate_prefix'])
        self.assertFalse(events[0]['response_valid']);self.assertIn('rejected_input_file',events[0]['response_diagnostic'])
        self.assertIsNone(client.response_capture);self.assertEqual(env.data.time,0.)

    def test_trace_injection_is_factory_scoped_and_preserves_other_backends(self):
        class BaseService:
            def __init__(self,**kwargs): self.dependencies=kwargs
        factory=install_isolated(BaseService)
        service=factory()
        self.assertIs(service.dependencies['policy_client_factory'],traced_client)
        self.assertIs(service.dependencies['policy_timing_class'],TracingTiming)
        self.assertEqual(set(service.dependencies),{'policy_client_factory','policy_timing_class'})
        self.assertIs(install_isolated(factory),factory)
        marker=object()
        with patch('harness.policy_trace.make_client',return_value=marker) as fallback:
            self.assertIs(traced_client({'backend':'act'},seed=3),marker)
        fallback.assert_called_once_with({'backend':'act'},3)


if __name__=='__main__':unittest.main()
