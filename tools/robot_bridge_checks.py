"""Local loopback integration of the real harness with kinematic TEST DOUBLES.

This never connects to SSH, a model API, a camera or CAN. HTTP, ASPIRE execution,
session ownership and failure reporting are real software paths; physics and
grasp outcomes are fixtures, not measured robot success.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from deploy import robot
from deploy.relative import CONTRACT
from deploy.relative_checks import Fixture
from deploy.service import Service,Journal,audit_log,serve

ROOT=Path(__file__).resolve().parents[1]


class ProtocolFixture(Fixture):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.hardware={'startup_target':[0.]*16}
        self.failure=None;self.real_device_calls=0
        self.servo.state={'q':[0.]*16,'timestamp_unix_s':time.time()}
    def begin_session(self):
        self.servo.state['timestamp_unix_s']=time.time()
        return super().begin_session()
    def observe(self):
        if self.failure:raise RuntimeError('test double stopped; fresh observation unavailable')
        return super().observe()
    def stop(self,reason):
        if self.failure is None:self.failure=reason


class RobotBridgeChecks(unittest.TestCase):
    def execute(self,fixture,*,agent=False):
        official=Path('/home/wsk/ASPIRE/libero-pro/source/src')
        sys.path.insert(0,str(official))
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/'.runtime',prefix='bridge-') as td:
                root=Path(td);(root/'app/skill').mkdir(parents=True);(root/'config').mkdir();(root/'.runtime').mkdir()
                program=(ROOT/'deploy/relative_program.py').read_bytes()
                (root/'app/skill/program.py').write_bytes(program)
                config={'robot_contract':CONTRACT,'program_sha256':hashlib.sha256(program).hexdigest()}
                journal=Journal(root/'service-record',{'scope':'local kinematic double; no hardware or physics'})
                fixture.record=journal.record
                svc=Service(config,{'calibration_verified':True,'missing':[]},journal,fixture)
                server=serve(svc,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                token=root/'.runtime/session-token';token.write_text(svc.token);token.chmod(0o600)
                config['robot_service']='http://127.0.0.1:'+str(server.server_port)
                (root/'config/robot.json').write_text(json.dumps(config))
                output=root/'skill-record';output.mkdir()
                try:
                    with patch.object(robot,'ROOT',root):
                        client=robot.Robot(config,execute=True,journal=journal)
                        if agent:
                            replies=[{'choices':[{'message':{'content':None,'tool_calls':[{
                                'id':'fixture-status','type':'function','function':{'name':'get_status','arguments':'{}'}}]}}]},
                                {'choices':[{'message':{'content':None,'tool_calls':[{
                                    'id':'fixture-skill','type':'function','function':{'name':'run_bottle_skill','arguments':'{}'}}]}}]},
                                {'choices':[{'message':{'content':'本地软件协议测试结束。'}}]}]
                            # Canned model replies test routing ONLY, not model intelligence.
                            with patch.object(robot,'model',side_effect=replies) as model:
                                chat=robot.chat('本地测试替身任务',client,output)
                            self.assertEqual(model.call_count,3)
                            self.assertEqual([r['tool'] for r in chat['tools']],['get_status','run_bottle_skill'])
                            result=json.loads((output/'skill-result.json').read_text())
                        else:result=robot.run_skill(client,output)
                        session_released=svc.session is None
                    saved=json.loads((output/'skill-result.json').read_text())
                    self.assertEqual(saved['status'],result['status'])
                    result['test_service_calls']=svc.calls
                    result['test_fixture_completed']=list(fixture.completed)
                    result['test_session_released']=session_released
                    self.assertEqual(fixture.real_device_calls,0)
                finally:
                    server.shutdown();server.server_close();thread.join(2);journal.close()
                self.assertTrue(audit_log(journal.folder)['sha256_chain_valid'])
                return result
        finally:sys.path.remove(str(official))

    def test_full_http_aspireskill_flow_and_bounded_recovery(self):
        for args in [dict(x=.46),dict(x=.50,fail_close=True),dict(x=.54,fail_receiver=True)]:
            with self.subTest(args=args):
                result=self.execute(ProtocolFixture(**args))
                self.assertEqual(result['status'],'program_finished',result['error'])
                # This success flag belongs ONLY to a synthetic contract double.
                self.assertEqual(result['test_fixture_completed'],['handover','placement'])
                self.assertTrue(result['test_session_released'])
    def test_wrong_initial_pose_is_returned_before_any_tool_dispatch(self):
        fixture=ProtocolFixture();fixture.servo.state['q'][0]=.1
        result=self.execute(fixture)
        self.assertEqual(result['status'],'program_failed');self.assertIn('ACT startup pose',result['error'])
        self.assertEqual(result['test_service_calls'],0);self.assertEqual(result['motion_requests_attempted'],0)
        self.assertTrue(result['test_session_released'])
    def test_http_collision_failure_keeps_original_error_and_valid_terminal_record(self):
        fixture=ProtocolFixture();fixture.reject_motion=True
        result=self.execute(fixture)
        self.assertEqual(result['status'],'program_failed');self.assertIn('injected collision rejection',result['error'])
        self.assertIsNone(result['task_success']);self.assertIsNone(result['final_observation'])
        self.assertIsNotNone(result['last_valid_observation']);self.assertTrue(result['test_session_released'])
    def test_agent_tool_routing_reaches_full_skill_with_stubbed_model_only(self):
        result=self.execute(ProtocolFixture(),agent=True)
        self.assertEqual(result['status'],'program_finished')
        self.assertEqual(result['test_fixture_completed'],['handover','placement'])


if __name__=='__main__':unittest.main()
