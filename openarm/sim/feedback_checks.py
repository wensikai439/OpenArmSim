"""Internal observability must not change physics or public policy evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .environment import OpenArmEnv
from .feedback import FeedbackLog
from .recording import Recorder


class FeedbackChecks(unittest.TestCase):
    def setUp(self):
        self.env = OpenArmEnv(images=False)
        self.env.reset(seed=0)

    def tearDown(self):
        self.env.close()

    def test_sampling_has_no_state_control_or_clock_side_effect(self):
        env = self.env
        before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.last_action.copy()]
        with tempfile.TemporaryDirectory() as folder:
            feedback = FeedbackLog(Path(folder), env)
            feedback.read(env, env.state())
            result = feedback.finish()
        for a,b in zip(before, [env.data.qpos, env.data.qvel, env.data.ctrl, env.last_action]):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(env.tick,0); self.assertEqual(env.data.time,0.)
        self.assertFalse(result['policy_input'])
        self.assertEqual(result['max_bottle_lift_m'],0.)
        self.assertEqual(result['max_two_finger_hold_seconds'],{'left':0.,'right':0.})

    def test_recording_saves_internal_feedback_without_public_summary_leak(self):
        env = self.env
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(env,'feedback-check',record_images=False,seed=0,root=folder)
            before = recorder.observation(env); action = env.last_action.copy()
            _,_,term,trunc,info = env.step(action)
            recorder.append(before,action,recorder.observation(env),info,terminated=term,truncated=trunc)
            recorder.finish('completed',info)
            private = json.loads((recorder.path/'internal-summary.json').read_text())
            public = json.loads((recorder.path/'summary.json').read_text())
            self.assertEqual(private['samples'],2)
            self.assertNotIn('min_tcp_bottle_distance_m',json.dumps(public))
            self.assertEqual(len((recorder.path/'internal.jsonl').read_text().splitlines()),2)

    def test_native_gripper_adapter_is_explicit_and_never_clips_prediction(self):
        env = self.env; raw = env.last_action.copy(); raw[15] = .056
        with self.assertRaisesRegex(ValueError,'gripper'):
            env.prepare_policy_action(raw)
        applied,_ = env.prepare_policy_action(raw,native_fingers=True)
        self.assertEqual(applied[15],raw[15])
        env.step_policy(applied,native_fingers=True)
        self.assertEqual(env.tick,1)
        self.assertEqual(env.last_action[15],raw[15])
        raw[15] = .2
        with self.assertRaisesRegex(ValueError,'gripper'):
            env.prepare_policy_action(raw,native_fingers=True)

    def test_native_gripper_adapter_retains_arm_jump_limit(self):
        env = self.env; raw = env.last_action.copy(); raw[:7] += 2
        applied,_ = env.prepare_policy_action(raw,native_fingers=True)
        self.assertLessEqual(np.linalg.norm(applied[:7]-env.state()[:7]),1.000001)


if __name__ == '__main__':
    unittest.main()
