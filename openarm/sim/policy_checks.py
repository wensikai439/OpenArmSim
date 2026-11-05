"""No model/hardware calls: exact policy inputs, identity and session guards."""
import copy
from io import BytesIO
import unittest
from unittest.mock import Mock

import numpy as np

from .policy_wire import PROTOCOL, TASK, IMAGE_SHAPES, encode, decode
from .pi05_server import ACTION_NAMES, Policy, validate_checkpoint_config
from .policy import LeRobotClient


def observation():
    rng = np.random.default_rng(4)
    return {"state": rng.standard_normal(16).astype(np.float32), "task": np.array(TASK),
            "session": np.array("a"*32), "sequence": np.array(0, np.int64),
            **{name: rng.integers(0, 256, shape, dtype=np.uint8) for name, shape in IMAGE_SHAPES.items()}}


def metadata():
    return {"protocol": PROTOCOL, "checkpoint": "a"*64, "action_semantics": "openarm_absolute_joint16",
            "state_semantics": "openarm_encoder16", "camera_shapes": {k:list(v) for k,v in IMAGE_SHAPES.items()},
            "strict_weights_verified": True, "chunk_size": 50, "num_inference_steps": 10}


class PolicyChecks(unittest.TestCase):
    def test_wire_preserves_every_rgb_pixel_and_state_value(self):
        original = observation()
        restored = decode(encode(original))
        for key in original:
            np.testing.assert_array_equal(original[key], restored[key])
            self.assertEqual(original[key].dtype, restored[key].dtype)

    def test_bad_cameras_and_task_are_rejected_before_inference(self):
        data = observation()
        for key, bad in (("head", np.zeros((480, 640, 3), np.uint8)),
                         ("state", np.zeros(7, np.float32)), ("left", data["left"].astype(np.float32)),
                         ("task", np.array("pick up only"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                encode({**data, key: bad})

    def test_archive_cannot_load_pickled_objects_or_compressed_payloads(self):
        data = observation()
        buffer = BytesIO()
        np.savez(buffer, **{**data, "state": np.array([object()], dtype=object)})
        with self.assertRaises(ValueError):
            decode(buffer.getvalue())
        buffer = BytesIO()
        np.savez_compressed(buffer, **data)
        with self.assertRaises(ValueError):
            decode(buffer.getvalue())

    def test_checkpoint_contract_rejects_relative_actions_and_changed_normalizer(self):
        config = {"type": "pi05", "use_relative_actions": False, "use_visual_memory": False,
                  "use_proprioceptive_memory": False, "action_feature_names": ACTION_NAMES,
                  "n_obs_steps": 1, "chunk_size": 50, "n_action_steps": 50, "num_inference_steps": 10,
                  "dtype": "bfloat16", "rtc_config": None,
                  "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
                  "input_features": {"observation.state": {"type": "STATE", "shape": [16]},
                    **{"observation.images."+k: {"type": "VISUAL", "shape": [3,v[0],v[1]]} for k,v in IMAGE_SHAPES.items()}},
                  "output_features": {"action": {"type": "ACTION", "shape": [16]}}}
        pre = {"steps": [{"registry_name": "relative_actions_processor", "config": {"enabled": False}},
                         {"registry_name": "normalizer_processor", "config": {"norm_map": config["normalization_mapping"]}}]}
        post = {"steps": [{"registry_name": "absolute_actions_processor", "config": {"enabled": False}},
                          {"registry_name": "unnormalizer_processor", "config": {"norm_map": config["normalization_mapping"]}}]}
        validate_checkpoint_config(config, pre, post)
        with self.assertRaises(ValueError):
            validate_checkpoint_config({**config, "use_relative_actions": True}, pre, post)
        bad = copy.deepcopy(pre)
        bad["steps"][1]["config"]["norm_map"]["STATE"] = "MEAN_STD"
        with self.assertRaises(ValueError):
            validate_checkpoint_config(config, bad, post)

    def test_wrong_server_identity_never_starts_a_session(self):
        config = metadata()
        client = LeRobotClient(config)
        client.request = Mock(return_value={**metadata(), "checkpoint": "wrong"})
        with self.assertRaises(ValueError):
            client.__enter__()
        client.request.assert_called_once_with("/metadata")

    def test_stale_response_cannot_reach_the_action_adapter(self):
        data = observation()
        env = Mock()
        env.state.return_value = data["state"]
        env.render.side_effect = lambda name: data[name]
        client = LeRobotClient({"action_key": "actions", "execute_horizon": 50})
        client.session = "a"*32
        client.request = Mock(return_value={"sequence": 1, "actions": np.zeros((50,16)).tolist()})
        with self.assertRaises(ValueError):
            client.infer(env, TASK)
        env.prepare_policy_action.assert_not_called()

    def test_exclusive_session_seed_and_owner(self):
        policy = Policy.__new__(Policy)
        policy.torch, policy.model = Mock(), Mock()
        policy.metadata, policy.session, policy.deadline = metadata(), None, 0.
        with self.assertRaises(ValueError):
            policy.begin(True)
        session = policy.begin(42)
        with self.assertRaises(ValueError):
            policy.begin(99)
        with self.assertRaises(ValueError):
            policy.end("wrong-owner")
        policy.torch.manual_seed.assert_called_once_with(42)
        policy.end(session["session"])
        self.assertIsNone(policy.session)
