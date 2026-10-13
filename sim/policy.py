"""Bounded client for the official OpenPI NumPy/MessagePack websocket protocol.

Wire representation follows openpi-client/msgpack_numpy.py. Observation keys
are checkpoint-specific and must be supplied in a config; never infer a
LIBERO checkpoint's 7-D actions as OpenArm's 16-D commands.
"""
from __future__ import annotations

import json
import http.client
from pathlib import Path
from urllib.parse import urlsplit

import msgpack
import numpy as np
from websockets.sync.client import connect
from .policy_wire import PROTOCOL, IMAGE_SHAPES, encode


def pack_array(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OVc":
            raise ValueError("unsupported NumPy dtype")
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(type(value).__name__)


def unpack_array(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "OVc":
            raise ValueError("unsupported array response")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def load_config(path):
    config = json.loads(Path(path).read_text())
    config.setdefault('policy_type', 'pi05')
    if config['policy_type'] not in {'pi05', 'act'}:
        raise ValueError('policy_type must be pi05 or act')
    config.setdefault("backend", "openpi_websocket")
    if config["backend"] not in {"openpi_websocket", "lerobot_http"}:
        raise ValueError("unsupported local policy backend")
    required = {"uri", "checkpoint", "state_key", "prompt_key", "image_keys", "action_key", "action_semantics", "state_semantics", "image_layout", "capability"}
    if required - config.keys():
        raise ValueError("policy config is incomplete: " + ", ".join(sorted(required - config.keys())))
    if urlsplit(config["uri"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("policy endpoint must be local; the simulator never invokes the SSH robot")
    if urlsplit(config["uri"]).username or urlsplit(config["uri"]).password:
        raise ValueError("do not put credentials in the policy config")
    uri = urlsplit(config["uri"])
    schemes = {"http"} if config["backend"] == "lerobot_http" else {"ws", "wss"}
    if uri.scheme not in schemes or uri.query or uri.fragment:
        raise ValueError("policy URI does not match its declared backend")
    if config["backend"] == "lerobot_http" and (uri.path not in {"", "/"} or config["image_layout"] != "HWC"):
        raise ValueError("LeRobot bridge requires a root HTTP endpoint and HWC RGB")
    if config["action_semantics"] != "openarm_absolute_joint16" or config["state_semantics"] != "openarm_encoder16":
        raise ValueError("policy must declare the OpenArm state and action contract")
    if set(config["image_keys"]) != {"head", "left", "right"} or config["image_layout"] not in {"HWC", "CHW"}:
        raise ValueError("declare all three cameras and image layout")
    if config["capability"] != "atomic_right_to_left_handover":
        raise ValueError("checkpoint capability is incompatible with this task")
    if not config["checkpoint"] or "REPLACE" in config["checkpoint"]:
        raise ValueError("record the actual checkpoint identity before running")
    config.setdefault("execute_horizon", 50)
    config.setdefault('chunk_size', 32 if config['policy_type']=='act' else 50)
    if type(config['chunk_size']) is not int or not 1 <= config['chunk_size'] <= 200:
        raise ValueError('invalid declared policy chunk size')
    if config['policy_type']=='pi05' and config['backend']=='lerobot_http' and config['chunk_size']!=50:
        raise ValueError('reviewed LeRobot PI05 requires a 50-step chunk')
    config.setdefault("max_steps", 512)
    config.setdefault("timeout_seconds", 30)
    config.setdefault("timing_mode", "paused")
    if config["timing_mode"] not in {"paused", "measured_delay"}:
        raise ValueError("timing_mode must be paused or measured_delay")
    if not 1 <= config["execute_horizon"] <= min(50,config['chunk_size']) or not 1 <= config["max_steps"] <= 1200:
        raise ValueError("invalid policy budget")
    if not 1 <= config["timeout_seconds"] <= 60:
        raise ValueError("policy timeout must be 1..60 seconds")
    keys = [config["state_key"], config["prompt_key"], *config["image_keys"].values()]
    if len(set(keys)) != len(keys) or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("observation keys must be unique non-empty strings")
    if config["backend"] == "lerobot_http":
        expected = {"state_key": "observation.state", "prompt_key": "task", "action_key": "actions",
                    "image_keys": {name:"observation.images."+name for name in IMAGE_SHAPES}}
        if any(config.get(k) != v for k,v in expected.items()):
            raise ValueError("LeRobot bridge observation keys are fixed by the reviewed checkpoint")
    return config


def validate_chunk(result, env, config):
    actions = np.asarray(result[config["action_key"]], np.float32)
    if actions.ndim != 2 or actions.shape[1] != 16 or not 1 <= len(actions) <= 200 or not np.isfinite(actions).all():
        raise ValueError("policy response must have shape [1..200,16] and finite values")
    for action in actions:
        env.prepare_policy_action(action)
    return actions[:config["execute_horizon"]]


class OpenPiClient:
    def __init__(self, config):
        self.config = config
        self.socket = None

    def __enter__(self):
        self.socket = connect(self.config["uri"], compression=None, max_size=32 * 1024 * 1024,
                              open_timeout=self.config["timeout_seconds"], close_timeout=2)
        try:
            self.metadata = self._receive()
        except Exception:
            self.socket.close()
            raise
        return self

    def _receive(self):
        data = self.socket.recv(timeout=self.config["timeout_seconds"])
        if isinstance(data, str):
            raise RuntimeError("policy server returned an error response")
        return msgpack.unpackb(data, object_hook=unpack_array, raw=False, strict_map_key=False)

    def infer(self, env, prompt):
        config = self.config
        obs = {config["state_key"]: env.state(), config["prompt_key"]: prompt}
        for camera, key in config["image_keys"].items():
            rgb = env.render(camera)
            obs[key] = rgb if config["image_layout"] == "HWC" else np.moveaxis(rgb, -1, 0)
        self.socket.send(msgpack.packb(obs, default=pack_array))
        result = self._receive()
        # Validate shape/gripper semantics before executing a prefix. Arm
        # targets are transformed by the reviewed deployment L2/limit adapter
        # against fresh feedback at each control step, never against stale
        # feedback from the start of an entire action chunk.
        return validate_chunk(result, env, config)

    def __exit__(self, *_):
        if self.socket:
            self.socket.close()


class LeRobotClient:
    """Bounded local HTTP bridge; exact RGB, verified identity, ordered session."""
    def __init__(self, config, seed=0):
        self.config, self.seed = config, seed
        self.session, self.sequence = None, 0

    def request(self, path, body=None, content_type="application/json"):
        uri = urlsplit(self.config["uri"])
        connection = http.client.HTTPConnection(uri.hostname, uri.port, timeout=self.config["timeout_seconds"])
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode()
        try:
            connection.request("POST" if body is not None else "GET", path, body=body,
                               headers={"Content-Type": content_type})
            response = connection.getresponse()
            raw = response.read(1024*1024+1)
            if len(raw) > 1024*1024:
                raise ValueError("policy response exceeds budget")
            result = json.loads(raw)
            if response.status != 200:
                raise RuntimeError(result.get("error", "policy endpoint failed"))
            return result
        finally:
            connection.close()

    def __enter__(self):
        self.metadata = self.request("/metadata")
        expected = {"protocol": PROTOCOL, "checkpoint": self.config["checkpoint"],
                    "action_semantics": self.config["action_semantics"],
                    "state_semantics": self.config["state_semantics"],
                    "camera_shapes": {k:list(v) for k,v in IMAGE_SHAPES.items()},
                    "strict_weights_verified": True, "chunk_size": self.config.get('chunk_size',50), "num_inference_steps": 10}
        if self.config.get('policy_type') == 'act':
            expected.update(policy_type='act', num_inference_steps=0)
        if any(self.metadata.get(k) != value for k, value in expected.items()):
            raise ValueError("policy server identity or I/O contract does not match configuration")
        result = self.request("/session", {"seed": self.seed})
        self.session = result["session"]
        if result["metadata"] != self.metadata:
            self.__exit__()
            raise ValueError("policy server changed during session creation")
        return self

    def infer(self, env, prompt):
        data = {"state": env.state(), "task": np.array(prompt), "session": np.array(self.session),
                "sequence": np.array(self.sequence, np.int64)}
        data.update({name: env.render(name) for name in IMAGE_SHAPES})
        result = self.request("/infer", encode(data), "application/x-npz")
        if result.get("sequence") != self.sequence:
            raise ValueError("policy response belongs to a different observation")
        self.sequence += 1
        chunk_size = self.config.get('chunk_size',50)
        if np.asarray(result["actions"]).shape != (chunk_size, 16):
            raise ValueError(f'LeRobot response must match the declared {chunk_size}x16 action chunk')
        return validate_chunk(result, env, self.config)

    def __exit__(self, *_):
        if self.session:
            try:
                self.request("/close", {"session": self.session})
            finally:
                self.session = None


def make_client(config, seed=0):
    return LeRobotClient(config, seed) if config.get("backend") == "lerobot_http" else OpenPiClient(config)
