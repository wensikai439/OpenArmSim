"""Local-only official LeRobot PI05 inference; no robot or camera drivers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import uuid

import numpy as np

from .policy_wire import PROTOCOL, TASK, IMAGE_SHAPES, MAX_BYTES, decode

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/pi05/checkpoint"
ACTION_NAMES = [name for side in range(2) for name in [*(f"qpos_{side}_{j}" for j in range(7)), f"gripper_{side}"]]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_checkpoint_config(config, pre, post):
    expected = {"type": "pi05", "use_relative_actions": False, "use_visual_memory": False,
                "use_proprioceptive_memory": False, "action_feature_names": ACTION_NAMES,
                "n_obs_steps": 1, "chunk_size": 50, "n_action_steps": 50, "num_inference_steps": 10,
                "dtype": "bfloat16", "rtc_config": None,
                "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("checkpoint does not match the reviewed OpenArm PI05 contract")
    expected_inputs = {"observation.state": {"type": "STATE", "shape": [16]}}
    expected_inputs.update({"observation.images."+name: {"type": "VISUAL", "shape": [3, shape[0], shape[1]]} for name, shape in IMAGE_SHAPES.items()})
    if config.get("input_features") != expected_inputs or config.get("output_features") != {"action": {"type": "ACTION", "shape": [16]}}:
        raise ValueError("checkpoint camera/state/action dimensions mismatch")
    for pipeline, name in ((pre, "relative_actions_processor"), (post, "absolute_actions_processor")):
        steps = [s for s in pipeline["steps"] if s["registry_name"] == name]
        if len(steps) != 1 or steps[0]["config"].get("enabled") is not False:
            raise ValueError("relative/absolute conversion must be explicitly disabled")
    for pipeline, name in ((pre, "normalizer_processor"), (post, "unnormalizer_processor")):
        steps = [s for s in pipeline["steps"] if s["registry_name"] == name]
        if len(steps) != 1 or steps[0]["config"].get("norm_map") != expected["normalization_mapping"]:
            raise ValueError("stored processing chain must preserve QUANTILES normalization")


def audit(checkpoint=CHECKPOINT):
    checkpoint = Path(checkpoint)
    if not (checkpoint.parent / "source.json").is_file() or not checkpoint.is_dir():
        raise ValueError("本地 OpenArm π0.5 检查点缺失；需先取得公司模型复制授权并核验副本。")
    source = read_json(checkpoint.parent / "source.json")
    if not source.get("remote_hash_verified") or not source.get("local_copy_complete"):
        raise ValueError("checkpoint copy and remote content hashes have not been verified")
    hashes = {}
    for name, item in source["files"].items():
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint.resolve()) or not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("checkpoint file missing or size changed: " + name)
        hashes[name] = digest(path)
        if hashes[name] != item["sha256"]:
            raise ValueError("checkpoint content changed: " + name)
    required = {"config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
                "policy_preprocessor_step_3_normalizer_processor.safetensors", "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "train_config.json"}
    if set(hashes) != required:
        raise ValueError("checkpoint inventory differs from reviewed files")
    config = read_json(checkpoint / "config.json")
    validate_checkpoint_config(config, read_json(checkpoint / "policy_preprocessor.json"), read_json(checkpoint / "policy_postprocessor.json"))
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"protocol": PROTOCOL, "checkpoint": identity, "files_sha256": hashes,
            "action_semantics": "openarm_absolute_joint16", "state_semantics": "openarm_encoder16",
            "camera_shapes": {k: list(v) for k, v in IMAGE_SHAPES.items()}, "chunk_size": 50,
            "num_inference_steps": 10, "task": TASK, "physical_transfer_validated": False}


class Policy:
    def __init__(self, device="cuda"):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        import lerobot

        self.metadata = audit()
        runtime = read_json(ROOT / "sim/learning.json")
        actual_source = Path(lerobot.__file__).resolve().parent.parent
        revision = subprocess.check_output(["git", "-C", str(actual_source), "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(["git", "-C", str(actual_source), "status", "--porcelain", "--untracked-files=no"], text=True)
        if actual_source != Path(runtime["source"]).resolve() or revision != runtime["source_revision"] or changes.strip():
            raise ValueError("PI05 must use the pinned clean official LeRobot runtime")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU fallback")
        torch.set_num_threads(4)
        self.torch = torch
        config = PreTrainedConfig.from_pretrained(str(CHECKPOINT), local_files_only=True)
        config.device = device
        config.compile_model = False
        config.gradient_checkpointing = False  # inference only; no learned parameter changes
        self.model = PI05Policy(config)
        # The upstream from_pretrained currently catches loading exceptions.
        # Use its key conversion but let strict load failures stop this server.
        weights = load_file(str(CHECKPOINT / "model.safetensors"), device="cpu")
        converted = self.model._fix_pytorch_state_dict_keys(weights, config)
        converted = {k if k.startswith("model.") else "model."+k: v for k, v in converted.items()}
        self.model.load_state_dict(converted, strict=True)
        # Check every mapped tensor, including expected dtype conversion.
        actual = self.model.state_dict()
        for name, tensor in converted.items():
            if not torch.equal(actual[name].detach().cpu(), tensor.to(dtype=actual[name].dtype)):
                raise RuntimeError("loaded tensor verification failed: " + name)
        del weights, converted, actual
        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT / "tokenizer"), local_files_only=True,
                                                   extra_special_tokens={"image_token": "<image>"})
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=str(CHECKPOINT),
            preprocessor_overrides={"device_processor": {"device": device, "float_dtype": None},
                                    "tokenizer_processor": {"tokenizer": tokenizer}},
            postprocessor_overrides={"device_processor": {"device": "cpu", "float_dtype": None}})
        self.device = device
        self.metadata.update(device=device, strict_weights_verified=True, runtime_revision=revision,
            adapter_sha256=digest(Path(__file__)), wire_sha256=digest(ROOT / "sim/policy_wire.py"),
            packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "lerobot", "safetensors")})
        self.session = None
        self.deadline = 0.
        self.sequence = 0

    def begin(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if self.session and time.monotonic() < self.deadline:
            raise ValueError("a policy session is already active")
        self.torch.manual_seed(seed)
        self.model.reset()
        self.session, self.sequence, self.deadline = uuid.uuid4().hex, 0, time.monotonic()+180
        return {"session": self.session, "seed": seed, "metadata": self.metadata}

    def infer(self, data):
        if not self.session or data["session"].item() != self.session or time.monotonic() >= self.deadline:
            raise ValueError("missing, stale or mismatched policy session")
        if data["sequence"].item() != self.sequence:
            raise ValueError("observation sequence mismatch; do not retry uncertain inference")
        if self.sequence >= 24:
            raise ValueError("policy session exhausted its 1200-action budget")
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(data["state"].copy()), "task": TASK}
        for name in IMAGE_SHAPES:
            batch["observation.images."+name] = torch.from_numpy(data[name].copy()).permute(2, 0, 1).float().div(255).contiguous()
        started = time.monotonic()
        # Consume the sequence before inference; failures cannot be retried as if
        # they had not advanced model RNG/state.
        self.sequence += 1
        with torch.inference_mode():
            actions = self.post(self.model.predict_action_chunk(self.pre(batch)))[0].cpu().float().numpy()
        if actions.shape != (50, 16) or not np.isfinite(actions).all():
            raise RuntimeError("invalid model output")
        self.deadline = time.monotonic()+180
        return {"actions": actions.tolist(), "sequence": self.sequence-1,
                "inference_seconds": time.monotonic()-started}

    def end(self, session):
        if session != self.session:
            raise ValueError("session ownership mismatch")
        self.session = None
        self.model.reset()


class Handler(BaseHTTPRequestHandler):
    def check_local(self):
        port = self.server.server_port
        if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"} or self.headers.get("Origin"):
            raise ValueError("local programmatic requests only")

    def reply(self, status, body):
        data = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        try:
            self.check_local()
            if self.path != "/metadata":
                raise ValueError("unknown inference endpoint")
            self.reply(200, self.server.policy.metadata)
        except ValueError as exc:
            self.reply(400, {"error": str(exc)})

    def do_POST(self):
        try:
            self.check_local()
            self.connection.settimeout(15)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= (MAX_BYTES if self.path == "/infer" else 1024):
                raise ValueError("invalid request size")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("incomplete request")
            if self.path == "/infer":
                if self.headers.get("Content-Type") != "application/x-npz":
                    raise ValueError("expected NumPy observation archive")
                result = self.server.policy.infer(decode(body))
            elif self.path == "/session":
                data = json.loads(body)
                if set(data) != {"seed"}:
                    raise ValueError("unexpected session arguments")
                result = self.server.policy.begin(data["seed"])
            elif self.path == "/close":
                data = json.loads(body)
                self.server.policy.end(data["session"])
                result = {"closed": True}
            else:
                raise ValueError("unknown inference endpoint")
            self.reply(200, result)
        except (ValueError, KeyError) as exc:
            self.reply(400, {"error": str(exc)})
        except Exception as exc:
            self.reply(500, {"error": type(exc).__name__+": "+str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(audit(), indent=2))
        return
    if not 1024 <= args.port <= 65535:
        raise ValueError("port must be 1024..65535")
    with tempfile.TemporaryDirectory(prefix="openarm-pi05-") as cache:
        os.environ.update(HF_HOME=cache, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_HOME=cache)
        policy = Policy(args.device)
        server = HTTPServer(("127.0.0.1", args.port), Handler)
        server.policy = policy
        record = ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S")+"-policy-server-"+uuid.uuid4().hex[:6])
        record.mkdir()
        (record / "manifest.json").write_text(json.dumps({"schema": "openarm-policy-server/1", "kind": "policy-server",
            "scope": "local_inference_only", "metadata": policy.metadata, "adapter_sha256": digest(Path(__file__)),
            "wire_sha256": digest(ROOT / "sim/policy_wire.py")}, indent=2)+"\n")
        print(json.dumps({"status": "ready", "port": args.port, "metadata": policy.metadata}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            (record / "summary.json").write_text(json.dumps({"status": "stopped", "task_performance_measured": False})+"\n")


if __name__ == "__main__":
    main()
