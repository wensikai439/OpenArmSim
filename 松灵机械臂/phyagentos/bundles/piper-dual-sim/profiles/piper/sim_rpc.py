# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified 2026-10-15: standalone HTTP transport and direct shared-asset setup.
"""Owned-process PiPER RPC adapter; physics stays in piper_robocasa."""
from __future__ import annotations
import argparse, base64, hashlib, importlib, json, signal, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence
import imageio.v2 as imageio
import numpy as np

B2_RPC_SCHEMA = "piper_sim_b2_readonly/v1"
B2_ACTION_RPC_SCHEMA = "piper_sim_b2_actions/v1"
B2_CAMERAS = ("cam_right", "overview")
B2_ACTION_IDS = ("set_gripper", "move_arm", "grasp_bottle", "release_bottle", "return_arm_to_initial", "place_bottle_at_origin")

def expected_rpc_meta(
    instance_id: str, *, action_enabled: bool = False
) -> dict[str, Any]:
    """Return the exact identity contract for one B2 server."""
    return {
        "rpc_schema": B2_ACTION_RPC_SCHEMA if action_enabled else B2_RPC_SCHEMA,
        "robot": "piper_sim",
        "source_kind": "sim_rpc",
        "instance_id": str(instance_id),
        "camera_id": "cam_right",
        "logical_side": "right",
        "cameras": list(B2_CAMERAS),
        "read_only": not action_enabled,
        "actions": list(B2_ACTION_IDS) if action_enabled else [],
        "reset_on_connect": False,
    }

class PiperSimReadOnlyBackend:
    """Adapt A's task to the B2 frame/observe contract and optional actions."""

    def __init__(
        self,
        task: Any,
        instance_id: str,
        *,
        action_enabled: bool = False,
        reset_factory: Any = None,
    ) -> None:
        self._task = task
        self._reset_factory = reset_factory
        self._reset_receipts: dict[str, dict[str, Any]] = {}
        self._instance_id = str(instance_id)
        self.action_enabled = bool(action_enabled)
        self._last_captured: dict[str, dict[str, Any]] = {}

    def get_env_meta(self) -> dict[str, Any]:
        return expected_rpc_meta(self._instance_id, action_enabled=self.action_enabled)

    def capture_frames(self, cameras: Sequence[str] = B2_CAMERAS) -> dict[str, Any]:
        cameras = tuple(cameras)
        if cameras != B2_CAMERAS:
            raise ValueError("B2 capture must request exactly (cam_right, overview)")
        records = self._task.capture_frames(cameras)
        if not isinstance(records, list) or len(records) != len(cameras):
            raise ValueError("A capture_frames returned an unexpected frame list")
        frames = []
        captured: dict[str, dict[str, Any]] = {}
        output_root = Path(self._task.output).resolve()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("A capture_frames returned a non-object record")
            camera_id = record.get("camera_id")
            if camera_id not in B2_CAMERAS:
                raise ValueError("A returned a camera outside the B2 allowlist")
            path = (output_root / str(record["path"])).resolve()
            if output_root not in path.parents:
                raise ValueError("A frame path escaped the owned output directory")
            png_bytes = path.read_bytes()
            record_sha256 = record.get("sha256")
            if not isinstance(record_sha256, str) or len(record_sha256) != 64:
                raise ValueError("A frame is missing its PNG SHA-256")
            if hashlib.sha256(png_bytes).hexdigest() != record_sha256:
                raise ValueError("A frame PNG SHA-256 mismatch")
            rgb = np.asarray(imageio.imread(BytesIO(png_bytes)))
            if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
                raise ValueError("A frame must decode as 640x480 uint8 RGB")
            item = dict(record)
            item["rgb"] = rgb
            captured[camera_id] = item
            frames.append(item)
        self._last_captured = captured
        return {"instance_id": self._instance_id, "frames": frames}

    def observe(self) -> dict[str, Any]:
        if set(self._last_captured) != set(B2_CAMERAS):
            raise RuntimeError(
                "capture_frames(cam_right, overview) is required before observe"
            )
        state = self._task.observe()
        if not isinstance(state, dict):
            raise ValueError("A observe returned a non-object state")
        bindings = {}
        for camera_id, frame in self._last_captured.items():
            bindings[camera_id] = {
                key: frame[key]
                for key in (
                    "frame_id",
                    "episode_id",
                    "state_index",
                    "physics_step",
                    "sim_time_s",
                )
            }
        result = dict(state)
        result["b2_frame_bindings"] = bindings
        result["b2_instance_id"] = self._instance_id
        return result

    def step_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Delegate one existing A primitive without duplicating control logic."""
        if not self.action_enabled:
            raise RuntimeError("SIM action RPC is disabled for this server")
        result = self._task.step(action)
        if not isinstance(result, dict):
            raise ValueError("A task.step returned a non-object result")
        return result

    def reset_scene(self, request_id: str) -> dict[str, Any]:
        """Rebuild the owned simulator on its RPC thread, with a fresh episode."""
        if not self.action_enabled or not callable(self._reset_factory):
            raise RuntimeError("scene reset is unavailable")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 1024:
            raise ValueError("reset requires a bounded request identifier")
        if request_id in self._reset_receipts:
            receipt = self._reset_receipts[request_id]
            if receipt["status"] != "succeeded":
                raise RuntimeError(
                    "previous reset failed or is unresolved; not replayed"
                )
            return dict(receipt)
        receipt = {"request_id": request_id, "status": "started"}
        self._reset_receipts[request_id] = receipt
        old_output = str(self._task.output) if self._task is not None else None
        try:
            self.close()
            self._last_captured = {}
            self._task = self._reset_factory()
            self.capture_frames()
            state = self.observe()
            receipt.update(
                status="succeeded",
                previous_output=old_output,
                output=str(self._task.output),
                episode_id=state["episode_id"],
                live_overview_path=str(Path(self._task.output) / "live_overview.jpg"),
            )
        except Exception as exc:
            receipt.update(status="failed", error=str(exc))
            raise
        return dict(receipt)

    def close(self) -> None:
        if self._task is None:
            return
        try:
            save_artifacts = getattr(self._task, "save_artifacts", None)
            if callable(save_artifacts):
                save_artifacts()
        finally:
            try:
                self._task.close()
            finally:
                self._task = None

class PiperSimEnvFacade:
    """Run RPC and MuJoCo rendering on the same owning main thread."""
    def __init__(self, backend):
        self._backend = backend
        self._rpc = {}
        self._readonly_methods = set()
        self._stopped = threading.Event()
        self._register_rpc()

    def _register_rpc(self):
        self._rpc.update({"env.get_env_meta": self._backend.get_env_meta,
                          "env.capture_frames": self._backend.capture_frames,
                          "env.observe": self._backend.observe,
                          "shutdown": self.shutdown})
        self._readonly_methods.update({"env.get_env_meta", "env.capture_frames", "env.observe"})
        if self._backend.action_enabled:
            self._rpc["env.step_action"] = self.step_action
            if callable(self._backend._reset_factory):
                self._rpc["env.reset_scene"] = self._backend.reset_scene

    def step_action(self, action):
        return self._backend.step_action(action)

    def shutdown(self):
        self._stopped.set()
        return {"status": "shutdown_requested"}

    def close(self):
        self._backend.close()

    def serve(self, *, transport, host, port, parent_watch=False):
        if transport != "http" or host != "127.0.0.1":
            raise ValueError("This adapter serves only its loopback HTTP endpoint")
        owner = self
        def encode(value):
            if isinstance(value, np.ndarray):
                return {"__ndarray__": base64.b64encode(value.tobytes()).decode("ascii"),
                        "dtype": str(value.dtype), "shape": list(value.shape)}
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Path):
                return str(value)
            raise TypeError("Unsupported RPC value: " + type(value).__name__)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                if self.path != "/call":
                    self.send_error(404)
                    return
                try:
                    call = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    method = call["method"]
                    if method not in owner._rpc:
                        raise ValueError("Unknown PiPER RPC method: " + str(method))
                    args, kwargs = call.get("args", []), call.get("kwargs", {})
                    if not isinstance(args, list) or not isinstance(kwargs, dict):
                        raise ValueError("RPC args/kwargs must be a list/object")
                    result = owner._rpc[method](*args, **kwargs)
                    payload = json.dumps({"ok": True, "result": result}, default=encode, allow_nan=False).encode()
                except Exception as error:
                    payload = json.dumps({"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        def parent_closed():
            while sys.stdin.buffer.read(1):
                pass
            self._stopped.set()
        if parent_watch:
            threading.Thread(target=parent_closed, daemon=True).start()
        previous = signal.signal(signal.SIGTERM, lambda *_: self._stopped.set())
        try:
            with HTTPServer((host, port), Handler) as server:
                server.timeout = 0.5
                while not self._stopped.is_set():
                    server.handle_request()
        finally:
            signal.signal(signal.SIGTERM, previous)
            self.close()

def _make_real_backend(args: argparse.Namespace) -> PiperSimReadOnlyBackend:
    source_root = Path(args.a_source_root).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"A source root does not exist: {source_root}")
    sys.path.insert(0, str(source_root))
    variant = getattr(args, "task_variant", "tabletop")
    module_name, class_name = {
        "tabletop": ("tabletop_task", "PiperTabletopRightTask"),
        "photo-relay": ("photo_relay_task", "PiperPhotoRelayTask"),
        "air-handoff": ("air_handoff_task", "PiperAirHandoffTask"),
    }[variant]
    task_module = importlib.import_module(f"piper_robocasa.{module_name}")
    task_class = getattr(task_module, class_name)
    output_root = Path(args.output).resolve()

    def make_task(output: Path, episode_id: str) -> Any:
        task = task_class(
            output=output,
            render=True,
            run_id=episode_id,
            config_path=args.config_path,
        )
        if variant in {"photo-relay", "air-handoff"}:
            from live_overview import attach_live_overview

            attach_live_overview(task)
        return task

    def reset_factory() -> Any:
        import uuid

        episode_id = f"{args.instance_id}-reset-{uuid.uuid4().hex}"
        return make_task(output_root / "episodes" / episode_id, episode_id)

    return PiperSimReadOnlyBackend(
        make_task(output_root, args.instance_id),
        args.instance_id,
        action_enabled=args.enable_actions,
        reset_factory=reset_factory,
    )

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("http",), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--a-source-root", required=True)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--task-variant", choices=("air-handoff",), default="air-handoff")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--enable-actions", action="store_true")
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args(argv)
    if not args.render:
        raise ValueError("Native PiPER simulation requires explicit --render")
    backend = _make_real_backend(args)
    try:
        imports = {name: str(getattr(module, "__file__", "")) for name, module in sys.modules.copy().items()
                   if getattr(module, "__file__", None)}
        forbidden = {name: path for name, path in imports.items()
                     if name == "rpent" or name.startswith("rpent.") or "/projects/rpent/" in path}
        Path(args.output, "startup-imports.json").write_text(json.dumps(
            {"module_origins": imports, "sys_path": sys.path, "rpent_modules": forbidden,
             "adapter_root": str(Path(__file__).parent), "asset_root": str(Path(args.a_source_root).parent)}, indent=2))
        if forbidden:
            raise RuntimeError("RPent runtime imports are forbidden in this independent adapter")
        facade = PiperSimEnvFacade(backend)
    except BaseException:
        backend.close()
        raise
    facade.serve(transport=args.transport, host=args.host, port=args.port, parent_watch=args.parent_watch)
    return 0
