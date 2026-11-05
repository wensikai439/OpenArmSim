"""Append-only transition log plus lossless synchronized camera frames."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .environment import ROOT, TASK


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serial, allow_nan=False) + "\n")


class Recorder:
    def __init__(self, env, kind, *, record_images, seed, details=None, root=None):
        env.require_current_sources()
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + kind + "-" + uuid.uuid4().hex[:6]
        self.path = (Path(root) if root else ROOT / "runs") / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.record_images = record_images
        self.count = 0
        self._closed = False
        self.started = time.monotonic()
        self.file = (self.path / "transitions.jsonl").open("w", buffering=1)
        self.manifest = {**env.manifest(), "run_id": run_id, "kind": kind, "seed": seed,
                         "task": TASK, "details": details or {},
                         "start_tick": env.tick, "start_sim_seconds": float(env.data.time),
                         "images": "synchronized lossless PNG every control tick" if record_images else "keyframes only; not VLA training data",
                         "format": "native transitions, not yet a LeRobot dataset",
                         "status": "running", "started_at_unix": time.time()}
        dump(self.path / "manifest.json", self.manifest)
        self._internal = None
        try:
            from .feedback import FeedbackLog
            self._internal = FeedbackLog(self.path, env)
            self._environment = env
        except Exception as exc:
            self.event({'type':'internal_feedback_error','error':str(exc)})

    def observation(self, env):
        obs = {"state": env.state(), "tick": env.tick, "sim_time": float(env.data.time)}
        if self.record_images:
            images = {}
            for camera in env.profile["camera_map"]:
                path = self.path / "images" / camera / f"{env.tick:06d}.png"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(env.render(camera)).save(path)
                images[camera] = str(path.relative_to(self.path))
            obs["images"] = images
        return obs

    def append(self, before, action, after, info, *, terminated=False, truncated=False, diagnostics=None):
        row = {"index": self.count, "observation": before, "action": action,
               "next_observation": after, "terminated": terminated, "truncated": truncated,
               "evaluation": info["evaluation"], "diagnostics": diagnostics,
               "wall_elapsed_s": time.monotonic() - self.started}
        self.file.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
        self.count += 1
        if self._internal is not None:
            try:
                self._internal.read(self._environment, action)
            except Exception as exc:
                self._internal.stream.close()
                self._internal = None
                self.event({'type':'internal_feedback_error','error':str(exc)})

    def event(self, event):
        with (self.path / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=serial, allow_nan=False) + "\n")

    def finish(self, status, info, *, error=None, metrics=None):
        if self._closed:
            return
        self.file.close()
        self._closed = True
        if self._internal is not None:
            try:
                self._internal.finish()
            except Exception as exc:
                self.event({'type':'internal_feedback_error','error':str(exc)})
        summary = {"status": status, "transitions": self.count,
                   "wall_seconds": time.monotonic() - self.started,
                   "sim_seconds": info["sim_time"], "evaluation": info["evaluation"],
                   "duration_sim_seconds": info['sim_time']-self.manifest['start_sim_seconds'],
                   "events": info["events"], "error": error, "metrics": metrics or {},
                   "transfer_ready": False}
        dump(self.path / "summary.json", summary)
        self.manifest["status"] = status
        dump(self.path / "manifest.json", self.manifest)
