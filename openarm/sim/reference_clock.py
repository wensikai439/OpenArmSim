"""Locally audited logger timestamps; never a hardware or camera clock claim."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load():
    data = json.loads((ROOT / "reference/acquisition.json").read_text())
    raw = (ROOT / "reference/real_trajectory.json").read_bytes()
    if data.get("schema") != "openarm-reference-clock/1" or data.get("reference_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("reference clock provenance differs from the audited trajectories")
    reference = json.loads(raw)["episodes"]
    if set(data["episodes"]) != set(reference):
        raise ValueError("logger clock does not cover every reference episode")
    for key, row in data["episodes"].items():
        t = np.asarray(row["recorded_seconds"], float)
        if (t.shape != (reference[key]["frames"],) or not np.isfinite(t).all()
                or t[0] != 0 or np.any(np.diff(t) <= 0)
                or not row["alignment"]["all_frames_and_all_16_channels_checked"]):
            raise ValueError("invalid or unaligned logger time series")
    return data


def summary():
    data = load()
    return {"timestamp_semantics": data["timestamp_semantics"],
            "episodes": {k: v["intervals"] for k, v in data["episodes"].items()},
            "head_camera": data["head_camera_evidence"], "limitations": data["limitations"]}
