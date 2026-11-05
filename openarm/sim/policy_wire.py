"""Exact RGB/state transport shared by the simulator and LeRobot runtime."""
from io import BytesIO
import zipfile

import numpy as np

PROTOCOL = "openarm-lerobot-pi05/1"
TASK = "Pass the black bottle from the right hand to the left, then place it into the white box."
IMAGE_SHAPES = {"head": (960, 1280, 3), "left": (480, 640, 3), "right": (480, 640, 3)}
MAX_BYTES = 8 * 1024 * 1024
FIELDS = {"state", "task", "session", "sequence", *IMAGE_SHAPES}


def validate_observation(data):
    if set(data) != FIELDS:
        raise ValueError("observation fields differ from the reviewed policy contract")
    state = data["state"]
    if state.shape != (16,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("expected 16 finite float32 encoder values")
    for name, shape in IMAGE_SHAPES.items():
        if data[name].shape != shape or data[name].dtype != np.uint8:
            raise ValueError(f"{name} must be RGB uint8 {shape}")
    for key in ("task", "session"):
        if data[key].shape != () or data[key].dtype.kind != "U":
            raise ValueError(f"{key} must be a scalar string")
    if data["task"].item() != TASK:
        raise ValueError("this checkpoint supports the reviewed whole-task instruction only")
    if len(data["session"].item()) != 32:
        raise ValueError("invalid policy session")
    if data["sequence"].shape != () or data["sequence"].dtype != np.int64 or data["sequence"].item() < 0:
        raise ValueError("invalid observation sequence")


def encode(data):
    validate_observation(data)
    buffer = BytesIO()
    np.savez(buffer, **data)
    return buffer.getvalue()


def decode(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError("observation payload exceeds budget")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        infos = archive.infolist()
        if len(infos) != len(FIELDS) or {i.filename for i in infos} != {f+".npy" for f in FIELDS}:
            raise ValueError("unexpected observation archive members")
        if sum(i.file_size for i in infos) > MAX_BYTES or any(i.compress_type != zipfile.ZIP_STORED for i in infos):
            raise ValueError("only bounded uncompressed NumPy arrays are accepted")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        data = {key: archive[key] for key in FIELDS}
    validate_observation(data)
    return data
