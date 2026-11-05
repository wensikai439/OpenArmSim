#!/usr/bin/env python3
"""Turn a simulated episode into the same LeRobot dataset layout as the lab data.

The point is that a simulated episode and a recorded one end up in the same
directory shape, with the same column names and the same 37-number layout, so a
training run can mix them without caring which is which:

    <dataset>/
      meta/info.json          robot_type, fps, feature names, episode count
      meta/tasks.jsonl        the instruction text
      meta/episodes.jsonl     episode lengths
      data/chunk-000/file-000.parquet
      videos/<key>/chunk-000/file-000.mp4     optional, from the keyframes

Nothing here is specific to the simulator - it reads what the workbench wrote.

Run it with an interpreter that has pyarrow and pandas, for example:

    ~/miniconda3/bin/python tools/to_lerobot.py \
        --episode runs/handover-2026-09-30T12-00-00 \
        --out     runs/lerobot_sim \
        --task    "Pass the black bottle from the right hand to the left, then place it into the white box."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FEATURES = [
    ("qpos_0_0", "left joint 1"), ("qpos_0_1", "left joint 2"), ("qpos_0_2", "left joint 3"),
    ("qpos_0_3", "left joint 4"), ("qpos_0_4", "left joint 5"), ("qpos_0_5", "left joint 6"),
    ("qpos_0_6", "left joint 7"), ("gripper_0", "left gripper"),
    ("qpos_1_0", "right joint 1"), ("qpos_1_1", "right joint 2"), ("qpos_1_2", "right joint 3"),
    ("qpos_1_3", "right joint 4"), ("qpos_1_4", "right joint 5"), ("qpos_1_5", "right joint 6"),
    ("qpos_1_6", "right joint 7"), ("gripper_1", "right gripper"),
    ("epos_0_qw", ""), ("epos_0_qx", ""), ("epos_0_qy", ""), ("epos_0_qz", ""),
    ("epos_0_x", ""), ("epos_0_y", ""), ("epos_0_z", ""),
    ("epos_1_qw", ""), ("epos_1_qx", ""), ("epos_1_qy", ""), ("epos_1_qz", ""),
    ("epos_1_x", ""), ("epos_1_y", ""), ("epos_1_z", ""),
    ("body_pose_qw", ""), ("body_pose_qx", ""), ("body_pose_qy", ""), ("body_pose_qz", ""),
    ("body_pose_x", ""), ("body_pose_y", ""), ("body_pose_z", ""),
]

# No video features are declared. The workbench writes keyframe PNGs at stage
# boundaries, not a synchronised per-frame stream, and a dataset that announces
# video it does not contain is worse than one that admits it has none: a loader
# will happily accept it and then fail, or silently train on nothing.
# `provenance.keyframes` points at the PNGs so a caller can encode them itself.



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, type=Path, help="a run directory written by /api/episode")
    ap.add_argument("--out", required=True, type=Path, help="dataset directory to write")
    ap.add_argument("--task", default="", help="instruction text; defaults to the episode's own")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    record = json.loads((args.episode / "episode.json").read_text())
    traj = record.get("trajectory") or []
    if not traj:
        raise SystemExit(f"{args.episode}/episode.json has no 30 Hz trajectory - export a run first.")
    fps = float(args.fps or record.get("trajectory_hz") or 30.0)
    task = args.task or record.get("task") or ""
    names = [n for n, _ in FEATURES]

    states = np.array([row["state"] for row in traj], dtype=np.float32)
    actions = np.array([row["action"] for row in traj], dtype=np.float32)
    if states.shape[1] != len(FEATURES) or actions.shape[1] != len(FEATURES):
        raise SystemExit(f"expected {len(FEATURES)} columns, got state {states.shape} action {actions.shape}")

    frames = len(traj)
    out = args.out
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # Real sampling times, not frame index over fps: the harness samples on a
    # control tick that can be a millisecond or two off, and rebuilding the
    # clock from the frame number throws that away.
    times = np.array([row["t"] for row in traj], dtype=np.float32)
    table = pa.table({
        "action": pa.array(list(actions)),
        "observation.state": pa.array(list(states)),
        "timestamp": pa.array(times),
        "frame_index": pa.array(np.arange(frames, dtype=np.int64)),
        "episode_index": pa.array(np.zeros(frames, dtype=np.int64)),
        "index": pa.array(np.arange(frames, dtype=np.int64)),
        "task_index": pa.array(np.zeros(frames, dtype=np.int64)),
    })
    pq.write_table(table, out / "data" / "chunk-000" / "file-000.parquet")

    episodes = pa.table({
        "episode_index": pa.array([0], dtype=np.int64),
        "tasks": pa.array([[task]], type=pa.list_(pa.string())),
        "length": pa.array([frames], dtype=np.int64),
        "dataset_from_index": pa.array([0], dtype=np.int64),
        "dataset_to_index": pa.array([frames], dtype=np.int64),
    })
    pq.write_table(episodes, out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    def feature(shape, names_=None, extra=None):
        spec = {"dtype": "float32", "shape": list(shape), "names": names_}
        if extra:
            spec.update(extra)
        return spec

    info = {
        "codebase_version": "v3.0",
        "robot_type": "openarm_sim",
        "total_episodes": 1,
        "total_frames": frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": {
            "action": feature((len(FEATURES),), names, {"fps": fps}),
            "observation.state": feature((len(FEATURES),), names, {"fps": fps}),
            "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": fps},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
            "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        },
        "provenance": {
            "source": "OpenArmSim simulation",
            "episode": record.get("id"),
            "planner": record.get("planner"),
            "calibration": record.get("calibration"),
            "geometry": record.get("geometry"),
            "state_frame": "robot base frame (the scene's world frame minus Robot.BASE_TO_WORLD)",
            "gripper_channel": "measured finger joint, mapped to the recordings' convention",
            "keyframes": [f["file"] for f in record.get("keyframes", [])],
            "not_included": [
                "per-frame camera video - the workbench records keyframe PNGs, not a stream",
                "force or torque - the lab recorder never logged it either",
            ],
            "validation": "structure checked against the v3 layout; the official loader round-trip has not been run",
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    (out / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n")

    stages: dict[int, int] = {}
    for row in traj:
        if row.get("stage") is not None:
            stages[row["stage"]] = stages.get(row["stage"], 0) + 1
    print(f"wrote {out}")
    print(f"  {frames} frames at {fps} Hz, {len(FEATURES)} columns, {times[-1]:.1f} s of recorded time")
    print(f"  per-stage frames: {dict(sorted(stages.items()))}")
    print("  no video features declared - see provenance.keyframes for the PNGs")


if __name__ == "__main__":
    main()
