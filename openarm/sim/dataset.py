"""Export native observations/actions via the official LeRobot v3 writer.

Imports no simulator or robot driver. Run in the configured learning runtime.
Raw physics timestamps are additional features: LeRobot's canonical timestamp
is frame_index / fps and must not silently replace the actual simulation clock.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("head", "left", "right")
NAMES = [name for arm in (0, 1) for name in [*(f"qpos_{arm}_{i}" for i in range(7)), f"gripper_{arm}"]]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def image_path(run, relative):
    path = (run / relative).resolve()
    if not path.is_relative_to(run) or not path.is_file():
        raise ValueError("missing image or image path outside the source run")
    return path


def validate_training_clock(manifest, rows, purpose):
    """A frame/fps index is not a resampling operation."""
    import numpy as np
    if purpose not in {"imitation", "diagnostic"}:
        raise ValueError("unknown dataset purpose")
    hz = float(manifest["profile"]["control_hz"])
    before = np.asarray([r["observation"]["sim_time"] for r in rows], float)
    after = np.asarray([r["next_observation"]["sim_time"] for r in rows], float)
    # MuJoCo ticks are quantized to 1 ms; use the global phase, not mean FPS.
    uniform = bool(len(rows) and np.isfinite(before).all() and np.isfinite(after).all()
                   and np.max(abs(before-before[0]-np.arange(len(rows))/hz)) <= .001001
                   and np.max(abs(after-before[0]-(np.arange(len(rows))+1)/hz)) <= .001001)
    recorded = manifest.get("details", {}).get("replay_clock", {}).get("mode") == "recorded"
    if purpose == "imitation" and (recorded or not uniform):
        raise ValueError("nonuniform recorded replay requires explicit temporal resampling before imitation training; diagnostic export preserves physical time only")
    return {"uniform_control_clock": uniform, "recorded_replay": recorded,
            "resampled": False, "eligible_clock_for_imitation": uniform and not recorded}


def read_source(run, purpose):
    import numpy as np
    from PIL import Image
    run = run.resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("source must be an OpenArmSim run")
    manifest, summary = read_json(run / "manifest.json"), read_json(run / "summary.json")
    if manifest.get("schema") != "openarm-sim/1":
        raise ValueError("only native transition logs are accepted")
    if purpose=='imitation' and manifest.get('details',{}).get('tamp',{}).get('native_finger_range'):
        raise ValueError('TAMP uses the full simulated finger range; explicitly match the training/deployment action adapter before imitation export. Diagnostic export remains available.')
    if purpose == "imitation" and (summary.get("status") != "success" or not summary["evaluation"].get("success")):
        raise ValueError("failed or incomplete runs cannot be exported as imitation demonstrations; use diagnostic purpose for inspection")
    rows = [json.loads(line) for line in (run / "transitions.jsonl").read_text().splitlines()]
    if not rows or len(rows) != summary["transitions"]:
        raise ValueError("transition count is empty or inconsistent with the closed run")
    validate_training_clock(manifest, rows, purpose)
    sizes = {}
    previous = None
    for i, row in enumerate(rows):
        before, after = row["observation"], row["next_observation"]
        if row["index"] != i or after["tick"] != before["tick"] + 1 or not before["sim_time"] < after["sim_time"]:
            raise ValueError(f"transition ordering invalid at frame {i}")
        for vector in (before["state"], row["action"], after["state"]):
            a = np.asarray(vector)
            if a.shape != (16,) or not np.isfinite(a).all():
                raise ValueError(f"invalid state/action at frame {i}")
        if previous is not None:
            if before["tick"] != previous["tick"] or before["sim_time"] != previous["sim_time"]:
                raise ValueError("gap in observations")
            np.testing.assert_allclose(before["state"], previous["state"], rtol=0, atol=1e-7)
        for observation in (before, after):
            if set(observation.get("images", {})) != set(CAMERAS):
                raise ValueError("three synchronized camera images are required on every observation; keyframes are insufficient")
            for camera in CAMERAS:
                with Image.open(image_path(run, observation["images"][camera])) as im:
                    if im.mode != "RGB":
                        raise ValueError("recorded camera images must be RGB")
                    if camera in sizes and sizes[camera] != im.size:
                        raise ValueError("camera resolution changed within an episode")
                    sizes[camera] = im.size
                    im.verify()
        previous = after
    return manifest, summary, rows, sizes


def export(run, *, purpose="imitation", output=None):
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    run = Path(run).resolve()
    manifest, summary, rows, sizes = read_source(run, purpose)
    out = Path(output).resolve() if output else run / "lerobot"
    if not out.is_relative_to(run) or out == run:
        raise ValueError("export must stay inside its source run directory")
    if out.exists():
        provenance = read_json(out / "provenance.json")
        if provenance["purpose"] != purpose or provenance["source_manifest_sha256"] != hashlib.sha256((run/"manifest.json").read_bytes()).hexdigest() or provenance["source_transitions_sha256"] != hashlib.sha256((run/"transitions.jsonl").read_bytes()).hexdigest():
            raise FileExistsError("existing dataset belongs to a different source or purpose")
        validate_dataset(out, run, manifest, rows, purpose)
        return out
    if CODEBASE_VERSION != "v3.0":
        raise ValueError("this exporter was verified against official LeRobot v3.0")
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (16,), "names": NAMES},
        "next.observation.state": {"dtype": "float32", "shape": (16,), "names": NAMES},
        "observation.sim_time": {"dtype": "float64", "shape": (1,), "names": ["seconds"]},
        "next.sim_time": {"dtype": "float64", "shape": (1,), "names": ["seconds"]},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["episode_boundary"]},
        "next.terminated": {"dtype": "bool", "shape": (1,), "names": ["environment_terminal"]},
        "next.truncated": {"dtype": "bool", "shape": (1,), "names": ["environment_time_limit"]},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": ["simulator_success"]},
    }
    for camera, (width, height) in sizes.items():
        features[f"observation.images.{camera}"] = {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    repo_id = "openarm-sim/" + run.name
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=int(manifest["profile"]["control_hz"]),
                                    features=features, root=out, robot_type="openarm_sim",
                                    use_videos=False, video_backend="pyav", image_writer_threads=2)
    try:
        for i, row in enumerate(rows):
            frame = {
                "observation.state": np.asarray(row["observation"]["state"], np.float32),
                "action": np.asarray(row["action"], np.float32),
                "next.observation.state": np.asarray(row["next_observation"]["state"], np.float32),
                "observation.sim_time": np.array([row["observation"]["sim_time"]], np.float64),
                "next.sim_time": np.array([row["next_observation"]["sim_time"]], np.float64),
                "next.done": np.array([i == len(rows)-1], bool),
                "next.terminated": np.array([row["terminated"]], bool),
                "next.truncated": np.array([row["truncated"]], bool),
                "next.reward": np.array([float(row["evaluation"]["success"])], np.float32),
                "task": manifest["task"],
            }
            for camera in CAMERAS:
                with Image.open(image_path(run, row["observation"]["images"][camera])) as im:
                    frame[f"observation.images.{camera}"] = np.array(im)
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    provenance = {"source_run": run.name, "source_manifest_sha256": hashlib.sha256((run/"manifest.json").read_bytes()).hexdigest(),
                  "source_transitions_sha256": hashlib.sha256((run/"transitions.jsonl").read_bytes()).hexdigest(),
                  "source_status": summary["status"], "purpose": purpose,
                  "eligible_as_imitation_demonstration": purpose == "imitation",
                  "transfer_ready": False, "format": CODEBASE_VERSION,
                  "writer_module": inspect.getfile(LeRobotDataset), "lerobot_version": importlib.metadata.version("lerobot"),
                  "clock": "timestamp = frame/fps for LeRobot indexing; observation.sim_time and next.sim_time retain actual physics times",
                  "clock_validation": validate_training_clock(manifest, rows, purpose),
                  "images": "lossless images embedded by official writer; per-pixel roundtrip required",
                  "terminal_observation": "final next-state is included; final next-images remain in original native run",
                  "boundary": "next.done marks dataset episode boundary, not task success; separate terminated/truncated flags are preserved",
                  "controller": "action is the applied command; policy requested_action, when present, remains in native diagnostics",
                  "calibration_gaps": manifest["profile"]["uncertainties"]}
    write_json(out / "provenance.json", provenance)
    validate_dataset(out, run, manifest, rows, purpose)
    return out


def verify_export(path):
    """Verify live source and pixels, not just a possibly stale PASS file."""
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "runs"):
        raise ValueError("dataset must be a local run export")
    provenance = read_json(path / "provenance.json")
    run = path.parent
    if provenance["source_run"] != run.name or path.name != "lerobot":
        raise ValueError("dataset source binding is invalid")
    for name in ("manifest", "transitions"):
        source_file = run / (name + (".jsonl" if name == "transitions" else ".json"))
        if hashlib.sha256(source_file.read_bytes()).hexdigest() != provenance[f"source_{name}_sha256"]:
            raise ValueError("dataset source changed after export")
    manifest, _, rows, _ = read_source(run, provenance["purpose"])
    validate_dataset(path, run, manifest, rows, provenance["purpose"])
    return provenance


def validate_dataset(out, run, manifest, rows, purpose):
    import numpy as np
    from PIL import Image
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    repo_id = "openarm-sim/" + run.name
    # Reopen with the official reader; compare all values and every image pixel.
    reader = LeRobotDataset(repo_id=repo_id, root=out, return_uint8=True, video_backend="pyav", token=False)
    if len(reader) != len(rows) or reader.num_episodes != 1:
        raise RuntimeError("official reader reports unexpected length")
    image_checks = 0
    for i, row in enumerate(rows):
        item = reader[i]
        for key, expected in (("observation.state", row["observation"]["state"]), ("action", row["action"]),
                              ("next.observation.state", row["next_observation"]["state"])):
            np.testing.assert_array_equal(item[key].numpy(), np.asarray(expected, np.float32))
        np.testing.assert_allclose(item["observation.sim_time"].numpy(), [row["observation"]["sim_time"]], rtol=2e-7, atol=1e-7)
        if item["task"] != manifest["task"]:
            raise RuntimeError("task language did not survive roundtrip")
        for camera in CAMERAS:
            with Image.open(image_path(run, row["observation"]["images"][camera])) as im:
                expected = np.array(im)
            restored = item[f"observation.images.{camera}"].numpy().transpose(1, 2, 0)
            if np.issubdtype(restored.dtype, np.floating):
                np.testing.assert_allclose(restored, expected.astype(np.float32)/255, rtol=0, atol=1e-7)
                restored = np.rint(restored * 255).astype(np.uint8)
            np.testing.assert_array_equal(restored, expected)
            image_checks += 1
    stored_times = []
    for file in sorted(out.glob("data/*/*.parquet")):
        stored_times.extend(pq.read_table(file, columns=["observation.sim_time"])["observation.sim_time"].to_pylist())
    np.testing.assert_array_equal(stored_times, [r["observation"]["sim_time"] for r in rows])
    result = {"result": "PASS", "reader": "official LeRobotDataset", "format": CODEBASE_VERSION,
              "frames_checked": len(rows), "images_compared_pixelwise": image_checks,
              "purpose": purpose, "simulator_task_passed": purpose == "imitation", "demonstration_quality_verified": False,
              "image_reader_representation": "float32 RGB [0,1] or uint8; pixel values reconstructed and compared exactly",
              "physics_timestamps_float64_preserved_in_parquet": True,
              "transfer_ready": False}
    write_json(out / "validation.json", result)
    print(json.dumps({"dataset": str(out), **result}, ensure_ascii=False, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description="Export and verify native observations with official LeRobot v3")
    parser.add_argument("run", type=Path)
    parser.add_argument("--purpose", choices=("imitation", "diagnostic"), default="imitation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="openarm-dataset-") as cache:
        os.environ.update({"HF_HOME": cache, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        export(args.run, purpose=args.purpose, output=args.output)


if __name__ == "__main__":
    main()
