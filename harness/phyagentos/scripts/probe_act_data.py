"""Read actual sim/real data through LeRobot and check ACT interfaces; no training."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[key] = "1"

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT = Path(str(Path(__file__).resolve().parents[1]))
TRAINING = PROJECT.parent / "piper_training_project"
ASSET = PROJECT.parent / "piper_robocasa"
GRIP_CONTROLLER = ASSET / "vendor/robosuite-5ce6643f/robosuite/controllers/parts/gripper/simple_grip.py"
STATE_NAMES = [f"{side}_{name}" for side in ("left", "right")
               for name in [*(f"joint{i}" for i in range(1, 7)), "gripper"]]
ACTION_NAMES = [f"{side}.tcp.{axis}" for side in ("left", "right")
                for axis in ("x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad", "gripper_width_target_m")]
SIM_SCHEMA = "piper_dual_tcp_absolute_world_rotvec_width_left_right_v1"
REAL_SCHEMA = "piper_dual_joint_absolute_width_left_right_v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   default=lambda x: x.value) + "\n")


def state14(state):
    # Same software opening as native observe(): sum(abs(slide qpos)).
    return torch.cat((state[..., 8:14], state[..., 14:16].abs().sum(-1, keepdim=True),
                      state[..., :6], state[..., 6:8].abs().sum(-1, keepdim=True)), -1)


def action14(action):
    # PiperGripper [-a,+a] and GRIP affine servo ranges [0,.05],[-.05,0].
    # Arm goals remain TCP goals; never substitute achieved qpos for a command.
    return torch.cat((action[..., 6:12], .05 * (1 - action[..., 13:14]),
                      action[..., :6], .05 * (1 - action[..., 12:13])), -1)


class SimACTView(Dataset):
    def __init__(self, loaded, episode):
        self.loaded, self.episode = loaded, episode

    def __len__(self):
        return len(self.loaded)

    def __getitem__(self, index):
        row = dict(self.loaded[index])
        row["observation.state"] = state14(row["observation.state"])
        row["action"] = action14(row["action"])
        row["observation.assist_active"] = row["observation.assist_active"].flip(-1)
        row["meta.assist_known"] = torch.ones(2, dtype=torch.bool)
        row["meta.source"] = "simulation_contact_assisted"
        row["meta.image_source"] = "state_replay_from_recorded_simulation"
        row["meta.source_task_id"] = self.episode["task_id"]
        row["meta.source_episode_id"] = self.episode["episode_id"]
        row["meta.action_schema"] = SIM_SCHEMA
        return row


def normalize_batch(batch, state, action, cameras):
    batch = dict(batch)
    stats = {}
    for key, array in (("observation.state", state), ("action", action)):
        mean = torch.from_numpy(array.mean(0).astype(np.float32))
        std = torch.from_numpy(array.std(0).astype(np.float32)).clamp_min(1e-6)
        batch[key] = (batch[key] - mean) / std
        stats[key] = {"mean": mean.tolist(), "std": std.tolist(), "fit_frames": len(array)}
    for key in cameras:
        image = torch.nn.functional.interpolate(batch[key], size=(240, 320),
                                                 mode="bilinear", align_corners=False)
        mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
        std = image.new_tensor([.229, .224, .225])[None, :, None, None]
        batch[key] = (image - mean) / std
    return batch, stats


def forward_probe(batch, cameras, state, action, out, name):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    normalized, stats = normalize_batch(batch, state, action, cameras)
    inputs = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(int(state.shape[-1]),))}
    inputs.update({k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320)) for k in cameras})
    cfg = ACTConfig(input_features=inputs,
                    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
                    device="cpu", chunk_size=32, n_action_steps=16,
                    pretrained_backbone_weights=None, dim_model=64, n_heads=4,
                    dim_feedforward=128, n_encoder_layers=1, n_decoder_layers=1,
                    n_vae_encoder_layers=1, latent_dim=8, push_to_hub=False)
    torch.manual_seed(42)
    policy = ACTPolicy(cfg).eval()
    with torch.inference_mode():
        prediction = policy.predict_action_chunk(normalized)
    assert prediction.shape == (2, 32, 14) and torch.isfinite(prediction).all()
    assert all(not p.requires_grad or p.grad is None for p in policy.parameters())
    save(out / f"{name}-act-config.json", dataclasses.asdict(cfg))
    save(out / f"{name}-normalization.json", stats)
    np.savez_compressed(out / f"{name}-interface-batch.npz",
                        state=batch["observation.state"].numpy(), action=batch["action"].numpy(),
                        action_is_pad=batch["action_is_pad"].numpy(),
                        random_model_prediction=prediction.numpy())
    return {"actual_policy": "lerobot.policies.act.modeling_act.ACTPolicy",
            "batch_shapes": {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)},
            "prediction_shape": list(prediction.shape), "prediction_finite": True,
            "device": "cpu", "model": "reduced random initialization; interface check only",
            "backbone_weights": None, "optimizer_updates": 0, "training_started": False,
            "actions_dispatched": 0, "camera_keys": cameras,
            "metadata_is_model_input": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = PROJECT / "runs/lifecycle-repair-20261010T150721Z"
    ap.add_argument("--prepared", type=Path, default=root / "formal-root1-prepared")
    ap.add_argument("--sim-export", type=Path, default=root / "formal-root1-lerobot-0.4.4")
    ap.add_argument("--real-registry", type=Path, default=TRAINING / "datasets/pick_place_gripper_v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-mix", action="store_true", help="Reject incompatible direct mixing before any model execution")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "alignment-audit.json").exists():
        raise FileExistsError("Use a fresh output directory; retained evidence is never overwritten")
    manifest = read(args.prepared / "manifest.json")
    episodes = [x for x in manifest["episodes"] if x["selected_for_training"]]
    if len(episodes) != 1:
        raise ValueError("This thin probe covers the inspected single approved episode")
    episode = episodes[0]
    folder = args.prepared / episode["label"]
    sim_info = read(args.sim_export / "meta/info.json")
    sim_provenance = read(args.sim_export / "export-provenance.json")
    real_cfg = read(args.real_registry / "dataset.json")
    real_root = TRAINING / real_cfg["root"]
    real_info = read(real_root / "meta/info.json")
    split = read(args.real_registry / "split.json")
    expected_state = [f"robot0_{side}_joint{i}" for side in ("right", "left") for i in range(1, 7)]
    for side, start in (("right", 0), ("left", 8)):
        assert episode["state_names"][start:start+6] == [f"robot0_{side}_joint{i}" for i in range(1, 7)]
    assert manifest["action_schema"] == "piper_dual_tcp_absolute_world_rotvec_v1"
    assert sim_info["fps"] == manifest["fps"] == 20 and real_info["fps"] == real_cfg["fps"] == 30
    assert real_info["features"]["action"]["names"] == STATE_NAMES
    assert real_info["features"]["observation.state"]["names"] == STATE_NAMES
    assert sim_provenance["episodes"][0]["source_task_id"] == episode["task_id"]
    assert sim_provenance["source_manifest_sha256"] == sha(args.prepared / "manifest.json")
    numeric = np.load(folder / "transitions.npz", allow_pickle=False)
    assert sha(folder / "transitions.npz") == episode["transitions_sha256"]
    assert sha(folder / "image-index.json") == episode["image_index_sha256"]
    sources = list(episode["source_sha256"]) + [
        args.prepared / "manifest.json", folder / "transitions.npz", folder / "image-index.json",
        args.sim_export / "meta/info.json", args.sim_export / "export-provenance.json",
        args.real_registry / "dataset.json", args.real_registry / "dataset.lock.json",
        args.real_registry / "split.json", real_root / "meta/info.json",
        real_root / "meta/piper_sources.json", real_root / "meta/piper_gripper_mapping.json",
        ASSET / "configs/robot_controller.json", ASSET / "src/piper_robocasa/gripper.py", GRIP_CONTROLLER]
    before = {str(p): sha(p) for p in sources}
    for p, expected in episode["source_sha256"].items():
        assert before[p] == expected
    sim_cameras = [k for k in sim_info["features"] if k.startswith("observation.images.")]
    issues = [
        {"field": "action", "sim": SIM_SCHEMA, "real": REAL_SCHEMA,
         "missing": "A real joint-position expert command source; OSC arm motor ctrl is torque, not a joint position"},
        {"field": "state", "sim": "named MuJoCo qpos and sum(abs(finger slides))",
         "real": "recorded follower joints and signed gripper feedback",
         "missing": "verified joint zero/sign and gripper-feedback calibration across the asset and recorded robot"},
        {"field": "frequency", "sim": 20, "real": 30,
         "missing": "explicit shared action horizon and causal resampling decision"},
        {"field": "cameras", "sim": sim_cameras, "real": real_cfg["cameras"],
         "missing": "common camera contract; overview cannot be renamed to cam_middle"},
        {"field": "image_and_assistance", "sim": "state-replay RGB; contact assistance explicitly recorded",
         "real": "captured RGB; assistance status not recorded",
         "missing": "explicit domain/assistance treatment; unknown real assistance is not relabeled as false"}
    ]
    audit = {"sim_task_id": episode["task_id"], "source_episode_id": episode["episode_id"],
             "sim_original": {"state_dim": 16, "action_schema": manifest["action_schema"], "fps": 20},
             "sim_view": {"state_dim": 14, "state_names": STATE_NAMES, "action_dim": 14,
                          "action_names": ACTION_NAMES, "action_schema": SIM_SCHEMA,
                          "arm_action_unchanged_semantics": "absolute TCP world position and rotvec",
                          "gripper_action_transform": ".05*(1-a), exact software servo-opening target in m",
                          "state_gripper_transform": "abs(l_slide)+abs(r_slide); no clipping or hardware equivalence claim"},
             "real": {"root": str(real_root), "episodes": 98, "frames": 56381,
                      "fps": 30, "action_schema": REAL_SCHEMA, "order": real_cfg["action_order"],
                      "action_gripper_range_m": [0, .1], "signed_state_gripper_preserved": True},
             "incompatibilities": issues, "direct_mixing_ready": False,
             "joint_command_labels_available": False, "next_qpos_used_as_action": False,
             "historical_joint_command_recovery": "Unavailable: OSC_POSE produces motor torque, not joint-position commands",
             "source_sha256": before, "script_sha256": sha(__file__)}
    save(args.out / "alignment-audit.json", audit)
    if args.require_mix:
        save(args.out / "mix-refusal.json", {"exit_code": 2, "reasons": issues,
                                            "model_executed": False, "training_started": False})
        print(json.dumps({"direct_mixing_rejected": True, "audit": str(args.out / "alignment-audit.json")}), flush=True)
        return 2
    torch.set_num_threads(2)
    state = state14(torch.from_numpy(numeric["state"])).numpy()
    action = action14(torch.from_numpy(numeric["action"])).numpy()
    next_state = state14(torch.from_numpy(numeric["next_state"])).numpy()
    assert action.shape == state.shape == (3928, 14)
    np.testing.assert_array_equal(action[:, :6], numeric["action"][:, 6:12])
    np.testing.assert_array_equal(action[:, 7:13], numeric["action"][:, :6])
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert action[:, [6, 13]].min() >= 0 and action[:, [6, 13]].max() <= .1
    np.savez_compressed(args.out / "sim-act-view-transitions.npz",
                        state=state, action=action, next_state=next_state,
                        sim_time=numeric["sim_time"], control_step=numeric["control_step"],
                        next_control_step=numeric["next_control_step"],
                        assist=numeric["assist"][:, [1, 0]], next_assist=numeric["next_assist"][:, [1, 0]],
                        phase=numeric["phase"])
    sim = LeRobotDataset("local/phyagentos-piper-assisted-tcp-v1", root=args.sim_export,
                         delta_timestamps={"action": [i/20 for i in range(32)]}, download_videos=False)
    view = SimACTView(sim, episode)
    sim_checks = []
    for n in (0, len(view)//2, len(view)-1):
        row = view[n]
        expected = np.stack([action[min(n+i, len(view)-1)] for i in range(32)])
        np.testing.assert_allclose(row["action"].numpy(), expected, atol=1e-6)
        np.testing.assert_array_equal(row["observation.state"].numpy(), state[n])
        np.testing.assert_array_equal(row["action_is_pad"].numpy(), [n+i >= len(view) for i in range(32)])
        np.testing.assert_array_equal(row["observation.assist_active"].numpy(), numeric["assist"][n, [1, 0]])
        sim_checks.append({"row": n, "source_state_action_assist_match": True,
                           "padded_actions": int(row["action_is_pad"].sum())})
    sim_batch = next(iter(DataLoader(view, batch_size=2, num_workers=0, shuffle=False)))
    print(json.dumps({"phase": "actual_sim_loader", "checks": sim_checks}), flush=True)
    sim_forward = forward_probe(sim_batch, sim_cameras, state, action, args.out, "sim")
    table = pq.read_table(real_root / "data/chunk-000/file-000.parquet",
                          columns=["observation.state", "action", "episode_index", "index"])
    real_states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    real_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    ep_index = table["episode_index"].to_numpy()
    assert len(real_actions) == 56381
    real = LeRobotDataset(real_cfg["repo_id"], root=real_root,
                          delta_timestamps={"action": [i/30 for i in range(32)]},
                          video_backend="pyav", download_videos=False)
    train_episode = split["train"][0]
    positions = np.flatnonzero(ep_index == train_episode)
    start, end = int(positions[0]), int(positions[-1])
    real_checks = []
    for n in (start, (start+end)//2, end):
        row = real[n]
        expected = np.stack([real_actions[min(n+i, end)] for i in range(32)])
        np.testing.assert_allclose(row["action"].numpy(), expected, atol=1e-6)
        np.testing.assert_array_equal(row["observation.state"].numpy(), real_states[n])
        np.testing.assert_array_equal(row["action_is_pad"].numpy(), [n+i > end for i in range(32)])
        for camera in real_cfg["cameras"]:
            assert tuple(row[camera].shape) == (3, 480, 640) and torch.isfinite(row[camera]).all()
        real_checks.append({"row": n, "episode": int(train_episode), "source_state_action_match": True,
                            "padded_actions": int(row["action_is_pad"].sum()),
                            "next_episode_not_used_as_future_action": True})
    rows = [dict(real[start]), dict(real[start+1])]
    for row in rows:
        row["meta.source"] = "recorded_robot_demonstration"
        row["meta.image_source"] = "captured_camera_video"
        row["meta.action_schema"] = REAL_SCHEMA
        row["meta.assist_active"] = torch.full((2,), -1, dtype=torch.int64)
        row["meta.assist_known"] = torch.zeros(2, dtype=torch.bool)
    real_batch = next(iter(DataLoader(rows, batch_size=2, num_workers=0)))
    fit = np.isin(ep_index, split["train"])
    assert not np.isin(ep_index[fit], split["validation"]).any()
    print(json.dumps({"phase": "actual_real_loader", "checks": real_checks}), flush=True)
    real_forward = forward_probe(real_batch, real_cfg["cameras"], real_states[fit], real_actions[fit],
                                 args.out, "real")
    after = {p: sha(p) for p in before}
    assert before == after
    report = {"actual_loader": "lerobot.datasets.lerobot_dataset.LeRobotDataset",
              "versions": {p: importlib.metadata.version(p) for p in ("lerobot", "torch", "torchvision", "numpy")},
              "sim": {"frames": len(view), "episodes": sim.num_episodes, "fps": sim.fps,
                      "checks": sim_checks, "forward": sim_forward,
                      "future_action_offsets_s": [i/20 for i in range(32)],
                      "normalization_scope": "one inspected successful episode; no independent validation split",
                      "data_view_npz_sha256": sha(args.out / "sim-act-view-transitions.npz")},
              "real": {"frames": len(real), "episodes": real.num_episodes, "fps": real.fps,
                       "checks": real_checks, "forward": real_forward,
                       "future_action_offsets_s": [i/30 for i in range(32)],
                       "normalization_fit_frames": int(fit.sum()),
                       "normalization_validation_excluded": True,
                       "assistance_label": "unknown (-1); known=false"},
              "joint_position_actions_fabricated": False, "direct_mixing_ready": False,
              "source_files_unchanged": True, "source_sha256_after": after,
              "new_model_requests": 0, "training_started": False, "optimizer_updates": 0,
              "actions_dispatched": 0, "shared_asset_and_ACT_sources_modified": False,
              "image_processing": "retain named cameras; bilinear to 240x320 and fixed ImageNet normalization; no camera alias",
              "incompatibilities": issues, "script_sha256": sha(__file__)}
    save(args.out / "acceptance.json", report)
    print(json.dumps({"completed": str(args.out), "sim_forward": sim_forward["prediction_shape"],
                      "real_forward": real_forward["prediction_shape"], "direct_mixing_ready": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
