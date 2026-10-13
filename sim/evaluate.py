"""Held-out action prediction through the saved official ACT processing chain.

This is not a rollout success-rate test and cannot promote a model for execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .dataset import verify_export
from .train import ROOT, save, checkpoint_hashes


def validate_split(training, evaluation, train_provenance, eval_provenance):
    if train_provenance["source_run"] == eval_provenance["source_run"] or train_provenance["source_transitions_sha256"] == eval_provenance["source_transitions_sha256"]:
        raise ValueError("evaluation duplicates the training episode")
    if training.get("kind") == evaluation.get("kind") == "replay":
        if training.get("details", {}).get("episode") == evaluation.get("details", {}).get("episode") and training.get("hashes", {}).get("reference/real_trajectory.json") == evaluation.get("hashes", {}).get("reference/real_trajectory.json"):
            raise ValueError("two runs replay the same source episode; this is not an independent episode")


def evaluate(trained, dataset_path, output, frames=300):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    trained, dataset_path, output = [Path(p).resolve() for p in (trained, dataset_path, output)]
    if any(not p.is_relative_to(ROOT / "runs") for p in (trained, dataset_path, output)):
        raise ValueError("evaluation must stay inside local runs")
    manifest = json.loads((trained / "manifest.json").read_text())
    provenance = verify_export(dataset_path)
    training_provenance = json.loads((Path(manifest["dataset"]) / "provenance.json").read_text())
    train_source = json.loads((Path(manifest["dataset"]).parent / "manifest.json").read_text())
    eval_source = json.loads((dataset_path.parent / "manifest.json").read_text())
    validate_split(train_source, eval_source, training_provenance, provenance)
    if not 1 <= frames <= 3000:
        raise ValueError("invalid frame budget")
    torch.set_num_threads(4)
    checkpoint = trained / "checkpoint"
    validation = json.loads((trained / "validation.json").read_text())
    if validation.get("checkpoint_roundtrip") != "PASS" or validation.get("checkpoint_sha256") != checkpoint_hashes(checkpoint):
        raise ValueError("checkpoint/processor files changed or have no verified content hashes")
    policy = ACTPolicy.from_pretrained(checkpoint).cpu().eval()
    policy.config.device = "cpu"
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}}, postprocessor_overrides={"device_processor": {"device": "cpu"}})
    dataset = LeRobotDataset(repo_id="openarm-sim/" + provenance["source_run"], root=dataset_path, video_backend="pyav", token=False)
    indices = np.linspace(0, len(dataset)-1, min(frames, len(dataset)), dtype=int)
    predictions, targets = [], []
    with torch.inference_mode():
        for i in indices:
            raw = dataset[int(i)]
            batch = {key: raw[key].unsqueeze(0) for key in policy.config.input_features}
            batch["task"] = [raw["task"]]
            for key in policy.config.image_features:
                batch[key] = F.interpolate(batch[key], size=manifest["image_preprocessing"]["resize_hw"], mode="bilinear", align_corners=False, antialias=True)
            policy.reset()
            native = post(policy.predict_action_chunk(pre(batch)))
            predictions.append(native[0, 0].cpu().numpy())
            targets.append(raw["action"].cpu().numpy())
    prediction, target = np.stack(predictions), np.stack(targets)
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite predicted actions")
    error = prediction - target
    joint_indices = [*range(7), *range(8, 15)]
    result = {"status": "evaluated", "scope": "offline_action_prediction_only", "training_run": str(trained.relative_to(ROOT / "runs")),
              "evaluation_source": provenance["source_run"], "frames": len(indices), "frame_indices": indices.tolist(),
              "separate_episode": True, "independent_real_world_holdout": False,
              "diagnostic_model": manifest["scope"] == "pipeline_validation_only", "task_success_rate": None,
              "task_performance_measured": False, "eligible_for_task_execution": False, "transfer_ready": False,
              "joint_rmse_rad": float(np.sqrt(np.mean(error[:, joint_indices]**2))),
              "joint_p95_abs_rad": float(np.percentile(abs(error[:, joint_indices]), 95)),
              "gripper_rmse_command_units": np.sqrt(np.mean(error[:, [7, 15]]**2, axis=0)).tolist(),
              "per_channel_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
              "note": "First predicted action vs recorded applied command; no physics rollout or success-rate claim.",
              "dataset_provenance_sha256": hashlib.sha256((dataset_path / "provenance.json").read_bytes()).hexdigest(),
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result["checkpoint_sha256"] = validation["checkpoint_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    save(output / "evaluation.json", result)
    save(output / "predictions.json", {"frame_indices": indices.tolist(), "prediction": prediction.tolist(), "target": target.tolist()})
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trained", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="openarm-evaluation-") as cache:
        os.environ.update({"HF_HOME": cache, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TORCH_HOME": cache})
        evaluate(args.trained, args.dataset, args.output, args.frames)


if __name__ == "__main__":
    main()
