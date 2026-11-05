"""Offline, bounded ACT training using official LeRobot policy/processors.

The smoke recipe tests the learning pipeline, not task performance. Checkpoints
from diagnostic data are explicitly ineligible for task execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def checkpoint_hashes(folder):
    values = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            values[str(path.relative_to(folder))] = digest.hexdigest()
    return values


def train(dataset_root, *, recipe="act_smoke", steps=5, seed=42, output=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.is_relative_to(ROOT / "runs"):
        raise ValueError("training data must be a reviewed local export")
    from .dataset import verify_export
    provenance = verify_export(dataset_root)
    validation = json.loads((dataset_root / "validation.json").read_text())
    if validation["result"] != "PASS":
        raise ValueError("dataset has not passed the official loader roundtrip")
    smoke = recipe == "act_smoke"
    if recipe not in {"act_smoke", "act_baseline"}:
        raise ValueError("unsupported training recipe")
    if not smoke and not provenance["eligible_as_imitation_demonstration"]:
        raise ValueError("diagnostic/failure data cannot start imitation training")
    if not 1 <= steps <= (10 if smoke else 10000):
        raise ValueError("training step budget exceeds recipe limits")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    chunk = 4 if smoke else 50
    height, width = (96, 128) if smoke else (240, 320)
    device = "cpu" if smoke else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("baseline recipe needs a working CUDA device")
    config = ACTConfig(device=device, push_to_hub=False, chunk_size=chunk, n_action_steps=chunk,
                       pretrained_backbone_weights=None,
                       input_features={"observation.state": PolicyFeature(FeatureType.STATE, (16,)),
                           **{f"observation.images.{c}": PolicyFeature(FeatureType.VISUAL, (3, height, width)) for c in ("head", "left", "right")}},
                       output_features={"action": PolicyFeature(FeatureType.ACTION, (16,))},
                       **({"dim_model": 64, "n_heads": 4, "dim_feedforward": 256, "n_encoder_layers": 1,
                           "n_decoder_layers": 1, "latent_dim": 8, "n_vae_encoder_layers": 1, "kl_weight": 1.0} if smoke else {}))
    info = json.loads((dataset_root / "meta/info.json").read_text())
    dataset = LeRobotDataset(repo_id="openarm-sim/" + provenance["source_run"], root=dataset_root,
                            delta_timestamps={"action": [i/info["fps"] for i in range(chunk)]}, video_backend="pyav", token=False)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, generator=generator)
    policy = ACTPolicy(config).to(device)
    pre, post = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr, weight_decay=config.optimizer_weight_decay)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + recipe.replace("_", "-") + "-" + uuid.uuid4().hex[:6]
    run = Path(output).resolve() if output else ROOT / "runs" / run_id
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("training output must stay in runs")
    run.mkdir(parents=True, exist_ok=False)
    metadata = {"schema": "openarm-training/1", "recipe": recipe, "dataset": str(dataset_root),
                "source_run": provenance["source_run"], "seed": seed, "steps": steps, "device": device,
                "image_preprocessing": {"resize_hw": [height, width], "interpolation": "bilinear antialias", "preserve_aspect_ratio": True},
                "policy": "official LeRobot ACT", "scope": "pipeline_validation_only" if smoke else "simulation_imitation",
                "task_performance_measured": False, "transfer_ready": False, "eligible_for_task_execution": False,
                "eligible_for_simulation_evaluation": not smoke, "promotion": "requires independent closed-loop evaluation",
                "learning_runtime": json.loads((ROOT / "sim/learning.json").read_text()),
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "lerobot", "numpy")},
                "dataset_provenance_sha256": hashlib.sha256((dataset_root / "provenance.json").read_bytes()).hexdigest(),
                "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(run / "manifest.json", metadata)
    config.save_pretrained(run / "config")
    started, last_batch, last_raw = time.monotonic(), None, None
    status, error = "error", None
    metrics = []
    try:
        batches = iter(loader)
        watched = next(p for name, p in policy.named_parameters() if name.endswith("action_head.weight"))
        initial = watched.detach().cpu().clone()
        with (run / "metrics.jsonl").open("w", buffering=1) as log:
            for step in range(steps):
                try:
                    raw = next(batches)
                except StopIteration:
                    batches = iter(loader)
                    raw = next(batches)
                batch = {k: raw[k] for k in [*config.input_features, "action", "action_is_pad", "task"]}
                for key in config.image_features:
                    batch[key] = F.interpolate(batch[key], size=(height, width), mode="bilinear", align_corners=False, antialias=True)
                last_raw = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                batch = pre(batch)
                last_batch = batch
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                loss, details = policy(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                row = {"step": step+1, "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm),
                       "elapsed_seconds": time.monotonic()-started, **details}
                metrics.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
        delta = float(torch.linalg.norm(watched.detach().cpu()-initial))
        if not delta > 0:
            raise RuntimeError("optimizer did not change the action head")
        checkpoint = run / "checkpoint"
        policy.save_pretrained(checkpoint)
        pre.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
        post.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
        save(checkpoint / "capabilities.json", {"scope": metadata["scope"], "eligible_for_task_execution": False,
             "eligible_for_simulation_evaluation": not smoke,
             "action_semantics": "openarm_absolute_joint16", "task_performance_measured": False,
             "image_preprocessing": metadata["image_preprocessing"], "transfer_ready": False})
        policy.eval()
        with torch.inference_mode():
            original = policy.predict_action_chunk(last_batch)
        restored = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
        restored_pre, restored_post = make_pre_post_processors(restored.config, pretrained_path=str(checkpoint))
        restored_batch = restored_pre(last_raw)
        for key in [*config.input_features, "action"]:
            torch.testing.assert_close(last_batch[key], restored_batch[key], rtol=1e-5, atol=1e-6)
        with torch.inference_mode():
            reloaded = restored.predict_action_chunk(restored_batch)
            torch.testing.assert_close(original, reloaded, rtol=1e-5, atol=1e-6)
            native_action = restored_post(reloaded)
        if native_action.shape[-1] != 16 or not torch.isfinite(native_action).all():
            raise RuntimeError("reloaded checkpoint emitted invalid actions")
        status = "validation_passed" if smoke else "training_completed"
        save(run / "validation.json", {"checkpoint_roundtrip": "PASS", "optimizer_updated_action_head": True,
             "checkpoint_sha256": checkpoint_hashes(checkpoint),
             "action_head_delta_l2": delta, "predicted_action_shape": list(native_action.shape),
             "preprocessor_roundtrip": "PASS", "scope": metadata["scope"], "task_performance_measured": False, "transfer_ready": False})
    except KeyboardInterrupt:
        status, error = "cancelled", "training interrupted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        save(run / "summary.json", {"status": status, "error": error, "optimizer_steps": len(metrics),
             "wall_seconds": time.monotonic()-started, "scope": metadata["scope"],
             "task_performance_measured": False, "transfer_ready": False})
    print(json.dumps({"run": str(run), "status": status, "error": error}, ensure_ascii=False), flush=True)
    return 0 if error is None else 1


def main():
    parser = argparse.ArgumentParser(description="Bounded offline training with official LeRobot ACT")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--recipe", choices=("act_smoke", "act_baseline"), default="act_smoke")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="openarm-learning-") as cache:
        os.environ.update({"HF_HOME": cache, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                           "TORCH_HOME": cache, "TOKENIZERS_PARALLELISM": "false"})
        raise SystemExit(train(args.dataset, recipe=args.recipe, steps=args.steps, seed=args.seed, output=args.output))


if __name__ == "__main__":
    main()
