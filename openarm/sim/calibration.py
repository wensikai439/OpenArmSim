"""Auditable calibration evidence and bounded, held-out controller studies.

This fits the current MuJoCo model to recorded commands/encoders, never writes
runtime parameters, and never connects to hardware. Effective gains and sample
offsets are hypotheses, not measurements of the real controller or bus latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from mujoco import minimize
import numpy as np

from .environment import ROOT, OpenArmEnv
from .recording import dump as write_data
from .reference_clock import load as load_clock, summary as clock_summary


def dump(path, value):
    temporary = Path(path).with_suffix(".tmp")
    write_data(temporary, value)
    temporary.replace(path)

PROTOCOL = {
    "id": "right_early_v4", "fit_episodes": [0, 1], "holdout_episodes": [2],
    "frames": 120, "window_seconds": 4., "warmup_frames": 10, "fps": 30,
    "delay_steps": [0, 1, 2, 3], "scale_bounds": [.25, 4.0],
    "max_iterations": 12, "wall_budget_seconds": 600,
    "selection": "lowest fit RMSE; holdout is evaluated only after selection",
    "scope": "right-arm early-motion response, not contact or full-task calibration",
    "timing": "audited nonuniform record() logger timestamps; not sensor exposure or CAN send timestamps",
    "offset_semantics": "right-arm command offset in recorded sample indices; not a constant delay in milliseconds",
    "initial_velocity": "zero; same reset as deployment simulator",
    "window_basis": "first 4 s, before first recorded gripper-close command in these episodes; real contact absence remains unverified",
    "contact_guard": "every 1 ms solved physics step and final-state check; reject robot external or self contact above 0.05 N",
    "validation_scope": "episode 2 was inspected in earlier studies; development validation, not a fresh final test set",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_dependencies(model_manifest):
    """Study compatibility depends on dynamics/data, not UI or harness tests.

    Keep the full source manifest for audit. This narrower set only answers
    whether the recorded controller response still describes this model.
    """
    names={'sim/environment.py','sim/controller.py','sim/contacts.py','sim/gripper.py','sim/reference_clock.py',
           'sim/profile.json','calibrate.json','task/bottle_to_box.xml','task/cell-v1.xml',
           'reference/real_trajectory.json','reference/acquisition.json','source.json'}
    return {name:value for name,value in model_manifest['hashes'].items()
            if name in names or name.startswith('openarm_mujoco/')}


def validate_protocol(protocol, reference):
    fit, holdout = protocol["fit_episodes"], protocol["holdout_episodes"]
    if not fit or not holdout or set(fit) & set(holdout):
        raise ValueError("fit and holdout episodes must be nonempty and disjoint")
    if len(set(fit + holdout)) != len(fit + holdout):
        raise ValueError("duplicate episode in protocol")
    for episode in fit + holdout:
        e = reference[str(episode)]
        if e["frames"] < protocol["frames"] or e["fps"] != protocol["fps"]:
            raise ValueError("reference does not cover this protocol")
        for key in ("right_joints", "right_action_joints", "left_joints", "left_action_joints"):
            a = np.asarray(e[key], float)
            if a.shape != (e["frames"], 7) or not np.isfinite(a).all():
                raise ValueError("malformed reference joint data")


def metrics(prediction, target, warmup):
    error = np.asarray(prediction)[warmup:] - np.asarray(target)[warmup:]
    return {"rmse_rad": float(np.sqrt(np.mean(error**2))),
            "p95_abs_rad": float(np.percentile(abs(error), 95)),
            "per_joint_rmse_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "measured_excursion_rad": np.ptp(target, axis=0).tolist(),
            "scored_frames": len(error)}


def select_candidate(candidates):
    # Deliberately cannot use validation scores for selection.
    return min(candidates, key=lambda c: c["fit_rmse_rad"])


class ControllerStudy:
    def __init__(self, protocol=None):
        self.protocol = json.loads(json.dumps(protocol or PROTOCOL))
        self.env = OpenArmEnv(images=False)
        validate_protocol(self.protocol, self.env.reference)
        self.clock = load_clock()
        self.aids = self.env.aids["right"]
        self.kp = self.env.model.actuator_gainprm[self.aids, 0].copy()
        self.kd = -self.env.model.actuator_biasprm[self.aids, 2].copy()
        self.started = time.monotonic()
        self.calls = 0

    def rollout(self, episode, scales, delay):
        if time.monotonic() - self.started > self.protocol["wall_budget_seconds"]:
            raise TimeoutError("controller study wall budget exceeded")
        env = self.env
        env.model.actuator_gainprm[self.aids, 0] = self.kp * scales[0]
        env.model.actuator_biasprm[self.aids, 1] = -self.kp * scales[0]
        env.model.actuator_biasprm[self.aids, 2] = -self.kd * scales[1]
        env.reset(seed=0, options={"episode": episode})
        times = np.asarray(self.clock["episodes"][str(episode)]["recorded_seconds"], float)
        count = min(self.protocol["frames"], int(np.searchsorted(times, self.protocol["window_seconds"], side="right")))
        times = times[:count]
        target = np.asarray(env.reference[str(episode)]["right_joints"][:count], float)
        predicted, contact_steps, self_contact_steps, contact_evidence = [], 0, 0, []
        simulated_times = []
        for frame in range(count):
            # Pre-command observation matches reference[frame]. Delay only the
            # right arm being fitted; left and gripper commands are unchanged.
            predicted.append(env.data.qpos[env.qids["right"]].copy())
            simulated_times.append(float(env.data.time))
            if frame + 1 == count:
                break  # No invented duration after the last recorded observation.
            action = env.reference_action(episode, frame)
            action[8:15] = env.reference_action(episode, max(0, frame-delay))[8:15]
            env.step_until(action, times[frame + 1])
            window = env.evaluation["contact_window"]
            external = any(p["robot_external"] for p in window["pairs"])
            robot_self = any(p["robot_self"] for p in window["pairs"])
            contact_steps += int(external)
            self_contact_steps += int(robot_self)
            if external or robot_self:
                contact_evidence.append({"frame": frame, "window": window})
            if env._done:
                raise ValueError("candidate physics terminated before calibration window ended")
        self.calls += 1
        prediction = np.asarray(predicted)
        return {"episode": episode, "prediction": prediction, "target": target,
                "recorded_seconds": times, "simulated_seconds": simulated_times,
                "metrics": metrics(prediction, target, self.protocol["warmup_frames"]),
                "robot_external_contact_steps": contact_steps,
                "robot_self_contact_steps": self_contact_steps, "contact_evidence": contact_evidence,
                "max_penetration_m": env.evaluation["max_penetration_m"]}

    def residual(self, x, delay):
        x = np.asarray(x).reshape(2, -1)
        columns = []
        for column in x.T:
            rows = [self.rollout(ep, np.exp(column), delay) for ep in self.protocol["fit_episodes"]]
            if any(r["robot_external_contact_steps"] or r["robot_self_contact_steps"] for r in rows):
                raise ValueError("contact contaminates this early-motion fitting window")
            error = np.concatenate([(r["prediction"]-r["target"])[self.protocol["warmup_frames"]:].ravel() for r in rows])
            columns.append(error / np.sqrt(len(error)))
        return np.stack(columns, axis=1)

    def close(self):
        self.env.close()


def study(output, protocol_name=PROTOCOL["id"]):
    if protocol_name != PROTOCOL["id"]:
        raise ValueError("unknown calibration protocol")
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "runs") or output.exists():
        raise ValueError("use a new calibration artifacts directory inside runs")
    output.mkdir(parents=True)
    worker = ControllerStudy()
    manifest = {"schema": "openarm-calibration/1", "protocol": PROTOCOL,
                "optimizer": "official mujoco.minimize.least_squares; finite differences of native dynamics",
                "model": worker.env.manifest(), "study_sha256": sha(__file__),
                "registry_sha256": sha(ROOT / "sim/calibration.json"),
                "acquisition_sha256": sha(ROOT / "reference/acquisition.json"),
                "clock_reader_sha256": sha(ROOT / "sim/reference_clock.py"),
                "acquisition": clock_summary(),
                "plotter_sha256": sha(ROOT / "tools/plot_calibration.py"), "transfer_ready": False}
    dump(output / "manifest.json", manifest)
    try:
        baseline_error = worker.residual(np.zeros(2), 0)
        baseline_rmse = float(np.linalg.norm(baseline_error))
        candidates = []
        for delay in PROTOCOL["delay_steps"]:
            x, trace = minimize.least_squares(
                np.zeros(2), lambda x: worker.residual(x, delay),
                bounds=(np.full(2, np.log(.25)), np.full(2, np.log(4.))),
                max_iter=PROTOCOL["max_iterations"], eps=1e-3, xtol=1e-4,
                gtol=1e-8, verbose=minimize.Verbosity.SILENT)
            error = worker.residual(x, delay)
            row = {"delay_steps": delay,
                   "kp_scale": float(np.exp(x.ravel()[0])), "kd_scale": float(np.exp(x.ravel()[1])),
                   "fit_rmse_rad": float(np.linalg.norm(error)), "iterations": len(trace)}
            candidates.append(row)
            dump(output / "progress.json", {"phase": "fitting", "completed": len(candidates), "total": 4,
                 "baseline_fit_rmse_rad": baseline_rmse, "candidates": candidates, "rollouts": worker.calls})
            print(json.dumps(row), flush=True)
        best = select_candidate(candidates)
        x = np.log([best["kp_scale"], best["kd_scale"]])
        r = worker.residual(x, best["delay_steps"])
        jac, _ = minimize.jacobian_fd(lambda x: worker.residual(x, best["delay_steps"]),
                                      x.reshape(2, 1), r, 1e-3, 0)
        sv = np.linalg.svd(jac, compute_uv=False)
        boundary = any(min(abs(v-.25), abs(v-4.)) < .02 for v in (best["kp_scale"], best["kd_scale"]))
        condition = float(sv[0]/sv[-1]) if sv[-1] > 1e-12 else None
        # Freeze selection on disk before reading the holdout trajectory.
        dump(output / "selected.json", {**best, "selection_uses_holdout": False})
        predictions, rows = {}, []
        for split, episodes in (("fit", PROTOCOL["fit_episodes"]), ("holdout", PROTOCOL["holdout_episodes"])):
            for episode in episodes:
                base = worker.rollout(episode, [1., 1.], 0)
                candidate = worker.rollout(episode, [best["kp_scale"], best["kd_scale"]], best["delay_steps"])
                predictions[str(episode)] = {"split": split, "target": base["target"],
                                            "recorded_seconds": base["recorded_seconds"], "simulated_seconds": base["simulated_seconds"],
                                            "baseline": base["prediction"], "candidate": candidate["prediction"]}
                rows.append({"episode": episode, "split": split, "baseline": base["metrics"], "candidate": candidate["metrics"],
                             "baseline_contact_steps": base["robot_external_contact_steps"],
                             "candidate_contact_steps": candidate["robot_external_contact_steps"],
                             "baseline_self_contact_steps": base["robot_self_contact_steps"],
                             "candidate_self_contact_steps": candidate["robot_self_contact_steps"],
                             "baseline_contact_evidence": base["contact_evidence"], "candidate_contact_evidence": candidate["contact_evidence"]})
        holdout = [row for row in rows if row["split"] == "holdout"]
        ratio = float(np.mean([r["candidate"]["rmse_rad"] for r in holdout]) / np.mean([r["baseline"]["rmse_rad"] for r in holdout]))
        result = {"status": "studied", "scope": PROTOCOL["scope"], "protocol": PROTOCOL["id"],
                  "baseline_fit_rmse_rad": baseline_rmse, "candidates": candidates, "selected": best,
                  "episodes": rows, "holdout_error_ratio": ratio, "holdout_improved": ratio < 1.,
                  "fit_jacobian_singular_values": sv.tolist(), "fit_jacobian_condition": condition,
                  "at_parameter_bound": boundary, "rollouts": worker.calls,
                  "wall_seconds": time.monotonic()-worker.started,
                  "applied_to_runtime": False, "calibrated": False, "transfer_ready": False,
                  "limitations": ["Only one development-validation early-motion episode, previously inspected; no fresh test set or full-task validation.",
                     "Logger timestamps recovered, but sensor exposure/feedback and CAN send timestamps remain unavailable; offsets are sample indices, not measured bus latency.",
                     "The recorded clock assumes commands persist between logger entries; collection-time source revision and motor watchdog semantics remain unverified.",
                     "Gains come from another deployment log; this study does not establish physical motor parameters.",
                     "Left arm, grippers, cameras, base transform, friction and object properties are not fitted.",
                     "Window contact absence is checked in sim only; real no-contact assumption needs synchronized video.",
                     "Boundary solutions and ill conditioning require richer data, not automatic parameter promotion."]}
        dump(output / "predictions.json", predictions)
        dump(output / "calibration.json", result)
        plot_python = Path(json.loads((ROOT / "sim/learning.json").read_text())["plot_python"])
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH") if k in os.environ}
        subprocess.run([str(plot_python), "-B", str(ROOT / "tools/plot_calibration.py"), str(output)],
                       check=True, timeout=60, env=env, cwd=ROOT)
        dump(output / "progress.json", {"phase": "completed", "completed": 4, "total": 4})
        return result
    finally:
        worker.close()


def report():
    registry = json.loads((ROOT / "sim/calibration.json").read_text())
    profile = json.loads((ROOT / "sim/profile.json").read_text())
    historical = json.loads((ROOT / "calibrate.json").read_text())
    for item in registry["parameters"]:
        if item.get("profile_key"):
            item["value"] = profile[item["profile_key"]]
        elif item.get("calibration_key"):
            item["value"] = historical["actuators"][item["calibration_key"]]
    registry.update(transfer_ready=False, auto_apply=False, protocol=PROTOCOL,
                    acquisition=clock_summary(),
                    profile_sha256=sha(ROOT / "sim/profile.json"))
    registry["latest_study"] = None
    for path in sorted((ROOT / "runs").glob("*-calibrate-*/artifacts/calibration.json"), reverse=True):
        result = json.loads(path.read_text())
        if result.get("status") != "studied":
            continue
        manifest = json.loads((path.parent / "manifest.json").read_text())
        dependencies=response_dependencies(manifest['model'])
        matching = bool(dependencies) and all(sha(ROOT / name) == value for name, value in dependencies.items())
        matching = matching and manifest.get("protocol") == PROTOCOL and manifest.get("acquisition_sha256") == sha(ROOT / "reference/acquisition.json")
        registry["latest_study"] = {"run_id": path.parent.parent.name, "selected": result["selected"],
            "holdout_error_ratio": result["holdout_error_ratio"], "at_parameter_bound": result["at_parameter_bound"],
            "matches_current_model": matching, "applied_to_runtime": False,
            "result_url": "/runs/" + str(path.relative_to(ROOT / "runs")),
            "plot_url": "/runs/" + str((path.parent / "calibration.png").relative_to(ROOT / "runs")) if (path.parent / "calibration.png").is_file() else None}
        break
    from .wrist import latest as latest_wrist
    wrist = latest_wrist()
    registry["latest_wrist"] = {k:v for k,v in wrist.items() if k != "samples"} if wrist['status'] == 'completed' else None
    from .gripper_study import latest as latest_gripper
    gripper = latest_gripper()
    registry["latest_gripper"] = ({k: gripper[k] for k in (
        "run_id", "aggregate_rmse_rad", "matches_current_model", "physically_calibrated", "report_url")}
        if gripper else None)
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=[PROTOCOL["id"]], default=PROTOCOL["id"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = study(args.output, args.protocol)
    print(json.dumps({"status": result["status"], "selected": result["selected"], "holdout_error_ratio": result["holdout_error_ratio"]}), flush=True)


if __name__ == "__main__":
    main()
