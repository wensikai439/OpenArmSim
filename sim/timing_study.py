"""Fixed, paired latency sensitivity study using recorded commands, not a VLA."""
from __future__ import annotations

import json
import argparse
import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .environment import ROOT, OpenArmEnv, digest
from .recording import Recorder, dump
from .timing import PolicyTiming


class RecordedChunks:
    def __init__(self, env, episode):
        self.env, self.episode, self.index = env, episode, 0

    def infer(self, env, task):
        end = min(self.index + 50, env.reference[str(self.episode)]["frames"])
        values = np.array([env.reference_action(self.episode, i) for i in range(self.index, end)])
        self.index = end
        return values


def run_study(output=None):
    root = Path(output).resolve() if output else ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-timing-" + uuid.uuid4().hex[:6])
    if not root.is_relative_to(ROOT / "runs") or root == ROOT / "runs":
        raise ValueError("timing output must stay in a new local run directory")
    root.mkdir(parents=True, exist_ok=False)
    env = OpenArmEnv(images=False)
    manifest = {"kind": "timing_study", "schema": "openarm-timing-study/1", "status": "running",
                "scope": "paired diagnostic with recorded commands, NOT a VLA or benchmark",
                "protocol": {"episodes": [0, 1, 2], "delays_seconds": [0., .1, .3], "chunk_size": 50,
                             "command_clock": "uniform 30 Hz policy-like schedule, not original logger-time replay",
                             "seed": 0, "time_limit_seconds": 40, "selection": "all conditions reported; no best-run selection"},
                "model": env.manifest(), "study_sha256": digest(__file__),
                "limitations": ["synthetic delays are not measured hardware or model latency",
                                "recorded commands are deliberately retimed to a 30 Hz policy-like schedule; this does not reconstruct acquisition timing",
                                "initial object poses are not reconstructed independently for episodes 1 and 2",
                                "records contain states and commands, not VLA observations or predictions"],
                "transfer_ready": False}
    dump(root / "manifest.json", manifest)
    rows, command_traces = [], {}
    status, error = "completed", None
    try:
        for episode in (0, 1, 2):
            for delay in (0., .1, .3):
                env.reset(seed=0, options={"episode": episode})
                timing = PolicyTiming("measured_delay", fixed_delay=delay)
                recorder = Recorder(env, f"ep{episode}-delay{round(delay*1000)}ms", record_images=False, seed=0,
                                    root=root / "trials", details={"episode": episode, "timing": timing.contract(),
                                    "policy": "NONE: recorded command sequence grouped into 50-step chunks",
                                    "purpose": "diagnostic only"})
                before = recorder.observation(env)
                result, failure, trace = "reference_exhausted", None, []
                impulse, max_force = 0., 0.
                try:
                    source = RecordedChunks(env, episode)
                    for requested, diagnostics, held in timing.actions(env, source,
                            {"max_steps": env.reference[str(episode)]["frames"]}, recorder.event):
                        action = requested if held else env.prepare_policy_action(requested)[0]
                        diagnostics["requested_action"] = requested.tolist()
                        if not held:
                            trace.append({"command": timing.commands, "sim_time": float(env.data.time),
                                          "state": env.state().tolist(), "bottle": env.data.xpos[env.bottle_id].tolist()})
                        _, _, terminal, truncated, info = env.step(action)
                        after = recorder.observation(env)
                        recorder.append(before, action, after, info, terminated=terminal, truncated=truncated, diagnostics=diagnostics)
                        timing.committed(held, before["sim_time"], after["sim_time"])
                        before = after
                        contact = info["evaluation"]["contact_window"]
                        impulse += contact["robot_external_impulse_Ns"]
                        max_force = max(max_force, contact["robot_external_peak_force_N"])
                        if terminal or truncated:
                            result = "success" if info["evaluation"]["success"] else "invalid" if terminal else "truncated"
                            break
                except KeyboardInterrupt:
                    result, failure = "cancelled", "interrupted"
                    raise
                except Exception as exc:
                    result, failure = "error", str(exc)
                    raise
                finally:
                    recorder.finish(result, env.info(), error=failure, metrics={"timing": timing.summary()})
                Image.fromarray(env.render("overview")).save(recorder.path / "final.png")
                key = f"ep{episode}-delay{round(delay*1000)}ms"
                command_traces[key] = trace
                row = {"episode": episode, "delay_ms": round(delay*1000), "run": str(recorder.path.relative_to(root)),
                       "status": result, "success": env.evaluation["success"],
                       "stages": [e["name"] for e in env.events], "sim_seconds": float(env.data.time),
                       "max_penetration_mm": env.evaluation["max_penetration_m"]*1000,
                       "robot_external_impulse_Ns": impulse, "robot_external_peak_force_N": max_force,
                       "timing": timing.summary()}
                rows.append(row)
                dump(root / "progress.json", {"completed": len(rows), "total": 9, "latest": row})
                print(json.dumps({k: row[k] for k in ("episode", "delay_ms", "status", "sim_seconds")}), flush=True)
        for row in rows:
            baseline = command_traces[f"ep{row['episode']}-delay0ms"]
            current = command_traces[f"ep{row['episode']}-delay{row['delay_ms']}ms"]
            count = min(len(baseline), len(current))
            error_q = np.array([a["state"] for a in current[:count]]) - np.array([a["state"] for a in baseline[:count]])
            error_b = np.array([a["bottle"] for a in current[:count]]) - np.array([a["bottle"] for a in baseline[:count]])
            row["paired_common_commands"] = count
            row["joint_rmse_vs_no_delay_rad"] = float(np.sqrt(np.mean(error_q[:, [*range(7), *range(8,15)]]**2)))
            row["bottle_rms_distance_vs_no_delay_m"] = float(np.sqrt(np.mean(np.sum(error_b**2, axis=1))))
        dump(root / "command_traces.json", command_traces)
    except KeyboardInterrupt:
        status, error = "cancelled", "interrupted"
    except Exception as exc:
        status, error = "error", str(exc)
    finally:
        env.close()
        manifest["status"] = status
        dump(root / "manifest.json", manifest)
        dump(root / "summary.json", {"status": status, "scope": manifest["scope"], "trials": rows,
                                     "error": error, "transfer_ready": False})
    if status == "completed":
        python = json.loads((ROOT / "sim/learning.json").read_text())["plot_python"]
        plot_env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH") if k in os.environ}
        subprocess.run([python, "-B", str(ROOT / "tools/plot_timing.py"), str(root)],
                       check=True, timeout=60, cwd=ROOT, env=plot_env)
    print(json.dumps({"run": str(root), "status": status, "error": error}), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run_study(parser.parse_args().output))
