"""Command replay on audited logger time; shared by CLI and UI.

N recorded observations delimit N-1 command intervals. The last command has
no known end timestamp and is never given an invented execution duration.
This is a zero-order-hold hypothesis, not a sensor/CAN timing measurement.
"""
import numpy as np

from .environment import ROOT, digest
from .reference_clock import load


class ReferenceReplay:
    def __init__(self, env, episode, clock="recorded"):
        if clock not in {"recorded", "uniform30"}:
            raise ValueError("replay clock must be recorded or uniform30")
        if env.tick != 0 or float(env.data.time) != 0 or env.episode != episode:
            raise ValueError("reset to the selected episode before creating a replay")
        self.env, self.episode, self.clock = env, episode, clock
        audit = load()
        row = audit["episodes"][str(episode)]
        self.recorded = np.asarray(row["recorded_seconds"], dtype=float)
        self.times = self.recorded if clock == "recorded" else np.arange(len(self.recorded)) / 30.
        self.intervals = len(self.times) - 1
        self.errors, self.sim_times, self.phase = [], [], []
        self._sample(0)
        self.contract = {
            "mode": clock, "intervals": self.intervals, "samples": len(self.times),
            "source": audit["timestamp_semantics"],
            "recorded_duration_seconds": float(self.recorded[-1]),
            "scheduled_duration_seconds": float(self.times[-1]),
            "acquisition_sha256": digest(ROOT / "reference/acquisition.json"),
            "reference_sha256": audit["reference_sha256"], "replay_sha256": digest(__file__),
            "final_command": "not executed: no following recorded timestamp",
            "hold_assumption": "command[f] held over [timestamp[f], timestamp[f+1]); zero offset",
            "sampling_error_bound_seconds": float(env.model.opt.timestep) / 2,
            "limitations": audit["limitations"] + [
                "Initial velocity is zero, not a measured acquisition velocity.",
                "Scene/object initial poses remain candidates; this is not a task benchmark.",
                *( ["Uniform 30 Hz deliberately time-warps the recording; diagnostic comparison only."] if clock == "uniform30" else []),
            ],
        }

    def _sample(self, index):
        e = self.env.reference[str(self.episode)]
        measured = np.array([v for side in ("left", "right") for v in e[f"{side}_joints"][index]])
        error = self.env.state()[[*range(7), *range(8, 15)]] - measured
        self.errors.append(error.tolist())
        self.sim_times.append(float(self.env.data.time))
        self.phase.append(float(self.env.data.time) - float(self.times[index]))

    def advance(self):
        index = len(self.errors) - 1
        if index >= self.intervals:
            raise StopIteration("no following observation to delimit another command")
        if self.env.tick != index:
            raise ValueError("environment advanced outside this replay")
        action = self.env.reference_action(self.episode, index)
        outcome = (self.env.step_until(action, self.times[index+1]) if self.clock == "recorded"
                   else self.env.step(action))
        self._sample(index+1)
        diagnostics = {
            "reference_frame": index, "next_reference_frame": index+1,
            "reference_time_seconds": float(self.recorded[index]),
            "next_reference_time_seconds": float(self.recorded[index+1]),
            "scheduled_time_seconds": float(self.times[index]),
            "next_scheduled_time_seconds": float(self.times[index+1]),
            "clock_phase_error_seconds": self.phase[-1],
            "pre_action_joint_error_rad": self.errors[-2],
            "next_joint_error_rad": self.errors[-1],
            "alignment": "same recorded sample index; command/feedback offset not identified",
        }
        return action, outcome, diagnostics

    def metrics(self):
        errors, times = np.asarray(self.errors), np.asarray(self.sim_times)
        values = {
            "joint_rmse_rad": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            "joint_p95_abs_rad": np.percentile(abs(errors), 95, axis=0).tolist(),
            "scored_observations": len(errors), "includes_final_observation": True,
            "alignment": "initial observation and every endpoint; zero command/feedback offset hypothesis",
            "max_clock_phase_error_seconds": float(np.max(abs(np.asarray(self.phase)))),
            "replay_clock": self.contract, "holdout": False, "physical_calibration_validated": False,
        }
        if len(times) > 1:
            integral = np.sum(.5 * (errors[:-1]**2 + errors[1:]**2) * np.diff(times)[:, None], axis=0)
            values["time_weighted_joint_rmse_rad"] = np.sqrt(integral / (times[-1]-times[0])).tolist()
            values["time_weighting"] = "trapezoidal squared residuals on realized physics times"
        return values
