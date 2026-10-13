"""Plot one native transition log. Requires matplotlib; never changes input data."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    folder = args.run.resolve()
    rows = [json.loads(line) for line in (folder / "transitions.jsonl").read_text().splitlines()]
    if not rows:
        raise SystemExit("No transitions to plot")
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    # Font cache belongs only to this task and is removed after plotting.
    with tempfile.TemporaryDirectory(prefix="openarm-plot-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "axes.edgecolor": "#cccccc", "axes.labelcolor": "#556057"})
        times = np.array([row["observation"]["sim_time"] for row in rows])
        errors = []
        for row in rows:
            d = row.get("diagnostics") or {}
            errors.append(d.get("pre_action_joint_error_rad", d.get("joint_error_rad", [float("nan")] * 14)))
        errors = np.array(errors)
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
        colors = ("#b67559", "#56786b")
        for side, start, color in zip(("Left", "Right"), (0, 7), colors):
            rms = np.sqrt(np.mean(errors[:, start:start+7] ** 2, axis=1))
            axes[0].plot(times, rms, color=color, label=f"{side} arm (7 joints)", lw=1.3)
        axes[0].set_ylabel("Joint RMSE (rad)")
        axes[0].legend(frameon=False, ncol=2)
        axes[1].plot(times, [1000*row["evaluation"]["penetration_m"] for row in rows], color="#8c7868", lw=1.2)
        axes[1].axhline(10, color="#b85f58", linestyle="--", lw=1, label="10 mm invalidity threshold")
        axes[1].set_ylabel("Penetration (mm)")
        axes[1].legend(frameon=False)
        windows = [row["evaluation"].get("contact_window", {}) for row in rows]
        for key, color, label in (("robot_external_peak_force_N", "#b67559", "1 ms samples: interval peak"),
                                  ("endpoint_robot_external_force_N", "#56786b", "Control endpoint only")):
            axes[2].plot(times, [w.get(key, float("nan")) for w in windows], color=color, label=label, lw=1)
        axes[2].set_ylabel("Robot external force (N)")
        axes[2].legend(frameon=False)
        for side, color in zip(("left", "right"), colors):
            axes[3].step(times, [int(row["evaluation"]["held"][side]) for row in rows], where="post", label=side+" gripper", color=color)
        axes[3].set_yticks([0, 1], ["No", "Yes"])
        axes[3].set_ylabel("Verified two-finger hold")
        axes[3].set_xlabel("Simulation time (s); reference alignment assumes zero lag")
        axes[3].legend(frameon=False)
        for ax in axes:
            ax.grid(alpha=.15)
            ax.set_xlim(times[0], times[-1] if len(times)>1 else times[0]+.034)
        episode = manifest.get("details", {}).get("episode", "-")
        fig.suptitle(f"OpenArm / episode {episode} / {summary['status']}\nCandidate model diagnostic — not a sim-to-real benchmark", fontsize=14, color="#37483a")
        fig.savefig(folder / "diagnostics.png", dpi=150, facecolor="#fafbf8")
        plt.close(fig)
    print(folder / "diagnostics.png")


if __name__ == "__main__":
    main()
