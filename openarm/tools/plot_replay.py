"""Compare paired clock replays on common reference samples, without selection."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    root = parser.parse_args().study.resolve()
    report = json.loads((root / "summary.json").read_text())
    trials = report["trials"]
    if {(r["episode"], r["clock"]) for r in trials} != {(ep, c) for ep in range(3) for c in ("recorded", "uniform30")} or len(trials) != 6:
        raise ValueError("all six predeclared conditions are required")
    with tempfile.TemporaryDirectory(prefix="openarm-plot-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained")
        colors = {"recorded": "#56786b", "uniform30": "#b67559"}
        comparisons = []
        for ep in range(3):
            records = {}
            manifests = []
            for row in (r for r in trials if r["episode"] == ep):
                path = (root / row["run"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("trial outside study")
                manifests.append(json.loads((path / "manifest.json").read_text()))
                records[row["clock"]] = [json.loads(s) for s in (path / "transitions.jsonl").read_text().splitlines()]
            if manifests[0]["hashes"] != manifests[1]["hashes"] or manifests[0]["profile"] != manifests[1]["profile"]:
                raise ValueError("clock comparison changed the physical model")
            count = min(map(len, records.values()))
            comparison = {"episode": ep, "common_transitions": count, "conditions": {}}
            for mode, rows in records.items():
                rows = rows[:count]
                t = np.array([r["diagnostics"]["reference_time_seconds"] for r in rows] + [rows[-1]["diagnostics"]["next_reference_time_seconds"]])
                errors = np.array([r["diagnostics"]["pre_action_joint_error_rad"] for r in rows] + [rows[-1]["diagnostics"]["next_joint_error_rad"]])
                sim = np.array([r["observation"]["sim_time"] for r in rows] + [rows[-1]["next_observation"]["sim_time"]])
                label = "Recorded clock" if mode == "recorded" else "Uniform 30 Hz"
                axes[0, ep].plot(t, np.sqrt(np.mean(errors**2, axis=1)), color=colors[mode], label=label, lw=1.1)
                force = [r["evaluation"]["contact_window"]["robot_external_peak_force_N"] for r in rows]
                axes[1, ep].plot(t[:-1], force, color=colors[mode], lw=1)
                axes[2, ep].plot(t, sim-t, color=colors[mode], lw=1.2)
                comparison["conditions"][mode] = {
                    "common_sample_joint_rmse_rad": float(np.sqrt(np.mean(errors**2))),
                    "first_four_logger_seconds_joint_rmse_rad": float(np.sqrt(np.mean(errors[t <= 4]**2))),
                    "last_common_reference_seconds": float(t[-1]), "last_common_sim_seconds": float(sim[-1]),
                }
            comparisons.append(comparison)
            axes[0, ep].set_title(f"Episode {ep+1} / {count+1} common observations")
            axes[2, ep].set_xlabel("Reference logger time (s)")
            for ax in axes[:, ep]:
                ax.grid(alpha=.17)
                ax.set_xlim(0, t[-1])
        axes[0, 0].set_ylabel("14-joint RMSE (rad)")
        axes[1, 0].set_ylabel("Robot external peak force (N)")
        axes[2, 0].set_ylabel("Sim time - logger time (s)")
        axes[0, 0].legend(frameon=False)
        successes = sum(bool(row["success"]) for row in trials)
        fig.suptitle(f"OpenArm / replay clock comparison\nSame commands and model; common sample prefix only; {successes}/6 diagnostic task successes", fontsize=14)
        fig.savefig(root / "comparison.png", dpi=150, facecolor="#fafbf8")
        plt.close(fig)
        report["comparisons"] = comparisons
        report["comparison_scope"] = "Same recorded sample indices over common prefix; 14 joints, no gripper. Not a benchmark or a calibration success claim."
        (root / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(comparisons, indent=2))
        print(root / "comparison.png")


if __name__ == "__main__":
    main()
