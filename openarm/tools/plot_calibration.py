"""Plot a completed controller study with Matplotlib; inputs stay unchanged."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    folder = args.artifacts.resolve()
    root = Path(__file__).resolve().parents[1]
    if not folder.is_relative_to(root / "runs"):
        raise SystemExit("plots must stay inside local runs")
    result = json.loads((folder / "calibration.json").read_text())
    predictions = json.loads((folder / "predictions.json").read_text())
    with tempfile.TemporaryDirectory(prefix="openarm-plot-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                             "axes.spines.top":False, "axes.spines.right":False})
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for row, color, label in (("baseline", "#ad816e", "Current model"), ("candidate", "#527c6d", "Fit candidate")):
            ys = [r[row]["rmse_rad"] * 1000 for r in result["episodes"]]
            x = np.arange(len(ys)) + (-.18 if row == "baseline" else .18)
            axes[0,0].bar(x, ys, width=.35, color=color, label=label)
        axes[0,0].set_xticks(range(len(result["episodes"])), [f"ep{r['episode']} / {r['split']}" for r in result["episodes"]])
        axes[0,0].set_ylabel("Right-arm RMSE (mrad)")
        axes[0,0].set_title("Fit episodes and development validation")
        axes[0,0].legend(frameon=False)
        candidates = result["candidates"]
        axes[0,1].plot([r["delay_steps"] for r in candidates], [r["fit_rmse_rad"]*1000 for r in candidates], "o-", color="#527c6d")
        axes[0,1].axhline(result["baseline_fit_rmse_rad"]*1000, color="#ad816e", ls="--", label="Current model")
        axes[0,1].set(xlabel="Effective offset (recorded samples)", ylabel="Fit RMSE (mrad)", title="Selection uses fit data only")
        axes[0,1].legend(frameon=False)
        holdout = next(v for v in predictions.values() if v["split"] == "holdout")
        q = np.asarray(holdout["target"]); times = np.asarray(holdout.get("recorded_seconds", np.arange(len(q))/30))
        time_label = "Logger time (s)" if "recorded_seconds" in holdout else "Inferred time (s)"
        # Show the most excited measured holdout joint, not the best-fitting one.
        joint = int(np.argmax(np.ptp(q, axis=0)))
        axes[1,0].plot(times, q[:,joint], color="#303c38", label="Recorded encoder")
        for key, color, label in (("baseline","#ad816e","Current model"),("candidate","#527c6d","Fit candidate")):
            a = np.asarray(holdout[key])
            axes[1,0].plot(times, a[:,joint], color=color, label=label)
            axes[1,1].plot(times, np.sqrt(np.mean((a-q)**2, axis=1))*1000, color=color, label=label)
        axes[1,0].set(xlabel=time_label, ylabel="Joint position (rad)", title=f"Holdout / right J{joint+1}")
        axes[1,0].legend(frameon=False, fontsize=8)
        axes[1,1].set(xlabel=time_label, ylabel="Right-arm RMSE (mrad)", title="Holdout trajectory error")
        axes[1,1].axvspan(0, times[min(10, len(times)-1)], color="#aaaaaa", alpha=.15, label="Unscored warmup")
        axes[1,1].legend(frameon=False, fontsize=8)
        for ax in axes.flat: ax.grid(alpha=.13, axis="y")
        s=result["selected"]
        fig.suptitle(f"OpenArm controller study / Kp x{s['kp_scale']:.2f}, Kd x{s['kd_scale']:.2f}, offset {s['delay_steps']} samples\n"
                     f"Early response only; {time_label}; candidate NOT applied; no transfer claim", fontsize=13)
        fig.savefig(folder/"calibration.png", dpi=150, facecolor="#fafbf8")
        plt.close(fig)
    print(folder/"calibration.png")


if __name__ == "__main__":
    main()
