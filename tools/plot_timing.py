"""Plot the paired delay study; cache is temporary and removed on exit."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    folder = parser.parse_args().run.resolve()
    result = json.loads((folder / "summary.json").read_text())
    traces = json.loads((folder / "command_traces.json").read_text())
    if result["status"] != "completed" or len(result["trials"]) != 9:
        raise ValueError("a completed nine-condition study is required")
    with tempfile.TemporaryDirectory(prefix="openarm-timing-plot-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        colors = ["#61756b", "#c78658", "#77689c"]
        for delay, color in zip((0, 100, 300), colors):
            trace = traces[f"ep0-delay{delay}ms"]
            ax[0, 0].plot([r["command"] for r in trace], [r["bottle"][2] for r in trace],
                          color=color, label=f"{delay} ms / chunk")
        ax[0, 0].set(title="Episode 0 / bottle trajectory", xlabel="Recorded command index", ylabel="Bottle body height (m)")
        ax[0, 0].legend(frameon=False)
        configs = [("joint_rmse_vs_no_delay_rad", 1000, "Joint deviation from zero delay", "RMSE (mrad)", ax[0, 1]),
                   ("bottle_rms_distance_vs_no_delay_m", 1000, "Bottle deviation from zero delay", "RMS position distance (mm)", ax[1, 0]),
                   ("sim_seconds", 1, "Physical time until termination", "Simulation seconds", ax[1, 1])]
        for key, scale, title, ylabel, axis in configs:
            for j, (delay, color) in enumerate(zip((0, 100, 300), colors)):
                rows = [r for r in result["trials"] if r["delay_ms"] == delay]
                values = [r[key]*scale for r in rows]
                axis.bar(np.arange(3) + (j-1)*.24, values, width=.22, color=color, label=f"{delay} ms")
            axis.set_xticks(range(3), ["Episode 0", "Episode 1", "Episode 2"])
            axis.set(title=title, ylabel=ylabel)
            axis.grid(axis="y", alpha=.15)
        ax[1, 1].legend(frameon=False, ncol=3, fontsize=9)
        fig.suptitle("OpenArm / delay sensitivity\nRecorded commands only — no trained VLA; synthetic latency", fontsize=16, color="#33463b")
        passed = sum(row["success"] for row in result["trials"])
        fig.supxlabel(f"Deviations use common command prefixes, not matched wall times. Task predicate met in {passed}/9 diagnostic conditions.", fontsize=10)
        fig.savefig(folder / "timing.png", dpi=150, facecolor="#fafbf8")
        plt.close(fig)
    print(folder / "timing.png")


if __name__ == "__main__":
    main()
