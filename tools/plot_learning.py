"""Plot actual learning metrics and a separate offline evaluation artifact."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    train = args.training.resolve()
    evaluation = args.evaluation.resolve()
    rows = [json.loads(line) for line in (train / "metrics.jsonl").read_text().splitlines()]
    result = json.loads((evaluation / "evaluation.json").read_text())
    with tempfile.TemporaryDirectory(prefix="openarm-plot-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "axes.edgecolor": "#c8cec4", "axes.labelcolor": "#4d5c50"})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
        for key, label, color in (("loss", "Total loss", "#ac6e50"), ("l1_loss", "L1 loss", "#587e68"), ("kld_loss", "KL loss", "#8e9690")):
            axes[0].plot([r["step"] for r in rows], [r[key] for r in rows], marker="o", ms=4, color=color, label=label)
        axes[0].set(xlabel="Optimizer step", ylabel="Loss", title="Official ACT / CPU pipeline smoke test")
        axes[0].set_xticks([r["step"] for r in rows])
        axes[0].legend(frameon=False)
        indices = [*range(7), *range(8, 15)]
        axes[1].bar(np.arange(14), [result["per_channel_rmse"][i] for i in indices], color=["#587e68"]*7+["#ac6e50"]*7, width=.7)
        axes[1].set_xticks(np.arange(14), [f"L{i+1}" for i in range(7)]+[f"R{i+1}" for i in range(7)], fontsize=8)
        axes[1].set(ylabel="Joint command RMSE (rad)", title=f"Separate episode / {result['frames']} recorded frames")
        for ax in axes:
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
        fig.suptitle("OpenArm learning pipeline — diagnostic data only\nNo closed-loop task success or sim-to-real performance measured", fontsize=13, color="#465347")
        fig.savefig(evaluation / "learning.png", dpi=150, facecolor="#fbfcf7")
        plt.close(fig)
    print(evaluation / "learning.png")


if __name__ == "__main__":
    main()
