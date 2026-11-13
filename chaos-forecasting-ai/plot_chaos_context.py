"""Show long Van der Pol dynamics alongside the short accuracy-based display."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_long_horizon import COLORS, local_accuracy_strip


def plot_context(out, display_units=40.0):
    out = Path(out)
    config = json.loads((out / "config.json").read_text())
    metrics = {
        name: value
        for name, value in json.loads((out / "metrics.json").read_text()).items()
        if name not in {"Persistence", "Ridge"}
    }
    window = json.loads((out / "plot-window.json").read_text())["maximum_time"]
    with np.load(out / "data.npz") as data:
        truth = data["truth"][0]
        dt = config["dt"]
        lead = np.arange(1, len(truth) + 1) * dt
        period = 2 * np.pi / config["forcing_frequency"]
        plt.style.use("dark_background")
        figure, axes = plt.subplots(3, 1, figsize=(13, 10))
        for axis, state in zip(axes[:2], [0, 1], strict=True):
            axis.plot(lead, truth[:, state], color="white", linewidth=0.9)
            axis.axvspan(
                0,
                window,
                color="#f6ca67",
                alpha=0.25,
                label=f"Current close-up: 0–{window:g}",
            )
            axis.set(
                xlim=(0, lead[-1]),
                ylabel=config["labels"][state],
                xlabel="Time since last observation",
            )
            axis.legend(loc="upper right", fontsize=8)
            axis.grid(alpha=0.15)
        # Use absolute simulation times to sample a fixed forcing phase.
        absolute = (int(data["reference_origin"]) + np.arange(len(truth))) * dt
        strobes = (
            np.arange(
                np.ceil(absolute[0] / period), np.floor(absolute[-1] / period) + 1
            )
            * period
        )
        x = np.interp(strobes, absolute, truth[:, 0])
        v = np.interp(strobes, absolute, truth[:, 1])
        axes[2].scatter(x, v, c=np.arange(len(x)), cmap="viridis", s=22)
        axes[2].set(
            xlabel="x at forcing phase 0",
            ylabel="v at forcing phase 0",
            title=f"Fixed-phase section: {len(strobes)} sampled forcing cycles",
        )
        axes[2].grid(alpha=0.15)
        figure.suptitle(
            f"Forced Van der Pol: full {lead[-1]:g}-unit truth trajectory; "
            f"forcing period {period:.2f}"
        )
        figure.text(
            0.03,
            0.01,
            "Yellow: short accuracy view. Same saved simulation over a longer window.\n"
            "Fixed-phase samples interpolate the saved grid; "
            "scatter alone does not prove chaos.",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.045, 1, 0.96))
        for extension in ["png", "pdf"]:
            figure.savefig(out / "figures" / f"chaos-context.{extension}", dpi=150)
        plt.close(figure)
        figure, axes = plt.subplots(
            len(metrics),
            2,
            figsize=(14, 2.7 * len(metrics)),
            sharex=True,
            squeeze=False,
        )
        maximum = min(display_units, lead[-1])
        for row, (name, color) in enumerate(zip(metrics, COLORS, strict=True)):
            with np.load(out / (name.lower().replace(" ", "-") + ".npz")) as saved:
                for state, axis in enumerate(axes[row]):
                    axis.plot(
                        lead,
                        truth[:, state],
                        color="white",
                        linewidth=1,
                        label="Simulated truth",
                    )
                    axis.plot(
                        lead,
                        saved["mean"][0, :, state],
                        color=color,
                        linewidth=0.9,
                        label="Forecast mean",
                    )
                    local_accuracy_strip(
                        axis,
                        lead,
                        saved["within_tolerance_by_origin_state"][0, :, state],
                        saved["episode_status_by_origin"][0]
                        if "episode_status_by_origin" in saved
                        else None,
                    )
                    axis.set(
                        xlim=(0, maximum),
                        ylabel=config["labels"][state],
                        xlabel="Time since last observation",
                        title=f"{name} | {config['labels'][state]}",
                    )
                    low, high = truth[:, state].min(), truth[:, state].max()
                    axis.set_ylim(low - 0.25 * (high - low), high + 0.25 * (high - low))
                    axis.legend(fontsize=8, loc="upper right")
                    axis.grid(alpha=0.15)
        figure.suptitle(
            f"Forced Van der Pol: {maximum:g}-unit context; "
            "shared scale across models and variables"
        )
        figure.text(
            0.03,
            0.01,
            "White: truth; colored: mean. "
            "Upper strip: tracking / temporary mismatch / sustained breakdown.\n"
            "Original free forecasts, no truth updates. "
            "Y limits use truth ±25%; all errors remain scored.",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.045, 1, 0.97))
        for extension in ["png", "pdf"]:
            figure.savefig(
                out / "figures" / f"long-context-overlays.{extension}", dpi=150
            )
        plt.close(figure)
        (out / "chaos-context.json").write_text(
            json.dumps(
                {
                    "forcing_period": period,
                    "accuracy_display_units": window,
                    "accuracy_display_forcing_cycles": window / period,
                    "additional_comparison_display_units": maximum,
                    "truth_horizon_units": float(lead[-1]),
                    "fixed_phase_samples": len(strobes),
                    "note": "Same forecasts and dynamics; longer display only.",
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--display-units", type=float, default=40.0)
    args = parser.parse_args()
    plot_context(args.results, args.display_units)
