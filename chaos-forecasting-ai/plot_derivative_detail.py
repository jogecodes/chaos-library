"""Inspect a forecast crossing using value, slope, curvature and local errors."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from long_horizon_metrics import causal_derivatives
from plot_long_horizon import feature_error_lines, local_accuracy_strip


def plot_detail(out, model="Ridge", start=5.0, end=7.0, state=0):
    out = Path(out)
    config = json.loads((out / "config.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())[model]
    settings = metrics["coherence_settings"]
    dt = config["dt"]
    slug = model.lower().replace(" ", "-")
    with np.load(out / "data.npz") as data, np.load(out / f"{slug}.npz") as saved:
        truth, prediction = data["truth"], saved["mean"]
        w = settings["derivative_window_steps"]
        past = data["test_contexts"][:, -(w - 1) :]
        actual_derivatives = causal_derivatives(
            np.concatenate([past, truth], axis=1), dt, w
        )[:, w - 1 :]
        predicted_derivatives = causal_derivatives(
            np.concatenate([past, prediction], axis=1), dt, w
        )[:, w - 1 :]
        lead = np.arange(1, truth.shape[1] + 1) * dt
        plt.style.use("dark_background")
        figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        series = [(truth, prediction, config["labels"][state])]
        series.extend(
            (actual_derivatives[..., k], predicted_derivatives[..., k], label)
            for k, label in enumerate(["Slope / time", "Curvature / time²"])
        )
        for axis, (actual, predicted, label) in zip(axes[:3], series, strict=True):
            axis.plot(lead, actual[0, :, state], color="white", label="Simulated truth")
            axis.plot(lead, predicted[0, :, state], color="#80b6ed", label=model)
            selection = (lead >= start) & (lead <= end)
            values = np.concatenate(
                [actual[0, selection, state], predicted[0, selection, state]]
            )
            lo, hi = np.nanmin(values), np.nanmax(values)
            margin = max(0.1 * (hi - lo), 0.01)
            axis.set(ylim=(lo - margin, hi + margin), ylabel=label)
            local_accuracy_strip(
                axis, lead, saved["within_tolerance_by_origin_state"][0, :, state]
            )
            axis.legend(fontsize=8)
        feature_error_lines(axes[3], lead, saved, state)
        axes[3].plot(
            lead,
            saved["rolling_nrmse_by_origin_state"][0, :, state],
            color="white",
            linewidth=1.5,
            label="Maximum of three errors",
        )
        axes[3].axhline(0.5, color="#ed807c", linestyle=":", label="Tolerance = 0.5")
        axes[3].set(
            ylabel="Error / own training std",
            xlabel="Time since last observation",
            ylim=(0, 4),
        )
        axes[3].legend(fontsize=8, ncol=3)
        for axis in axes:
            axis.set_xlim(start, end)
            axis.grid(alpha=0.2)
        figure.suptitle(f"{out.name}: {model} crossing — value, slope and curvature")
        figure.text(
            0.03,
            0.01,
            "Causal quadratic estimates; same observed context. "
            "Green requires all three errors to pass.\n"
            "Each feature uses its training scale. "
            "Local agreement does not prove lasting trajectory tracking.",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.05, 1, 0.96))
        target = out / "figures" / f"{slug}-crossing-detail"
        for extension in ["png", "pdf"]:
            figure.savefig(target.with_suffix(f".{extension}"), dpi=150)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--model", default="Ridge")
    parser.add_argument("--start", type=float, default=5.0)
    parser.add_argument("--end", type=float, default=7.0)
    parser.add_argument("--state", type=int, default=0)
    args = parser.parse_args()
    plot_detail(args.results, args.model, args.start, args.end, args.state)
