"""Overlay saved forecasts and truth with variable-specific coherence limits."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from improved_evaluation import safe_correlation
from metrics import coherence_length

COLORS = ["#70d6c5", "#c9be90", "#b8a4ef", "#ff8778", "#75b9ed", "#f7c55f"]


def diagnostics(truth, prediction, state):
    """Match the benchmark and distinguish no failure from failure at last lead."""
    actual, predicted = truth[:, state], prediction[:, state]
    scale = max(float(np.std(actual)), np.finfo(float).eps)
    leads = np.arange(2, len(actual) + 1)
    correlation = np.asarray(
        [safe_correlation(actual[:n], predicted[:n]) for n in leads]
    )
    nrmse = np.asarray(
        [np.sqrt(np.mean((actual[:n] - predicted[:n]) ** 2)) / scale for n in leads]
    )
    failed = np.flatnonzero((correlation < 0.8) | (nrmse > 0.5))
    failure = int(leads[failed[0]]) if len(failed) else None
    length = failure if failure is not None else len(actual)
    assert length == coherence_length(truth, prediction, state=state)
    return length, failure, leads, correlation, nrmse


def plot_coherence(results, origin_index=0):
    """Generate PNG/PDF overlays directly from recorded numerical forecasts."""
    results = Path(results)
    config = json.loads((results / "config.json").read_text())
    report = json.loads((results / "metrics.json").read_text())
    origins = json.loads((results / "origins.json").read_text())["test"]
    if not 0 <= origin_index < len(origins):
        raise ValueError(f"origin-index must lie between 0 and {len(origins) - 1}")
    data = np.load(results / "data.npz")
    truth = data["truth"][origin_index]
    origin, dt = origins[origin_index], config["dt"]
    history = data["states"][origin - 20 : origin]
    lead = np.arange(1, len(truth) + 1) * dt
    past = np.arange(-len(history) + 1, 1) * dt
    names = list(report["models"])
    predictions, info = {}, {}
    for name in names:
        slug = name.lower().replace(" ", "-").replace("(", "").replace(")", "")
        predictions[name] = np.load(results / f"{slug}.npz")["forecast"][origin_index]
        info[name] = [diagnostics(truth, predictions[name], state) for state in [0, 1]]
    out = results / "figures" / f"coherence-origin-{origin_index:02d}"
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    labels = ["Position x", "Velocity v"]
    title = (
        f"Actual vs predicted | test origin {origin_index} "
        f"(first forecast t={data['time'][origin]:.2f})"
    )
    note = (
        "Coherence ends at the first lead where prefix correlation < 0.8 or "
        "prefix RMSE / full-horizon truth std > 0.5.\n"
        "Limits are computed separately for x and v; the benchmark uses x. "
        "Gray: observed context. Red: after first coherence failure."
    )

    def save(figure, filename):
        figure.savefig(
            out / f"{filename}.png", dpi=160, facecolor=figure.get_facecolor()
        )
        figure.savefig(out / f"{filename}.pdf", facecolor=figure.get_facecolor())
        plt.close(figure)

    def decorate(axis, state):
        axis.axvspan(past[0], 0, color="white", alpha=0.06)
        axis.axvline(0, color="white", alpha=0.4, linewidth=0.8)
        axis.set(
            xlabel="Time since last observation",
            ylabel=labels[state],
            xlim=(past[0], lead[-1]),
        )
        axis.grid(alpha=0.16)

    def overlay(axis, name, state, color):
        length, failure, *_ = info[name][state]
        limit = length * dt
        axis.plot(
            np.r_[past, lead],
            np.r_[history[:, state], truth[:, state]],
            color="#f4f4f4",
            linewidth=2.0,
            label="Actual simulated truth",
        )
        axis.plot(
            np.r_[0, lead],
            np.r_[history[-1, state], predictions[name][:, state]],
            color=color,
            linewidth=1.6,
            label="Predicted",
        )
        if failure is not None:
            axis.axvspan(limit, lead[-1], color="#ff6b6b", alpha=0.08)
            axis.axvline(limit, color=color, linestyle="--", linewidth=1.2)
            caption = f"L = {length} steps = {limit:.2f} time units"
        else:
            caption = f"L >= {length} steps ({limit:.2f}); no failure observed"
        axis.set_title(f"{name} | {caption}", fontsize=10, loc="left")
        decorate(axis, state)
        axis.legend(fontsize=8, loc="upper right")

    figure, axes = plt.subplots(len(names), 2, figsize=(16, 3 * len(names)))
    for row, (name, color) in enumerate(zip(names, COLORS, strict=True)):
        for state in [0, 1]:
            overlay(axes[row, state], name, state, color)
        single, panels = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for state in [0, 1]:
            overlay(panels[state], name, state, color)
        single.suptitle(title, fontsize=12)
        single.text(0.05, 0.01, note, fontsize=8, va="bottom")
        single.tight_layout(rect=(0, 0.09, 1, 0.95))
        slug = name.lower().replace(" ", "-").replace("(", "").replace(")", "")
        save(single, slug)
    figure.suptitle(title, fontsize=16)
    figure.text(0.04, 0.01, note, fontsize=10, va="bottom")
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    save(figure, "all-models")

    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for state, axis in enumerate(axes):
        axis.plot(
            np.r_[past, lead],
            np.r_[history[:, state], truth[:, state]],
            color="white",
            linewidth=2.8,
            label="Actual simulated truth",
        )
        for name, color in zip(names, COLORS, strict=True):
            length, failure, *_ = info[name][state]
            qualifier = "=" if failure is not None else ">="
            axis.plot(
                lead,
                predictions[name][:, state],
                color=color,
                alpha=0.9,
                label=f"{name} | L {qualifier} {length * dt:.2f}",
            )
            if failure is not None:
                axis.axvline(length * dt, color=color, linestyle=":", alpha=0.65)
        decorate(axis, state)
        axis.legend(fontsize=8, ncol=2, loc="lower left")
    figure.suptitle(title, fontsize=14)
    figure.text(0.04, 0.01, note, fontsize=9, va="bottom")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(figure, "superposed-variables")

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for state in [0, 1]:
        for name, color in zip(names, COLORS, strict=True):
            _, _, leads, correlation, nrmse = info[name][state]
            axes[0, state].plot(leads * dt, correlation, color=color, label=name)
            axes[1, state].plot(leads * dt, nrmse, color=color, label=name)
        axes[0, state].axhline(0.8, color="white", linestyle="--")
        axes[1, state].axhline(0.5, color="white", linestyle="--")
        axes[0, state].set(
            title=labels[state], ylabel="Prefix correlation", ylim=(-1.05, 1.05)
        )
        axes[1, state].set(
            xlabel="Forecast lead (time units)", ylabel="Prefix normalized RMSE"
        )
        for axis in axes[:, state]:
            axis.grid(alpha=0.16)
        axes[1, state].legend(fontsize=7)
    figure.suptitle("Why coherence ends | " + title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save(figure, "coherence-thresholds")

    summary = {
        "test_origin_index": origin_index,
        "sample": origin,
        "first_forecast_time": float(data["time"][origin]),
        "dt": dt,
        "models": {},
    }
    for name in names:
        summary["models"][name] = {}
        for state, key in enumerate(["x", "v"]):
            length, failure, *_ = info[name][state]
            summary["models"][name][key] = {
                "coherence_steps": length,
                "coherence_time": length * dt,
                "failure_observed": failure is not None,
            }
    (out / "coherence.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved PNG/PDF figures to {out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--origin-index", type=int, default=0)
    args = parser.parse_args()
    plot_coherence(args.results, args.origin_index)
