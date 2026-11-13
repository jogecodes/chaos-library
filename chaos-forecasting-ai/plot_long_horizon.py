"""Truth overlays, coherence retention, and long-run single-member diagnostics."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#ff8b78", "#f6ca67", "#b9a0f0", "#9fdb88"]


def local_accuracy_strip(axis, lead, within, episode_status=None):
    """Show current accuracy, including recovery after an earlier breach."""
    for condition, color in [(within, "#76d2a0"), (~within, "#ed807c")]:
        axis.fill_between(
            lead,
            0,
            0.035,
            where=condition,
            step="mid",
            color=color,
            alpha=0.8,
            transform=axis.get_xaxis_transform(),
            linewidth=0,
        )

    if episode_status is not None:
        for code, color in enumerate(["#76d2a0", "#f6ca67", "#ed807c"]):
            axis.fill_between(
                lead,
                0.04,
                0.085,
                where=episode_status == code,
                step="mid",
                color=color,
                alpha=0.9,
                transform=axis.get_xaxis_transform(),
                linewidth=0,
            )


def reference_breakdown(metric, state=None):
    """Prefer confirmed joint episode onset; retain raw first-breach fallback."""
    if "episode_coherence" in metric:
        report = metric["episode_coherence"]
        onset = report["onset_time_by_origin"][0]
        return (report["horizon_time"] if onset is None else onset), onset is None
    times = metric["coherence_time_by_origin_state"][0]
    if state is None:
        return max(times), any(metric["coherence_censored_by_origin_state"][0])
    return times[state], metric["coherence_censored_by_origin_state"][0][state]


def feature_error_lines(axis, lead, saved, state):
    """Expose which aspect of the local trajectory disagrees with truth."""
    styles = [("Value", "#80b6ed"), ("Slope", "#f6ca67"), ("Curvature", "#b9a0f0")]
    components = saved["feature_nrmse_by_origin_state"][0, :, state]
    for i, (label, color) in enumerate(styles[: components.shape[-1]]):
        axis.plot(
            lead,
            components[:, i],
            color=color,
            linewidth=0.9,
            alpha=0.85,
            linestyle="--",
            label=label,
        )


def autocorrelation(values, maximum_lag):
    """Pool within-trajectory lag pairs without crossing trajectory boundaries."""
    centered = values - values.mean(axis=1, keepdims=True)
    result = []
    for lag in range(maximum_lag + 1):
        a = centered if lag == 0 else centered[:, :-lag]
        b = centered if lag == 0 else centered[:, lag:]
        denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
        result.append(float(np.sum(a * b) / denominator) if denominator > 0 else 0)
    return np.asarray(result)


def comparison_window(metrics, horizon, dt, full_horizon=False):
    """Use one limit across models and variables at the displayed reference origin."""
    longest = max(reference_breakdown(model)[0] for model in metrics.values())
    if full_horizon or longest >= horizon:
        return float(horizon)
    context = 4 * max(
        m.get("episode_coherence", {}).get("policy", {}).get("window_time", 0)
        for m in metrics.values()
    )
    padded = longest + max(longest, context, 5 * dt)
    tick = 10 ** np.floor(np.log10(max(longest, dt))) / 10
    return float(min(horizon, np.ceil(padded / tick) * tick))


def plot_system(out, full_horizon=False):
    out = Path(out)
    config = json.loads((out / "config.json").read_text())
    metrics = {
        name: value
        for name, value in json.loads((out / "metrics.json").read_text()).items()
        if name not in {"Persistence", "Ridge"}
    }
    data = np.load(out / "data.npz")
    actual = data["truth"]
    dt, d = config["dt"], config["physical_size"]
    scale = data["train"].reshape(-1, data["train"].shape[-1]).std(0)[:d]
    lead = np.arange(1, actual.shape[1] + 1) * dt
    maximum = comparison_window(metrics, lead[-1], dt, full_horizon)
    names = list(metrics)
    (out / "plot-window.json").write_text(
        json.dumps(
            {
                "mode": "full-horizon" if full_horizon else "coherence",
                "maximum_time": maximum,
                "reference_origin_index": 0,
                "longest_reference_coherence": max(
                    reference_breakdown(model)[0] for model in metrics.values()
                ),
                "padding_fraction": 1.0,
                "minimum_post_breakdown_confirmation_windows": 4,
                "window_rule": (
                    "Longest reference episode onset plus at least its own duration; "
                    "all systems get at least four confirmation windows after it. "
                    "Rounded up and clipped to saved forecast horizon."
                ),
                "available_horizon": float(lead[-1]),
                "shared_across": "Time-series model panels and physical variables",
            },
            indent=2,
        )
        + "\n"
    )
    forecasts = {
        name: np.load(out / (name.lower().replace(" ", "-") + ".npz")) for name in names
    }
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.style.use("dark_background")
    system = out.name

    def save(figure, name):
        figure.savefig(figures / f"{name}.png", dpi=150)
        figure.savefig(figures / f"{name}.pdf")
        plt.close(figure)

    def overlays(maximum, filename):
        figure, axes = plt.subplots(
            len(names), d, figsize=(6 * d, 2.7 * len(names)), squeeze=False
        )
        for row, (name, color) in enumerate(zip(names, COLORS, strict=True)):
            saved = forecasts[name]
            for state in range(d):
                axis = axes[row, state]
                axis.plot(
                    lead,
                    actual[0, :, state],
                    color="white",
                    linewidth=1.4,
                    label="Truth",
                )
                for member in saved["members"]:
                    axis.plot(
                        lead,
                        member[0, :, state],
                        color=color,
                        alpha=0.23,
                        linewidth=0.7,
                    )
                axis.plot(
                    lead,
                    saved["mean"][0, :, state],
                    color=color,
                    linewidth=1,
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
                coherence, censored = reference_breakdown(metrics[name], state)
                if not censored:
                    axis.axvline(coherence, color=color, linestyle="--", linewidth=0.8)
                qualifier = ">=" if censored else "="
                axis.set(
                    title=(
                        f"{name} | {config['labels'][state]} | "
                        f"Joint breakdown {qualifier} {coherence:.2f}"
                    ),
                    xlabel="Time since last observation",
                    ylabel=config["labels"][state],
                    xlim=(0, maximum),
                )
                low, high = np.min(actual[:, :, state]), np.max(actual[:, :, state])
                margin = 0.25 * (high - low)
                axis.set_ylim(low - margin, high + margin)
                selection = lead <= maximum
                values = saved["mean"][0, selection, state]
                offscale = (
                    (~np.isfinite(values))
                    | (values < low - margin)
                    | (values > high + margin)
                )
                if offscale.any():
                    axis.text(
                        0.02,
                        0.87,
                        f"{offscale.mean():.0%} off-scale/invalid",
                        transform=axis.transAxes,
                        fontsize=8,
                        color="#ff8b78",
                    )
                axis.legend(fontsize=7, loc="upper right")
                axis.grid(alpha=0.15)
        figure.suptitle(
            f"{system}: free forecast vs truth | reference origin", fontsize=15
        )
        figure.text(
            0.03,
            0.01,
            "Thin colored lines: individual neural members; bold: mean. "
            "Marker: retrospective joint sustained-breakdown onset.\n"
            "Upper strip: green tracking, amber mismatch, red sustained loss. "
            "No truth updates after the initial context. Axes show truth range ±25%; "
            "off-scale and invalid values remain counted in metrics.",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.045, 1, 0.97))
        save(figure, filename)

    overlays(maximum, "long-overlays")
    # Preserve the earlier export path with the same shared display window.
    for extension in ["png", "pdf"]:
        shutil.copyfile(
            figures / f"long-overlays.{extension}",
            figures / f"early-overlays.{extension}",
        )
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, color in zip(names, COLORS, strict=True):
        values = forecasts[name]
        # Physical-state point RMSE normalized by training scales, across test origins.
        error = (values["mean"][:, :, :d] - actual[:, :, :d]) / scale
        rmse = np.sqrt(np.mean(error**2, axis=(0, 2)))
        axes[0].plot(lead, rmse, label=name, color=color)
        axes[1].plot(lead, values["joint_coherence_survival"], label=name, color=color)
    axes[0].set(ylabel="Normalized point RMSE")
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[1].set(
        ylabel="Fraction with no joint accuracy breach yet",
        xlabel="Forecast lead (time units)",
        ylim=(-0.02, 1.02),
    )
    for axis in axes:
        axis.set_xlim(0, maximum)
        axis.legend(fontsize=8, ncol=3)
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{system}: accuracy and first breaches across nine dependent origins"
    )
    figure.tight_layout()
    save(figure, "error-and-coherence")
    settings = metrics[names[0]]["coherence_settings"]
    figure, axes = plt.subplots(
        len(names), d, figsize=(6 * d, 2.5 * len(names)), squeeze=False
    )
    for row, (name, color) in enumerate(zip(names, COLORS, strict=True)):
        for state in range(d):
            axis = axes[row, state]
            error = forecasts[name]["rolling_nrmse_by_origin_state"][0, :, state]
            axis.plot(
                lead,
                error,
                color=color,
                linewidth=1,
                label="Maximum value/slope/curvature error",
            )
            feature_error_lines(axis, lead, forecasts[name], state)
            axis.axhline(
                settings["threshold"],
                color="white",
                linestyle=":",
                label="Loss threshold = 0.5",
            )
            lost = metrics[name]["coherence_time_by_origin_state"][0][state]
            censored = metrics[name]["coherence_censored_by_origin_state"][0][state]
            if not censored:
                axis.axvline(
                    lost,
                    color=color,
                    linestyle="--",
                    label=f"First confirmed breach {lost:.2f}",
                )
            axis.set(
                xlim=(0, maximum),
                ylim=(0, 2),
                xlabel="Time since last observation",
                ylabel="Rolling normalized RMS error",
                title=f"{name} | {config['labels'][state]}",
            )
            axis.grid(alpha=0.15)
            axis.legend(fontsize=7)
    figure.suptitle(f"{system}: check the coherence threshold directly", fontsize=15)
    figure.text(
        0.03,
        0.01,
        f"Trailing window {settings['window_steps']} samples "
        f"({settings['window_steps'] * dt:g} time units); "
        f"require {settings['persistence_steps']} consecutive exceedances. "
        "Dashed marker: first confirmed breach.\n"
        "Each error uses its training scale; all three must pass for recovery. "
        "Errors >2 clipped for display. Same origin as overlays.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    save(figure, "coherence-threshold")

    def review_model(name, filename):
        figure, axes = plt.subplots(
            2, d, figsize=(6 * d, 7), squeeze=False, sharex=True
        )
        values = forecasts[name]
        model_metrics = metrics[name]
        for state in range(d):
            top, bottom = axes[:, state]
            top.plot(lead, actual[0, :, state], color="white", label="Simulated truth")
            top.plot(
                lead,
                values["mean"][0, :, state],
                color="#ff8b78",
                label=f"{name} forecast",
            )
            local_accuracy_strip(
                top,
                lead,
                values["within_tolerance_by_origin_state"][0, :, state],
                values["episode_status_by_origin"][0]
                if "episode_status_by_origin" in values
                else None,
            )
            top.set(
                title=f"{config['labels'][state]}: forecast and local accuracy",
                ylabel=config["labels"][state],
            )
            error = values["rolling_nrmse_by_origin_state"][0, :, state]
            bottom.plot(lead, error, color="#ff8b78", label="Maximum of three errors")
            feature_error_lines(bottom, lead, values, state)
            bottom.axhline(
                settings["threshold"],
                color="white",
                linestyle=":",
                label="Tolerance = 0.5 training std",
            )
            lost = model_metrics["coherence_time_by_origin_state"][0][state]
            censored = model_metrics["coherence_censored_by_origin_state"][0][state]
            recovered = model_metrics["first_breach_recovery_time_by_origin_state"][0][
                state
            ]
            for axis in [top, bottom]:
                if not censored:
                    axis.axvline(
                        lost,
                        color="#ff8b78",
                        linestyle="--",
                        label=f"First confirmed breach {lost:.2f}",
                    )
                axis.set_xlim(0, maximum)
                axis.grid(alpha=0.15)
            if recovered is not None and recovered <= maximum:
                bottom.scatter(
                    [recovered],
                    [error[int(round(recovered / dt)) - 1]],
                    color="#76d2a0",
                    s=35,
                    label=f"Back within tolerance {recovered:.2f}",
                    zorder=5,
                )
            bottom.set(
                ylim=(0, 2),
                ylabel="Error / training std",
                xlabel="Time since last observation",
            )
            top.legend(fontsize=8)
            bottom.legend(fontsize=8)
        figure.suptitle(f"{system}: {name} value, slope, curvature, and recovery")
        figure.text(
            0.03,
            0.01,
            "Green: value, slope and curvature all pass; red: at least one fails. "
            "Dashed curves show each component.\n"
            "Tolerance 0.5 per training scale; validation window and confirmation. "
            "Error axes clipped at 2 for display.",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.95))
        save(figure, filename)

    review_model("Polynomial dynamics", "polynomial-accuracy-review")
    figure, axis = plt.subplots(figsize=(10, 5))
    for name, color in zip(names, COLORS, strict=True):
        sensitivity = metrics[name].get("coherence_threshold_sensitivity", {})
        thresholds = sorted(float(t) for t in sensitivity)
        medians = [sensitivity[str(t)]["joint_median_capped"] for t in thresholds]
        axis.plot(thresholds, medians, "o-", color=color, label=name)
    axis.set(
        xlabel="Rolling error threshold / training std",
        ylabel="Median first joint accuracy breach (time units)",
        title=f"{system}: dependence on the declared accuracy tolerance",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    save(figure, "coherence-sensitivity")
    tail = actual.shape[1] // 10
    truth_tail = actual[:, tail:, :d]
    phase_y = 1 if d == 2 else 2
    figure, axes = plt.subplots(len(names), 3, figsize=(16, 3 * len(names)))
    max_lag = min(int(20 / dt), truth_tail.shape[1] // 4)
    lags = np.arange(max_lag + 1) * dt
    actual_acf = autocorrelation(truth_tail[:, :, 0], max_lag)
    flattened = truth_tail.reshape(-1, d)
    limits = [
        (flattened[:, i].min() - 0.1 * scale[i], flattened[:, i].max() + 0.1 * scale[i])
        for i in range(d)
    ]
    bins = np.linspace(*limits[0], 61)
    actual_counts, _ = np.histogram(flattened[:, 0], bins=bins)
    for row, (name, color) in enumerate(zip(names, COLORS, strict=True)):
        members = forecasts[name]["members"][:, :, tail:, :d]
        # Phase portraits show a single free trajectory, never the ensemble average.
        first = members[0, 0]
        if np.isfinite(first[-1]).all():
            axes[row, 0].scatter(
                first[-1, 0], first[-1, phase_y], color=color, s=18, zorder=5
            )
        axes[row, 0].plot(
            truth_tail[0, :, 0],
            truth_tail[0, :, phase_y],
            color="white",
            alpha=0.55,
            linewidth=0.7,
            label="Truth",
        )
        axes[row, 0].plot(
            first[:, 0],
            first[:, phase_y],
            color=color,
            alpha=0.8,
            linewidth=0.7,
            label="One member",
        )
        axes[row, 0].set(
            xlim=limits[0],
            ylim=limits[phase_y],
            xlabel=config["labels"][0],
            ylabel=config["labels"][phase_y],
            title=name + " | state projection",
        )
        axes[row, 1].stairs(
            actual_counts / (len(flattened) * np.diff(bins)),
            bins,
            color="white",
            label="Truth",
        )
        for member in members:
            sample = member.reshape(-1, d)
            finite = np.isfinite(sample).all(axis=1)
            counts, _ = np.histogram(sample[finite, 0], bins=bins)
            axes[row, 1].stairs(
                counts / (len(sample) * np.diff(bins)), bins, color=color, alpha=0.6
            )
            if finite.all():
                acf = autocorrelation(member[:, :, 0], max_lag)
                axes[row, 2].plot(lags, acf, color=color, alpha=0.65)
        axes[row, 2].plot(lags, actual_acf, color="white", label="Truth")
        axes[row, 1].set(
            xlabel=config["labels"][0],
            ylabel="Probability density",
            title="Individual-member distribution",
        )
        axes[row, 2].set(
            xlabel="Lag (time units)",
            ylabel="Autocorrelation",
            ylim=(-1.05, 1.05),
            title="Within-trajectory correlation",
        )
        for axis in axes[row]:
            axis.grid(alpha=0.15)
        axes[row, 0].legend(fontsize=7)
    figure.suptitle(f"{system}: behavior after point forecasts diverge", fontsize=15)
    figure.text(
        0.03,
        0.01,
        "First 10% of each forecast omitted. State projections show the "
        "reference trajectory; densities and correlations pool nine origins within "
        "each member.\nNeural ensemble means are excluded from these distribution "
        "diagnostics. Display limits use truth ranges; no histogram renormalization "
        "after excluding off-scale samples.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    save(figure, "long-run-behavior")
    # A compact comparison is easier to open than the full all-variable grid.
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for axis, name in zip(axes, ["Transformer", "GRU", "MLP"], strict=True):
        saved = forecasts[name]
        axis.plot(lead, actual[0, :, 0], color="white", linewidth=1, label="Truth")
        for i, member in enumerate(saved["members"]):
            axis.plot(
                lead,
                member[0, :, 0],
                linewidth=0.8,
                alpha=0.65,
                label=f"Seed member {i + 1}",
            )
        axis.plot(
            lead,
            saved["mean"][0, :, 0],
            color="gold",
            linewidth=1.3,
            label="Ensemble mean",
        )
        axis.set(title=name, ylabel=config["labels"][0], xlim=(0, maximum))
        axis.grid(alpha=0.15)
        axis.legend(fontsize=7, ncol=5)
    axes[-1].set_xlabel("Forecast lead (time units)")
    figure.suptitle(f"{system}: individual free trajectories and their mean")
    figure.tight_layout()
    save(figure, "neural-members")
    plot_compact(out, full_horizon=full_horizon)
    print(f"{system}: plots saved to {figures}", flush=True)


def plot_compact(out, full_horizon=False):
    """Make a compact four-model view for opening beside the full diagnostics."""
    out = Path(out)
    config = json.loads((out / "config.json").read_text())
    metrics = {
        name: value
        for name, value in json.loads((out / "metrics.json").read_text()).items()
        if name not in {"Persistence", "Ridge"}
    }
    data = np.load(out / "data.npz")
    truth = data["truth"][0, :, 0]
    lead = np.arange(1, len(truth) + 1) * config["dt"]
    maximum = comparison_window(metrics, lead[-1], config["dt"], full_horizon)
    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    for axis, name, color in zip(
        axes.flat,
        ["Polynomial dynamics", "Transformer", "GRU", "MLP"],
        ["#ff8b78", "#f6ca67", "#b9a0f0", "#9fdb88"],
        strict=True,
    ):
        values = np.load(out / (name.lower().replace(" ", "-") + ".npz"))
        axis.plot(lead, truth, color="white", linewidth=1.0, label="Simulated truth")
        if len(values["members"]) > 1:
            for member in values["members"]:
                axis.plot(lead, member[0, :, 0], color=color, alpha=0.2, linewidth=0.6)
        axis.plot(
            lead,
            values["mean"][0, :, 0],
            color=color,
            linewidth=1.1,
            label="Prediction" if len(values["members"]) == 1 else "Ensemble mean",
        )
        local_accuracy_strip(
            axis,
            lead,
            values["within_tolerance_by_origin_state"][0, :, 0],
            values["episode_status_by_origin"][0]
            if "episode_status_by_origin" in values
            else None,
        )
        coherence, censored = reference_breakdown(metrics[name], 0)
        if not censored:
            axis.axvline(coherence, color=color, linestyle="--", linewidth=0.8)
        qualifier = ">=" if censored else "="
        axis.set(
            title=f"{name} | joint sustained breakdown {qualifier} {coherence:.2f}",
            xlabel="Forecast lead (time units)",
            ylabel=config["labels"][0],
            xlim=(0, maximum),
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    figure.suptitle(f"{out.name}: same observed context | shared display 0–{maximum:g}")
    figure.text(
        0.03,
        0.01,
        "Thin lines: individual neural trajectories. Dashed markers: "
        "retrospective joint sustained-breakdown onset.\n"
        "Upper strip: green tracking, amber mismatch, red sustained loss. "
        "No truth updates.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(out / "figures/long-comparison.png", dpi=150)
    figure.savefig(out / "figures/long-comparison.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    tail = len(truth) // 10
    phase_y = 1 if config["physical_size"] == 2 or out.name == "rossler" else 2
    actual = data["truth"][0, tail:]
    for axis, name, color in zip(
        axes.flat[: len(metrics)], metrics, COLORS, strict=True
    ):
        saved = np.load(out / (name.lower().replace(" ", "-") + ".npz"))
        member = saved["members"][0, 0, tail:]
        if np.isfinite(member).all():
            axis.scatter(
                member[-1, 0], member[-1, phase_y], color=color, s=24, zorder=5
            )
            span = np.ptp(member[:, [0, phase_y]], axis=0)
            reference_span = np.ptp(actual[:, [0, phase_y]], axis=0)
            if np.all(span < 0.01 * reference_span):
                axis.annotate(
                    "Nearly stationary",
                    (member[-1, 0], member[-1, phase_y]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
        axis.plot(
            actual[:, 0],
            actual[:, phase_y],
            color="white",
            alpha=0.65,
            linewidth=0.7,
            label="Simulated truth",
        )
        axis.plot(
            member[:, 0],
            member[:, phase_y],
            color=color,
            alpha=0.8,
            linewidth=0.7,
            label="One free trajectory",
        )
        axis.set(
            title=name, xlabel=config["labels"][0], ylabel=config["labels"][phase_y]
        )
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(metrics) :]:
        axis.set_visible(False)
    figure.suptitle(f"{out.name}: long-run state-space projections")
    figure.text(
        0.03,
        0.01,
        "Same reference origin; first 10% of forecast omitted. "
        "Each model panel uses one trajectory, not an ensemble mean. "
        "Projection similarity alone does not establish correct dynamics.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(out / "figures/state-projections.png", dpi=150)
    figure.savefig(out / "figures/state-projections.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--full-horizon",
        action="store_true",
        help="Show the complete forecast instead of the coherence window",
    )
    args = parser.parse_args()
    plot_system(args.results, full_horizon=args.full_horizon)
