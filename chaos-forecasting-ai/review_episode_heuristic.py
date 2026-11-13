"""Compare episode onsets with development labels and unannotated trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from coherence_episodes import detect_episodes
from plot_long_horizon import COLORS, local_accuracy_strip


def review(root, system="van-der-pol"):
    out = Path(root) / system
    metrics = json.loads((out / "metrics.json").read_text())
    label_path = out / "visual-coherence-annotations.json"
    labels = json.loads(label_path.read_text()) if label_path.exists() else None
    config = json.loads((out / "config.json").read_text())
    d = config["physical_size"]
    maximum = json.loads((out / "plot-window.json").read_text())["maximum_time"]
    if system == "van-der-pol":
        maximum = 40.0
    report = {
        "scope": (
            "Four models on one annotated development trajectory. "
            "x/v labels are not independent cases."
        ),
        "unannotated_scope": (
            "Eight dependent origins, four other ICs. "
            "No visual labels; visual accuracy is unmeasured."
        ),
        "selected_rule": "5 memory times; >=80% bad; gaps <=1 memory time.",
        "approximate_label_tolerance": 0.5,
        "models": {},
    }
    forecasts = {
        name: np.load(out / (name.lower().replace(" ", "-") + ".npz"))
        for name in metrics
    }
    for name, m in metrics.items():
        ep = m["episode_coherence"]
        label = (
            next(
                (
                    a
                    for a in labels["annotations"]
                    if a["model"] == name and a["variable"] == "x"
                ),
                None,
            )
            if labels
            else None
        )
        interval = label.get("break_time_range") if label else None
        approximate = label.get("break_time_approx") if label else None
        onset = ep["onset_time_by_origin"][0]
        if interval is not None:
            distance = (
                max(interval[0] - onset, onset - interval[1], 0)
                if onset is not None
                else None
            )
        elif approximate is not None:
            distance = abs(onset - approximate) if onset is not None else None
        else:
            distance = None
        variants = []
        base = ep["policy"]
        for factor in [0.75, 1.0, 1.25]:
            for fraction in [0.7, 0.8, 0.9]:
                for recovery_multiplier in [0.5, 1.0, 1.5]:
                    policy = dict(
                        base,
                        window_steps=max(5, round(base["window_steps"] * factor)),
                        bad_fraction=fraction,
                        max_good_gap_steps=int(
                            recovery_multiplier * base["memory_steps"]
                        ),
                    )
                    # Recovery duration is varied separately from confirmation duration.
                    result, _ = detect_episodes(
                        forecasts[name]["rolling_nrmse_by_origin_state"][:1], policy
                    )
                    variants.append(
                        {
                            "window_factor": factor,
                            "bad_fraction": fraction,
                            "recovery_memory_multiplier": recovery_multiplier,
                            "onset": result["onset_time_by_origin"][0],
                        }
                    )
        values = ep["onset_time_by_origin"][1:]
        horizon = ep["horizon_time"]
        report["models"][name] = {
            "visual_main_break_range": interval,
            "visual_main_break_approx": approximate,
            "heuristic_onset": onset,
            "confirmed_at": ep["confirmation_time_by_origin"][0],
            "distance_from_label": distance,
            "within_development_label": None
            if label is None
            else (distance is not None and distance <= (0 if interval else 0.5)),
            "other_origin_onsets": values,
            "other_origins_median_capped": float(
                np.median([horizon if v is None else v for v in values])
            ),
            "other_origins_censored_fraction": float(
                np.mean([v is None for v in values])
            ),
            "other_initial_conditions_paired_onsets": [
                values[i : i + 2] for i in range(0, 8, 2)
            ],
            "parameter_sensitivity": variants,
        }
    if labels is None:
        report["scope"] = "Unannotated system: visual agreement is unmeasured."
    (out / "episode-review-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if labels is not None:
        (out / "episode-development-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    data = np.load(out / "data.npz")
    dt = config["dt"]
    lead = np.arange(1, data["truth"].shape[1] + 1) * dt
    plt.style.use("dark_background")

    def plot_origin(origin, filename):
        figure, axes = plt.subplots(
            len(metrics),
            d + 1,
            figsize=(5 * (d + 1), 3 * len(metrics)),
            sharex=True,
            squeeze=False,
        )
        for row, (name, color) in enumerate(zip(metrics, COLORS, strict=True)):
            saved = forecasts[name]
            ep = metrics[name]["episode_coherence"]
            onset, confirmation = (
                ep["onset_time_by_origin"][origin],
                ep["confirmation_time_by_origin"][origin],
            )
            for state in range(d):
                axis = axes[row, state]
                axis.plot(
                    lead,
                    data["truth"][origin, :, state],
                    color="white",
                    lw=1,
                    label="Truth",
                )
                axis.plot(
                    lead,
                    saved["mean"][origin, :, state],
                    color=color,
                    lw=1,
                    label="Forecast mean",
                )
                local_accuracy_strip(
                    axis,
                    lead,
                    saved["within_tolerance_by_origin_state"][origin, :, state],
                    saved["episode_status_by_origin"][origin],
                )
                axis.set(
                    ylabel=config["labels"][state],
                    title=f"{name} | {config['labels'][state]}",
                )
            bad = (saved["rolling_nrmse_by_origin_state"][origin] > 0.5).any(-1)
            window = ep["policy"]["window_steps"]
            fraction = np.full(len(lead), np.nan)
            fraction[window - 1 :] = np.convolve(
                bad.astype(float), np.ones(window) / window, mode="valid"
            )
            axes[row, d].plot(
                lead, fraction, color=color, label="Trailing disagreement fraction"
            )
            axes[row, d].axhline(
                0.8, color="white", ls=":", label="80%; gap rule must also pass"
            )
            axes[row, d].set(
                ylim=(-0.03, 1.03), ylabel="Fraction", title="Evidence and confirmation"
            )
            for axis in axes[row]:
                if onset is not None:
                    axis.axvline(
                        onset, color=color, ls="--", label=f"Onset {onset:.2f}"
                    )
                    axis.axvline(
                        confirmation,
                        color="gray",
                        ls=":",
                        label=f"Confirmed {confirmation:.2f}",
                    )
                axis.set_xlim(0, maximum)
                axis.set_xlabel("Time since last observation")
                axis.grid(alpha=0.15)
                axis.legend(fontsize=7, loc="upper right")
        label = (
            ("annotated development example" if labels else "unannotated reference")
            if origin == 0
            else f"unannotated origin {origin} (different test IC)"
        )
        figure.suptitle(f"{system}: {label} | mismatch, recovery and breakdown")
        figure.text(
            0.02,
            0.01,
            "Upper strip: green tracking, amber mismatch, red sustained episode. "
            "Lower: strict local accuracy.\n"
            "Dashed: retrospective onset; dotted: confirmation. "
            "Recovery remains possible. Same rule for all models.",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.045, 1, 0.96))
        for extension in ["png", "pdf"]:
            figure.savefig(out / "figures" / f"{filename}.{extension}", dpi=150)
        plt.close(figure)

    plot_origin(0, "episode-review")
    for origin in [1, 3, 5, 7]:
        plot_origin(origin, f"episode-unannotated-origin-{origin:02d}")
    figure, axes = plt.subplots(1, len(metrics), figsize=(18, 4), sharey=True)
    for axis, (name, color) in zip(
        axes, zip(metrics, COLORS, strict=True), strict=True
    ):
        variants = report["models"][name]["parameter_sensitivity"]
        for factor, style in [(0.75, "o-"), (1.0, "s-"), (1.25, "^-")]:
            subset = [
                v
                for v in variants
                if v["window_factor"] == factor
                and v["recovery_memory_multiplier"] == 1.0
            ]
            axis.plot(
                [v["bad_fraction"] for v in subset],
                [np.nan if v["onset"] is None else v["onset"] for v in subset],
                style,
                label=f"{factor:g} duration",
            )
        axis.set(
            title=name,
            xlabel="Required disagreement fraction",
            ylabel="Onset time",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle(
        f"{system}: sensitivity; recovery gap held at one validation memory time"
    )
    figure.tight_layout()
    for extension in ["png", "pdf"]:
        figure.savefig(out / "figures" / f"episode-sensitivity.{extension}", dpi=150)
    plt.close(figure)
    for saved in forecasts.values():
        saved.close()
    data.close()
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in m.items()
                    if k
                    in [
                        "heuristic_onset",
                        "confirmed_at",
                        "other_origins_median_capped",
                    ]
                }
                for name, m in report["models"].items()
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--system", choices=["all", "van-der-pol", "lorenz", "rossler"], default="all"
    )
    args = parser.parse_args()
    systems = (
        ["van-der-pol", "lorenz", "rossler"] if args.system == "all" else [args.system]
    )
    for system in systems:
        review(args.results, system)
