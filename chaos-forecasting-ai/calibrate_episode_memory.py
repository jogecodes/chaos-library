"""Calibrate one dimensionless persistence multiplier on the annotated example."""

import argparse
import json
from pathlib import Path

import numpy as np
from coherence_episodes import (
    CONFIRMATION_MEMORY_MULTIPLIER,
    detect_episodes,
    episode_policy,
)


def calibrate(root):
    out = Path(root) / "van-der-pol"
    labels = json.loads((out / "visual-coherence-annotations.json").read_text())[
        "annotations"
    ]
    metrics = json.loads((out / "metrics.json").read_text())
    with np.load(out / "data.npz") as data:
        base = episode_policy(data["validation"], 2, 0.05)
    intervals = {}
    for name in metrics:
        label = next(a for a in labels if a["model"] == name and a["variable"] == "x")
        intervals[name] = label.get("break_time_range") or [
            label["break_time_approx"] - 0.5,
            label["break_time_approx"] + 0.5,
        ]
    trials = []
    for multiplier in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        policy = dict(
            base,
            confirmation_memory_multiplier=multiplier,
            window_steps=max(5, int(np.ceil(multiplier * base["memory_steps"]))),
        )
        results = {}
        for name in metrics:
            with np.load(out / (name.lower().replace(" ", "-") + ".npz")) as saved:
                result, _ = detect_episodes(
                    saved["rolling_nrmse_by_origin_state"][:1], policy
                )
            onset = result["onset_time_by_origin"][0]
            lo, hi = intervals[name]
            error = (
                base["dt"] * 4000 if onset is None else max(lo - onset, onset - hi, 0)
            )
            results[name] = {"onset": onset, "distance_from_label_interval": error}
        trials.append(
            {
                "confirmation_multiplier": multiplier,
                "mean_interval_error": float(
                    np.mean(
                        [m["distance_from_label_interval"] for m in results.values()]
                    )
                ),
                "models": results,
            }
        )
    selected = min(
        trials, key=lambda t: (t["mean_interval_error"], t["confirmation_multiplier"])
    )
    withheld = {}
    for name in metrics:
        choice = min(
            trials,
            key=lambda t: (
                np.mean(
                    [
                        m["distance_from_label_interval"]
                        for model, m in t["models"].items()
                        if model != name
                    ]
                ),
                t["confirmation_multiplier"],
            ),
        )
        withheld[name] = {
            "selected_without_model": choice["confirmation_multiplier"],
            **choice["models"][name],
        }
    report = {
        "inputs": "One annotated trajectory; four models; x/v are dependent.",
        "time_scale": "Median validation ACF 1/e crossing.",
        "memory_time": base["memory_time"],
        "recovery_multiplier_fixed": 1.0,
        "bad_fraction_fixed": 0.8,
        "approximate_label_interval_half_width": 0.5,
        "selection": "Min mean interval error; smallest multiplier breaks ties.",
        "trials": trials,
        "selected_confirmation_multiplier": selected["confirmation_multiplier"],
        "model_withholding_stress_check": withheld,
        "limitations": "Small set; model withholding is not independent validation.",
    }
    (out / "episode-memory-calibration.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if selected["confirmation_multiplier"] != CONFIRMATION_MEMORY_MULTIPLIER:
        raise RuntimeError(
            "Frozen multiplier differs from calibration; inspect before updating"
        )
    print("Selected memory multiplier:", selected["confirmation_multiplier"])
    print("Memory time:", base["memory_time"])
    print("Withheld-model check:", withheld)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    calibrate(parser.parse_args().results)
