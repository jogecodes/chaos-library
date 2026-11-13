"""Recompute coherence and plots from saved forecasts, without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
from long_horizon_metrics import evaluate_long_run, validation_coherence_settings
from plot_long_horizon import plot_system


def remeasure(root):
    summary = {}
    verification = {}
    for system in ["van-der-pol", "lorenz", "rossler"]:
        out = root / system
        config = json.loads((out / "config.json").read_text())
        old_metrics = json.loads((out / "metrics.json").read_text())
        with np.load(out / "data.npz") as data:
            d, dt = config["physical_size"], config["dt"]
            scale = data["train"].reshape(-1, data["train"].shape[-1]).std(0)
            settings = validation_coherence_settings(
                data["validation"],
                d,
                dt,
                training=data["train"],
            )
            (out / "coherence-settings.json").write_text(
                json.dumps(settings, indent=2) + "\n"
            )
            metrics = {}
            for name, old in old_metrics.items():
                if name in {"Persistence", "Ridge"}:
                    continue
                path = out / (name.lower().replace(" ", "-") + ".npz")
                with np.load(path) as saved:
                    members = saved["members"].copy()
                    mean = saved["mean"].copy()
                cutoffs = [float(c) for c in old["point_by_cutoff"]]
                result, arrays = evaluate_long_run(
                    data["truth"],
                    members,
                    scale,
                    d,
                    dt,
                    cutoffs,
                    settings,
                    contexts=data["test_contexts"],
                )
                assert np.array_equal(members, arrays["members"], equal_nan=True)
                assert np.array_equal(mean, arrays["mean"], equal_nan=True)
                for cutoff, score in old["point_by_cutoff"].items():
                    assert score == result["point_by_cutoff"][str(float(cutoff))]
                assert (
                    old["individual_member_climate"]
                    == result["individual_member_climate"]
                )
                np.savez_compressed(path, **arrays)
                metrics[name] = result
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        summary[system] = metrics
        verification[system] = {
            "forecasts_unchanged": True,
            "point_errors_unchanged": True,
            "climate_scores_unchanged": True,
            "settings": settings,
        }
        print(system, settings, flush=True)
        for name, result in metrics.items():
            print(
                name,
                "reference",
                result["coherence_time_by_origin_state"][0],
                "joint median",
                result["coherence_joint_median_capped"],
                flush=True,
            )
        plot_system(out)
        print(system, "replotted", flush=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "coherence-verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    remeasure(parser.parse_args().results)
