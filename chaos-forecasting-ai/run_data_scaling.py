"""Measure whether more independent simulated training trajectories help."""

import argparse
import json
import platform
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import generate_series
from improved_evaluation import calibrated_diagnostics
from metrics import coherence_length
from models import RidgeForecaster
from plot_coherence import plot_coherence
from split_integrity import audit_initial_conditions
from torch_models import select_device
from trajectory_training import fit_trajectories

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_examples(trajectories, origins, horizon=180, window=24):
    contexts, truth = [], []
    for trajectory in trajectories:
        for origin in origins:
            contexts.append(trajectory[origin - window : origin])
            truth.append(trajectory[origin : origin + horizon])
    return np.asarray(contexts), np.asarray(truth)


def evaluate(calibration, cal_truth, forecast, truth, scale):
    metric, detail = calibrated_diagnostics(
        calibration, cal_truth, forecast, truth, scale
    )
    errors = forecast[:, :, 0] - truth[:, :, 0]
    metric["first_origin_position_rmse"] = float(np.sqrt(np.mean(errors[0] ** 2)))
    coherence = [coherence_length(t, p) for t, p in zip(truth, forecast, strict=True)]
    metric["coherence_first_origin"] = coherence[0]
    metric["mean_coherence_steps"] = float(np.mean(coherence))
    return metric, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/data-scaling-2026-11-05"
    )
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1:
        parser.error("updates and batch-size must be positive")
    out = args.output_dir
    if (out / "metrics.json").exists():
        parser.error("Choose a new results directory to preserve prior measurements")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    device = select_device(args.device)
    rng = np.random.default_rng(20261105)
    initials = rng.uniform(-2, 2, (44, 2))
    groups = {
        "train": initials[:32],
        "validation": initials[32:36],
        "calibration": initials[36:40],
        "fresh_test": initials[40:44],
    }
    save_json(out / "split-audit.json", audit_initial_conditions(groups, [0.1, 0.0]))
    config = {
        "dt": 0.05,
        "horizon": 180,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "data_seed": 20261105,
        "model_seeds": args.seeds,
        "trajectory_counts": [1, 8, 32],
        "updates_per_seed": args.updates,
        "batch_size": args.batch_size,
        "rollout_training_steps": 4,
        "validation_horizon": 180,
        "validation_check_every_updates": 200,
        "normalization": "training trajectories only",
        "joint_score_scale": "standard deviation of the first training trajectory",
        "optimizer": {
            "name": "AdamW",
            "lr": 8e-4,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "gradient_clip": 1,
        },
        "initial_conditions": {k: v.tolist() for k, v in groups.items()},
        "forcing": "same known external forcing and system parameters in every group",
        "note": "Nested training trajectory subsets; equal optimizer-update budgets. "
        "All groups have disjoint initial conditions. Historical test data "
        "was inspected earlier; fresh episodes are additional holdouts. "
        "Windows within trajectories overlap and are temporally dependent.",
    }
    save_json(out / "config.json", config)
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.glob("*.py"):
            archive.write(path, path.name)
        archive.write(ROOT.parent / "pyproject.toml", "pyproject.toml")
        for path in (ROOT.parent / "src").rglob("*.py"):
            archive.write(path, str(path.relative_to(ROOT.parent)))
    series = {}
    for group, initial_states in groups.items():
        trajectories = []
        for i, initial in enumerate(initial_states):
            duration = 180 if group == "fresh_test" else 100
            _, states = generate_series(duration=duration, initial_state=tuple(initial))
            trajectories.append(states)
            if (i + 1) % 8 == 0 or i + 1 == len(initial_states):
                print(f"Simulated {group}: {i + 1}/{len(initial_states)}", flush=True)
        series[group] = np.asarray(trajectories)
    np.savez_compressed(out / "simulation-pool.npz", **series)
    val_contexts, val_truth = make_examples(
        series["validation"], [424, 824, 1224, 1624]
    )
    cal_contexts, cal_truth = make_examples(
        series["calibration"], range(424, 1821, 100)
    )
    fresh_contexts, fresh_truth = make_examples(
        series["fresh_test"], range(424, 3421, 200)
    )
    time, original = generate_series()
    historical_origins = list(range(2620, 3421, 40))
    historical_contexts, historical_truth = make_examples(
        [original], historical_origins
    )
    contexts = np.concatenate((cal_contexts, historical_contexts, fresh_contexts))
    ncal, nhistorical = len(cal_truth), len(historical_truth)
    save_json(
        out / "origins.json",
        {
            "test": historical_origins,
            "calibration": {
                "trajectory_origins": list(range(424, 1821, 100)),
                "trajectory_count": 4,
            },
            "fresh_test": {
                "trajectory_origins": list(range(424, 3421, 200)),
                "trajectory_count": 4,
            },
        },
    )
    np.savez_compressed(
        out / "data.npz",
        time=time,
        states=original,
        truth=historical_truth,
        calibration_truth=cal_truth,
        fresh_truth=fresh_truth,
    )
    training, fitted, ridge_trials = {}, {}, {}
    # Select every checkpoint using validation before evaluating either test set.
    for count in config["trajectory_counts"]:
        train = series["train"][:count]
        ridge_trials[str(count)] = []
        for regularization in [1e-6, 1e-4, 1e-2, 1.0]:
            candidate = RidgeForecaster(regularization=regularization).fit_trajectories(
                train
            )
            predicted = np.asarray([candidate.rollout(c, 180) for c in val_contexts])
            score = float(np.mean((predicted[:, :, :2] - val_truth[:, :, :2]) ** 2))
            ridge_trials[str(count)].append(
                {"regularization": regularization, "validation_mse": score}
            )
        best = min(ridge_trials[str(count)], key=lambda r: r["validation_mse"])
        ridge = RidgeForecaster(regularization=best["regularization"]).fit_trajectories(
            train
        )
        np.save(out / f"checkpoints/ridge-{count}.npy", ridge.weights)
        fitted[
            f"Ridge ({count} " + ("trajectory)" if count == 1 else "trajectories)")
        ] = ridge
        members = []
        training[str(count)] = []
        print(
            f"Training {count} trajectories: "
            f"{train.shape[0] * train.shape[1]} samples.",
            flush=True,
        )
        for seed in args.seeds:
            model, history = fit_trajectories(
                train,
                val_contexts,
                val_truth,
                seed=seed,
                updates=args.updates,
                batch_size=args.batch_size,
                device=str(device),
            )
            members.append(model)
            training[str(count)].append(history)
            torch.save(
                model.model.state_dict(),
                out / f"checkpoints/transformer-{count}-{seed}.pt",
            )
            save_json(out / "training.json", training)
        fitted[
            f"Transformer ({count} "
            + ("trajectory)" if count == 1 else "trajectories)")
        ] = members
    print(
        "Fitting complete. Evaluating frozen models on test trajectories.",
        flush=True,
    )
    historical, fresh, seed_scores = {}, {}, {}
    for name, model in fitted.items():
        if isinstance(model, list):
            predictions = np.asarray(
                [member.rollout_batch(contexts, 180) for member in model]
            )
            prediction = predictions.mean(0)
            seed_scores[name] = {
                "historical_position_rmse": np.sqrt(
                    np.mean(
                        (
                            predictions[:, ncal : ncal + nhistorical, :, 0]
                            - historical_truth[None, :, :, 0]
                        )
                        ** 2,
                        axis=(1, 2),
                    )
                ).tolist(),
                "fresh_position_rmse": np.sqrt(
                    np.mean(
                        (
                            predictions[:, ncal + nhistorical :, :, 0]
                            - fresh_truth[None, :, :, 0]
                        )
                        ** 2,
                        axis=(1, 2),
                    )
                ).tolist(),
            }
        else:
            prediction = np.asarray([model.rollout(c, 180) for c in contexts])
        calibration = prediction[:ncal]
        forecast = prediction[ncal : ncal + nhistorical]
        transferred = prediction[ncal + nhistorical :]
        historical[name], detail = evaluate(
            calibration,
            cal_truth,
            forecast,
            historical_truth,
            series["train"][0].std(0),
        )
        fresh[name], fresh_detail = evaluate(
            calibration, cal_truth, transferred, fresh_truth, series["train"][0].std(0)
        )
        slug = name.lower().replace(" ", "-").replace("(", "").replace(")", "")
        np.savez_compressed(
            out / f"{slug}.npz", calibration=calibration, forecast=forecast, **detail
        )
        np.savez_compressed(
            out / f"{slug}-fresh.npz", forecast=transferred, **fresh_detail
        )
        if isinstance(model, list):
            np.savez_compressed(out / f"{slug}-members.npz", predictions=predictions)
    report = {
        "models": historical,
        "fresh_trajectories": fresh,
        "seed_scores": seed_scores,
        "ridge_validation": ridge_trials,
        "historical_origin_count": nhistorical,
        "fresh_origin_count": len(fresh_truth),
        "calibration_origin_count": ncal,
        "reference_previous_improved_run": {
            "transformer_historical_rmse": 1.3407902404962588,
            "ridge_historical_rmse": 1.040196889456642,
            "note": "Different training/validation protocol; reference only.",
        },
    }
    save_json(out / "metrics.json", report)
    plot_learning_curve(out, report)
    plot_coherence(out)
    print(
        json.dumps(
            {
                "historical_position_rmse": {
                    k: v["position_rmse"] for k, v in historical.items()
                },
                "fresh_position_rmse": {
                    k: v["position_rmse"] for k, v in fresh.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


def plot_learning_curve(out, report):
    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, key, title in zip(
        axes,
        ["models", "fresh_trajectories"],
        ["Original test windows", "Four new test trajectories"],
        strict=True,
    ):
        for family, color in [("Ridge", "#75b9ed"), ("Transformer", "#f7c55f")]:
            sizes, scores = [], []
            for count in [1, 8, 32]:
                name = f"{family} ({count} " + (
                    "trajectory)" if count == 1 else "trajectories)"
                )
                sizes.append(count * 2001)
                scores.append(report[key][name]["position_rmse"])
            axis.plot(sizes, scores, "-o", label=family, color=color)
        axis.set(
            title=title, xlabel="Training samples", ylabel="Position RMSE", xscale="log"
        )
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("More simulated data | fixed architecture and update budget")
    figure.tight_layout()
    figure.savefig(out / "figures/data-scaling.png", dpi=160)
    figure.savefig(out / "figures/data-scaling.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
