"""Run a chronological, multi-origin comparison and separate PINN evaluation."""

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import generate_series
from improved_evaluation import calibrated_diagnostics
from improved_models import NeuralForecaster, fit_pinn, fit_transformer
from metrics import coherence_length
from model_comparison import transformer_forecast
from models import PolynomialDynamics, RidgeForecaster
from run_torch_experiments import train_pinn
from torch_models import select_device

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def batch_rollout(model, contexts, horizon):
    if hasattr(model, "rollout_batch"):
        return model.rollout_batch(contexts, horizon)
    return np.asarray([model.rollout(c, horizon) for c in contexts])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pinn-epochs", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/improved-run-2026-11-05"
    )
    args = parser.parse_args()
    if min(args.epochs, args.pinn_epochs) < 1:
        parser.error("Epoch counts must be positive")
    out = args.output_dir
    if (out / "metrics.json").exists():
        parser.error("Results already exist; select a new output directory")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    source = out / "source"
    source.mkdir(exist_ok=True)
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, source / path.name)
    shutil.copytree(
        ROOT.parent / "src",
        source / "src",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    shutil.copy2(ROOT.parent / "pyproject.toml", source / "pyproject.toml")
    shutil.make_archive(str(out / "source"), "zip", source)
    shutil.rmtree(source)
    device = select_device(args.device)
    config = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "seeds": args.seeds,
        "maximum_transformer_epochs": args.epochs,
        "pinn_epochs": args.pinn_epochs,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "train_range": [0, 2000],
        "inner_fit_range": [0, 1600],
        "inner_validation_range": [1600, 2000],
        "calibration_range": [2000, 2600],
        "test_range": [2600, 3601],
        "horizon": 180,
        "dt": 0.05,
        "nominal_coverage": 0.9,
        "caveat": "One deterministic trajectory; rolling windows are dependent. "
        "Empirical conformal intervals have no asserted exchangeability guarantee.",
        "event": "position x > 0",
        "interval_states": ["x", "v"],
        "phase": "Known forcing phase: improved Transformer and polynomial dynamics",
        "pinn": "Separate reconstruction experiment, not a forecasting competitor",
    }
    save_json(out / "config.json", config)
    time, states = generate_series()
    train = states[:2000]
    horizon, window = 180, 24
    cal_origins = np.arange(2024, 2600 - horizon + 1, 20)
    test_origins = np.arange(2620, len(states) - horizon + 1, 40)

    def windows(origins, size=window):
        return np.asarray([states[o - size : o] for o in origins])

    def truths(origins):
        return np.asarray([states[o : o + horizon] for o in origins])

    contexts = np.concatenate((windows(cal_origins), windows(test_origins)))
    ncal = len(cal_origins)
    cal_truth, truth = truths(cal_origins), truths(test_origins)
    save_json(
        out / "origins.json",
        {"calibration": cal_origins.tolist(), "test": test_origins.tolist()},
    )
    np.savez_compressed(
        out / "data.npz",
        time=time,
        states=states,
        calibration_truth=cal_truth,
        truth=truth,
    )
    forecasts = {}
    ridge = RidgeForecaster().fit(train)
    forecasts["Ridge (legacy)"] = batch_rollout(ridge, contexts, horizon)
    # Hyperparameter selection uses only the inner validation window.
    val_origins = np.arange(1624, 2000 - 60 + 1, 60)
    ridge_trials = []
    for regularization in [1e-6, 1e-4, 1e-2, 1.0]:
        candidate = RidgeForecaster(regularization=regularization).fit(train[:1600])
        predicted = batch_rollout(candidate, windows(val_origins), 60)
        actual = np.asarray([states[o : o + 60] for o in val_origins])
        score = float(np.mean((predicted[:, :, :2] - actual[:, :, :2]) ** 2))
        ridge_trials.append({"regularization": regularization, "validation_mse": score})
    best_ridge = min(ridge_trials, key=lambda t: t["validation_mse"])
    tuned = RidgeForecaster(regularization=best_ridge["regularization"]).fit(train)
    forecasts["Ridge (validated)"] = batch_rollout(tuned, contexts, horizon)
    polynomial = PolynomialDynamics().fit(train)
    forecasts["Polynomial dynamics"] = polynomial.rollout_batch(contexts, horizon)
    np.save(out / "checkpoints/polynomial-coefficients.npy", polynomial.coefficients)
    print("Baselines fitted. Training legacy Transformer (35 epochs).", flush=True)
    _, legacy_model = transformer_forecast(
        train, contexts[ncal, -20:], device=str(device), return_model=True
    )
    legacy = NeuralForecaster(legacy_model, device)
    forecasts["Transformer (legacy)"] = legacy.rollout_batch(contexts[:, -20:], horizon)
    torch.save(legacy_model.state_dict(), out / "checkpoints/transformer-legacy.pt")
    training = []
    members = []
    standalone_members = []
    for seed in args.seeds:
        print(
            f"Training improved Transformer seed {seed} (max {args.epochs} epochs).",
            flush=True,
        )
        model, history = fit_transformer(
            train, seed=seed, epochs=args.epochs, device=str(device)
        )
        training.append(history)
        torch.save(model.model.state_dict(), out / f"checkpoints/transformer-{seed}.pt")
        members.append(model.rollout_batch(contexts, horizon))
        standalone_members.append(model.rollout(states[2000:2024], horizon))
        print(
            f"Seed {seed}: selected epoch {history['best_epoch']}, "
            f"validation MSE {history['validation_normalized_rollout_mse']:.5f}",
            flush=True,
        )
        save_json(out / "training.json", training)
    members = np.asarray(members)
    forecasts["Transformer (improved ensemble)"] = members.mean(0)
    metrics, details = {}, {}
    for name, forecast in forecasts.items():
        if not np.isfinite(forecast).all():
            raise RuntimeError(f"Nonfinite forecasts for {name}")
        metric, detail = calibrated_diagnostics(
            forecast[:ncal], cal_truth, forecast[ncal:], truth, train.std(0)
        )
        metric["first_origin_position_rmse"] = float(
            np.sqrt(np.mean((forecast[ncal, :, 0] - truth[0, :, 0]) ** 2))
        )
        metric["coherence_first_origin"] = coherence_length(truth[0], forecast[ncal])
        metric["mean_coherence_steps"] = float(
            np.mean(
                [
                    coherence_length(t, p)
                    for t, p in zip(truth, forecast[ncal:], strict=True)
                ]
            )
        )
        metrics[name], details[name] = metric, detail
        file_name = name.lower().replace(" ", "-").replace("(", "").replace(")", "")
        np.savez_compressed(
            out / f"{file_name}.npz",
            calibration=forecast[:ncal],
            forecast=forecast[ncal:],
            **detail,
        )
    seed_rmse = np.sqrt(
        np.mean((members[:, ncal:, :, 0] - truth[None, :, :, 0]) ** 2, axis=(1, 2))
    )
    standalone_truth = states[2024:2204]
    standalone_legacy = NeuralForecaster(legacy_model, device).rollout(
        states[2000:2024], horizon
    )
    standalone = np.asarray(standalone_members).mean(0)
    standalone_metric = {
        "origin": 2024,
        "legacy_position_rmse": float(
            np.sqrt(np.mean((standalone_legacy[:, 0] - standalone_truth[:, 0]) ** 2))
        ),
        "improved_ensemble_position_rmse": float(
            np.sqrt(np.mean((standalone[:, 0] - standalone_truth[:, 0]) ** 2))
        ),
        "note": "Retrospective point comparison inside the calibration period.",
    }
    np.savez_compressed(
        out / "standalone-comparison.npz",
        truth=standalone_truth,
        legacy=standalone_legacy,
        improved=standalone,
    )
    print("Forecasting evaluated. Training legacy PINN (800 epochs).", flush=True)
    old_pinn, old_loss = train_pinn(time, states, device=str(device), return_model=True)
    short_time = time[time <= 25]
    short_states = states[time <= 25]
    with torch.no_grad():
        old_prediction = (
            old_pinn(
                torch.tensor(short_time[:, None], dtype=torch.float32, device=device)
            )
            .cpu()
            .numpy()
        )
    print(f"Training corrected PINN ({args.pinn_epochs} epochs).", flush=True)
    pinn, pinn_prediction, pinn_metrics = fit_pinn(
        short_time, short_states, epochs=args.pinn_epochs, device=str(device)
    )
    torch.save(pinn.state_dict(), out / "checkpoints/pinn-improved.pt")
    torch.save(old_pinn.state_dict(), out / "checkpoints/pinn-legacy.pt")
    interpolation = np.flatnonzero(short_time <= 20)[1::2]
    pinn_metrics["legacy_training_loss"] = old_loss
    pinn_metrics["legacy_interpolation_rmse_x_v"] = np.sqrt(
        np.mean(
            (old_prediction[interpolation] - short_states[interpolation, :2]) ** 2,
            axis=0,
        )
    ).tolist()
    pinn_metrics["legacy_extrapolation_20_25_rmse_x_v"] = np.sqrt(
        np.mean(
            (old_prediction[short_time > 20] - short_states[short_time > 20, :2]) ** 2,
            axis=0,
        )
    ).tolist()
    pinn_metrics["legacy_observation_samples"] = int(np.sum(short_time <= 20))
    np.savez_compressed(
        out / "pinn.npz",
        time=short_time,
        truth=short_states[:, :2],
        legacy=old_prediction,
        improved=pinn_prediction,
    )
    save_json(out / "pinn-metrics.json", pinn_metrics)
    report = {
        "models": metrics,
        "ridge_validation": ridge_trials,
        "seed_position_rmse": dict(
            zip(map(str, args.seeds), seed_rmse.tolist(), strict=True)
        ),
        "seed_rmse_mean": float(seed_rmse.mean()),
        "seed_rmse_std": float(seed_rmse.std()),
        "standalone_same_window": standalone_metric,
        "calibration_origins": ncal,
        "test_origins": len(test_origins),
    }
    save_json(out / "metrics.json", report)
    plot_results(
        out,
        truth,
        forecasts,
        details,
        ncal,
        short_time,
        short_states,
        pinn_prediction,
        old_prediction,
        training,
    )
    from improved_diagnostics import run_diagnostics

    run_diagnostics(out, str(device), args.pinn_epochs)
    print(
        json.dumps(
            {
                "rolling_position_rmse": {
                    k: v["position_rmse"] for k, v in metrics.items()
                },
                "standalone": standalone_metric,
                "pinn_interpolation_rmse": pinn_metrics["interpolation_rmse_x_v"],
                "pinn_extrapolation_rmse": pinn_metrics["extrapolation_20_25_rmse_x_v"],
            },
            indent=2,
        ),
        flush=True,
    )


def plot_results(
    out, truth, forecasts, details, ncal, time, states, pinn, old_pinn, training
):
    plt.style.use("dark_background")
    figure, axes = plt.subplots(3, 1, figsize=(13, 12))
    lead = np.arange(1, truth.shape[1] + 1) * 0.05
    axes[0].plot(lead, truth[0, :, 0], color="white", label="truth")
    for name, values in forecasts.items():
        forecast = values[ncal:]
        axes[0].plot(lead, forecast[0, :, 0], label=name, alpha=0.8)
        axes[1].plot(
            lead,
            np.sqrt(np.mean((forecast[:, :, 0] - truth[:, :, 0]) ** 2, axis=0)),
            label=name,
        )
        inside = (truth[:, :, 0] >= details[name]["lower"][:, :, 0]) & (
            truth[:, :, 0] <= details[name]["upper"][:, :, 0]
        )
        axes[2].plot(lead, inside.mean(0), label=name)
    axes[2].axhline(0.9, linestyle="--", color="white", label="nominal 90%")
    for axis, label in zip(
        axes,
        ["first-origin x", "rolling-origin position RMSE", "position coverage"],
        strict=True,
    ):
        axis.set(xlabel="forecast lead (time units)", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(out / "figures/forecast-comparison.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, axis in enumerate(axes):
        axis.plot(time, states[:, i], label="truth", color="white")
        axis.plot(time, old_pinn[:, i], label="legacy PINN", alpha=0.7)
        axis.plot(time, pinn[:, i], label="corrected PINN", alpha=0.8)
        axis.axvline(20, linestyle="--", color="gold", label="observation cutoff")
        axis.set(ylabel=["position x", "velocity v"][i])
        axis.legend()
        axis.grid(alpha=0.2)
    axes[1].set_xlabel("time")
    figure.tight_layout()
    figure.savefig(out / "figures/pinn-reconstruction-and-extrapolation.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    report = json.loads((out / "metrics.json").read_text())
    for name, metric in report["models"].items():
        bins = metric["reliability_x_positive"]
        axes[0].plot(
            [b["probability"] for b in bins],
            [b["frequency"] for b in bins],
            marker="o",
            label=name,
        )
        axes[1].plot(lead, metric["by_horizon"]["crps"], label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="white")
    axes[0].set(
        xlabel="predicted P(x > 0)",
        ylabel="observed frequency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].set(xlabel="forecast lead (time units)", ylabel="position CRPS")
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(out / "figures/reliability-and-crps.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5))
    for member in training:
        axis.semilogy(
            [r["epoch"] for r in member["history"]],
            [r["validation_rollout_mse"] for r in member["history"]],
            label=f"seed {member['seed']}",
        )
    axis.set(xlabel="epoch", ylabel="inner validation normalized rollout MSE")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(out / "figures/validation.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
