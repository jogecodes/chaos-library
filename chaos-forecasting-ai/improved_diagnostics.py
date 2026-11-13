"""Frozen-model transfer checks, reliability figures, and forecast animations."""

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import generate_series
from improved_evaluation import calibrated_diagnostics
from improved_models import NeuralForecaster, ResidualTransformer, fit_pinn
from matplotlib.animation import FuncAnimation, PillowWriter
from metrics import coherence_length
from models import PolynomialDynamics, RidgeForecaster
from torch_models import SequenceTransformer, select_device


class Ensemble:
    def __init__(self, members):
        self.members = members

    def rollout_batch(self, contexts, horizon):
        return np.mean(
            [m.rollout_batch(contexts, horizon) for m in self.members], axis=0
        )


def run_diagnostics(out, device="auto", pinn_epochs=4000, reuse_pinn=False):
    """Evaluate fixed hyperparameters on two additional initial conditions."""
    from run_improved import batch_rollout, plot_results, save_json

    out = Path(out)
    device = select_device(device)
    report = json.loads((out / "metrics.json").read_text())
    config = json.loads((out / "config.json").read_text())
    data = np.load(out / "data.npz")
    train = data["states"][:2000]
    truth, cal_truth = data["truth"], data["calibration_truth"]
    members = []
    for seed in config["seeds"]:
        weights = torch.load(
            out / f"checkpoints/transformer-{seed}.pt",
            map_location=device,
            weights_only=True,
        )
        model = ResidualTransformer(np.zeros(4), np.ones(4)).to(device)
        model.load_state_dict(weights)
        members.append(NeuralForecaster(model, device))
    legacy = SequenceTransformer().to(device)
    legacy.load_state_dict(
        torch.load(
            out / "checkpoints/transformer-legacy.pt",
            map_location=device,
            weights_only=True,
        )
    )
    polynomial = PolynomialDynamics()
    polynomial.dt = 0.05
    polynomial.coefficients = np.load(out / "checkpoints/polynomial-coefficients.npy")
    regularization = min(report["ridge_validation"], key=lambda t: t["validation_mse"])[
        "regularization"
    ]
    models = {
        "Ridge (legacy)": RidgeForecaster().fit(train),
        "Ridge (validated)": RidgeForecaster(regularization=regularization).fit(train),
        "Polynomial dynamics": polynomial,
        "Transformer (legacy)": NeuralForecaster(legacy, device),
        "Transformer (improved ensemble)": Ensemble(members),
    }
    origins = json.loads((out / "origins.json").read_text())
    test_origins = origins["test"]
    saved, forecasts, details = {}, {}, {}
    for name in models:
        filename = name.lower().replace(" ", "-").replace("(", "").replace(")", "")
        saved[name] = np.load(out / f"{filename}.npz")
        forecasts[name] = np.concatenate(
            (saved[name]["calibration"], saved[name]["forecast"])
        )
        details[name] = {k: saved[name][k] for k in ["lower", "upper"]}
    transfer = {}
    previous_transfer = (
        json.loads((out / "transfer-metrics.json").read_text())
        if (out / "transfer-metrics.json").exists()
        else {}
    )
    for number, initial in enumerate([(0.2, 0.0), (0.1, 0.1)], start=1):
        print(f"Frozen-model transfer check: initial state {initial}", flush=True)
        time, states = generate_series(initial_state=initial)
        contexts = np.asarray([states[o - 24 : o] for o in test_origins])
        actual = np.asarray([states[o : o + 180] for o in test_origins])
        result, arrays = {}, {"time": time, "states": states, "truth": actual}
        for name, model in models.items():
            context = contexts[:, -20:] if name == "Transformer (legacy)" else contexts
            forecast = batch_rollout(model, context, 180)
            metric, _ = calibrated_diagnostics(
                saved[name]["calibration"], cal_truth, forecast, actual, train.std(0)
            )
            result[name] = {k: v for k, v in metric.items() if k != "by_horizon"}
            arrays[name] = forecast
        np.savez_compressed(out / f"transfer-{number}.npz", **arrays)
        existing = previous_transfer
        if (
            reuse_pinn
            and str(initial) in existing
            and "validation_samples" in existing[str(initial)]["pinn"]
        ):
            pinn_metric = existing[str(initial)]["pinn"]
        else:
            print(
                f"Corrected PINN reconstruction for initial state {initial}", flush=True
            )
            mask = time <= 25
            pinn, prediction, pinn_metric = fit_pinn(
                time[mask], states[mask], epochs=pinn_epochs, device=str(device)
            )
            torch.save(
                pinn.state_dict(), out / f"checkpoints/pinn-transfer-{number}.pt"
            )
            np.savez_compressed(
                out / f"pinn-transfer-{number}.npz",
                time=time[mask],
                truth=states[mask, :2],
                prediction=prediction,
            )
        transfer[str(initial)] = {"models": result, "pinn": pinn_metric}
        save_json(out / "transfer-metrics.json", transfer)
    pinn_data = np.load(out / "pinn.npz")
    training = json.loads((out / "training.json").read_text())
    plot_results(
        out,
        truth,
        forecasts,
        details,
        len(origins["calibration"]),
        pinn_data["time"],
        pinn_data["truth"],
        pinn_data["improved"],
        pinn_data["legacy"],
        training,
    )
    animate_forecasts(out, truth, saved)
    from plot_coherence import plot_coherence

    plot_coherence(out)
    with zipfile.ZipFile(
        out / "diagnostics-source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in Path(__file__).parent.glob("*.py"):
            archive.write(path, path.name)
    print("Transfer evaluation, reliability figures, and animation saved.", flush=True)


def animate_forecasts(out, truth, forecasts):
    """Show each model's own intervals and first-origin coherence markers."""
    plt.style.use("dark_background")
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    lead = np.arange(1, 181) * 0.05
    lines = []
    for axis, (name, values) in zip(axes.flat, forecasts.items(), strict=True):
        forecast = values["forecast"][0]
        axis.fill_between(
            lead,
            values["lower"][0, :, 0],
            values["upper"][0, :, 0],
            color="gold",
            alpha=0.18,
            label="90% marginal x interval",
        )
        coherence = coherence_length(truth[0], forecast)
        axis.axvline(
            coherence * 0.05, color="cyan", linestyle=":", label="coherence limit"
        )
        (actual_line,) = axis.plot([], [], color="white", label="truth")
        (predicted_line,) = axis.plot([], [], color="gold", label="forecast")
        lines.append((actual_line, predicted_line, forecast[:, 0]))
        axis.set(
            title=name, xlabel="lead (time units)", ylabel="position x", xlim=(0, 9)
        )
        axis.legend(fontsize=6)
        axis.grid(alpha=0.2)

    def update(frame):
        for actual, predicted, forecast in lines:
            actual.set_data(lead[:frame], truth[0, :frame, 0])
            predicted.set_data(lead[:frame], forecast[:frame])
        return [line for a, b, _ in lines for line in (a, b)]

    figure.tight_layout()
    animation = FuncAnimation(figure, update, frames=range(1, 181, 3), blit=False)
    animation.save(
        out / "figures/forecast-comparison.gif", writer=PillowWriter(fps=15), dpi=80
    )
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pinn-epochs", type=int, default=4000)
    parser.add_argument("--reuse-pinn", action="store_true")
    args = parser.parse_args()
    run_diagnostics(args.results, args.device, args.pinn_epochs, args.reuse_pinn)
