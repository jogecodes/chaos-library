"""Legacy comparison using shared Gaussian residual-scale intervals."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import generate_series
from improved_evaluation import safe_correlation
from matplotlib.animation import FuncAnimation, PillowWriter
from metrics import brier_score, coherence_length
from models import RidgeForecaster
from scipy.special import ndtr
from torch_models import SequenceTransformer, select_device

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
HORIZON = 180
STATE_SIZE = 4
NOMINAL_COVERAGE = 0.9


def predictive_probability(
    forecast: np.ndarray,
    truth: np.ndarray,
    calibration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build Gaussian interval probabilities from calibration residual scales."""
    residuals = np.asarray(calibration)
    scale = np.maximum(np.std(residuals, axis=0), 1e-6)
    z = 1.6448536269514722
    radius = z * scale
    lower = forecast - radius
    upper = forecast + radius
    probabilities = np.prod(
        ndtr((upper - forecast) / scale) - ndtr((lower - forecast) / scale),
        axis=1,
    )
    outcomes = np.all((truth >= lower) & (truth <= upper), axis=1)
    return probabilities, outcomes, lower, upper


def evaluate_model(
    name: str,
    forecast: np.ndarray,
    truth: np.ndarray,
    calibration_residuals: np.ndarray,
) -> tuple[dict, dict]:
    """Compute point and probabilistic metrics for one model."""
    probabilities, outcomes, lower, upper = predictive_probability(
        forecast,
        truth,
        calibration_residuals,
    )
    errors = forecast - truth
    correlation = safe_correlation(truth[:, 0], forecast[:, 0])
    metrics = {
        "model": name,
        "rmse_position": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "mae_position": float(np.mean(np.abs(errors[:, 0]))),
        "final_rmse_position": float(np.sqrt(np.mean(errors[-20:, 0] ** 2))),
        "position_correlation": float(correlation),
        "coherence_length": coherence_length(truth, forecast),
        "coverage": float(np.mean(outcomes)),
        "nominal_coverage": NOMINAL_COVERAGE,
        "calibration_error": float(abs(np.mean(outcomes) - NOMINAL_COVERAGE)),
        "brier_score": brier_score(probabilities, outcomes),
        "mean_log_score": float(
            np.mean(
                outcomes * np.log(np.maximum(probabilities, 1e-12))
                + (1.0 - outcomes) * np.log(np.maximum(1.0 - probabilities, 1e-12))
            )
        ),
        "mean_interval_width": float(np.mean(upper[:, 0] - lower[:, 0])),
    }
    return metrics, {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "probabilities": probabilities,
        "outcomes": outcomes,
    }


def transformer_forecast(
    train: np.ndarray,
    initial: np.ndarray,
    *,
    device: str = "auto",
    epochs: int = 35,
    horizon: int = HORIZON,
    return_model: bool = False,
) -> np.ndarray | tuple[np.ndarray, torch.nn.Module]:
    """Train a small Transformer and return a recursive forecast."""
    device = select_device(device)
    torch.manual_seed(7)
    window = 24
    features = np.asarray(
        [train[index - window : index] for index in range(window, len(train))]
    )
    inputs = torch.tensor(features, dtype=torch.float32, device=device)
    labels = torch.tensor(train[window:], dtype=torch.float32, device=device)
    model = SequenceTransformer(state_size=train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_function = torch.nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    history = torch.tensor(initial, dtype=torch.float32, device=device)
    predictions = []
    with torch.no_grad():
        for _ in range(horizon):
            next_state = model(history[-window:].unsqueeze(0))[0]
            predictions.append(next_state.cpu().numpy())
            history = torch.cat((history, next_state[None]), dim=0)
    if return_model:
        return np.asarray(predictions), model
    return np.asarray(predictions)


def main() -> None:
    """Run the comparison and generate static and animated diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    device = select_device(args.device)
    results = args.output_dir
    time, states = generate_series(duration=180.0, dt=0.05)
    train_end = 2_000
    calibration_end = 2_600
    train = states[:train_end]
    calibration = states[train_end:calibration_end]
    truth = states[calibration_end + 20 : calibration_end + 20 + HORIZON]
    initial = states[calibration_end : calibration_end + 20]
    calibration_residuals = calibration[20:] - calibration[:-20]

    ridge = RidgeForecaster().fit(train)
    forecasts = {
        "Ridge": ridge.rollout(initial, HORIZON),
        "Transformer": transformer_forecast(train, initial, device=str(device)),
    }
    metrics = []
    details = {}
    for name, forecast in forecasts.items():
        result, detail = evaluate_model(
            name,
            forecast,
            truth,
            calibration_residuals,
        )
        if name == "Transformer":
            result.update(
                device=str(device),
                device_name=(
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
                torch_version=torch.__version__,
                training_epochs=35,
                seed=7,
            )
        metrics.append(result)
        details[name] = detail

    (results / "figures").mkdir(parents=True, exist_ok=True)
    (results / "model-comparison-metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    np.savez_compressed(
        results / "model-comparison.npz",
        truth=truth,
        ridge=details["Ridge"]["forecast"],
        transformer=details["Transformer"]["forecast"],
    )

    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    lead = np.arange(1, HORIZON + 1)
    axes[0].plot(lead, truth[:, 0], color="#5ee7f2", label="truth")
    for name, color in (
        ("Ridge", "#f4c95d"),
        ("Transformer", "#b8f26d"),
    ):
        detail = details[name]
        axes[0].plot(lead, detail["forecast"][:, 0], color=color, label=name)
        axes[0].fill_between(
            lead,
            detail["lower"][:, 0],
            detail["upper"][:, 0],
            color=color,
            alpha=0.12,
        )
    axes[0].set_ylabel("position x")
    axes[0].legend()
    names = [item["model"] for item in metrics]
    axes[1].bar(
        np.arange(len(names)) - 0.2,
        [item["brier_score"] for item in metrics],
        width=0.4,
        color="#b8f26d",
        label="Brier (lower is better)",
    )
    axes[1].bar(
        np.arange(len(names)) + 0.2,
        [item["calibration_error"] for item in metrics],
        width=0.4,
        color="#ff8066",
        label="coverage error",
    )
    axes[1].set_xticks(np.arange(len(names)), names)
    axes[1].set_ylabel("score")
    axes[1].legend()
    for axis in axes:
        axis.set_facecolor("#182936")
        axis.grid(alpha=0.16, color="#8fa8b5")
    figure.suptitle("Chaotic forecasting model comparison")
    figure.tight_layout()
    figure.savefig(results / "figures" / "model-comparison.png", dpi=160)
    plt.close(figure)

    animation_figure, axis = plt.subplots(figsize=(12, 5))
    axis.set(
        xlim=(0, HORIZON),
        ylim=(truth[:, 0].min() - 0.5, truth[:, 0].max() + 0.5),
        xlabel="forecast lead",
        ylabel="position x",
        title="Forecasts losing coherence in a chaotic system",
    )
    axis.set_facecolor("#182936")
    (truth_line,) = axis.plot([], [], color="#5ee7f2", label="truth", linewidth=2)
    prediction_lines = {
        name: axis.plot([], [], color=color, label=name)[0]
        for name, color in (
            ("Ridge", "#f4c95d"),
            ("Transformer", "#b8f26d"),
        )
    }
    axis.legend()

    def update(frame: int):
        lead_slice = slice(0, frame + 1)
        truth_line.set_data(lead[lead_slice], truth[lead_slice, 0])
        for name, line in prediction_lines.items():
            line.set_data(lead[lead_slice], details[name]["forecast"][lead_slice, 0])
        return truth_line, *prediction_lines.values()

    animation = FuncAnimation(
        animation_figure,
        update,
        frames=range(1, HORIZON + 1, 3),
        interval=50,
        blit=True,
    )
    animation.save(
        results / "figures" / "model-comparison.gif",
        writer=PillowWriter(fps=20),
        dpi=100,
    )
    plt.close(animation_figure)


if __name__ == "__main__":
    main()
