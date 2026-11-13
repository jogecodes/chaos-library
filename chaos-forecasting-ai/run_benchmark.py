"""Run the reproducible chaotic forecasting benchmark."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data import generate_series, save_series
from metrics import brier_score, calibration_error, coherence_length
from models import RidgeForecaster
from uncertainty import conformal_radius, ensemble_rollouts

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def main() -> None:
    """Generate data, forecasts, conformal intervals, and figures."""
    time, states = generate_series()
    save_series(RESULTS / "forced-van-der-pol.npz", time, states)
    train_end = 2_000
    calibration_end = 2_600
    horizon = 180
    train = states[:train_end]
    calibration = states[train_end:calibration_end]
    test = states[calibration_end:]
    initial = test[:20]

    model = RidgeForecaster().fit(train)
    forecast = model.rollout(initial, horizon)
    members = ensemble_rollouts(train, initial, horizon)
    radius = conformal_radius(
        train,
        calibration,
        horizon=horizon,
    )
    lower = forecast - radius
    upper = forecast + radius
    truth = test[20 : 20 + horizon]
    inside = (truth >= lower) & (truth <= upper)
    coverage = np.mean(inside, axis=0)
    coverage_by_horizon = np.mean(inside, axis=1)
    ensemble_inside = np.all(
        (members >= lower[None]) & (members <= upper[None]),
        axis=2,
    )
    interval_probability = np.mean(ensemble_inside, axis=0)
    interval_outcome = np.all(inside, axis=1)
    rmse = np.sqrt(np.mean((forecast - truth) ** 2, axis=1))
    coherence = coherence_length(truth, forecast)
    nominal_coverage = 0.9
    metrics = {
        "system": "forced_van_der_pol",
        "train_samples": train_end,
        "calibration_samples": len(calibration),
        "forecast_horizon": horizon,
        "nominal_coverage": nominal_coverage,
        "coverage_by_state": coverage.tolist(),
        "coverage_by_horizon": coverage_by_horizon.tolist(),
        "mean_coverage": float(coverage.mean()),
        "calibration_error": calibration_error(
            coverage_by_horizon,
            nominal_coverage,
        ),
        "brier_score_interval_containment": brier_score(
            interval_probability,
            interval_outcome,
        ),
        "coherence_length": coherence,
        "coherence_definition": {
            "state": "position",
            "correlation_threshold": 0.8,
            "normalized_rmse_threshold": 0.5,
        },
        "rmse_final_state": rmse[-1].tolist(),
        "mean_interval_width": float(np.mean(2.0 * radius)),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    lead = np.arange(1, horizon + 1)
    axes[0].plot(lead, truth[:, 0], label="truth", color="#5ee7f2")
    axes[0].plot(lead, forecast[:, 0], label="Ridge forecast", color="#f4c95d")
    axes[0].fill_between(
        lead,
        lower[:, 0],
        upper[:, 0],
        color="#ff8066",
        alpha=0.25,
        label="90% conformal interval",
    )
    axes[0].set_ylabel("position x")
    axes[0].legend()
    axes[1].plot(
        lead,
        coverage_by_horizon,
        color="#b8f26d",
        label="coverage across state",
    )
    axes[1].axhline(
        nominal_coverage,
        color="#ff8066",
        linestyle="--",
        label="nominal 90%",
    )
    axes[1].axvline(
        coherence,
        color="#5ee7f2",
        linestyle=":",
        label=f"coherence length: {coherence}",
    )
    axes[1].set(xlabel="forecast lead", ylabel="empirical coverage")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    for axis in axes:
        axis.set_facecolor("#182936")
        axis.grid(alpha=0.16, color="#8fa8b5")
    figure.suptitle("Chaotic forecasting: error growth and calibrated uncertainty")
    figure.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(RESULTS / "figures" / "forecast-and-uncertainty.png", dpi=160)
    plt.close(figure)
    np.savez_compressed(
        RESULTS / "forecasts.npz",
        truth=truth,
        forecast=forecast,
        ensemble=members,
        lower=lower,
        upper=upper,
        coverage=coverage,
        coverage_by_horizon=coverage_by_horizon,
        interval_probability=interval_probability,
        interval_outcome=interval_outcome,
    )


if __name__ == "__main__":
    main()
