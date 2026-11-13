"""Metrics for trajectory coherence and calibrated uncertainty."""

import numpy as np


def coherence_length(
    truth: np.ndarray,
    forecast: np.ndarray,
    *,
    state: int = 0,
    correlation_threshold: float = 0.8,
    normalized_rmse_threshold: float = 0.5,
) -> int:
    """Return the first lead where a forecast loses trajectory coherence.

    Coherence is defined operationally as both a Pearson correlation above the
    threshold and a cumulative RMSE below the truth standard deviation times
    ``normalized_rmse_threshold``. The returned lead is one-based.
    """
    actual = np.asarray(truth)[:, state]
    predicted = np.asarray(forecast)[:, state]
    scale = max(float(np.std(actual)), np.finfo(float).eps)
    for lead in range(2, len(actual) + 1):
        a = actual[:lead] - actual[:lead].mean()
        b = predicted[:lead] - predicted[:lead].mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        tolerance = np.finfo(float).eps * max(
            1.0, np.linalg.norm(actual[:lead]) * np.linalg.norm(predicted[:lead])
        )
        correlation = (
            float(np.dot(a, b) / denominator) if denominator > tolerance else 0.0
        )
        normalized_rmse = (
            np.sqrt(np.mean((actual[:lead] - predicted[:lead]) ** 2)) / scale
        )
        if (
            not np.isfinite(correlation)
            or correlation < correlation_threshold
            or normalized_rmse > normalized_rmse_threshold
        ):
            return lead
    return len(actual)


def calibration_error(
    coverage_by_horizon: np.ndarray,
    nominal_coverage: float,
) -> float:
    """Return mean absolute deviation from nominal empirical coverage."""
    return float(np.mean(np.abs(coverage_by_horizon - nominal_coverage)))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Return the mean squared error of probabilistic binary forecasts."""
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if probabilities.shape != outcomes.shape:
        raise ValueError("probabilities and outcomes must have the same shape")
    if not np.isfinite(probabilities).all() or not np.isfinite(outcomes).all():
        raise ValueError("probabilities and outcomes must be finite")
    if np.any((outcomes != 0.0) & (outcomes != 1.0)):
        raise ValueError("outcomes must be binary")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie between zero and one")
    return float(np.mean((probabilities - outcomes) ** 2))
