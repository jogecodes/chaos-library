"""Model-specific horizon calibration and rolling-origin diagnostics."""

import numpy as np


def safe_correlation(actual, predicted):
    """Return zero for a constant segment without NumPy divide warnings."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) < 2:
        return 0.0
    a, b = actual - actual.mean(), predicted - predicted.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    tolerance = np.finfo(float).eps * max(
        1.0, np.linalg.norm(actual) * np.linalg.norm(predicted)
    )
    return float(np.dot(a, b) / norm) if norm > tolerance else 0.0


def conformal_quantile(scores, alpha=0.1):
    """Use the finite-sample rank, including infinity if calibration is too small."""
    scores = np.asarray(scores)
    if scores.shape[0] == 0 or not 0 < alpha < 1:
        raise ValueError("Need calibration scores and alpha between zero and one")
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if rank > len(scores):
        return np.full(scores.shape[1:], np.inf)
    return np.sort(scores, axis=0)[rank - 1]


def calibrated_diagnostics(cal_forecast, cal_truth, forecast, truth, scale, alpha=0.1):
    """Calibrate each lead and physical state on separate pre-test origins."""
    arrays = [cal_forecast, cal_truth, forecast, truth]
    if any(np.asarray(value).ndim != 3 for value in arrays):
        raise ValueError("Forecasts and truth must have origin, horizon, state axes")
    if cal_forecast.shape != cal_truth.shape or forecast.shape != truth.shape:
        raise ValueError("Forecast and truth shapes must agree")
    if cal_forecast.shape[1:] != forecast.shape[1:]:
        raise ValueError("Calibration and test horizons/states must agree")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Forecasts and truth must be finite")
    scale = np.maximum(np.asarray(scale), 1e-8)
    residuals = cal_truth[:, :, :2] - cal_forecast[:, :, :2]
    radius = conformal_quantile(np.abs(residuals), alpha)
    # Joint x,v coverage at each lead, separate from marginal x or v coverage.
    joint_radius = conformal_quantile(
        np.max(np.abs(residuals) / scale[:2], axis=2), alpha
    )
    lower, upper = forecast[:, :, :2] - radius, forecast[:, :, :2] + radius
    inside = (truth[:, :, :2] >= lower) & (truth[:, :, :2] <= upper)
    joint_inside = np.all(
        np.abs(truth[:, :, :2] - forecast[:, :, :2]) / scale[:2]
        <= joint_radius[None, :, None],
        axis=2,
    )
    # A fixed observable event rather than membership in a model-defined interval.
    samples = forecast[None, :, :, 0] + residuals[:, None, :, 0]
    probabilities = np.mean(samples > 0, axis=0)
    outcomes = truth[:, :, 0] > 0
    brier = np.mean((probabilities - outcomes) ** 2, axis=0)
    ordered = np.sort(samples, axis=0)
    n = len(ordered)
    coefficients = (2 * np.arange(1, n + 1) - n - 1)[:, None, None]
    crps = (
        np.abs(samples - truth[None, :, :, 0]).mean(axis=0)
        - np.sum(coefficients * ordered, axis=0) / n**2
    )
    error = forecast[:, :, 0] - truth[:, :, 0]
    rmse = np.sqrt(np.mean(error**2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    correlation = np.asarray(
        [
            safe_correlation(truth[:, h, 0], forecast[:, h, 0])
            for h in range(truth.shape[1])
        ]
    )
    # Central 90% interval score penalizes both width and missed observations.
    interval_score = (
        upper[:, :, 0]
        - lower[:, :, 0]
        + 2 / alpha * np.maximum(lower[:, :, 0] - truth[:, :, 0], 0)
        + 2 / alpha * np.maximum(truth[:, :, 0] - upper[:, :, 0], 0)
    )
    reliability = []
    for low in np.linspace(0, 0.8, 5):
        mask = (probabilities >= low) & (probabilities < low + 0.2)
        if low > 0.79:
            mask = (probabilities >= low) & (probabilities <= 1)
        if mask.any():
            reliability.append(
                {
                    "count": int(mask.sum()),
                    "probability": float(probabilities[mask].mean()),
                    "frequency": float(outcomes[mask].mean()),
                }
            )
    metrics = {
        "position_rmse": float(np.sqrt(np.mean(error**2))),
        "position_mae": float(np.mean(np.abs(error))),
        "position_coverage": float(inside[:, :, 0].mean()),
        "velocity_coverage": float(inside[:, :, 1].mean()),
        "joint_physical_state_coverage": float(joint_inside.mean()),
        "position_calibration_error": float(
            np.mean(np.abs(inside[:, :, 0].mean(0) - (1 - alpha)))
        ),
        "position_mean_interval_width": float(np.mean(2 * radius[:, 0])),
        "position_interval_score": float(interval_score.mean()),
        "brier_x_positive": float(brier.mean()),
        "crps_position": float(crps.mean()),
        "reliability_x_positive": reliability,
        "by_horizon": {
            "rmse": rmse.tolist(),
            "mae": mae.tolist(),
            "correlation": correlation.tolist(),
            "position_coverage": inside[:, :, 0].mean(0).tolist(),
            "joint_coverage": joint_inside.mean(0).tolist(),
            "brier_x_positive": brier.tolist(),
            "crps": crps.mean(0).tolist(),
        },
    }
    return metrics, {
        "lower": lower,
        "upper": upper,
        "joint_radius": joint_radius,
        "event_probability": probabilities,
    }
