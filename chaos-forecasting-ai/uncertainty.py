"""Uncertainty calibration for recursive forecasts."""

import numpy as np
from improved_evaluation import conformal_quantile
from models import RidgeForecaster


def ensemble_rollouts(
    train: np.ndarray,
    initial: np.ndarray,
    horizon: int,
    *,
    window: int = 20,
    members: int = 8,
) -> np.ndarray:
    """Return forecasts from contiguous-window Ridge ensemble members."""
    starts = np.linspace(
        max(window + 1, len(train) - 800),
        len(train) - horizon - 1,
        members,
        dtype=int,
    )
    forecasts = []
    for start in starts:
        model = RidgeForecaster(window=window).fit(train[:start])
        forecasts.append(model.rollout(initial, horizon))
    return np.asarray(forecasts)


def conformal_radius(
    train: np.ndarray,
    calibration: np.ndarray,
    *,
    horizon: int,
    alpha: float = 0.1,
    window: int = 20,
) -> np.ndarray:
    """Calibrate per-horizon absolute residual radii with split conformal."""
    model = RidgeForecaster(window=window).fit(train)
    residuals = []
    for origin in range(window, len(calibration) - horizon + 1):
        initial = calibration[origin - window : origin]
        forecast = model.rollout(initial, horizon)
        residuals.append(np.abs(forecast - calibration[origin : origin + horizon]))
    if not residuals:
        raise ValueError("calibration series is too short")
    residuals = np.asarray(residuals)
    return conformal_quantile(residuals, alpha)
