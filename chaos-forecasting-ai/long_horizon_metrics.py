"""Separate trajectory accuracy from long-run single-member state statistics."""

import numpy as np
from coherence_episodes import detect_episodes, episode_policy
from scipy.stats import wasserstein_distance


def sanitize_forecasts(forecast, scale, physical_size):
    """Censor at the first nonfinite/extreme prediction; never hide failures."""
    result = np.asarray(forecast).copy()
    normalized = result[:, :, :physical_size] / scale[:physical_size]
    invalid = ~np.isfinite(normalized).all(axis=2) | (
        np.abs(normalized).max(axis=2) > 1e6
    )
    failed = np.maximum.accumulate(invalid, axis=1)
    result[failed] = np.nan
    return result


def causal_derivatives(values, dt, window=5):
    """Endpoint slope/curvature of a trailing quadratic fit, with no future data.

    The first window-1 outputs are unavailable (NaN). Caller may prepend the
    observed context, equally to truth and forecast, to score the first lead.
    Derivatives are estimated from each trajectory, never from the true ODE.
    """
    values = np.asarray(values, dtype=float)
    if dt <= 0 or window < 3:
        raise ValueError("Positive dt and at least three fit samples are required")
    result = np.full((*values.shape, 2), np.nan)
    if values.shape[1] < window:
        return result
    # Sample units keep the least-squares system well conditioned at small dt.
    times = np.arange(1 - window, 1, dtype=float)
    weights = np.linalg.pinv(np.polynomial.polynomial.polyvander(times, 2))[[1, 2]]
    weights[0] /= dt
    weights[1] *= 2 / dt**2
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=1)
    result[:, window - 1 :] = np.einsum("nhdw,kw->nhdk", windows, weights)
    return result


def validation_coherence_settings(validation, physical_size, dt, training=None):
    """Set a short window from validation dynamics, without using test forecasts.

    Use the fastest physical variable's median first 0.95 autocorrelation
    crossing across validation trajectories. The accuracy budget (0.5 training
    standard deviations) is an explicit convention, not a fitted constant.
    """
    values = np.asarray(validation)[:, :, :physical_size]
    lags = []
    for state in range(physical_size):
        crossings = []
        for trajectory in values[:, :, state]:
            centered = trajectory - trajectory.mean()
            energy = np.dot(centered, centered)
            if energy <= 0:
                raise ValueError("Validation variables must have positive variance")
            acf = (
                np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
                / energy
            )
            crossed = np.flatnonzero(acf < 0.95)
            crossings.append(int(crossed[0]) if len(crossed) else len(centered) - 1)
        lags.append(float(np.median(crossings)))
    window = max(2, int(np.ceil(min(lags))))
    persistence = max(2, int(np.ceil(window / 2)))
    settings = {
        "version": 2,
        "threshold": 0.5,
        "window_steps": window,
        "persistence_steps": persistence,
        "window_time": window * dt,
        "confirmation_delay_time": (persistence - 1) * dt,
        "validation_acf_095_median_lags_by_state": lags,
        "window_selection": "Fastest validation variable median first ACF<0.95 lag",
        "threshold_selection": "Accuracy budget: 0.5 training std; not fitted to test",
        "startup": "Partial trailing windows until a full window is available",
        "marker": "Onset of the first confirmed sustained run",
        "joint_state": "Earliest confirmed physical-variable loss",
        "forecast": "Ensemble mean; individual member coherence is also reported",
    }
    if training is not None:
        derivative_window = max(5, window + 1)
        derivatives = causal_derivatives(
            np.asarray(training)[:, :, :physical_size], dt, derivative_window
        )[:, derivative_window - 1 :]
        scales = derivatives.std(axis=(0, 1))
        if not np.isfinite(scales).all():
            raise ValueError("Training trajectories too short for derivative scales")
        settings.update(
            {
                "version": 3,
                "dt": dt,
                "derivative_window_steps": derivative_window,
                "derivative_scales_by_state": np.maximum(scales, 1e-12).tolist(),
                "derivative_scale_floor": 1e-12,
                "features": ["value", "slope", "curvature"],
                "derivative_estimator": "Trailing quadratic endpoint derivatives",
                "combination": "Max RMS; all three features must pass",
                "reference_context": "Same observed past for truth and forecast",
            }
        )
    if training is not None:
        settings["episode_policy"] = episode_policy(validation, physical_size, dt)
    return settings


def sustained_lengths(exceeded, invalid, persistence):
    """Return 1-based onset lead, censor flag, and survival for N,H,D flags."""
    n, horizon, d = exceeded.shape
    lengths = np.full((n, d), horizon, dtype=int)
    censored = np.ones((n, d), dtype=bool)
    count = np.zeros((n, d), dtype=int)
    for lead in range(horizon):
        count = np.where(exceeded[:, lead], count + 1, 0)
        numeric = invalid[:, lead] & censored
        confirmed = (count >= persistence) & censored & ~numeric
        lengths[numeric] = lead + 1
        lengths[confirmed] = lead - persistence + 2
        censored[numeric | confirmed] = False
    survival = (
        (np.arange(1, horizon + 1)[None, :, None] < lengths[:, None, :])
        | censored[:, None, :]
    ).mean(axis=0)
    return lengths, censored, survival


def coherence_errors(truth, forecast, scale, settings, contexts=None):
    """Separate training-scaled local errors for value and optional derivatives."""
    scale = np.asarray(scale)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Training scales must be finite and positive")
    window = settings["window_steps"]
    if window < 1 or settings["persistence_steps"] < 1 or settings["threshold"] <= 0:
        raise ValueError("Window, persistence, and threshold must be positive")
    invalid = ~np.isfinite(forecast) | ~np.isfinite(truth)
    features = [((forecast - truth) / scale)[..., None]]
    if "derivative_scales_by_state" in settings:
        fit_window = settings["derivative_window_steps"]
        actual, predicted = truth, forecast
        prefix = 0
        if contexts is not None:
            past = np.asarray(contexts)[:, -(fit_window - 1) :, : truth.shape[2]]
            if past.shape[1] != fit_window - 1 or not np.isfinite(past).all():
                raise ValueError(
                    "A finite observed context must cover the derivative fit"
                )
            prefix = past.shape[1]
            actual = np.concatenate([past, truth], axis=1)
            predicted = np.concatenate([past, forecast], axis=1)
        slopes = causal_derivatives(predicted, settings["dt"], fit_window)[:, prefix:]
        actual_slopes = causal_derivatives(actual, settings["dt"], fit_window)[
            :, prefix:
        ]
        derivative_scale = np.asarray(settings["derivative_scales_by_state"])
        if not np.isfinite(derivative_scale).all() or (derivative_scale <= 0).any():
            raise ValueError("Derivative training scales must be finite and positive")
        features.append((slopes - actual_slopes) / derivative_scale)
    normalized = np.concatenate(features, axis=-1)
    available = np.isfinite(normalized)
    squared = np.where(available, normalized, 0) ** 2
    cumulative = np.concatenate(
        [np.zeros_like(squared[:, :1]), np.cumsum(squared, axis=1)], axis=1
    )
    counts = np.concatenate(
        [np.zeros_like(squared[:, :1]), np.cumsum(available, axis=1)], axis=1
    )
    ends = np.arange(1, truth.shape[1] + 1)
    starts = np.maximum(0, ends - window)
    count = counts[:, ends] - counts[:, starts]
    rolling = np.sqrt(
        np.divide(
            np.maximum(cumulative[:, ends] - cumulative[:, starts], 0),
            count,
            out=np.full_like(squared, np.nan),
            where=count > 0,
        )
    )
    rolling[invalid] = np.inf
    combined = np.max(np.where(np.isnan(rolling), -np.inf, rolling), axis=-1)
    return combined, rolling


def coherence_curves(truth, forecast, scale, settings=None, contexts=None):
    """Sustained local error breach, with recoverable value/shape agreement."""
    settings = settings or {"window_steps": 5, "persistence_steps": 3, "threshold": 0.5}
    rolling, _ = coherence_errors(truth, forecast, scale, settings, contexts)
    invalid = ~np.isfinite(forecast) | ~np.isfinite(truth)
    lengths, censored, survival = sustained_lengths(
        rolling > settings["threshold"], invalid, settings["persistence_steps"]
    )
    return lengths, censored, survival, rolling


def evaluate_long_run(
    truth,
    members,
    scale,
    physical_size,
    dt,
    cutoffs,
    coherence_settings=None,
    contexts=None,
):
    """Report ensemble point errors and individual-trajectory distribution errors."""
    members = np.asarray([sanitize_forecasts(m, scale, physical_size) for m in members])
    mean = members.mean(0)
    actual = truth[:, :, :physical_size]
    predicted = mean[:, :, :physical_size]
    lengths, censored, survival, rolling = coherence_curves(
        actual, predicted, scale[:physical_size], coherence_settings, contexts
    )
    member_coherence = [
        coherence_curves(
            actual,
            member[:, :, :physical_size],
            scale[:physical_size],
            coherence_settings,
            contexts,
        )[:2]
        for member in members
    ]
    _, components = coherence_errors(
        actual,
        predicted,
        scale[:physical_size],
        coherence_settings
        or {"window_steps": 5, "persistence_steps": 3, "threshold": 0.5},
        contexts,
    )
    # A first breach records past accuracy loss; it does not forbid recovery.
    tolerance = (coherence_settings or {}).get("threshold", 0.5)
    within = np.isfinite(rolling) & (rolling <= tolerance)
    recoveries = []
    for origin in range(len(actual)):
        recovery = []
        for state in range(physical_size):
            start = lengths[origin, state] - 1
            later = np.flatnonzero(within[origin, start:, state])
            recovery.append(
                float((start + later[0] + 1) * dt)
                if not censored[origin, state] and len(later)
                else None
            )
        recoveries.append(recovery)
    valid = np.isfinite(predicted).all(axis=2)
    numeric_failure = ~valid
    first_numeric = np.argmax(numeric_failure, axis=1) + 1
    first_numeric[~numeric_failure.any(axis=1)] = 0
    errors = predicted - actual
    metrics = {
        "accuracy_interpretation": (
            "Coherence fields retain the first confirmed local accuracy breach. "
            "They do not measure permanent loss of coherence. "
            "Local accuracy can recover; see within_tolerance and recovery times."
        ),
        "first_breach_recovery_time_by_origin_state": recoveries,
        "coherence_definition": (
            "First sustained breach of the maximum training-scaled rolling RMS "
            "over configured value/slope/curvature features; no correlation. "
            "Marker is onset of confirmed run. Numerical failures are immediate. "
            "Joint loss is earliest confirmed physical-variable loss."
        ),
        "coherence_settings": coherence_settings
        or {"window_steps": 5, "persistence_steps": 3, "threshold": 0.5},
        "coherence_joint_time_by_origin": (lengths.min(axis=1) * dt).tolist(),
        "coherence_joint_censored_by_origin": censored.all(axis=1).tolist(),
        "coherence_joint_median_capped": float(np.median(lengths.min(axis=1)) * dt),
        "coherence_time_by_origin_state": (lengths * dt).tolist(),
        "coherence_censored_by_origin_state": censored.tolist(),
        "coherence_x_median_capped": float(np.median(lengths[:, 0]) * dt),
        "coherence_x_mean_capped": float(np.mean(lengths[:, 0]) * dt),
        "coherence_x_censored_fraction": float(censored[:, 0].mean()),
        "member_coherence_time_by_origin_state": [
            (length * dt).tolist() for length, _ in member_coherence
        ],
        "member_coherence_censored_by_origin_state": [
            censor.tolist() for _, censor in member_coherence
        ],
        "numerical_failure_fraction": float(numeric_failure.any(axis=1).mean()),
        "numerical_failure_lead_by_origin": first_numeric.tolist(),
        "point_by_cutoff": {},
    }
    metrics["coherence_threshold_sensitivity"] = {}
    for threshold in [0.25, 0.5, 0.75]:
        sensitivity_settings = dict(metrics["coherence_settings"], threshold=threshold)
        sl, sc, _, _ = coherence_curves(
            actual, predicted, scale[:physical_size], sensitivity_settings, contexts
        )
        metrics["coherence_threshold_sensitivity"][str(threshold)] = {
            "joint_median_capped": float(np.median(sl.min(axis=1)) * dt),
            "joint_censored_fraction": float(sc.all(axis=1).mean()),
        }
    episode_arrays = {}
    if coherence_settings and "episode_policy" in coherence_settings:
        report, status = detect_episodes(
            rolling, coherence_settings["episode_policy"], numeric_failure
        )
        metrics["episode_coherence"] = report
        episode_arrays["episode_status_by_origin"] = status
    for cutoff in cutoffs:
        lead = min(int(round(cutoff / dt)), truth.shape[1])
        partial = errors[:, :lead]
        if np.isfinite(partial).all():
            rmse = np.sqrt(np.mean(partial**2, axis=(0, 1))).tolist()
            normalized = float(np.sqrt(np.mean((partial / scale[:physical_size]) ** 2)))
        else:
            rmse, normalized = None, None
        metrics["point_by_cutoff"][str(cutoff)] = {
            "rmse_by_state": rmse,
            "normalized_rmse_all_states": normalized,
            "valid_origin_fraction": float(valid[:, :lead].all(axis=1).mean()),
        }
    tail = truth.shape[1] // 10
    actual_tail = actual[:, tail:].reshape(-1, physical_size)
    climate = []
    for member in members:
        sample = member[:, tail:, :physical_size].reshape(-1, physical_size)
        finite = np.isfinite(sample).all(axis=1)
        if finite.all():
            w = [
                wasserstein_distance(actual_tail[:, i], sample[:, i]) / scale[i]
                for i in range(physical_size)
            ]
            std_ratio = (sample.std(0) / actual_tail.std(0)).tolist()
            mean_bias = (
                (sample.mean(0) - actual_tail.mean(0)) / scale[:physical_size]
            ).tolist()
        else:
            w, std_ratio, mean_bias = None, None, None
        climate.append(
            {
                "normalized_wasserstein_by_state": w,
                "std_ratio_by_state": std_ratio,
                "normalized_mean_bias_by_state": mean_bias,
                "finite_fraction": float(finite.mean()),
            }
        )
    scores = [
        np.mean(c["normalized_wasserstein_by_state"])
        for c in climate
        if c["normalized_wasserstein_by_state"] is not None
    ]
    metrics["individual_member_climate"] = climate
    metrics["climate_valid_member_fraction"] = len(scores) / len(members)
    metrics["mean_normalized_member_wasserstein"] = (
        float(np.mean(scores)) if len(scores) == len(members) else None
    )
    metrics["climate_tail_start_time"] = tail * dt
    metrics["climate_note"] = (
        "Distribution statistics use individual free trajectories, "
        "not the ensemble mean; first 10% of forecast excluded. "
        "Distribution agreement alone does not establish correct "
        "dynamics."
    )
    # Plot arrays retain explicit NaNs for numerical failures.
    point_rmse = np.sqrt(np.mean(errors**2, axis=0))
    return metrics, {
        **episode_arrays,
        "members": members,
        "mean": mean,
        "point_rmse_by_lead": point_rmse,
        "coherence_survival": survival,
        "within_tolerance_by_origin_state": within,
        "feature_nrmse_by_origin_state": components,
        "value_within_tolerance_by_origin_state": components[..., 0] <= tolerance,
        "rolling_nrmse_by_origin_state": rolling,
        "rolling_nrmse_mean": rolling.mean(axis=0),
        "joint_coherence_survival": (
            (np.arange(1, actual.shape[1] + 1)[None, :] < lengths.min(axis=1)[:, None])
            | censored.all(axis=1)[:, None]
        ).mean(axis=0),
    }
