"""Separate temporary disagreement, recovery, and sustained local breakdown."""

import numpy as np

CONFIRMATION_MEMORY_MULTIPLIER = 5.0
RECOVERY_MEMORY_MULTIPLIER = 1.0


def validation_memory(validation, physical_size, dt):
    """Median first 1/e ACF crossing in physical validation coordinates."""
    values = np.asarray(validation)[:, :, :physical_size]
    if dt <= 0 or not np.isfinite(dt) or values.shape[1] < 3:
        raise ValueError("Need positive dt and at least three validation samples")
    if not np.isfinite(values).all():
        raise ValueError("Validation states must be finite")
    lags = []
    for trajectory in values:
        row = []
        for series in trajectory.T:
            x = series - series.mean()
            energy = np.dot(x, x)
            if energy <= 0:
                raise ValueError("Validation variables need positive variance")
            acf = np.correlate(x, x, mode="full")[len(x) - 1 :] / energy
            crossed = np.flatnonzero(acf < np.exp(-1))
            if not len(crossed):
                raise ValueError("Validation sequence does not resolve decorrelation")
            row.append(int(crossed[0]))
        lags.append(row)
    median_steps = float(np.median(lags))
    return {
        "memory_steps": median_steps,
        "memory_time": median_steps * dt,
        "acf_1e_crossing_steps_by_trajectory_state": lags,
    }


def episode_policy(
    validation,
    physical_size,
    dt,
    confirmation_multiplier=CONFIRMATION_MEMORY_MULTIPLIER,
    recovery_multiplier=RECOVERY_MEMORY_MULTIPLIER,
):
    """Same validation-memory rule for every system; no forcing input."""
    if not 0 < recovery_multiplier < confirmation_multiplier:
        raise ValueError(
            "Confirmation multiplier must exceed positive recovery multiplier"
        )
    memory = validation_memory(validation, physical_size, dt)
    window = max(5, int(np.ceil(confirmation_multiplier * memory["memory_steps"])))
    gap = min(window - 1, int(np.floor(recovery_multiplier * memory["memory_steps"])))
    return {
        "version": 2,
        "dt": dt,
        "local_error_threshold": 0.5,
        "window_steps": window,
        "window_time": window * dt,
        "bad_fraction": 0.8,
        "max_good_gap_steps": gap,
        "time_scale_source": "Median validation first ACF<1/e crossing",
        **memory,
        "confirmation_memory_multiplier": confirmation_multiplier,
        "recovery_memory_multiplier": recovery_multiplier,
        "joint_rule": "Any physical variable fails value, slope or curvature",
        "confirmation_delay_time": (window - 1) * dt,
        "interpretation": (
            "Retrospective onset requires a complete following window. "
            "Recovery remains possible."
        ),
        "development_scope": (
            "Shared multiplier calibrated on four annotated model cases of one "
            "Van der Pol trajectory; autonomous systems have no visual labels."
        ),
    }


def detect_episodes(error, policy, numeric_failure=None):
    """Detect disagreement episodes with a complete confirmation window.

    A candidate starts on a bad sample. At least bad_fraction of the next window
    must be bad, with no good run longer than max_good_gap_steps. A longer good
    run separates episodes. Status is retrospective: 0 tracking, 1 temporary or
    unconfirmed mismatch, 2 sustained breakdown. Numeric failure is immediate.
    """
    error = np.asarray(error)
    if error.ndim != 3:
        raise ValueError("Errors must have origin, lead, physical-variable axes")
    n, horizon, _ = error.shape
    window = policy["window_steps"]
    gap = policy["max_good_gap_steps"]
    fraction = policy["bad_fraction"]
    dt = policy["dt"]
    if window < 1 or not 0 <= gap < window or not 0 < fraction <= 1 or dt <= 0:
        raise ValueError("Invalid episode policy")
    if numeric_failure is None:
        numeric_failure = ~np.isfinite(error).all(axis=2)
    numeric_failure = np.asarray(numeric_failure, dtype=bool)
    if numeric_failure.shape != (n, horizon):
        raise ValueError("Numerical failure mask must match origin and lead")
    bad = (error > policy["local_error_threshold"]).any(axis=2)
    status = bad.astype(np.uint8)
    episodes, onsets, confirmations, censored = [], [], [], []
    for origin in range(n):
        failures = np.flatnonzero(numeric_failure[origin])
        stop = int(failures[0]) if len(failures) else horizon
        b = bad[origin, :stop]
        candidates = np.zeros(stop, dtype=bool)
        if stop >= window:
            starts = np.arange(stop - window + 1)
            counts = np.r_[0, np.cumsum(b)]
            good_run = np.lib.stride_tricks.sliding_window_view(~b, gap + 1).all(-1)
            runs = np.r_[0, np.cumsum(good_run)]
            eligible = (runs[starts + window - gap] - runs[starts]) == 0
            candidates[starts] = (
                b[starts]
                & eligible
                & ((counts[starts + window] - counts[starts]) >= fraction * window)
            )
        bad_indices = np.flatnonzero(b)
        groups = np.split(
            bad_indices, np.flatnonzero(np.diff(bad_indices) > gap + 1) + 1
        )
        records = []
        for indices in groups:
            if not len(indices):
                continue
            first, last = int(indices[0]), int(indices[-1])
            qualifying = np.flatnonzero(candidates[first : last + 1])
            sustained = bool(len(qualifying))
            onset = first + int(qualifying[0]) if sustained else first
            recovery_confirm = last + gap + 1
            recovered = recovery_confirm < stop
            end = last + 1 if recovered else stop
            status[origin, first:end] = 1
            if sustained:
                status[origin, onset:end] = 2
            records.append(
                {
                    "kind": "sustained" if sustained else "temporary_or_unconfirmed",
                    "start_time": (first + 1) * dt,
                    "breakdown_onset_time": (onset + 1) * dt if sustained else None,
                    "confirmation_time": (onset + window) * dt if sustained else None,
                    "last_bad_time": (last + 1) * dt,
                    "recovery_onset_time": (last + 2) * dt if recovered else None,
                    "recovery_confirmation_time": (recovery_confirm + 1) * dt
                    if recovered
                    else None,
                    "recovery_censored": not recovered,
                }
            )
        if stop < horizon:
            status[origin, stop:] = 2
            records.append(
                {
                    "kind": "numerical_failure",
                    "start_time": (stop + 1) * dt,
                    "breakdown_onset_time": (stop + 1) * dt,
                    "confirmation_time": (stop + 1) * dt,
                    "last_bad_time": horizon * dt,
                    "recovery_onset_time": None,
                    "recovery_confirmation_time": None,
                    "recovery_censored": True,
                }
            )
        sustained = [e for e in records if e["breakdown_onset_time"] is not None]
        episodes.append(records)
        onsets.append(sustained[0]["breakdown_onset_time"] if sustained else None)
        confirmations.append(sustained[0]["confirmation_time"] if sustained else None)
        censored.append(not sustained)
    capped = [horizon * dt if t is None else t for t in onsets]
    report = {
        "policy": policy,
        "horizon_time": horizon * dt,
        "onset_time_by_origin": onsets,
        "confirmation_time_by_origin": confirmations,
        "censored_by_origin": censored,
        "median_onset_capped": float(np.median(capped)),
        "censored_fraction": float(np.mean(censored)),
        "episodes_by_origin": episodes,
        "status_codes": {
            "tracking": 0,
            "temporary_or_unconfirmed_mismatch": 1,
            "sustained_breakdown": 2,
        },
        "status_note": "Retrospective full-state labels; separate confirmation times.",
    }
    return report, status
