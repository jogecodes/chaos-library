"""Generate deterministic benchmark data."""

from pathlib import Path

import numpy as np

from chaos_toy_models.flows import forced_van_der_pol, integrate_flow

PARAMETERS = {
    "mu": 3.0,
    "amplitude": 4.4,
    "forcing_frequency": 1.68,
}


def generate_series(
    *,
    duration: float = 180.0,
    dt: float = 0.05,
    initial_state: tuple[float, float] = (0.1, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return time and the augmented state ``(x, v, sin(phi), cos(phi))``."""
    time, states = integrate_flow(
        forced_van_der_pol,
        initial_state,
        (0.0, duration),
        dt=dt,
        **PARAMETERS,
    )
    phase = PARAMETERS["forcing_frequency"] * time
    return time, np.column_stack((states, np.sin(phase), np.cos(phase)))


def save_series(path: Path, time: np.ndarray, states: np.ndarray) -> None:
    """Save a generated series as a compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, time=time, states=states)
