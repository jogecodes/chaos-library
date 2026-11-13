"""Classical discrete maps."""

from collections.abc import Iterable

import numpy as np


def logistic(x: float, r: float) -> float:
    """Return one step of the logistic map, x[n+1] = r*x[n]*(1-x[n])."""
    return r * x * (1.0 - x)


def henon(state: Iterable[float], a: float = 1.4, b: float = 0.3) -> np.ndarray:
    """Return one step of the Hénon map."""
    x, y = np.asarray(tuple(state), dtype=float)
    return np.array([1.0 - a * x * x + y, b * x])


def ikeda(
    state: Iterable[float],
    u: float = 0.918,
) -> np.ndarray:
    """Return one step of the two-dimensional Ikeda map."""
    x, y = np.asarray(tuple(state), dtype=float)
    theta = 0.4 - 6.0 / (1.0 + x * x + y * y)
    return np.array([1.0 + u * (x * np.cos(theta) - y * np.sin(theta)),
                     u * (x * np.sin(theta) + y * np.cos(theta))])


def iterate_map(
    function,
    initial_state,
    steps: int,
    *,
    discard: int = 0,
) -> np.ndarray:
    """Iterate a map and return states after the discarded transient."""
    if steps < 0 or discard < 0:
        raise ValueError("steps and discard must be non-negative")
    state = np.asarray(initial_state, dtype=float)
    trajectory = []
    for index in range(steps + discard):
        state = np.asarray(function(state), dtype=float)
        if index >= discard:
            trajectory.append(state.copy())
    return np.asarray(trajectory)
