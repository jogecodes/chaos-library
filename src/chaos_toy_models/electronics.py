"""Toy electronic circuits."""

from collections.abc import Iterable

import numpy as np


def chua(
    _time: float,
    state: Iterable[float],
    alpha: float = 15.6,
    beta: float = 28.0,
    m0: float = -1.143,
    m1: float = -0.714,
) -> np.ndarray:
    """Evaluate the dimensionless Chua circuit with a piecewise-linear diode."""
    x, y, z = state
    diode = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))
    return np.array([alpha * (y - x - diode), x - y + z, -beta * y])
