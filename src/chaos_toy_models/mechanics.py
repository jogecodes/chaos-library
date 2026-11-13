"""Mechanical toy systems."""

from collections.abc import Iterable

import numpy as np


def double_pendulum(
    _time: float,
    state: Iterable[float],
    length_1: float = 1.0,
    length_2: float = 1.0,
    mass_1: float = 1.0,
    mass_2: float = 1.0,
    gravity: float = 9.81,
) -> np.ndarray:
    """Evaluate the planar double-pendulum equations.

    Angles are measured from the downward vertical and the state is
    ``(theta_1, theta_2, omega_1, omega_2)``.
    """
    theta_1, theta_2, omega_1, omega_2 = state
    difference = theta_1 - theta_2
    denominator = 2.0 * mass_1 + mass_2 - mass_2 * np.cos(2.0 * difference)
    acceleration_1 = (
        -gravity * (2.0 * mass_1 + mass_2) * np.sin(theta_1)
        - mass_2 * gravity * np.sin(theta_1 - 2.0 * theta_2)
        - 2.0
        * np.sin(difference)
        * mass_2
        * (omega_2**2 * length_2 + omega_1**2 * length_1 * np.cos(difference))
    ) / (length_1 * denominator)
    acceleration_2 = (
        2.0
        * np.sin(difference)
        * (
            omega_1**2 * length_1 * (mass_1 + mass_2)
            + gravity * (mass_1 + mass_2) * np.cos(theta_1)
            + omega_2**2 * length_2 * mass_2 * np.cos(difference)
        )
    ) / (length_2 * denominator)
    return np.array([omega_1, omega_2, acceleration_1, acceleration_2])
