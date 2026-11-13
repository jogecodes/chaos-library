"""Discrete population dynamics."""

import numpy as np


def ricker(population: float, growth: float = 4.0) -> float:
    """Return one step of the overcompensating Ricker population map."""
    if population < 0.0:
        raise ValueError("population must be non-negative")
    return population * np.exp(growth * (1.0 - population))


def forced_ricker(
    state: np.ndarray,
    growth: float = 3.2,
    forcing: float = 0.5,
    phase_step: float = 0.7,
) -> np.ndarray:
    """Return one step of a seasonally forced Ricker population map."""
    population, phase = np.asarray(state, dtype=float)
    if population < 0.0:
        raise ValueError("population must be non-negative")
    next_population = population * np.exp(
        growth * (1.0 - population) + forcing * np.sin(phase)
    )
    return np.array([next_population, (phase + phase_step) % (2.0 * np.pi)])
