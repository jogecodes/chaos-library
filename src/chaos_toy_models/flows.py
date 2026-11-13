"""Continuous-time chaotic flows."""

from collections.abc import Callable, Iterable

import numpy as np
from scipy.integrate import solve_ivp


def lorenz(
    _time: float,
    state: Iterable[float],
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """Evaluate the Lorenz equations."""
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def rossler(
    _time: float,
    state: Iterable[float],
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
) -> np.ndarray:
    """Evaluate the Rössler equations."""
    x, y, z = state
    return np.array([-y - z, x + a * y, b + z * (x - c)])


def duffing(
    _time: float,
    state: Iterable[float],
    delta: float = 0.2,
    alpha: float = -1.0,
    beta: float = 1.0,
    gamma: float = 0.3,
    omega: float = 1.2,
) -> np.ndarray:
    """Evaluate the periodically forced Duffing oscillator."""
    position, velocity = state
    return np.array([
        velocity,
        -delta * velocity - alpha * position - beta * position**3
        + gamma * np.cos(omega * _time),
    ])


def forced_van_der_pol(
    time: float,
    state: Iterable[float],
    mu: float = 3.0,
    amplitude: float = 4.4,
    forcing_frequency: float = 1.68,
) -> np.ndarray:
    """Evaluate the externally forced Van der Pol oscillator."""
    position, velocity = state
    forcing = amplitude * np.cos(forcing_frequency * time)
    return np.array([
        velocity,
        mu * (1.0 - position**2) * velocity - position + forcing,
    ])


def integrate_flow(
    function: Callable,
    initial_state: Iterable[float],
    time_span: tuple[float, float],
    *,
    dt: float = 0.01,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    **parameters: float,
):
    """Integrate a flow on a uniform output grid with explicit tolerances."""
    if dt <= 0 or time_span[1] <= time_span[0]:
        raise ValueError("dt must be positive and time_span must be increasing")
    times = np.arange(time_span[0], time_span[1] + dt / 2.0, dt)
    result = solve_ivp(
        lambda time, state: function(time, state, **parameters),
        time_span,
        np.asarray(initial_state, dtype=float),
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"Integration failed: {result.message}")
    return result.t, result.y.T
