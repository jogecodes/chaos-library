"""Malkus-Howard waterwheel models.

The reduced model is the Lorenz system after nondimensionalisation. The
discrete model is intentionally simple: each bucket is a point mass with
exponential leakage and constant inflow into the top bucket.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


def mass_moments(trajectory: np.ndarray, bucket_count: int) -> np.ndarray:
    """Return physical first Fourier mass moments for a bucket trajectory.

    The returned columns are ``theta``, ``omega``, ``C1``, and ``S1``.
    ``C1`` and ``S1`` use the instantaneous bucket angles, including theta.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != bucket_count + 2:
        raise ValueError("trajectory shape does not match bucket_count")
    theta = trajectory[:, 0]
    omega = trajectory[:, 1]
    relative_angles = 2.0 * np.pi * np.arange(bucket_count) / bucket_count
    angles = theta[:, None] + relative_angles[None, :]
    masses = trajectory[:, 2:]
    cosine_moment = np.sum(masses * np.cos(angles), axis=1)
    sine_moment = np.sum(masses * np.sin(angles), axis=1)
    return np.column_stack((theta, omega, cosine_moment, sine_moment))


@dataclass(frozen=True)
class ReducedWaterwheel:
    """Three-mode Lorenz reduction with physical parameter names."""

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    def derivative(self, time: float, state: np.ndarray) -> np.ndarray:
        """Return the reduced waterwheel derivative."""
        x, y, z = state
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ])

    def integrate(self, initial_state=(1.0, 1.0, 1.0), time_span=(0.0, 40.0), dt=0.01):
        """Integrate the reduced model with explicit solver tolerances."""
        times = np.arange(time_span[0], time_span[1] + dt / 2.0, dt)
        result = solve_ivp(
            self.derivative,
            time_span,
            np.asarray(initial_state, dtype=float),
            t_eval=times,
            rtol=1e-9,
            atol=1e-11,
        )
        if not result.success:
            raise RuntimeError(f"Integration failed: {result.message}")
        return result.t, result.y.T


@dataclass
class DiscreteWaterwheel:
    """Bucket waterwheel with smooth top inflow, leakage, and viscous drag."""

    bucket_count: int = 24
    radius: float = 1.0
    inertia: float = 1.0
    drag: float = 0.8
    leak_rate: float = 0.1
    inflow: float = 1.0
    gravity: float = 9.81
    inflow_concentration: float = 0.5

    def derivative(self, _time: float, state: np.ndarray) -> np.ndarray:
        """Return angular acceleration and bucket mass derivatives."""
        theta, omega = state[:2]
        masses = state[2:]
        angles = theta + 2.0 * np.pi * np.arange(self.bucket_count) / self.bucket_count
        weights = np.exp(self.inflow_concentration * np.cos(angles))
        inflow_profile = self.inflow * weights / weights.sum()
        dm = inflow_profile - self.leak_rate * masses
        torque = self.gravity * self.radius * np.sum(masses * np.sin(angles))
        domega = (torque - self.drag * omega) / self.inertia
        return np.concatenate(([omega, domega], dm))

    def integrate(self, initial_state=None, time_span=(0.0, 100.0), dt=0.02):
        """Integrate the bucket model with bounded non-negative masses."""
        if initial_state is None:
            initial_state = np.zeros(self.bucket_count + 2)
            initial_state[0] = 0.1
        initial_state = np.asarray(initial_state, dtype=float)
        if initial_state.shape != (self.bucket_count + 2,):
            raise ValueError(
                "initial_state must contain theta, omega, and bucket masses"
            )
        times = np.arange(time_span[0], time_span[1] + dt / 2.0, dt)
        result = solve_ivp(
            self.derivative,
            time_span,
            initial_state,
            t_eval=times,
            rtol=1e-8,
            atol=1e-10,
        )
        if not result.success:
            raise RuntimeError(f"Integration failed: {result.message}")
        trajectory = result.y.T
        trajectory[:, 2:] = np.maximum(trajectory[:, 2:], 0.0)
        return result.t, trajectory
