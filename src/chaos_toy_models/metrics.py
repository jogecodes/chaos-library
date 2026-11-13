"""Numerical diagnostics for maps and flows."""

import numpy as np
from scipy.integrate import solve_ivp


def map_lyapunov(
    map_function,
    derivative_function,
    initial_state: np.ndarray,
    *,
    steps: int = 10_000,
    discard: int = 1_000,
) -> float:
    """Estimate the largest Lyapunov exponent of a differentiable map."""
    state = np.asarray(initial_state, dtype=float)
    vector = np.ones_like(state)
    total = 0.0
    for index in range(steps + discard):
        vector = np.asarray(derivative_function(state), dtype=float) @ vector
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return float("-inf")
        vector /= norm
        state = np.asarray(map_function(state), dtype=float)
        if index >= discard:
            total += np.log(norm)
    return total / steps


def flow_lyapunov(
    function,
    initial_state: np.ndarray,
    time_span: tuple[float, float],
    *,
    dt: float = 0.02,
    discard: float = 5.0,
    jacobian_step: float = 1e-6,
) -> float:
    """Estimate a flow's largest Lyapunov exponent by tangent-vector renormalisation."""
    state = np.asarray(initial_state, dtype=float)
    dimension = len(state)
    vector = np.ones(dimension) / np.sqrt(dimension)
    total = 0.0
    count = 0
    time = time_span[0]
    end = time_span[1]
    while time < end - 1e-12:
        step_end = min(time + dt, end)

        def augmented(current_time, augmented_state):
            current = augmented_state[:dimension]
            tangent = augmented_state[dimension:]
            jacobian = np.column_stack([
                (
                    np.asarray(
                        function(current_time, current + jacobian_step * basis)
                    )
                    - np.asarray(
                        function(current_time, current - jacobian_step * basis)
                    )
                )
                / (2.0 * jacobian_step)
                for basis in np.eye(dimension)
            ])
            return np.concatenate((
                np.asarray(function(current_time, current)),
                jacobian @ tangent,
            ))

        result = solve_ivp(
            augmented,
            (time, step_end),
            np.concatenate((state, vector)),
            rtol=1e-7,
            atol=1e-9,
        )
        if not result.success:
            raise RuntimeError(f"Integration failed: {result.message}")
        state = result.y[:dimension, -1]
        vector = result.y[dimension:, -1]
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return float("-inf")
        vector /= norm
        if step_end > discard:
            total += np.log(norm)
            count += 1
        time = step_end
    if count == 0:
        raise ValueError("time_span must extend beyond discard")
    return total / (count * dt)


def logistic_lyapunov(
    r: float, x0: float = 0.2, steps: int = 100_000, discard: int = 1_000
) -> float:
    """Estimate the logistic-map Lyapunov exponent from its derivative."""
    x = x0
    total = 0.0
    for index in range(steps + discard):
        x = r * x * (1.0 - x)
        if index >= discard:
            total += np.log(abs(r * (1.0 - 2.0 * x)))
    return total / steps


def bifurcation_diagram(
    map_function,
    parameter_values: np.ndarray,
    initial_state: float,
    *,
    parameter_name: str,
    discard: int = 500,
    samples: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect post-transient scalar samples over a parameter sweep."""
    parameters = []
    values = []
    for parameter in parameter_values:
        state = initial_state
        for _ in range(discard):
            state = map_function(state, **{parameter_name: parameter})
        for _ in range(samples):
            state = map_function(state, **{parameter_name: parameter})
            parameters.append(parameter)
            values.append(state)
    return np.asarray(parameters), np.asarray(values)


def poincare_section(
    times: np.ndarray, states: np.ndarray, coordinate: int = 0
) -> np.ndarray:
    """Return upward crossings of coordinate zero, linearly interpolated."""
    values = states[:, coordinate]
    crossings = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
    fractions = -values[crossings] / (values[crossings + 1] - values[crossings])
    return states[crossings] + fractions[:, None] * (
        states[crossings + 1] - states[crossings]
    )


def box_counting_dimension(points: np.ndarray, scales: int = 8) -> float:
    """Estimate a planar box-counting dimension over dyadic scales."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("points must contain at least two two-dimensional samples")
    minimum = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - minimum, np.finfo(float).eps)
    sizes = 2 ** np.arange(1, scales + 1)
    counts = []
    for size in sizes:
        occupied = np.floor((points - minimum) / span * (size - 1)).astype(int)
        counts.append(len(np.unique(occupied, axis=0)))
    slope, _ = np.polyfit(np.log(sizes), np.log(counts), 1)
    return float(slope)
