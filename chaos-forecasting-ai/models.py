"""Small dependency-free forecasting models."""

import numpy as np


class RidgeForecaster:
    """Recursive multi-output Ridge regression over flattened state windows."""

    def __init__(self, window: int = 20, regularization: float = 1e-4) -> None:
        self.window = window
        self.regularization = regularization
        self.weights: np.ndarray | None = None

    def fit(self, states: np.ndarray) -> "RidgeForecaster":
        """Fit one-step state prediction on a contiguous training series."""
        return self.fit_trajectories([states])

    def fit_trajectories(self, trajectories) -> "RidgeForecaster":
        """Pool valid windows within trajectories, never across boundaries."""
        windows = [make_windows(states, self.window) for states in trajectories]
        if not windows:
            raise ValueError("Need at least one training trajectory")
        features = np.concatenate([features for features, _ in windows])
        targets = np.concatenate([targets for _, targets in windows])
        design = np.column_stack((features, np.ones(len(features))))
        penalty = self.regularization * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        self.weights = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ targets,
        )
        return self

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        """Predict one state from the most recent window."""
        if self.weights is None:
            raise RuntimeError("fit must be called before predict_next")
        window = np.asarray(history[-self.window :], dtype=float).reshape(1, -1)
        return (np.column_stack((window, np.ones(1))) @ self.weights).ravel()

    def rollout(self, initial: np.ndarray, horizon: int) -> np.ndarray:
        """Generate a recursive forecast."""
        history = [row.copy() for row in np.asarray(initial, dtype=float)]
        for _ in range(horizon):
            history.append(self.predict_next(np.asarray(history)))
        return np.asarray(history[-horizon:])


def make_windows(states: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Create flattened lag windows and one-step targets."""
    if len(states) <= window:
        raise ValueError("states must contain more samples than window")
    features = np.asarray(
        [
            states[index - window : index].reshape(-1)
            for index in range(window, len(states))
        ]
    )
    return features, states[window:]


class PolynomialDynamics:
    """Fit cubic physical-state derivatives and integrate with known forcing.

    This model has an explicit polynomial dynamics prior. Coefficients are
    learned from training data; governing-equation coefficients are not supplied.
    """

    @staticmethod
    def features(state):
        x, v, sine, cosine = np.asarray(state).T
        return np.stack(
            (
                np.ones_like(x),
                x,
                v,
                x * x,
                x * v,
                v * v,
                x**3,
                x * x * v,
                x * v * v,
                v**3,
                sine,
                cosine,
            ),
            axis=-1,
        )

    def fit(self, train, dt=0.05):
        self.dt = dt
        # Fourth-order central differences; every stencil stays inside training.
        derivative = (
            train[:-4, :2] - 8 * train[1:-3, :2] + 8 * train[3:-1, :2] - train[4:, :2]
        ) / (12 * dt)
        features = self.features(train[2:-2])
        self.coefficients = np.linalg.lstsq(features, derivative, rcond=None)[0]
        return self

    def rollout_batch(self, contexts, horizon):
        from data import PARAMETERS

        states = np.asarray(contexts)[:, -1].copy()
        frequency = PARAMETERS["forcing_frequency"]

        def derivative(value):
            physical = self.features(value) @ self.coefficients
            return np.column_stack(
                (physical, frequency * value[:, 3], -frequency * value[:, 2])
            )

        result = []
        # Substeps keep integration error below the learned derivative error.
        step = self.dt / 4
        for _ in range(horizon):
            for _ in range(4):
                k1 = derivative(states)
                k2 = derivative(states + step * k1 / 2)
                k3 = derivative(states + step * k2 / 2)
                k4 = derivative(states + step * k3)
                states = states + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            result.append(states.copy())
        return np.stack(result, axis=1)

    def rollout(self, context, horizon):
        return self.rollout_batch(np.asarray(context)[None], horizon)[0]
