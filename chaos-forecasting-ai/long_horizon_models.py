"""Polynomial dynamics and batched linear baselines for long free rollouts."""

from itertools import combinations_with_replacement

import numpy as np


class CubicDynamics:
    """Learn polynomial derivatives; equation coefficients are not supplied."""

    def __init__(self, physical_size, dt, forcing_frequency=None, degree=3):
        self.physical_size, self.dt = physical_size, dt
        self.forcing_frequency = forcing_frequency
        self.degree = degree
        self.terms = [()] + [
            term
            for degree in range(1, degree + 1)
            for term in combinations_with_replacement(range(physical_size), degree)
        ]

    def features(self, state):
        columns = [
            np.prod(state[..., term], axis=-1) if term else np.ones(state.shape[:-1])
            for term in self.terms
        ]
        if self.forcing_frequency is not None:
            columns.extend([state[..., -2], state[..., -1]])
        return np.stack(columns, axis=-1)

    def fit(self, trajectories):
        features, derivatives = [], []
        for state in trajectories:
            derivative = (
                state[:-4, : self.physical_size]
                - 8 * state[1:-3, : self.physical_size]
                + 8 * state[3:-1, : self.physical_size]
                - state[4:, : self.physical_size]
            ) / (12 * self.dt)
            features.append(self.features(state[2:-2]))
            derivatives.append(derivative)
        self.coefficients = np.linalg.lstsq(
            np.concatenate(features), np.concatenate(derivatives), rcond=None
        )[0]
        return self

    def rollout_batch(self, contexts, horizon):
        state = np.asarray(contexts)[:, -1].copy()
        active = np.ones(len(state), dtype=bool)
        output = np.full((len(state), horizon, state.shape[1]), np.nan)

        def derivative(value):
            physical = self.features(value) @ self.coefficients
            if self.forcing_frequency is None:
                return physical
            return np.column_stack(
                (
                    physical,
                    self.forcing_frequency * value[:, -1],
                    -self.forcing_frequency * value[:, -2],
                )
            )

        step = self.dt / 4
        for lead in range(horizon):
            # Catch unstable numerical trajectories, preserve them as explicit failures.
            for _ in range(4):
                indices = np.flatnonzero(active)
                if not len(indices):
                    break
                value = state[indices]
                with np.errstate(over="ignore", invalid="ignore"):
                    a = derivative(value)
                    b = derivative(value + step * a / 2)
                    c = derivative(value + step * b / 2)
                    d = derivative(value + step * c)
                    next_state = value + step * (a + 2 * b + 2 * c + d) / 6
                finite = np.isfinite(next_state).all(axis=1) & (
                    np.abs(next_state).max(axis=1) < 1e8
                )
                state[indices[finite]] = next_state[finite]
                active[indices[~finite]] = False
            output[active, lead] = state[active]
        return output


class BatchedRidge:
    def __init__(self, model):
        self.model = model

    def rollout_batch(self, contexts, horizon):
        history = np.asarray(contexts)[:, -self.model.window :].copy()
        output = []
        for _ in range(horizon):
            features = np.column_stack(
                (history.reshape(len(history), -1), np.ones(len(history)))
            )
            value = features @ self.model.weights
            output.append(value)
            history = np.concatenate((history[:, 1:], value[:, None]), axis=1)
        return np.stack(output, axis=1)
