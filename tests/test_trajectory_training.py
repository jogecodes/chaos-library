"""Guard against false transition windows when scaling simulated data."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chaos-forecasting-ai"))


def test_ridge_pools_windows_within_trajectories():
    models = importlib.import_module("models")
    trajectories = [np.full((32, 1), value) for value in [0.1, 2.0, -3.0]]
    model = models.RidgeForecaster(window=4, regularization=1e-8).fit_trajectories(
        trajectories
    )
    for trajectory in trajectories:
        np.testing.assert_allclose(
            model.rollout(trajectory[-4:], 10), trajectory[:10], atol=1e-6
        )


def test_pool_excludes_boundary_transitions():
    pytest.importorskip("torch")
    training = importlib.import_module("trajectory_training")
    trajectories = [np.full((40, 4), value) for value in [1.0, 100.0, -100.0]]
    values, origins = training.trajectory_pool(trajectories)
    assert len(origins) == 3 * (40 - 24 - 4 + 1)
    for origin in origins:
        assert np.unique(values[origin - 24 : origin + 4]).size == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_multi_trajectory_fit_uses_only_training_normalization(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    training = importlib.import_module("trajectory_training")
    time = np.arange(96) * 0.05
    values = np.column_stack(
        (np.sin(time), np.cos(time), np.sin(1.68 * time), np.cos(1.68 * time))
    )
    train = [values, values * 0.8]
    validation = values + 10
    model, report = training.fit_trajectories(
        train,
        validation[:24][None],
        validation[24:32][None],
        updates=2,
        batch_size=4,
        device=device,
    )
    np.testing.assert_allclose(
        model.model.mean.cpu(), np.concatenate(train).mean(0), atol=1e-6
    )
    assert report["training_samples"] == 192
    assert np.isfinite(model.rollout(values[-24:], 8)).all()
