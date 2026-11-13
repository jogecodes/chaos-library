"""Numerical checks for the improved forecasting and calibration protocol."""

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chaos-forecasting-ai"))
evaluation = importlib.import_module("improved_evaluation")
models = importlib.import_module("models")
metrics = importlib.import_module("metrics")


def test_constant_forecast_has_finite_warning_free_coherence():
    truth = np.arange(10, dtype=float)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert metrics.coherence_length(truth, np.ones_like(truth)) == 2
        assert evaluation.safe_correlation(truth[:, 0], np.ones(10)) == 0


def test_conformal_rank_and_insufficient_samples():
    assert evaluation.conformal_quantile(np.arange(1, 10), 0.1) == 9
    assert np.isinf(evaluation.conformal_quantile(np.arange(8), 0.1))
    with pytest.raises(ValueError):
        evaluation.conformal_quantile(np.arange(9), 1.0)


def test_perfect_predictive_distribution_has_zero_proper_scores():
    truth = np.ones((20, 5, 4))
    result, detail = evaluation.calibrated_diagnostics(
        truth, truth, truth[:3], truth[:3], np.ones(4)
    )
    assert result["crps_position"] == 0
    assert result["brier_x_positive"] == 0
    assert result["position_coverage"] == 1
    assert np.array_equal(detail["lower"], truth[:3, :, :2])


def test_polynomial_derivative_fit_recovers_harmonic_motion():
    time = np.arange(2000) * 0.05
    states = np.column_stack(
        (np.sin(time), np.cos(time), np.sin(1.68 * time), np.cos(1.68 * time))
    )
    model = models.PolynomialDynamics().fit(states[:1600])
    forecast = model.rollout(states[1600:1624], 180)
    assert np.max(np.abs(forecast[:, :2] - states[1624:1804, :2])) < 1e-4


def test_pinn_includes_kinematic_equation():
    torch = pytest.importorskip("torch")
    improved = importlib.import_module("improved_models")

    class Analytic(torch.nn.Module):
        def forward(self, time):
            return torch.cat((time**2, 2 * time), dim=1)

    time = torch.linspace(0, 2, 10)[:, None].requires_grad_()
    _, kinematic, dynamic = improved.pinn_residuals(Analytic(), time)
    assert torch.max(torch.abs(kinematic)).item() == 0
    assert torch.isfinite(dynamic).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_improved_transformer_and_pinn_smoke(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    improved = importlib.import_module("improved_models")
    time = np.arange(512) * 0.05
    states = np.column_stack(
        (np.sin(time), np.cos(time), np.sin(1.68 * time), np.cos(1.68 * time))
    )
    model, history = improved.fit_transformer(states, epochs=1, device=device)
    np.testing.assert_allclose(model.model.mean.cpu(), states[:409].mean(0), atol=1e-6)
    forecast = model.rollout(states[-24:], 8)
    assert np.isfinite(forecast).all()
    expected_phase = (512 + np.arange(8)) * 0.05 * 1.68
    np.testing.assert_allclose(forecast[:, 2], np.sin(expected_phase), atol=2e-6)
    assert history["best_epoch"] == 1
    _, prediction, report = improved.fit_pinn(time, states, epochs=2, device=device)
    assert np.isfinite(prediction).all()
    assert len(report["interpolation_rmse_x_v"]) == 2
    assert report["validation_samples"] == 100
    assert report["interpolation_samples"] == 200
    # Test and future labels must never affect fitting or checkpoint selection.
    changed = states.copy()
    held_out = (np.arange(len(time)) % 2 == 1) | (time > 20)
    changed[held_out, :2] += 100
    _, prediction_changed, _ = improved.fit_pinn(time, changed, epochs=2, device=device)
    np.testing.assert_allclose(prediction_changed, prediction, atol=1e-6)


@pytest.mark.parametrize(
    "probability,outcome", [(np.nan, 1), (0.5, np.inf), (0.5, 0.2)]
)
def test_brier_rejects_invalid_values(probability, outcome):
    with pytest.raises(ValueError):
        metrics.brier_score(np.array([probability]), np.array([outcome]))


def test_legacy_conformal_uses_last_valid_origin():
    uncertainty = importlib.import_module("uncertainty")
    train = np.sin(np.arange(30))[:, None]
    radius = uncertainty.conformal_radius(
        train, train[:3], horizon=1, window=2, alpha=0.5
    )
    assert radius.shape == (1, 1)
    assert np.isfinite(radius).all()


def test_rollout_grows_short_context_to_training_window():
    torch = pytest.importorskip("torch")
    improved = importlib.import_module("improved_models")

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lengths = []

        def forward(self, history):
            self.lengths.append(history.shape[1])
            return history[:, -1]

    model = Recorder()
    forecast = improved.NeuralForecaster(model, "cpu").rollout(np.ones((20, 4)), 6)
    assert model.lengths == [20, 21, 22, 23, 24, 24]
    assert forecast.shape == (6, 4)
