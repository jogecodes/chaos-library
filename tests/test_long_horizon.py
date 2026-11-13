"""Check autonomous-state learning, known forcing, and long-horizon baselines."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chaos-forecasting-ai"))


@pytest.mark.parametrize("family", ["transformer", "gru", "mlp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_autonomous_models_and_training(family, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    training = importlib.import_module("trajectory_training")
    t = np.arange(96) * 0.01
    states = np.column_stack((np.sin(t), np.cos(t), np.sin(2 * t)))
    model, report = training.fit_trajectories(
        [states],
        states[:24][None],
        states[24:32][None],
        family=family,
        phase_increment=None,
        device=device,
        updates=2,
        batch_size=4,
    )
    forecast = model.rollout(states[-24:], 10)
    assert forecast.shape == (10, 3)
    assert np.isfinite(forecast).all()
    assert report["physical_size"] == 3


@pytest.mark.parametrize("family", ["transformer", "gru", "mlp"])
def test_forcing_phase_is_exact(family):
    torch = pytest.importorskip("torch")
    models = importlib.import_module("improved_models")
    model = models.make_residual_model(
        family, np.zeros(4), np.ones(4), phase_increment=0.084
    )
    history = torch.zeros((2, 24, 4))
    history[:, :, 3] = 1
    value = model(history).detach().numpy()
    np.testing.assert_allclose(value[:, 2], np.sin(0.084), atol=1e-7)
    np.testing.assert_allclose(value[:, 3], np.cos(0.084), atol=1e-7)


def test_cubic_dynamics_keeps_episode_derivative_stencils_separate():
    models = importlib.import_module("long_horizon_models")
    t = np.arange(1200) * 0.05
    trajectories = [np.column_stack((np.sin(t + p), np.cos(t + p))) for p in [0, 1]]
    model = models.CubicDynamics(2, 0.05).fit(trajectories)
    forecast = model.rollout_batch(np.asarray([trajectories[0][900:924]]), 180)
    np.testing.assert_allclose(forecast[0], trajectories[0][924:1104], atol=1e-4)


def test_batched_ridge_matches_serial_rollout():
    models = importlib.import_module("models")
    long = importlib.import_module("long_horizon_models")
    t = np.arange(500) * 0.05
    state = np.column_stack((np.sin(t), np.cos(t)))
    fitted = models.RidgeForecaster().fit(state[:400])
    contexts = np.asarray([state[400:424], state[425:449]])
    actual = long.BatchedRidge(fitted).rollout_batch(contexts, 50)
    expected = np.asarray([fitted.rollout(c, 50) for c in contexts])
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_long_metrics_distinguish_no_failure_from_horizon_limit():
    metrics = importlib.import_module("long_horizon_metrics")
    time = np.arange(100) * 0.05
    truth = np.column_stack((np.sin(time), np.cos(time)))[None]
    length, censored, survival, _ = metrics.coherence_curves(truth, truth, np.ones(2))
    assert censored.all()
    assert (length == 100).all()
    assert (survival == 1).all()


def test_numerical_failure_is_not_silently_excluded():
    metrics = importlib.import_module("long_horizon_metrics")
    time = np.arange(100) * 0.05
    truth = np.column_stack((np.sin(time), np.cos(time)))[None]
    prediction = truth.copy()
    prediction[:, 30] = np.nan
    result, arrays = metrics.evaluate_long_run(
        truth, prediction[None], np.ones(2), 2, 0.05, [1, 5]
    )
    assert result["numerical_failure_fraction"] == 1
    assert result["point_by_cutoff"]["5"]["rmse_by_state"] is None
    assert np.isnan(arrays["members"][:, :, 30:]).all()
    assert result["mean_normalized_member_wasserstein"] is None


def test_distribution_scores_use_individual_members_not_the_mean():
    metrics = importlib.import_module("long_horizon_metrics")
    truth = np.tile([-1.0, 1.0], 10)[None, :, None]
    members = np.stack((truth, -truth))
    result, _ = metrics.evaluate_long_run(truth, members, np.ones(1), 1, 1, [20])
    assert result["mean_normalized_member_wasserstein"] == 0
    assert result["point_by_cutoff"]["20"]["rmse_by_state"] == [1.0]


def test_rolling_loss_is_backdated_and_forgets_early_accuracy():
    metrics = importlib.import_module("long_horizon_metrics")
    truth = np.zeros((1, 100, 2))
    forecast = truth.copy()
    forecast[:, 80:, 0] = 1
    settings = {"window_steps": 4, "persistence_steps": 3, "threshold": 0.5}
    length, censored, survival, rolling = metrics.coherence_curves(
        truth, forecast, np.ones(2), settings
    )
    assert length[0, 0] == 82  # first .707 crossing, confirmed two steps later
    assert not censored[0, 0]
    assert censored[0, 1]
    assert survival[80, 0] == 1 and survival[81, 0] == 0
    assert rolling[0, 83, 0] == 1


def test_short_excursion_and_unconfirmed_tail_remain_censored():
    metrics = importlib.import_module("long_horizon_metrics")
    truth = np.zeros((1, 20, 1))
    forecast = truth.copy()
    forecast[:, 5] = 1
    forecast[:, -2:] = 1
    settings = {"window_steps": 1, "persistence_steps": 3, "threshold": 0.5}
    _, censored, _, _ = metrics.coherence_curves(truth, forecast, np.ones(1), settings)
    assert censored.all()


def test_nonfinite_failure_is_immediate_and_startup_is_checked():
    metrics = importlib.import_module("long_horizon_metrics")
    truth = np.zeros((1, 20, 2))
    forecast = truth.copy()
    forecast[:, 0, 0] = np.nan
    forecast[:, :3, 1] = 1
    settings = {"window_steps": 5, "persistence_steps": 3, "threshold": 0.5}
    length, censored, _, _ = metrics.coherence_curves(
        truth, forecast, np.ones(2), settings
    )
    assert (length == 1).all() and not censored.any()


def test_validation_window_uses_only_physical_states():
    metrics = importlib.import_module("long_horizon_metrics")
    t = np.arange(400) * 0.05
    validation = np.stack(
        [
            np.column_stack([np.sin(t + p), np.cos(t + p), np.zeros_like(t)])
            for p in [0, 1]
        ]
    )
    settings = metrics.validation_coherence_settings(validation, 2, 0.05)
    assert settings["window_steps"] >= 2
    assert settings["threshold"] == 0.5
    assert (
        settings["confirmation_delay_time"]
        == (settings["persistence_steps"] - 1) * 0.05
    )


def test_first_breach_and_later_recovery_are_reported_separately():
    metrics = importlib.import_module("long_horizon_metrics")
    truth = np.tile([-1.0, 1.0], 10)[None, :, None]
    forecast = truth.copy()
    forecast[:, 4:7] += 1
    result, arrays = metrics.evaluate_long_run(
        truth,
        forecast[None],
        np.ones(1),
        1,
        0.1,
        [1],
        {"window_steps": 1, "persistence_steps": 2, "threshold": 0.5},
    )
    assert result["coherence_time_by_origin_state"] == [[0.5]]
    assert result["first_breach_recovery_time_by_origin_state"] == [[0.8]]
    assert not arrays["within_tolerance_by_origin_state"][0, 4:7, 0].any()
    assert arrays["within_tolerance_by_origin_state"][0, 7:, 0].all()
    assert not arrays["coherence_survival"][7:, 0].any()


def test_causal_derivatives_reproduce_quadratic_and_ignore_future():
    metrics = importlib.import_module("long_horizon_metrics")
    t = np.arange(60) * 0.02
    values = (3 * t * t + 2 * t + 7)[None, :, None]
    derivatives = metrics.causal_derivatives(values, 0.02, 5)
    np.testing.assert_allclose(derivatives[0, 4:, 0, 0], 6 * t[4:] + 2, atol=1e-10)
    np.testing.assert_allclose(derivatives[0, 4:, 0, 1], 6, atol=1e-9)
    changed = values.copy()
    changed[:, 30:] += 100
    np.testing.assert_allclose(
        metrics.causal_derivatives(changed, 0.02, 5)[:, :30], derivatives[:, :30]
    )


@pytest.mark.parametrize("power,feature", [(1, 1), (2, 2)])
def test_value_crossing_rejected_by_slope_or_curvature(power, feature):
    metrics = importlib.import_module("long_horizon_metrics")
    t = np.arange(-30, 31) * 0.05
    truth = (t**power)[None, :, None]
    forecast = -truth
    settings = {
        "window_steps": 1,
        "persistence_steps": 2,
        "threshold": 0.5,
        "derivative_window_steps": 5,
        "dt": 0.05,
        "derivative_scales_by_state": [[1.0, 1.0]],
    }
    combined, components = metrics.coherence_errors(
        truth, forecast, np.ones(1), settings
    )
    assert components[0, 30, 0, 0] == 0
    assert components[0, 30, 0, feature] > 0.5
    assert combined[0, 30, 0] > 0.5


def test_observed_context_scores_derivatives_at_first_lead():
    metrics = importlib.import_module("long_horizon_metrics")
    truth = np.sin(np.arange(20) * 0.1)[None, :, None]
    context = np.sin(np.arange(-4, 0) * 0.1)[None, :, None]
    settings = {
        "window_steps": 2,
        "persistence_steps": 2,
        "threshold": 0.5,
        "derivative_window_steps": 5,
        "dt": 0.1,
        "derivative_scales_by_state": [[1.0, 1.0]],
    }
    combined, components = metrics.coherence_errors(
        truth, truth.copy(), np.ones(1), settings, context
    )
    assert np.isfinite(components).all()
    assert (combined == 0).all()


def test_derivative_scales_come_from_training_trajectories():
    metrics = importlib.import_module("long_horizon_metrics")
    t = np.arange(300) * 0.05
    train = np.stack([np.column_stack([np.sin(t + p), np.cos(t + p)]) for p in [0, 1]])
    validation = train * 7
    settings = metrics.validation_coherence_settings(
        validation, 2, 0.05, training=train
    )
    w = settings["derivative_window_steps"]
    expected = metrics.causal_derivatives(train, 0.05, w)[:, w - 1 :].std(axis=(0, 1))
    np.testing.assert_allclose(settings["derivative_scales_by_state"], expected)
