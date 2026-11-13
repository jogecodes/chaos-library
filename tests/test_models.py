import numpy as np

from chaos_toy_models.electronics import chua
from chaos_toy_models.flows import (
    forced_van_der_pol,
    integrate_flow,
    lorenz,
)
from chaos_toy_models.maps import henon, logistic
from chaos_toy_models.mechanics import double_pendulum
from chaos_toy_models.metrics import (
    flow_lyapunov,
    logistic_lyapunov,
    map_lyapunov,
)
from chaos_toy_models.population import forced_ricker, ricker
from chaos_toy_models.wheel import (
    DiscreteWaterwheel,
    ReducedWaterwheel,
    mass_moments,
)


def test_logistic_map_fixed_point_at_zero():
    assert logistic(0.0, 4.0) == 0.0


def test_logistic_lyapunov_at_r_four_is_log_two():
    assert np.isclose(
        logistic_lyapunov(4.0, steps=20_000, discard=1_000), np.log(2), atol=0.02
    )


def test_general_map_lyapunov_matches_logistic_reference():
    exponent = map_lyapunov(
        lambda state: np.array([logistic(state[0], 4.0)]),
        lambda state: np.array([[4.0 * (1.0 - 2.0 * state[0])]]),
        np.array([0.2]),
        steps=10_000,
        discard=500,
    )
    assert np.isclose(exponent, np.log(2), atol=0.03)


def test_henon_returns_two_coordinates():
    assert henon((0.0, 0.0)).shape == (2,)


def test_lorenz_flow_has_expected_shape():
    times, states = integrate_flow(lorenz, (1.0, 1.0, 1.0), (0.0, 1.0), dt=0.02)
    assert states.shape == (len(times), 3)


def test_chua_has_three_state_variables():
    assert chua(0.0, (0.1, 0.0, 0.0)).shape == (3,)


def test_double_pendulum_has_four_state_variables():
    assert double_pendulum(0.0, (0.1, 0.2, 0.0, 0.0)).shape == (4,)


def test_forced_van_der_pol_has_two_state_variables():
    assert forced_van_der_pol(0.0, (0.1, 0.0)).shape == (2,)


def test_forced_van_der_pol_has_positive_lyapunov_estimate():
    exponent = flow_lyapunov(
        forced_van_der_pol,
        (0.1, 0.0),
        (0.0, 150.0),
        dt=0.1,
        discard=50.0,
    )
    assert exponent > 0.05


def test_population_maps_remain_non_negative():
    assert ricker(0.2) >= 0.0
    assert forced_ricker(np.array([0.2, 0.0])).shape == (2,)


def test_ricker_map_has_positive_lyapunov_estimate():
    exponent = map_lyapunov(
        lambda state: np.array([ricker(state[0])]),
        lambda state: np.array(
            [[np.exp(4.0 * (1.0 - state[0])) * (1.0 - 4.0 * state[0])]]
        ),
        np.array([0.9]),
        steps=5_000,
        discard=500,
    )
    assert exponent > 0.4


def test_lorenz_flow_lyapunov_is_positive():
    exponent = flow_lyapunov(lorenz, (1.0, 1.0, 1.0), (0.0, 25.0), discard=5.0)
    assert exponent > 0.5


def test_reduced_waterwheel_matches_lorenz():
    wheel = ReducedWaterwheel()
    expected = lorenz(0.0, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        wheel.derivative(0.0, np.array([1.0, 2.0, 3.0])), expected
    )


def test_discrete_waterwheel_can_start_turning():
    wheel = DiscreteWaterwheel(bucket_count=12, inflow=1.0)
    _, trajectory = wheel.integrate(time_span=(0.0, 5.0), dt=0.05)
    assert np.max(np.abs(trajectory[:, 1])) > 0.0
    assert np.min(trajectory[:, 2:]) >= 0.0


def test_default_discrete_wheel_reverses_direction():
    wheel = DiscreteWaterwheel()
    _, trajectory = wheel.integrate(time_span=(0.0, 30.0), dt=0.05)
    assert np.min(trajectory[:, 1]) < 0.0
    assert np.max(trajectory[:, 1]) > 0.0


def test_mass_moments_have_physical_observable_columns():
    wheel = DiscreteWaterwheel(bucket_count=4)
    trajectory = np.zeros((2, 6))
    trajectory[:, 2:] = 1.0
    moments = mass_moments(trajectory, wheel.bucket_count)
    assert moments.shape == (2, 4)
    np.testing.assert_allclose(moments[:, 2:], 0.0, atol=1e-14)
