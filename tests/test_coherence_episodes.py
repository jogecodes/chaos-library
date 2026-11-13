"""Episode-level loss needs sustained disagreement and permits recovery."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chaos-forecasting-ai"))
from coherence_episodes import detect_episodes, episode_policy, validation_memory  # noqa: E402


def policy(window=10, gap=2):
    return {
        "window_steps": window,
        "max_good_gap_steps": gap,
        "bad_fraction": 0.8,
        "local_error_threshold": 0.5,
        "dt": 0.1,
    }


def test_temporary_episode_then_sustained_breakdown_and_recovery():
    error = np.zeros((1, 50, 2))
    error[:, 3:8] = 1
    error[:, 20:35] = 1
    report, status = detect_episodes(error, policy())
    assert report["onset_time_by_origin"] == [2.1]
    assert report["confirmation_time_by_origin"] == [3.0]
    assert (status[0, 3:8] == 1).all()
    assert (status[0, 20:35] == 2).all()
    assert (status[0, 35:] == 0).all()
    episode = report["episodes_by_origin"][0][1]
    assert episode["recovery_onset_time"] == 3.6
    assert episode["recovery_confirmation_time"] == pytest.approx(3.8)


def test_chance_agreement_does_not_end_a_sustained_episode():
    error = np.ones((1, 30, 2))
    error[:, 5:7] = 0
    report, status = detect_episodes(error, policy())
    assert report["onset_time_by_origin"] == [0.1]
    assert (status[0] == 2).all()


def test_long_good_gap_separates_temporary_episodes():
    error = np.zeros((1, 30, 1))
    error[:, :7] = 1
    error[:, 10:17] = 1
    report, status = detect_episodes(error, policy())
    assert report["censored_by_origin"] == [True]
    assert report["onset_time_by_origin"] == [None]
    assert not (status == 2).any()


def test_incomplete_tail_is_not_confirmed_and_no_onset_is_invented():
    error = np.zeros((1, 20, 1))
    error[:, 15:] = 1
    report, _ = detect_episodes(error, policy())
    assert report["onset_time_by_origin"] == [None]
    assert report["confirmation_time_by_origin"] == [None]
    assert report["median_onset_capped"] == 2.0


def test_any_physical_variable_can_trigger_joint_breakdown():
    error = np.zeros((1, 20, 3))
    error[:, :, 2] = 1
    report, _ = detect_episodes(error, policy())
    assert report["onset_time_by_origin"] == [0.1]


def test_numerical_failure_is_immediate_without_confirmation_delay():
    error = np.zeros((1, 20, 1))
    error[:, 3:] = np.inf
    report, status = detect_episodes(error, policy())
    assert report["onset_time_by_origin"] == [0.4]
    assert report["confirmation_time_by_origin"] == [0.4]
    assert (status[0, 3:] == 2).all()


def test_policy_has_no_forcing_input_and_uses_validation_memory():
    t = np.arange(300) * 0.05
    validation = np.column_stack([np.sin(t), np.cos(t)])[None]
    p = episode_policy(validation, 2, 0.05)
    memory = validation_memory(validation, 2, 0.05)
    assert "forcing_frequency" not in inspect.signature(episode_policy).parameters
    assert p["window_steps"] == max(5, int(np.ceil(5 * memory["memory_steps"])))
    assert p["max_good_gap_steps"] == int(np.floor(memory["memory_steps"]))


def test_memory_policy_is_invariant_to_coordinate_units_and_offsets():
    t = np.arange(500) * 0.05
    validation = np.column_stack([np.sin(t), np.cos(1.7 * t)])[None]
    original = episode_policy(validation, 2, 0.05)
    converted = episode_policy(
        validation * np.array([100.0, 0.02]) + [20.0, -7.0], 2, 0.05
    )
    assert original == converted


def test_memory_policy_respects_time_unit_changes():
    t = np.arange(500) * 0.05
    validation = np.column_stack([np.sin(t), np.cos(1.7 * t)])[None]
    original = episode_policy(validation, 2, 0.05)
    converted = episode_policy(validation, 2, 0.1)
    assert converted["window_steps"] == original["window_steps"]
    assert converted["memory_time"] == 2 * original["memory_time"]
    assert (
        converted["confirmation_delay_time"] == 2 * original["confirmation_delay_time"]
    )


def test_memory_policy_ignores_nonphysical_phase_channels():
    t = np.arange(500) * 0.05
    validation = np.column_stack([np.sin(t), np.cos(t)])[None]
    augmented = np.concatenate([validation, np.zeros_like(validation)], axis=2)
    assert episode_policy(validation, 2, 0.05) == episode_policy(augmented, 2, 0.05)


def test_brief_agreement_inside_temporary_episode_is_not_recovery():
    error = np.zeros((1, 30, 1))
    error[:, 5:7] = 1
    error[:, 8:10] = 1
    _, status = detect_episodes(error, policy())
    assert (status[0, 5:10] == 1).all()
    assert (status[0, 10:] == 0).all()


def test_unconfirmed_recovery_at_horizon_stays_in_episode():
    error = np.ones((1, 20, 1))
    error[:, -1] = 0
    report, status = detect_episodes(error, policy())
    assert status[0, -1] == 2
    assert report["episodes_by_origin"][0][0]["recovery_censored"]
