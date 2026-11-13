"""Prevent accidental reuse of initial conditions and cached trajectories."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chaos-forecasting-ai"))
from split_integrity import audit_initial_conditions, audit_long_horizon_split  # noqa: E402


def test_disjoint_initial_conditions_report_separation():
    report = audit_initial_conditions(
        {"train": [[0, 0], [1, 1]], "test": [[2, 2]]}, [3, 3]
    )
    assert report["initial_conditions_disjoint"]
    assert report["trajectory_counts"] == {"train": 2, "test": 1, "reference": 1}
    assert report["minimum_initial_condition_distance"]["train/test"] > 0


@pytest.mark.parametrize("test", [[[1, 1]], [[1 + 1e-14, 1]]])
def test_repeated_and_nearly_identical_test_initials_are_rejected(test):
    with pytest.raises(ValueError, match="overlap"):
        audit_initial_conditions({"train": [[1, 1]], "test": test})


def test_reference_cannot_reuse_training_initial():
    with pytest.raises(ValueError, match="overlap"):
        audit_initial_conditions({"train": [[1, 1]], "test": [[2, 2]]}, [1, 1])


def test_saved_trajectories_must_match_manifest_and_not_share_states():
    initials = {"train": [[0, 0]], "validation": [[2, 2]], "test": [[4, 4]]}
    data = {
        "train": np.array([[[0.0, 0.0], [1.0, 1.0]]]),
        "validation": np.array([[[2.0, 2.0], [3.0, 3.0]]]),
        "fresh_trajectories": np.array([[[4.0, 4.0], [5.0, 5.0]]]),
        "reference": np.array([[6.0, 6.0], [7.0, 7.0]]),
    }
    assert not any(
        audit_long_horizon_split(data, initials, [6, 6])[
            "exact_state_overlap_counts"
        ].values()
    )
    data["fresh_trajectories"][0, 1] = [1, 1]
    with pytest.raises(ValueError, match="exact physical states"):
        audit_long_horizon_split(data, initials, [6, 6])
    data["fresh_trajectories"][0, 0] = [9, 9]
    with pytest.raises(ValueError, match="do not match"):
        audit_long_horizon_split(data, initials, [6, 6])
