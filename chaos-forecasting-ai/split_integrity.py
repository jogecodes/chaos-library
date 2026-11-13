"""Check initial-condition and exact-state separation before fitting or scoring."""

from itertools import combinations

import numpy as np


def audit_initial_conditions(groups, reference_initial=None):
    """Reject repeated or near-identical ICs, including the plotted reference."""
    groups = {name: np.asarray(values, dtype=float) for name, values in groups.items()}
    if reference_initial is not None:
        groups["reference"] = np.asarray(reference_initial, dtype=float)[None]
    if any(
        x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all() for x in groups.values()
    ):
        raise ValueError("Initial-condition groups must be nonempty finite matrices")
    if len({x.shape[1] for x in groups.values()}) != 1:
        raise ValueError("Initial-condition dimensions must agree")
    for name, values in groups.items():
        distance = np.linalg.norm(values[:, None] - values[None, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        if np.any(distance <= 1e-12):
            raise ValueError(f"Repeated initial conditions within {name}")
    separation = {}
    for (a, x), (b, y) in combinations(groups.items(), 2):
        minimum = float(np.linalg.norm(x[:, None] - y[None, :], axis=-1).min())
        if minimum <= 1e-12:
            raise ValueError(f"Initial-condition overlap between {a} and {b}")
        separation[f"{a}/{b}"] = minimum
    return {
        "trajectory_counts": {name: len(x) for name, x in groups.items()},
        "minimum_initial_condition_distance": separation,
        "initial_conditions_disjoint": True,
        "limitation": "Disjoint ICs do not imply independent windows or new dynamics.",
    }


def audit_long_horizon_split(data, initials, reference_initial):
    """Check cached data as well as the manifest; report reproducible evidence."""
    report = audit_initial_conditions(initials, reference_initial)
    physical_size = len(reference_initial)
    groups = {
        "train": data["train"],
        "validation": data["validation"],
        "test": data["fresh_trajectories"],
        "reference": data["reference"][None],
    }
    fingerprints = {}
    for name, trajectories in groups.items():
        expected = [reference_initial] if name == "reference" else initials[name]
        if not np.allclose(
            trajectories[:, 0, :physical_size], expected, rtol=0, atol=1e-12
        ):
            raise ValueError(
                f"{name} saved trajectories do not match initial conditions"
            )
        rows = (
            np.asarray(trajectories[:, :, :physical_size], dtype=np.float64)
            .reshape(-1, physical_size)
            .copy()
        )
        if not np.isfinite(rows).all():
            raise ValueError(f"Nonfinite states in {name}")
        rows[rows == 0] = 0  # Canonicalize signed zero before exact fingerprints.
        fingerprints[name] = {row.tobytes() for row in rows}
    report["exact_state_overlap_counts"] = {}
    for a, b in combinations(groups, 2):
        count = len(fingerprints[a] & fingerprints[b])
        report["exact_state_overlap_counts"][f"{a}/{b}"] = count
        if count:
            raise ValueError(f"{count} exact physical states shared by {a} and {b}")
    report["scope"] = "Same equations/parameters; held-out initial conditions."
    report["test_sampling"] = (
        "Two overlapping origins/test IC plus a reference; nine dependent origins."
    )
    report["test_reuse"] = (
        "Holdouts inspected in previous experiments; not an untouched final evaluation."
    )
    return report
