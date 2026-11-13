"""Reproducible toy models for deterministic chaos."""

from .electronics import chua
from .flows import duffing, forced_van_der_pol, lorenz, rossler
from .maps import henon, ikeda, logistic
from .mechanics import double_pendulum
from .population import forced_ricker, ricker
from .wheel import DiscreteWaterwheel, ReducedWaterwheel

__all__ = [
    "DiscreteWaterwheel",
    "ReducedWaterwheel",
    "duffing",
    "forced_van_der_pol",
    "forced_ricker",
    "chua",
    "double_pendulum",
    "henon",
    "ikeda",
    "logistic",
    "lorenz",
    "rossler",
    "ricker",
]
