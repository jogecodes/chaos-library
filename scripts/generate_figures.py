"""Generate the small set of plots referenced by the README."""

from pathlib import Path

import matplotlib.pyplot as plt

from chaos_toy_models.flows import integrate_flow, lorenz
from chaos_toy_models.wheel import DiscreteWaterwheel

OUTPUT = Path(__file__).parents[1] / "docs" / "figures"
WATERWHEEL_OUTPUT = (
    Path(__file__).parents[1] / "toy-models" / "waterwheel" / "figures"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

time, states = integrate_flow(lorenz, (1.0, 1.0, 1.0), (0.0, 40.0), dt=0.01)

fig, axis = plt.subplots()
axis.plot(time, states[:, 0], linewidth=0.6)
axis.set(xlabel="time", ylabel="x", title="Lorenz state with sign changes")
fig.tight_layout()
fig.savefig(OUTPUT / "lorenz-time-series.png", dpi=140)
plt.close(fig)

fig = plt.figure()
axis = fig.add_subplot(projection="3d")
axis.plot(
    states[:, 0],
    states[:, 1],
    states[:, 2],
    linewidth=0.4,
    color="#246bce",
    label="trajectory",
)
axis.scatter(
    states[0, 0],
    states[0, 1],
    states[0, 2],
    color="#2ca02c",
    s=24,
    label="initial state",
)
axis.set(
    xlabel="x: convective intensity",
    ylabel="y: horizontal temperature contrast",
    zlabel="z: vertical temperature profile",
    title="Lorenz attractor (sigma=10, rho=28, beta=8/3)",
)
axis.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT / "lorenz-attractor.png", dpi=140)
plt.close(fig)

fig, axis = plt.subplots()
for inflow in (0.5, 1.0, 2.0):
    _, trajectory = DiscreteWaterwheel(inflow=inflow).integrate(
        time_span=(0.0, 40.0), dt=0.02
    )
    axis.plot(trajectory[:, 1], alpha=0.7, label=f"inflow={inflow:g}")
axis.set(
    xlabel="sample",
    ylabel="angular velocity",
    title="Bucket-wheel response at selected inflows",
)
axis.legend()
fig.tight_layout()
WATERWHEEL_OUTPUT.mkdir(parents=True, exist_ok=True)
fig.savefig(WATERWHEEL_OUTPUT / "waterwheel-regimes.png", dpi=140)
plt.close(fig)
