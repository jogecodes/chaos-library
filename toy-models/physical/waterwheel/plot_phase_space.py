"""Plot physical variables and mass-moment portraits of the bucket wheel."""

from pathlib import Path

import matplotlib.pyplot as plt

from chaos_toy_models.plotting import (
    CYAN,
    LIME,
    save_figure,
    save_rotating_gif,
    style_3d_axis,
    use_dark_style,
)
from chaos_toy_models.wheel import DiscreteWaterwheel, mass_moments

ROOT = Path(__file__).parent
OUTPUT = ROOT / "figures" / "waterwheel-physical-portraits.png"
GIF_OUTPUT = ROOT / "figures" / "waterwheel-physical-portraits-rotation.gif"


def main() -> None:
    """Integrate the bucket wheel and save physical-coordinate plots."""
    use_dark_style()
    wheel = DiscreteWaterwheel(bucket_count=24, inflow=1.0)
    time, trajectory = wheel.integrate(time_span=(0.0, 100.0), dt=0.02)
    theta, omega, cosine_moment, sine_moment = mass_moments(
        trajectory, wheel.bucket_count
    ).T

    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(2, 2)
    time_axis = figure.add_subplot(grid[0, :])
    phase_axis = figure.add_subplot(grid[1, 0])
    portrait_axis = figure.add_subplot(grid[1, 1], projection="3d")
    transient_end = 15.0
    transient = time <= transient_end
    settled = ~transient

    time_axis.axvspan(
        time[0],
        transient_end,
        color="#f4c95d",
        alpha=0.08,
        label="transient",
    )
    time_axis.plot(
        time,
        omega,
        label="omega: angular velocity",
        color="#f4c95d",
        linewidth=1.8,
    )
    time_axis.plot(
        time,
        cosine_moment,
        label="C1: cosine mass moment",
        color=CYAN,
        linewidth=1.2,
    )
    time_axis.plot(
        time,
        sine_moment,
        label="S1: sine mass moment",
        color=LIME,
        linewidth=1.2,
    )
    time_axis.axvline(transient_end, color="#ff6b6b", linewidth=1, alpha=0.8)
    time_axis.set(
        xlabel="time t",
        ylabel="moment / angular velocity",
        title="What the bucket model actually does",
    )
    time_axis.legend(ncol=2, loc="upper right", framealpha=0.25)
    time_axis.grid(alpha=0.15)
    time_axis.text(
        transient_end + 2,
        omega[settled].mean(),
        "post-transient regime",
        color="#ff9b85",
        va="center",
    )

    phase_axis.scatter(
        sine_moment[transient],
        omega[transient],
        c=time[transient],
        cmap="plasma",
        s=4,
        alpha=0.8,
        linewidths=0,
        label="transient",
    )
    phase_axis.scatter(
        sine_moment[settled],
        omega[settled],
        color="#dcecf2",
        s=2,
        alpha=0.35,
        linewidths=0,
        label="settled",
    )
    phase_axis.set(
        xlabel="S1: sine mass moment",
        ylabel="omega: angular velocity",
        title="Physical phase portrait",
    )
    phase_axis.legend(framealpha=0.25)
    phase_axis.grid(alpha=0.15)

    portrait_axis.scatter(
        omega,
        sine_moment,
        cosine_moment,
        c=time,
        cmap="plasma",
        s=2.0,
        linewidths=0,
    )
    portrait_axis.set(
        xlabel="omega",
        ylabel="S1",
        zlabel="C1",
        title="Physical state space (colour = time)",
    )
    style_3d_axis(portrait_axis)
    portrait_axis.view_init(elev=25, azim=-55)

    figure.suptitle(
        "Discrete Malkus-Howard waterwheel",
        color="#f4c95d",
        fontsize=18,
        fontweight="bold",
    )
    figure.tight_layout()
    save_rotating_gif(figure, portrait_axis, GIF_OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, OUTPUT)


if __name__ == "__main__":
    main()
