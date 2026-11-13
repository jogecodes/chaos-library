"""Generate a double-pendulum trajectory and phase portrait."""

from pathlib import Path

import matplotlib.pyplot as plt

from chaos_toy_models.flows import integrate_flow
from chaos_toy_models.mechanics import double_pendulum
from chaos_toy_models.plotting import (
    ACCENT,
    BRIGHT_TIME,
    CYAN,
    save_figure,
    save_rotating_gif,
    style_3d_axis,
    style_axes,
    use_dark_style,
)

OUTPUT = Path(__file__).parent / "figures" / "double-pendulum.png"
GIF_OUTPUT = Path(__file__).parent / "figures" / "double-pendulum-rotation.gif"


def main() -> None:
    """Integrate a double pendulum and save its trajectory."""
    use_dark_style()
    time, states = integrate_flow(
        double_pendulum,
        (2.0, 1.0, 0.0, 0.0),
        (0.0, 30.0),
        dt=0.01,
    )
    figure = plt.figure(figsize=(13, 9))
    grid = figure.add_gridspec(2, 2)
    time_axis = figure.add_subplot(grid[0, :])
    phase_axis = figure.add_subplot(grid[1, 0])
    projection_axis = figure.add_subplot(grid[1, 1], projection="3d")
    time_axis.plot(time, states[:, 0], label="theta 1", color=ACCENT)
    time_axis.plot(time, states[:, 1], label="theta 2", color=CYAN)
    time_axis.set(xlabel="time", ylabel="angle", title="Double-pendulum angles")
    time_axis.legend()
    phase_axis.plot(states[:, 0], states[:, 2], linewidth=0.6, color=ACCENT)
    phase_axis.set(
        xlabel="theta 1",
        ylabel="omega 1",
        title="First-link phase portrait",
    )
    projection_axis.scatter(
        states[:, 0],
        states[:, 2],
        states[:, 1],
        c=time,
        cmap=BRIGHT_TIME,
        s=1.5,
        linewidths=0,
    )
    projection_axis.set(
        xlabel="theta 1",
        ylabel="omega 1",
        zlabel="theta 2",
        title="3D state-space projection (colour = time)",
    )
    style_axes((time_axis, phase_axis))
    style_3d_axis(projection_axis)
    figure.tight_layout()
    save_rotating_gif(figure, projection_axis, GIF_OUTPUT)
    save_figure(figure, OUTPUT)


if __name__ == "__main__":
    main()
