"""Create a reproducible waterwheel animation.

Examples:
    python toy-models/physical/waterwheel/animate.py --format gif
    python toy-models/physical/waterwheel/animate.py --format mp4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon

from chaos_toy_models.wheel import DiscreteWaterwheel


def create_animation(output: Path, file_format: str) -> None:
    """Integrate the discrete wheel and save an animation."""
    wheel = DiscreteWaterwheel(bucket_count=24, inflow=1.0)
    time, trajectory = wheel.integrate(time_span=(0.0, 40.0), dt=0.05)
    angles = 2.0 * np.pi * np.arange(wheel.bucket_count) / wheel.bucket_count
    plt.style.use("dark_background")
    figure = plt.figure(figsize=(11, 6), facecolor="#101820")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.25, 1), wspace=0.18)
    axis = figure.add_subplot(grid[0], aspect="equal", facecolor="#182936")
    plot_axis = figure.add_subplot(grid[1], facecolor="#182936")
    axis.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35))
    axis.axis("off")
    axis.add_patch(Circle((0, 0), 1.02, fill=False, lw=2, color="#8da9b8"))
    axis.add_patch(Circle((0, 0), 0.08, color="#f4c95d", zorder=5))
    radius = 1.0
    wheel_line, = axis.plot([], [], "-", lw=1.2, color="#dcecf2", alpha=0.7)
    arm_points, = axis.plot([], [], "o", ms=5, color="#f4c95d", zorder=4)
    bucket_patches = [
        Polygon(np.zeros((4, 2)), closed=True, color="#50b7c1", ec="#b8f2e6", lw=0.6)
        for _ in angles
    ]
    for patch in bucket_patches:
        axis.add_patch(patch)
    trail, = axis.plot([], [], color="#f4c95d", lw=1.0, alpha=0.35)
    plot_axis.set(
        xlim=(time[0], time[-1]),
        ylim=(trajectory[:, 1].min() * 1.1, trajectory[:, 1].max() * 1.1),
        xlabel="time",
        ylabel="angular velocity",
        title="Motion",
    )
    plot_axis.grid(alpha=0.15)
    velocity_line, = plot_axis.plot([], [], color="#f4c95d", lw=1.5)
    cursor_line = plot_axis.axvline(time[0], color="#ff6b6b", lw=1, alpha=0.8)
    title = figure.suptitle(
        "Discrete Malkus–Howard waterwheel",
        color="#f4c95d",
        fontsize=16,
    )

    def update(frame: int):
        theta = trajectory[frame, 0]
        positions = angles + theta
        masses = trajectory[frame, 2:]
        levels = np.clip(masses / max(masses.max(), 1e-12), 0.02, 1.0)
        x = radius * np.sin(positions)
        y = -radius * np.cos(positions)
        spoke_x = np.column_stack((np.zeros_like(x), x, np.full_like(x, np.nan)))
        spoke_y = np.column_stack((np.zeros_like(y), y, np.full_like(y, np.nan)))
        wheel_line.set_data(spoke_x.ravel(), spoke_y.ravel())
        arm_points.set_data(x, y)
        for index, patch in enumerate(bucket_patches):
            width = 0.18
            height = 0.12 + 0.22 * levels[index]
            radial = np.array([np.cos(positions[index]), np.sin(positions[index])])
            tangent = np.array([-radial[1], radial[0]])
            center = np.array([x[index], y[index]])
            corners = np.array([
                center - radial * height / 2 - tangent * width / 2,
                center + radial * height / 2 - tangent * width / 2,
                center + radial * height / 2 + tangent * width / 2,
                center - radial * height / 2 + tangent * width / 2,
            ])
            patch.set_xy(corners)
        start = max(0, frame - 90)
        trail.set_data(x[start:frame + 1], y[start:frame + 1])
        velocity_line.set_data(time[:frame + 1], trajectory[:frame + 1, 1])
        cursor_line.set_xdata([time[frame], time[frame]])
        title.set_text(
            f"Discrete waterwheel   t={time[frame]:.1f}   "
            f"ω={trajectory[frame, 1]:+.2f}"
        )
        return (
            wheel_line,
            arm_points,
            trail,
            velocity_line,
            cursor_line,
            title,
            *bucket_patches,
        )

    animation = FuncAnimation(
        figure, update, frames=range(0, len(time), 2), interval=33, blit=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "mp4":
        animation.save(output, writer=FFMpegWriter(fps=30), dpi=120)
    else:
        animation.save(output, writer=PillowWriter(fps=30), dpi=80)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    args = parser.parse_args()
    suffix = f".{args.format}"
    create_animation(
        Path(__file__).parent / "animations" / f"waterwheel{suffix}",
        args.format,
    )
