"""Shared visual style for toy-model figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

BACKGROUND = "#101820"
PANEL = "#182936"
ACCENT = "#f4c95d"
TEXT = "#dcecf2"
MUTED = "#8fa8b5"
CYAN = "#5ee7f2"
CORAL = "#ff8066"
LIME = "#b8f26d"
BRIGHT_TIME = LinearSegmentedColormap.from_list(
    "bright_time",
    [CYAN, "#8d7cff", "#ff6bd6", CORAL, ACCENT, LIME],
)


def use_dark_style() -> None:
    """Apply the repository's dark figure style."""
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": MUTED,
            "text.color": TEXT,
            "legend.facecolor": PANEL,
            "legend.edgecolor": MUTED,
        }
    )


def style_axes(axes) -> None:
    """Apply shared grid and spine styling to one or more axes."""
    for axis in axes if hasattr(axes, "__iter__") else (axes,):
        axis.set_facecolor(PANEL)
        axis.grid(alpha=0.16, color=MUTED)
        for spine in axis.spines.values():
            spine.set_color(MUTED)


def style_3d_axis(axis) -> None:
    """Apply the dark panel styling supported by a 3D Matplotlib axis."""
    axis.set_facecolor(PANEL)
    axis.xaxis.pane.set_facecolor(PANEL)
    axis.yaxis.pane.set_facecolor(PANEL)
    axis.zaxis.pane.set_facecolor(PANEL)
    axis.xaxis.pane.set_edgecolor(MUTED)
    axis.yaxis.pane.set_edgecolor(MUTED)
    axis.zaxis.pane.set_edgecolor(MUTED)
    axis.grid(alpha=0.16, color=MUTED)


def save_figure(figure, output: Path, *, dpi: int = 160) -> None:
    """Save a figure, creating its parent directory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_rotating_gif(
    figure,
    axis,
    output: Path,
    *,
    frames: int = 180,
    fps: int = 24,
) -> None:
    """Save a rotating-camera GIF for a three-dimensional axis."""
    output.parent.mkdir(parents=True, exist_ok=True)
    initial_azimuth = axis.azim

    def update(frame: int) -> None:
        axis.view_init(elev=25, azim=initial_azimuth + 360 * frame / frames)

    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=100)
