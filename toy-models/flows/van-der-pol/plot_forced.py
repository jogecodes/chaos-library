"""Generate diagnostics for a chaotic forced Van der Pol oscillator."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chaos_toy_models.flows import forced_van_der_pol, integrate_flow
from chaos_toy_models.metrics import flow_lyapunov
from chaos_toy_models.plotting import (
    ACCENT,
    BRIGHT_TIME,
    CORAL,
    CYAN,
    save_figure,
    save_rotating_gif,
    style_3d_axis,
    style_axes,
    use_dark_style,
)

OUTPUT = Path(__file__).parent / "figures" / "forced-van-der-pol-chaos.png"
ATTRACTOR_OUTPUT = (
    Path(__file__).parent / "figures" / "forced-van-der-pol-attractor-3d.png"
)
ATTRACTOR_GIF_OUTPUT = (
    Path(__file__).parent / "figures" / "forced-van-der-pol-attractor-3d.gif"
)


def main() -> None:
    """Integrate the forced oscillator and save chaos diagnostics."""
    use_dark_style()
    time, states = integrate_flow(
        forced_van_der_pol,
        (0.1, 0.0),
        (0.0, 250.0),
        dt=0.02,
    )
    exponent = flow_lyapunov(
        forced_van_der_pol,
        (0.1, 0.0),
        (0.0, 250.0),
        dt=0.1,
        discard=80.0,
    )
    forcing_period = 2.0 * np.pi / 1.68
    sample_times = np.arange(80.0, 250.0, forcing_period)
    stroboscopic = np.column_stack([
        np.interp(sample_times, time, states[:, 0]),
        np.interp(sample_times, time, states[:, 1]),
    ])
    phase = (1.68 * time) % (2.0 * np.pi)

    attractor_figure = plt.figure(figsize=(10, 8))
    attractor_axis = attractor_figure.add_subplot(projection="3d")
    attractor_axis.scatter(
        states[::3, 0],
        states[::3, 1],
        phase[::3],
        c=time[::3],
        cmap=BRIGHT_TIME,
        s=2.2,
        linewidths=0,
    )
    attractor_axis.set(
        xlabel="position x",
        ylabel="velocity v",
        zlabel="forcing phase phi",
        title="Forced Van der Pol strange attractor",
    )
    style_3d_axis(attractor_axis)
    attractor_axis.view_init(elev=25, azim=-55)
    attractor_figure.tight_layout()
    save_rotating_gif(attractor_figure, attractor_axis, ATTRACTOR_GIF_OUTPUT)

    attractor_figure = plt.figure(figsize=(10, 8))
    attractor_axis = attractor_figure.add_subplot(projection="3d")
    attractor_axis.scatter(
        states[::3, 0],
        states[::3, 1],
        phase[::3],
        c=time[::3],
        cmap=BRIGHT_TIME,
        s=2.2,
        linewidths=0,
    )
    attractor_axis.set(
        xlabel="position x",
        ylabel="velocity v",
        zlabel="forcing phase phi",
        title="Forced Van der Pol strange attractor",
    )
    style_3d_axis(attractor_axis)
    attractor_axis.view_init(elev=25, azim=-55)
    attractor_figure.tight_layout()
    save_figure(attractor_figure, ATTRACTOR_OUTPUT)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(time[-3_000:], states[-3_000:, 0], color=CYAN, linewidth=1)
    axes[0, 0].set(
        xlabel="time",
        ylabel="position x",
        title="Forced voltage / position",
    )
    axes[0, 1].scatter(
        states[::4, 0],
        states[::4, 1],
        c=time[::4],
        cmap=BRIGHT_TIME,
        s=2,
        linewidths=0,
    )
    axes[0, 1].set(
        xlabel="position x",
        ylabel="velocity v",
        title="Forced phase portrait",
    )
    axes[1, 0].scatter(
        stroboscopic[:, 0],
        stroboscopic[:, 1],
        color=ACCENT,
        s=12,
        alpha=0.9,
        linewidths=0,
    )
    axes[1, 0].set(
        xlabel="x at forcing period",
        ylabel="v at forcing period",
        title="Stroboscopic Poincare section",
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.98,
        "External forcing",
        color=CORAL,
        fontsize=14,
        fontweight="bold",
    )
    axes[1, 1].text(
        0.05,
        0.68,
        "mu = 3.0\nA = 4.4\nforcing frequency = 1.68",
        color="white",
        fontsize=13,
        linespacing=1.8,
    )
    axes[1, 1].text(
        0.05,
        0.30,
        f"largest Lyapunov estimate\nlambda = {exponent:.3f} > 0",
        color=ACCENT,
        fontsize=14,
        fontweight="bold",
        linespacing=1.8,
    )
    style_axes(axes[:1].flat)
    style_axes(axes[1, :1])
    figure.suptitle(
        "Externally forced Van der Pol: chaotic regime",
        color=ACCENT,
        fontsize=18,
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, OUTPUT)


if __name__ == "__main__":
    main()
