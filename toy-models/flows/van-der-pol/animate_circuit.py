"""Animate the chaotic forced Van der Pol circuit with oscilloscope traces.

Run:
    python toy-models/flows/van-der-pol/animate_circuit.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from chaos_toy_models.flows import forced_van_der_pol, integrate_flow
from chaos_toy_models.plotting import (
    ACCENT,
    CORAL,
    CYAN,
    LIME,
    MUTED,
    style_axes,
    use_dark_style,
)

ROOT = Path(__file__).parent
FIGURE_OUTPUT = ROOT / "figures" / "forced-van-der-pol-circuit.png"
GIF_OUTPUT = ROOT / "animations" / "forced-van-der-pol-oscilloscope.gif"


def main() -> None:
    """Simulate the forced circuit interpretation and save a scope animation."""
    use_dark_style()
    forcing_frequency = 1.68
    time, states = integrate_flow(
        forced_van_der_pol,
        (0.1, 0.0),
        (0.0, 250.0),
        dt=0.01,
        mu=3.0,
        amplitude=4.4,
        forcing_frequency=forcing_frequency,
    )
    transient_end = 80.0
    frame_indices = np.linspace(
        np.searchsorted(time, transient_end),
        len(time) - 1,
        240,
        dtype=int,
    )
    forcing_period = 2.0 * np.pi / forcing_frequency
    sample_times = np.arange(transient_end, time[-1], forcing_period)
    poincare_states = np.column_stack([
        np.interp(sample_times, time, states[:, 0]),
        np.interp(sample_times, time, states[:, 1]),
    ])

    figure = plt.figure(figsize=(13, 9))
    grid = figure.add_gridspec(3, 2, width_ratios=(0.85, 1.6), hspace=0.45)
    circuit_axis = figure.add_subplot(grid[:, 0])
    voltage_axis = figure.add_subplot(grid[0, 1])
    phase_axis = figure.add_subplot(grid[1, 1])
    poincare_axis = figure.add_subplot(grid[2, 1])

    circuit_axis.set(
        xlim=(-0.2, 4.2),
        ylim=(-0.7, 2.8),
        title="Chaotic forced Van der Pol circuit",
    )
    circuit_axis.axis("off")
    circuit_axis.plot([0.5, 3.7], [2.2, 2.2], color=MUTED, linewidth=2)
    circuit_axis.plot([0.5, 3.7], [0.3, 0.3], color=MUTED, linewidth=2)
    circuit_axis.plot([0.5, 0.5], [0.3, 2.2], color=MUTED, linewidth=2)
    circuit_axis.plot([3.7, 3.7], [0.3, 2.2], color=MUTED, linewidth=2)
    circuit_axis.add_patch(
        Rectangle((1.0, 1.9), 0.8, 0.6, fill=False, edgecolor=CYAN, linewidth=2)
    )
    circuit_axis.text(1.4, 2.2, "L", ha="center", va="center", color=CYAN, fontsize=15)
    circuit_axis.add_patch(
        Circle((2.35, 2.2), 0.32, fill=False, edgecolor=ACCENT, linewidth=2)
    )
    circuit_axis.text(
        2.35,
        2.2,
        "C",
        ha="center",
        va="center",
        color=ACCENT,
        fontsize=15,
    )
    circuit_axis.plot([2.35, 2.35], [1.88, 1.2], color=MUTED, linewidth=2)
    circuit_axis.plot([2.1, 2.6], [1.2, 1.2], color=ACCENT, linewidth=3)
    circuit_axis.plot([2.15, 2.55], [1.05, 1.05], color=ACCENT, linewidth=3)
    circuit_axis.plot([2.2, 2.5], [0.9, 0.9], color=ACCENT, linewidth=3)
    circuit_axis.plot([2.35, 2.35], [0.9, 0.3], color=MUTED, linewidth=2)
    nonlinear = circuit_axis.add_patch(
        Rectangle((2.9, 1.85), 0.8, 0.7, fill=False, edgecolor=LIME, linewidth=2)
    )
    circuit_axis.text(
        3.25,
        2.2,
        "N(x)",
        ha="center",
        va="center",
        color=LIME,
        fontsize=13,
    )
    current_dot, = circuit_axis.plot([], [], "o", color=CORAL, markersize=8)
    status = circuit_axis.text(2.1, -0.35, "", ha="center", color="white")

    voltage_axis.plot(time, states[:, 0], color="#334857", linewidth=1)
    voltage_line, = voltage_axis.plot([], [], color=CYAN, linewidth=1.8)
    voltage_cursor = voltage_axis.axvline(time[0], color=CORAL, linewidth=1)
    voltage_axis.set(
        xlabel="time",
        ylabel="voltage x",
        title="Oscilloscope channel 1: forced capacitor voltage",
    )

    phase_axis.plot(states[:, 0], states[:, 1], color="#334857", linewidth=1)
    phase_line, = phase_axis.plot([], [], color=ACCENT, linewidth=1.8)
    phase_marker, = phase_axis.plot([], [], "o", color=CORAL, markersize=7)
    phase_axis.set(
        xlabel="voltage x",
        ylabel="current / velocity v",
        title="Scope XY mode: 2D projection",
    )
    poincare_axis.scatter(
        poincare_states[:, 0],
        poincare_states[:, 1],
        color="#334857",
        s=10,
        linewidths=0,
    )
    poincare_points = poincare_axis.scatter(
        [],
        [],
        color=ACCENT,
        s=20,
        linewidths=0,
    )
    poincare_axis.set(
        xlabel="x at forcing period",
        ylabel="v at forcing period",
        title="Stroboscopic Poincare section",
    )
    style_axes((voltage_axis, phase_axis, poincare_axis))

    def update(frame: int):
        index = frame_indices[frame]
        start = max(0, index - 900)
        voltage_line.set_data(time[start : index + 1], states[start : index + 1, 0])
        voltage_cursor.set_xdata([time[index], time[index]])
        phase_line.set_data(states[start : index + 1, 0], states[start : index + 1, 1])
        phase_marker.set_data([states[index, 0]], [states[index, 1]])
        completed = sample_times <= time[index]
        poincare_points.set_offsets(poincare_states[completed])
        current_dot.set_data([2.35], [1.65 + 0.25 * np.tanh(states[index, 1])])
        damping = 1.0 - states[index, 0] ** 2
        nonlinear.set_edgecolor(LIME if damping < 0 else CYAN)
        status.set_text(
            f"t={time[index]:.1f}   x={states[index, 0]:+.2f}   "
            f"v={states[index, 1]:+.2f}"
        )
        return (
            voltage_line,
            voltage_cursor,
            phase_line,
            phase_marker,
            poincare_points,
            current_dot,
            status,
            nonlinear,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000 / 30,
        blit=False,
    )
    GIF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    animation.save(GIF_OUTPUT, writer=PillowWriter(fps=30), dpi=90)
    figure.savefig(FIGURE_OUTPUT, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
