"""Generate a phase portrait of the Chua circuit."""

from pathlib import Path

import matplotlib.pyplot as plt

from chaos_toy_models.electronics import chua
from chaos_toy_models.flows import integrate_flow
from chaos_toy_models.plotting import (
    CYAN,
    save_figure,
    save_rotating_gif,
    style_3d_axis,
    use_dark_style,
)

OUTPUT = Path(__file__).parent / "figures" / "chua-attractor.png"
GIF_OUTPUT = Path(__file__).parent / "figures" / "chua-attractor-rotation.gif"


def main() -> None:
    """Integrate Chua's circuit and save its phase portrait."""
    use_dark_style()
    time, states = integrate_flow(
        chua, (0.1, 0.0, 0.0), (0.0, 100.0), dt=0.02
    )
    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(projection="3d")
    axis.plot(
        states[:, 0],
        states[:, 1],
        states[:, 2],
        linewidth=0.45,
        color=CYAN,
        alpha=0.9,
    )
    style_3d_axis(axis)
    axis.set(
        xlabel="capacitor voltage x",
        ylabel="capacitor voltage y",
        zlabel="inductor current z",
        title="Chua circuit phase portrait",
    )
    figure.tight_layout()
    save_rotating_gif(figure, axis, GIF_OUTPUT)
    save_figure(figure, OUTPUT)


if __name__ == "__main__":
    main()
