"""Generate chaotic population-map figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chaos_toy_models.maps import iterate_map
from chaos_toy_models.plotting import ACCENT, save_figure, style_axes, use_dark_style
from chaos_toy_models.population import forced_ricker, ricker

OUTPUT = Path(__file__).parent / "figures" / "population-chaos.png"


def main() -> None:
    """Simulate Ricker and seasonally forced Ricker populations."""
    use_dark_style()
    ricker_states = iterate_map(
        lambda state: np.array([ricker(state[0])]),
        (0.9,),
        2_000,
        discard=500,
    )[:, 0]
    forced_states = iterate_map(
        forced_ricker,
        (0.2, 0.0),
        2_000,
        discard=500,
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ricker_states[-500:], linewidth=0.7, color=ACCENT)
    axes[0].set(xlabel="post-transient step", ylabel="population", title="Ricker map")
    axes[1].scatter(
        forced_states[:, 1],
        forced_states[:, 0],
        c=np.arange(len(forced_states)),
        cmap="magma",
        s=2,
    )
    axes[1].set(
        xlabel="seasonal phase",
        ylabel="population",
        title="Forced Ricker map",
    )
    style_axes(axes)
    figure.tight_layout()
    save_figure(figure, OUTPUT)


if __name__ == "__main__":
    main()
