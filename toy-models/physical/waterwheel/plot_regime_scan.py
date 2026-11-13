"""Plot the discrete wheel's response as inflow is varied."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chaos_toy_models.plotting import (
    ACCENT,
    save_figure,
    style_axes,
    use_dark_style,
)
from chaos_toy_models.wheel import DiscreteWaterwheel

ROOT = Path(__file__).parent
OUTPUT = ROOT / "figures" / "waterwheel-inflow-scan.png"


def main() -> None:
    """Run an inflow sweep and save the measured long-time response."""
    use_dark_style()
    inflows = np.linspace(0.0, 5.0, 11)
    final_omega = []
    residual_omega = []
    for inflow in inflows:
        _, trajectory = DiscreteWaterwheel(inflow=float(inflow)).integrate(
            time_span=(0.0, 40.0), dt=0.1
        )
        tail = trajectory[len(trajectory) // 2 :, 1]
        final_omega.append(tail.mean())
        residual_omega.append(tail.std())

    figure, (response_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True
    )
    style_axes((response_axis, residual_axis))

    response_axis.plot(inflows, final_omega, "o-", color=ACCENT)
    response_axis.set(
        ylabel="mean omega in final half",
        title="Discrete waterwheel: inflow response",
    )
    response_axis.axvline(0.0, color="#ff6b6b", linewidth=1)
    response_axis.text(
        0.12,
        final_omega[1],
        "rest at zero inflow",
        color="#ff9b85",
        va="bottom",
    )

    residual_axis.semilogy(
        inflows,
        np.maximum(residual_omega, 1e-12),
        "o-",
        color="#62c370",
    )
    residual_axis.set(
        xlabel="inflow",
        ylabel="std(omega) in final half",
        title="Residual motion after transients",
    )

    figure.suptitle(
        "No regime transition is detected in the default bucket model",
        color="#f4c95d",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, OUTPUT)


if __name__ == "__main__":
    main()
