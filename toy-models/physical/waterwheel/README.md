# Waterwheel toy model

This directory contains waterwheel-specific visualisations, plots, and
derivation notes. The numerical implementation remains in
`src/chaos_toy_models/wheel.py`.

The physical variables used in the plots are:

- `theta`: accumulated wheel angle;
- `omega`: angular velocity;
- `C1`: cosine first Fourier mass moment;
- `S1`: sine first Fourier mass moment;
- `t`: simulation time.

The physical two-dimensional phase portrait is `omega` versus `S1`. The
three-dimensional state-space panel uses `(omega, S1, C1)`, the angular
velocity and the two first Fourier moments of the bucket masses.

Generate the plot with:

```bash
python toy-models/physical/waterwheel/plot_phase_space.py
```

The result is written to
`toy-models/physical/waterwheel/figures/waterwheel-physical-portraits.png`.

The figure contains:

1. `omega(t)`, `C1(t)`, and `S1(t)`;
2. the physical phase portrait `(S1, omega)`;
3. the physical state-space projection `(omega, S1, C1)`.

The same 3D projection is also rendered as
`figures/waterwheel-physical-portraits-rotation.gif`.

The earlier point-inflow version settled to a steady rotation. The current
smooth-inflow model is designed to test reversals directly from bucket data.

The dark dashboard separates the transient (`t < 15`) from later motion, uses
colour to encode time, and keeps the physical observables visible.

The physical portraits and regime scan use only `DiscreteWaterwheel` output:
`omega` comes from the simulated wheel and `C1`/`S1` are calculated from the
simulated bucket masses.

The default exploratory parameters are `leak_rate=0.1`, `drag=0.8`, and
`inflow_concentration=0.5`. They were selected by simulation to produce
direction reversals; they are not fitted to a physical apparatus.

The inflow response scan is generated with:

```bash
python toy-models/physical/waterwheel/plot_regime_scan.py
```

It is saved as
`toy-models/physical/waterwheel/figures/waterwheel-inflow-scan.png`. It summarizes
the mean and standard deviation of angular velocity in the final half of each
simulation, rather than a bifurcation diagram. The script's older title claiming
no regime transition is detected must not be read as evidence that no transition
exists. Older steady-response descriptions predate the current smooth-inflow
defaults: current runs show direction reversals and reported positive finite-time
sensitivity estimates. These finite numerical observations do not establish
asymptotic chaos or experimentally validate the bucket model. See
[the current research account](../../../docs/CURRENT_STATUS.md).
