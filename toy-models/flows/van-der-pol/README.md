# Forced Van der Pol oscillator

This directory contains the externally forced Van der Pol oscillator, the
chaotic member of the Van der Pol family used in this repository:

```text
x'' - mu * (1 - x^2) * x' + x = A cos(omega_f t)
```

The documented parameters are `mu=3`, `A=4.4`, and `omega_f=1.68`. Earlier
finite-time Lyapunov estimates were approximately `0.12` to `0.14`; later
conditional checks at fixed forcing phase gave approximately `0.17` to `0.18`
with different windows/settings. These are finite numerical estimates, not
exact asymptotic constants. The stroboscopic Poincare section is scattered;
its appearance alone does not prove chaos. See the
[current research account](../../../docs/CURRENT_STATUS.md).

Generate the diagnostic figure:

```bash
python toy-models/flows/van-der-pol/plot_forced.py
```

Generate the circuit-style oscilloscope animation:

```bash
python toy-models/flows/van-der-pol/animate_circuit.py
```

Outputs:

- `figures/forced-van-der-pol-chaos.png`
- `figures/forced-van-der-pol-attractor-3d.png`
- `figures/forced-van-der-pol-attractor-3d.gif`
- `figures/forced-van-der-pol-circuit.png`
- `animations/forced-van-der-pol-oscilloscope.gif`

The 3D attractor uses the extended state `(x, v, phi)`, where `phi` is the
phase of the external forcing. This is the appropriate autonomous
three-dimensional representation of the periodically forced oscillator.

The circuit drawing is an idealized interpretation, not a calibrated
electronic design.
