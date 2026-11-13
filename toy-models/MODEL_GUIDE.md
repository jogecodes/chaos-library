# Model guide

## What counts as a model?

The maps are models too. They are discrete-time dynamical systems: the state
is updated once per iteration rather than by integrating an ordinary
differential equation. They are useful minimal models because they isolate
mechanisms such as nonlinear growth, stretching, and folding.

## Current dynamical regimes

| Model | Time | Default behaviour |
| --- | --- | --- |
| Logistic, Hénon, Ikeda | Discrete | Chaotic for documented parameter choices |
| Lorenz, Rössler | Continuous | Chaotic for the defaults used here |
| Duffing | Continuous and forced | Parameter-dependent; not automatically chaotic |
| Forced Van der Pol | Continuous and forced | Chaotic for the documented parameters |
| Chua | Continuous | Double-scroll chaotic attractor for the documented defaults |
| Double pendulum | Continuous | Sensitive mechanical motion; chaos depends on energy and initial state |
| Ricker | Discrete population | Positive Lyapunov estimate at `growth=4` |
| Forced Ricker | Discrete population | Positive Lyapunov estimate for the documented forcing |
| Discrete waterwheel | Continuous bucket ODE | Reversals and positive finite-time tangent estimate for its exploratory defaults |

## Phase space versus time plots

An autonomous system's phase space contains its state variables. Adding time
as a third plotting axis gives a spacetime trajectory, not a new attractor.
For example, the double pendulum has four state variables:

```text
(theta_1, theta_2, omega_1, omega_2)
```

There is no unique three-dimensional Lorenz-style attractor for it. Useful
projections include `(theta_1, omega_1, theta_2)`, `(theta_1, omega_1)`, and
Poincare sections. A time-coloured 3D trajectory is useful for presentation,
but it should be labelled as a projection or spacetime view.

All model-specific directories use the same layout:

```text
toy-models/<category>/<model>/
├── README.md
├── plot_*.py
├── figures/
└── animations/
```
