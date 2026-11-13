# Numerical methods and protocols

## Simulation models

The `chaos_toy_models` package implements deterministic maps, differential-equation
flows, double-pendulum mechanics, circuits and a finite bucket waterwheel.
Illustrative plots are numerical simulations rather than empirical measurements.
See model-local derivations and docstrings for equations and parameters.

## Laboratory forecasting

Lorenz, Rössler and forced Van der Pol each use 32 training trajectories and four
validation initial conditions. Four held-out test trajectories each provide two
overlapping forecast windows, plus a separate reference origin: nine dependent
origins per system. The complete original arrays are retained in `results/`.

Active families are polynomial dynamics, Transformer, GRU and residual MLP.
Neural fits use seeds 7/11/23, 1,600 updates and full observed physical states.
Sequence models use a 24-state context. Models predict increments and recursively
reuse their outputs without truth corrections. The known external forcing phase
for Van der Pol is advanced exactly. Polynomial degree 2/3 is selected on
validation and learned derivatives are integrated with RK4.

Checkpoint selection uses separate 180-step validation forecasts. The reference
long free horizons are 50 Lorenz time units and 200 Rössler/Van der Pol units.
Native time units differ between systems. Test examples have been inspected
throughout development: these are exploratory comparisons, not blind evaluation.

## Scores and diagnostics

Coordinate RMSE is the square root of the average squared prediction error.
Normalized full-state RMSE divides each coordinate's error by its training-data
standard deviation before averaging across coordinates, origins and time.
`point_by_cutoff` retains coordinate and normalized errors at recorded horizons.
The verifier reconstructs those scores directly from original members and truth,
using the recorded nonfinite/extreme-state handling of the scientific evaluator.

Current retrospective episodes use value, slope and curvature, training-derived
scales and a validation-memory policy. Slopes and curvatures use trailing
quadratic fits to current/past observations. The episode rule uses window 5τ,
occupancy 0.8 and passing gaps at most τ. It was development-calibrated on four
model cases on one inspected Van der Pol trajectory. Lorenz/Rössler have no
independent human visual labels. Preserve onset, confirmation, recovery and
censoring in `episode_coherence`; historical `coherence_*` fields may describe
superseded local checks. This is not a live confidence estimate or universal
predictability threshold. Recovery does not prove global phase coherence.

## Physical forecasting

Double pendulum state comprises two angles and two angular velocities. The
waterwheel state comprises angle, angular velocity and all 24 bucket masses.
Each has 32 training states, four validation states and three fixed test states
from separate RNG streams (1701/1702/1703). All 39 initial states per system are
unique. The reference example plus two fixed random states form the test group.

Truth uses SciPy DOP853, rtol 1e-10 and atol 1e-12. Sampling is 0.01 seconds for
the pendulum and 0.02 for the wheel. Training lasts 24 seconds. Validation/test
forecast origins follow a 4-second pendulum or 12-second wheel warmup. Full
forecast horizons are 25 and 45 seconds respectively. Warmup is observed history,
not part of the scored forecast.

- Residual MLP: original fit, two 128-unit SiLU layers, angle sine/cosine inputs,
  full-state increments, AdamW, batch 256, seed 7, 3,200 updates and four-step
  recursive loss.
- Transformer: two encoder layers, width 48, four attention heads, feed-forward
  width 96, sinusoidal positions, no dropout and the previous 24 states.
- GRU: two recurrent layers of 64 units and the same 24-state history.
- Polynomial dynamics: fixed quadratic polynomials of standardized periodic angle
  features, velocities and masses; least squares (rcond 1e-8), fourth-order finite
  difference derivative targets and RK4 with four substeps per sample.

The additional neural fits use the same optimizer, batch, seed, update budget,
four-step loss and 180-step validation checkpoint selection as the physical MLP.
Only training data determine feature and increment scales. Reconstructed observed
contexts are verified against the original arrays. MLP and polynomial models
consume the final observed state; sequence models consume the preceding 24.

Physical polynomial features approximate trigonometric, rational and exponential
equations; they are a different prior from the polynomial laboratory equations.
One neural seed cannot calibrate uncertainty. No physical equation coefficients
are supplied to the learned models.

## Original physical outputs and invalid states

`forecast.npz` always retains the entire raw numerical output, including failed
trajectories. `scientific-metrics.json` reports full and early (first 1.8-second)
errors, nonfinite-state counts, first nonfinite time and first negative bucket
mass time. Angle differences are wrapped to [-π,π]. Motion RMSE uses lower-bob
Euclidean separation or absolute angular-velocity difference. The exactly
observed initial sample is excluded from scores.

Nonfinite errors make a full score unavailable. Negative masses are reported
as invalid physical states independently of finite point errors. No shorter
trajectory is substituted for a full-horizon result, and no accuracy threshold
is treated as proof of chaos.
