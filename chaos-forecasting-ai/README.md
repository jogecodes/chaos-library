# Forecasting experiments

Simulation-data forecasting models, training/evaluation runners and original
numerical records for Lorenz, Rössler, forced Van der Pol, the double pendulum
and a 24-bucket waterwheel.

Read [methods](../docs/METHODS.md), [experiment index](../docs/EXPERIMENTS.md) and
[reproduction commands](../docs/REPRODUCING.md). New fits should write to ignored
`reproductions/` rather than overwriting the retained `results/` selection.

Ridge/persistence models and earlier PINN protocols remain for historical
comparisons and regression tests. They are distinct from the four active
polynomial/Transformer/GRU/MLP families. Reconstruction, extrapolation, overlapping
rolling windows and fresh-initial-state evaluation must not be treated as a
single benchmark ranking.
