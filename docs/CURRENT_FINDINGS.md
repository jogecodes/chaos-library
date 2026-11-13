# Scientific findings and limits

- Controlled Van der Pol training-data scaling with 1/8/32 trajectories reduced
  fresh-test Transformer position RMSE from 1.06856 to 0.44486 and 0.39344 at a
  fixed architecture and optimizer-update budget. It tests data coverage under
  the same equations, not robustness to different systems or parameters.
- Polynomial dynamics benefits from a prior compatible with the laboratory's
  low-order equations. The results do not establish a universal ranking.
- PINN reconstruction of held-out observations inside its fitted interval was
  accurate (reference position RMSE approximately 0.00119), but extrapolation was
  poor (approximately 2.90639 over t=20..25). These are different tasks/intervals.
- Exact paths can diverge while broad attractor geometry remains recognizable.
  Distribution or geometry agreement alone does not establish correct dynamics.
- Temporary discrepancy and recovery complicate a single first-threshold-breach
  definition of forecast failure. The retrospective episode rule remains
  development-calibrated and sensitive to its settings.
- The physical comparison fits polynomial dynamics, Transformer, GRU and MLP
  separately to each of the pendulum and bucket-waterwheel systems. Full raw
  outputs, including nonfinite values and negative masses, are retained. Inspect
  `scientific-metrics.json` beside each fitted model for results on the actual
  complete forecast horizon.

All studies use simulations and exploratory, now-inspected test examples.
Overlapping forecast windows are dependent. Consult METHODS.md and each saved
configuration before interpreting metrics from different protocols.
