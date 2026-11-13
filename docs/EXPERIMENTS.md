# Experiment index

Paths are relative to `chaos-forecasting-ai/results/`.

| Study | Directory | Numerical evidence |
| --- | --- | --- |
| Long-horizon laboratory | `long-horizon-2026-11-05/` | All nine original origins per system, datasets, member predictions, checkpoints, metrics, configs and split/episode audits |
| Physical MLP baseline | `physical-forecasting/run-2026-11-05/` | Original datasets, MLP weights, full forecasts, initial states, validation history and equation/checkpoint audit |
| Physical comparison | `physical-forecasting/comparison-2026-11-12/` | Reconstructed contexts, polynomial/Transformer/GRU weights, full outputs and scientific metrics |
| Controlled scaling | `data-scaling-2026-11-05/` | Fixed 1/8/32-trajectory protocol, checkpoints, full forecasts and fresh-test scores |
| Improved model/PINN | `improved-run-2026-11-05/` | Earlier forecasting and reconstruction/extrapolation evidence |
| Historical experiments | `full-run-2026-11-05/` and loose result files | Earlier numerical records with distinct protocols |

Scientific source is under `src/`, `chaos-forecasting-ai/` and `toy-models/`.
Each physical model's `scientific-metrics.json` describes its full raw output.
The original arrays/weights are unchanged; portable runner and metadata changes
are recorded in `provenance/selection.json`.
