# Physical forecasting experiments

`train_physical.py` generates the fixed train/validation/test split, fits original
residual-MLP models, and saves datasets, checkpoints and full numerical forecasts.
`train_comparison.py` reads a baseline and fits polynomial dynamics, Transformer
and GRU models independently for each system. `physical_io.py` provides atomic
JSON writing. `evaluation.py` computes scientific metrics without truncating
raw forecasts or hiding invalid physical states.

Both runners need only the simulation and training dependencies. Output is
entirely controlled by `--artifacts`; original source arrays are read only.
See [the reproduction guide](../../docs/REPRODUCING.md) for complete commands.

Retained evidence lives in `../results/physical-forecasting/run-2026-11-05/` and
`../results/physical-forecasting/comparison-2026-11-12/`. New experiments belong
under ignored `reproductions/`. No real-apparatus, noise-robustness or calibrated
uncertainty claim is made.
