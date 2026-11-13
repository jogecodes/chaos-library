# Reproducing the numerical experiments

Run from the library root after installing `.[dev,ai]`. Choose a fresh output
directory for every new experiment. Solver, hardware and PyTorch differences can
change numerical results; the retained arrays are the exact original record,
not a promise of bitwise-identical GPU retraining.

## Verification

```bash
python scripts/verify_library.py
pytest
```

The verifier uses original datasets, model outputs and scientific metrics. It
does not fit models or load checkpoint pickle objects. Tests include short
training checks on synthetic data. CUDA-specific tests skip when unavailable.

## Physical baseline and four-family comparison

```bash
python chaos-forecasting-ai/physical_forecasting/train_physical.py --artifacts reproductions/physical/new-baseline --device cpu
python chaos-forecasting-ai/physical_forecasting/train_comparison.py --source reproductions/physical/new-baseline --artifacts reproductions/physical/new-comparison --system pendulum --device cpu
python chaos-forecasting-ai/physical_forecasting/train_comparison.py --source reproductions/physical/new-baseline --artifacts reproductions/physical/new-comparison --system waterwheel --device cpu
```

The first runner writes training/validation/test arrays, original MLP checkpoints,
full forecasts, configs, metrics and source snapshots in the artifact directory.
The comparison runner reads that baseline, reconstructs observed contexts and
fits polynomial, Transformer and GRU models. Use `--device cuda` on an available
GPU. Reference fits use 3,200 updates and seed 7; checkpoints are selected only on
separate 180-step validation forecasts. The comparison runner verifies that its
reconstructed contexts end at the original forecast origin.

To fit only the additional models from the shipped baseline:

```bash
python chaos-forecasting-ai/physical_forecasting/train_comparison.py --source chaos-forecasting-ai/results/physical-forecasting/run-2026-11-05 --artifacts reproductions/physical/comparison --system pendulum --device cpu
python chaos-forecasting-ai/physical_forecasting/train_comparison.py --source chaos-forecasting-ai/results/physical-forecasting/run-2026-11-05 --artifacts reproductions/physical/comparison --system waterwheel --device cpu
```

Completed model families are reused when their summary exists. Choose a new
artifact directory when changing settings. These are full scientific fits,
not quick smoke tests.

## Laboratory and controlled data scaling

```bash
python chaos-forecasting-ai/run_data_scaling.py --device cpu --output-dir reproductions/data-scaling
python chaos-forecasting-ai/run_long_horizon.py --device cpu --previous reproductions/data-scaling --output-dir reproductions/long-horizon
```

Defaults are 1,600 updates and seeds 7/11/23. The laboratory runner can reuse
Van der Pol models from the selected scaling run. Use `--help` to inspect the
system and output options, and retain generated configuration files with results.

## Illustrative numerical models

```bash
python scripts/generate_figures.py
python toy-models/physical/waterwheel/plot_phase_space.py
```

Figure scripts write to their documented figure directories. Regeneration may
change the local working tree; keep the checksum inventory if you need to verify
the original selection.
