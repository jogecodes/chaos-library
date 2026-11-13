# Chaos Library

Repository: [jogecodes/chaos-library](https://github.com/jogecodes/chaos-library).

Reproducible numerical experiments in deterministic chaos and forecasting by
Jorge Casajús-Setién ([jogecodes](https://github.com/jogecodes)).

The library contains simulation equations, model implementations, training and
evaluation runners, original datasets, trained checkpoints, full predictions,
scientific metrics, numerical audits and regression tests.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ai]"
python scripts/verify_library.py
pytest
```

Python 3.10+ is required. NumPy/SciPy support simulation and metric verification;
PyTorch supports neural fitting and checkpoint replay; Matplotlib generates
scientific figures. A GPU is optional and CUDA tests skip when unavailable.
On Windows activate `.venv\Scripts\activate` instead. The recorded validation
environment is in [provenance/environment.json](provenance/environment.json).
Installing this project provides the `chaos_toy_models` Python package.

```python
from chaos_toy_models.flows import integrate_flow, lorenz

time, trajectory = integrate_flow(lorenz, (1., 1., 1.), (0., 40.), dt=0.01)
```

## Contents

| Path | Purpose |
| --- | --- |
| `src/chaos_toy_models/` | Maps, continuous flows, mechanical systems, circuits and numerical diagnostics |
| `toy-models/` | Model descriptions, derivations, scientific plots and plotting scripts |
| `chaos-forecasting-ai/` | Forecasting models, training and evaluation runners |
| `chaos-forecasting-ai/physical_forecasting/` | Standalone MLP, polynomial, Transformer and GRU physical experiments |
| `chaos-forecasting-ai/results/` | Original numerical datasets, checkpoints, full predictions and audits |
| `scripts/verify_library.py` | Checksums and independent checks against original scientific metrics |
| `tests/` | Simulation, model, splitting and training regression tests |
| `docs/` | Methods, scientific findings and reproduction commands |
| `provenance/` | Source revisions, selection inventory, checksums and preparation audit |

## Experiments

- Lorenz, Rössler and forced Van der Pol: polynomial dynamics, Transformer, GRU
  and residual MLP; 32 training trajectories, four validation initial conditions,
  three neural seeds and long free forecasts. All nine original forecast origins
  are retained, including overlapping windows.
- Double pendulum and 24-bucket waterwheel: the same four model families,
  32 training trajectories, four validation trajectories and three fixed test
  trajectories per system; one neural fitting seed. Full failed predictions are
  retained, including nonfinite states and negative mass predictions.
- Supporting experiments: controlled training-data scaling with 1/8/32
  trajectories, earlier benchmarks and PINN reconstruction versus extrapolation.

Start with the [reproduction guide](docs/REPRODUCING.md),
[methods](docs/METHODS.md), [experiment index](docs/EXPERIMENTS.md) and
[findings](docs/CURRENT_FINDINGS.md). Write fresh fits under ignored
`reproductions/` so the retained scientific evidence stays unchanged.

## Evidence and limitations

`python scripts/verify_library.py` checks publication checksums, the laboratory
metrics across the 108 original model/origin combinations, and all 24 physical
forecasts directly from their original NumPy arrays. Physical scores are computed
on full saved horizons; nonfinite values make full metrics unavailable rather
than being removed. Negative bucket masses are reported separately. No shortened
trajectory is substituted for the full scientific record.

These are exploratory noiseless simulations, not measurements of real apparatus.
Test examples have been inspected during development. Overlapping forecast
windows are dependent, and different initial states still share equations and
parameters. Retrospective episode diagnostics are development-calibrated;
ensemble trajectories are not calibrated uncertainty intervals. Native time
units differ across systems. Reconstruction and extrapolation are distinct tasks.

The numerical arrays and checkpoint bytes retain their source hashes. Read
[publication scope](docs/PUBLICATION.md) for the deliberate selection, source
adaptations and omitted historical development archives.

## License

MIT. The original copyright and license are preserved in [LICENSE](LICENSE).
