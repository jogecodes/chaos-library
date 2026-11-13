"""Exercise optional neural training and recursive rollout on CPU and CUDA."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chaos-forecasting-ai"))
experiment = importlib.import_module("run_torch_experiments")
comparison = importlib.import_module("model_comparison")
select_device = importlib.import_module("torch_models").select_device


def test_unavailable_cuda_fails_and_auto_uses_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_and_rollout_on_selected_device(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Small batches still exercise gradients, PINN autograd, and CPU export.
    monkeypatch.setattr(experiment, "TRAIN_END", 32)
    time = np.linspace(0.0, 4.0, 64)
    states = np.column_stack((np.sin(time), np.cos(time), time / 4, -time / 4))
    forecast, loss = experiment.train_transformer(
        states, device=device, epochs=2, horizon=3
    )
    assert forecast.shape == (3, 4)
    assert np.isfinite(forecast).all() and np.isfinite(loss)
    assert np.isfinite(experiment.train_pinn(time, states, device=device, epochs=2))
    forecast = comparison.transformer_forecast(
        states[:32], states[32:56], device=device, epochs=2, horizon=3
    )
    assert forecast.shape == (3, 4)
    assert np.isfinite(forecast).all()
