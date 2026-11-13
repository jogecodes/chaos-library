"""Train the Transformer and PINN experiments."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import PARAMETERS, generate_series
from models import make_windows
from torch_models import ForcedVanDerPolPINN, SequenceTransformer, select_device

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
WINDOW = 24
TRAIN_END = 2_000


def train_transformer(
    states: np.ndarray,
    *,
    device: str = "auto",
    epochs: int = 35,
    horizon: int = 180,
) -> tuple[np.ndarray, float]:
    """Train a small one-step Transformer and recursively forecast."""
    torch.manual_seed(7)
    device = select_device(device)
    window = WINDOW
    features, targets = make_windows(states[:TRAIN_END], window)
    inputs = torch.tensor(
        features.reshape(-1, window, states.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(targets, dtype=torch.float32, device=device)
    model = SequenceTransformer(state_size=states.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_function = torch.nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    history = torch.tensor(
        states[TRAIN_END : TRAIN_END + window], dtype=torch.float32, device=device
    )
    predictions = []
    with torch.no_grad():
        for _ in range(horizon):
            next_state = model(history[-window:].unsqueeze(0))
            predictions.append(next_state[0].cpu().numpy())
            history = torch.cat((history, next_state), dim=0)
    return np.asarray(predictions), float(loss.item())


def train_pinn(
    time: np.ndarray,
    states: np.ndarray,
    *,
    device: str = "auto",
    epochs: int = 800,
    return_model: bool = False,
) -> float | tuple[torch.nn.Module, float]:
    """Fit a short trajectory while penalising the forced Van der Pol residual."""
    torch.manual_seed(11)
    device = select_device(device)
    mask = time <= 20.0
    observed_time = torch.tensor(time[mask, None], dtype=torch.float32, device=device)
    observed_states = torch.tensor(states[mask, :2], dtype=torch.float32, device=device)
    collocation = torch.linspace(
        0.0, 20.0, 500, device=device
    )[:, None].requires_grad_()
    model = ForcedVanDerPolPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted = model(observed_time)
        data_loss = torch.mean((predicted - observed_states) ** 2)
        values = model(collocation)
        position, velocity = values[:, 0], values[:, 1]
        derivative = torch.autograd.grad(
            velocity,
            collocation,
            torch.ones_like(velocity),
            create_graph=True,
        )[0][:, 0]
        forcing = PARAMETERS["amplitude"] * torch.cos(
            PARAMETERS["forcing_frequency"] * collocation[:, 0]
        )
        residual = (
            derivative
            - PARAMETERS["mu"] * (1.0 - position**2) * velocity
            + position
            - forcing
        )
        loss = data_loss + 0.1 * torch.mean(residual**2)
        loss.backward()
        optimizer.step()
    if return_model:
        return model, float(loss.item())
    return float(loss.item())


def main() -> None:
    """Run both PyTorch experiments and save metrics and a comparison plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--transformer-epochs", type=int, default=35)
    parser.add_argument("--pinn-epochs", type=int, default=800)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if min(args.transformer_epochs, args.pinn_epochs, args.horizon) < 1:
        parser.error("Epoch counts and horizon must be positive")
    device = select_device(args.device)
    results = args.output_dir
    time, states = generate_series(
        duration=max(180.0, (TRAIN_END + WINDOW + args.horizon) * 0.05), dt=0.05
    )
    transformer_forecast, transformer_loss = train_transformer(
        states, device=str(device), epochs=args.transformer_epochs, horizon=args.horizon
    )
    pinn_loss = train_pinn(
        time, states, device=str(device), epochs=args.pinn_epochs
    )
    forecast_start = TRAIN_END + WINDOW
    truth = states[forecast_start : forecast_start + args.horizon]
    metrics = {
        "transformer_training_loss": transformer_loss,
        "transformer_position_rmse": float(
            np.sqrt(np.mean((transformer_forecast[:, 0] - truth[:, 0]) ** 2))
        ),
        "pinn_training_loss": pinn_loss,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torch_version": torch.__version__,
        "transformer_epochs": args.transformer_epochs,
        "pinn_epochs": args.pinn_epochs,
        "horizon": args.horizon,
        "forecast_start_index": forecast_start,
    }
    (results / "figures").mkdir(parents=True, exist_ok=True)
    (results / "torch-metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        results / "torch-forecasts.npz",
        truth=truth,
        transformer=transformer_forecast,
    )
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(truth[:, 0], label="truth", color="#5ee7f2")
    axis.plot(transformer_forecast[:, 0], label="Transformer", color="#f4c95d")
    axis.set(
        xlabel="forecast lead",
        ylabel="position x",
        title="Numeric Transformer on forced Van der Pol",
    )
    axis.legend()
    axis.grid(alpha=0.16)
    figure.tight_layout()
    figure.savefig(results / "figures" / "transformer-forecast.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
