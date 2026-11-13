"""PyTorch models for chaotic forecasting and physics-informed fitting."""

import torch
from torch import nn


def select_device(requested: str = "auto") -> torch.device:
    """Select CPU or CUDA, failing clearly for an unavailable explicit GPU."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, and cuda[:index]")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device.index}")
    return device


class SequenceTransformer(nn.Module):
    """Small causal Transformer predicting the next augmented state."""

    def __init__(
        self,
        state_size: int = 4,
        width: int = 32,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(state_size, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.output = nn.Linear(width, state_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Predict the next state from a state sequence."""
        length = sequence.shape[1]
        mask = torch.triu(
            torch.ones(length, length, device=sequence.device),
            diagonal=1,
        ).bool()
        encoded = self.encoder(self.input(sequence), mask=mask)
        return self.output(encoded[:, -1])


class ForcedVanDerPolPINN(nn.Module):
    """Network mapping time to position and velocity."""

    def __init__(self, width: int = 64, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Return position and velocity at queried times."""
        return self.network(time)
