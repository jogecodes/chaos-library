"""Validated residual forecasting and equation-constrained reconstruction."""

import copy
import math

import numpy as np
import torch
from data import PARAMETERS
from torch import nn
from torch_models import select_device


class ResidualTransformer(nn.Module):
    """Predict state increments, optionally advancing a known external phase."""

    def __init__(
        self,
        mean,
        scale,
        window=24,
        width=48,
        phase_increment=PARAMETERS["forcing_frequency"] * 0.05,
    ):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        positions = torch.arange(window)[:, None]
        rates = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        encoding = torch.zeros(window, width)
        encoding[:, 0::2] = torch.sin(positions * rates)
        encoding[:, 1::2] = torch.cos(positions * rates)
        self.register_buffer("position", encoding)
        self.phase_increment = phase_increment
        self.physical_size = len(mean) if phase_increment is None else len(mean) - 2
        self.input = nn.Linear(len(mean), width)
        layer = nn.TransformerEncoderLayer(
            width, 4, 2 * width, dropout=0.0, batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        # Independently initialize the cloned encoder layers.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.output = nn.Linear(width, self.physical_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history):
        encoded = self.input((history - self.mean) / self.scale)
        encoded = self.encoder(encoded + self.position[-history.shape[1] :])
        physical = (
            history[:, -1, : self.physical_size]
            + self.output(encoded[:, -1]) * self.scale[: self.physical_size]
        )
        return append_known_phase(physical, history[:, -1], self.phase_increment)


def append_known_phase(physical, last, increment):
    """Append an externally known phase; autonomous systems need no phase state."""
    if increment is None:
        return physical
    sine, cosine = last[:, -2], last[:, -1]
    phase = torch.stack(
        (
            sine * math.cos(increment) + cosine * math.sin(increment),
            cosine * math.cos(increment) - sine * math.sin(increment),
        ),
        dim=-1,
    )
    return torch.cat((physical, phase), dim=-1)


class ResidualMLP(nn.Module):
    """State-to-increment network without hand-chosen polynomial features."""

    def __init__(self, mean, scale, phase_increment=None):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.phase_increment = phase_increment
        self.physical_size = len(mean) if phase_increment is None else len(mean) - 2
        self.network = nn.Sequential(
            nn.Linear(len(mean), 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, self.physical_size),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, history):
        last = history[:, -1]
        delta = self.network((last - self.mean) / self.scale)
        physical = (
            last[:, : self.physical_size] + delta * self.scale[: self.physical_size]
        )
        return append_known_phase(physical, last, self.phase_increment)


class ResidualGRU(nn.Module):
    """Recurrent residual predictor over the same context as the Transformer."""

    def __init__(self, mean, scale, phase_increment=None):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.phase_increment = phase_increment
        self.physical_size = len(mean) if phase_increment is None else len(mean) - 2
        self.encoder = nn.GRU(len(mean), 64, num_layers=2, batch_first=True)
        self.output = nn.Linear(64, self.physical_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history):
        encoded, _ = self.encoder((history - self.mean) / self.scale)
        physical = (
            history[:, -1, : self.physical_size]
            + self.output(encoded[:, -1]) * self.scale[: self.physical_size]
        )
        return append_known_phase(physical, history[:, -1], self.phase_increment)


def make_residual_model(family, mean, scale, phase_increment=None):
    """Build one explicitly configured architecture for safe checkpoint replay."""
    models = {
        "transformer": ResidualTransformer,
        "mlp": ResidualMLP,
        "gru": ResidualGRU,
    }
    if family not in models:
        raise ValueError(f"Unknown forecasting family: {family}")
    return models[family](mean, scale, phase_increment=phase_increment)


class NeuralForecaster:
    """Batched recursive rollout retaining only the required context."""

    def __init__(self, model, device, window=24):
        self.model, self.device, self.window = model, device, window

    def rollout_batch(self, contexts, horizon):
        history = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        self.model.eval()
        output = []
        with torch.no_grad():
            for _ in range(horizon):
                value = self.model(history)
                output.append(value)
                history = torch.cat((history, value[:, None]), dim=1)[:, -self.window :]
        return torch.stack(output, dim=1).cpu().numpy()

    def rollout(self, context, horizon):
        return self.rollout_batch(np.asarray(context)[None], horizon)[0]


def fit_transformer(train, *, seed=7, epochs=100, device="auto", batch_size=128):
    """Use only an inner chronological validation split for checkpoint selection."""
    if epochs < 1 or len(train) < 500:
        raise ValueError("Need positive epochs and at least 500 training samples")
    device = select_device(device)
    torch.manual_seed(seed)
    split, window, steps = int(0.8 * len(train)), 24, 4
    mean = train[:split].mean(axis=0)
    scale = np.maximum(train[:split].std(axis=0), 1e-6)
    model = ResidualTransformer(mean, scale).to(device)
    values = torch.as_tensor(train, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    origins = torch.arange(window, split - steps + 1, device=device)
    lags = torch.arange(-window, 0, device=device)
    valid_origins = np.arange(split + window, len(train) - 60 + 1, 60)
    contexts = np.asarray([train[o - window : o] for o in valid_origins])
    truth = np.asarray([train[o : o + 60, :2] for o in valid_origins])
    best, best_state, stale, log = float("inf"), None, 0, []
    forecaster = NeuralForecaster(model, device)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total, count = 0.0, 0
        for indices in origins[torch.randperm(len(origins), device=device)].split(
            batch_size
        ):
            history = values[indices[:, None] + lags]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for lead in range(steps):
                predicted = model(history)
                target = values[indices + lead, :2]
                loss = loss + torch.mean(
                    ((predicted[:, :2] - target) / model.scale[:2]) ** 2
                )
                history = torch.cat((history[:, 1:], predicted[:, None]), dim=1)
            loss = loss / steps
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite Transformer training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_total += loss.item() * len(indices)
            count += len(indices)
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            prediction = forecaster.rollout_batch(contexts, 60)[:, :, :2]
            score = float(np.mean(((prediction - truth) / scale[:2]) ** 2))
            log.append(
                {
                    "epoch": epoch,
                    "training_loss": loss_total / count,
                    "validation_rollout_mse": score,
                }
            )
            if score < best:
                best, stale = score, 0
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            else:
                stale += 1
            if stale >= 6:
                break
    model.load_state_dict(best_state)
    model.eval()
    return forecaster, {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "fit_samples": split,
        "validation_samples": len(train) - split,
        "validation_normalized_rollout_mse": best,
        "history": log,
    }


class FourierPINN(nn.Module):
    """Normalized time/state network with smooth Fourier features."""

    def __init__(self, mean, scale, duration=20.0):
        super().__init__()
        self.duration = duration
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("frequencies", torch.linspace(0.25, 5.0, 16))
        self.network = nn.Sequential(
            nn.Linear(33, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, time):
        phase = time * self.frequencies
        features = torch.cat(
            (2 * time / self.duration - 1, phase.sin(), phase.cos()), dim=1
        )
        return self.mean + self.scale * self.network(features)


def pinn_residuals(model, time):
    """Return both first-order governing-equation residuals in physical units."""
    values = model(time)
    gradients = [
        torch.autograd.grad(
            values[:, i], time, torch.ones_like(values[:, i]), create_graph=True
        )[0][:, 0]
        for i in range(2)
    ]
    x, v = values[:, 0], values[:, 1]
    kinematic = gradients[0] - v
    dynamic = (
        gradients[1]
        - PARAMETERS["mu"] * (1 - x * x) * v
        + x
        - PARAMETERS["amplitude"]
        * torch.cos(PARAMETERS["forcing_frequency"] * time[:, 0])
    )
    return values, kinematic, dynamic


def fit_pinn(time, states, *, epochs=4000, seed=11, device="auto"):
    """Fit sparse observations up to t=20; reserve interpolation and t>20 tests."""
    device = select_device(device)
    torch.manual_seed(seed)
    observed = np.flatnonzero(time <= 20.0)[::4]
    candidates = np.flatnonzero(time <= 20.0)
    validation = candidates[2::4]
    interpolation = candidates[1::2]
    if epochs < 1 or len(validation) == 0 or not np.any((time > 20) & (time <= 25)):
        raise ValueError("Need positive epochs and a series spanning t=0 through t>20")
    mean, scale = states[observed, :2].mean(0), states[observed, :2].std(0)
    model = FourierPINN(mean, np.maximum(scale, 1e-6)).to(device)
    times = torch.tensor(time[:, None], dtype=torch.float32, device=device)
    truth = torch.tensor(states[:, :2], dtype=torch.float32, device=device)
    collocation = torch.linspace(0, 20, 401, device=device)[:, None].requires_grad_()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=1e-5
    )
    best, log = float("inf"), []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        collocation.grad = None
        data_loss = (
            ((model(times[observed]) - truth[observed]) / model.scale) ** 2
        ).mean()
        values, kinematic, dynamic = pinn_residuals(model, collocation)
        kin_loss = (kinematic / model.scale[1]).square().mean()
        dyn_loss = (dynamic / model.scale[1]).square().mean()
        initial_loss = (((model(times[:1]) - truth[:1]) / model.scale) ** 2).mean()
        # Warm up the data fit, then enforce both equations with fixed final weights.
        weight = 0.1 * min(1.0, epoch / 500)
        loss = data_loss + weight * (kin_loss + dyn_loss) + initial_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite PINN training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            with torch.no_grad():
                val = (
                    (
                        ((model(times[validation]) - truth[validation]) / model.scale)
                        ** 2
                    )
                    .mean()
                    .item()
                )
            log.append(
                {
                    "epoch": epoch,
                    "data_loss": data_loss.item(),
                    "kinematic_loss": kin_loss.item(),
                    "dynamic_loss": dyn_loss.item(),
                    "validation_normalized_mse": val,
                }
            )
            if val < best:
                best, best_epoch = val, epoch
                checkpoint = copy.deepcopy(model.state_dict())
    model.load_state_dict(checkpoint)
    with torch.no_grad():
        prediction = model(times).cpu().numpy()
    _, kin, dyn = pinn_residuals(model, collocation)

    def rmse(indices):
        return np.sqrt(
            np.mean((prediction[indices] - states[indices, :2]) ** 2, axis=0)
        ).tolist()

    metrics = {
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "observed_samples": len(observed),
        "validation_samples": len(validation),
        "validation_rmse_x_v": rmse(validation),
        "interpolation_samples": len(interpolation),
        "observation_end": 20.0,
        "interpolation_rmse_x_v": rmse(interpolation),
        "extrapolation_20_25_rmse_x_v": rmse((time > 20) & (time <= 25)),
        "kinematic_residual_rmse": float(kin.square().mean().sqrt().item()),
        "dynamic_residual_rmse": float(dyn.square().mean().sqrt().item()),
        "history": log,
    }
    return model, prediction, metrics
