"""Controlled data scaling with windows that never cross simulation boundaries."""

import copy

import numpy as np
import torch
from improved_models import NeuralForecaster, make_residual_model
from torch_models import select_device


def trajectory_pool(trajectories, window=24, steps=4):
    """Return joined storage and valid origins, preserving trajectory boundaries."""
    arrays = [np.asarray(trajectory) for trajectory in trajectories]
    if not arrays or any(len(a) < window + steps for a in arrays):
        raise ValueError("Each trajectory must contain a context and rollout targets")
    if any(a.ndim != 2 for a in arrays) or len({a.shape[1] for a in arrays}) != 1:
        raise ValueError("Trajectories must have matching state dimensions")
    origins, offset = [], 0
    for array in arrays:
        origins.append(np.arange(offset + window, offset + len(array) - steps + 1))
        offset += len(array)
    return np.concatenate(arrays), np.concatenate(origins)


def fit_trajectories(
    trajectories,
    validation_contexts,
    validation_truth,
    *,
    seed=7,
    updates=1600,
    batch_size=256,
    device="auto",
    family="transformer",
    phase_increment=1.68 * 0.05,
):
    """Fix optimizer updates across data amounts; select only on separate episodes."""
    if updates < 1 or batch_size < 1:
        raise ValueError("updates and batch_size must be positive")
    values, origins = trajectory_pool(trajectories)
    device = select_device(device)
    torch.manual_seed(seed)
    mean, scale = values.mean(0), np.maximum(values.std(0), 1e-6)
    model = make_residual_model(family, mean, scale, phase_increment).to(device)
    physical_size = model.physical_size
    forecaster = NeuralForecaster(model, device)
    tensors = torch.tensor(values, dtype=torch.float32, device=device)
    indices = torch.tensor(origins, device=device)
    lags = torch.arange(-24, 0, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates)
    best, logs, total = float("inf"), [], 0.0
    for update in range(1, updates + 1):
        model.train()
        selection = torch.randint(len(indices), (batch_size,), device=device)
        starts = indices[selection]
        history = tensors[starts[:, None] + lags]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for lead in range(4):
            predicted = model(history)
            target = tensors[starts + lead, :physical_size]
            loss = (
                loss
                + (
                    (
                        (predicted[:, :physical_size] - target)
                        / model.scale[:physical_size]
                    )
                    ** 2
                ).mean()
            )
            history = torch.cat((history[:, 1:], predicted[:, None]), dim=1)
        loss = loss / 4
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        if update == 1 or update % 200 == 0 or update == updates:
            predicted = forecaster.rollout_batch(
                validation_contexts, validation_truth.shape[1]
            )
            score = float(
                np.mean(
                    (
                        (
                            predicted[:, :, :physical_size]
                            - validation_truth[:, :, :physical_size]
                        )
                        / scale[:physical_size]
                    )
                    ** 2
                )
            )
            if not np.isfinite(score):
                raise RuntimeError("Nonfinite validation rollout")
            if score < best:
                best, selected = score, update
                checkpoint = copy.deepcopy(model.state_dict())
            logs.append(
                {
                    "update": update,
                    "training_loss": total / update,
                    "validation_normalized_rollout_mse": score,
                }
            )
            print(
                f"  seed {seed}, update {update}/{updates}: validation={score:.5f}, "
                f"best={best:.5f} at {selected}",
                flush=True,
            )
    model.load_state_dict(checkpoint)
    model.eval()
    return forecaster, {
        "seed": seed,
        "family": family,
        "physical_size": physical_size,
        "phase_increment": phase_increment,
        "updates": updates,
        "selected_update": selected,
        "batch_size": batch_size,
        "trajectory_count": len(trajectories),
        "training_samples": len(values),
        "training_windows": len(origins),
        "validation_normalized_rollout_mse": best,
        "history": logs,
    }
