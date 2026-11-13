"""Long free forecasts across Van der Pol, Lorenz, and Rössler model families."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from data import generate_series
from improved_models import NeuralForecaster, make_residual_model
from long_horizon_metrics import evaluate_long_run, validation_coherence_settings
from long_horizon_models import CubicDynamics
from split_integrity import audit_long_horizon_split
from torch_models import select_device
from trajectory_training import fit_trajectories

from chaos_toy_models.flows import integrate_flow, lorenz, rossler

ROOT = Path(__file__).resolve().parent
SYSTEMS = {
    "van-der-pol": {
        "dt": 0.05,
        "physical_size": 2,
        "phase_increment": 1.68 * 0.05,
        "forcing_frequency": 1.68,
        "long_units": 200,
        "labels": ["x", "v"],
        "seed": 20261105,
        "reference_initial": [0.1, 0.0],
    },
    "lorenz": {
        "dt": 0.01,
        "physical_size": 3,
        "phase_increment": None,
        "forcing_frequency": None,
        "long_units": 50,
        "labels": ["x", "y", "z"],
        "seed": 20261115,
        "reference_initial": [1, 1, 1],
    },
    "rossler": {
        "dt": 0.05,
        "physical_size": 3,
        "phase_increment": None,
        "forcing_frequency": None,
        "long_units": 200,
        "labels": ["x", "y", "z"],
        "seed": 20261125,
        "reference_initial": [0.1, 0, 0.1],
    },
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def simulate(system, initial, duration):
    cfg = SYSTEMS[system]
    if system == "van-der-pol":
        return generate_series(
            duration=duration, dt=cfg["dt"], initial_state=tuple(initial)
        )
    function = lorenz if system == "lorenz" else rossler
    return integrate_flow(function, initial, (0, duration), dt=cfg["dt"])


def examples(trajectories, origins, horizon=180):
    return (
        np.asarray([s[o - 24 : o] for s in trajectories for o in origins]),
        np.asarray([s[o : o + horizon] for s in trajectories for o in origins]),
    )


def prepare(system, out, previous):
    cfg = SYSTEMS[system]
    if (out / "data.npz").exists():
        return np.load(out / "data.npz"), json.loads(
            (out / "initials.json").read_text()
        )
    if system == "van-der-pol":
        pool = np.load(previous / "simulation-pool.npz")
        old = json.loads((previous / "config.json").read_text())["initial_conditions"]
        train, validation = pool["train"], pool["validation"]
        initials = {
            "train": old["train"],
            "validation": old["validation"],
            "test": old["fresh_test"],
        }
    else:
        rng = np.random.default_rng(cfg["seed"])
        if system == "lorenz":
            initial = rng.uniform([-15, -15, 5], [15, 15, 35], size=(40, 3))
        else:
            initial = rng.uniform([-5, -5, 0.1], [5, 5, 2], size=(40, 3))
        initials = {
            "train": initial[:32].tolist(),
            "validation": initial[32:36].tolist(),
            "test": initial[36:40].tolist(),
        }
        train, validation = [], []
        for group, target in [("train", train), ("validation", validation)]:
            for i, ic in enumerate(initials[group]):
                _, state = simulate(system, ic, 2000 * cfg["dt"])
                target.append(state)
                if (i + 1) % 8 == 0 or i + 1 == len(initials[group]):
                    print(
                        f"{system}: simulated {group} {i + 1}/{len(initials[group])}",
                        flush=True,
                    )
        train, validation = np.asarray(train), np.asarray(validation)
    horizon = int(round(cfg["long_units"] / cfg["dt"]))
    origins = [424, 1424]
    duration = (max(origins) + horizon + 10) * cfg["dt"]
    tests = []
    for initial in initials["test"]:
        _, state = simulate(system, initial, duration)
        tests.append(state)
    contexts, truth = examples(tests, origins, horizon)
    # An additional reference origin supplies an easy-to-identify plot.
    reference_origin = 2620 if system == "van-der-pol" else 424
    _, reference = simulate(
        system, cfg["reference_initial"], (reference_origin + horizon + 10) * cfg["dt"]
    )
    contexts = np.concatenate(
        (reference[None, reference_origin - 24 : reference_origin], contexts)
    )
    truth = np.concatenate(
        (reference[None, reference_origin : reference_origin + horizon], truth)
    )
    save_json(out / "initials.json", initials)
    np.savez_compressed(
        out / "data.npz",
        train=train,
        validation=validation,
        test_contexts=contexts,
        truth=truth,
        reference=reference,
        reference_origin=np.asarray(reference_origin),
        fresh_trajectories=np.asarray(tests),
    )
    return np.load(out / "data.npz"), initials


def run_system(system, args, device):
    from plot_long_horizon import plot_system

    out = args.output_dir / system
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    cfg = SYSTEMS[system]
    data, initials = prepare(system, out, args.previous)
    split_audit = audit_long_horizon_split(data, initials, cfg["reference_initial"])
    save_json(out / "split-audit.json", split_audit)
    train, validation = data["train"], data["validation"]
    val_contexts, val_truth = examples(validation, [424, 824, 1224, 1624])
    scale = np.maximum(train.reshape(-1, train.shape[-1]).std(0), 1e-6)
    horizon = data["truth"].shape[1]
    config = {
        **cfg,
        "updates": args.updates,
        "batch_size": 256,
        "model_seeds": args.seeds,
        "training_trajectories": 32,
        "training_samples": int(train.shape[0] * train.shape[1]),
        "validation_trajectories": 4,
        "fresh_test_trajectories": 4,
        "test_origins": 9,
        "reference_origin_index": 0,
        "device": str(device),
        "torch": torch.__version__,
        "hardware": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "prediction": "Free recursive rollout; no truth updates after context.",
        "long_uncertainty": (
            "No calibration claim beyond the prior 9-unit intervals. "
            "Neural members and mean are displayed separately."
        ),
        "selection": (
            "180-step validation only; all long test outputs are "
            "evaluated after fits are fixed."
        ),
        "sampling_note": (
            "Two long, overlapping origins per fresh trajectory plus "
            "one reference; dependent origins, not nine independent "
            "trajectories."
        ),
        "climate_note": (
            "Single-member distribution metrics avoid averaging "
            "unrelated chaotic phases."
        ),
        "numerical_limit": (
            "Nonfinite states or |state/training_std|>1e6 are explicit "
            "numerical failures; no clipping into a plausible range."
        ),
    }
    save_json(out / "config.json", config)
    models, training = {}, {}
    print(
        f"{system}: models and {cfg['long_units']}-unit forecasts",
        flush=True,
    )
    poly_trials = []
    for degree in [2, 3]:
        polynomial = CubicDynamics(
            cfg["physical_size"], cfg["dt"], cfg["forcing_frequency"], degree
        ).fit(train)
        pred = polynomial.rollout_batch(val_contexts, 180)
        score = float(
            np.mean(
                (
                    (
                        pred[:, :, : cfg["physical_size"]]
                        - val_truth[:, :, : cfg["physical_size"]]
                    )
                    / scale[: cfg["physical_size"]]
                )
                ** 2
            )
        )
        poly_trials.append(
            {
                "degree": degree,
                "validation_normalized_mse": score if np.isfinite(score) else None,
            }
        )
    best = min(
        [t for t in poly_trials if t["validation_normalized_mse"] is not None],
        key=lambda t: t["validation_normalized_mse"],
    )
    polynomial = CubicDynamics(
        cfg["physical_size"], cfg["dt"], cfg["forcing_frequency"], best["degree"]
    ).fit(train)
    np.save(out / "checkpoints/polynomial.npy", polynomial.coefficients)
    models["Polynomial dynamics"] = [polynomial]
    save_json(
        out / "baseline-validation.json",
        {
            "polynomial": poly_trials,
            "polynomial_degree": best["degree"],
        },
    )
    for family in ["transformer", "gru", "mlp"]:
        members = []
        training[family] = []
        for seed in args.seeds:
            path = out / f"checkpoints/{family}-{seed}.pt"
            record = out / f"checkpoints/{family}-{seed}.json"
            source = (
                (args.previous / f"checkpoints/transformer-32-{seed}.pt")
                if system == "van-der-pol" and family == "transformer"
                else path
            )
            if source.exists():
                model = make_residual_model(
                    family,
                    np.zeros(train.shape[-1]),
                    np.ones(train.shape[-1]),
                    cfg["phase_increment"],
                ).to(device)
                model.load_state_dict(
                    torch.load(source, map_location=device, weights_only=True)
                )
                forecaster = NeuralForecaster(model, device)
                history = (
                    json.loads(record.read_text())
                    if record.exists()
                    else {
                        "seed": seed,
                        "source": str(source),
                        "note": "Previously selected 32-trajectory checkpoint.",
                    }
                )
                print(f"{system}: loaded {family} seed {seed}", flush=True)
            else:
                print(f"{system}: train {family} seed {seed}", flush=True)
                forecaster, history = fit_trajectories(
                    train,
                    val_contexts,
                    val_truth,
                    seed=seed,
                    updates=args.updates,
                    device=str(device),
                    family=family,
                    phase_increment=cfg["phase_increment"],
                )
            torch.save(forecaster.model.state_dict(), path)
            save_json(record, history)
            members.append(forecaster)
            training[family].append(history)
            save_json(out / "training.json", training)
        models[family.upper() if family != "transformer" else "Transformer"] = members
    # All data-dependent selection is finished before any long-horizon scoring.
    contexts = data["test_contexts"]
    coherence_settings = validation_coherence_settings(
        data["validation"],
        cfg["physical_size"],
        cfg["dt"],
        training=train,
    )
    save_json(out / "coherence-settings.json", coherence_settings)
    metrics = {}
    for name, members in models.items():
        slug = name.lower().replace(" ", "-")
        print(
            f"{system}: long rollout {name} ({horizon} steps, {len(contexts)} origins)",
            flush=True,
        )
        predictions = np.asarray([m.rollout_batch(contexts, horizon) for m in members])
        cutoffs = [v for v in [1.8, 9, 25, 50, 100, 200] if v <= cfg["long_units"]]
        metric, arrays = evaluate_long_run(
            data["truth"],
            predictions,
            scale,
            cfg["physical_size"],
            cfg["dt"],
            cutoffs,
            coherence_settings,
            contexts=contexts,
        )
        metrics[name] = metric
        np.savez_compressed(out / f"{slug}.npz", **arrays)
        save_json(out / "metrics.json", metrics)
        print(
            f"{system}: {name} "
            f"first x breach={metric['coherence_x_median_capped']:.2f}, "
            f"member-climate W1={metric['mean_normalized_member_wasserstein']}",
            flush=True,
        )
    plot_system(out)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument(
        "--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS)
    )
    parser.add_argument(
        "--previous", type=Path, default=ROOT / "results/data-scaling-2026-11-05"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/long-horizon-2026-11-05"
    )
    args = parser.parse_args()
    if args.updates < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Need positive updates and distinct model seeds")
    for system in args.systems:
        saved_config = args.output_dir / system / "config.json"
        if saved_config.exists():
            previous_config = json.loads(saved_config.read_text())
            if (
                previous_config["updates"] != args.updates
                or previous_config["model_seeds"] != args.seeds
            ):
                parser.error(
                    "Existing run uses different settings; choose a new directory"
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "source.zip"
    if source_path.exists():
        source_path = args.output_dir / "source-resume.zip"
    with zipfile.ZipFile(source_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.glob("*.py"):
            archive.write(path, path.name)
        for path in (ROOT.parent / "src").rglob("*.py"):
            archive.write(path, str(path.relative_to(ROOT.parent)))
    device = select_device(args.device)
    summary = {}
    for system in args.systems:
        if (args.output_dir / system / "metrics.json").exists():
            saved = json.loads((args.output_dir / system / "metrics.json").read_text())
            if len(saved) == 6:
                from plot_long_horizon import plot_system

                print(
                    f"{system}: completed results retained; refreshing plots.",
                    flush=True,
                )
                plot_system(args.output_dir / system)
                summary[system] = saved
                continue
        summary[system] = run_system(system, args, device)
    save_json(args.output_dir / "summary.json", summary)
    print("Long-horizon comparison complete.", flush=True)


if __name__ == "__main__":
    main()
