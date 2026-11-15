"""Chapter 09: real single-thread CPU MLP training, optimizer comparisons and exact resume."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from torch import nn


def learning_rate(config, step):
    if config["schedule"] == "constant":
        return config["lr"]
    warmup = 20
    if step < warmup:
        return config["lr"] * (step + 1) / warmup
    progress = (step - warmup) / (config["steps"] - warmup - 1)
    return config["lr"] * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def make_model():
    torch.manual_seed(20261115)
    return nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1)).double()


def make_optimizer(model, config):
    parameters = model.parameters()
    common = {"lr": config["lr"], "weight_decay": config["weight_decay"], "foreach": False}
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(parameters, momentum=config.get("momentum", 0.0), **common)
    if config["optimizer"] == "adam":
        return torch.optim.Adam(parameters, betas=(0.9, 0.999), eps=1e-8,
                                decoupled_weight_decay=False, **common)
    return torch.optim.AdamW(parameters, betas=(0.9, 0.999), eps=1e-8, **common)


def data():
    x = torch.linspace(-2.0, 2.0, 64, dtype=torch.float64).reshape(-1, 1)
    xv = ((x[:-1] + x[1:]) / 2).clone()
    return x, torch.sin(2 * x) + 0.1 * x, xv, torch.sin(2 * xv) + 0.1 * xv


def run_training(config, stop=None, checkpoint=None, save_checkpoint=None):
    x, y, xv, yv = data()
    model = make_model()
    optimizer = make_optimizer(model, config)
    generator = torch.Generator(device="cpu").manual_seed(17)
    order = torch.empty(0, dtype=torch.int64)
    cursor, start = 64, 0
    history = []
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError("Resume configuration differs from checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        order, cursor, start = saved["order"], saved["cursor"], saved["next_step"]
        history = saved["history"]
    else:
        with torch.no_grad():
            history.append({"step": 0, "passes": 0.0, "train_mse": float(nn.functional.mse_loss(model(x), y)),
                            "validation_mse": float(nn.functional.mse_loss(model(xv), yv)), "lr": 0.0})
    final_step = config["steps"] if stop is None else stop
    for step in range(start, final_step):
        if cursor >= len(x):
            order = torch.randperm(len(x), generator=generator)
            cursor = 0
        indices = order[cursor:cursor + config["batch_size"]]
        cursor += len(indices)
        eta = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = eta
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[indices])
        loss = nn.functional.mse_loss(prediction, y[indices])
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss in {config["name"]}')
        loss.backward()
        before_clip = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            history.append({"step": step + 1, "passes": (step + 1) * config["batch_size"] / len(x),
                            "train_mse": float(nn.functional.mse_loss(model(x), y)),
                            "validation_mse": float(nn.functional.mse_loss(model(xv), yv)),
                            "lr": eta, "gradient_norm_before_clip": float(before_clip)})
    if save_checkpoint is not None:
        torch.save({"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "sampler_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                    "order": order, "cursor": cursor, "next_step": final_step, "history": history}, save_checkpoint)
    return {"model": model, "optimizer": optimizer, "history": history, "sampler_rng": generator.get_state(),
            "order": order, "cursor": cursor, "torch_rng": torch.get_rng_state()}


def quadratic_paths():
    output = {}
    specifications = [("sgd", 0.15, 0.0), ("momentum", 0.08, 0.9), ("adam", 0.15, 0.0)]
    for name, rate, momentum in specifications:
        parameter = nn.Parameter(torch.tensor([4.0, 2.0], dtype=torch.float64))
        if name == "adam":
            optimizer = torch.optim.Adam([parameter], lr=rate, foreach=False)
        else:
            optimizer = torch.optim.SGD([parameter], lr=rate, momentum=momentum, foreach=False)
        points = [parameter.detach().tolist()]
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            objective = 0.5 * (parameter[0]**2 + 10 * parameter[1]**2)
            objective.backward(); optimizer.step(); points.append(parameter.detach().tolist())
        output[name] = {"lr": rate, "momentum": momentum, "points": points,
                        "final_objective": float((0.5 * (parameter[0]**2 + 10 * parameter[1]**2)).detach())}
    unstable = torch.tensor([4.0, 2.0], dtype=torch.float64)
    losses = []
    for _ in range(12):
        losses.append(float(0.5 * (unstable[0]**2 + 10 * unstable[1]**2)))
        unstable -= 0.3 * torch.tensor([unstable[0], 10 * unstable[1]])
    output["unstable_sgd"] = {"lr": 0.3, "objectives": losses}
    return output


def scalar_demonstrations():
    result = {}
    for name in ["sgd", "momentum", "adam", "adamw"]:
        parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        if name == "sgd": optimizer = torch.optim.SGD([parameter], lr=0.1)
        elif name == "momentum": optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
        elif name == "adam": optimizer = torch.optim.Adam([parameter], lr=0.1, eps=1e-8, foreach=False)
        else: optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.1, eps=1e-8, foreach=False)
        points = [1.0]
        for _ in range(2):
            parameter.grad = torch.tensor(2.0, dtype=torch.float64)
            optimizer.step(); points.append(float(parameter.detach()))
        result[name] = points
    p = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    (0.5 * p**2).backward(); first = float(p.grad)
    (0.5 * p**2).backward(); accumulated = float(p.grad)
    p.grad = None
    (0.5 * p**2).backward(); after_clear = float(p.grad)
    result["gradient_accumulation"] = {"first": first, "without_clear_second": accumulated, "after_clear": after_clear}
    clip_parameter = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    clip_parameter.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    norm_before = nn.utils.clip_grad_norm_([clip_parameter], max_norm=2.0)
    result["clipping"] = {"before": [3.0, 4.0], "norm_before": float(norm_before),
                          "after": clip_parameter.grad.tolist(), "norm_after": float(clip_parameter.grad.norm())}
    scalar = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    targets = torch.arange(1.0, 6.0, dtype=torch.float64)
    gradients = []
    for subset in [targets, targets[:2], targets[2:]]:
        scalar.grad = None
        ((scalar - subset) ** 2).mean().backward()
        gradients.append(float(scalar.grad))
    result["unequal_batch_gradients"] = {"full_mean_gradient": gradients[0],
                                       "first_batch_mean_gradient": gradients[1], "second_batch_mean_gradient": gradients[2],
                                       "weighted_mean_gradient": 0.4 * gradients[1] + 0.6 * gradients[2],
                                       "naive_equal_mean_gradient": 0.5 * gradients[1] + 0.5 * gradients[2]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/experiments/09-training-and-optimizers"))
    parser.add_argument("--write-figures", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {"lr": 0.01, "batch_size": 16, "steps": 240, "weight_decay": 0.0, "schedule": "constant"}
    specifications = [
        {"name": "sgd", "optimizer": "sgd", "lr": 0.03},
        {"name": "momentum", "optimizer": "sgd", "lr": 0.03, "momentum": 0.9},
        {"name": "adam", "optimizer": "adam"},
        {"name": "adamw_no_decay", "optimizer": "adamw"},
        {"name": "adam_coupled_decay", "optimizer": "adam", "weight_decay": 0.01},
        {"name": "adamw_decay", "optimizer": "adamw", "weight_decay": 0.01},
        {"name": "adam_small_lr", "optimizer": "adam", "lr": 0.001},
        {"name": "adam_full_batch", "optimizer": "adam", "batch_size": 64, "steps": 60},
        {"name": "adamw_warmup_cosine", "optimizer": "adamw", "weight_decay": 0.01, "schedule": "warmup_cosine"},
    ]
    configs = [{**base, **item} for item in specifications]
    results = {}
    full_main = None
    for config in configs:
        running = run_training(config)
        results[config["name"]] = {"config": config, "history": running["history"],
                                   "initial_train_mse": running["history"][0]["train_mse"],
                                   "final_train_mse": running["history"][-1]["train_mse"],
                                   "final_validation_mse": running["history"][-1]["validation_mse"],
                                   "samples_processed": config["steps"] * config["batch_size"],
                                   "data_passes": config["steps"] * config["batch_size"] / 64}
        if config["name"] == "adamw_warmup_cosine": full_main = running
    checkpoint_path = arguments.output_dir / "step-93-checkpoint.pt"
    main_config = configs[-1]
    run_training(main_config, stop=93, save_checkpoint=checkpoint_path)
    resumed = run_training(main_config, checkpoint=checkpoint_path)
    resume_max_error = max(float((a - b).detach().abs().max()) for a, b in zip(full_main["model"].parameters(), resumed["model"].parameters()))
    adam_zero_difference = max(abs(a["train_mse"] - b["train_mse"]) for a, b in zip(results["adam"]["history"], results["adamw_no_decay"]["history"]))
    demonstrations = scalar_demonstrations()
    paths = quadratic_paths()
    checks = []
    def require(condition, name):
        if not bool(condition): raise RuntimeError(f"Validation failed: {name}")
        checks.append(name)
    require(sum(p.numel() for p in make_model().parameters()) == 49, "mlp_has_49_parameters")
    for name, result in results.items():
        require(all(math.isfinite(row["train_mse"]) and math.isfinite(row["validation_mse"]) for row in result["history"]), f"finite_losses_{name}")
        require(result["final_train_mse"] < result["initial_train_mse"], f"training_loss_decreases_{name}")
    require(resume_max_error == 0.0, "resumed_parameters_exactly_match_uninterrupted")
    require(resumed["history"] == full_main["history"], "resumed_entire_history_matches")
    require(torch.equal(resumed["sampler_rng"], full_main["sampler_rng"]) and resumed["cursor"] == full_main["cursor"], "resumed_sampler_state_matches")
    require(adam_zero_difference < 1e-14, "adam_and_adamw_match_without_decay")
    require(abs(demonstrations["sgd"][1] - 0.8) < 1e-14, "sgd_hand_first_update")
    require(abs(demonstrations["momentum"][2] - 0.42) < 1e-14, "momentum_hand_second_update")
    require(abs(demonstrations["adam"][1] - (1 - 0.1 * 2 / (2 + 1e-8))) < 1e-12, "adam_bias_corrected_first_update")
    require(abs(demonstrations["adamw"][1] - (0.99 - 0.1 * 2 / (2 + 1e-8))) < 1e-12, "adamw_decoupled_first_update")
    require(demonstrations["gradient_accumulation"] == {"first": 1.0, "without_clear_second": 2.0, "after_clear": 1.0}, "gradients_accumulate_until_cleared")
    require(abs(demonstrations["clipping"]["norm_before"] - 5.0) < 1e-14 and demonstrations["clipping"]["norm_after"] <= 2.0, "global_norm_is_clipped")
    require(paths["unstable_sgd"]["objectives"][-1] > paths["unstable_sgd"]["objectives"][0] * 1000, "too_large_quadratic_learning_rate_diverges")
    require(all(result["samples_processed"] == 3840 for result in results.values()), "comparisons_share_60_data_passes")
    batch_demo = demonstrations["unequal_batch_gradients"]
    require(abs(batch_demo["weighted_mean_gradient"] - batch_demo["full_mean_gradient"]) < 1e-14, "weighted_batch_gradients_match_full_gradient")
    require(abs(batch_demo["naive_equal_mean_gradient"] - batch_demo["full_mean_gradient"]) > 0.1, "equal_weighting_unequal_batches_is_wrong")
    report = {
        "schema_version": 1, "chapter": "09", "collected_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "python_executable": sys.executable, "platform": platform.platform(), "device": "CPU", "dtype": "float64", "threads": 1},
        "data": {"source": "synthetic y=sin(2x)+0.1x", "train_points": 64, "validation_midpoints": 63,
                 "split": "fixed disjoint interpolation grids; no test set", "initialization_seed": 20261115, "sampler_seed": 17},
        "model": {"architecture": "Linear(1,16) -> tanh -> Linear(16,1)", "parameter_count": 49,
                  "loss": "mean squared error over batch and its one output", "gradient_clip_max_norm": 1.0},
        "experiments": results, "scalar_demonstrations": demonstrations, "quadratic_paths": paths,
        "resume": {"checkpoint": str(checkpoint_path), "saved_after_step": 93, "total_steps": 240,
                   "maximum_parameter_absolute_error": resume_max_error, "entire_history_equal": resumed["history"] == full_main["history"],
                   "contents": ["model", "optimizer", "next_step", "schedule_configuration", "sampler_order", "sampler_cursor", "sampler_rng", "torch_rng", "history"]},
        "resources": {"elapsed_seconds_excluding_imports": time.perf_counter() - started, "parameter_payload_bytes": 49 * 8,
                      "memory_scope": "weights only; optimizer states and total process memory not included"},
        "checks": {"status": "passed", "count": len(checks), "names": checks},
    }
    result_path = arguments.output_dir / "results.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if arguments.write_figures:
        from figures import write_figures
        write_figures(report, Path(__file__).resolve().parents[2] / "docs/assets/09-training-and-optimizers")
    print(json.dumps({"collected_at": report["collected_at"], "environment": report["environment"], "experiments": {name: {key: result[key] for key in ["initial_train_mse", "final_train_mse", "final_validation_mse", "samples_processed", "data_passes"]} for name, result in results.items()}, "resume": report["resume"], "scalar_demonstrations": demonstrations, "resources": report["resources"], "checks": report["checks"], "output": str(result_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
