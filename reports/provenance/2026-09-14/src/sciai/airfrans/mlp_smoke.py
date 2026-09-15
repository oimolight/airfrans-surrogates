from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from torch import nn


EXPECTED_SELECTION_SHA256 = "8c532a15c2af6538177ae9885de8978553b162c4eb788d33a2998c9edc447014"
EXPECTED_CASE_ID = "airFoil2D_SST_31.468_13.713_3.339_3.51_6.993"
DATASET_ROOT = Path("data/raw/airfrans_hf_selected/data/Dataset")


@dataclass(frozen=True)
class SmokeConfig:
    seed: int = 20260914
    surface_samples: int = 512
    volume_samples: int = 1536
    hidden_width: int = 64
    hidden_layers: int = 2
    steps: int = 2000
    learning_rate: float = 1e-3


class PointwiseMLP(nn.Module):
    def __init__(self, input_features: int, hidden_width: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_features, hidden_width), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_width, hidden_width), nn.ReLU()))
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audited_training_case(selection_path: Path) -> str:
    actual_hash = file_sha256(selection_path)
    if actual_hash != EXPECTED_SELECTION_SHA256:
        raise RuntimeError(f"Frozen selection changed: {actual_hash}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    case_id = sorted(selection["splits"]["train"])[0]
    memberships = [name for name, case_ids in selection["splits"].items() if case_id in case_ids]
    if case_id != EXPECTED_CASE_ID or memberships != ["train"]:
        raise RuntimeError(f"Expected the one audited training case, got {case_id} in {memberships}")
    return case_id


def fixed_sample_indices(implicit_distance: np.ndarray, config: SmokeConfig) -> tuple[np.ndarray, int, int]:
    geometry_surface = np.asarray(implicit_distance) == 0
    surface_indices = np.flatnonzero(geometry_surface)
    volume_indices = np.flatnonzero(~geometry_surface)
    if len(surface_indices) < config.surface_samples or len(volume_indices) < config.volume_samples:
        raise RuntimeError("The audited case does not contain enough surface or volume points")
    rng = np.random.default_rng(config.seed)
    selected_surface = rng.choice(surface_indices, size=config.surface_samples, replace=False)
    selected_volume = rng.choice(volume_indices, size=config.volume_samples, replace=False)
    selected = np.concatenate((selected_surface, selected_volume))
    rng.shuffle(selected)
    return selected, len(selected_surface), len(selected_volume)


def input_features(
    position: np.ndarray,
    implicit_distance: np.ndarray,
    case_id: str,
) -> tuple[np.ndarray, list[str]]:
    tokens = case_id.split("_")
    inlet_speed = float(tokens[2])
    angle_radians = math.radians(float(tokens[3]))
    inlet_x = inlet_speed * math.cos(angle_radians)
    inlet_y = inlet_speed * math.sin(angle_radians)
    signed_distance = -np.asarray(implicit_distance)
    geometry_surface = np.asarray(implicit_distance) == 0
    features = np.column_stack(
        (
            np.asarray(position)[:, :2],
            np.full(len(position), inlet_x),
            np.full(len(position), inlet_y),
            signed_distance,
            geometry_surface.astype(np.float64),
        )
    ).astype(np.float32)
    names = ["x", "y", "inlet_velocity_x", "inlet_velocity_y", "signed_distance", "geometry_surface"]
    return features, names


def normalize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    statistics = values.to(torch.float64)
    mean = statistics.mean(dim=0, keepdim=True)
    standard_deviation = statistics.std(dim=0, unbiased=False, keepdim=True)
    constant = statistics.amax(dim=0, keepdim=True) == statistics.amin(dim=0, keepdim=True)
    scale = torch.where(constant, torch.ones_like(standard_deviation), standard_deviation.clamp_min(1e-12))
    normalized = ((statistics - mean) / scale).to(values.dtype)
    return normalized, mean, scale


def nested_mismatches(before: Any, after: Any, path: str = "checkpoint") -> list[str]:
    if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
        return [] if torch.equal(before, after) else [path]
    if isinstance(before, dict) and isinstance(after, dict):
        if before.keys() != after.keys():
            return [f"{path}.keys"]
        return [
            mismatch
            for key in before
            for mismatch in nested_mismatches(before[key], after[key], f"{path}.{key}")
        ]
    if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        if len(before) != len(after):
            return [f"{path}.length"]
        return [
            mismatch
            for index, (before_item, after_item) in enumerate(zip(before, after))
            for mismatch in nested_mismatches(before_item, after_item, f"{path}[{index}]")
        ]
    return [] if before == after else [path]


def load_smoke_checkpoint(
    path: Path,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, Any], PointwiseMLP]:
    model_device = torch.device(device)
    checkpoint = torch.load(path, map_location=model_device, weights_only=True)
    config = checkpoint["config"]
    model = PointwiseMLP(
        len(checkpoint["input_names"]),
        config["hidden_width"],
        config["hidden_layers"],
    ).to(model_device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return checkpoint, model


def markdown_report(result: dict[str, Any]) -> str:
    mismatch_text = "none" if not result["mismatches"] else ", ".join(result["mismatches"])
    return f"""# AirfRANS one-case pointwise MLP smoke test

Status: COMPLETED; NOT A BENCHMARK

## Purpose

Verify the real preprocessing -> pointwise MLP -> loss -> backward -> Adam optimizer -> checkpoint -> reload path on fixed points from exactly one audited training case. No validation/test case, GNN, FNO, or multi-case training was used.

## Scope and data

- Case ID: `{result["case_id"]}`
- Fixed sampled points: {result["sample_count"]} ({result["surface_samples"]} geometry-surface + {result["volume_samples"]} nonsurface)
- Sampling seed: {result["config"]["seed"]}
- Model inputs: {", ".join(f'`{name}`' for name in result["input_names"])}
- Target: pressure `p/rho` only
- Input shape: `{tuple(result["input_shape"])}`
- Target shape: `{tuple(result["target_shape"])}`
- Target-derived model inputs: none

The fixed sample is selected using only point indices and `implicit_distance == 0`. Raw velocity `U`, pressure `p`, turbulent viscosity `nut`, `Simulation.surface`, and `Simulation.normals` are not model inputs. Pressure is read only after indices are frozen and is used only as the target.

## Model and optimization

- Model: pointwise ReLU MLP, {result["config"]["hidden_layers"]} hidden layers x {result["config"]["hidden_width"]} units, scalar output
- Trainable parameters: {result["parameter_count"]}
- Device: `{result["device"]}`
- PyTorch: `{result["torch_version"]}`
- Loss: mean squared error on pressure normalized from these fixed training targets
- Optimizer: Adam, learning rate {result["config"]["learning_rate"]}
- Steps: {result["steps"]}
- All gradients finite: {result["all_gradients_finite"]}

## Result

- Starting loss: {result["starting_loss"]:.12g}
- Final loss: {result["final_loss"]:.12g}
- Final / starting loss: {result["loss_ratio"]:.12g}
- Starting physical RMSE: {result["starting_rmse_physical"]:.9g} m^2/s^2
- Final physical RMSE: {result["final_rmse_physical"]:.9g} m^2/s^2
- Elapsed optimization time: {result["elapsed_seconds"]:.6f} s
- Checkpoint: `{result["checkpoint_path"]}`
- Checkpoint SHA-256: `{result["checkpoint_sha256"]}`
- Reload maximum absolute prediction difference: {result["reload_max_abs_prediction_difference"]:.12g}
- Reload exact prediction parity: {result["reload_exact_prediction_parity"]}
- Mismatches: {mismatch_text}

## Interpretation and limit

The loss change tests whether this small network can fit the fixed sampled labels and whether every pipeline stage executes. It is not evidence of performance on unseen points, geometries, flow conditions, validation cases, or test cases. The normalizers are fitted only on this one case's fixed training labels and must not be reused as evidence for the planned multi-case experiment.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("data/manifests/airfrans_selected_cases.json"))
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/airfrans_one_case_mlp_smoke.pt"))
    parser.add_argument("--result", type=Path, default=Path("reports/airfrans_one_case_mlp_smoke_test.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/airfrans_one_case_mlp_smoke_test.md"))
    parser.add_argument("--steps", type=int, default=SmokeConfig.steps)
    args = parser.parse_args()

    config = SmokeConfig(steps=args.steps)
    if config.steps < 1:
        raise RuntimeError("--steps must be at least 1")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")

    case_id = audited_training_case(args.selection)
    internal_path = args.dataset_root / case_id / f"{case_id}_internal.vtu"
    if not internal_path.is_file():
        raise RuntimeError(f"Audited internal mesh is missing: {internal_path}")
    internal = pv.read(internal_path)
    implicit_distance = np.asarray(internal.point_data["implicit_distance"])
    sample_indices, surface_samples, volume_samples = fixed_sample_indices(implicit_distance, config)
    all_features, input_names = input_features(internal.points, implicit_distance, case_id)
    sampled_features = torch.from_numpy(all_features[sample_indices]).to(device)
    sampled_target = torch.from_numpy(np.asarray(internal.point_data["p"])[sample_indices, None]).float().to(device)
    normalized_features, feature_mean, feature_scale = normalize(sampled_features)
    normalized_target, target_mean, target_scale = normalize(sampled_target)

    model = PointwiseMLP(normalized_features.shape[1], config.hidden_width, config.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    with torch.no_grad():
        starting_prediction = model(normalized_features)
        starting_loss = float(loss_function(starting_prediction, normalized_target))
        starting_rmse_physical = float(torch.sqrt(torch.mean(((starting_prediction - normalized_target) * target_scale) ** 2)))

    all_gradients_finite = True
    started = time.perf_counter()
    model.train()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(normalized_features)
        loss = loss_function(prediction, normalized_target)
        loss.backward()
        all_gradients_finite = all_gradients_finite and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        optimizer.step()
    elapsed_seconds = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        final_prediction = model(normalized_features)
        final_loss = float(loss_function(final_prediction, normalized_target))
        final_rmse_physical = float(torch.sqrt(torch.mean(((final_prediction - normalized_target) * target_scale) ** 2)))

    checkpoint = {
        "case_id": case_id,
        "config": asdict(config),
        "input_names": input_names,
        "sample_indices": torch.from_numpy(sample_indices),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "steps_completed": config.steps,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    reloaded, reloaded_model = load_smoke_checkpoint(args.checkpoint, device)
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=config.learning_rate)
    reloaded_optimizer.load_state_dict(reloaded["optimizer_state"])
    with torch.no_grad():
        reloaded_prediction = reloaded_model(normalized_features)

    reload_max_difference = float(torch.max(torch.abs(final_prediction - reloaded_prediction)))
    mismatches = nested_mismatches(checkpoint, reloaded)
    if not all_gradients_finite:
        mismatches.append("nonfinite_gradient")
    if not math.isfinite(final_loss):
        mismatches.append("nonfinite_final_loss")
    result = {
        "status": "COMPLETED_NOT_A_BENCHMARK",
        "case_id": case_id,
        "selection_sha256": file_sha256(args.selection),
        "internal_file": str(internal_path),
        "input_names": input_names,
        "input_shape": list(normalized_features.shape),
        "target_shape": list(normalized_target.shape),
        "sample_count": len(sample_indices),
        "surface_samples": surface_samples,
        "volume_samples": volume_samples,
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "steps": config.steps,
        "starting_loss": starting_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / starting_loss,
        "starting_rmse_physical": starting_rmse_physical,
        "final_rmse_physical": final_rmse_physical,
        "elapsed_seconds": elapsed_seconds,
        "all_gradients_finite": all_gradients_finite,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "reload_max_abs_prediction_difference": reload_max_difference,
        "reload_exact_prediction_parity": bool(torch.equal(final_prediction, reloaded_prediction)),
        "mismatches": mismatches,
        "prohibited_model_inputs": ["U", "p", "nut", "Simulation.surface", "Simulation.normals"],
        "benchmark_claim": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
