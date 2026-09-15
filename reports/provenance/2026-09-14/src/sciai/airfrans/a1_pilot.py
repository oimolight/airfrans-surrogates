from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import resource
import signal
import statistics
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from torch import nn

from sciai.common.artifacts import initialize_run, write_manifest

from .mlp_smoke import SmokeConfig, fixed_sample_indices, input_features, normalize
from .pilot_models import knn_edges, make_pilot_model, trainable_scalar_count


EXPECTED_SELECTION_SHA256 = "8c532a15c2af6538177ae9885de8978553b162c4eb788d33a2998c9edc447014"
EXPERIMENT_ID = "airfrans-a1-compute-pilot"
MODEL_NAMES = ("mlp", "gnn", "fno")


@dataclass
class PointCase:
    case_id: str
    split: str
    features: torch.Tensor
    target: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor | None = None


@dataclass
class GridCase:
    case_id: str
    split: str
    features: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor


class PilotTimeout(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_subset(
    selection: Mapping[str, Any],
    seed: int,
    train_count: int,
    validation_count: int,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for split, count in (("train", train_count), ("validation", validation_count)):
        case_ids = selection["splits"][split]
        ranked = sorted(
            case_ids,
            key=lambda case_id: hashlib.sha256(f"{seed}:{split}:{case_id}".encode()).hexdigest(),
        )
        if count < 1 or count > len(ranked):
            raise ValueError(f"Invalid {split} subset count: {count}")
        selected[split] = ranked[:count]
    if set(selected["train"]) & set(selected["validation"]):
        raise RuntimeError("Pilot train and validation subsets overlap")
    return selected


def case_seed(seed: int, split: str, case_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split}:{case_id}:points".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def process_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def fit_statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, mean, scale = normalize(values)
    return mean.to(values.dtype), scale.to(values.dtype)


def apply_statistics(values: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (values - mean) / scale


def load_point_case(
    dataset_root: Path,
    case_id: str,
    split: str,
    config: Mapping[str, Any],
) -> PointCase:
    path = dataset_root / case_id / f"{case_id}_internal.vtu"
    mesh = pv.read(path)
    distance = np.asarray(mesh.point_data["implicit_distance"])
    sample_config = SmokeConfig(
        seed=case_seed(config["seed"], split, case_id),
        surface_samples=config["sampling"]["surface_samples"],
        volume_samples=config["sampling"]["volume_samples"],
        steps=1,
    )
    indices, surface_count, volume_count = fixed_sample_indices(distance, sample_config)
    if surface_count + volume_count != config["sampling"]["points_per_case"]:
        raise RuntimeError("Sample count differs from the pilot config")
    features, _ = input_features(mesh.points, distance, case_id)
    sampled_features = torch.from_numpy(features[indices])
    sampled_target = torch.from_numpy(np.asarray(mesh.point_data["p"])[indices, None]).float()
    del mesh
    gc.collect()
    return PointCase(
        case_id=case_id,
        split=split,
        features=sampled_features,
        target=sampled_target,
        positions=sampled_features[:, :2].clone(),
    )


def load_grid_case(
    dataset_root: Path,
    case_id: str,
    split: str,
    config: Mapping[str, Any],
) -> GridCase:
    path = dataset_root / case_id / f"{case_id}_internal.vtu"
    mesh = pv.read(path)
    height = config["models"]["fno"]["grid_height"]
    width = config["models"]["fno"]["grid_width"]
    bounds = mesh.bounds
    spacing_x = (bounds.x_max - bounds.x_min) / (width - 1)
    spacing_y = (bounds.y_max - bounds.y_min) / (height - 1)
    grid = pv.ImageData(
        dimensions=(width, height, 1),
        spacing=(spacing_x, spacing_y, 1.0),
        origin=(bounds.x_min, bounds.y_min, bounds.z_min),
    )
    sampled = grid.sample(mesh)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"]).astype(bool)
    distance = np.asarray(sampled.point_data["implicit_distance"])
    features, _ = input_features(sampled.points, distance, case_id)
    signed_distance = features[:, 4]
    features[:, 5] = signed_distance <= 0.5 * math.hypot(spacing_x, spacing_y)
    pressure = np.asarray(sampled.point_data["p"], dtype=np.float32)

    def grid_array(values: np.ndarray) -> np.ndarray:
        return values.reshape((width, height), order="F").T

    feature_grid = np.stack([grid_array(features[:, index]) for index in range(features.shape[1])])
    target_grid = grid_array(pressure)[None, ...]
    valid_grid = grid_array(valid)[None, ...]
    del mesh, sampled, grid
    gc.collect()
    return GridCase(
        case_id=case_id,
        split=split,
        features=torch.from_numpy(feature_grid.copy()).float(),
        target=torch.from_numpy(target_grid.copy()).float(),
        valid=torch.from_numpy(valid_grid.copy()),
    )


def normalize_point_cases(
    train: list[PointCase],
    validation: list[PointCase],
    neighbors: int | None,
) -> dict[str, Any]:
    feature_mean, feature_scale = fit_statistics(torch.cat([case.features for case in train]))
    target_mean, target_scale = fit_statistics(torch.cat([case.target for case in train]))
    for case in [*train, *validation]:
        case.features = apply_statistics(case.features, feature_mean, feature_scale)
        case.target = apply_statistics(case.target, target_mean, target_scale)
        case.positions = case.features[:, :2]
        if neighbors is not None:
            case.edge_index = knn_edges(case.positions, neighbors)
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


def normalize_grid_cases(train: list[GridCase], validation: list[GridCase]) -> dict[str, Any]:
    train_features = torch.cat(
        [case.features.permute(1, 2, 0)[case.valid.squeeze(0)] for case in train]
    )
    train_targets = torch.cat(
        [case.target.permute(1, 2, 0)[case.valid.squeeze(0)] for case in train]
    )
    feature_mean, feature_scale = fit_statistics(train_features)
    target_mean, target_scale = fit_statistics(train_targets)
    feature_mean_grid = feature_mean.reshape(-1, 1, 1)
    feature_scale_grid = feature_scale.reshape(-1, 1, 1)
    target_mean_grid = target_mean.reshape(-1, 1, 1)
    target_scale_grid = target_scale.reshape(-1, 1, 1)
    for case in [*train, *validation]:
        case.features = apply_statistics(case.features, feature_mean_grid, feature_scale_grid)
        case.target = apply_statistics(case.target, target_mean_grid, target_scale_grid)
        case.features[:, ~case.valid.squeeze(0)] = 0
        case.target[:, ~case.valid.squeeze(0)] = 0
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


def masked_grid_loss(prediction: torch.Tensor, case: GridCase) -> torch.Tensor:
    mask = case.valid[None, ...].to(prediction.dtype)
    return ((prediction - case.target[None, ...]).square() * mask).sum() / mask.sum()


def case_loss(model_name: str, model: nn.Module, case: PointCase | GridCase) -> torch.Tensor:
    if model_name == "mlp" and isinstance(case, PointCase):
        return nn.functional.mse_loss(model(case.features), case.target)
    if model_name == "gnn" and isinstance(case, PointCase):
        if case.edge_index is None:
            raise RuntimeError("GNN case has no edge index")
        return nn.functional.mse_loss(model(case.features, case.positions, case.edge_index), case.target)
    if model_name == "fno" and isinstance(case, GridCase):
        return masked_grid_loss(model(case.features[None, ...]), case)
    raise TypeError(f"Unexpected case type for {model_name}")


def loss_decreases_normally(losses: list[float]) -> tuple[bool, dict[str, object]]:
    window = min(5, len(losses) // 2)
    if window < 1:
        return False, {"reason": "INSUFFICIENT_STEPS"}
    first_median = statistics.median(losses[:window])
    last_median = statistics.median(losses[-window:])
    normal = all(math.isfinite(loss) for loss in losses) and last_median < first_median
    return normal, {
        "criterion": "all losses finite and median(last 5) < median(first 5)",
        "first_window_median": first_median,
        "last_window_median": last_median,
    }


def estimate_full_run(
    seconds_per_step: float,
    validation_seconds: float,
    preprocessing_seconds: float,
    validation_case_count: int,
    config: Mapping[str, Any],
) -> dict[str, object]:
    assumptions = config["estimate_assumptions"]
    epochs = assumptions["epochs"]
    validation_passes = math.ceil(epochs / assumptions["validation_interval_epochs"])
    preprocessing_per_case = preprocessing_seconds / (
        config["subset"]["train_cases"] + config["subset"]["validation_cases"]
    )
    validation_per_case = validation_seconds / validation_case_count
    preprocessing_estimate = preprocessing_per_case * (
        assumptions["full_train_cases"] + assumptions["full_validation_cases"]
    )
    training_estimate = seconds_per_step * assumptions["full_train_cases"] * epochs
    validation_estimate = (
        validation_per_case
        * assumptions["full_validation_cases"]
        * validation_passes
    )
    total = preprocessing_estimate + training_estimate + validation_estimate
    return {
        "estimated_full_run_seconds": total,
        "estimated_full_run_hours": total / 3600,
        "preprocessing_seconds": preprocessing_estimate,
        "training_seconds": training_estimate,
        "validation_seconds": validation_estimate,
        "assumptions": dict(assumptions),
        "formula": "preprocess_per_case*200 + seconds_per_step*160*epochs + validation_per_case*40*validation_passes",
        "omissions": ["checkpoint I/O", "summary generation", "model-specific final evaluation"],
        "estimate_not_measurement": True,
    }


def timeout_handler(signum: int, frame: object) -> None:
    raise PilotTimeout("Pilot exceeded the 600-second per-model wall-time limit")


def resolved_config(base_config: Mapping[str, Any], model_name: str, subset: Mapping[str, list[str]]) -> dict[str, Any]:
    return {
        "schema_version": base_config["schema_version"],
        "purpose": base_config["purpose"],
        "seed": base_config["seed"],
        "device": base_config["device"],
        "subset": {
            **base_config["subset"],
            "train_case_ids": subset["train"],
            "validation_case_ids": subset["validation"],
            "test_case_count": 0,
        },
        "sampling": dict(base_config["sampling"]),
        "training": dict(base_config["training"]),
        "model_name": model_name,
        "model": dict(base_config["models"][model_name]),
        "estimate_assumptions": dict(base_config["estimate_assumptions"]),
        "limitations": {
            "systems_pilot_not_benchmark": True,
            "test_set_evaluated": False,
            "gnn_wing_crossing_edge_filter": "NOT_IMPLEMENTED_IN_PILOT",
            "fno_common_point_evaluation": "NOT_IMPLEMENTED_IN_PILOT",
        },
    }


def run_model(
    model_name: str,
    config_path: Path,
    selection_path: Path,
    source_manifest_path: Path,
    dataset_root: Path,
    runs_root: Path,
    repository_root: Path,
) -> Path:
    overall_started = time.perf_counter()
    baseline_rss = process_peak_rss_bytes()
    base_config = read_json(config_path)
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    if base_config["seed"] != 17 or base_config["device"] != "cpu":
        raise RuntimeError("Pilot must use seed 17 on CPU")
    max_wall_seconds = base_config["training"]["max_wall_seconds_per_model"]
    if max_wall_seconds != 600:
        raise RuntimeError("Pilot wall-time limit must be exactly 600 seconds per model")
    if file_sha256(selection_path) != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("Frozen selection changed")

    selection = read_json(selection_path)
    subset = deterministic_subset(
        selection,
        seed=base_config["seed"],
        train_count=base_config["subset"]["train_cases"],
        validation_count=base_config["subset"]["validation_cases"],
    )
    config = resolved_config(base_config, model_name, subset)
    run_id, run_directory = initialize_run(
        runs_root=runs_root,
        experiment_id=f"{EXPERIMENT_ID}-{model_name}",
        model=model_name,
        seed=17,
        device="cpu",
        source_airfrans_manifest=source_manifest_path,
        selected_case_manifest=selection_path,
        resolved_config=config,
        actual_case_counts={"train": len(subset["train"]), "validation": len(subset["validation"]), "test": 0},
        repository_root=repository_root,
    )
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_wall_seconds)
    try:
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(base_config["training"]["torch_threads"])
        torch.set_num_interop_threads(1)
        preprocessing_started = time.perf_counter()
        if model_name == "fno":
            train_cases: list[PointCase | GridCase] = [
                load_grid_case(dataset_root, case_id, "train", base_config)
                for case_id in subset["train"]
            ]
            validation_cases: list[PointCase | GridCase] = [
                load_grid_case(dataset_root, case_id, "validation", base_config)
                for case_id in subset["validation"]
            ]
            normalizers = normalize_grid_cases(train_cases, validation_cases)
        else:
            train_cases = [
                load_point_case(dataset_root, case_id, "train", base_config)
                for case_id in subset["train"]
            ]
            validation_cases = [
                load_point_case(dataset_root, case_id, "validation", base_config)
                for case_id in subset["validation"]
            ]
            neighbors = base_config["models"]["gnn"]["neighbors"] if model_name == "gnn" else None
            normalizers = normalize_point_cases(train_cases, validation_cases, neighbors)
        preprocessing_seconds = time.perf_counter() - preprocessing_started

        model = make_pilot_model(model_name)
        parameter_count = trainable_scalar_count(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=base_config["training"]["learning_rate"])
        history = []
        gradients_finite = True
        model.train()
        training_started = time.perf_counter()
        for step in range(base_config["training"]["steps"]):
            if time.perf_counter() - overall_started >= max_wall_seconds:
                raise PilotTimeout("Pilot reached the per-model wall-time limit")
            case = train_cases[step % len(train_cases)]
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = case_loss(model_name, model, case)
            loss.backward()
            step_gradients_finite = all(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            gradients_finite = gradients_finite and step_gradients_finite
            optimizer.step()
            history.append(
                {
                    "step": step + 1,
                    "case_id": case.case_id,
                    "normalized_mse": float(loss.detach()),
                    "seconds": time.perf_counter() - step_started,
                    "gradients_finite": step_gradients_finite,
                }
            )
        training_seconds = time.perf_counter() - training_started

        validation_started = time.perf_counter()
        model.eval()
        validation_records = []
        with torch.no_grad():
            for case in validation_cases:
                validation_records.append(
                    {
                        "case_id": case.case_id,
                        "split": "validation",
                        "normalized_mse": float(case_loss(model_name, model, case)),
                    }
                )
        validation_seconds = time.perf_counter() - validation_started
        timed_steps = history[base_config["training"]["timing_warmup_steps"] :]
        seconds = [record["seconds"] for record in timed_steps]
        seconds_per_step = statistics.mean(seconds)
        decreases, decrease_details = loss_decreases_normally(
            [record["normalized_mse"] for record in history]
        )
        estimate = estimate_full_run(
            seconds_per_step,
            validation_seconds,
            preprocessing_seconds,
            len(validation_cases),
            base_config,
        )

        checkpoint_path = run_directory / "checkpoints/model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": model_name,
                "config": config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "normalizers": normalizers,
                "steps_completed": len(history),
            },
            checkpoint_path,
        )
        peak_rss = process_peak_rss_bytes()
        total_seconds = time.perf_counter() - overall_started
        metrics = {
            "purpose": base_config["purpose"],
            "parameter_count": parameter_count,
            "parameter_count_definition": "trainable real scalar values; each complex parameter counts as two",
            "peak_rss_bytes": peak_rss,
            "baseline_peak_rss_bytes": baseline_rss,
            "incremental_peak_rss_bytes": max(0, peak_rss - baseline_rss),
            "memory_measurement": "process peak resident set size via resource.getrusage",
            "loss_decreases_normally": decreases,
            "loss_decrease_details": decrease_details,
            "all_gradients_finite": gradients_finite,
            "validation_normalized_mse": statistics.mean(
                record["normalized_mse"] for record in validation_records
            ),
            "test_metrics": None,
            "test_set_evaluated": False,
            "benchmark_claim": False,
        }
        timing = {
            "preprocessing_seconds": preprocessing_seconds,
            "training_seconds": training_seconds,
            "seconds_per_training_step": seconds_per_step,
            "median_seconds_per_training_step": statistics.median(seconds),
            "timing_warmup_steps_excluded": base_config["training"]["timing_warmup_steps"],
            "validation_seconds": validation_seconds,
            "validation_case_count": len(validation_cases),
            "total_pilot_seconds": total_seconds,
            "maximum_wall_seconds": max_wall_seconds,
            "estimated_full_run": estimate,
        }
        per_case_metrics = [
            {
                "case_id": case.case_id,
                "split": "train",
                "last_observed_normalized_mse": next(
                    record["normalized_mse"]
                    for record in reversed(history)
                    if record["case_id"] == case.case_id
                ),
            }
            for case in train_cases
        ] + validation_records
        write_json(run_directory / "training_history.json", history)
        write_json(run_directory / "metrics.json", metrics)
        write_json(run_directory / "per_case_metrics.json", per_case_metrics)
        write_json(run_directory / "timing.json", timing)
        manifest = read_json(run_directory / "manifest.json")
        manifest.update(
            {
                "status": "COMPLETED",
                "failure_reason": None,
                "checkpoint_path": checkpoint_path.resolve().relative_to(repository_root.resolve()).as_posix(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "pilot_scope": {
                    "purpose": base_config["purpose"],
                    "test_case_ids_read": False,
                    "test_case_files_opened": 0,
                    "test_cases_evaluated": 0,
                    "quality_comparison_claim": False,
                },
                "measurements": {
                    "parameter_count": parameter_count,
                    "peak_rss_bytes": peak_rss,
                    "seconds_per_training_step": seconds_per_step,
                    "validation_seconds": validation_seconds,
                    "estimated_full_run_seconds": estimate["estimated_full_run_seconds"],
                    "loss_decreases_normally": decreases,
                },
            }
        )
        write_manifest(run_directory, manifest)
        print(json.dumps({"run_id": run_id, "run_directory": str(run_directory), **manifest["measurements"]}, indent=2))
        return run_directory
    except Exception as error:
        manifest = read_json(run_directory / "manifest.json")
        manifest.update(
            {
                "status": "FAILED",
                "failure_reason": f"{type(error).__name__}: {error}",
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        write_manifest(run_directory, manifest)
        raise
    finally:
        signal.alarm(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/airfrans/a1_compute_pilot.json"))
    parser.add_argument("--selection", type=Path, default=Path("data/manifests/airfrans_selected_cases.json"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/airfrans_hf/data/Dataset/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/airfrans_hf_selected/data/Dataset"))
    parser.add_argument("--runs-root", type=Path, default=Path("reports/runs"))
    args = parser.parse_args()
    run_model(
        model_name=args.model,
        config_path=args.config,
        selection_path=args.selection,
        source_manifest_path=args.source_manifest,
        dataset_root=args.dataset_root,
        runs_root=args.runs_root,
        repository_root=Path.cwd(),
    )


if __name__ == "__main__":
    main()
