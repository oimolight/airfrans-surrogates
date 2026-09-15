from __future__ import annotations

import gc
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from torch import nn

from .a1_contract import bilinear_grid_to_points, case_pressure_metrics
from .a1_pilot import apply_statistics, fit_statistics, masked_grid_loss
from .mlp_smoke import SmokeConfig, fixed_sample_indices, input_features
from .pilot_models import wing_filtered_knn_edges


@dataclass
class EvaluationPoints:
    positions: torch.Tensor
    target: torch.Tensor
    surface_mask: torch.Tensor
    inlet_velocity: tuple[float, float]


@dataclass
class FinalPointCase:
    case_id: str
    split: str
    features: torch.Tensor
    target: torch.Tensor
    evaluation: EvaluationPoints
    raw_positions: torch.Tensor
    wing_polygon: torch.Tensor | None
    edge_index: torch.Tensor | None = None


@dataclass
class FinalGridCase:
    case_id: str
    split: str
    features: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    bounds: tuple[float, float, float, float]
    evaluation: EvaluationPoints | None


def deterministic_point_seed(namespace: str, split: str, case_id: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{split}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def ordered_boundary_polygon(surface: pv.PolyData) -> torch.Tensor:
    cells: list[tuple[int, int]] = []
    offset = 0
    lines = np.asarray(surface.lines)
    while offset < len(lines):
        point_count = int(lines[offset])
        if point_count != 2:
            raise ValueError("Aerofoil boundary must contain only two-point line cells")
        cells.append((int(lines[offset + 1]), int(lines[offset + 2])))
        offset += point_count + 1
    adjacency: dict[int, list[int]] = {index: [] for index in range(surface.n_points)}
    for first, second in cells:
        adjacency[first].append(second)
        adjacency[second].append(first)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("Aerofoil boundary must be one closed degree-two cycle")
    start = min(adjacency)
    order = [start]
    previous = -1
    current = start
    while True:
        following = next(node for node in adjacency[current] if node != previous)
        if following == start:
            break
        if following in order:
            raise ValueError("Aerofoil boundary closes before visiting every point")
        order.append(following)
        previous, current = current, following
    if len(order) != surface.n_points:
        raise ValueError("Aerofoil boundary contains more than one component")
    return torch.from_numpy(np.asarray(surface.points)[order, :2].copy()).float()


def _sample_indices(
    distance: np.ndarray,
    case_id: str,
    split: str,
    config: Mapping[str, Any],
    training: bool,
) -> np.ndarray:
    namespace = (
        config["sampling"]["training_point_seed_namespace"]
        if training
        else config["evaluation"]["point_seed_namespace"]
    )
    if training:
        namespace = f"{namespace}:seed-{config['seed']}"
    sample_config = SmokeConfig(
        seed=deterministic_point_seed(namespace, split, case_id),
        surface_samples=config["sampling"]["surface_samples"],
        volume_samples=config["sampling"]["volume_samples"],
        steps=1,
    )
    indices, surface_count, volume_count = fixed_sample_indices(distance, sample_config)
    if surface_count + volume_count != config["sampling"]["points_per_case"]:
        raise RuntimeError("Sample count differs from the final A1 protocol")
    return indices


def _evaluation_points(
    mesh: pv.DataSet,
    distance: np.ndarray,
    case_id: str,
    split: str,
    config: Mapping[str, Any],
    training: bool,
) -> tuple[np.ndarray, np.ndarray, EvaluationPoints]:
    indices = _sample_indices(distance, case_id, split, config, training)
    features, _ = input_features(mesh.points, distance, case_id)
    sampled_features = features[indices]
    pressure = np.asarray(mesh.point_data["p"], dtype=np.float32)[indices, None]
    evaluation = EvaluationPoints(
        positions=torch.from_numpy(sampled_features[:, :2].copy()).float(),
        target=torch.from_numpy(pressure.copy()).float(),
        surface_mask=torch.from_numpy((distance[indices] == 0).copy()),
        inlet_velocity=(float(sampled_features[0, 2]), float(sampled_features[0, 3])),
    )
    return indices, sampled_features, evaluation


def load_point_case(
    dataset_root: Path,
    case_id: str,
    split: str,
    config: Mapping[str, Any],
    model_name: str,
) -> FinalPointCase:
    case_root = dataset_root / case_id
    mesh = pv.read(case_root / f"{case_id}_internal.vtu")
    distance = np.asarray(mesh.point_data["implicit_distance"])
    indices, sampled_features, evaluation = _evaluation_points(
        mesh,
        distance,
        case_id,
        split,
        config,
        training=split == "train",
    )
    wing_polygon = None
    if model_name == "gnn":
        surface = pv.read(case_root / f"{case_id}_aerofoil.vtp")
        wing_polygon = ordered_boundary_polygon(surface)
        del surface
    case = FinalPointCase(
        case_id=case_id,
        split=split,
        features=torch.from_numpy(sampled_features.copy()).float(),
        target=evaluation.target.clone(),
        evaluation=evaluation,
        raw_positions=torch.from_numpy(np.asarray(mesh.points)[indices, :2].copy()).float(),
        wing_polygon=wing_polygon,
    )
    del mesh
    gc.collect()
    return case


def load_grid_case(
    dataset_root: Path,
    case_id: str,
    split: str,
    config: Mapping[str, Any],
    include_evaluation: bool,
) -> FinalGridCase:
    path = dataset_root / case_id / f"{case_id}_internal.vtu"
    mesh = pv.read(path)
    height = config["model"]["grid_height"]
    width = config["model"]["grid_width"]
    mesh_bounds = mesh.bounds
    bounds = (
        float(mesh_bounds.x_min),
        float(mesh_bounds.x_max),
        float(mesh_bounds.y_min),
        float(mesh_bounds.y_max),
    )
    spacing_x = (bounds[1] - bounds[0]) / (width - 1)
    spacing_y = (bounds[3] - bounds[2]) / (height - 1)
    grid = pv.ImageData(
        dimensions=(width, height, 1),
        spacing=(spacing_x, spacing_y, 1.0),
        origin=(bounds[0], bounds[2], float(mesh_bounds.z_min)),
    )
    sampled = grid.sample(mesh)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"]).astype(bool)
    distance = np.asarray(sampled.point_data["implicit_distance"])
    features, _ = input_features(sampled.points, distance, case_id)
    features[:, 5] = features[:, 4] <= 0.5 * math.hypot(spacing_x, spacing_y)
    pressure = np.asarray(sampled.point_data["p"], dtype=np.float32)

    def grid_array(values: np.ndarray) -> np.ndarray:
        return values.reshape((width, height), order="F").T

    feature_grid = np.stack([grid_array(features[:, index]) for index in range(features.shape[1])])
    target_grid = grid_array(pressure)[None, ...]
    valid_grid = grid_array(valid)[None, ...]
    evaluation = None
    if include_evaluation:
        mesh_distance = np.asarray(mesh.point_data["implicit_distance"])
        _, _, evaluation = _evaluation_points(
            mesh,
            mesh_distance,
            case_id,
            split,
            config,
            training=False,
        )
    del mesh, sampled, grid
    gc.collect()
    return FinalGridCase(
        case_id=case_id,
        split=split,
        features=torch.from_numpy(feature_grid.copy()).float(),
        target=torch.from_numpy(target_grid.copy()).float(),
        valid=torch.from_numpy(valid_grid.copy()),
        bounds=bounds,
        evaluation=evaluation,
    )


def normalize_point_cases(
    train: list[FinalPointCase],
    evaluation: list[FinalPointCase],
    model_name: str,
    neighbors: int | None,
    normalizers: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if normalizers is None:
        if not train:
            raise ValueError("training cases are required to fit normalizers")
        feature_mean, feature_scale = fit_statistics(torch.cat([case.features for case in train]))
        target_mean, target_scale = fit_statistics(torch.cat([case.target for case in train]))
    else:
        feature_mean = normalizers["feature_mean"]
        feature_scale = normalizers["feature_scale"]
        target_mean = normalizers["target_mean"]
        target_scale = normalizers["target_scale"]
    for case in [*train, *evaluation]:
        if model_name == "gnn":
            if neighbors is None or case.wing_polygon is None:
                raise RuntimeError("GNN case is missing graph configuration")
            case.edge_index = wing_filtered_knn_edges(
                case.raw_positions,
                neighbors,
                case.wing_polygon,
            )
        case.features = apply_statistics(case.features, feature_mean, feature_scale)
        case.target = apply_statistics(case.target, target_mean, target_scale)
        case.raw_positions = case.features[:, :2]
        case.wing_polygon = None
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


def normalize_grid_cases(
    train: list[FinalGridCase],
    evaluation: list[FinalGridCase],
    normalizers: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if normalizers is None:
        if not train:
            raise ValueError("training cases are required to fit normalizers")
        train_features = torch.cat(
            [case.features.permute(1, 2, 0)[case.valid.squeeze(0)] for case in train]
        )
        train_targets = torch.cat(
            [case.target.permute(1, 2, 0)[case.valid.squeeze(0)] for case in train]
        )
        feature_mean, feature_scale = fit_statistics(train_features)
        target_mean, target_scale = fit_statistics(train_targets)
    else:
        feature_mean = normalizers["feature_mean"]
        feature_scale = normalizers["feature_scale"]
        target_mean = normalizers["target_mean"]
        target_scale = normalizers["target_scale"]
    feature_mean_grid = feature_mean.reshape(-1, 1, 1)
    feature_scale_grid = feature_scale.reshape(-1, 1, 1)
    target_mean_grid = target_mean.reshape(-1, 1, 1)
    target_scale_grid = target_scale.reshape(-1, 1, 1)
    for case in [*train, *evaluation]:
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


def training_loss(
    model_name: str,
    model: nn.Module,
    case: FinalPointCase | FinalGridCase,
) -> torch.Tensor:
    if model_name == "mlp" and isinstance(case, FinalPointCase):
        return nn.functional.mse_loss(model(case.features), case.target)
    if model_name == "gnn" and isinstance(case, FinalPointCase):
        if case.edge_index is None:
            raise RuntimeError("GNN case has no edge index")
        return nn.functional.mse_loss(
            model(case.features, case.raw_positions, case.edge_index),
            case.target,
        )
    if model_name == "fno" and isinstance(case, FinalGridCase):
        return masked_grid_loss(model(case.features[None, ...]), case)
    raise TypeError(f"Unexpected case type for {model_name}")


def evaluate_case(
    model_name: str,
    model: nn.Module,
    case: FinalPointCase | FinalGridCase,
    normalizers: Mapping[str, torch.Tensor],
) -> dict[str, float | str]:
    if isinstance(case, FinalPointCase):
        if model_name == "mlp":
            normalized_prediction = model(case.features)
        elif model_name == "gnn" and case.edge_index is not None:
            normalized_prediction = model(case.features, case.raw_positions, case.edge_index)
        else:
            raise TypeError(f"Unexpected point case for {model_name}")
        evaluation = case.evaluation
    elif model_name == "fno" and isinstance(case, FinalGridCase):
        if case.evaluation is None:
            raise RuntimeError("FNO evaluation case has no common points")
        normalized_grid = model(case.features[None, ...])[0]
        normalized_prediction = bilinear_grid_to_points(
            normalized_grid,
            case.evaluation.positions,
            case.bounds,
            valid=case.valid.squeeze(0),
        )
        evaluation = case.evaluation
    else:
        raise TypeError(f"Unexpected case type for {model_name}")
    prediction = (
        normalized_prediction * normalizers["target_scale"]
        + normalizers["target_mean"]
    )
    metrics = case_pressure_metrics(
        prediction,
        evaluation.target,
        evaluation.surface_mask,
        evaluation.inlet_velocity,
    )
    return {"case_id": case.case_id, "split": case.split, **metrics}
