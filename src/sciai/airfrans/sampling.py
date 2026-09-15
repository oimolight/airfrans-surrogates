from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class SamplingConfig:
    seed: int
    surface_samples: int
    volume_samples: int


class GridCase(Protocol):
    target: torch.Tensor
    valid: torch.Tensor


def fixed_sample_indices(
    implicit_distance: np.ndarray,
    config: SamplingConfig,
) -> tuple[np.ndarray, int, int]:
    geometry_surface = np.asarray(implicit_distance) == 0
    surface_indices = np.flatnonzero(geometry_surface)
    volume_indices = np.flatnonzero(~geometry_surface)
    if len(surface_indices) < config.surface_samples or len(volume_indices) < config.volume_samples:
        raise RuntimeError("Case does not contain enough surface or volume points")
    generator = np.random.default_rng(config.seed)
    selected_surface = generator.choice(
        surface_indices,
        size=config.surface_samples,
        replace=False,
    )
    selected_volume = generator.choice(
        volume_indices,
        size=config.volume_samples,
        replace=False,
    )
    selected = np.concatenate((selected_surface, selected_volume))
    generator.shuffle(selected)
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
    names = [
        "x",
        "y",
        "inlet_velocity_x",
        "inlet_velocity_y",
        "signed_distance",
        "geometry_surface",
    ]
    return features, names


def normalize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    statistics = values.to(torch.float64)
    mean = statistics.mean(dim=0, keepdim=True)
    standard_deviation = statistics.std(dim=0, unbiased=False, keepdim=True)
    constant = statistics.amax(dim=0, keepdim=True) == statistics.amin(dim=0, keepdim=True)
    scale = torch.where(
        constant,
        torch.ones_like(standard_deviation),
        standard_deviation.clamp_min(1e-12),
    )
    normalized = ((statistics - mean) / scale).to(values.dtype)
    return normalized, mean, scale


def fit_statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, mean, scale = normalize(values)
    return mean.to(values.dtype), scale.to(values.dtype)


def apply_statistics(
    values: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return (values - mean) / scale


def masked_grid_loss(prediction: torch.Tensor, case: GridCase) -> torch.Tensor:
    mask = case.valid[None, ...].to(prediction.dtype)
    return ((prediction - case.target[None, ...]).square() * mask).sum() / mask.sum()
