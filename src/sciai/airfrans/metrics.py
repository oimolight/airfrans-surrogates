from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F


def bilinear_grid_to_points(
    values: torch.Tensor,
    points: torch.Tensor,
    bounds: Sequence[float],
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("values must have shape [channels, height, width]")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [points, 2]")
    if len(bounds) != 4:
        raise ValueError("bounds must be (x_min, x_max, y_min, y_max)")
    x_min, x_max, y_min, y_max = bounds
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("grid bounds must have positive width and height")
    tolerance = 32 * torch.finfo(points.dtype).eps
    if bool(
        (points[:, 0] < x_min - tolerance).any()
        or (points[:, 0] > x_max + tolerance).any()
        or (points[:, 1] < y_min - tolerance).any()
        or (points[:, 1] > y_max + tolerance).any()
    ):
        raise ValueError("evaluation points fall outside the FNO grid")
    normalized_x = 2 * (points[:, 0] - x_min) / (x_max - x_min) - 1
    normalized_y = 2 * (points[:, 1] - y_min) / (y_max - y_min) - 1
    sampling_grid = torch.stack((normalized_x, normalized_y), dim=-1).reshape(1, -1, 1, 2)
    if valid is not None:
        if valid.shape != values.shape[-2:]:
            raise ValueError("valid must match the grid height and width")
        weights = valid.to(dtype=values.dtype)[None, None, ...]
        weighted_values = values[None, ...] * weights
    else:
        weights = torch.ones((1, 1, *values.shape[-2:]), dtype=values.dtype)
        weighted_values = values[None, ...]
    sampled_values = F.grid_sample(
        weighted_values,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled_weights = F.grid_sample(
        weights,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    unsupported = sampled_weights[0, 0, :, 0] <= 0
    denominator = sampled_weights.clamp_min(torch.finfo(values.dtype).tiny)
    sampled = (sampled_values / denominator)[0, :, :, 0].transpose(0, 1)
    if bool(unsupported.any()):
        if valid is None or not bool(valid.any()):
            raise ValueError("FNO grid has no valid values")
        valid_indices = valid.nonzero()
        height, width = valid.shape
        valid_x = x_min + valid_indices[:, 1] * (x_max - x_min) / (width - 1)
        valid_y = y_min + valid_indices[:, 0] * (y_max - y_min) / (height - 1)
        valid_positions = torch.stack((valid_x, valid_y), dim=1).to(points.dtype)
        nearest = torch.cdist(points[unsupported], valid_positions).argmin(dim=1)
        nearest_indices = valid_indices[nearest]
        sampled[unsupported] = values[
            :,
            nearest_indices[:, 0],
            nearest_indices[:, 1],
        ].transpose(0, 1)
    return sampled


def case_pressure_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    surface_mask: torch.Tensor,
    inlet_velocity: Sequence[float],
) -> dict[str, float]:
    prediction = prediction.reshape(-1).to(torch.float64)
    target = target.reshape(-1).to(torch.float64)
    surface_mask = surface_mask.reshape(-1).bool()
    if prediction.shape != target.shape or target.shape != surface_mask.shape:
        raise ValueError("prediction, target, and surface_mask must have equal lengths")
    if not bool(surface_mask.any()):
        raise ValueError("surface_mask must contain at least one evaluation point")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise ValueError("prediction and target must be finite")
    if len(inlet_velocity) != 2:
        raise ValueError("inlet_velocity must contain x and y components")
    dynamic_pressure_per_density = 0.5 * sum(float(component) ** 2 for component in inlet_velocity)
    if dynamic_pressure_per_density <= 0:
        raise ValueError("inlet speed must be positive")
    nondimensional_error = (prediction - target) / dynamic_pressure_per_density
    return {
        "pressure_rmse_nondimensional": float(torch.sqrt(nondimensional_error.square().mean())),
        "surface_mae_nondimensional": float(nondimensional_error[surface_mask].abs().mean()),
        "dynamic_pressure_per_density_m2_per_s2": dynamic_pressure_per_density,
        "evaluation_point_count": float(len(target)),
        "surface_point_count": float(surface_mask.sum()),
    }


def select_validation_checkpoint(
    records: Sequence[Mapping[str, float | int | str]],
) -> Mapping[str, float | int | str]:
    if not records:
        raise ValueError("at least one validation record is required")
    required = {"epoch", "validation_pressure_rmse_nondimensional"}
    if any(not required.issubset(record) for record in records):
        raise ValueError("validation records are missing selection fields")
    return min(
        records,
        key=lambda record: (
            float(record["validation_pressure_rmse_nondimensional"]),
            int(record["epoch"]),
        ),
    )
