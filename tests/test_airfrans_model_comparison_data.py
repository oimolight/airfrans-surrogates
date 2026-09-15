import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from torch import nn

from sciai.airfrans.data import (
    EvaluationPoints,
    FinalGridCase,
    deterministic_point_seed,
    evaluate_case,
    load_grid_case,
    ordered_boundary_polygon,
)


class ConstantGridModel(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.ones((len(features), 1, *features.shape[-2:]))


class TestModelComparisonData:
    def test_evaluation_seed_is_deterministic_and_split_specific(self) -> None:
        first = deterministic_point_seed("evaluation", "validation", "case-a")
        second = deterministic_point_seed("evaluation", "validation", "case-a")
        test_seed = deterministic_point_seed("evaluation", "id_test", "case-a")
        assert first == second
        assert first != test_seed

    def test_ordered_boundary_polygon_follows_closed_line_cycle(self) -> None:
        surface = pv.PolyData(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        )
        surface.lines = np.array([2, 1, 2, 2, 3, 0, 2, 0, 1, 2, 2, 3])
        polygon = ordered_boundary_polygon(surface)
        assert polygon.shape == (4, 2)
        assert set(map(tuple, polygon.tolist())) == {
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        }

    def test_grid_loader_uses_resolved_model_config(self) -> None:
        case_id = "airFoil2D_SST_10.0_2.0"
        mesh = pv.ImageData(dimensions=(2, 2, 1)).cast_to_unstructured_grid()
        mesh.point_data["implicit_distance"] = np.ones(mesh.n_points)
        mesh.point_data["p"] = np.arange(mesh.n_points, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_directory = Path(temporary_directory) / case_id
            case_directory.mkdir()
            mesh.save(case_directory / f"{case_id}_internal.vtu")
            case = load_grid_case(
                Path(temporary_directory),
                case_id,
                "train",
                {"model": {"grid_height": 2, "grid_width": 2}},
                include_evaluation=False,
            )
        assert case.features.shape == (6, 2, 2)
        assert case.target.shape == (1, 2, 2)

    def test_fno_evaluation_interpolates_denormalizes_and_scores(self) -> None:
        case = FinalGridCase(
            case_id="case-a",
            split="validation",
            features=torch.zeros(6, 2, 2),
            target=torch.zeros(1, 2, 2),
            valid=torch.ones(1, 2, 2, dtype=torch.bool),
            bounds=(0.0, 1.0, 0.0, 1.0),
            evaluation=EvaluationPoints(
                positions=torch.tensor([[0.5, 0.5]]),
                target=torch.tensor([[10.0]]),
                surface_mask=torch.tensor([True]),
                inlet_velocity=(2.0, 0.0),
            ),
        )
        metrics = evaluate_case(
            "fno",
            ConstantGridModel(),
            case,
            {
                "target_mean": torch.tensor([[10.0]]),
                "target_scale": torch.tensor([[2.0]]),
            },
        )
        assert metrics["pressure_rmse_nondimensional"] == 1.0
        assert metrics["surface_mae_nondimensional"] == 1.0
