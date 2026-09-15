import pytest
import torch

from sciai.airfrans.metrics import (
    bilinear_grid_to_points,
    case_pressure_metrics,
    select_validation_checkpoint,
)
from sciai.airfrans.models import wing_filtered_knn_edges


class TestModelComparisonContract:
    def test_wing_filter_excludes_edges_through_polygon(self) -> None:
        positions = torch.tensor(
            [[-2.0, 0.0], [2.0, 0.0], [-2.0, 1.5], [2.0, 1.5]],
            dtype=torch.float64,
        )
        wing = torch.tensor(
            [[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]],
            dtype=torch.float64,
        )
        edges = wing_filtered_knn_edges(positions, neighbors=1, wing_polygon=wing)
        edge_pairs = set(zip(edges[0].tolist(), edges[1].tolist()))
        assert (0, 1) not in edge_pairs
        assert (1, 0) not in edge_pairs
        assert edges.shape == (2, 4)

    def test_wing_filter_expands_beyond_initial_candidate_window(self) -> None:
        right_side = torch.column_stack(
            (torch.full((70,), 2.0), torch.linspace(-0.4, 0.4, 70))
        )
        positions = torch.cat(
            (torch.tensor([[-2.0, 0.0]]), right_side, torch.tensor([[-2.0, 10.0]]))
        )
        wing = torch.tensor(
            [[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]]
        )
        edges = wing_filtered_knn_edges(positions, neighbors=1, wing_polygon=wing)
        sender_for_first_node = int(edges[1, edges[0] == 0].item())
        assert sender_for_first_node == 71

    def test_bilinear_interpolation_preserves_xy_axis_order(self) -> None:
        x = torch.linspace(-2.0, 2.0, 9)
        y = torch.linspace(-1.0, 3.0, 7)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        values = (2 * x_grid - 3 * y_grid + 5)[None, ...]
        points = torch.tensor([[-1.5, -0.5], [0.25, 1.5], [1.75, 2.5]])
        actual = bilinear_grid_to_points(values, points, (-2.0, 2.0, -1.0, 3.0))
        expected = (2 * points[:, 0] - 3 * points[:, 1] + 5)[:, None]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_bilinear_interpolation_ignores_invalid_grid_values(self) -> None:
        values = torch.tensor([[[1.0, 1.0], [1.0, 1000.0]]])
        valid = torch.tensor([[True, True], [True, False]])
        actual = bilinear_grid_to_points(
            values,
            torch.tensor([[0.5, 0.5]]),
            (0.0, 1.0, 0.0, 1.0),
            valid=valid,
        )
        torch.testing.assert_close(actual, torch.ones(1, 1))

    def test_bilinear_interpolation_uses_nearest_valid_fallback(self) -> None:
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        valid = torch.tensor([[True, False], [False, False]])
        actual = bilinear_grid_to_points(
            values,
            torch.tensor([[1.0, 1.0]]),
            (0.0, 1.0, 0.0, 1.0),
            valid=valid,
        )
        torch.testing.assert_close(actual, torch.tensor([[1.0]]))

    def test_metrics_use_case_dynamic_pressure_and_surface_mask(self) -> None:
        metrics = case_pressure_metrics(
            prediction=torch.tensor([2.0, -2.0, 4.0]),
            target=torch.zeros(3),
            surface_mask=torch.tensor([True, False, True]),
            inlet_velocity=(2.0, 0.0),
        )
        assert metrics["pressure_rmse_nondimensional"] == pytest.approx(2**0.5)
        assert metrics["surface_mae_nondimensional"] == pytest.approx(1.5)
        assert metrics["dynamic_pressure_per_density_m2_per_s2"] == 2.0

    def test_checkpoint_selection_uses_rmse_then_earliest_epoch(self) -> None:
        records = [
            {"epoch": 20, "validation_pressure_rmse_nondimensional": 0.4, "path": "b"},
            {"epoch": 10, "validation_pressure_rmse_nondimensional": 0.4, "path": "a"},
            {"epoch": 30, "validation_pressure_rmse_nondimensional": 0.5, "path": "c"},
        ]
        assert select_validation_checkpoint(records)["path"] == "a"
