from pathlib import Path

import torch

from sciai.airfrans import model_comparison
from sciai.airfrans.edge_feature_ablation import (
    SOURCE_FILE,
    EXPERIMENT_ID,
    ZeroRelativeEdgeFeatureGNN,
    evaluator_source_hashes,
    installed_ablation_runner,
    make_ablation_model,
    validate_ablation_protocol,
)
from sciai.airfrans.models import GraphNeuralNetwork, trainable_scalar_count


class TestEdgeFeatureAblation:
    repository_root = Path(__file__).resolve().parents[1]

    def test_ablation_preserves_baseline_parameters(self) -> None:
        torch.manual_seed(17)
        baseline = GraphNeuralNetwork(input_features=6, hidden_width=8, blocks=2)
        torch.manual_seed(17)
        ablation = ZeroRelativeEdgeFeatureGNN(input_features=6, hidden_width=8, blocks=2)

        assert baseline.state_dict().keys() == ablation.state_dict().keys()
        assert trainable_scalar_count(baseline) == trainable_scalar_count(ablation)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, ablation.state_dict()[name])

    def test_ablation_ignores_positions_after_adjacency_is_fixed(self) -> None:
        torch.manual_seed(17)
        model = ZeroRelativeEdgeFeatureGNN(input_features=6, hidden_width=8, blocks=2)
        features = torch.randn(6, 6)
        positions = torch.randn(6, 2)
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]]
        )

        first = model(features, positions, edge_index)
        second = model(features, positions * 11.0 + 7.0, edge_index)

        torch.testing.assert_close(first, second)

    def test_ablation_retains_node_features_and_adjacency(self) -> None:
        torch.manual_seed(29)
        model = ZeroRelativeEdgeFeatureGNN(input_features=6, hidden_width=8, blocks=2)
        features = torch.randn(6, 6)
        positions = torch.randn(6, 2)
        first_edges = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]]
        )
        second_edges = first_edges.clone()
        second_edges[1] = second_edges[1].roll(1)

        baseline = model(features, positions, first_edges)
        changed_geometry_features = features.clone()
        changed_geometry_features[:, :2] += 1.0

        assert not torch.allclose(baseline, model(changed_geometry_features, positions, first_edges))
        assert not torch.allclose(baseline, model(features, positions, second_edges))

    def test_config_changes_only_the_edge_feature(self) -> None:
        config = validate_ablation_protocol(
            self.repository_root / "configs/airfrans/edge_feature_ablation.json",
            self.repository_root / "configs/airfrans/model_comparison.json",
            self.repository_root / "data/manifests/airfrans_selected_cases.json",
        )
        model = make_ablation_model("gnn", {"model": config["models"]["gnn"]})
        assert isinstance(model, ZeroRelativeEdgeFeatureGNN)

    def test_runner_patch_is_scoped_and_source_is_hashed(self) -> None:
        original_experiment_id = model_comparison.EXPERIMENT_ID
        original_factory = model_comparison.make_model
        with installed_ablation_runner():
            assert model_comparison.EXPERIMENT_ID == EXPERIMENT_ID
            assert model_comparison.make_model is make_ablation_model
        assert model_comparison.EXPERIMENT_ID == original_experiment_id
        assert model_comparison.make_model is original_factory
        assert SOURCE_FILE.as_posix() in evaluator_source_hashes(self.repository_root)
