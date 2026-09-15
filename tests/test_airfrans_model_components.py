import pytest
import torch

from sciai.airfrans.models import (
    FourierNeuralOperator,
    GraphNeuralNetwork,
    PointwiseMLP,
    knn_edges,
    trainable_scalar_count,
)


class TestModelComponents:
    def test_mlp_shape_and_gradient(self) -> None:
        torch.manual_seed(17)
        model = PointwiseMLP(input_features=6, hidden_width=8, hidden_layers=2)
        features = torch.randn(16, 6)
        prediction = model(features)
        prediction.square().mean().backward()
        assert prediction.shape == (16, 1)
        assert all(parameter.grad is not None for parameter in model.parameters())

    def test_gnn_shape_gradient_and_edges(self) -> None:
        torch.manual_seed(17)
        features = torch.randn(16, 6)
        positions = torch.randn(16, 2)
        edges = knn_edges(positions, neighbors=3)
        model = GraphNeuralNetwork(input_features=6, hidden_width=8, blocks=2)
        prediction = model(features, positions, edges)
        prediction.square().mean().backward()
        assert prediction.shape == (16, 1)
        assert edges.shape == (2, 48)
        assert all(parameter.grad is not None for parameter in model.parameters())

    def test_fno_shape_gradient_and_real_parameter_count(self) -> None:
        torch.manual_seed(17)
        model = FourierNeuralOperator(input_features=6, width=4, blocks=2, modes=3)
        features = torch.randn(2, 6, 8, 8)
        prediction = model(features)
        prediction.square().mean().backward()
        real_count = sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
        )
        assert prediction.shape == (2, 1, 8, 8)
        assert trainable_scalar_count(model) == real_count
        assert torch.isfinite(prediction).all()

    def test_knn_rejects_invalid_neighbor_count(self) -> None:
        with pytest.raises(ValueError):
            knn_edges(torch.zeros(4, 2), neighbors=4)
