from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


class MessageBlock(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_width + 2, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        receivers, senders = edge_index
        relative_position = positions[senders] - positions[receivers]
        messages = self.message(
            torch.cat((hidden[receivers], hidden[senders], relative_position), dim=-1)
        )
        aggregated = torch.zeros_like(hidden)
        aggregated.index_add_(0, receivers, messages)
        degree = torch.bincount(receivers, minlength=len(hidden)).clamp_min(1).to(hidden.dtype)
        aggregated = aggregated / degree[:, None]
        return hidden + self.update(torch.cat((hidden, aggregated), dim=-1))


class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_features: int = 6, hidden_width: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_features, hidden_width), nn.ReLU())
        self.blocks = nn.ModuleList(MessageBlock(hidden_width) for _ in range(blocks))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(features)
        for block in self.blocks:
            hidden = block(hidden, positions, edge_index)
        return self.decoder(hidden)


def knn_edges(positions: torch.Tensor, neighbors: int) -> torch.Tensor:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [nodes, 2]")
    if neighbors < 1 or neighbors >= len(positions):
        raise ValueError("neighbors must be between 1 and node_count - 1")
    distances = torch.cdist(positions, positions)
    nearest = distances.topk(neighbors + 1, largest=False).indices[:, 1:]
    receivers = torch.arange(len(positions), device=positions.device).repeat_interleave(neighbors)
    return torch.stack((receivers, nearest.reshape(-1)))


def _cross_2d(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _segments_cross_wing(
    starts: torch.Tensor,
    ends: torch.Tensor,
    polygon: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    boundary_starts = polygon
    boundary_ends = polygon.roll(-1, dims=0)
    boundary_directions = boundary_ends - boundary_starts
    tolerance = 32 * torch.finfo(starts.dtype).eps
    crosses = torch.empty(len(starts), dtype=torch.bool)
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        batch_ends = ends[offset : offset + batch_size]
        directions = batch_ends - batch_starts
        relative_boundary_starts = boundary_starts[None, ...] - batch_starts[:, None, :]
        denominators = _cross_2d(directions[:, None, :], boundary_directions[None, ...])
        nonparallel = denominators.abs() > tolerance
        safe_denominators = torch.where(nonparallel, denominators, torch.ones_like(denominators))
        segment_fractions = _cross_2d(
            relative_boundary_starts,
            boundary_directions[None, ...],
        ) / safe_denominators
        boundary_fractions = _cross_2d(
            relative_boundary_starts,
            directions[:, None, :],
        ) / safe_denominators
        intersects_boundary = (
            nonparallel
            & (segment_fractions > tolerance)
            & (segment_fractions < 1 - tolerance)
            & (boundary_fractions >= 0)
            & (boundary_fractions <= 1)
        ).any(dim=1)

        midpoints = (batch_starts + batch_ends) / 2
        midpoint_y = midpoints[:, 1, None]
        y1 = boundary_starts[None, :, 1]
        y2 = boundary_ends[None, :, 1]
        ray_candidates = (y1 > midpoint_y) != (y2 > midpoint_y)
        safe_y_delta = torch.where(ray_candidates, y2 - y1, torch.ones_like(y2 - y1))
        crossing_x = (
            (boundary_ends[None, :, 0] - boundary_starts[None, :, 0])
            * (midpoint_y - y1)
            / safe_y_delta
            + boundary_starts[None, :, 0]
        )
        midpoint_inside = (
            ray_candidates & (midpoints[:, 0, None] < crossing_x)
        ).sum(dim=1) % 2 == 1
        crosses[offset : offset + len(batch_starts)] = intersects_boundary | midpoint_inside
    return crosses


def wing_filtered_knn_edges(
    positions: torch.Tensor,
    neighbors: int,
    wing_polygon: torch.Tensor,
) -> torch.Tensor:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [nodes, 2]")
    if wing_polygon.ndim != 2 or wing_polygon.shape[1] != 2 or len(wing_polygon) < 3:
        raise ValueError("wing_polygon must have shape [boundary_nodes, 2]")
    if neighbors < 1 or neighbors >= len(positions):
        raise ValueError("neighbors must be between 1 and node_count - 1")
    if positions.device.type != "cpu" or wing_polygon.device.type != "cpu":
        raise ValueError("wing-filtered graph construction currently requires CPU tensors")
    if not positions.is_floating_point() or not wing_polygon.is_floating_point():
        raise ValueError("positions and wing_polygon must use floating-point dtypes")

    distances = torch.cdist(positions, positions)
    candidate_count = min(len(positions) - 1, max(64, 4 * neighbors))
    candidates = distances.topk(candidate_count + 1, largest=False).indices[:, 1:]
    candidate_receivers = torch.arange(len(positions)).repeat_interleave(candidate_count)
    candidate_senders = candidates.reshape(-1)
    crosses_wing = _segments_cross_wing(
        positions[candidate_receivers],
        positions[candidate_senders],
        wing_polygon.to(dtype=positions.dtype),
    ).reshape(len(positions), candidate_count)
    receivers: list[int] = []
    senders: list[int] = []
    for receiver in range(len(positions)):
        valid_senders = candidates[receiver][~crosses_wing[receiver]]
        if len(valid_senders) < neighbors:
            all_candidates = distances[receiver].argsort()
            all_candidates = all_candidates[all_candidates != receiver]
            expanded_crosses = _segments_cross_wing(
                positions[receiver].expand(len(all_candidates), -1),
                positions[all_candidates],
                wing_polygon.to(dtype=positions.dtype),
            )
            valid_senders = all_candidates[~expanded_crosses]
        if len(valid_senders) < neighbors:
            raise RuntimeError(f"Node {receiver} has only {len(valid_senders)} non-crossing neighbors")
        for sender in valid_senders[:neighbors].tolist():
            receivers.append(receiver)
            senders.append(sender)
    return torch.tensor((receivers, senders), dtype=torch.long, device=positions.device)


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        scale = 1 / (width * width)
        shape = (width, width, modes, modes)
        self.modes = modes
        self.positive_weights = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.negative_weights = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = values.shape
        transformed = torch.fft.rfft2(values)
        output = torch.zeros(
            batch,
            values.shape[1],
            height,
            width // 2 + 1,
            dtype=transformed.dtype,
            device=values.device,
        )
        x_modes = min(self.modes, height // 2)
        y_modes = min(self.modes, width // 2 + 1)
        output[:, :, :x_modes, :y_modes] = self.multiply(
            transformed[:, :, :x_modes, :y_modes],
            self.positive_weights[:, :, :x_modes, :y_modes],
        )
        output[:, :, -x_modes:, :y_modes] = self.multiply(
            transformed[:, :, -x_modes:, :y_modes],
            self.negative_weights[:, :, :x_modes, :y_modes],
        )
        return torch.fft.irfft2(output, s=(height, width))


class FourierNeuralOperator(nn.Module):
    def __init__(
        self,
        input_features: int = 6,
        width: int = 32,
        blocks: int = 4,
        modes: int = 12,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(input_features, width, 1)
        self.spectral_layers = nn.ModuleList(SpectralConv2d(width, modes) for _ in range(blocks))
        self.local_layers = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(blocks))
        self.projection = nn.Sequential(
            nn.Conv2d(width, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.lift(features)
        for index, (spectral, local) in enumerate(zip(self.spectral_layers, self.local_layers)):
            values = spectral(values) + local(values)
            if index + 1 < len(self.spectral_layers):
                values = F.gelu(values)
        return self.projection(values)


def trainable_scalar_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
        if parameter.requires_grad
    )


