import math

import numpy as np
import torch

from sciai.airfrans.sampling import (
    SamplingConfig,
    fixed_sample_indices,
    input_features,
    normalize,
)


class TestAirfransSampling:
    def test_geometry_only_sampling_is_fixed(self) -> None:
        distance = np.array([0.0] * 8 + [-1.0] * 12)
        config = SamplingConfig(seed=11, surface_samples=4, volume_samples=6)
        first, surface_count, volume_count = fixed_sample_indices(distance, config)
        second, _, _ = fixed_sample_indices(distance, config)
        np.testing.assert_array_equal(first, second)
        assert (surface_count, volume_count) == (4, 6)
        assert np.count_nonzero(distance[first] == 0) == 4

    def test_input_features_do_not_accept_target_fields(self) -> None:
        position = np.array([[0.0, 1.0, 0.5], [2.0, 3.0, 0.5]])
        distance = np.array([0.0, -2.0])
        features, names = input_features(position, distance, "airFoil2D_SST_10.0_30.0_0_0_12")
        assert names == [
            "x",
            "y",
            "inlet_velocity_x",
            "inlet_velocity_y",
            "signed_distance",
            "geometry_surface",
        ]
        np.testing.assert_allclose(features[:, 2], 10.0 * math.cos(math.radians(30.0)))
        np.testing.assert_allclose(features[:, 3], 5.0)
        np.testing.assert_array_equal(features[:, 4:], [[0.0, 1.0], [2.0, 0.0]])

    def test_constant_feature_normalizes_to_exact_zero(self) -> None:
        values = torch.full((2048, 1), 30.05554962158203, dtype=torch.float32)
        normalized, _, scale = normalize(values)
        assert torch.equal(normalized, torch.zeros_like(normalized))
        assert float(scale) == 1.0