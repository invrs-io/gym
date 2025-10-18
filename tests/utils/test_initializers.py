"""Tests for `utils.initializers`.

Copyright (c) 2025 invrs.io LLC
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized
from totypes import types

from invrs_gym.utils import initializers


class TestNoisyDensityInitializer(unittest.TestCase):
    @parameterized.expand(
        [
            [(20, 30), 0.2, -1, 1, 1, 1],
            [(20, 30), 0.2, -1, 1, 7, 1],
            [(20, 30), 0.2, -1, 1, 1, 7],
            [(20, 30), 0.2, -1, 1, 7, 7],
            [(5, 10, 20, 30), 0.5, 2, 4, 1, 1],
            [(5, 10, 20, 30), 0.8, -4, -2, 1, 1],
        ]
    )
    def test_density_has_expected_mean(
        self, shape, relative_mean, lower_bound, upper_bound, min_spacing, min_width
    ):
        seed_density = types.Density2DArray(
            array=jnp.full(shape, -1),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            minimum_spacing=min_spacing,
            minimum_width=min_width,
        )
        density_with_noise = initializers.noisy_density_initializer(
            key=jax.random.PRNGKey(0),
            seed_density=seed_density,
            relative_mean=relative_mean,
            relative_noise_amplitude=1e-4,
        )
        expected_mean = lower_bound + (upper_bound - lower_bound) * relative_mean
        onp.testing.assert_allclose(
            jnp.mean(density_with_noise.array), expected_mean, rtol=1e-4
        )
