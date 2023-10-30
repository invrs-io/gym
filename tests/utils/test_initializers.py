"""Tests for `utils.initializers`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized
from totypes import types

from invrs_gym.utils import initializers


class TestNoisyInitializer(unittest.TestCase):
    @parameterized.expand(
        [
            [(20, 30), 0.2, -1, 1, 1, 1],
            [(20, 30), 0.2, -1, 1, 7, 1],
            [(20, 30), 0.2, -1, 1, 1, 7],
            [(20, 30), 0.2, -1, 1, 7, 7],
            [(5, 10, 20, 30), 2.7, 2, 4, 1, 1],
            [(5, 10, 20, 30), -3.1, -4, -2, 1, 1],
        ]
    )
    def test_noise_is_additive(
        self, shape, mean_value, lower_bound, upper_bound, min_spacing, min_width
    ):
        density = types.Density2DArray(
            array=jnp.full(shape, mean_value),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            minimum_spacing=min_spacing,
            minimum_width=min_width,
        )
        density_with_noise = initializers.noisy_density_initializer(
            key=jax.random.PRNGKey(0),
            seed_density=density,
            relative_stddev=0.0,
        )
        onp.testing.assert_allclose(
            jnp.mean(density_with_noise.array), mean_value, rtol=1e-5
        )
