"""Tests for `utils.transforms`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized
from totypes import types

from invrs_gym.utils import transforms


class InterpolateTest(unittest.TestCase):
    @parameterized.expand(
        [
            (4.0, 2.0, 0.0, 2.0),
            (4.0, 2.0, 1.0, 4.0),
            (4.0, 2.0, 0.5, (onp.sqrt(2) * 0.5 + onp.sqrt(4) * 0.5) ** 2),
            (4.0 + 1.0j, 2.0, 0.0, 2.0),
            (4.0 + 1.0j, 2.0, 1.0, 4.0 + 1.0j),
            (4.0 + 1.0j, 2.0, 0.5, (onp.sqrt(2) * 0.5 + onp.sqrt(4 + 1.0j) * 0.5) ** 2),
        ]
    )
    def test_interpolated_matches_expected(self, p_solid, p_void, density, expected):
        result = transforms.interpolate_permittivity(p_solid, p_void, density)
        onp.testing.assert_allclose(result, expected, rtol=1e-6)


class RescaledDensityTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                types.Density2DArray(
                    array=onp.asarray([[0.0, 1.0, 2.0]]),
                    lower_bound=0,
                    upper_bound=1,
                ),
                0.0,  # lower bound
                2.0,  # upper bound
                onp.asarray([[0.0, 2.0, 4.0]]),
            ],
            [
                types.Density2DArray(
                    array=onp.asarray([[0.0, 1.0, 2.0]]),
                    lower_bound=0,
                    upper_bound=1,
                ),
                -1.0,  # lower bound
                1.0,  # upper bound
                onp.asarray([[-1.0, 1.0, 3.0]]),
            ],
        ]
    )
    def test_rescaled_density(self, density, lower_bound, upper_bound, expected):
        onp.testing.assert_array_equal(
            transforms.rescaled_density_array(density, lower_bound, upper_bound),
            expected,
        )


class ResampleTest(unittest.TestCase):
    @parameterized.expand([[(1, 3)], [(2, 3)], [(1, 6)]])
    def test_downsampled_matches_expected(self, target_shape):
        # Downsampling where the target shape evenly divides the original
        # shape is equivalent to box downsampling.
        x = jnp.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        )
        expected = transforms.box_downsample(x, target_shape)
        result = transforms.resample(x, target_shape, method=jax.image.ResizeMethod.CUBIC)
        onp.testing.assert_allclose(result, expected)

    @parameterized.expand([[(4, 12)], [(2, 12)], [(3, 9)]])
    def test_upsampled_matches_expected(self, target_shape):
        # Upsampling is equivalent to `jax.image.resize`.
        x = jnp.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        )
        expected = jax.image.resize(
            x, target_shape, method=jax.image.ResizeMethod.CUBIC
        )
        result = transforms.resample(x, target_shape, method=jax.image.ResizeMethod.CUBIC)
        onp.testing.assert_allclose(result, expected)


class BoxDownsampleTest(unittest.TestCase):
    @parameterized.expand(
        [
            [(4, 4), (3, 3)],
            [(4, 3), (3, 1)],
            [(4, 4, 1), (1, 1)],
        ]
    )
    def test_downsample_factor_validation(self, arr_shape, target_shape):
        with self.assertRaisesRegex(
            ValueError, "Each axis of `shape` must evenly divide "
        ):
            transforms.box_downsample(jnp.ones(arr_shape), shape=target_shape)

    def test_downsampled_matches_expected(self):
        x = jnp.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        )
        expected = jnp.asarray([[14 / 4, 22 / 4, 30 / 4]])
        result = transforms.box_downsample(x, (1, 3))
        onp.testing.assert_allclose(result, expected)

    def test_upsample_downsample(self):
        onp.random.seed(0)
        shape = (5, 2, 8, 3, 3)
        factor = 4
        original = jnp.asarray(onp.random.rand(*shape))

        kernel = jnp.ones((factor,) * original.ndim, dtype=original.dtype)
        upsampled = jnp.kron(original, kernel)

        expected_shape = tuple([factor * d for d in shape])
        self.assertSequenceEqual(upsampled.shape, expected_shape)
        downsampled_upsampled = transforms.box_downsample(upsampled, original.shape)
        onp.testing.assert_allclose(downsampled_upsampled, original, rtol=1e-5)

    def test_upsample_downsample_asymmetric(self):
        onp.random.seed(0)
        shape = (20, 30)
        factor = 4
        original = jnp.asarray(onp.random.rand(*shape))

        kernel = jnp.ones((factor, 1), dtype=original.dtype)
        upsampled = jnp.kron(original, kernel)

        expected_shape = (factor * shape[0], shape[1])
        self.assertSequenceEqual(upsampled.shape, expected_shape)
        downsampled_upsampled = transforms.box_downsample(upsampled, original.shape)
        onp.testing.assert_allclose(downsampled_upsampled, original, rtol=1e-5)
