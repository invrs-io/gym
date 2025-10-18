"""Tests for `utils.metrics`.

Copyright (c) 2025 invrs.io LLC
"""

import unittest

import jax.numpy as jnp
from parameterized import parameterized
from totypes import types

from invrs_gym.utils import metrics, transforms


class BinarizationDegreeTest(unittest.TestCase):
    def test_no_densities_is_none(self):
        params = [1.0, 2.0]
        self.assertIsNone(metrics.binarization_degree(params))

    @parameterized.expand(
        [
            [[(1, -1)], 1, 1],
            [[(0, 0)], 0, 1],
            [[(1, 2)], 0, 1],
            [[(1, 3)], -1, 1],
            [[(1, -1)], 1, 10],
            [[(0, 0)], 0, 10],
            [[(1, 2)], 0, 10],
            [[(1, 3)], -1, 10],
            [[(1, -1)], 1, 0.1],
            [[(0, 0)], 0, 0.1],
            [[(1, 2)], 0, 0.1],
            [[(1, 3)], -1, 0.1],
        ]
    )
    def test_binarization_degree_matches_expected(self, array, expected, scale):
        array = jnp.kron(jnp.asarray(array), jnp.ones((5, 5)))
        self.assertEqual(
            metrics.binarization_degree(
                types.Density2DArray(
                    array=array * scale,
                    lower_bound=-scale,
                    upper_bound=scale,
                ),
            ),
            expected,
        )

    @parameterized.expand(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [10, 0],
            [0, 0],
            [1, 1],
            [2, 2],
            [10, 10],
        ]
    )
    def test_binarization_for_downsampled_binary_circle(self, xshift, yshift):
        x, y = jnp.meshgrid(jnp.arange(-50, 50), jnp.arange(-50, 50))
        circle = (((x - xshift) ** 2 + (y - yshift) ** 2) < 35**2).astype(float)
        circle = transforms.box_downsample(circle, (20, 20))
        circle_density = types.Density2DArray(
            array=circle,
            lower_bound=0,
            upper_bound=1,
        )
        self.assertEqual(metrics.binarization_degree(circle_density), 1.0)

    @parameterized.expand(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [10, 0],
            [0, 0],
            [1, 1],
            [2, 2],
            [10, 10],
        ]
    )
    def test_binarization_for_downsampled_binary_square(self, xshift, yshift):
        x, y = jnp.meshgrid(jnp.arange(-50, 50), jnp.arange(-50, 50))
        square = ((x - xshift) ** 2 < 35**2) & ((y - yshift) ** 2 < 35**2)
        square = square.astype(float)
        square = transforms.box_downsample(square, (20, 20))
        square_density = types.Density2DArray(
            array=square,
            lower_bound=0,
            upper_bound=1,
        )
        self.assertEqual(metrics.binarization_degree(square_density), 1.0)

    @parameterized.expand(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [10, 0],
            [0, 0],
            [1, 1],
            [2, 2],
            [10, 10],
        ]
    )
    def test_binarization_for_downsampled_triangle(self, xshift, yshift):
        x, y = jnp.meshgrid(jnp.arange(-50, 50), jnp.arange(-50, 50))
        triangle = x + y > 0
        triangle = transforms.box_downsample(triangle, (20, 20))
        triangle_density = types.Density2DArray(
            array=triangle,
            lower_bound=0,
            upper_bound=1,
        )
        self.assertEqual(metrics.binarization_degree(triangle_density), 1.0)

    def test_batch_dim(self):
        params = {
            "a": types.Density2DArray(
                array=jnp.ones((10, 10)), lower_bound=0, upper_bound=1
            ),
            "b": types.Density2DArray(
                array=jnp.zeros((5, 10, 10)), lower_bound=0, upper_bound=1
            ),
            "c": types.Density2DArray(
                array=jnp.ones((5, 5, 10, 10)), lower_bound=0, upper_bound=1
            ),
        }
        self.assertEqual(metrics.binarization_degree(params), 1.0)
