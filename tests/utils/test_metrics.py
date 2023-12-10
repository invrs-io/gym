"""Tests for `utils.metrics`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax.numpy as jnp
from parameterized import parameterized
from totypes import types

from invrs_gym.utils import metrics


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
        self.assertEqual(
            metrics.binarization_degree(
                types.Density2DArray(
                    array=jnp.asarray(array) * scale,
                    lower_bound=-scale,
                    upper_bound=scale,
                ),
            ),
            expected,
        )
