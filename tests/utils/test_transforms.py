"""Tests for `utils.transforms`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

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
        onp.testing.assert_allclose(result, expected)


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
