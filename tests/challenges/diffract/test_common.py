"""Tests for `diffract.common`."""

import unittest

import jax.numpy as jnp
import numpy as onp
from fmmax import basis
from jax import tree_util

from invrs_gym.challenges.diffract import common


class GatingResponseTest(unittest.TestCase):
    def test_flatten_unflatten(self):
        original = common.GratingResponse(
            wavelength=jnp.arange(3),
            transmission_efficiency=jnp.arange(4),
            reflection_efficiency=jnp.arange(5),
            expansion=basis.Expansion(
                basis_coefficients=onp.arange(10).reshape((5, 2))
            ),
        )
        leaves, treedef = tree_util.tree_flatten(original)
        restored = tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(restored.wavelength, original.wavelength)
        onp.testing.assert_array_equal(
            restored.transmission_efficiency, original.transmission_efficiency
        )
        onp.testing.assert_array_equal(
            restored.reflection_efficiency, original.reflection_efficiency
        )
        onp.testing.assert_array_equal(restored.expansion, original.expansion)
