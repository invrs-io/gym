"""Tests for `metagrating.component`."""

import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from fmmax import basis
from jax import tree_util

from invrs_gym.challenge.metagrating import component

DESIGNS_DIR = pathlib.Path(__file__).resolve().parent / "designs"


class MetagratingComponentTest(unittest.TestCase):
    def test_density_has_expected_properties(self):
        mc = component.MetagratingComponent(
            spec=component.MetagratingSpec(),
            sim_params=component.MetagratingSimParams(),
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        self.assertEqual(params.lower_bound, 0.0)
        self.assertEqual(params.upper_bound, 1.0)
        self.assertSequenceEqual(params.periodic, (True, True))

    def test_can_jit_response(self):
        mc = component.MetagratingComponent(
            spec=component.MetagratingSpec(),
            sim_params=component.MetagratingSimParams(),
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))

        @jax.jit
        def jit_response_fn(params):
            return mc.response(params)

        jit_response_fn(params)


class MetagratingResponseTest(unittest.TestCase):
    def test_flatten_unflatten(self):
        original = component.MetagratingResponse(
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
