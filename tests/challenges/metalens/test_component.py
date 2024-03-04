"""Tests for `metalens.component`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from fmmax import fmm
from jax import tree_util

from invrs_gym.challenges.metalens import challenge, component


class MetalensComponentTest(unittest.TestCase):
    def test_density_has_expected_properties(self):
        mc = component.MetalensComponent(
            spec=challenge.METALENS_SPEC,
            sim_params=challenge.METALENS_SIM_PARAMS,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        self.assertEqual(params.lower_bound, 0.0)
        self.assertEqual(params.upper_bound, 1.0)
        self.assertSequenceEqual(params.periodic, (False, False))

        onp.testing.assert_array_equal(params.fixed_solid[:, :-1], False)
        onp.testing.assert_array_equal(params.fixed_solid[:, -1], True)
        onp.testing.assert_array_equal(params.fixed_void[:, 0], True)
        onp.testing.assert_array_equal(params.fixed_void[:300, :-1], True)
        onp.testing.assert_array_equal(params.fixed_void[-300:, :-1], True)
        onp.testing.assert_array_equal(params.fixed_void[300:-300, 1:-1], False)

    def test_can_jit_response(self):
        mc = component.MetalensComponent(
            spec=challenge.METALENS_SPEC,
            sim_params=dataclasses.replace(
                challenge.METALENS_SIM_PARAMS,
                approximate_num_terms=100,
                formulation=fmm.Formulation.FFT,
                num_layers=1,
            ),
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))

        @jax.jit
        def jit_response_fn(params):
            return mc.response(params)

        jit_response_fn(params)


class MetalensResponseTest(unittest.TestCase):
    def test_flatten_unflatten(self):
        original = component.MetalensResponse(
            wavelength=jnp.arange(3),
            enhancement_ex=jnp.arange(3, 6),
            enhancement_ey=jnp.arange(6, 9),
        )
        leaves, treedef = tree_util.tree_flatten(original)
        restored = tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(restored.wavelength, original.wavelength)
        onp.testing.assert_array_equal(restored.enhancement_ex, original.enhancement_ex)
        onp.testing.assert_array_equal(restored.enhancement_ey, original.enhancement_ey)
