"""Tests for `extractor.component`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import fmmax
import jax
import jax.numpy as jnp
import numpy as onp
from jax import tree_util

from invrs_gym.challenges.extractor import challenge, component


class ExtractorComponentTest(unittest.TestCase):
    def test_density_has_expected_properties(self):
        ec = component.ExtractorComponent(
            spec=challenge.EXTRACTOR_SPEC,
            sim_params=challenge.EXTRACTOR_SIM_PARAMS,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = ec.init(jax.random.PRNGKey(0))
        self.assertEqual(params.lower_bound, 0.0)
        self.assertEqual(params.upper_bound, 1.0)
        self.assertSequenceEqual(params.periodic, (False, False))
        onp.testing.assert_array_equal(params.fixed_solid, False)

        pad = (ec.spec.grid_shape[0] - 300) // 2
        expected_fixed_void = onp.pad(
            onp.zeros((300, 300), bool),
            ((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=True,
        )
        onp.testing.assert_array_equal(params.fixed_void, expected_fixed_void)

    def test_can_jit_response(self):
        ec = component.ExtractorComponent(
            spec=challenge.EXTRACTOR_SPEC,
            sim_params=dataclasses.replace(
                challenge.EXTRACTOR_SIM_PARAMS,
                approximate_num_terms=100,
                formulation=fmmax.Formulation.FFT,
            ),
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = ec.init(jax.random.PRNGKey(0))

        @jax.jit
        def jit_response_fn(params):
            return ec.response(params)

        jit_response_fn(params)

    def test_multiple_wavelengths(self):
        ec = component.ExtractorComponent(
            spec=challenge.EXTRACTOR_SPEC,
            sim_params=dataclasses.replace(
                challenge.EXTRACTOR_SIM_PARAMS,
                approximate_num_terms=100,
                formulation=fmmax.Formulation.FFT,
            ),
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = ec.init(jax.random.PRNGKey(0))

        response, aux = ec.response(params, wavelength=jnp.asarray([0.637, 0.638]))
        self.assertSequenceEqual(
            response.extracted_power.shape,
            (2, 3),
        )


class ExtractorResponseTest(unittest.TestCase):
    def test_flatten_unflatten(self):
        original = component.ExtractorResponse(
            wavelength=jnp.arange(3),
            emitted_power=jnp.arange(4),
            extracted_power=jnp.arange(5),
            collected_power=jnp.arange(6),
            bare_substrate_emitted_power=jnp.arange(7),
            bare_substrate_extracted_power=jnp.arange(8),
            bare_substrate_collected_power=jnp.arange(9),
        )
        leaves, treedef = tree_util.tree_flatten(original)
        restored = tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(restored.wavelength, original.wavelength)
        onp.testing.assert_array_equal(
            restored.emitted_power,
            original.emitted_power,
        )
        onp.testing.assert_array_equal(
            restored.extracted_power,
            original.extracted_power,
        )
        onp.testing.assert_array_equal(
            restored.collected_power,
            original.collected_power,
        )
        onp.testing.assert_array_equal(
            restored.bare_substrate_emitted_power,
            original.bare_substrate_emitted_power,
        )
        onp.testing.assert_array_equal(
            restored.bare_substrate_extracted_power,
            original.bare_substrate_extracted_power,
        )
        onp.testing.assert_array_equal(
            restored.bare_substrate_collected_power,
            original.bare_substrate_collected_power,
        )
