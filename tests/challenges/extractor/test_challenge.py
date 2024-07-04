"""Tests for `extractor.challenge`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import optax
import pytest
from fmmax import fmm
from parameterized import parameterized
from totypes import symmetry

from invrs_gym.challenges.extractor import challenge


class ExtractorChallengeTest(unittest.TestCase):
    @parameterized.expand([[lambda fn: fn], [jax.jit]])
    def test_optimize(self, step_fn_decorator):
        ec = challenge.photon_extractor(
            sim_params=dataclasses.replace(
                challenge.EXTRACTOR_SIM_PARAMS,
                approximate_num_terms=100,
                formulation=fmm.Formulation.FFT,
            )
        )

        def loss_fn(params):
            response, aux = ec.component.response(params)
            loss = ec.loss(response)
            return loss, (response, aux)

        opt = optax.adam(0.05)
        params = ec.component.init(jax.random.PRNGKey(0))
        state = opt.init(params)

        @step_fn_decorator
        def step_fn(params, state):
            (value, (response, aux)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            eval_metric = ec.eval_metric(response)
            metrics = ec.metrics(response, params, aux)
            updates, state = opt.update(grad, state)
            params = optax.apply_updates(params, updates)
            return params, state, eval_metric, metrics

        step_fn(params, state)

    @parameterized.expand([[1, 1], [2, 3]])
    def test_density_has_expected_attrs(self, min_width, min_spacing):
        ec = challenge.photon_extractor(
            minimum_width=min_width,
            minimum_spacing=min_spacing,
        )
        params = ec.component.init(jax.random.PRNGKey(0))

        self.assertEqual(params.lower_bound, 0.0)
        self.assertEqual(params.upper_bound, 1.0)
        self.assertSequenceEqual(params.periodic, (False, False))
        self.assertEqual(
            set(params.symmetries),
            {
                symmetry.REFLECTION_E_W,
                symmetry.REFLECTION_N_S,
                symmetry.REFLECTION_NE_SW,
                symmetry.REFLECTION_NW_SE,
            },
        )
        self.assertEqual(params.minimum_width, min_width)
        self.assertEqual(params.minimum_spacing, min_spacing)
        pad = (ec.component.spec.grid_shape[0] - 300) // 2
        expected_fixed_void = onp.pad(
            onp.zeros((300, 300), bool),
            ((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=True,
        )
        onp.testing.assert_array_equal(params.fixed_void, expected_fixed_void)
        onp.testing.assert_array_equal(
            params.fixed_solid, onp.zeros_like(expected_fixed_void)
        )


class BareSubstrateTest(unittest.TestCase):
    @pytest.mark.slow
    def test_bare_substrate_response_matches_expected(self):
        ec = challenge.photon_extractor()
        params = ec.component.init(jax.random.PRNGKey(0))
        params.array = jnp.zeros_like(params.array)

        bare_substrate_response, _ = ec.component.response(params)

        with self.subTest("collected power"):
            onp.testing.assert_allclose(
                bare_substrate_response.collected_power,
                bare_substrate_response.bare_substrate_collected_power,
                rtol=1e-2,
            )
        with self.subTest("emitted power"):
            onp.testing.assert_allclose(
                bare_substrate_response.emitted_power,
                bare_substrate_response.bare_substrate_emitted_power,
                rtol=1e-2,
            )
