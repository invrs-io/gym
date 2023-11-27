"""Tests for `diffract.splitter_challenge`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import optax
from fmmax import fmm
from parameterized import parameterized

from invrs_gym.challenges.diffract import splitter_challenge

LIGHTWEIGHT_SIM_PARAMS = dataclasses.replace(
    splitter_challenge.DIFFRACTIVE_SPLITTER_SIM_PARAMS,
    approximate_num_terms=100,
    formulation=fmm.Formulation.FFT,
)


class SplitterComponentTest(unittest.TestCase):
    def test_can_jit_response(self):
        mc = splitter_challenge.DiffractiveSplitterComponent(
            spec=splitter_challenge.DIFFRACTIVE_SPLITTER_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            thickness_initializer=lambda _, thickness: thickness,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))

        @jax.jit
        def jit_response_fn(params):
            return mc.response(params)

        jit_response_fn(params)

    def test_multiple_wavelengths(self):
        mc = splitter_challenge.DiffractiveSplitterComponent(
            spec=splitter_challenge.DIFFRACTIVE_SPLITTER_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            thickness_initializer=lambda _, thickness: thickness,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        response, aux = mc.response(params, wavelength=jnp.asarray([1.045, 1.055]))
        self.assertSequenceEqual(
            response.transmission_efficiency.shape,
            (2, mc.expansion.num_terms, 1),
        )


class SplitterChallengeTest(unittest.TestCase):
    @parameterized.expand([[lambda fn: fn], [jax.jit]])
    def test_optimize(self, step_fn_decorator):
        mc = splitter_challenge.diffractive_splitter(sim_params=LIGHTWEIGHT_SIM_PARAMS)

        def loss_fn(params):
            response, aux = mc.component.response(params)
            loss = mc.loss(response)
            return loss, (response, aux)

        opt = optax.adam(0.05)
        params = mc.component.init(jax.random.PRNGKey(0))
        state = opt.init(params)

        @step_fn_decorator
        def step_fn(params, state):
            (value, (response, aux)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            metrics = mc.metrics(response, params, aux)
            updates, state = opt.update(grad, state)
            params = optax.apply_updates(params, updates)
            return params, state, metrics

        step_fn(params, state)

    @parameterized.expand([[1, 1], [2, 3]])
    def test_density_has_expected_attrs(self, min_width, min_spacing):
        mc = splitter_challenge.diffractive_splitter(
            spec=splitter_challenge.DIFFRACTIVE_SPLITTER_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            thickness_initializer=lambda _, thickness: thickness,
            density_initializer=lambda _, seed_density: seed_density,
            minimum_spacing=min_spacing,
            minimum_width=min_width,
        )
        params = mc.component.init(jax.random.PRNGKey(0))

        self.assertEqual(set(params.keys()), {"density", "thickness"})

        self.assertEqual(params["density"].lower_bound, 0.0)
        self.assertEqual(params["density"].upper_bound, 1.0)
        self.assertSequenceEqual(params["density"].periodic, (True, True))
        self.assertSequenceEqual(params["density"].symmetries, ())
        self.assertEqual(params["density"].minimum_width, min_width)
        self.assertEqual(params["density"].minimum_spacing, min_spacing)
        self.assertIsNone(params["density"].fixed_solid)
        self.assertIsNone(params["density"].fixed_void)

        self.assertEqual(
            params["thickness"].array,
            splitter_challenge.DIFFRACTIVE_SPLITTER_SPEC.thickness_grating.array,
        )
        self.assertEqual(params["thickness"].lower_bound, 0.5)
        self.assertEqual(params["thickness"].upper_bound, 1.5)
