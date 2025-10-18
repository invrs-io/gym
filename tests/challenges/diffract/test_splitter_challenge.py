"""Tests for `diffract.splitter_challenge`.

Copyright (c) 2025 invrs.io LLC
"""

import dataclasses
import unittest

import fmmax
import jax
import optax
from parameterized import parameterized

from invrs_gym.challenges.diffract import splitter_challenge

LIGHTWEIGHT_SIM_PARAMS = dataclasses.replace(
    splitter_challenge.DIFFRACTIVE_SPLITTER_SIM_PARAMS,
    approximate_num_terms=100,
    formulation=fmmax.Formulation.FFT,
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
            eval_metric = mc.eval_metric(response)
            metrics = mc.metrics(response, params, aux)
            updates, state = opt.update(grad, state)
            params = optax.apply_updates(params, updates)
            return params, state, eval_metric, metrics

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

        self.assertEqual(
            set(params.keys()),
            {"density", "thickness_grating", "thickness_spacer", "thickness_cap"},
        )

        self.assertEqual(params["density"].lower_bound, 0.0)
        self.assertEqual(params["density"].upper_bound, 1.0)
        self.assertSequenceEqual(params["density"].periodic, (True, True))
        self.assertSequenceEqual(params["density"].symmetries, ())
        self.assertEqual(params["density"].minimum_width, min_width)
        self.assertEqual(params["density"].minimum_spacing, min_spacing)
        self.assertIsNone(params["density"].fixed_solid)
        self.assertIsNone(params["density"].fixed_void)

        self.assertEqual(
            params["thickness_grating"].array,
            splitter_challenge.DIFFRACTIVE_SPLITTER_SPEC.thickness_grating.array,
        )
        self.assertEqual(params["thickness_grating"].lower_bound, 0.5)
        self.assertEqual(params["thickness_grating"].upper_bound, 1.5)

    def test_default_grid_shape(self):
        mc = splitter_challenge.diffractive_splitter()
        params = mc.component.init(jax.random.PRNGKey(0))
        self.assertSequenceEqual(params["density"].shape, (180, 180))
