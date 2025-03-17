"""Tests for `metalens.challenge`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import fmmax
import jax
import optax
from parameterized import parameterized
from totypes import symmetry

from invrs_gym.challenges.metalens import challenge


class MetalensChallengeTest(unittest.TestCase):
    @parameterized.expand([[lambda fn: fn], [jax.jit]])
    def test_optimize(self, step_fn_decorator):
        mc = challenge.metalens(
            sim_params=dataclasses.replace(
                challenge.METALENS_SIM_PARAMS,
                approximate_num_terms=100,
                formulation=fmmax.Formulation.FFT,
                num_layers=1,
            )
        )

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
        mc = challenge.metalens(
            minimum_width=min_width,
            minimum_spacing=min_spacing,
        )
        params = mc.component.init(jax.random.PRNGKey(0))

        self.assertEqual(params.lower_bound, 0.0)
        self.assertEqual(params.upper_bound, 1.0)
        self.assertSequenceEqual(params.periodic, (False, False))
        self.assertEqual(
            set(params.symmetries),
            {
                symmetry.REFLECTION_N_S,
            },
        )
        self.assertEqual(params.minimum_width, min_width)
        self.assertEqual(params.minimum_spacing, min_spacing)
