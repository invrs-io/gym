"""Tests for `bayer.challenge.bayer_sorter` challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import optax
from fmmax import fmm
from parameterized import parameterized

from invrs_gym.challenges.bayer import challenge

LIGHTWEIGHT_SIM_PARAMS = dataclasses.replace(
    challenge.BAYER_SIM_PARAMS,
    wavelength=jnp.asarray([0.45, 0.55, 0.65]),
    polar_angle=0.0,
    azimuthal_angle=0.0,
    formulation=fmm.Formulation.FFT,
    approximate_num_terms=100,
)


class BayerChallengeTest(unittest.TestCase):
    @parameterized.expand([[lambda fn: fn], [jax.jit]])
    def test_optimize(self, step_fn_decorator):
        cc = challenge.bayer_sorter(sim_params=LIGHTWEIGHT_SIM_PARAMS)

        def loss_fn(params):
            response, aux = cc.component.response(params)
            loss = cc.loss(response)
            return loss, (response, aux)

        opt = optax.adam(0.05)
        params = cc.component.init(jax.random.PRNGKey(0))
        state = opt.init(params)

        @step_fn_decorator
        def step_fn(params, state):
            (value, (response, aux)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            eval_metric = cc.eval_metric(response)
            metrics = cc.metrics(response, params, aux)
            updates, state = opt.update(grad, state)
            params = optax.apply_updates(params, updates)
            return params, state, eval_metric, metrics

        step_fn(params, state)

    def test_jit_with_random_wavelengths(self):
        cc = challenge.bayer_sorter(sim_params=LIGHTWEIGHT_SIM_PARAMS)

        opt = optax.adam(0.05)
        params = cc.component.init(jax.random.PRNGKey(0))
        state = opt.init(params)

        @jax.jit
        def step_fn(key, params, state):
            wavelength = jax.random.uniform(key, (3,)) * 0.5 + 0.5

            def loss_fn(params):
                response, aux = cc.component.response(params, wavelength=wavelength)
                loss = cc.loss(response)
                return loss, (response, aux)

            (value, (response, aux)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            eval_metric = cc.eval_metric(response)
            metrics = cc.metrics(response, params, aux)
            updates, state = opt.update(grad, state)
            params = optax.apply_updates(params, updates)
            return params, state, eval_metric, metrics

        step_fn(jax.random.PRNGKey(0), params, state)
