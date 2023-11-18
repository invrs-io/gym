"""Tests for `sorter.polarization_challenge`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import jax
import optax
from fmmax import fmm
from parameterized import parameterized

from invrs_gym.challenges.sorter import polarization_challenge

LIGHTWEIGHT_SIM_PARAMS = dataclasses.replace(
    polarization_challenge.POLARIZATION_SORTER_SIM_PARAMS,
    approximate_num_terms=100,
    formulation=fmm.Formulation.FFT,
)


class SplitterChallengeTest(unittest.TestCase):
    @parameterized.expand([[lambda fn: fn], [jax.jit]])
    def test_optimize(self, step_fn_decorator):
        pc = polarization_challenge.polarization_sorter(
            sim_params=LIGHTWEIGHT_SIM_PARAMS
        )

        def loss_fn(params):
            response, aux = pc.component.response(params)
            loss = pc.loss(response)
            return loss, (response, aux)

        opt = optax.adam(0.05)
        params = pc.component.init(jax.random.PRNGKey(0))
        state = opt.init(params)

        @step_fn_decorator
        def step_fn(params, state):
            (value, (response, aux)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            metrics = pc.metrics(response, params, aux)
            updates, state = opt.update(grad, state)
            params = optax.apply_updates(params, updates)
            return params, state, metrics

        step_fn(params, state)
