"""Tests for `sorter.polarization_challenge`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import optax
from fmmax import fmm
from parameterized import parameterized

from invrs_gym.challenges.sorter import common, polarization_challenge

LIGHTWEIGHT_SIM_PARAMS = dataclasses.replace(
    polarization_challenge.POLARIZATION_SORTER_SIM_PARAMS,
    approximate_num_terms=100,
    formulation=fmm.Formulation.FFT,
)


class SorterChallengeTest(unittest.TestCase):
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

    @parameterized.expand([[0.4, 10], [0.36, 6]])
    def test_distance(self, efficiency_target, ratio_target):
        pc = polarization_challenge.polarization_sorter(
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            efficiency_target=efficiency_target,
            polarization_ratio_target=ratio_target,
        )

        dummy_ideal_response = common.SorterResponse(
            wavelength=1.0,
            polar_angle=0.0,
            azimuthal_angle=0.0,
            transmission=jnp.asarray(
                [
                    [0.5, 0.25, 0.25, 0.0],
                    [0.25, 0.5, 0.0, 0.25],
                    [0.25, 0.0, 0.5, 0.25],
                    [0.0, 0.25, 0.25, 0.5],
                ]
            ),
            reflection=jnp.asarray([0, 0, 0, 0]),
        )

        t1 = efficiency_target
        t2 = efficiency_target / ratio_target
        dummy_successful_response = common.SorterResponse(
            wavelength=1.0,
            polar_angle=0.0,
            azimuthal_angle=0.0,
            transmission=jnp.asarray(
                [
                    [t1, 0, 0, t2],
                    [0, t1, t2, 0],
                    [0, t2, t1, 0],
                    [t2, 0, 0, t1],
                ]
            ),
            reflection=jnp.asarray([0, 0, 0, 0]),
        )
        dummy_unsuccessful_response = common.SorterResponse(
            wavelength=1.0,
            polar_angle=0.0,
            azimuthal_angle=0.0,
            transmission=jnp.asarray(
                [
                    [t1 - 0.001, 0, 0, t2],
                    [0, t1, t2, 0],
                    [0, t2, t1, 0],
                    [t2 + 0.001, 0, 0, t1],
                ]
            ),
            reflection=jnp.asarray([0, 0, 0, 0]),
        )

        self.assertGreater(pc.distance_to_target(dummy_unsuccessful_response), 0)
        self.assertEqual(pc.distance_to_target(dummy_successful_response), 0)
        self.assertLess(pc.distance_to_target(dummy_ideal_response), 0)
