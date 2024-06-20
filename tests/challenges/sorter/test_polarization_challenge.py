"""Tests for `sorter.polarization_challenge`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
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

        t1 = efficiency_target
        t2 = efficiency_target / ratio_target

        # Successful response where the targets are exactly met.
        dummy_successful_response_0 = common.SorterResponse(
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
        # Successful response where efficiency exceeds the goal slightly,
        # and the ratio exactly matches the target.
        dummy_successful_response_1 = common.SorterResponse(
            wavelength=1.0,
            polar_angle=0.0,
            azimuthal_angle=0.0,
            transmission=jnp.asarray(
                [
                    [t1 + 0.1, 0, 0, (t1 + 0.1) / ratio_target],
                    [0, t1, t2, 0],
                    [0, t2, t1, 0],
                    [t2, 0, 0, t1],
                ]
            ),
            reflection=jnp.asarray([0, 0, 0, 0]),
        )

        # Unsuccessful response where efficiency is too low for one pixel.
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

        self.assertEqual(pc._distance_to_target(dummy_successful_response_0), 0)
        self.assertEqual(pc._distance_to_target(dummy_successful_response_1), 0)
        self.assertGreater(pc._distance_to_target(dummy_unsuccessful_response), 0)

    def test_on_target_transmission(self):
        dummy_response = common.SorterResponse(
            wavelength=1.0,
            polar_angle=0.0,
            azimuthal_angle=0.0,
            transmission=jnp.asarray(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ]
            ),
            reflection=jnp.asarray([0, 0, 0, 0]),
        )
        onp.testing.assert_array_equal(
            polarization_challenge._on_target_transmission(dummy_response),
            onp.asarray([0, 5, 10, 15]),
        )

    def test_off_target_transmission(self):
        dummy_response = common.SorterResponse(
            wavelength=1.0,
            polar_angle=0.0,
            azimuthal_angle=0.0,
            transmission=jnp.asarray(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ]
            ),
            reflection=jnp.asarray([0, 0, 0, 0]),
        )
        onp.testing.assert_array_equal(
            polarization_challenge._off_target_transmission(dummy_response),
            onp.asarray([3, 6, 9, 12]),
        )
