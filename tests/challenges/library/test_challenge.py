"""Tests for the meta-atom library challenge.

Copyright (c) 2024 The INVRS-IO authors.
"""

import dataclasses
import unittest

import fmmax
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from parameterized import parameterized
from totypes import types

from invrs_gym.challenges.library import challenge


class DensityInitializerTest(unittest.TestCase):
    @parameterized.expand(
        [
            [0.0, (0.5, 0.5, 0.5, 0.5)],
            [1.0, (0.125, 0.375, 0.625, 0.875)],
        ]
    )
    def test_mean_matches_expected(self, relative_mean_range, expected):
        seed_density = types.Density2DArray(
            array=jnp.zeros((4, 10, 10)), lower_bound=0.0, upper_bound=1.0
        )
        density = challenge.library_density_initializer(
            key=jax.random.PRNGKey(0),
            seed_density=seed_density,
            relative_mean_range=relative_mean_range,
            relative_noise_amplitude=0.0,
        )
        expected = onp.asarray(expected)
        expected = onp.broadcast_to(expected[:, onp.newaxis, onp.newaxis], (4, 10, 10))
        onp.testing.assert_allclose(density.array, expected)


LIGHTWEIGHT_SIM_PARAMS = dataclasses.replace(
    challenge.LIBRARY_SIM_PARAMS,
    approximate_num_terms=100,
    formulation=fmmax.Formulation.FFT,
)


class MetagratingChallengeTest(unittest.TestCase):
    @parameterized.expand([[lambda fn: fn], [jax.jit]])
    def test_optimize(self, step_fn_decorator):
        mc = challenge.meta_atom_library(sim_params=LIGHTWEIGHT_SIM_PARAMS)

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

    def test_multiple_wavelengths(self):
        mc = challenge.meta_atom_library(sim_params=LIGHTWEIGHT_SIM_PARAMS)
        params = mc.component.init(jax.random.PRNGKey(0))
        response, _ = mc.component.response(
            params, wavelength=jnp.asarray([[0.88, 0.89]])
        )
        loss = mc.loss(response)
        self.assertSequenceEqual(loss.shape, ())

    @parameterized.expand([[1, 1], [2, 3]])
    def test_density_has_expected_attrs(self, min_width, min_spacing):
        mc = challenge.meta_atom_library(
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            minimum_spacing=min_spacing,
            minimum_width=min_width,
        )
        params = mc.component.init(jax.random.PRNGKey(0))
        density = params["density"]

        self.assertEqual(density.lower_bound, 0.0)
        self.assertEqual(density.upper_bound, 1.0)
        self.assertSequenceEqual(density.periodic, (False, False))
        self.assertSequenceEqual(density.symmetries, ("reflection_n_s",))
        self.assertEqual(density.minimum_width, min_width)
        self.assertEqual(density.minimum_spacing, min_spacing)
        self.assertIsNone(density.fixed_solid)
        self.assertIsNotNone(density.fixed_void)

    def test_default_density_shape(self):
        mc = challenge.meta_atom_library()
        params = mc.component.init(jax.random.PRNGKey(0))
        self.assertSequenceEqual(params["density"].shape, (8, 80, 80))
