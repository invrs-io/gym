"""Tests for `diffract.metagrating_challenge`."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import optax
from fmmax import fmm
from parameterized import parameterized
from totypes import symmetry  # type: ignore[import,attr-defined,unused-ignore]

from invrs_gym.challenges.diffract import metagrating_challenge

LIGHTWEIGHT_SIM_PARAMS = dataclasses.replace(
    metagrating_challenge.METAGRATING_SIM_PARAMS,
    approximate_num_terms=100,
    formulation=fmm.Formulation.FFT,
)


class MetagratingComponentTest(unittest.TestCase):
    def test_density_has_expected_properties(self):
        mc = metagrating_challenge.MetagratingComponent(
            spec=metagrating_challenge.METAGRATING_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        self.assertEqual(params.lower_bound, 0.0)
        self.assertEqual(params.upper_bound, 1.0)
        self.assertSequenceEqual(params.periodic, (True, True))

    def test_can_jit_response(self):
        mc = metagrating_challenge.MetagratingComponent(
            spec=metagrating_challenge.METAGRATING_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))

        @jax.jit
        def jit_response_fn(params):
            return mc.response(params)

        jit_response_fn(params)

    def test_multiple_wavelengths(self):
        mc = metagrating_challenge.MetagratingComponent(
            spec=metagrating_challenge.METAGRATING_SPEC,
            sim_params=LIGHTWEIGHT_SIM_PARAMS,
            density_initializer=lambda _, seed_density: seed_density,
        )
        params = mc.init(jax.random.PRNGKey(0))
        response, aux = mc.response(params, wavelength=jnp.asarray([1.045, 1.055]))
        self.assertSequenceEqual(
            response.transmission_efficiency.shape,
            (2, mc.expansion.num_terms, 1),
        )


class MetagratingChallengeTest(unittest.TestCase):
    @parameterized.expand([[lambda fn: fn], [jax.jit]])
    def test_optimize(self, step_fn_decorator):
        mc = metagrating_challenge.metagrating(sim_params=LIGHTWEIGHT_SIM_PARAMS)

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
        mc = metagrating_challenge.metagrating(
            minimum_width=min_width,
            minimum_spacing=min_spacing,
        )
        params = mc.component.init(jax.random.PRNGKey(0))
        self.assertEqual(params.lower_bound, 0.0)
        self.assertEqual(params.upper_bound, 1.0)
        self.assertSequenceEqual(params.periodic, (True, True))
        self.assertSequenceEqual(params.symmetries, (symmetry.REFLECTION_E_W,))
        self.assertEqual(params.minimum_width, min_width)
        self.assertEqual(params.minimum_spacing, min_spacing)
        self.assertIsNone(params.fixed_solid)
        self.assertIsNone(params.fixed_void)
