"""Tests for `metagrating.challenge`."""

import pathlib
import unittest

import jax
import optax
from parameterized import parameterized
from totypes import symmetry  # type: ignore[import,attr-defined,unused-ignore]

from invrs_gym.challenge.metagrating import challenge

DESIGNS_DIR = pathlib.Path(__file__).resolve().parent / "designs"


class MetagratingChallengeTest(unittest.TestCase):
    @parameterized.expand([[lambda fn: fn], [jax.jit]])
    def test_optimize(self, step_fn_decorator):
        mc = challenge.metagrating()

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
        mc = challenge.metagrating(
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
