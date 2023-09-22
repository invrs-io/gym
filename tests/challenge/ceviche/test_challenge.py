"""Tests for `invrs_gym.challenge.ceviche.challenge`."""

import unittest

import jax
import jax.numpy as jnp
import optax
from parameterized import parameterized

from invrs_gym.challenge.ceviche import challenge


class CreateChallengesTest(unittest.TestCase):
    @parameterized.expand(
        [
            [challenge.lightweight_beam_splitter_challenge],
            [challenge.lightweight_mode_converter_challenge],
            [challenge.lightweight_waveguide_bend_challenge],
            [challenge.lightweight_wdm_challenge],
        ]
    )
    def test_optimize(self, ceviche_challenge):
        c = ceviche_challenge()
        params = c.component.init(jax.random.PRNGKey(0))

        def loss_fn(params):
            response, aux = c.component.response(params)
            loss = c.loss(response)
            return loss, (response, aux)

        values = []
        opt = optax.adam(learning_rate=0.01)
        state = opt.init(params)
        for _ in range(3):
            (value, (response, aux)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            values.append(value)
            updates, state = opt.update(grad, state)
            params = optax.apply_updates(params, updates)

        # Assert that loss decreases.
        self.assertLess(values[1], values[0])
        self.assertLess(values[2], values[1])

        self.assertEqual(set(aux.keys()), {"sparams", "fields"})

        num_ports = len(c.component.ceviche_model.ports)
        self.assertSequenceEqual(aux["sparams"].shape, (2, 1, num_ports))

        density = c.component.ceviche_model.density(params.array)
        self.assertSequenceEqual(aux["fields"].shape, (2, 1) + density.shape)

        # Check that the gradient is nonzero.
        self.assertGreater(jnp.sum(jnp.abs(grad.array)), 0.0)

        # Check that metrics can be computed.
        metrics = c.metrics(response, params=params, aux=aux)
        self.assertEqual(set(metrics.keys()), {"distance_to_window"})

    @parameterized.expand(
        [
            [challenge.beam_splitter_challenge],
            [challenge.lightweight_beam_splitter_challenge],
            [challenge.mode_converter_challenge],
            [challenge.lightweight_mode_converter_challenge],
            [challenge.waveguide_bend_challenge],
            [challenge.lightweight_waveguide_bend_challenge],
            [challenge.wdm_challenge],
            [challenge.lightweight_wdm_challenge],
        ]
    )
    def test_with_dummy_response(self, ceviche_challenge):
        c = ceviche_challenge()
        _ = c.component.init(jax.random.PRNGKey(0))

        num_ports = len(c.component.ceviche_model.ports)
        num_wavelengths = len(c.component.ceviche_model.params.wavelengths)
        dummy_response = jnp.zeros((num_wavelengths, 1, num_ports))

        _ = c.loss(dummy_response)
