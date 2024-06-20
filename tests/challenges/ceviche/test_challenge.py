"""Tests for `ceviche.challenge`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax
import jax.numpy as jnp
import optax
from parameterized import parameterized

from invrs_gym.challenges.ceviche import challenge


class CevicheChallengesTest(unittest.TestCase):
    @parameterized.expand(
        [
            [challenge.lightweight_beam_splitter],
            [challenge.lightweight_mode_converter],
            [challenge.lightweight_waveguide_bend],
            [challenge.lightweight_wdm],
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

    @parameterized.expand(
        [
            [challenge.beam_splitter],
            [challenge.lightweight_beam_splitter],
            [challenge.mode_converter],
            [challenge.lightweight_mode_converter],
            [challenge.waveguide_bend],
            [challenge.lightweight_waveguide_bend],
            [challenge.wdm],
            [challenge.lightweight_wdm],
        ]
    )
    def test_with_dummy_response(self, ceviche_challenge):
        c = ceviche_challenge()
        dummy_params = c.component.init(jax.random.PRNGKey(0))

        num_ports = len(c.component.ceviche_model.ports)
        num_wavelengths = len(c.component.ceviche_model.params.wavelengths)
        dummy_response = challenge.CevicheResponse(
            s_parameters=jnp.zeros((num_wavelengths, 1, num_ports)),
            wavelengths_nm=jnp.arange(num_wavelengths),
            excite_port_idxs=jnp.asarray([0]),
        )
        loss = c.loss(dummy_response)
        metrics = c.metrics(dummy_response, params=dummy_params, aux={})
        distance = metrics["distance_to_target"]
        self.assertSequenceEqual(loss.shape, ())
        self.assertSequenceEqual(distance.shape, ())

    def test_can_jit(self):
        c = challenge.lightweight_waveguide_bend()
        params = c.component.init(jax.random.PRNGKey(0))
        jax.jit(c.component.response)(params)
