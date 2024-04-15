"""Tests that challenge-related quantities are serializable.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized
from totypes import json_utils

from invrs_gym import challenges

# Ignore expensive challenges which have lightweight counterparts for which the testing
# below suffices to establish serializability of response, metrics, etc.
IGNORED_CHALLENGES = (
    "ceviche_beam_splitter",
    "ceviche_mode_converter",
    "ceviche_waveguide_bend",
    "ceviche_wdm",
)
CHALLENGE_NAMES = [
    name for name in challenges.BY_NAME.keys() if name not in IGNORED_CHALLENGES
]


class TestSerialize(unittest.TestCase):
    @parameterized.expand(CHALLENGE_NAMES)
    def test_can_serialize(self, challenge_name):
        challenge = challenges.BY_NAME[challenge_name]()
        params = challenge.component.init(jax.random.PRNGKey(0))
        response, aux = challenge.component.response(params)
        loss = challenge.loss(response)
        metrics = challenge.metrics(response, params, aux)
        distance = challenge.distance_to_target(response)
        original = {
            "params": params,
            "loss": loss,
            "distance": distance,
            "metrics": metrics,
            "aux": aux,
            "response": response,
        }
        serialized = json_utils.json_from_pytree(original)
        restored = json_utils.pytree_from_json(serialized)
        for key in original.keys():
            if isinstance(original[key], jnp.ndarray):
                self.assertIsInstance(restored[key], onp.ndarray)
            else:
                self.assertEqual(type(original[key]), type(restored[key]))

    @parameterized.expand(CHALLENGE_NAMES)
    def test_can_serialize_batch(self, challenge_name):
        challenge = challenges.BY_NAME[challenge_name]()

        keys = jax.random.split(jax.random.PRNGKey(0))

        params = jax.vmap(challenge.component.init)(keys)
        response, aux = jax.vmap(challenge.component.response)(params)
        loss = jax.vmap(challenge.loss)(response)
        metrics = jax.vmap(challenge.metrics)(response, params, aux)
        distance = jax.vmap(challenge.distance_to_target)(response)
        original = {
            "params": params,
            "loss": loss,
            "distance": distance,
            "metrics": metrics,
            "aux": aux,
            "response": response,
        }
        serialized = json_utils.json_from_pytree(original)
        restored = json_utils.pytree_from_json(serialized)
        for key in original.keys():
            if isinstance(original[key], jnp.ndarray):
                self.assertIsInstance(restored[key], onp.ndarray)
            else:
                self.assertEqual(type(original[key]), type(restored[key]))
