"""Tests for `utils.optimizer`"""

import unittest

import optax

from invrs_gym import challenge
from invrs_gym.utils import optimizer


class OptimizerTest(unittest.TestCase):
    def test_can_optimize(self):
        params, state, step_fn = optimizer.setup_optimization(
            challenge=challenge.metagrating(),
            optimizer=optax.adam(0.02),
        )

        for _ in range(2):
            params, state, _ = step_fn(params, state)