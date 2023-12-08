"""Tests for `ceviche.transmission_loss`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import functools
import unittest

import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from invrs_gym.loss import transmission_loss


class OrthotopeSmoothLossTest(unittest.TestCase):
    def test_loss_with_target_inside_space(self):
        loss_fn = functools.partial(
            transmission_loss.orthotope_smooth_transmission_loss,
            window_lower_bound=0.4,
            window_upper_bound=0.6,
            transmission_exponent=0.5,
            scalar_exponent=2.0,
        )

        self.assertGreater(loss_fn(0.0), loss_fn(0.2))
        self.assertGreater(loss_fn(0.2), loss_fn(0.3))
        self.assertGreater(loss_fn(0.3), loss_fn(0.4))
        self.assertGreater(loss_fn(0.4), loss_fn(0.5))

        self.assertLess(loss_fn(0.5), loss_fn(0.6))
        self.assertLess(loss_fn(0.6), loss_fn(0.7))
        self.assertLess(loss_fn(0.7), loss_fn(0.8))
        self.assertLess(loss_fn(0.8), loss_fn(1.0))

    def test_loss_with_target_bordering_space(self):
        loss_fn = functools.partial(
            transmission_loss.orthotope_smooth_transmission_loss,
            window_lower_bound=0.8,
            window_upper_bound=1.0,
            transmission_exponent=0.5,
            scalar_exponent=2.0,
        )

        self.assertGreater(loss_fn(0.0), loss_fn(0.5))
        self.assertGreater(loss_fn(0.5), loss_fn(0.7))
        self.assertGreater(loss_fn(0.7), loss_fn(0.8))
        self.assertGreater(loss_fn(0.8), loss_fn(0.9))
        self.assertGreater(loss_fn(0.9), loss_fn(1.0))

    @parameterized.expand(
        [
            [(1,), None, ()],
            [(1,), 0, ()],
            [(6, 1, 3), None, ()],
            [(6, 1, 3), (1, 2), (6,)],
        ]
    )
    def test_loss_has_expected_shape(self, shape, axis, expected_shape):
        transmission = jnp.arange(jnp.prod(jnp.asarray(shape))).reshape(shape)
        loss = transmission_loss.orthotope_smooth_transmission_loss(
            transmission,
            window_lower_bound=jnp.zeros_like(transmission),
            window_upper_bound=jnp.ones_like(transmission),
            transmission_exponent=1,
            scalar_exponent=1,
            axis=axis,
        )
        self.assertSequenceEqual(loss.shape, expected_shape)


class DistanceToWindowTest(unittest.TestCase):
    def test_loss_with_target_inside_space(self):
        loss_fn = functools.partial(
            transmission_loss.distance_to_window,
            window_lower_bound=0.4,
            window_upper_bound=0.6,
        )

        self.assertGreater(loss_fn(0.0), loss_fn(0.2))
        self.assertGreater(loss_fn(0.2), loss_fn(0.3))
        self.assertGreater(loss_fn(0.3), loss_fn(0.4))
        self.assertEqual(loss_fn(0.4), 0.0)
        self.assertEqual(loss_fn(0.5), 0.0)
        self.assertEqual(loss_fn(0.6), 0.0)
        self.assertLess(loss_fn(0.6), loss_fn(0.7))
        self.assertLess(loss_fn(0.7), loss_fn(0.8))
        self.assertLess(loss_fn(0.8), loss_fn(1.0))


class L2NormTest(unittest.TestCase):
    def test_norm_axis_none(self):
        a = jnp.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
        result = transmission_loss._l2_norm(a, axis=None)
        expected = jnp.linalg.norm(result)
        onp.testing.assert_array_equal(result, expected)

    def test_norm_axis_tuple(self):
        a = jnp.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
        result = transmission_loss._l2_norm(a, axis=(0, 2, 3))
        expected = jnp.stack(
            [
                jnp.linalg.norm(a[:, 0, ...]),
                jnp.linalg.norm(a[:, 1, ...]),
                jnp.linalg.norm(a[:, 2, ...]),
            ]
        )
        onp.testing.assert_array_equal(result, expected)

    def test_norm_axis_int(self):
        a = jnp.arange(2 * 3 * 4 * 5).reshape((6, 20))
        result = transmission_loss._l2_norm(a, axis=1)
        expected = jnp.stack(
            [
                jnp.linalg.norm(a[0, :]),
                jnp.linalg.norm(a[1, :]),
                jnp.linalg.norm(a[2, :]),
                jnp.linalg.norm(a[3, :]),
                jnp.linalg.norm(a[4, :]),
                jnp.linalg.norm(a[5, :]),
            ]
        )
        onp.testing.assert_array_equal(result, expected)
