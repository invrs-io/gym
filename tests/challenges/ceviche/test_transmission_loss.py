"""Tests for `ceviche.transmission_loss`.

Copyright (c) 2023 Martin F. Schubert
"""

import functools
import unittest

from invrs_gym.challenges.ceviche import transmission_loss


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
