"""Tests for the meta-atom library challenge related to nanostructure rotation.

Copyright (c) 2025 invrs.io LLC
"""

import copy
import pathlib
import unittest

import jax.numpy as jnp
import numpy as onp
from totypes import json_utils

from invrs_gym.challenges.library import challenge as library_challenge

REPO_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent
DESIGNS_DIR = REPO_PATH / "reference_designs/meta_atom_library"


class RotationTest(unittest.TestCase):
    def test_rotated_response_matches_expected(self):
        challenge = library_challenge.meta_atom_library()

        # Compute response of rotated nanostructures by modifying the response of
        # nanostructures in their original orientation.
        with open(DESIGNS_DIR / "library1.json") as f:
            serialized = f.read()
        params = json_utils.pytree_from_json(serialized)
        response, _ = challenge.component.response(params)
        rotated_response = library_challenge._rotate_response(
            response, jnp.ones((8,), dtype=bool)
        )

        # Compute response of rotated nanostructures by directly simulating the
        # rotated nanostructures.
        rotated_params = copy.deepcopy(params)
        rotated_params["density"].array = jnp.stack(
            [jnp.rot90(x) for x in rotated_params["density"].array],
            axis=0,
        )
        expected_rotated_response, _ = challenge.component.response(rotated_params)

        with self.subTest("transmission_rhcp"):
            onp.testing.assert_allclose(
                rotated_response.transmission_rhcp,
                expected_rotated_response.transmission_rhcp,
                rtol=0.01,
                atol=0.001,
            )
        with self.subTest("transmission_lhcp"):
            onp.testing.assert_allclose(
                rotated_response.transmission_lhcp,
                expected_rotated_response.transmission_lhcp,
                rtol=0.01,
                atol=0.001,
            )
        with self.subTest("reflection_rhcp"):
            onp.testing.assert_allclose(
                rotated_response.reflection_rhcp,
                expected_rotated_response.reflection_rhcp,
                rtol=0.01,
                atol=0.001,
            )
        with self.subTest("reflection_lhcp"):
            onp.testing.assert_allclose(
                rotated_response.reflection_lhcp,
                expected_rotated_response.reflection_lhcp,
                rtol=0.01,
                atol=0.001,
            )

    def test_optimal_rotation(self):
        challenge = library_challenge.meta_atom_library()

        # Compute response of rotated nanostructures by modifying the response of
        # nanostructures in their original orientation.
        with open(DESIGNS_DIR / "library1.json") as f:
            serialized = f.read()
        params = json_utils.pytree_from_json(serialized)
        response, _ = challenge.component.response(params)

        optimal_rotation_idx = library_challenge.optimal_rotation(
            response, challenge.component.spec
        )
        self.assertEqual(optimal_rotation_idx, 0)

        # Apply a random rotation, and check that the optimal rotation is the one
        # that undoes the random rotation.
        is_rotated = jnp.asarray([0, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
        rotated_response = library_challenge._rotate_response(response, is_rotated)

        optimal_rotation_idx_for_rotated_response = library_challenge.optimal_rotation(
            rotated_response, challenge.component.spec
        )

        expected_rotation_idx = library_challenge.rotation_idx_from_is_rotated(
            ~is_rotated
        )
        self.assertNotEqual(optimal_rotation_idx_for_rotated_response, 0.0)
        self.assertEqual(
            optimal_rotation_idx_for_rotated_response,
            expected_rotation_idx,
        )

    def test_rotation_for_idx(self):
        num_nanostructures = 8

        def expected_fn(idx):
            # Reference implementation of `_rotation_for_idx`.
            is_rotated = [
                int(j) for j in onp.binary_repr(idx, width=num_nanostructures)
            ]
            return onp.asarray(is_rotated).astype(bool)[::-1]

        for i in range(128):
            expected = expected_fn(idx=i)
            result = library_challenge._rotation_for_idx(
                idx=i, num_nanostructures=num_nanostructures
            )
            onp.testing.assert_array_equal(result, expected)
