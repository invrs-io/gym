"""Tests for the meta-atom library challenge related to nanostructure rotation.

Copyright (c) 2024 The INVRS-IO authors.
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
    def test_transformed_response_matches_expected(self):
        # Compare post-process rotation of response to actual simulation of rotatated
        # meta-atoms.
        challenge = library_challenge.meta_atom_library()

        # Compute response of rotated nanostructures by modifying the response of
        # nanostructures in their original orientation.
        with open(DESIGNS_DIR / "library1.json") as f:
            serialized = f.read()
        params = json_utils.pytree_from_json(serialized)
        response, _ = challenge.component.response(params)
        rotated_response = library_challenge.reordered_rotated_response(
            response,
            ordering=jnp.arange(8, dtype=int),
            rotation=jnp.ones((8,), dtype=bool),
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

        (
            reference_optimal_ordering,
            reference_optimal_rotation,
        ) = library_challenge.optimal_ordering_and_rotation(
            response, challenge.component.spec
        )
        onp.testing.assert_array_equal(reference_optimal_ordering, jnp.arange(8))
        onp.testing.assert_array_equal(
            reference_optimal_rotation, jnp.zeros((8,), dtype=bool)
        )

        # Apply a random rotation, and check that the optimal rotation is the one
        # that undoes the random rotation.
        random_rotation = jnp.asarray([0, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
        rotated_response = library_challenge.reordered_rotated_response(
            response,
            ordering=jnp.arange(8),
            rotation=random_rotation,
        )

        (
            optimal_ordering,
            optimal_rotation,
        ) = library_challenge.optimal_ordering_and_rotation(
            rotated_response, challenge.component.spec
        )
        onp.testing.assert_array_equal(optimal_ordering, jnp.arange(8))
        onp.testing.assert_array_equal(optimal_rotation, random_rotation)

    def test_optimal_ordering(self):
        challenge = library_challenge.meta_atom_library()

        # Compute response of rotated nanostructures by modifying the response of
        # nanostructures in their original orientation.
        with open(DESIGNS_DIR / "library1.json") as f:
            serialized = f.read()
        params = json_utils.pytree_from_json(serialized)
        response, _ = challenge.component.response(params)

        (
            reference_optimal_ordering,
            reference_optimal_rotation,
        ) = library_challenge.optimal_ordering_and_rotation(
            response, challenge.component.spec
        )
        onp.testing.assert_array_equal(reference_optimal_ordering, jnp.arange(8))
        onp.testing.assert_array_equal(
            reference_optimal_rotation, jnp.zeros((8,), dtype=bool)
        )

        # Apply a random ordering, and check that the optimal ordering is the one
        # that undoes the random ordering. Keep the first meta-atom fixed, since the
        # optimal ordering also keeps the first elemetn fixed.
        random_ordering = jnp.asarray([0, 5, 3, 1, 2, 7, 4, 6])
        rotated_response = library_challenge.reordered_rotated_response(
            response,
            ordering=random_ordering,
            rotation=jnp.zeros((8,), dtype=bool),
        )

        (
            optimal_ordering,
            optimal_rotation,
        ) = library_challenge.optimal_ordering_and_rotation(
            rotated_response, challenge.component.spec
        )
        onp.testing.assert_array_equal(
            optimal_ordering,
            jnp.asarray([0, 3, 4, 2, 6, 1, 7, 5]),
        )
        onp.testing.assert_array_equal(
            optimal_rotation,
            jnp.zeros((8,), dtype=bool),
        )

    def test_optimal_ordering_and_rotation(self):
        challenge = library_challenge.meta_atom_library()

        # Compute response of rotated nanostructures by modifying the response of
        # nanostructures in their original orientation.
        with open(DESIGNS_DIR / "library1.json") as f:
            serialized = f.read()
        params = json_utils.pytree_from_json(serialized)
        response, _ = challenge.component.response(params)

        (
            reference_optimal_ordering,
            reference_optimal_rotation,
        ) = library_challenge.optimal_ordering_and_rotation(
            response, challenge.component.spec
        )
        onp.testing.assert_array_equal(reference_optimal_ordering, jnp.arange(8))
        onp.testing.assert_array_equal(
            reference_optimal_rotation, jnp.zeros((8,), dtype=bool)
        )

        # Apply a random ordering, and check that the optimal ordering is the one
        # that undoes the random ordering. Keep the first meta-atom fixed, since the
        # optimal ordering also keeps the first elemetn fixed.
        random_ordering = jnp.asarray([0, 5, 3, 1, 2, 7, 4, 6])
        random_rotation = jnp.asarray([0, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
        rotated_response = library_challenge.reordered_rotated_response(
            response,
            ordering=random_ordering,
            rotation=random_rotation,
        )

        (
            optimal_ordering,
            optimal_rotation,
        ) = library_challenge.optimal_ordering_and_rotation(
            rotated_response, challenge.component.spec
        )
        onp.testing.assert_array_equal(
            optimal_ordering,
            jnp.asarray([0, 3, 4, 2, 6, 1, 7, 5]),
        )
        onp.testing.assert_array_equal(
            optimal_rotation,
            random_rotation[random_ordering],
        )
