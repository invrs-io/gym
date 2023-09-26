"""Tests that simulations of reference devices give expected results."""

import dataclasses
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from invrs_gym.challenge.metagrating import component

DESIGNS_DIR = pathlib.Path(__file__).resolve().parent / "designs"


class ReferenceDeviceTest(unittest.TestCase):
    @parameterized.expand(
        [
            # device name, expected, tolerance
            ["device1.csv", 0.957, 0.005],  # Reticolo 0.957, Meep 0.955
            ["device2.csv", 0.933, 0.005],  # Reticolo 0.933, Meep 0.938
            ["device3.csv", 0.966, 0.010],  # Reticolo 0.966, Meep 0.950
            ["device4.csv", 0.933, 0.005],  # Reticolo 0.933, Meep 0.925
            ["device5.csv", 0.841, 0.015],  # Reticolo 0.841, Meep 0.843
        ]
    )
    def test_efficiency_matches_expected(self, fname, expected_efficiency, tol):
        # Compares efficiencies against those reported at.
        # https://github.com/NanoComp/photonics-opt-testbed/tree/main/Metagrating3D

        path = DESIGNS_DIR / fname
        density_array = onp.genfromtxt(path, delimiter=",")

        if density_array.ndim == 1:
            density_array = jnp.broadcast_to(density_array[:, jnp.newaxis], (119, 45))

        mc = component.MetagratingComponent(
            spec=component.MetagratingSpec(),
            sim_params=component.MetagratingSimParams(
                grid_shape=density_array.shape,
            ),
            density_initializer=lambda _, seed_density: seed_density,
        )

        density = mc.init(jax.random.PRNGKey(0))
        density = dataclasses.replace(density, array=density_array)
        response, _ = mc.response(density)

        output_order = (1, 0)
        ((order_idx,),) = onp.where(
            onp.all(response.expansion.basis_coefficients == (output_order), axis=1)
        )
        self.assertSequenceEqual(
            tuple(response.expansion.basis_coefficients[order_idx, :]), output_order
        )

        efficiency = response.transmission_efficiency[order_idx]
        self.assertEqual(efficiency.size, 1)

        onp.testing.assert_allclose(efficiency, expected_efficiency, rtol=tol)
