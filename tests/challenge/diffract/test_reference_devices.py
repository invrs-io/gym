"""Tests that simulations of reference metagratings give expected results."""

import dataclasses
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from invrs_gym.challenge.diffract import metagrating_challenge, splitter_challenge

METAGRATING_DIR = pathlib.Path(__file__).resolve().parent / "metagrating_designs"
SPLITTER_DIR = pathlib.Path(__file__).resolve().parent / "splitter_designs"


class ReferenceMetagratingTest(unittest.TestCase):
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

        path = METAGRATING_DIR / fname
        density_array = onp.genfromtxt(path, delimiter=",")

        if density_array.ndim == 1:
            density_array = jnp.broadcast_to(density_array[:, jnp.newaxis], (119, 45))

        sim_params = dataclasses.replace(
            metagrating_challenge.METAGRATING_SIM_PARAMS,
            grid_shape=density_array.shape,
        )

        mc = metagrating_challenge.MetagratingComponent(
            spec=metagrating_challenge.METAGRATING_SPEC,
            sim_params=sim_params,
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


class ReferenceDiffractiveSplitterTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                "device1.csv",
                0.705,  # total efficiency expected
                0.010,  # total efficiency rtol
                0.014,  # average efficiency expected
                0.030,  # average efficiency rtol
                0.080,  # zeroth order efficiency expected
                0.060,  # zeroth order efficiency rtol
            ],
            [
                "device2.csv",
                0.702,  # total efficiency expected
                0.010,  # total efficiency rtol
                0.014,  # average efficiency expected
                0.030,  # average efficiency rtol
                0.029,  # zeroth order efficiency expected
                0.100,  # zeroth order efficiency rtol
            ],
            [
                "device3.csv",
                0.738,  # total efficiency expected
                0.010,  # total efficiency rtol
                0.015,  # average efficiency expected
                0.030,  # average efficiency rtol
                0.023,  # zeroth order efficiency expected
                0.050,  # zeroth order efficiency rtol
            ],
        ]
    )
    def test_efficiency_matches_expected(
        self,
        fname,
        total_efficiency_expected,
        total_efficiency_rtol,
        average_efficiency_expected,
        average_efficiency_rtol,
        zeroth_order_efficiency_expected,
        zeroth_order_efficiency_rtol,
    ):
        # Compares results to those reported at
        # https://www.lighttrans.com/fileadmin/shared/UseCases/Application_UC_Rigorous%20Analysis%20of%20Non-paraxial%20Diffractive%20Beam%20Splitter.pdf

        path = SPLITTER_DIR / fname
        density_array = onp.genfromtxt(path, delimiter=",")

        challenge = splitter_challenge.diffractive_splitter()
        params = challenge.component.init(jax.random.PRNGKey(0))

        # Upsample the resolution density array to match the default of the splitter
        # challenge. This is required, since the reference density array does not
        # have sufficient resolution for the default Fourier expansion.
        density_array = onp.kron(density_array, onp.ones((20, 20)))
        assert density_array.shape == params["density"].shape
        params["density"].array = density_array

        response, aux = challenge.component.response(params)
        metrics = challenge.metrics(response, params, aux)

        onp.testing.assert_allclose(
            metrics["total_efficiency"],
            total_efficiency_expected,
            rtol=total_efficiency_rtol,
        )

        onp.testing.assert_allclose(
            metrics["average_efficiency"],
            average_efficiency_expected,
            rtol=average_efficiency_rtol,
        )

        onp.testing.assert_allclose(
            metrics["zeroth_order_efficiency"],
            zeroth_order_efficiency_expected,
            rtol=zeroth_order_efficiency_rtol,
        )
