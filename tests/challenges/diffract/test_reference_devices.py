"""Tests that simulations of reference metagratings give expected results.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from parameterized import parameterized

from invrs_gym.challenges.diffract import metagrating_challenge, splitter_challenge

REPO_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent
METAGRATING_DIR = REPO_PATH / "reference_designs/metagrating"
SPLITTER_DIR = REPO_PATH / "reference_designs/diffractive_splitter"


class ReferenceMetagratingTest(unittest.TestCase):
    @parameterized.expand(
        [
            # device name, expected, tolerance
            ["device1.csv", 0.957, 0.01],  # Reticolo 0.957, Meep 0.955
            ["device2.csv", 0.933, 0.01],  # Reticolo 0.933, Meep 0.938
            ["device3.csv", 0.966, 0.02],  # Reticolo 0.966, Meep 0.950
            ["device4.csv", 0.933, 0.025],  # Reticolo 0.933, Meep 0.925
            ["device5.csv", 0.841, 0.02],  # Reticolo 0.841, Meep 0.843
        ]
    )
    @pytest.mark.slow
    def test_efficiency_matches_expected(self, fname, expected_efficiency, rtol):
        # Compares efficiencies against those reported at
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

        efficiency = response.transmission_efficiency[order_idx, :]
        self.assertEqual(efficiency.size, 1)

        onp.testing.assert_allclose(efficiency, expected_efficiency, rtol=rtol)


class ReferenceDiffractiveSplitterTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                "device1.csv",
                0.705,  # total efficiency expected
                0.010,  # total efficiency rtol
                0.014,  # average efficiency expected
                0.040,  # average efficiency rtol
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
                0.010,  # average efficiency rtol
                0.023,  # zeroth order efficiency expected
                0.080,  # zeroth order efficiency rtol
            ],
        ]
    )
    @pytest.mark.slow
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
        density_array = onp.kron(density_array, onp.ones((10, 10)))
        assert density_array.shape == params["density"].shape
        params["density"].array = density_array

        response, aux = challenge.component.response(params)
        metrics = challenge.metrics(response, params, aux)

        print(metrics["total_efficiency"])
        print(metrics["average_efficiency"])
        print(metrics["zeroth_order_efficiency"])

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
