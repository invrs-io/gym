"""Tests that simulations of reference metalenses give expected results.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import pathlib
import unittest

import jax
import numpy as onp
import pytest
from parameterized import parameterized

from invrs_gym.challenges.metalens import challenge as metalens_challenge
from invrs_gym.challenges.metalens import component as metalens_component

REPO_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent
METALENS_PATH = REPO_PATH / "reference_designs/metalens/"


def load_reference_design(path):
    density_array = onp.genfromtxt(path, delimiter=",")

    assert onp.all(density_array[:, 0] == 0)  # Adjacent to ambient
    assert onp.all(density_array[:, -1] == 1)  # Adjacent to substrate

    polarization_str = str(path).split("/")[-2:]
    return density_array, polarization_str


def simulate_reference_design(path, compute_fields=False):
    density_array, _ = load_reference_design(path)
    mc = metalens_component.MetalensComponent(
        spec=metalens_challenge.METALENS_SPEC,
        sim_params=metalens_challenge.METALENS_SIM_PARAMS,
        density_initializer=lambda k, d: d,
    )
    params = dataclasses.replace(
        mc.init(jax.random.PRNGKey(0)),
        array=density_array,
        fixed_solid=None,
        fixed_void=None,
    )
    response, aux = mc.response(params, compute_fields=compute_fields)
    return params, response, aux


class ReferenceMetalensTest(unittest.TestCase):
    @parameterized.expand(
        [
            # device name, expected, tolerance
            # "Rasmus" designs have been adjusted from 10 -> 20 nm grid spacing.
            ["Rasmus70nm.csv", (21.8, 23.7, 24.2), 0.12],  # 73.8 nm
            ["Rasmus123nm.csv", (16.3, 14.9, 15.0), 0.05],  # 95.5 nm
            ["Rasmus209nm.csv", (12.1, 11.3, 11.0), 0.07],  # 182.4 nm
            ["Rasmus256nm.csv", (7.6, 7.8, 8.5), 0.24],  # 256.5 nm
            # "Mo" designs are at their original resolution.
            ["Mo86nm.csv", (14.9, 15.3, 16.7), 0.06],  # 85.7 nm
            ["Mo117nm.csv", (12.7, 12.1, 12.3), 0.08],  # 108.9 nm
            ["Mo180nm.csv", (12.8, 12.1, 12.6), 0.09],  # 147.6 nm
            ["Mo242nm.csv", (10.8, 11.4, 11.5), 0.09],  # 217.2 nm
        ]
    )
    @pytest.mark.slow
    def test_ex_efficiency_matches_expected(self, fname, expected_enhancement, rtol):
        # Compares enhancements against the FDTD values reported at
        # https://github.com/NanoComp/photonics-opt-testbed/tree/main/RGB_metalens

        path = METALENS_PATH / fname
        _, response, _ = simulate_reference_design(path)

        onp.testing.assert_allclose(
            response.enhancement_ex, expected_enhancement, rtol=rtol
        )
