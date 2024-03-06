"""Tests that simulations of reference metalenses give expected results.

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

from invrs_gym.challenges.metalens import challenge as metalens_challenge
from invrs_gym.challenges.metalens import component as metalens_component


REPO_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent
METALENS_PATH = REPO_PATH / "reference_designs/metalens/Ex"


def load_reference_design(path):
    density_array = onp.genfromtxt(path, delimiter=",")
    # Flip the orientation, so that indexing begins at the top of the metalens.
    density_array = density_array[:, ::-1]

    polarization_str, fname = str(path).split("/")[-2:]
    if fname.startswith("Mo"):
        grid_spacing = 0.020
    elif fname.startswith("Rasmus"):
        grid_spacing = 0.010
    elif fname.startswith("wenjin"):
        grid_spacing = 0.010
    else:
        raise ValueError(f"Unknown grid spacing for design {path}")
    return density_array, grid_spacing, polarization_str


def simulate_reference_design(path, compute_fields=False):
    density_array, grid_spacing, _ = load_reference_design(path)

    spec = metalens_challenge.METALENS_SPEC
    width_lens = density_array.shape[0] * grid_spacing
    thickness_lens = density_array.shape[1] * grid_spacing
    lens_offset = (spec.width - width_lens) / 2
    pml_lens_offset = lens_offset - spec.width_pml

    spec = dataclasses.replace(
        metalens_challenge.METALENS_SPEC,
        thickness_lens=thickness_lens,
        width_lens=width_lens,
        pml_lens_offset=pml_lens_offset,
        grid_spacing=grid_spacing,
    )
    mc = metalens_component.MetalensComponent(
        spec=spec,
        sim_params=metalens_challenge.METALENS_SIM_PARAMS,
        density_initializer=lambda k, d: d,
    )
    pad = (spec.grid_shape[0] - density_array.shape[0]) // 2
    assert 2 * pad + density_array.shape[0] == spec.grid_shape[0]
    params = dataclasses.replace(
        mc.init(jax.random.PRNGKey(0)),
        array=jnp.pad(density_array, ((pad, pad), (0, 0)), mode="edge"),
    )
    response, aux = mc.response(params, compute_fields=compute_fields)
    return params, response, aux


class ReferenceMetalensTest(unittest.TestCase):
    @parameterized.expand(
        [
            # device name, expected, tolerance
            ["Rasmus70nm.csv", (21.8, 23.7, 24.2), 0.140],  # 73.8 nm
            ["Mo86nm.csv", (14.9, 15.3, 16.7), 0.060],  # 85.7 nm
            ["Rasmus123nm.csv", (16.3, 14.9, 15.0), 0.060],  # 95.5 nm
            ["Mo117nm.csv", (12.7, 12.1, 12.3), 0.080],  # 108.9 nm
            ["Mo180nm.csv", (12.8, 12.1, 12.6), 0.100],  # 147.6 nm
            ["Rasmus209nm.csv", (12.1, 11.3, 11.0), 0.080],  # 182.4 nm
            ["Mo242nm.csv", (10.8, 11.4, 11.5), 0.12],  # 217.2 nm
            ["Rasmus256nm.csv", (7.6, 7.8, 8.5), 0.16],  # 256.5 nm
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
