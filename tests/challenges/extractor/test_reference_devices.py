"""Tests that simulations of the reference extractor gives expected results.

Copyright (c) 2025 invrs.io LLC
"""

import dataclasses
import pathlib
import unittest

import fmmax
import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from totypes import json_utils

from invrs_gym.challenges.extractor import challenge

REPO_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent
DESIGN_PATH = (
    REPO_PATH / "reference_designs/photon_extractor/"
    "240118_mfschubert_8569349cf4b44541ee37aa3eeed0127b70e29fb52674a4e97370fe7d95323bc7.json"
)


class ReferenceExtractorTest(unittest.TestCase):
    @pytest.mark.slow
    def test_bare_substrate_matches_expected(self):
        spec = dataclasses.replace(challenge.EXTRACTOR_SPEC, fwhm_source=0.0)
        sim_params = dataclasses.replace(
            challenge.EXTRACTOR_SIM_PARAMS, approximate_num_terms=1200
        )

        # Compute the response of the reference design.
        pec = challenge.photon_extractor(spec=spec, sim_params=sim_params)
        params = pec.component.init(jax.random.PRNGKey(0))
        params.array = jnp.zeros_like(params.array)

        response, _ = pec.component.response(params)

        expected_bare_substrate_emitted_power = (43.499092, 43.50009, 50.655725)
        expected_bare_substrate_extracted_power = (2.3140843, 2.314318, 0.14749196)
        expected_bare_substrate_collected_power = (1.3164568, 1.3165493, 0.06888947)

        with self.subTest("emitted_power"):
            onp.testing.assert_allclose(
                response.emitted_power, response.bare_substrate_emitted_power, rtol=1e-3
            )
            onp.testing.assert_allclose(
                response.emitted_power, expected_bare_substrate_emitted_power, rtol=1e-3
            )
        with self.subTest("extracted_power"):
            onp.testing.assert_allclose(
                response.extracted_power,
                response.bare_substrate_extracted_power,
                rtol=1e-3,
            )
            onp.testing.assert_allclose(
                response.extracted_power,
                expected_bare_substrate_extracted_power,
                rtol=1e-3,
            )
        with self.subTest("collected_power"):
            onp.testing.assert_allclose(
                response.collected_power,
                response.bare_substrate_collected_power,
                rtol=1e-3,
            )
            onp.testing.assert_allclose(
                response.collected_power,
                expected_bare_substrate_collected_power,
                rtol=1e-3,
            )

    @pytest.mark.slow
    def test_convergence(self):
        # Test that the simulations of the reference design are converged.
        pec = challenge.photon_extractor()

        with open(DESIGN_PATH) as f:
            serialized = f.read()
        extractor_params = json_utils.pytree_from_json(serialized)

        responses = []
        for approximate_num_terms in [1200, 1600]:
            expansion = fmmax.generate_expansion(
                primitive_lattice_vectors=fmmax.LatticeVectors(
                    u=pec.component.spec.pitch * fmmax.X,
                    v=pec.component.spec.pitch * fmmax.Y,
                ),
                approximate_num_terms=approximate_num_terms,
                truncation=pec.component.sim_params.truncation,
            )
            response, _ = pec.component.response(extractor_params, expansion=expansion)
            responses.append(response)

        response_1200, response_1600 = responses

        onp.testing.assert_allclose(
            response_1200.emitted_power,
            response_1600.emitted_power,
            rtol=0.05,
        )

        # Collected power for z-oriented dipoles converges a bit more slowly. Use a
        # larger tolerance for comparison.
        with self.subTest("xy dipoles"):
            onp.testing.assert_allclose(
                response_1200.collected_power[:2],
                response_1600.collected_power[:2],
                rtol=0.08,
            )
        with self.subTest("z dipole"):
            onp.testing.assert_allclose(
                response_1200.collected_power[2],
                response_1600.collected_power[2],
                rtol=0.14,
            )
        with self.subTest("xy dipoles bare substrate"):
            onp.testing.assert_allclose(
                response_1200.bare_substrate_collected_power[:2],
                response_1600.bare_substrate_collected_power[:2],
                rtol=0.07,
            )
        with self.subTest("z dipole bare substrate"):
            onp.testing.assert_allclose(
                response_1200.bare_substrate_collected_power[2],
                response_1600.bare_substrate_collected_power[2],
                rtol=0.01,
            )
